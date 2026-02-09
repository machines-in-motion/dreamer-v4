import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from functools import partial
from dreamerv4.models.utils import load_tokenizer, load_denoiser
from dreamerv4.models.dynamics import DenoiserWrapper
from dreamerv4.datasets import create_distributed_dataloader
from dreamerv4.utils.distributed import load_ddp_checkpoint, save_ddp_checkpoint
from dreamerv4.loss import ForwardDiffusionWithShortcut, compute_bootstrap_diffusion_loss, RMSLossScaler

from dreamerv4.utils.distributed import setup_distributed, cleanup_distributed
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime
import math
from datetime import timedelta
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.distributed.elastic.multiprocessing.errors import record
from pathlib import Path

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-8):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale to [min_lr/max_lr, 1.0]
        return max(min_lr / optimizer.defaults['lr'], cosine_decay)
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@record
@hydra.main(config_path="config", config_name="dynamics/pushT", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul
    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()
    # Print info only from rank 0
    if rank == 0:
        print(f"Initialized distributed training with {world_size} GPUs")
        print(f"Running on {world_size} GPUs")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
    
    torch.manual_seed(cfg.seed + rank)
        
    effective_global_batch = cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps
    if rank == 0:
        print(f"Effective global batch size: {effective_global_batch}")

    train_loader, train_sampler, train_dataset = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.denoiser.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1,
        seed=cfg.seed,
        split="train",
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,   # controls which episodes go to train/test
        shuffle=True,
        drop_last=True,
    )

    test_loader, test_sampler, test_dataset = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.denoiser.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1,
        seed=cfg.seed,          # seed for sampler sharding (not for split itself)
        split="test",
        train_fraction=cfg.dataset.train_episodes_fraction,
        split_seed=cfg.dataset.split_seed,   # must match train_dataset
        shuffle=False,    # evaluation: deterministic order
        drop_last=False,  # keep all test windows
    )
    # Create models
    if rank == 0:
        print("Creating encoder and decoder models...")
    tokenizer = load_tokenizer(cfg, device=device, max_num_forward_steps=cfg.denoiser.max_sequence_length) 
    
    if cfg.dynamics_ckpt:
        print(f"Loading dynamics from: {cfg.dynamics_ckpt}")
        denoiser = load_denoiser(cfg, device=device, model_key='model', max_num_forward_steps=cfg.denoiser.max_sequence_length)
    else:
        denoiser = DenoiserWrapper(cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length)
    diffuser = ForwardDiffusionWithShortcut(num_noise_levels=cfg.denoiser.num_noise_levels)
    # tokenizer = tokenizer.to(device, dtype=torch.bfloat16)
    # denoiser = denoiser.to(device, dtype=torch.bfloat16)
    tokenizer = tokenizer.to(device)
    denoiser = denoiser.to(device)
    # Print parameter counts
    if rank == 0:
        learnable_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {learnable_params:,}")

    # Freeze tokenizer
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)
    
    if cfg.train.use_compile:
        # denoiser = torch.compile(denoiser, mode="max-autotune-no-cudagraphs", fullgraph=False)
        denoiser = torch.compile(denoiser, mode="max-autotune-no-cudagraphs", fullgraph=True)
        tokenizer = torch.compile(tokenizer, mode="max-autotune-no-cudagraphs", fullgraph=False)


    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    # Create optimizer
    optim = torch.optim.AdamW(denoiser.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    steps_per_epoch = len(train_loader)
    steps_per_epoch = steps_per_epoch
    total_steps = cfg.train.num_epochs * steps_per_epoch // cfg.train.accum_grad_steps
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    
    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    start_epoch = 0
    global_update = 0  # counts optimizer updates
    ckpt_path = None

    if cfg.reload_checkpoint is not None:
        # Try loading checkpoint BEFORE initializing W&B
        print("Resuming from checkpoint:", cfg.reload_checkpoint)
        start_epoch, global_update, wandb_run_id, log_dir = load_ddp_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model=denoiser,
            optim=optim,
            scheduler=scheduler,
            rank=rank,
        )
    else:
        if rank == 0:
            print("Not resuming from checkpoint, starting from scratch.")
    
    # TensorBoard + wandb (rank 0 only)
    if rank == 0:
        # Use stable log_dir if resuming, otherwise create new one
        if log_dir is None:
            now = datetime.datetime.now()
            log_dir = cfg.output_dir
            os.makedirs(log_dir, exist_ok=True)
            ckpt_path = os.path.join(log_dir)
        else:
            # Resuming: reuse existing log_dir
            print(f"Reusing log directory: {log_dir}")

        tb_log_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)

        # Initialize W&B with resume logic
        if wandb_run_id is not None:
            # Resuming an existing run
            wandb.init(
                project=cfg.wandb.project,
                id=wandb_run_id,
                resume="must",  # or "must" if you want to error if run doesn't exist
                config= OmegaConf.to_container(cfg, resolve=True), 
                sync_tensorboard=True,
                dir=log_dir,
            )
        else:
            # Starting a new run
            wandb.init(
                project=cfg.wandb.project,
                name=f"run_num_gpus{world_size}",
                config=OmegaConf.to_container(cfg, resolve=True),
                sync_tensorboard=True,
                dir=log_dir,
            )
        # Capture the W&B run ID for saving in checkpoint
        wandb_run_id = wandb.run.id

        tb_writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        tb_writer = None
        wandb_run_id = None

    obj_list = [ckpt_path]
    dist.broadcast_object_list(obj_list, src=0)
    ckpt_path = obj_list[0]

    # Broadcast log_dir and wandb_run_id to all ranks so they can save it later
    obj_list = [log_dir, wandb_run_id]
    dist.broadcast_object_list(obj_list, src=0)
    log_dir, wandb_run_id = obj_list[0], obj_list[1]

    # Synchronize before training
    dist.barrier()
    
    if rank == 0:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    dist.barrier()
    
    if rank == 0:
        print("Starting warmup iterations...")
    # optim = torch.optim.AdamW(denoiser.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    # scheduler = get_cosine_schedule_with_warmup(optim, 300, total_steps)
    loss_scaler = RMSLossScaler()

    # Track statistics
    epoch_times = []
    epoch_losses = []
    epoch_losses_long = []
    epoch_fps = []
    epoch_data_times = []
    
    for epoch in range(start_epoch, cfg.train.num_epochs):
        # --- TRAIN PHASE ---
        denoiser.train()

        # CRITICAL: Set epoch for DistributedSampler to reshuffle data
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        epoch_loss_sum_long = 0.0
        step_times = []
        data_times = []
        data_start = time.perf_counter()

        num_updates_short = 0
        num_updates_long = 0
        # Accumulators over the current accumulation window
        accum_flow = 0.0
        accum_shortcut = 0.0
        accum_total = 0.0
        accum_flow_long = 0.0
        accum_shortcut_long = 0.0
        accum_total_long = 0.0
        long_seq_sample = False

        for step_idx, batch in enumerate(train_loader):
            micro_idx = step_idx % cfg.train.accum_grad_steps
            is_last_micro = (micro_idx == cfg.train.accum_grad_steps - 1)
            # --- measure how long we waited for this batch to arrive ---
            data_end = time.perf_counter()
            data_time = data_end - data_start
            data_times.append(data_time)

            # Move data to device
            images = batch['image'].to(device, non_blocking=True)  # (B, T, C, H, W)
            if not cfg.train.video_pretraining:
                actions = batch['action'].to(device, non_blocking=True)  # (B, T, action_dim)
            else:
                actions = torch.zeros(images.shape[0], images.shape[1], cfg.denoiser.n_actions).to(images.device, dtype=images.dtype)

            
            if long_seq_sample:
                B = cfg.train.long_seq_batch_per_gpu
                assert B <= cfg.train.batch_per_gpu, "long_seq_batch_per_gpu must be <= batch_per_gpu"
                images = images[:B]
                actions = actions[:B]
            else:
                T_max = cfg.denoiser.context_length
                assert T_max <= images.shape[1], "context_length exceeds sequence length in data"
                images = images[:, :T_max]
                actions = actions[:, :T_max]
            
            # Convert to bfloat16
            images = images.to(torch.bfloat16)
            actions = actions.to(torch.bfloat16)[:, :, :cfg.denoiser.n_actions]

            # Time the training step
            torch.cuda.synchronize(device)
            step_start = time.perf_counter()
            
            # Forward pass
            # Zero grads at the start of each accumulation window
            if micro_idx == 0:
                optim.zero_grad(set_to_none=True)
            # Encode (Frozen)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    z_clean = tokenizer.encode(images).detach().clone()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                diffused_info = diffuser(z_clean)
                flow_loss, bootstrap_loss = compute_bootstrap_diffusion_loss(diffused_info, denoiser, actions=actions)
                # 1. Calculate TOTAL loss for THIS micro-batch
                # Do NOT add to a persistent tensor variable like 'accum_flow' here
                loss_micro_raw = flow_loss + bootstrap_loss
                if long_seq_sample:
                    loss_micro = loss_scaler('long_seq_loss', loss_micro_raw)/cfg.train.accum_grad_steps
                else: 
                    loss_micro = loss_scaler('short_seq_loss', loss_micro_raw)/cfg.train.accum_grad_steps

            loss_micro.backward()
            
            # 3. Accumulate values for logging (DETACHED)
            if not long_seq_sample:
                accum_flow += flow_loss.item() 
                accum_shortcut += bootstrap_loss.item()
                accum_total += loss_micro.item()
            else:
                accum_flow_long += flow_loss.item() 
                accum_shortcut_long += bootstrap_loss.item()
                accum_total_long += loss_micro.item()
            
            # If this is the last micro-batch in the window, do optimizer step
            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)
                optim.step()
                scheduler.step()
                global_update += 1
                # Compute synchronized loss across all ranks
                
                if long_seq_sample:
                    total = torch.tensor([accum_total_long], device=device)
                    dist.all_reduce(total, op=dist.ReduceOp.AVG)
                    sync_loss_long = total.item()
                    epoch_loss_sum_long += sync_loss_long
                    num_updates_long += 1

                    if rank == 0:
                        tb_writer.add_scalar("train/total_loss_long_normalized", sync_loss_long, global_update)
                        tb_writer.add_scalar("train/flow_loss_long", accum_flow_long, global_update)
                        tb_writer.add_scalar("train/shortcut_loss_long", accum_shortcut_long, global_update)
                else:
                    total = torch.tensor([accum_total], device=device)
                    dist.all_reduce(total, op=dist.ReduceOp.AVG)
                    sync_loss = total.item()
                    epoch_loss_sum += sync_loss
                    num_updates_short += 1

                    if rank == 0:
                        tb_writer.add_scalar("train/total_loss_short_normalized", sync_loss, global_update)
                        tb_writer.add_scalar("train/flow_loss_short", accum_flow, global_update)
                        tb_writer.add_scalar("train/shortcut_loss_short", accum_shortcut, global_update)

                if rank == 0:
                    tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_update)
                    tb_writer.add_scalar("train/long_seq_sampled", float(long_seq_sample), global_update)

                long_next = torch.zeros((), device=device, dtype=torch.bool)
                
                if rank == 0:
                    # Checkpoint every N updates (within epoch)
                    if global_update % cfg.save_every == 0:
                        print(f"[Checkpoint] Saving at global_update={global_update}")
                        save_ddp_checkpoint(
                            ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                            epoch=epoch,
                            global_update=global_update,
                            model=denoiser,
                            optim=optim,
                            scheduler=scheduler,
                            rank=rank,
                            wandb_run_id=wandb_run_id,
                            log_dir=log_dir,
                        )

                    long_next = (torch.rand((), device=device) < cfg.train.long_seq_prob)

                dist.broadcast(long_next, src=0)
                long_seq_sample = bool(long_next.item())

                # Reset accumulators for next accumulation window
                accum_flow = 0.0
                accum_shortcut = 0.0
                accum_total = 0.0
                accum_flow_long = 0.0
                accum_shortcut_long = 0.0
                accum_total_long = 0.0

            torch.cuda.synchronize(device)
            step_end = time.perf_counter()
            step_time = step_end - step_start
            step_times.append(step_time)

            # Next data timing starts now (time until next batch is yielded)
            data_start = time.perf_counter()

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        # Compute epoch statistics
        avg_loss = epoch_loss_sum / num_updates_short if num_updates_short > 0 else 0.0
        avg_loss_long = epoch_loss_sum_long / num_updates_long if num_updates_long > 0 else 0.0

        total_frames = cfg.train.batch_per_gpu * cfg.denoiser.context_length * world_size * steps_per_epoch
        epoch_fps_val = total_frames / epoch_time
        avg_data_time_epoch = sum(data_times) / len(data_times)

        epoch_times.append(epoch_time)
        epoch_losses.append(avg_loss)
        epoch_losses_long.append(avg_loss_long)
        epoch_fps.append(epoch_fps_val)
        epoch_data_times.append(avg_data_time_epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{cfg.train.num_epochs} Summary:")
            print(f" Train Loss: {avg_loss:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            print(f"  Total Frames: {total_frames:,}")
            print(f"  Throughput: {epoch_fps_val:.2f} FPS")
            print(f"  Avg Data Time / step: {avg_data_time_epoch:.3f}s")
            print(f"  Avg Step Time: {sum(step_times)/len(step_times):.3f}s")
            print(f"{'='*60}\n")

            save_ddp_checkpoint(
                ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                epoch=epoch,
                global_update=global_update,
                model=denoiser,
                optim=optim,
                scheduler=scheduler,
                rank=rank,
                wandb_run_id=wandb_run_id,
                log_dir=log_dir,
            )
        # Synchronize after each epoch
        dist.barrier()
    
    # Final statistics
    if rank == 0:
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Overall Statistics (across {cfg.train.num_epochs} epochs):")
        print(f"  Average epoch time: {sum(epoch_times)/len(epoch_times):.2f}s")
        print(f"  Average loss (short): {sum(epoch_losses)/len(epoch_losses):.6f}")
        print(f"  Average loss (long): {sum(epoch_losses_long)/len(epoch_losses_long):.6f}")
        print(f"  Average FPS: {sum(epoch_fps)/len(epoch_fps):.2f}")
        print(f"  Min FPS: {min(epoch_fps):.2f}")
        print(f"  Max FPS: {max(epoch_fps):.2f}")
        
        # Memory report
        cur_alloc = torch.cuda.memory_allocated(device) / (1024 ** 3)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"\nGPU Memory (Rank 0):")
        print(f"  Current: {cur_alloc:.2f} GB")
        print(f"  Peak: {peak_alloc:.2f} GB")
        
        # Per-epoch breakdown
        print(f"\nPer-Epoch Breakdown:")
        for i in range(cfg.train.num_epochs):
            print(f"  Epoch {i+1}: Loss={epoch_losses[i]:.6f}, "
                  f"Time={epoch_times[i]:.2f}s, FPS={epoch_fps[i]:.2f}")
        print(f"{'='*60}")
    
    # Cleanup
    cleanup_distributed()
    if rank == 0:
        print("\nTraining completed successfully!")
        tb_writer.close()
        wandb.finish()

if __name__ == "__main__":
    main()
