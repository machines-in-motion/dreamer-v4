import os
import time
import torch
torch.compiler.reset()
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial
from dreamerv4.single_stream.utils import load_tokenizer 
from dreamerv4.single_stream.dynamics import DreamerV4DenoiserCfg, DreamerV4Denoiser
from dreamerv4.loss import ForwardDiffusionWithShortcut, compute_bootstrap_diffusion_loss

from dreamerv4.utils.distributed import setup_distributed, cleanup_distributed
from dreamerv4.datasets import ShardedHDF5Dataset
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
torch.autograd.set_detect_anomaly(True)

def save_checkpoint(
    ckpt_path: str,
    epoch: int,
    global_update: int,
    dyn: DDP,
    optim: torch.optim.Optimizer,
    scheduler,
    rank: int,
    wandb_run_id: str = None,
    log_dir: str = None,
):
    if rank == 0:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "global_update": global_update,
            "dyn": dyn.module.state_dict(),  # <- .module to unwrap DDP
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "wandb_run_id": wandb_run_id,
            "log_dir": log_dir,
        }
        torch.save(ckpt, ckpt_path)
        print(f"[rank0] Saved checkpoint to {ckpt_path}")

def load_checkpoint(
    ckpt_path: str,
    dyn: DDP,
    optim: torch.optim.Optimizer,
    scheduler,
    rank: int,
    device: torch.device,
):
    """
    Load FULL checkpoint saved by save_checkpoint().
    Returns (start_epoch, global_update, wandb_run_id, log_dir).
    """
    if not os.path.isfile(ckpt_path):
        if rank == 0:
            print(f"No checkpoint found at {ckpt_path}, starting from scratch.")
        return 0, 0, None, None  # epoch, global_update, wandb_run_id, log_dir

    # Rank 0 loads from disk
    if rank == 0:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print(f"[rank0] Loaded checkpoint from {ckpt_path}")
    else:
        ckpt = None

    # Broadcast checkpoint to all ranks
    obj_list = [ckpt]
    dist.broadcast_object_list(obj_list, src=0)
    ckpt = obj_list[0]

    start_epoch = ckpt.get("epoch", 0)
    global_update = ckpt.get("global_update", 0)
    wandb_run_id = ckpt.get("wandb_run_id", None)  # NEW
    log_dir = ckpt.get("log_dir", None)            # NEW

    dyn.module.load_state_dict(ckpt["dyn"])

    optim.load_state_dict(ckpt["optim"])
    scheduler.load_state_dict(ckpt["scheduler"])

    if rank == 0:
        print(f"Resuming from epoch {start_epoch+1}, global_update {global_update}")
        if wandb_run_id:
            print(f"Resuming W&B run ID: {wandb_run_id}")
        if log_dir:
            print(f"Resuming log directory: {log_dir}")

    return start_epoch, global_update, wandb_run_id, log_dir

def create_distributed_dataloader(
    data_dir: str,
    window_size: int,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    stride: int = 1,
    seed: int = 42,
    split: str = "train",
    train_fraction: float = 0.9,
    split_seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    Create DataLoader with DistributedSampler for sharded HDF5 dataset.
    """
    # Create the dataset with a fixed split
    dataset = ShardedHDF5Dataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
        split=split,
        train_fraction=train_fraction,
        split_seed=split_seed,
    )

    # Create DistributedSampler
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
    )

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader, sampler, dataset

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
@hydra.main(config_path="../config", config_name="dynamics/single_stream/soar", version_base=None)
def main(cfg: DictConfig):
    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()
    # Print info only from rank 0
    if rank == 0:
        print(f"Initialized distributed training with {world_size} GPUs")
        print(f"Running on {world_size // 8} nodes with 8 GPUs each")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
    
    torch.manual_seed(cfg.seed + rank)
        
    effective_global_batch = cfg.train.batch_per_gpu * world_size * cfg.train.accum_grad_steps
    if rank == 0:
        print(f"Effective global batch size: {effective_global_batch}")

    # Create models
    if rank == 0:
        print("Creating encoder and decoder models...")

    # tokenizer = load_tokenizer(cfg, device=device, compile=cfg.train.use_compile)
    tokenizer = load_tokenizer(cfg, device=device, compile=cfg.train.use_compile)
    tokenizer = tokenizer.to(torch.bfloat16)

    train_loader, train_sampler, train_dataset = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.tokenizer.max_context_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=4,
        stride=1,
        seed=cfg.seed,
        split="train",
        train_fraction=0.9,
        split_seed=cfg.dataset.split_seed,   # controls which episodes go to train/test
        shuffle=True,
        drop_last=True,
    )

    test_loader, test_sampler, test_dataset = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.tokenizer.max_context_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=4,
        stride=1,
        seed=cfg.seed,          # seed for sampler sharding (not for split itself)
        split="test",
        train_fraction=0.9,
        split_seed=cfg.dataset.split_seed,   # must match train_dataset
        shuffle=False,    # evaluation: deterministic order
        drop_last=False,  # keep all test windows
    )

    denoiser_cfg = DreamerV4DenoiserCfg(**OmegaConf.to_object(cfg.denoiser))
    denoiser = DreamerV4Denoiser(denoiser_cfg)
    diffuser = ForwardDiffusionWithShortcut(K_max=cfg.denoiser.K_max)
    denoiser.to(device)
    diffuser.to(device)
    # Print parameter counts
    if rank == 0:
        learnable_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {learnable_params:,}")


    if cfg.train.use_compile:
        denoiser = torch.compile(denoiser, mode="max-autotune-no-cudagraphs", fullgraph=False)

    denoiser = DDP(denoiser, device_ids=[local_rank], find_unused_parameters=False)

    # Create optimizer
    optim = torch.optim.AdamW(denoiser.parameters(), lr=1e-4, weight_decay=0.1)


    steps_per_epoch = len(train_loader)
    steps_per_epoch = steps_per_epoch
    total_steps = cfg.train.num_epochs * steps_per_epoch // cfg.train.accum_grad_steps
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    
    wandb_run_id = None
    log_dir = None
    ckpt_path = "./logs/dreamer_v4_dynamics/2025-11-30_08-05-07/latest.pt"
    start_epoch = 0
    global_update = 0  # counts optimizer updates
    
    if cfg.reload_checkpoint is not None:
        # Try loading checkpoint BEFORE initializing W&B
        start_epoch, global_update, wandb_run_id, log_dir = load_checkpoint(
            ckpt_path=ckpt_path,
            dyn=denoiser,
            optim=optim,
            scheduler=scheduler,
            rank=rank,
            device=device,
        )
    else:
        if rank == 0:
            print("Not resuming from checkpoint, starting from scratch.")

    # TensorBoard + wandb (rank 0 only)
    if rank == 0:
        # Use stable log_dir if resuming, otherwise create new one
        if log_dir is None:
            now = datetime.datetime.now()
            log_dir = f"./logs/dreamer_v4_dynamics/{now.strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(log_dir, exist_ok=True)
            ckpt_path = os.path.join(log_dir, "latest.pt")
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

        tb_writer = SummaryWriter(log_dir=log_dir)
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
        print("Starting warmup iterations...")

    # Track statistics
    epoch_times = []
    epoch_losses = []
    epoch_fps = []
    epoch_data_times = []

    for epoch in range(start_epoch, cfg.train.num_epochs):
        # --- TRAIN PHASE ---
        denoiser.train()

        # CRITICAL: Set epoch for DistributedSampler to reshuffle data
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        step_times = []
        data_times = []
        data_start = time.perf_counter()

        num_updates = 0

        # Accumulators over the current accumulation window
        accum_flow = 0.0
        accum_shortcut = 0.0
        accum_total = 0.0

        for step_idx, batch in enumerate(train_loader):
            #if step_idx > steps_per_epoch:
            #    break
            micro_idx = step_idx % cfg.train.accum_grad_steps
            is_last_micro = (micro_idx == cfg.train.accum_grad_steps - 1)

            # --- measure how long we waited for this batch to arrive ---
            data_end = time.perf_counter()
            data_time = data_end - data_start
            data_times.append(data_time)

            # Move data to device
            images = batch['image'].to(device, non_blocking=True)  # (B, T, C, H, W)
            actions = batch['action'].to(device, non_blocking=True)  # (B, T, action_dim)

            # Convert to bfloat16
            images = images.to(torch.bfloat16)
            actions = actions.to(torch.bfloat16)

            # Time the training step
            torch.cuda.synchronize(device)
            step_start = time.perf_counter()

            # Forward pass
            # Zero grads at the start of each accumulation window
            if micro_idx == 0:
                optim.zero_grad(set_to_none=True)

            # Encode (Frozen)
            with torch.no_grad():
                z_clean = tokenizer.encode(images).detach().clone()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # z_clean_bf16 = z_clean.to(torch.bfloat16)
                diffused_info = diffuser(z_clean)
                flow_loss, bootstrap_loss = compute_bootstrap_diffusion_loss(diffused_info, denoiser, actions=actions)
            # 1. Calculate TOTAL loss for THIS micro-batch
            # Do NOT add to a persistent tensor variable like 'accum_flow' here
            loss_micro = (flow_loss + bootstrap_loss) / cfg.train.accum_grad_steps
            
            # 2. Backward immediately on this specific graph
            loss_micro.backward()
            
            # 3. Accumulate values for logging (DETACHED)
            # Use .item() to ensure you store Python floats, not graph nodes
            accum_flow += flow_loss.item() 
            accum_shortcut += bootstrap_loss.item()
            accum_total += (flow_loss.item() + bootstrap_loss.item())

            # If this is the last micro-batch in the window, do optimizer step
            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(denoiser.parameters(), max_norm=1.0)

                optim.step()
                scheduler.step()
                global_update += 1

                total_mean = torch.tensor([accum_total], device=device)
                dist.all_reduce(total_mean, op=dist.ReduceOp.AVG)   # average across ranks
                sync_loss = total_mean.item()
                epoch_loss_sum += sync_loss
                num_updates += 1

                # sums over accumulation window on this rank
                flow_sum = torch.tensor([accum_flow], device=device)
                short_sum = torch.tensor([accum_shortcut], device=device)

                dist.all_reduce(flow_sum, op=dist.ReduceOp.AVG)
                dist.all_reduce(short_sum, op=dist.ReduceOp.AVG)

                flow_sum = flow_sum.item()
                short_sum = short_sum.item()

                flow_mean = flow_sum / cfg.train.accum_grad_steps
                short_mean = short_sum / cfg.train.accum_grad_steps

                if rank == 0:
                    # Sums over micro-batches in this update
                    tb_writer.add_scalar("train/flow_loss_mean", flow_mean, global_update)
                    tb_writer.add_scalar("train/shortcut_loss_mean", short_mean, global_update)
                    tb_writer.add_scalar("train/total_loss_mean", sync_loss, global_update)
                    current_lr = scheduler.get_last_lr()[0]
                    tb_writer.add_scalar("train/lr", current_lr, global_update)

                # Checkpoint every N updates (within epoch)
                if global_update % cfg.save_every == 0:
                    if rank == 0:
                        print(f"[Checkpoint] Saving at global_update={global_update}")
                    save_checkpoint(
                        ckpt_path=ckpt_path,
                        epoch=epoch,
                        global_update=global_update,
                        dyn=denoiser,
                        optim=optim,
                        scheduler=scheduler,
                        rank=rank,
                        wandb_run_id=wandb_run_id,
                        log_dir=log_dir,
                    )

                # Reset accumulators for next accumulation window
                accum_flow = 0.0
                accum_shortcut = 0.0
                accum_total = 0.0

            torch.cuda.synchronize(device)
            step_end = time.perf_counter()
            step_time = step_end - step_start
            step_times.append(step_time)

            # Next data timing starts now (time until next batch is yielded)
            data_start = time.perf_counter()

            # Print progress every 200 steps
            if rank == 0 and is_last_micro and (global_update % 200 == 0):
                avg_step_time = sum(step_times[-200:]) / len(step_times[-200:])
                avg_data_time = sum(data_times[-200:]) / len(data_times[-200:])

                frames_per_step = cfg.train.batch_per_gpu * cfg.denoiser.max_context_length * world_size
                step_fps = frames_per_step / avg_step_time
                data_fps = frames_per_step / avg_data_time if avg_data_time > 0 else float('inf')

                print(
                    f"Epoch {epoch+1}/{cfg.train.num_epochs} | "
                    f"Step {step_idx+1}/{steps_per_epoch} | "
                    f"Loss: {sync_loss:.6f} | "
                    f"Data: {avg_data_time:.3f}s ({data_fps:.1f} fps) | "
                    f"Compute: {avg_step_time:.3f}s ({step_fps:.1f} fps)"
                )

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        # Compute epoch statistics
        avg_loss = epoch_loss_sum / num_updates
        total_frames = cfg.train.batch_per_gpu * cfg.denoiser.max_context_length * world_size * steps_per_epoch
        epoch_fps_val = total_frames / epoch_time
        avg_data_time_epoch = sum(data_times) / len(data_times)

        epoch_times.append(epoch_time)
        epoch_losses.append(avg_loss)
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

        save_checkpoint(
            ckpt_path=ckpt_path,
            epoch=epoch,
            global_update=global_update,
            dyn=denoiser,
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
        print(f"  Average loss: {sum(epoch_losses)/len(epoch_losses):.6f}")
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
