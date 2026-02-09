import os
import time
import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from dreamerv4.models.tokenizer import TokenizerWrapper 
from dreamerv4.models.blocks import EfficientTransformerLayer
from dreamerv4.models.utils import load_tokenizer

from dreamerv4.datasets import ShardedHDF5Dataset
from dreamerv4.utils.distributed import setup_distributed, cleanup_distributed
import lpips
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime
import math
from pathlib import Path

assert torch.cuda.is_available(), "Tokenizer training requires CUDA"
class RMSLossScaler:
    """
    Tracks running RMS for named losses and returns normalized losses.
    """
    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq = {}  # name -> scalar tensor

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        # value is a scalar loss tensor (per-batch, per-rank)
        with torch.no_grad():
            sq = value.detach().pow(2)
            mean_sq = sq.mean()

            # Optionally average across ranks for a global RMS
            if dist.is_initialized():
                dist.all_reduce(mean_sq, op=dist.ReduceOp.AVG)

            if name not in self.ema_sq:
                self.ema_sq[name] = mean_sq
            else:
                self.ema_sq[name] = (
                    self.decay * self.ema_sq[name] + (1.0 - self.decay) * mean_sq
                )

            rms = (self.ema_sq[name] + self.eps).sqrt()

        # Normalize current loss by running RMS; gradient flows only through value
        return value / rms

def unwrap_model(m):
    # DDP / DataParallel
    while hasattr(m, "module"):
        m = m.module

    # torch.compile (OptimizedModule)
    for attr in ("_orig_mod", "_orig_module", "_orig_model"):
        if hasattr(m, attr):
            m = getattr(m, attr)

    return m

def save_fsdp_checkpoint(
    ckpt_path: str,
    epoch: int,
    global_update: int,
    model: FSDP,
    optim: torch.optim.Optimizer,
    scheduler,
    rms_norm: RMSLossScaler,
    rank: int,
    wandb_run_id: str = None,
    log_dir: str = None,
):
    """
    Save a FULL checkpoint (unsharded enc/dec) on rank 0.
    ALL RANKS must call this function for FSDP collectives to work.
    """
    # Gather full (unsharded) state dicts on CPU
    # ALL RANKS must participate in this, even though only rank 0 gets the result
    full_cfg = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        m = unwrap_model(model)
        model_state = m.state_dict()  # All ranks participate in gather

    # Only rank 0 saves to disk
    if rank == 0:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        
        ckpt = {
            "epoch": epoch,
            "global_update": global_update,
            "model": model_state,
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rms_norm": rms_norm.ema_sq,
            "wandb_run_id": wandb_run_id,
            "log_dir": log_dir,
        }

        torch.save(ckpt, ckpt_path)
        print(f"[rank0] Saved checkpoint to {ckpt_path}")

def load_fsdp_checkpoint(
    ckpt_path: str,
    model: FSDP,
    optim: torch.optim.Optimizer,
    scheduler,
    rms_norm: RMSLossScaler,
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

    # Load full enc/dec state dicts into FSDP models
    full_cfg = FullStateDictConfig(rank0_only=False, offload_to_cpu=True)

    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_cfg):
        model.load_state_dict(ckpt["model"])

    optim.load_state_dict(ckpt["optim"])
    scheduler.load_state_dict(ckpt["scheduler"])

    # Restore RMS loss scaler state
    rms_norm.ema_sq = ckpt.get("rms_norm", {})

    if rank == 0:
        print(f"Resuming from epoch {start_epoch+1}, global_update {global_update}")
        if wandb_run_id:
            print(f"Resuming W&B run ID: {wandb_run_id}")
        if log_dir:
            print(f"Resuming log directory: {log_dir}")

    return start_epoch, global_update, wandb_run_id, log_dir

def get_fsdp_wrap_policy():
    """Define which layers to wrap with FSDP."""
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            EfficientTransformerLayer
        },
    )

def setup_fsdp_model(model, mixed_precision=True, sharding_strategy="FULL_SHARD"):
    """Wrap model with FSDP for distributed training."""
    # Mixed precision policy for H200
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        mp_policy = None
    
    # Map string to ShardingStrategy enum
    strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    }
    
    fsdp_model = FSDP(
        model,
        sharding_strategy=strategy_map[sharding_strategy],
        auto_wrap_policy=get_fsdp_wrap_policy(),
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )
    
    return fsdp_model

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
@hydra.main(config_path="config", config_name="tokenizer/pushT", version_base=None)
def main(cfg: DictConfig):
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.enable_fast_matmul
    # Setup distributed training
    rank, local_rank, world_size, device = setup_distributed()

    # LPIPS model (perceptual loss), kept outside FSDP and frozen
    # Initialize only on rank 0 to avoid concurrent downloads; broadcast to other ranks
    if rank == 0:
        # Load with pretrained on rank 0 (succeeds reliably)
        lpips_model = lpips.LPIPS(net='vgg').to(device=device, dtype=torch.bfloat16)
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)
    else:
        # Other ranks: Create un-pretrained skeleton and load broadcasted weights
        lpips_model = lpips.LPIPS(net='vgg', pretrained=False).to(device=device, dtype=torch.bfloat16)  # No download
        lpips_model.eval()
        for p in lpips_model.parameters():
            p.requires_grad_(False)

    if rank == 0:
        sd = lpips_model.state_dict()
    else:
        sd = None

    obj_list = [sd]
    dist.broadcast_object_list(obj_list, src=0)
    sd = obj_list[0]  # now the same dict on all ranks

    if rank != 0:
        lpips_model.load_state_dict(sd)

    # Print info only from rank 0
    if rank == 0:
        print(f"Initialized distributed training with {world_size} GPUs")
        print(f"Running on {world_size // 8} nodes with 8 GPUs each")
        print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'not set')}")
        print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'not set')}")
    
    # Set random seed for reproducibility
    torch.manual_seed(cfg.seed + rank)
        
    # Batch size per GPU
    effective_global_batch = cfg.train.batch_per_gpu * world_size * cfg.train.grad_accum_steps
    if rank == 0:
        print(f"Effective global batch size: {effective_global_batch}")

    # Create models
    if rank == 0:
        print("Creating encoder and decoder models...")

    # Wrap with FSDP
    if rank == 0:
        print("Wrapping models with FSDP...")

    train_loader, train_sampler, train_dataset = create_distributed_dataloader(
        data_dir=cfg.dataset.data_dir,
        window_size=cfg.tokenizer.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
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
        window_size=cfg.tokenizer.max_sequence_length,
        batch_size=cfg.train.batch_per_gpu,
        rank=rank,
        world_size=world_size,
        num_workers=cfg.train.num_workers,
        stride=1,
        seed=cfg.seed,          # seed for sampler sharding (not for split itself)
        split="test",
        train_fraction=0.9,
        split_seed=cfg.dataset.split_seed,   # must match train_dataset
        shuffle=False,    # evaluation: deterministic order
        drop_last=False,  # keep all test windows
    )
     # Todo: Make it cofigurable with hydra
    if cfg.tokenizer_ckpt:
        print(f'Initializing the tokenizer with: {cfg.tokenizer_ckpt}')
        tokenizer = load_tokenizer(cfg, device=device, max_num_forward_steps=cfg.tokenizer.max_sequence_length)
        tokenizer.to(device)
    else:
        tokenizer = TokenizerWrapper(cfg, max_num_forward_steps=cfg.denoiser.max_sequence_length)
        tokenizer.to(device)

    if cfg.train.use_compile:
        tokenizer = torch.compile(tokenizer, mode="max-autotune-no-cudagraphs", fullgraph=False)
        
    # Print parameter counts
    if rank == 0:
        learnable_params = sum(p.numel() for p in tokenizer.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {learnable_params:,}")
    
    tokenizer = setup_fsdp_model(tokenizer, mixed_precision=cfg.train.mixed_precision, sharding_strategy="FULL_SHARD")
    
    # Create optimizer
    params = tokenizer.parameters()
    optim = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay) 

    steps_per_epoch = len(train_loader)
    STEPS_PER_EPOCH = steps_per_epoch
    total_steps = cfg.train.num_epochs * steps_per_epoch // cfg.train.grad_accum_steps
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    
    
    # RMS loss normalizer for tokenizer losses
    rms_norm = RMSLossScaler(decay=0.99, eps=1e-8)

    # Initialize model from a previously trained model but start training from scratch
    wandb_run_id = cfg.wandb.run_name
    log_dir = None
    start_epoch = 0
    global_update = 0  # counts optimizer updates
    ckpt_path = None
    
    if cfg.reload_checkpoint is not None:
        # Try loading checkpoint BEFORE initializing W&B
        start_epoch, global_update, wandb_run_id, log_dir = load_fsdp_checkpoint(
            ckpt_path=cfg.reload_checkpoint,
            model= tokenizer,
            optim=optim,
            scheduler=scheduler,
            rms_norm=rms_norm,
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
    # Track statistics
    epoch_times = []
    epoch_losses = []
    epoch_fps = []
    epoch_data_times = []

    for epoch in range(start_epoch, cfg.train.num_epochs):
        # --- TRAIN PHASE ---
        tokenizer.train()
        # CRITICAL: Set epoch for DistributedSampler to reshuffle data
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        step_times = []
        data_times = []
        data_start = time.perf_counter()

        num_updates = 0

        # Accumulators over the current accumulation window
        accum_mse = 0.0
        accum_lpips = 0.0
        accum_raw_loss = 0.0  # mse + 0.2 * lpips
        accum_norm = 0.0

        for step_idx, batch in enumerate(train_loader):
            micro_idx = step_idx % cfg.train.grad_accum_steps
            is_last_micro = (micro_idx == cfg.train.grad_accum_steps - 1)

            # --- measure how long we waited for this batch to arrive ---
            data_end = time.perf_counter()
            data_time = data_end - data_start
            data_times.append(data_time)

            # Move data to device
            images = batch['image'].to(device, non_blocking=True)  # (B, T, C, H, W)
            # actions are present for future conditioning but unused in tokenizer reconstruction training.
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
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                x_hat = tokenizer(images)
            # --- Tokenizer reconstruction losses: MSE + 0.2 * LPIPS, with RMS normalization ---
            # 1) Pixel MSE in the native scale of your data
            mse_loss = nn.functional.mse_loss(x_hat, images)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # 2) LPIPS perceptual loss expects images in [-1, 1]
                images_lpips = (images * 2.0) - 1.0          # [B, T, C, H, W] -> [-1,1]
                x_hat_lpips = (x_hat * 2.0) - 1.0

                # LPIPS implementation expects shape [B', C, H, W]; flatten time into batch
                B, T, C, H, W = images_lpips.shape
                images_lpips_flat = images_lpips.view(B * T, C, H, W)
                x_hat_lpips_flat = x_hat_lpips.view(B * T, C, H, W)

                lpips_val = lpips_model(x_hat_lpips_flat, images_lpips_flat)
                # Some LPIPS implementations return shape [B']; take mean over batch
                lpips_loss = lpips_val.mean()

            # Pre-RMS total loss (for logging only): L_raw = mse + 0.2 * lpips
            raw_loss = mse_loss + 0.2 * lpips_loss

            # Update accumulators (CPU scalars)
            accum_mse += mse_loss.detach().item()
            accum_lpips += lpips_loss.detach().item()
            accum_raw_loss += raw_loss.detach().item()

            # 3) RMS loss normalization per term
            mse_norm = rms_norm("mse", mse_loss)
            lpips_norm = rms_norm("lpips", lpips_loss)

            # 4) Combined tokenizer loss: L = LMSE + 0.2 * LLPIPS
            loss = (mse_norm + 0.2 * lpips_norm) / cfg.train.grad_accum_steps
            accum_norm = accum_norm + loss.detach().item()

            # Backward pass
            if not is_last_micro:
                with tokenizer.no_sync():
                    loss.backward()
            else:
                loss.backward()

            # If this is the last micro-batch in the window, do optimizer step
            if is_last_micro:
                #torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                tokenizer.clip_grad_norm_(max_norm=1.0)

                optim.step()
                scheduler.step()
                global_update += 1

                norm_mean = torch.tensor([accum_norm], device=device)
                dist.all_reduce(norm_mean, op=dist.ReduceOp.AVG)   # average across ranks
                sync_loss = norm_mean.item()
                epoch_loss_sum += sync_loss
                num_updates += 1

                # sums over accumulation window on this rank
                mse_sum = torch.tensor([accum_mse], device=device)
                lpips_sum = torch.tensor([accum_lpips], device=device)
                raw_sum = torch.tensor([accum_raw_loss], device=device)

                dist.all_reduce(mse_sum, op=dist.ReduceOp.AVG)
                dist.all_reduce(lpips_sum, op=dist.ReduceOp.AVG)
                dist.all_reduce(raw_sum, op=dist.ReduceOp.AVG)

                mse_sum = mse_sum.item()
                lpips_sum = lpips_sum.item()
                raw_sum = raw_sum.item()

                mse_mean = mse_sum / cfg.train.grad_accum_steps
                lpips_mean = lpips_sum / cfg.train.grad_accum_steps
                raw_mean = raw_sum / cfg.train.grad_accum_steps

                if rank == 0 and num_updates%5==0:
                    # Sums over micro-batches in this update
                    tb_writer.add_scalar("train/mse_mean", mse_mean, global_update)
                    tb_writer.add_scalar("train/lpips_mean", lpips_mean, global_update)
                    tb_writer.add_scalar("train/raw_loss_mean", raw_mean, global_update)
                    tb_writer.add_scalar("train/normalized_loss_mean", sync_loss, global_update)
                    current_lr = scheduler.get_last_lr()[0]
                    tb_writer.add_scalar("train/lr", current_lr, global_update)

                # Checkpoint every N updates (within epoch)
                if global_update % cfg.save_every == 0:
                    if rank == 0:
                        print(f"[Checkpoint] Saving at global_update={global_update}")
                    save_fsdp_checkpoint(
                        ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
                        epoch=epoch,
                        global_update=global_update,
                        model=tokenizer, 
                        optim=optim,
                        scheduler=scheduler,
                        rms_norm=rms_norm,
                        rank=rank,
                        wandb_run_id=wandb_run_id,
                        log_dir=log_dir,
                    )

                # Reset accumulators for next accumulation window
                accum_mse = 0.0
                accum_lpips = 0.0
                accum_raw_loss = 0.0
                accum_norm = 0.0

            torch.cuda.synchronize(device)
            step_end = time.perf_counter()
            step_time = step_end - step_start
            step_times.append(step_time)
            # if rank == 0:
            #     print(f'Step time: {step_time}')
            #     print(f'data time: {data_end-data_start}')
            # Next data timing starts now (time until next batch is yielded)
            data_start = time.perf_counter()

            # Print progress every 200 steps
            if rank == 0 and is_last_micro and (global_update % 200 == 0):
                avg_step_time = sum(step_times[-200:]) / len(step_times[-200:])
                avg_data_time = sum(data_times[-200:]) / len(data_times[-200:])

                frames_per_step = cfg.train.batch_per_gpu * cfg.tokenizer.max_sequence_length * world_size
                step_fps = frames_per_step / avg_step_time
                data_fps = frames_per_step / avg_data_time if avg_data_time > 0 else float('inf')

                print(
                    f"Epoch {epoch+1}/{cfg.train.num_epochs} | "
                    f"Step {step_idx+1}/{STEPS_PER_EPOCH} | "
                    f"Loss: {sync_loss:.6f} | "
                    f"Data: {avg_data_time:.3f}s ({data_fps:.1f} fps) | "
                    f"Compute: {avg_step_time:.3f}s ({step_fps:.1f} fps)"
                )

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        # Compute epoch statistics
        avg_loss = epoch_loss_sum / num_updates
        total_frames = cfg.train.batch_per_gpu * cfg.tokenizer.context_length * world_size * STEPS_PER_EPOCH
        epoch_fps_val = total_frames / epoch_time
        avg_data_time_epoch = sum(data_times) / len(data_times)

        epoch_times.append(epoch_time)
        epoch_losses.append(avg_loss)
        epoch_fps.append(epoch_fps_val)
        epoch_data_times.append(avg_data_time_epoch)

        # --- TEST PHASE ---
        optim.zero_grad(set_to_none=True)
        tokenizer.eval()

        # Even with shuffle=False, set_epoch is recommended for DistributedSampler
        test_sampler.set_epoch(epoch)

        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_lpips_sum = 0.0
        val_count = 0  # number of windows on this rank

        with torch.no_grad():
            for step_idx, batch in enumerate(test_loader):
                images = batch['image'].to(device, non_blocking=True).to(torch.bfloat16)
                # actions are present for future conditioning but unused in tokenizer reconstruction training.
                actions = batch['action'].to(device, non_blocking=True).to(torch.bfloat16)

                x_hat = tokenizer(images)

                # Raw validation loss (no RMS normalization needed for monitoring)
                mse_loss = nn.functional.mse_loss(x_hat, images)

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    images_lpips = (images * 2.0) - 1.0
                    x_hat_lpips = (x_hat * 2.0) - 1.0
                    Bv, Tv, C, H, W = images_lpips.shape
                    images_lpips_flat = images_lpips.view(Bv * Tv, C, H, W)
                    x_hat_lpips_flat = x_hat_lpips.view(Bv * Tv, C, H, W)
                    lpips_val = lpips_model(x_hat_lpips_flat, images_lpips_flat)
                    lpips_loss = lpips_val.mean()

                val_loss = mse_loss + 0.2 * lpips_loss

                # accumulate local sums
                val_loss_sum += val_loss.item()
                val_mse_sum += mse_loss.item()
                val_lpips_sum += lpips_loss.item()
                val_count += 1

        # convert to tensors and sum across ranks
        loss_sum_t = torch.tensor([val_loss_sum], device=device)
        mse_sum_t  = torch.tensor([val_mse_sum], device=device)
        lpips_sum_t = torch.tensor([val_lpips_sum], device=device)
        count_t    = torch.tensor([val_count], device=device, dtype=torch.long)

        dist.all_reduce(loss_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(mse_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(lpips_sum_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(count_t, op=dist.ReduceOp.SUM)

        total_count = max(int(count_t.item()), 1)

        avg_val_loss = loss_sum_t.item() / total_count
        avg_val_mse  = mse_sum_t.item() / total_count
        avg_val_lpips = lpips_sum_t.item() / total_count

        if rank == 0:
            tb_writer.add_scalar("val/raw_loss", avg_val_loss, epoch + 1)
            tb_writer.add_scalar("val/mse", avg_val_mse, epoch + 1)
            tb_writer.add_scalar("val/lpips", avg_val_lpips, epoch + 1)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{cfg.train.num_epochs} Summary:")
            print(f" Train Loss: {avg_loss:.6f}")
            print(f" Val Loss:   {avg_val_loss:.6f}")
            print(f"  Epoch Time: {epoch_time:.2f}s")
            print(f"  Total Frames: {total_frames:,}")
            print(f"  Throughput: {epoch_fps_val:.2f} FPS")
            print(f"  Avg Data Time / step: {avg_data_time_epoch:.3f}s")
            print(f"  Avg Step Time: {sum(step_times)/len(step_times):.3f}s")
            print(f"{'='*60}\n")
        print(f"Saving checkpoint for epoch {epoch+1} at path: {ckpt_path}")
        save_fsdp_checkpoint(
            ckpt_path=os.path.join(log_dir, f"{global_update}.pt"),
            epoch=epoch,
            global_update=global_update,
            model = tokenizer,
            optim=optim,
            scheduler=scheduler,
            rms_norm=rms_norm,
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
