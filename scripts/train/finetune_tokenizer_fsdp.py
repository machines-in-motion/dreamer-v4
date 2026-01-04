import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
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

from models import (
    DreamerV4Encoder,
    DreamerV4Decoder,
    BlockCausalEncoderLayer,
    BlockCausalDecoderLayer,
)

from dataset import HDF5SequenceDataset
import lpips
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime
import math

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

def save_checkpoint(
    ckpt_path: str,
    epoch: int,
    global_update: int,
    enc: FSDP,
    dec: FSDP,
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

    with FSDP.state_dict_type(enc, StateDictType.FULL_STATE_DICT, full_cfg):
        enc_state = enc.state_dict()  # All ranks participate in gather
    with FSDP.state_dict_type(dec, StateDictType.FULL_STATE_DICT, full_cfg):
        dec_state = dec.state_dict()  # All ranks participate in gather

    # Only rank 0 saves to disk
    if rank == 0:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        
        ckpt = {
            "epoch": epoch,
            "global_update": global_update,
            "enc": enc_state,
            "dec": dec_state,
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rms_norm": rms_norm.ema_sq,
            "wandb_run_id": wandb_run_id,
            "log_dir": log_dir,
        }

        torch.save(ckpt, ckpt_path)
        print(f"[rank0] Saved checkpoint to {ckpt_path}")

def load_checkpoint(
    ckpt_path: str,
    enc: FSDP,
    dec: FSDP,
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

    with FSDP.state_dict_type(enc, StateDictType.FULL_STATE_DICT, full_cfg):
        enc.load_state_dict(ckpt["enc"])
    with FSDP.state_dict_type(dec, StateDictType.FULL_STATE_DICT, full_cfg):
        dec.load_state_dict(ckpt["dec"])

    # Restore RMS loss scaler state
    rms_norm.ema_sq = ckpt.get("rms_norm", {})

    if rank == 0:
        print(f"Resuming from epoch {start_epoch+1}, global_update {global_update}")
        if wandb_run_id:
            print(f"Resuming W&B run ID: {wandb_run_id}")
        if log_dir:
            print(f"Resuming log directory: {log_dir}")

    return start_epoch, global_update, wandb_run_id, log_dir

def setup_distributed():
    """
    Initialize distributed process group using SLURM environment variables.
    SLURM sets these automatically when using srun with --ntasks-per-node.
    """
    # SLURM sets these environment variables automatically
    rank = int(os.environ['SLURM_PROCID'])           # Global rank
    local_rank = int(os.environ['SLURM_LOCALID'])    # Local rank on node
    world_size = int(os.environ['SLURM_NTASKS'])     # Total number of tasks
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group with NCCL backend
    # MASTER_ADDR and MASTER_PORT should be set in SLURM script
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    return rank, local_rank, world_size

def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()

def get_fsdp_wrap_policy():
    """Define which layers to wrap with FSDP."""
    return partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            BlockCausalEncoderLayer,
            BlockCausalDecoderLayer,
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
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    Create DataLoader with DistributedSampler for sharded HDF5 dataset.
    """
    # Create the dataset with a fixed split
    dataset = HDF5SequenceDataset(
        data_dir=data_dir,
        window_size=window_size,
        stride=stride,
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

def main():
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
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
    torch.manual_seed(42 + rank)
    
    # Model hyperparameters
    IMG_H, IMG_W = 128, 128 # 256, 256
    PATCH = 16
    CONTEXT_T = 96
    N_LATENTS = 512
    BOTTLENECK_D = 16
    D_MODEL = 768
    N_LAYERS = 12
    HEADS_Q = 12
    HEADS_KV_LATENT = 12
    MLP_RATIO = 4.0
    TEMPORAL_EVERY = 4
    IN_CH = 3
    
    # Batch size per GPU
    BATCH_PER_GPU = 2 # 1
    T = CONTEXT_T
    NUM_EPOCHS = 210
    #STEPS_PER_EPOCH = 100  # Limit steps per epoch for faster benchmarking
    GRAD_ACCUM_STEPS = 5 # 10  # simulate batch size 10 per GPU
    effective_global_batch = BATCH_PER_GPU * world_size * GRAD_ACCUM_STEPS
    if rank == 0:
        print(f"Effective global batch size: {effective_global_batch}")

    USE_COMPILE = True
    CHECKPOINT_EVERY_N_UPDATES = 1000

    # Create models
    if rank == 0:
        print("Creating encoder and decoder models...")
    
    enc = DreamerV4Encoder(
        image_size=(IMG_H, IMG_W),
        patch_size=PATCH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        num_heads_q=HEADS_Q,
        num_heads_kv_latent=HEADS_KV_LATENT,
        seq_len=CONTEXT_T,
        mlp_ratio=MLP_RATIO,
        dropout=0.0,
        n_latents=N_LATENTS,
        bottleneck_dim=BOTTLENECK_D,
        temporal_every=TEMPORAL_EVERY,
        in_channels=IN_CH,
        mae_max_mask_prob=0.9,  # enable MAE
    )
    
    dec = DreamerV4Decoder(
        image_size=(IMG_H, IMG_W),
        patch_size=PATCH,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        num_heads_q=HEADS_Q,
        num_heads_kv_latent=HEADS_KV_LATENT,
        bottleneck_dim=BOTTLENECK_D,
        seq_len=CONTEXT_T,
        mlp_ratio=MLP_RATIO,
        dropout=0.0,
        n_latents=N_LATENTS,
        in_channels=IN_CH,
        temporal_every=TEMPORAL_EVERY,
    )

    # Move to correct device before compile (recommended ordering)
    enc = enc.to(device)
    dec = dec.to(device)

    if USE_COMPILE:
        # Good starting config; you can try other modes later
        enc = torch.compile(enc, mode="max-autotune", fullgraph=False)
        dec = torch.compile(dec, mode="max-autotune", fullgraph=False)

    # Print parameter counts
    if rank == 0:
        learnable_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
        learnable_params += sum(p.numel() for p in dec.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {learnable_params:,}")
    
    # Wrap with FSDP
    if rank == 0:
        print("Wrapping models with FSDP...")

    train_loader, train_sampler, train_dataset = create_distributed_dataloader(
        data_dir="/scratch/ja5009/processed_logs/",
        window_size=CONTEXT_T,
        batch_size=BATCH_PER_GPU,
        rank=rank,
        world_size=world_size,
        num_workers=4,
        stride=1,
        seed=42,
        shuffle=True,
        drop_last=True,
    )

    enc = setup_fsdp_model(enc, mixed_precision=True, sharding_strategy="FULL_SHARD")
    dec = setup_fsdp_model(dec, mixed_precision=True, sharding_strategy="FULL_SHARD")

    # Create optimizer
    params = list(enc.parameters()) + list(dec.parameters())
    optim = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01) # try increasing to 0.03–0.05? (apparently normal range for tokenizer)

    steps_per_epoch = len(train_loader)
    STEPS_PER_EPOCH = steps_per_epoch
    total_steps = NUM_EPOCHS * steps_per_epoch // GRAD_ACCUM_STEPS
    warmup_steps = 1400 # int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # RMS loss normalizer for tokenizer losses
    rms_norm = RMSLossScaler(decay=0.99, eps=1e-8)

    wandb_run_id = None
    log_dir = None
    #ckpt_path = "./logs/dreamer_v4_tokenizer/2025-11-26_15-52-05/latest.pt"
    ckpt_path = "./logs/dreamer_v4_tokenizer/2025-11-18_08-12-02/latest.pt"
    start_epoch = 0
    global_update = 0  # counts optimizer updates
    _, _, _, _ = load_checkpoint(
        ckpt_path=ckpt_path,
        enc=enc,
        dec=dec,
        rms_norm=rms_norm,
        rank=rank,
        device=device,
    )

    # TensorBoard + wandb (rank 0 only)
    if rank == 0:
        # Use stable log_dir if resuming, otherwise create new one
        now = datetime.datetime.now()
        log_dir = f"./logs/dreamer_v4_tokenizer/{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(log_dir, exist_ok=True)
        ckpt_path = os.path.join(log_dir, "latest.pt")

        tb_log_dir = os.path.join(log_dir, "tensorboard")
        os.makedirs(tb_log_dir, exist_ok=True)

        # Initialize W&B
        if True:
            # Starting a new run
            wandb.init(
                project="dreamer-v4-tokenizer",
                name=f"finetune_run_nodes{world_size//8}_gpus{world_size}",
                config={
                    "img_h": IMG_H,
                    "img_w": IMG_W,
                    "context_t": CONTEXT_T,
                    "n_latents": N_LATENTS,
                    "bottleneck_d": BOTTLENECK_D,
                    "d_model": D_MODEL,
                    "n_layers": N_LAYERS,
                    "heads_q": HEADS_Q,
                    "heads_kv_latent": HEADS_KV_LATENT,
                    "mlp_ratio": MLP_RATIO,
                    "temporal_every": TEMPORAL_EVERY,
                    "batch_per_gpu": BATCH_PER_GPU,
                    "grad_accum_steps": GRAD_ACCUM_STEPS,
                    "lr": 1e-4,
                    "weight_decay": 0.01,
                    "use_compile": USE_COMPILE,
                },
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
        print("Starting warmup iterations...")

    # Track statistics
    epoch_times = []
    epoch_losses = []
    epoch_fps = []
    epoch_data_times = []

    for epoch in range(start_epoch, NUM_EPOCHS):
        # --- TRAIN PHASE ---
        enc.train()
        dec.train()

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
            #if step_idx > STEPS_PER_EPOCH:
            #    break
            micro_idx = step_idx % GRAD_ACCUM_STEPS
            is_last_micro = (micro_idx == GRAD_ACCUM_STEPS - 1)

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

            P_enc, L_enc, Z = enc(images)
            R_dec, x_hat = dec(Z)

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
            loss = (mse_norm + 0.2 * lpips_norm) / GRAD_ACCUM_STEPS
            accum_norm = accum_norm + loss.detach().item()

            # Backward pass
            loss.backward()

            # If this is the last micro-batch in the window, do optimizer step
            if is_last_micro:
                #torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                enc.clip_grad_norm_(max_norm=1.0)
                dec.clip_grad_norm_(max_norm=1.0)

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

                mse_mean = mse_sum / GRAD_ACCUM_STEPS
                lpips_mean = lpips_sum / GRAD_ACCUM_STEPS
                raw_mean = raw_sum / GRAD_ACCUM_STEPS

                if rank == 0:
                    # Sums over micro-batches in this update
                    tb_writer.add_scalar("train/mse_mean", mse_mean, global_update)
                    tb_writer.add_scalar("train/lpips_mean", lpips_mean, global_update)
                    tb_writer.add_scalar("train/raw_loss_mean", raw_mean, global_update)
                    tb_writer.add_scalar("train/normalized_loss_mean", sync_loss, global_update)
                    current_lr = scheduler.get_last_lr()[0]
                    tb_writer.add_scalar("train/lr", current_lr, global_update)

                # Checkpoint every N updates (within epoch)
                if global_update % CHECKPOINT_EVERY_N_UPDATES == 0:
                    if rank == 0:
                        print(f"[Checkpoint] Saving at global_update={global_update}")
                    save_checkpoint(
                        ckpt_path=ckpt_path,
                        epoch=epoch,
                        global_update=global_update,
                        enc=enc,
                        dec=dec,
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

            # Next data timing starts now (time until next batch is yielded)
            data_start = time.perf_counter()

            # Print progress every 200 steps
            if rank == 0 and is_last_micro and (global_update % 200 == 0):
                avg_step_time = sum(step_times[-200:]) / len(step_times[-200:])
                avg_data_time = sum(data_times[-200:]) / len(data_times[-200:])

                frames_per_step = BATCH_PER_GPU * CONTEXT_T * world_size
                step_fps = frames_per_step / avg_step_time
                data_fps = frames_per_step / avg_data_time if avg_data_time > 0 else float('inf')

                print(
                    f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                    f"Step {step_idx+1}/{STEPS_PER_EPOCH} | "
                    f"Loss: {sync_loss:.6f} | "
                    f"Data: {avg_data_time:.3f}s ({data_fps:.1f} fps) | "
                    f"Compute: {avg_step_time:.3f}s ({step_fps:.1f} fps)"
                )

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start

        # Compute epoch statistics
        avg_loss = epoch_loss_sum / num_updates
        total_frames = BATCH_PER_GPU * CONTEXT_T * world_size * STEPS_PER_EPOCH
        epoch_fps_val = total_frames / epoch_time
        avg_data_time_epoch = sum(data_times) / len(data_times)

        epoch_times.append(epoch_time)
        epoch_losses.append(avg_loss)
        epoch_fps.append(epoch_fps_val)
        epoch_data_times.append(avg_data_time_epoch)

        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
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
            enc=enc,
            dec=dec,
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
        print(f"Overall Statistics (across {NUM_EPOCHS} epochs):")
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
        for i in range(NUM_EPOCHS):
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
