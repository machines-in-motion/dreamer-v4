import os
import time
import hydra
import torch
import torch.nn as nn
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from torch.nn.parallel import DistributedDataParallel as DDP
from model.tokenizer import (CausalTokenizerDecoder, 
                             CausalTokenizerEncoder, 
                             CausalTokenizerConfig, 
                             LinearTokensToImageHead,
                             TokensToImageHead, 
                             ImagePatchifier)
from model.blocks import EfficientTransformerBlock
from model.utils import TokenMasker

from dataset import ShardedHDF5Dataset
import lpips
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils as nn_utils
import wandb
import datetime
import math

class ModelWrapper(nn.Module):
    def __init__(self, cfg:DictConfig):
        super().__init__()
        self.cfg = cfg
        tokenizer_cfg = CausalTokenizerConfig(**OmegaConf.to_object(cfg.tokenizer)) 
        self.encoder = CausalTokenizerEncoder(tokenizer_cfg)
        self.decoder = CausalTokenizerDecoder(tokenizer_cfg)
        self.patchifier = ImagePatchifier(cfg.tokenizer.patch_size, cfg.tokenizer.model_dim)
        self.image_head = LinearTokensToImageHead(cfg.tokenizer.model_dim, cfg.dataset.resolution, cfg.tokenizer.patch_size)
        self.masker = TokenMasker(cfg.tokenizer.model_dim, cfg.tokenizer.num_modality_tokens)

    def forward(self, images):
        images = (images*2.)-1. # Translate the images in +-1 range
        tokens = self.patchifier(images)
        masked_tokens = self.masker(tokens)
        z, _ = self.encoder(masked_tokens)
        z_decoded = self.decoder(z)
        recon_images = self.image_head(z_decoded)
        return  torch.clamp((recon_images + 1)/2., 0., 1.)

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


# ToDo: Save the state dicts of the model modules independently
def save_checkpoint(
    ckpt_path: str,
    epoch: int,
    global_update: int,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scheduler,
    rms_norm: RMSLossScaler,
    rank: int,
    wandb_run_id: str = None,
    log_dir: str = None,
):
    """
    Save a FULL checkpoint (unsharded enc/dec) on rank 0.
    """

    # Only rank 0 saves to disk
    if rank == 0:
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
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

def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
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
    # DDP: load into wrapped model
    state_dict = ckpt["model"]

    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

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

@hydra.main(config_path="config", config_name="tokenizer.yaml")
def main(cfg: DictConfig):
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
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
    IMG_H, IMG_W = 256, 256
    CONTEXT_T = 96
    N_LATENTS = 512
    BOTTLENECK_D = 16
    D_MODEL = 768
    N_LAYERS = 12
    HEADS_Q = 12
    HEADS_KV_LATENT = 16
    MLP_RATIO = 4.0
    TEMPORAL_EVERY = 4
    
    # Batch size per GPU
    BATCH_PER_GPU = 1
    NUM_EPOCHS = 3
    STEPS_PER_EPOCH = 50  # Limit steps per epoch for faster benchmarking
    GRAD_ACCUM_STEPS = 1  # simulate batch size 10 per GPU
    effective_global_batch = BATCH_PER_GPU * world_size * GRAD_ACCUM_STEPS
    if rank == 0:
        print(f"Effective global batch size: {effective_global_batch}")

    USE_COMPILE = True
    CHECKPOINT_EVERY_N_UPDATES = 1000

    # Create models
    if rank == 0:
        print("Creating encoder and decoder models...")
    
    model = ModelWrapper(cfg)
    model.to(device)

    if USE_COMPILE:
        model = torch.compile(model, mode="max-autotune", fullgraph=True)

    # Print parameter counts
    if rank == 0:
        learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {learnable_params:,}")
    

    train_loader, train_sampler, train_dataset = create_distributed_dataloader(
        data_dir="/scratch/ja5009/soar_data_sharded/",
        window_size=CONTEXT_T,
        batch_size=BATCH_PER_GPU,
        rank=rank,
        world_size=world_size,
        num_workers=4,
        stride=1,
        seed=42,
        split="train",
        train_fraction=0.9,
        split_seed=123,   # controls which episodes go to train/test
        shuffle=True,
        drop_last=True,
    )

    test_loader, test_sampler, test_dataset = create_distributed_dataloader(
        data_dir="/scratch/ja5009/soar_data_sharded/",
        window_size=CONTEXT_T,
        batch_size=BATCH_PER_GPU,
        rank=rank,
        world_size=world_size,
        num_workers=4,
        stride=1,
        seed=42,          # seed for sampler sharding (not for split itself)
        split="test",
        train_fraction=0.9,
        split_seed=123,   # must match train_dataset
        shuffle=False,    # evaluation: deterministic order
        drop_last=False,  # keep all test windows
    )

    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
        broadcast_buffers=True,
    )
    # Create optimizer
    params = model.parameters()
    optim = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01) # try increasing to 0.03–0.05? (apparently normal range for tokenizer)

    steps_per_epoch = len(train_loader)
    STEPS_PER_EPOCH = steps_per_epoch
    total_steps = NUM_EPOCHS * steps_per_epoch // GRAD_ACCUM_STEPS
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    # RMS loss normalizer for tokenizer losses
    rms_norm = RMSLossScaler(decay=0.99, eps=1e-8)

    RESUME_TRAINING = False
    wandb_run_id = None
    log_dir = None
    ckpt_path = None
    start_epoch = 0
    global_update = 0  # counts optimizer updates
    if RESUME_TRAINING:
        # Try loading checkpoint BEFORE initializing W&B
        start_epoch, global_update, wandb_run_id, log_dir = load_checkpoint(
            ckpt_path=ckpt_path,
            model= model,
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
            log_dir = f"./logs/dreamer_v4_tokenizer/{now.strftime('%Y-%m-%d_%H-%M-%S')}"
            os.makedirs(log_dir, exist_ok=True)
            ckpt_path = os.path.join(log_dir, "latest.pt")
        else:
            # Resuming: reuse existing log_dir
            print(f"Reusing log directory: {log_dir}")

        # Initialize W&B with resume logic
        if wandb_run_id is not None:
            # Resuming an existing run
            wandb.init(
                project="dreamer-v4-tokenizer",
                id=wandb_run_id,
                resume="must",  # or "must" if you want to error if run doesn't exist
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
        else:
            # Starting a new run
            wandb.init(
                project="dreamer-v4-tokenizer",
                name=f"run_nodes{world_size//8}_gpus{world_size}",
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
    
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        train_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        epoch_loss_sum = 0.0
        step_times = []
        data_times = []

        num_updates = 0
        accum_mse = 0.0
        accum_lpips = 0.0
        accum_raw_loss = 0.0
        accum_norm = 0.0
        torch.cuda.synchronize(device=None)
        data_start = time.perf_counter()

        for step_idx, batch in enumerate(train_loader):
            if num_updates >= STEPS_PER_EPOCH:
                break
            # ------------------------------
            # Measure data loading time
            # ------------------------------
            torch.cuda.synchronize(device=None)
            data_time = time.perf_counter() - data_start
            data_times.append(data_time)

            is_first_micro = (step_idx % GRAD_ACCUM_STEPS) == 0
            is_update_step = (step_idx + 1) % GRAD_ACCUM_STEPS == 0

            if is_first_micro:
                torch.compiler.cudagraph_mark_step_begin() 
                optim.zero_grad(set_to_none=True)

            # Move data
            images = batch['image'].to(device, non_blocking=True).to(torch.bfloat16)

            # ------------------------------
            # Forward
            # ------------------------------
            torch.cuda.synchronize(device=None)
            step_start = time.perf_counter()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                torch.cuda.synchronize(device=None)
                forward_time_tik = time.perf_counter()
                x_hat = model(images)
                torch.cuda.synchronize(device=None)
                forward_time = time.perf_counter() - forward_time_tik

                mse_loss = nn.functional.mse_loss(x_hat, images)

                # disabled LPIPS for now
                lpips_loss = torch.zeros_like(mse_loss)

            raw_loss = mse_loss + 0.2 * lpips_loss

            # update CPU-side accumulators
            accum_mse += mse_loss.item()
            accum_lpips += lpips_loss.item()
            accum_raw_loss += raw_loss.item()

            # normalized losses
            mse_norm = rms_norm("mse", mse_loss)
            lpips_norm = rms_norm("lpips", lpips_loss)

            loss = (mse_norm + 0.2 * lpips_norm) / GRAD_ACCUM_STEPS
            accum_norm += loss.item()
            torch.cuda.synchronize(device=None)
            backward_time_tik = time.perf_counter()
            # backward
            loss.backward()
            torch.cuda.synchronize(device=None)
            backward_time = time.perf_counter() - backward_time_tik

            # ------------------------------
            # UPDATE STEP
            # ------------------------------
            if is_update_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.cuda.synchronize(device=None)
                optim_step_tic = time.perf_counter()
                optim.step()
                scheduler.step()
                torch.cuda.synchronize(device=None)
                optim_step_time = time.perf_counter() - optim_step_tic

                global_update += 1
                num_updates += 1

                # ————— Sync every LOG_INTERVAL —————
                if global_update % 100 == 0:
                    metrics = torch.tensor(
                        [accum_norm, accum_mse, accum_lpips, accum_raw_loss],
                        device=device,
                        dtype=torch.float32,
                    )
                    dist.all_reduce(metrics, op=dist.ReduceOp.AVG)
                    sync_loss, mse_sum, lpips_sum, raw_sum = metrics.tolist()

                    epoch_loss_sum += sync_loss
                    torch.cuda.synchronize(device=None)
                    step_end = time.perf_counter()
                    if rank == 0:
                        mse_mean = mse_sum / GRAD_ACCUM_STEPS
                        lpips_mean = lpips_sum / GRAD_ACCUM_STEPS
                        raw_mean = raw_sum / GRAD_ACCUM_STEPS

                        tb_writer.add_scalar("train/mse_mean", mse_mean, global_update)
                        tb_writer.add_scalar("train/lpips_mean", lpips_mean, global_update)
                        tb_writer.add_scalar("train/raw_loss_mean", raw_mean, global_update)
                        tb_writer.add_scalar("train/normalized_loss_mean", sync_loss, global_update)
                        tb_writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_update)
                        tb_writer.add_scalar("timing/forward_time", forward_time, global_update)
                        tb_writer.add_scalar("timing/backward_time", backward_time, global_update)
                        tb_writer.add_scalar("timing/optim_step_time", optim_step_time, global_update)
                        tb_writer.add_scalar("timing/overall_step_time", step_end - step_start, global_update)
                        tb_writer.add_scalar("timing/data_time", data_time, global_update)

                # Reset accumulators
                accum_mse = accum_lpips = accum_raw_loss = accum_norm = 0.0

            # ------------------------------
            # End-of-step timing
            # ------------------------------
            torch.cuda.synchronize(device=None)
            step_end = time.perf_counter()
            step_times.append(step_end - step_start)

            # Next data load timing begins
            torch.cuda.synchronize(device=None)
            data_start = time.perf_counter()


            # Print progress every 200 steps
            # if rank == 0 and is_update_step and (global_update % 200 == 0):
            #     avg_step_time = sum(step_times[-200:]) / len(step_times[-200:])
            #     avg_data_time = sum(data_times[-200:]) / len(data_times[-200:])
            #     frames_per_step = BATCH_PER_GPU * CONTEXT_T * world_size
            #     step_fps = frames_per_step / avg_step_time
            #     data_fps = frames_per_step / avg_data_time if avg_data_time > 0 else float('inf')
            #     print(
            #         f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            #         f"Step {step_idx+1}/{STEPS_PER_EPOCH} | "
            #         f"Loss: {sync_loss:.6f} | "
            #         f"Data: {avg_data_time:.3f}s ({data_fps:.1f} fps) | "
            #         f"Compute: {avg_step_time:.3f}s ({step_fps:.1f} fps)"
            #     )





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

        # --- TEST PHASE ---
        optim.zero_grad(set_to_none=True)
        model.eval()

        # Even with shuffle=False, set_epoch is recommended for DistributedSampler
        test_sampler.set_epoch(epoch)

        val_loss_sum = 0.0
        val_mse_sum = 0.0
        val_lpips_sum = 0.0
        val_count = 0  # number of windows on this rank

        with torch.no_grad():
            for step_idx, batch in enumerate(test_loader):
                images = batch['image'].to(device, non_blocking=True).to(torch.bfloat16)
                actions = batch['action'].to(device, non_blocking=True).to(torch.bfloat16)

                x_hat = model(images)

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
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} Summary:")
            print(f" Train Loss: {avg_loss:.6f}")
            print(f" Val Loss:   {avg_val_loss:.6f}")
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
            model = model,
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
