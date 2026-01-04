import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from models import (
    DreamerV4Encoder,
    BlockCausalEncoderLayer,
    DreamerV4Dynamics,
    BlockCausalDynamicsLayer
)

from dataset import HDF5SequenceDataset
from torch.utils.tensorboard import SummaryWriter
import wandb
import datetime
import math
from datetime import timedelta
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP

class RMSLossScaler:
    """
    Tracks running RMS for named losses and returns normalized losses.
    """
    def __init__(self, decay: float = 0.99, eps: float = 1e-8):
        self.decay = decay
        self.eps = eps
        self.ema_sq = {}  # name -> scalar tensor

    def __call__(self, name: str, value: torch.Tensor) -> torch.Tensor:
        # value is a scalar loss tensor
        # We assume value is already averaged over the batch/tokens, 
        # but to compute RMS we strictly need the squared magnitude.
        
        with torch.no_grad():
            # If value is a mean loss, value^2 is approx mean(x^2) if var is low, 
            # but ideally we'd want mean(x^2). For scalar losses, value^2 is what we have.
            sq = value.detach().pow(2)
            
            # Sync across ranks if DDP
            if dist.is_initialized():
                dist.all_reduce(sq, op=dist.ReduceOp.AVG)

            if name not in self.ema_sq:
                self.ema_sq[name] = sq
            else:
                self.ema_sq[name] = (
                    self.decay * self.ema_sq[name] + (1.0 - self.decay) * sq
                )

            rms = (self.ema_sq[name] + self.eps).sqrt()

        return value / rms

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
    train_reward_model=False
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
    wandb_run_id = ckpt.get("wandb_run_id", None)
    log_dir = ckpt.get("log_dir", None)

    # LOAD STATE DICT with STRICT=FALSE
    # This allows loading the dynamics backbone even if reward_head is missing
    missing, unexpected = dyn.module.load_state_dict(ckpt["dyn"], strict=False)
    
    if rank == 0:
        if len(missing) > 0:
            print(f"[Warning] Missing keys in checkpoint: {missing}")
            if train_reward_model:
                print("This is expected if you are adding a reward head to a pre-trained model.")
        if len(unexpected) > 0:
            # Raise Error on Unexpected Keys
            raise RuntimeError(
                f"Checkpoint contains unexpected keys: {unexpected}. "
                "This implies a model architecture mismatch (e.g., layer count changed) "
                "that cannot be safely ignored."
            )

    # OPTIMIZER HANDLING
    # If we added new parameters, the optimizer state dict will not match.
    # We must reset the optimizer state for the new parameters or skip loading.
    try:
        optim.load_state_dict(ckpt["optim"])
    except ValueError as e:
        if rank == 0:
            print(f"[Info] Optimizer state mismatch (likely due to new reward head). Resetting optimizer.")
        # We simply do NOT load the optimizer state, effectively restarting optimization
        # You might want to keep the scheduler step if you want to continue LR decay,
        # but usually restarting/finetuning implies a fresh LR schedule or a specific one.
        pass

    scheduler.load_state_dict(ckpt["scheduler"])

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
    

    print(os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
    # Initialize process group with NCCL backend
    # MASTER_ADDR and MASTER_PORT should be set in SLURM script
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank,
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

@torch.no_grad()
def load_pretrained_tokenizer_encoder(ckpt_path: str, device: torch.device, enc_kwargs: dict, compile):
    """
    Load pretrained DreamerV4Encoder from a tokenizer checkpoint and freeze it.
    Expects a checkpoint saved by your tokenizer script with key 'enc'.
    """
    enc = DreamerV4Encoder(**enc_kwargs).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if compile:
        enc = torch.compile(enc, mode="max-autotune", fullgraph=False)
    enc_state = ckpt.get("enc", None)
    if enc_state is None:
        raise RuntimeError("Checkpoint missing 'enc' weights; provide a tokenizer ckpt saved by your tokenizer training script.")
    # Load full (unsharded) weights directly since we are not wrapping the encoder with FSDP
    enc.load_state_dict(enc_state)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc

def sample_shortcut_batch(B, T, device, micro_idx, GRAD_ACCUM_STEPS, num_tau_levels):
    """
    Samples (tau, d) parameters for Diffusion Forcing with Shortcut split.
    
    - Allocation is DETERMINISTIC based on distributed RANK.
    - First 75% of ranks -> Flow Matching (d = d_min).
    - Last 25% of ranks -> Shortcut (d > d_min).
    - Noise parameters are sampled per-step (B, T).
    
    Returns:
        tau_vals: (B,T) float
        d_vals:   (B,T) float
        tau_idxs: (B,T) long
        step_idxs: (B,T) long
    """
    # 1. Determine Task based on micro_idx
    # First 75% flow matching, rest shortcut
    cutoff = int(GRAD_ACCUM_STEPS * 0.75)
    is_flow_worker = (micro_idx < cutoff)
    
    # 2. Dynamic Grid Constants
    # If num_tau_levels = 64, d_min_idx = 6 (corresponding to 1/64 step)
    # Columns are 0..6.
    d_min_idx = int(np.log2(num_tau_levels))
    
    # Initialize indices
    tau_idxs = torch.zeros(B, T, dtype=torch.long, device=device)
    step_idxs = torch.zeros(B, T, dtype=torch.long, device=device)
    
    if is_flow_worker:
        # --- Flow Matching Case (d = d_min) ---
        # Step index is fixed to d_min_idx
        step_idxs.fill_(d_min_idx)
        
        # Valid tau rows i are 0..(2^d_min_idx - 1) i.e., 0..63
        # Sample uniformly for every step in (B, T)
        tau_idxs = torch.randint(0, num_tau_levels, (B, T), device=device)
        
    else:
        # --- Shortcut Case (d > d_min) ---
        # Sample step level j from 0..(d_min_idx - 1)
        # This corresponds to steps 1.0, 0.5, ..., strictly larger than d_min
        # (B, T) independent samples
        s_cols = torch.randint(0, d_min_idx, (B, T), device=device)
        step_idxs = s_cols
        
        # For a given col j, valid i is [0, 2^j - 1]
        # Vectorized sampling: i = floor(u * 2^j)
        max_rows = (2.0 ** s_cols) # Float tensor (B, T)
        u = torch.rand(B, T, device=device)
        
        # Clamp to ensure numerical stability doesn't push to 2^j
        s_rows = (u * max_rows).floor().long()
        tau_idxs = s_rows

    # 3. Reconstruct Float Values
    # Formula: val = (idx + 1) * (1 / 2^step)
    # We use 0.5 ** step_idxs for efficiency
    d_vals = (0.5 ** step_idxs.float())
    
    # tau = (i + 1) * d
    tau_vals = 1 - (tau_idxs.float() + 1.0) * d_vals
    
    # We need each noise level to be associated to a unique id
    real_tau_idxs = ((tau_idxs+1)*(2**(d_min_idx-step_idxs))).long()-1
    return tau_vals, d_vals, real_tau_idxs, step_idxs

def add_noise_flow_matching(z_clean, tau):
    """
    DreamerV4 / Flow Matching interpolation:
    x_tau = (1 - tau) * x_0 + tau * x_1
    where x_1 is data (z_clean), x_0 is noise.
    tau=1 -> Clean, tau=0 -> Noise.
    """
    B, T, N, D = z_clean.shape
    eps = torch.randn_like(z_clean)
    # tau shape (B,T) -> (B,T,1,1)
    tau_ex = tau.view(B, T, 1, 1)
    z_noisy = (1 - tau_ex) * eps + tau_ex * z_clean
    return z_noisy

def compute_dynamics_loss(dyn_model, actions, z_clean, reward_targets,
                          tau_vals, d_vals, tau_idxs, step_idxs, micro_idx,
                          GRAD_ACCUM_STEPS, num_tau_levels, loss_scaler):
    """
    Computes combined Flow Matching + Shortcut Loss.
    Determines Flow vs Shortcut based on Rank (same logic as sampler).
    """
    B, T, N, D = z_clean.shape

    tau_vals = tau_vals.to(dtype=z_clean.dtype)
    d_vals = d_vals.to(dtype=z_clean.dtype)

    # 1. Re-determine Task based on micro_idx
    cutoff = int(GRAD_ACCUM_STEPS * 0.75)
    is_flow_worker = (micro_idx < cutoff)
    d_min_idx = int(np.log2(num_tau_levels))

    # 2. Create Noisy Input
    z_input = add_noise_flow_matching(z_clean, tau_vals)

    # 3. Student Prediction
    pred_z1_student, pred_rewards = dyn_model(actions, z_input, tau_idxs, step_idxs)

    # 4. Ramp Weighting
    # w(tau) = 0.9 * tau + 0.1
    # Weight is applied to the squared error before averaging
    weights = 0.9 * tau_vals + 0.1
    # Expand for broadcast: (B, T) -> (B, T, 1, 1)
    weights = weights.view(*tau_vals.shape, 1, 1)

    loss_flow = 0.0
    loss_shortcut = 0.0
    loss_reward = 0.0 # Normalized reward loss
    loss_reward_raw = 0.0

    if dyn_model.module.train_reward_model and pred_rewards is not None:
        # pred_rewards: (B, T, L, Buckets)
        # reward_targets: (B, T) scalar
        
        B, T, L, Buckets = pred_rewards.shape
        
        # 1. Construct Rolling Targets (B, T, L)
        # We efficiently create windows [r_t, r_{t+1}, ..., r_{t+L-1}]
        # Pad with zeros to handle the end of sequence
        if reward_targets.dim() == 3 and reward_targets.shape[-1] == 1:                                 │        high_weight = indices - low.float()
            reward_targets = reward_targets.squeeze(-1)

        pad_len = L - 1
        rewards_padded = F.pad(reward_targets, (0, pad_len), value=0.0) # (B, T + L - 1)

        # Create a mask for valid data
        valid_mask = torch.ones_like(reward_targets, dtype=torch.bool) # (B, T)
        # Pad mask with False (invalid)
        valid_mask_padded = F.pad(valid_mask, (0, pad_len), value=False)
        # Unfold mask to match targets (B, T, L)
        mask_windowed = valid_mask_padded.unfold(dimension=1, size=L, step=1)

        # Unfold creates the windows
        # targets_windowed: (B, T, L)
        targets_windowed = rewards_padded.unfold(dimension=1, size=L, step=1)
        
        # 2. Compute Symlog Two-Hot Targets
        # We access the helper from the first head (they share bucket logic)
        head_module = dyn_model.module.reward_head.heads[0]
        
        # These returns are (B, T, L) indices and weights
        low, low_w, high, high_w = head_module.get_targets(targets_windowed)
        
        # 3. Cross Entropy Loss
        # We interpret pred_rewards as logits
        # Loss = -sum(target * log_prob)
        log_probs = F.log_softmax(pred_rewards, dim=-1) # (B, T, L, Buckets)
        
        # Gather log probs for the low and high buckets
        # low.unsqueeze(-1) -> (B, T, L, 1)
        lp_low = log_probs.gather(-1, low.unsqueeze(-1)).squeeze(-1)
        lp_high = log_probs.gather(-1, high.unsqueeze(-1)).squeeze(-1)
        
        # Two-hot cross entropy
        loss_per_step = -(low_w * lp_low + high_w * lp_high) # (B, T, L)

        # Apply mask to loss
        # loss_per_step is (B, T, L)
        loss_per_step = loss_per_step * mask_windowed.float()

        # 3. Normalize correctly (divide by number of valid elements)
        raw_r_loss = loss_per_step.sum() / mask_windowed.sum().clamp(min=1.0)
        # OR if you prefer per-sample weighting:
        # raw_r_loss = (loss_per_step.sum(dim=2) / mask_windowed.sum(dim=2).clamp(min=1.0)).mean()

        # Apply RMS Scaler
        loss_reward = loss_scaler("loss_reward", raw_r_loss)
        loss_reward_raw = raw_r_loss.detach().item()

    if is_flow_worker:
        # --- Flow Matching Loss (d = d_min) ---
        # Target: z_clean
        diff = pred_z1_student - z_clean
        raw_flow_loss = (weights * (diff ** 2)).mean()
        # Weighted MSE
        # Apply RMS Scaler
        loss_flow_norm = loss_scaler("loss_flow", raw_flow_loss)
        
        total_loss = loss_flow_norm + loss_reward
        
        return total_loss, raw_flow_loss.detach().item(), 0.0, loss_reward_raw
    else:
        # --- Shortcut Loss (d > d_min) ---
        # This worker is dedicated to Shortcut logic.
        # We assume the WHOLE batch on this GPU is shortcut data 

        with torch.no_grad():
            inner_model = dyn_model.module if hasattr(dyn_model, "module") else dyn_model
            # Calculate Next Half-Step Params
            # d_half = d / 2
            d_half = d_vals / 2.0

            # Step index for d/2 is step_idxs + 1
            step_idxs_half = step_idxs + 1

            tau_idxs_1 = tau_idxs

            z_in_short = z_input

            # --- Teacher 1: Start -> Middle ---
            # Pred z1 from (z_tau, tau, d/2)
            pred_z1_t1, _ = inner_model(
                actions,
                z_in_short,
                tau_idxs_1,
                step_idxs_half
            )

            # Velocity b' = (pred - z) / (1 - tau)
            # Prevent div by zero if tau=1 (though shortcut usually < 1)
            denom_t1 = (1.0 - tau_vals.view(B, T, 1, 1) + 1e-6)
            b_prime = (pred_z1_t1 - z_in_short) / denom_t1

            # Advance state: z' = z + b' * d/2
            z_prime = z_in_short + b_prime * d_half.view(B, T, 1, 1)

            # --- Teacher 2: Middle -> End ---
            # New tau = tau + d/2
            tau_new_vals = tau_vals + d_half

            # New unique ID for (tau + d/2)
            tau_idxs_2 = tau_idxs_1 - (2 ** (d_min_idx - step_idxs_half)).long()

            pred_z1_t2, _ = inner_model(
                actions,
                z_prime,
                tau_idxs_2,
                step_idxs_half
            )

            # Velocity b'' = (pred - z') / (1 - tau_new)
            denom_t2 = (1.0 - tau_new_vals.view(B, T, 1, 1) + 1e-6)
            b_double_prime = (pred_z1_t2 - z_prime) / denom_t2

            # Target Velocity
            v_target = (b_prime + b_double_prime) / 2.0

            # target_update = (1 - tau) * v_target
            # This cancels out the denominator in the student's velocity term.
            target_update = (1.0 - tau_vals.view(B, T, 1, 1)) * v_target

        # Student Loss
        # -------------------------------------------------
        # Loss = || (z_student - z_tau) - target_update ||^2
        student_update = pred_z1_student - z_input

        diff = student_update - target_update

        # Apply Ramp Weighting
        raw_short_loss = (weights * (diff ** 2)).mean()

        # Apply RMS Scaler
        loss_short_norm = loss_scaler("loss_shortcut", raw_short_loss)
        
        total_loss = loss_short_norm + loss_reward

        return total_loss, 0.0, raw_short_loss.detach().item(), loss_reward_raw

def main():
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

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
    N_LATENTS = 256 # 512
    BOTTLENECK_D = 16
    D_MODEL = 768
    N_LAYERS = 12
    HEADS_Q = 12
    HEADS_KV_LATENT = 12
    MLP_RATIO = 4.0
    TEMPORAL_EVERY = 4
    IN_CH = 3
    NUM_TAU_LEVELS = 128
    NUM_REGISTERS = 4

    D_MODEL_DYN = 2048
    N_LAYERS_DYN = 32
    HEADS_Q_DYN = 32
    CONTEXT_T_DYN = 32 # 96

    TRAIN_REWARD_MODEL = True
    MTP_LENGTH = 8

    # Tokenizer checkpoint (encoder-only used, frozen)
    #TOKENIZER_CKPT = "./logs/dreamer_v4_tokenizer/2025-12-27_16-15-06/latest.pt"  # path to a saved tokenizer ckpt
    TOKENIZER_CKPT = "/scratch/ja5009/dreamer_v4_tokenizer/2025-12-31_11-10-46/latest.pt"

    # Batch size per GPU
    BATCH_PER_GPU = 1
    T = CONTEXT_T_DYN
    NUM_EPOCHS = 210
    #STEPS_PER_EPOCH = 20 # Limit steps per epoch for faster benchmarking
    GRAD_ACCUM_STEPS = 10 # 10  # simulate batch size 10 per GPU
    effective_global_batch = BATCH_PER_GPU * world_size * GRAD_ACCUM_STEPS
    if rank == 0:
        print(f"Effective global batch size: {effective_global_batch}")

    USE_COMPILE = True
    CHECKPOINT_EVERY_N_UPDATES = 1000

    # Create models
    if rank == 0:
        print("Creating encoder and decoder models...")

    # Load frozen encoder from tokenizer checkpoint
    enc_kwargs = dict(
        image_size=(IMG_H, IMG_W),
        patch_size=16,
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
        in_channels=3,
        mae_max_mask_prob=0.0,  # disabled at eval
    )
    enc = load_pretrained_tokenizer_encoder(TOKENIZER_CKPT, device=device, enc_kwargs=enc_kwargs, compile=True)

    train_loader, train_sampler, train_dataset = create_distributed_dataloader(
        data_dir="/scratch/ja5009/processed_logs/",
        window_size=CONTEXT_T_DYN,
        batch_size=BATCH_PER_GPU,
        rank=rank,
        world_size=world_size,
        num_workers=4,
        stride=1,
        seed=42,
        shuffle=True,
        drop_last=True,
    )

    # Peek one batch to infer action_dim and instantiate dynamics + optimizer
    first_batch = next(iter(train_loader))
    action_dim = first_batch["action"].shape[-1]
    if rank == 0:
        print(f"Inferred action_dim={action_dim} from dataset batch")

    dyn = DreamerV4Dynamics(
        action_dim=action_dim,
        num_latents=N_LATENTS,
        latent_dim=BOTTLENECK_D,
        d_model=D_MODEL_DYN,
        num_layers=N_LAYERS_DYN,
        num_heads=HEADS_Q_DYN,
        num_registers=NUM_REGISTERS,
        seq_len=CONTEXT_T_DYN,
        dropout=0.0,
        mlp_ratio=MLP_RATIO,
        num_tau_levels=NUM_TAU_LEVELS,
        temporal_every=TEMPORAL_EVERY,
        train_reward_model=TRAIN_REWARD_MODEL,
        mtp_length=MTP_LENGTH
    ).to(device)

    # Print parameter counts
    if rank == 0:
        learnable_params = sum(p.numel() for p in dyn.parameters() if p.requires_grad)
        print(f"Total learnable parameters: {learnable_params:,}")

    #enc = setup_fsdp_model(enc, mixed_precision=True, sharding_strategy="FULL_SHARD")
    enc = enc.to(torch.bfloat16)

    if USE_COMPILE:
        #dyn = torch.compile(dyn, mode="max-autotune", fullgraph=False)
        dyn = torch.compile(dyn, mode="max-autotune-no-cudagraphs", fullgraph=False)
    # dynamics: FP32 params to avoid BF16-param / FP32-grad bug
    #dyn = setup_fsdp_model(dyn, mixed_precision=True, sharding_strategy="FULL_SHARD",
    #                   fp32_params=True)
    #dyn = setup_fsdp_model(dyn, mixed_precision=True, sharding_strategy="SHARD_GRAD_OP",
    #                   fp32_params=True)
    dyn = DDP(dyn, device_ids=[local_rank], find_unused_parameters=False)

    # Create optimizer
    optim = torch.optim.AdamW(dyn.parameters(), lr=1e-4, weight_decay=0.1)

    steps_per_epoch = len(train_loader)
    STEPS_PER_EPOCH = steps_per_epoch
    total_steps = NUM_EPOCHS * steps_per_epoch // GRAD_ACCUM_STEPS
    warmup_steps = int(0.05 * total_steps)  # 5% warmup
    scheduler = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)
    loss_scaler = RMSLossScaler()

    RESUME_TRAINING = True
    wandb_run_id = None
    log_dir = None
    ckpt_path = "/scratch/ja5009/dreamer_v4_dynamics/2025-12-31_21-26-52/latest.pt"
    start_epoch = 0
    global_update = 0  # counts optimizer updates
    if RESUME_TRAINING:
        # Try loading checkpoint BEFORE initializing W&B
        start_epoch, global_update, wandb_run_id, log_dir = load_checkpoint(
            ckpt_path=ckpt_path,
            dyn=dyn,
            optim=optim,
            scheduler=scheduler,
            rank=rank,
            device=device,
        )
    else:
        if rank == 0:
            print("Not resuming from checkpoint, starting from scratch.")

    # /!\ manual logdir override
    log_dir = "/scratch/ja5009/dreamer_v4_dynamics/2025-12-31_21-26-52/"

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
        dyn.train()

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
        accum_reward = 0.0
        accum_total = 0.0

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
            if 'is_demo' in batch:
                rewards = batch['is_demo'].to(device, non_blocking=True).float()
            else:
                rewards = torch.zeros(B, T, device=device)

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
                _, _, z_clean = enc(images)

            z_clean = z_clean.clone()

            B, T, _, _ = z_clean.shape

            tau_vals, d_vals, tau_idxs, step_idxs = sample_shortcut_batch(
                B, T, device, micro_idx, GRAD_ACCUM_STEPS, num_tau_levels=NUM_TAU_LEVELS
            )

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, l_flow, l_short, l_reward = compute_dynamics_loss(
                    dyn, actions, z_clean, rewards,
                    tau_vals, d_vals, tau_idxs, step_idxs,
                    micro_idx, GRAD_ACCUM_STEPS, NUM_TAU_LEVELS,
                    loss_scaler
                )

            # Update accumulators (CPU scalars)
            accum_flow += l_flow
            accum_shortcut += l_short
            accum_reward += l_reward
            accum_total += loss.detach().item()

            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()

            # If this is the last micro-batch in the window, do optimizer step
            if is_last_micro:
                torch.nn.utils.clip_grad_norm_(dyn.parameters(), max_norm=1.0)

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
                reward_sum = torch.tensor([accum_reward], device=device)

                dist.all_reduce(flow_sum, op=dist.ReduceOp.AVG)
                dist.all_reduce(short_sum, op=dist.ReduceOp.AVG)
                dist.all_reduce(reward_sum, op=dist.ReduceOp.AVG)

                flow_sum = flow_sum.item()
                short_sum = short_sum.item()
                reward_sum = reward_sum.item()

                flow_mean = flow_sum / GRAD_ACCUM_STEPS
                short_mean = short_sum / GRAD_ACCUM_STEPS
                reward_mean = reward_sum / GRAD_ACCUM_STEPS

                if rank == 0:
                    # Sums over micro-batches in this update
                    tb_writer.add_scalar("train/flow_loss_mean", flow_mean, global_update)
                    tb_writer.add_scalar("train/shortcut_loss_mean", short_mean, global_update)
                    tb_writer.add_scalar("train/reward_loss_mean", reward_mean, global_update)
                    tb_writer.add_scalar("train/total_loss_mean", sync_loss, global_update)
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
                        dyn=dyn,
                        optim=optim,
                        scheduler=scheduler,
                        rank=rank,
                        wandb_run_id=wandb_run_id,
                        log_dir=log_dir,
                    )

                # Reset accumulators for next accumulation window
                accum_flow = 0.0
                accum_shortcut = 0.0
                accum_reward = 0.0
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

                frames_per_step = BATCH_PER_GPU * CONTEXT_T_DYN * world_size
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
        total_frames = BATCH_PER_GPU * CONTEXT_T_DYN * world_size * STEPS_PER_EPOCH
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
            dyn=dyn,
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
