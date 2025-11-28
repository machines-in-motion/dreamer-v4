import os
import time
import torch
import hydra
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf


from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from functools import partial
import torch.distributed as dist


from model.dynamics import (DreamerV4Denoiser,
                            DreamerV4DenoiserCfg,
                            ForwardDiffusionWithShortcut)

from dataset import ShardedHDF5Dataset


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
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
    # DDP: load into wrapped model
    state_dict = ckpt["model"]

    if isinstance(model, DDP):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)

    optim.load_state_dict(ckpt["optim"])
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
    dataset: str,
    batch_size: int,
    rank: int,
    world_size: int,
    num_workers: int = 4,
    seed: int = 42,
    shuffle: bool = True,
    drop_last: bool = True,
):
    """
    Create DataLoader with DistributedSampler for sharded HDF5 dataset.
    """

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

    return dataloader


rank, local_rank, world_size = setup_distributed()

@hydra.main(config_path='config', config_name='dynamics_small.yaml')
def main(cfg: DictConfig):

    train_dataset = ShardedHDF5Dataset(
                                data_dir=cfg.dataset.base_path,
                                window_size=cfg.train.long_context,
                                stride=1,
                                split="train",
                                train_fraction=0.9,
                                split_seed=123)
    dev_dataset = ShardedHDF5Dataset(
                                data_dir=cfg.dataset.base_path,
                                window_size=cfg.train.long_context,
                                stride=1,
                                split="test",
                                train_fraction=0.9,
                                split_seed=123)
    
    train_loader = create_distributed_dataloader(train_dataset, cfg.train.batch_size, local_rank, world_size, cfg.train.num_workers)
    dev_loader = create_distributed_dataloader(dev_dataset, cfg.train.batch_size, local_rank, world_size, cfg.train.num_workers)
    


    model = DreamerV4Denoiser(DreamerV4DenoiserCfg(**OmegaConf.to_object(cfg.denoiser)))
    model.cuda(local_rank)

    n_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    diffuser = ForwardDiffusionWithShortcut(cfg.denoiser.K_max)
    print(f'The number of denoiser parameters is: {n_params/1e6:.2f} M')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    step = 0
    for epoch in range(cfg.train.num_epochs):
        for batch in train_loader:
            step +=1
            B, T, _, _, _ = batch['image'].shape
            fake_tokens = torch.randn(B, T, cfg.denoiser.num_latent_tokens, cfg.denoiser.latent_dim).cuda(local_rank).to(torch.float32)
            diffused_batch = diffuser(fake_tokens)
            tau_d = diffused_batch['tau_d']
            step_d = diffused_batch['step_d']
            x_tau = diffused_batch['x_tau']
            breakpoint()
            denoising_vel = model(x_tau, tau_d, step_d)


    breakpoint()
    

if __name__=='__main__':
    main()
    cleanup_distributed()
    