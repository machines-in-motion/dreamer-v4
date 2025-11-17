import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data import ConcatDataset
from tqdm import tqdm

from dataset import SingleViewSequenceDataset


def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
    )

    return rank, local_rank, world_size


def cleanup():
    dist.destroy_process_group()


rank, local_rank, world_size = setup()

# Dataset
train_dataset_paths = [
    "/scratch/rk4342/datasets/pushT/pusht-play-128/episode_0.h5"
]

seq_len = 64
load_to_ram = True
batch_size_per_gpu = 16
num_workers = 8

datasets = [SingleViewSequenceDataset(p, seq_len, load_to_ram=load_to_ram)
            for p in train_dataset_paths]

train_dataset = ConcatDataset(datasets)

sampler = DistributedSampler(train_dataset, shuffle=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_per_gpu,
    sampler=sampler,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)


# Training loop
for epoch in range(3):
    print(f'Running epoch {epoch} in rank: {local_rank}')
    sampler.set_epoch(epoch)
    for batch in train_loader:
        imgs = batch["observation.image"].to(local_rank, non_blocking=True)
        B, T, C, H, W = imgs.shape
        pass

cleanup()