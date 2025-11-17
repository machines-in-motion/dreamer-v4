from tqdm import tqdm
from dataset import SingleViewSequenceDataset
import torch.distributed as dist
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
import torch
from torchvision.transforms import transforms
import torchvision.transforms.functional as F
import numpy as np
import kornia.augmentation as K
torch.set_float32_matmul_precision('high')


device = 'cuda:0'
# dist.init_process_group('nccl')
kernel_size= 5
max_sigma= 1e-1
brightness= 1e-1
contrast= 1e-1
saturation= 1e-1
hue= 3e-2



train_dataset_paths = ["/scratch/rk4342/datasets/pushT/pusht-play-128/episode_0.h5"]
seq_len = 64
load_to_ram = True
batch_size_per_gpu = 16
num_workers = 8
train_datasets = [SingleViewSequenceDataset(dataset_path, seq_len, load_to_ram=load_to_ram) for dataset_path in train_dataset_paths]
train_dataset = ConcatDataset(train_datasets)
# sampler = DistributedSampler(train_dataset, shuffle = True)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_per_gpu,
    # sampler=sampler,
    num_workers=num_workers,
    # pin_memory=True,
    persistent_workers=True,
)

aug1 =  K.RandomGaussianBlur((5, 5), (1e-3, max_sigma), p=1.0)
aug1.set_rng_device_and_dtype(device=device, dtype=torch.float32)
aug2 = K.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03, p=1.0)
aug2.set_rng_device_and_dtype(device=device, dtype=torch.float32)
# K.RandomGaussianNoise(mean=0.0, std=0.01
augment = torch.nn.Sequential(
    aug1,
    aug2,
    # K.RandomGaussianNoise(mean=0.0, std=0.01)
)
# augment = torch.compile(augment)
# augment = augment.cuda()
for i in range(100):
    for batch in tqdm(train_loader):
        # imgs = batch["observation.image"].float().cuda(non_blocking=True) # (B, T, C, H, W)
        imgs = batch["observation.image"] # (B, T, C, H, W)
        B, T, C, H, W = imgs.shape
        imgs = imgs.view(B*T, C, H, W).contiguous().to(device)      # flatten time
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            imgs = augment(imgs) 
        # print(imgs.device)    
        imgs = imgs.view(B, T, C, H, W).to(torch.float32)     # restore shape
        pass