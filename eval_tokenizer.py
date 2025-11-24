"""
Qualitative evaluation script for DreamerV4 tokenizer.
Loads a checkpoint, encodes/decodes test trajectories, and saves visual comparisons.
"""

import os
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from models import DreamerV4Encoder, DreamerV4Decoder
from dataset import ShardedHDF5Dataset
from torch.utils.data import DataLoader
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from torchvision.datasets import Places365
import torchvision.transforms as transforms

def load_checkpoint_for_eval(ckpt_path, device):
    """
    Load checkpoint and create encoder/decoder models.
    Returns (enc, dec) on the specified device.
    """
    # Load checkpoint
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    print(f"Global update: {ckpt.get('global_update', 'unknown')}")
    
    # Model hyperparameters (must match training config)
    IMG_H, IMG_W = 256, 256
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

    USE_COMPILE = True

    # Create encoder
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
        mae_max_mask_prob=0.9,  # Disable MAE for eval
        activate_masking=True
    )
    
    # Create decoder
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

    if USE_COMPILE:
        # Good starting config; you can try other modes later
        enc = torch.compile(enc, mode="max-autotune", fullgraph=False)
        dec = torch.compile(dec, mode="max-autotune", fullgraph=False)

    # Load state dicts
    enc.load_state_dict(ckpt["enc"])
    dec.load_state_dict(ckpt["dec"])
    
    # Move to device and set to eval mode
    enc = enc.to(device, dtype=torch.bfloat16)
    dec = dec.to(device, dtype=torch.bfloat16)
    enc.eval()
    dec.eval()
    
    print(f"Models loaded and moved to {device}")
    return enc, dec

def create_comparison_image(original, reconstructed, num_frames=16, save_path=None):
    """
    Create a side-by-side comparison image with original on top, reconstructed on bottom.
    
    Args:
        original: [T, C, H, W] tensor in [0, 1]
        reconstructed: [T, C, H, W] tensor in [0, 1]
        num_frames: Number of frames to display
        save_path: Path to save the image
    """
    T = min(original.shape[0], num_frames)
    
    # Convert to numpy and denormalize to [0, 255]
    orig_np = (original[:T].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    recon_np = (reconstructed[:T].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
    
    # Create figure with 2 rows
    fig, axes = plt.subplots(2, T, figsize=(T * 2, 4))
    
    if T == 1:
        axes = axes.reshape(2, 1)
    
    for t in range(T):
        # Top row: original
        axes[0, t].imshow(orig_np[t])
        axes[0, t].axis('off')
        if t == 0:
            axes[0, t].set_title('Original', fontsize=12, fontweight='bold', color='white',
                                backgroundcolor='black', pad=10)
        
        # Bottom row: reconstructed
        axes[1, t].imshow(recon_np[t])
        axes[1, t].axis('off')
        if t == 0:
            axes[1, t].set_title('Reconstructed', fontsize=12, fontweight='bold', color='white',
                                backgroundcolor='black', pad=10)
    
    # Add row labels on the left
    fig.text(0.01, 0.75, 'Original', ha='left', va='center', 
            fontsize=14, fontweight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    fig.text(0.01, 0.25, 'Reconstructed', ha='left', va='center',
            fontsize=14, fontweight='bold', color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, wspace=0.02, hspace=0.1)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison to {save_path}")
    
    plt.close()

class PushTDatasetClass(Dataset):
    """
    Dataset for PushT HDF5 files that samples trajectory windows.
    Assumes the HDF5 file contains one single very long trajectory.
    """

    def __init__(
        self,
        hd5_file_path,
        window_size=96,
        stride=1,
        load_to_ram=False,
        target_size=(256, 256),  # Target size for upsampling
    ):
        self.load_to_ram = load_to_ram
        self.hd5_file_path = hd5_file_path
        self.window_size = window_size
        self.stride = stride
        self.target_size = target_size

        # Open HDF5 file
        self.hd5_file = h5py.File(self.hd5_file_path, "r")
        
        # Load data to RAM if requested
        if self.load_to_ram:
            self.data = {
                key: torch.from_numpy(self.hd5_file[key][:]).to(torch.float32) 
                for key in self.hd5_file
            }
        else:
            self.data = self.hd5_file

        # Get trajectory length
        self.len_traj, c, W, H = self.hd5_file['cam1'].shape
        
        # Build window indices (start positions for each window)
        self.windows = []
        for start in range(0, self.len_traj - self.window_size + 1, self.stride):
            self.windows.append(start)
        
        print(f"Loaded trajectory with {self.len_traj} frames")
        print(f"Created {len(self.windows)} windows of size {self.window_size} with stride {self.stride}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        assert idx < len(self), f'idx {idx} out of range for dataset with length {len(self)}'
        
        # Get window start and end
        start = self.windows[idx]
        end = start + self.window_size
        
        # Load data for the window
        if self.load_to_ram:
            actions = self.data['actions'][start:end]  # [T, action_dim]
            images = self.data['cam1'][start:end]      # [T, C, H, W]
        else:
            actions = torch.from_numpy(self.data['actions'][start:end]).to(torch.float32)
            images = torch.from_numpy(self.data['cam1'][start:end]).to(torch.float32)

        # Normalize images to [0, 1] if they're in [0, 255]
        if images.max() > 1.0:
            images = images / 255.0
        
        # Ensure correct shape: [T, C, H, W]
        if images.shape[1] != 3:  # If channels are not in dim 1
            images = images.permute(0, 3, 1, 2)  # [T, H, W, C] -> [T, C, H, W]

        # Upsample from 224x224 to 256x256 using bilinear interpolation
        if images.shape[-2:] != self.target_size:
            images = torch.nn.functional.interpolate(
                images,
                size=self.target_size,
                mode='bilinear',
                align_corners=False,
                antialias=True  # Better quality upsampling
            )

        print("images.shape", images.shape, images.min(), images.max())
        Dout = {
            "image": images,
            "action": actions,
        }
        
        return Dout

class Places365SingleImageDataset(Dataset):
    """
    Wrapper around Places365 that returns single images with sequence dimension of 1.
    Output shape: [1, 3, 256, 256]
    """
    def __init__(self, root='./data', split='val'):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ])
        
        self.places_dataset = Places365(
            root=root,
            split=split,  # 'val' is faster, 'train-standard' for full dataset
            small=True,   # 256x256 version
            download=True,
            transform=transform
        )
        
        print(f"Loaded Places365 {split} split: {len(self.places_dataset)} images")
    
    def __len__(self):
        return len(self.places_dataset)
    
    def __getitem__(self, idx):
        image, label = self.places_dataset[idx]  # [3, 256, 256]
        
        # Add sequence dimension of size 1: [3, 256, 256] -> [1, 3, 256, 256]
        images = image.unsqueeze(0)  # [1, 3, 256, 256]
        
        return {
            "image": images,
            "label": label,
        }

def visualize_masked_video(
    video: torch.Tensor,
    patch_size: int,
    patch_mask: torch.Tensor,
    color: tuple = (1.0, 0.0, 0.0),  # RGB for tint/border (red by default)
    alpha: float = 0.5,  # Transparency: 0=invisible, 1=opaque
) -> torch.Tensor:
    """
    Visualize masked patches while keeping underlying content visible.
    
    Args:
        video: [B, T, C, H, W], pixel space in [0, 1]
        patch_size: P
        patch_mask: [B, T, Np] bool tensor, True = masked patch
        color: RGB tuple for overlay/border color
        alpha: Transparency level (0-1)
    
    Returns:
        vis_video: [B, T, C, H, W] with masked regions visualized
    """
    B, T, C, H, W = video.shape
    P = patch_size
    Hp, Wp = H // P, W // P

    # 1. Create per-pixel mask from patch mask
    # [B,T,Np] -> [B,T,Hp,Wp] -> [B,T,1,H,W]
    pixel_mask = (
        patch_mask.view(B, T, Hp, Wp)
                  .unsqueeze(2)  # Add channel dim
                  .repeat_interleave(P, dim=3)  # Repeat along H
                  .repeat_interleave(P, dim=4)  # Repeat along W
    )  # [B,T,1,H,W]

    # Semi-transparent color overlay (best for clear identification)
    overlay = torch.tensor(color, device=video.device).view(1, 1, 3, 1, 1)
    overlay = overlay.expand(B, T, 3, H, W)

    # Alpha blend: result = alpha*overlay + (1-alpha)*original
    vis_video = torch.where(
        pixel_mask,
        alpha * overlay + (1 - alpha) * video,
        video,
    )

    return vis_video.clamp(0, 1)

def make_right_rect_mask_for_last_n_frames(
    video: torch.Tensor,
    patch_size: int,
    n_frames: int = 1,
) -> torch.Tensor:
    """
    Create a [B, T, Np] bool mask with a rectangular region masked
    on the last n frames, same rectangle for all sequences.

    Rectangle geometry (in patch grid coordinates):
      - Right side: near right image border
      - Left side: ends before middle of width
      - Top side: slightly below middle of height
      - Bottom side: at bottom of image

    Args:
        video: [B, T, C, H, W]
        patch_size: P (must divide H and W)
        n_frames: Number of last frames to apply the mask to (default: 1)

    Returns:
        mask: [B, T, Np] bool, True = masked patch
    """
    B, T, C, H, W = video.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "H and W must be multiples of patch_size"
    assert n_frames <= T, f"n_frames ({n_frames}) cannot exceed T ({T})"

    Hp = H // P  # patches along height
    Wp = W // P  # patches along width
    Np = Hp * Wp

    device = video.device

    # Initialize all-unmasked
    mask = torch.zeros(B, T, Np, dtype=torch.bool, device=device)

    # Define rectangle in patch grid:
    # h indices: from ~60% of height to bottom
    h_start = int(0.6 * Hp)
    h_end   = Hp  # bottom

    # w indices: from ~50% of width to right edge
    w_start = int(0.5 * Wp)
    w_end   = Wp  # right side

    # Build a [Hp, Wp] boolean grid for the rectangle
    rect_grid = torch.zeros(Hp, Wp, dtype=torch.bool, device=device)
    rect_grid[h_start:h_end, w_start:w_end] = True  # set rectangle True

    # Flatten to [Np]
    rect_flat = rect_grid.view(-1)  # [Np]

    # Apply this rectangle to the last n frames for all batch elements
    # last n frames indices: T - n_frames : T
    mask[:, T - n_frames:T, :] = rect_flat.unsqueeze(0).unsqueeze(0).expand(B, n_frames, -1)

    return mask

def mask_video_with_patch_mask(
    video: torch.Tensor,
    patch_size: int,
    patch_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        video: [B, T, C, H, W], pixel space in [0, 1] or [0, 255].
        patch_size: P, must divide H and W (same P as encoder).
        patch_mask: [B, T, Np] bool tensor, True = masked patch.
                    Np = (H // P) * (W // P), same ordering as MAE mask.

    Returns:
        masked_video: [B, T, C, H, W] with masked patches set to 0.
    """
    B, T, C, H, W = video.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, "H and W must be multiples of patch_size"
    Hp, Wp = H // P, W // P             # grid of patches

    # 1. Pixel patchify: [B,T,C,H,W] -> [B,T,Np,C,P,P]
    patches = (
        video.view(B, T, C, Hp, P, Wp, P)
             .permute(0, 1, 3, 5, 2, 4, 6)   # [B,T,Hp,Wp,C,P,P]
             .contiguous()
    )
    patches = patches.view(B, T, Hp * Wp, C, P, P)   # [B,T,Np,C,P,P]

    # 2. Apply mask over patches (True => zero patch)
    #    patch_mask: [B,T,Np] -> [B,T,Np,1,1,1] for broadcasting
    if patch_mask.dtype != torch.bool:
        patch_mask = patch_mask.bool()
    mask_expanded = patch_mask[..., None, None, None]  # [B,T,Np,1,1,1]

    # Zero masked patches
    patches_masked = torch.where(
        mask_expanded,
        torch.zeros_like(patches),
        patches,
    )  # [B,T,Np,C,P,P]

    # 3. Unpatchify back to image: [B,T,Np,C,P,P] -> [B,T,C,H,W]
    patches_masked = patches_masked.view(B, T, Hp, Wp, C, P, P)  # [B,T,Hp,Wp,C,P,P]
    masked_video = (
        patches_masked.permute(0, 1, 4, 2, 5, 3, 6)   # [B,T,C,Hp,P,Wp,P]
                     .contiguous()
                     .view(B, T, C, H, W)
    )

    return masked_video

def evaluate_trajectories(
    ckpt_path,
    data_dir,
    output_dir="./eval_outputs",
    num_trajectories=5,
    num_frames_per_traj=16,
    device="cuda",
):
    """
    Evaluate checkpoint on test trajectories and save visual comparisons.
    
    Args:
        ckpt_path: Path to checkpoint file
        data_dir: Path to sharded HDF5 dataset directory
        output_dir: Directory to save output images
        num_trajectories: Number of trajectories to evaluate
        num_frames_per_traj: Number of frames to display per trajectory
        device: Device to run evaluation on
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    enc, dec = load_checkpoint_for_eval(ckpt_path, device)
    
    # Create test dataset and dataloader
    CONTEXT_T = 32 # 96
    """test_dataset = ShardedHDF5Dataset(
        data_dir=data_dir,
        window_size=CONTEXT_T,
        stride=CONTEXT_T,  # Non-overlapping windows for diverse samples
        split="test",
        train_fraction=0.9,
        split_seed=123,
    )"""

    test_dataset = PushTDatasetClass(
        hd5_file_path='/scratch/ja5009/dataset/pusht-hd5-v1-224/episode_0.h5',
        window_size=CONTEXT_T,
        stride=CONTEXT_T,  # Non-overlapping windows for diverse samples
        load_to_ram=True
    )

    """test_dataset = Places365SingleImageDataset(
        root='/scratch/ja5009/data/places365',
        split='val',  # ~36K validation images
    )"""

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    print(f"\nTest dataset: {len(test_dataset)} windows")
    print(f"Evaluating {num_trajectories} trajectories...\n")

    # Evaluate trajectories
    with torch.no_grad():
        for traj_idx, batch in enumerate(test_loader):
            if traj_idx >= num_trajectories:
                break

            # Move to device
            images = batch['image'].to(device, dtype=torch.bfloat16)  # [1, T, C, H, W]

            # Encode and decode
            mask = make_right_rect_mask_for_last_n_frames(images, patch_size=16, n_frames=6)
            P_enc, L_enc, Z, mask = enc(images, mask=mask)
            Z = Z.clone()  # Prevents CUDA graph memory reuse conflict
            mask = mask.clone()
            R_dec, x_hat = dec(Z)

            # Remove batch dimension
            """images = mask_video_with_patch_mask(
                video=images,
                patch_size=16,
                patch_mask=mask
            )"""

            images = visualize_masked_video(
                images,
                patch_size=16,
                patch_mask=mask,
                color=(1.0, 0.0, 0.0),  # Red
                alpha=0.4,  # 40% overlay opacity
            )

            original = images[0]  # [T, C, H, W]
            reconstructed = x_hat[0]  # [T, C, H, W]

            # Compute metrics
            mse = nn.functional.mse_loss(reconstructed, original).item()

            print(f"Trajectory {traj_idx + 1}/{num_trajectories}")
            print(f"  MSE: {mse:.6f}")
            print(f"  Latent bottleneck shape: {Z.shape}")

            # Save comparison image
            save_path = output_dir / f"trajectory_{traj_idx:03d}_comparison.png"
            create_comparison_image(
                original.to(torch.float32),
                reconstructed.to(torch.float32),
                num_frames=num_frames_per_traj,
                save_path=save_path,
            )

    print(f"\n✓ Evaluation complete! Images saved to {output_dir}")

def main():
    """
    Main evaluation script.
    """
    # Configuration
    CHECKPOINT_PATH = "./logs/dreamer_v4_tokenizer/2025-11-18_08-12-02/latest.pt"
    DATA_DIR = "/scratch/ja5009/soar_data_sharded/"
    OUTPUT_DIR = "./eval_outputs"
    NUM_TRAJECTORIES = 32
    NUM_FRAMES_PER_TRAJ = 32
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("DreamerV4 Tokenizer Qualitative Evaluation")
    print("="*60)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print("="*60 + "\n")
    
    evaluate_trajectories(
        ckpt_path=CHECKPOINT_PATH,
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        num_trajectories=NUM_TRAJECTORIES,
        num_frames_per_traj=NUM_FRAMES_PER_TRAJ,
        device=DEVICE,
    )

if __name__ == "__main__":
    main()
