import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid
import torchvision.transforms.functional as TF

# Import models and dataset
from models import DreamerV4Encoder, DreamerV4Decoder, DreamerV4Dynamics
from dataset import ShardedHDF5Dataset, HDF5SequenceDataset

# --- Configuration (Matches Training) ---
IMG_H, IMG_W = 128, 128 # 256, 256
PATCH = 16
CONTEXT_T = 96
N_LATENTS = 256 # 512
BOTTLENECK_D = 16
D_MODEL_ENC = 768
N_LAYERS_ENC = 12
HEADS_Q_ENC = 12
HEADS_KV_LATENT_ENC = 12
MLP_RATIO = 4.0
TEMPORAL_EVERY = 4
IN_CH = 3

D_MODEL_DYN = 2048
N_LAYERS_DYN = 32
HEADS_Q_DYN = 32
NUM_REGISTERS = 4
NUM_TAU_LEVELS = 128
CONTEXT_T_DYN = 32

SEQ_COR_TAU_IDX = 12
SEQ_COR = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True, help="Path to tokenizer checkpoint")
    parser.add_argument("--dynamics_ckpt", type=str, required=True, help="Path to dynamics checkpoint")
    parser.add_argument("--data_dir", type=str, default="/scratch/ja5009/soar_data_sharded_128x128/", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of trajectories to evaluate")
    parser.add_argument("--context_len", type=int, default=10, help="Number of conditioning frames")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_tokenizer(ckpt_path, device):
    print(f"Loading tokenizer from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Encoder
    enc = DreamerV4Encoder(
        image_size=(IMG_H, IMG_W), patch_size=PATCH, d_model=D_MODEL_ENC,
        n_layers=N_LAYERS_ENC, num_heads_q=HEADS_Q_ENC, num_heads_kv_latent=HEADS_KV_LATENT_ENC,
        seq_len=CONTEXT_T, mlp_ratio=MLP_RATIO, n_latents=N_LATENTS,
        bottleneck_dim=BOTTLENECK_D, temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    enc = torch.compile(enc)
    enc.load_state_dict(ckpt["enc"])
    
    # Decoder (Assuming symmetric config and key 'dec' in checkpoint)
    dec = DreamerV4Decoder(
        image_size=(IMG_H, IMG_W), patch_size=PATCH, d_model=D_MODEL_ENC,
        n_layers=N_LAYERS_ENC, num_heads_q=HEADS_Q_ENC, num_heads_kv_latent=HEADS_KV_LATENT_ENC,
        bottleneck_dim=BOTTLENECK_D, seq_len=CONTEXT_T, mlp_ratio=MLP_RATIO,
        n_latents=N_LATENTS, temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    dec = torch.compile(dec)
    
    # Try loading decoder; warn if key missing
    if "dec" in ckpt:
        dec.load_state_dict(ckpt["dec"])
    else:
        print("WARNING: 'dec' key not found in tokenizer checkpoint. Using random decoder weights (Reconstruction will fail).")
    
    return enc, dec

def load_dynamics(ckpt_path, action_dim, device):
    print(f"Loading dynamics from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    dyn = DreamerV4Dynamics(
        action_dim=action_dim, num_latents=N_LATENTS, latent_dim=BOTTLENECK_D,
        d_model=D_MODEL_DYN, num_layers=N_LAYERS_DYN, num_heads=HEADS_Q_DYN,
        num_registers=NUM_REGISTERS, seq_len=CONTEXT_T_DYN, num_tau_levels=NUM_TAU_LEVELS,
        temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    dyn = torch.compile(dyn)
    
    # Load weights (Standard FSDP FULL_STATE_DICT save is compatible with standard load)
    dyn.load_state_dict(ckpt["dyn"])
    return dyn

@torch.no_grad()
def euler_solve_step(model, actions_seq, z_seq, t_idx, device, num_steps=4):
    """
    Generates z_{t} starting from noise, conditioned on z_{0:t-1}.
    Uses Euler method for Flow Matching.
    """
    B, _, N, D = z_seq.shape

    # Initialize z_t as noise
    z_t = torch.randn(B, 1, N, D, device=device, dtype=torch.bfloat16)

    step_val = 1 / num_steps
    step_idx = int(np.log2(num_steps))

    input_actions = actions_seq[:, :t_idx+1]

    # Constants for indices
    d_min_idx = int(np.log2(NUM_TAU_LEVELS)) # typically 7 for 128 levels

    for i in range(num_steps):
        curr_tau_idx = int(((num_steps - 1 - i) + 1)*(2**(d_min_idx-step_idx)))-1
        tau_curr = i / num_steps # signal level, 1. = clean
        print("curr_tau_idx: ", curr_tau_idx)
        print("tau_curr: ", tau_curr)

        # Construct noisy sequence input: [z_0, ..., z_{t-1}, z_t_current]
        # z_0...z_{t-1} are clean (from history)
        current_seq_input = torch.cat([z_seq[:, :t_idx], z_t], dim=1)

        # Create index tensors
        tau_idxs = torch.zeros(B, t_idx+1, dtype=torch.long, device=device)
        tau_idxs[:, :t_idx] = SEQ_COR_TAU_IDX if SEQ_COR else 0 # Slitghly corrupt history
        tau_idxs[:, t_idx] = curr_tau_idx # Current frame is at tau_curr

        step_idxs = torch.full((B, t_idx+1), d_min_idx, dtype=torch.long, device=device)
        step_idxs[:, -1] = step_idx

        # Predict 'vector field' (z_clean prediction)
        pred_z_clean = model(input_actions, current_seq_input, tau_idxs, step_idxs)
        # Extract prediction for the last frame
        z_1_pred = pred_z_clean[:, -1:] # [B, 1, N, D]

        # Euler Step: z_{new} = z_{old} + (z_1 - z_{old}) / (1 - tau) * dt
        denom = max(1.0 - tau_curr, 1e-5)
        velocity = (z_1_pred - z_t) / denom

        z_t = z_t + velocity * step_val

    return z_t

def add_label(tensor_img, text, top=True):
    """Add text label to a tensor image (C, H, W)"""
    # Convert to PIL
    img_np = tensor_img.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    
    draw = ImageDraw.Draw(pil_img)
    # Basic font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        
    # Draw text with black outline for visibility
    x, y = 10, 10
    draw.text((x-1, y), text, font=font, fill="black")
    draw.text((x+1, y), text, font=font, fill="black")
    draw.text((x, y-1), text, font=font, fill="black")
    draw.text((x, y+1), text, font=font, fill="black")
    draw.text((x, y), text, font=font, fill="white")
    
    return TF.to_tensor(pil_img)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"Setting up dataset from {args.data_dir}...")
    """dataset = ShardedHDF5Dataset(
        data_dir=args.data_dir, window_size=CONTEXT_T_DYN, stride=1, split="test",
        train_fraction=0.9, split_seed=123
    )"""
    dataset = HDF5SequenceDataset(data_dir=args.data_dir, window_size=CONTEXT_T_DYN, stride=1)
    # Standard DataLoader (no distributed sampler needed for eval)
    g = torch.Generator()
    g.manual_seed(42)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.num_samples, shuffle=True, generator=g)
    
    # Fetch one batch
    batch = next(iter(dataloader))
    images = batch['image'].to(args.device).to(torch.bfloat16)
    actions = batch['action'].to(args.device).to(torch.bfloat16)
    
    B, T, C, H, W = images.shape
    action_dim = actions.shape[-1]
    
    # 2. Load Models
    enc, dec = load_tokenizer(args.tokenizer_ckpt, args.device)
    dyn = load_dynamics(args.dynamics_ckpt, action_dim, args.device)
    
    # 3. Encode Ground Truth
    print("Encoding ground truth trajectories...")
    with torch.no_grad():
        # Encoder forward: P_enc, L_enc, Z
        # We need Z (bottleneck latents)
        print(images.shape, actions.shape)
        _, _, z_gt = enc(images) # [B, T, Nl, D]
        
    # 4. Autoregressive Generation
    print(f"Starting generation (Context={args.context_len})...")
    z_gen = z_gt.clone()
    final_z_gen = z_gt.clone()

    t_idx = args.context_len
    if SEQ_COR:
        SEQ_COR_TAU = torch.zeros(B, t_idx, dtype=torch.bfloat16, device=args.device)
        SEQ_COR_TAU.fill_(1. - ((SEQ_COR_TAU_IDX + 1) / NUM_TAU_LEVELS))
        eps = torch.randn_like(z_gen[:, :t_idx])
        # tau shape (B,T) -> (B,T,1,1)
        tau_ex = SEQ_COR_TAU.view(B, t_idx, 1, 1)
        z_gen[:, :t_idx] = (1 - tau_ex) * eps + tau_ex * z_gen[:, :t_idx]

    # We keep the first `context_len` frames fixed, generate the rest
    for t in range(args.context_len, T):
        print(f"Generating frame {t+1}/{T}...")
        with torch.no_grad():
            # Generate frame t conditioned on 0..t-1
            z_next = euler_solve_step(dyn, actions, z_gen, t, args.device, num_steps=4)
            z_gen[:, t:t+1] = z_next
            final_z_gen[:, t:t+1] = z_next.clone()

            if SEQ_COR:
                SEQ_COR_TAU = torch.zeros(B, 1, dtype=torch.bfloat16, device=args.device)
                SEQ_COR_TAU.fill_(1. - ((SEQ_COR_TAU_IDX + 1) / NUM_TAU_LEVELS))
                eps = torch.randn_like(z_gen[:, t:t+1])
                tau_ex = SEQ_COR_TAU.view(B, 1, 1, 1)
                z_gen[:, t:t+1] = (1 - tau_ex) * eps + tau_ex * z_gen[:, t:t+1]
            
    # 5. Decode and Visualize
    print("Decoding trajectories...")
    with torch.no_grad():
        # Decode GT (for verification/reconstruction)
        _, recon_gt = dec(z_gt)
        recon_gt = recon_gt.to(torch.float32)
        
        # Decode Generated
        _, recon_gen = dec(final_z_gen)
        recon_gen = recon_gen.to(torch.float32)
        
    # 6. Save Grids
    print("Saving images...")
    for i in range(B):
        traj_gt = recon_gt[i]   # [T, C, H, W]
        traj_gen = recon_gen[i] # [T, C, H, W]
        
        # Add labels to first frames
        traj_gt[0] = add_label(traj_gt[0], "Ground Truth (Recon)", top=True)
        traj_gen[0] = add_label(traj_gen[0], "Generated", top=False)
        
        # Clamp to valid range
        traj_gt = torch.clamp(traj_gt, 0, 1)
        traj_gen = torch.clamp(traj_gen, 0, 1)
        
        # Create grid: Top row GT, Bottom row Gen
        # Concatenate along height
        grid_gt = make_grid(traj_gt, nrow=T, padding=2)
        grid_gen = make_grid(traj_gen, nrow=T, padding=2)
        
        # Combine
        final_grid = torch.cat([grid_gt, grid_gen], dim=1) # dim 1 is height
        
        save_path = os.path.join(args.output_dir, f"traj_{i}.png")
        TF.to_pil_image(final_grid).save(save_path)
        print(f"Saved {save_path}")

if __name__ == "__main__":
    main()
