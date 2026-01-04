import argparse
import time
import torch
import numpy as np
import cv2
import torchvision.transforms.functional as TF

# --- Models ---
# Assuming models.py is available in the same folder
from models import DreamerV4Encoder, DreamerV4Decoder, DreamerV4Dynamics

# --- Configuration ---
IMG_H, IMG_W = 128, 128
PATCH = 16
CONTEXT_T = 96
N_LATENTS = 256
BOTTLENECK_D = 16
D_MODEL_ENC = 768
N_LAYERS_ENC = 12
HEADS_Q_ENC = 12
HEADS_KV_LATENT_ENC = 12
MLP_RATIO = 4.0
TEMPORAL_EVERY = 4

D_MODEL_DYN = 1536
N_LAYERS_DYN = 24
HEADS_Q_DYN = 24
NUM_REGISTERS = 4
NUM_TAU_LEVELS = 128
CONTEXT_T_DYN = 32 # Ensure this matches your saved trajectory length if possible

# --- Helpers ---

def get_device():
    if torch.backends.mps.is_available():
        print("Using Apple Metal (MPS) acceleration.")
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def load_models(tokenizer_path, dynamics_path, action_dim, device):
    print("Loading models...")
    ckpt_tok = torch.load(tokenizer_path, map_location="cpu")
    ckpt_dyn = torch.load(dynamics_path, map_location="cpu")

    # Encoder
    enc = DreamerV4Encoder(
        image_size=(IMG_H, IMG_W), patch_size=PATCH, d_model=D_MODEL_ENC,
        n_layers=N_LAYERS_ENC, num_heads_q=HEADS_Q_ENC, num_heads_kv_latent=HEADS_KV_LATENT_ENC,
        seq_len=CONTEXT_T, mlp_ratio=MLP_RATIO, n_latents=N_LATENTS,
        bottleneck_dim=BOTTLENECK_D, temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    # torch.compile on MPS can sometimes be flaky or slow to warm up. 
    enc = torch.compile(enc) 
    enc.load_state_dict(ckpt_tok["enc"])

    # Decoder
    dec = DreamerV4Decoder(
        image_size=(IMG_H, IMG_W), patch_size=PATCH, d_model=D_MODEL_ENC,
        n_layers=N_LAYERS_ENC, num_heads_q=HEADS_Q_ENC, num_heads_kv_latent=HEADS_KV_LATENT_ENC,
        bottleneck_dim=BOTTLENECK_D, seq_len=CONTEXT_T, mlp_ratio=MLP_RATIO,
        n_latents=N_LATENTS, temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    dec = torch.compile(dec)
    if "dec" in ckpt_tok:
        dec.load_state_dict(ckpt_tok["dec"])
    
    # Dynamics
    dyn = DreamerV4Dynamics(
        action_dim=action_dim, num_latents=N_LATENTS, latent_dim=BOTTLENECK_D,
        d_model=D_MODEL_DYN, num_layers=N_LAYERS_DYN, num_heads=HEADS_Q_DYN,
        num_registers=NUM_REGISTERS, seq_len=CONTEXT_T_DYN, num_tau_levels=NUM_TAU_LEVELS,
        temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    dyn = torch.compile(dyn)
    dyn.load_state_dict(ckpt_dyn["dyn"])

    return enc, dec, dyn


@torch.no_grad()
def solve_frame_cached(model, actions_t, z_gen, t, device, num_steps=4):
    """
    Generates ONE frame at index `t`.
    z_gen: The full latent tensor [B, T_total, N, D] used for initialization.
    """
    # Infer shape from z_gen (using the previous frame or general shape)
    B, _, N, D = z_gen.shape
    
    # 1. Initialize z_t as random noise
    z_t = torch.randn(B, 1, N, D, device=device, dtype=torch.bfloat16)
    
    step_val = 1 / num_steps
    step_idx = int(np.log2(num_steps))
    d_min_idx = int(np.log2(NUM_TAU_LEVELS)) 
    
    # ODE Loop
    for i in range(num_steps):
        tau_curr = i / num_steps
        curr_tau_idx = int(((num_steps - 1 - i) + 1)*(2**(d_min_idx-step_idx)))-1
        
        # Determine if this is the LAST step of the ODE solver
        # If it is, we want to commit to the cache.
        is_last_step = (i == num_steps - 1)
        
        # Indices
        tau_idxs = torch.full((B, 1), curr_tau_idx, dtype=torch.long, device=device)
        step_idxs = torch.full((B, 1), step_idx, dtype=torch.long, device=device)
        
        # Forward pass
        # update_cache=True ONLY on the last step
        pred = model.forward_step(
            action=actions_t,       # [B, 1, D]
            noisy_z=z_t,            # [B, 1, N, D]
            sigma_idx=tau_idxs,     # [B, 1]
            step_idx=step_idxs,     # [B, 1]
            start_step_idx=t,
            update_cache=is_last_step 
        )
        
        z_1_pred = pred # [B, 1, N, D]
        
        denom = max(1.0 - tau_curr, 1e-5)
        velocity = (z_1_pred - z_t) / denom
        z_t = z_t + velocity * step_val
        
    return z_t

def run_live_eval(args):
    device = get_device()
    
    print(f"Loading trajectory from {args.traj_path}...")
    traj_data = torch.load(args.traj_path, map_location="cpu")
    images = traj_data['images'].unsqueeze(0).to(device).to(torch.bfloat16)
    actions = traj_data['actions'].unsqueeze(0).to(device).to(torch.bfloat16)
    
    B, T_total, C, H, W = images.shape
    action_dim = actions.shape[-1]
    
    enc, dec, dyn = load_models(args.tokenizer_ckpt, args.dynamics_ckpt, action_dim, device)
    
    print("Initializing Cache...")
    dyn.init_cache(B, device, max_seq_len=CONTEXT_T_DYN)
    dec.init_cache(B, device, max_seq_len=CONTEXT_T_DYN)

    # --- PREFILL ---
    print(f"Prefilling context (0 to {args.context_len})...")
    with torch.no_grad():
        _, _, z_gt = enc(images)
        
    context_z = z_gt[:, :args.context_len]
    context_actions = actions[:, :args.context_len]
    
    dummy_sigma = torch.zeros((B, args.context_len), dtype=torch.long, device=device)
    dummy_step = torch.zeros((B, args.context_len), dtype=torch.long, device=device)
    
    with torch.no_grad():
        dyn.forward_step(
            action=context_actions,
            noisy_z=context_z,
            sigma_idx=dummy_sigma,
            step_idx=dummy_step,
            start_step_idx=0,
            update_cache=True # Always update cache for context
        )

        dec.forward_step(context_z, start_step_idx=0, update_cache=True)
        
    z_gen = z_gt.clone()
    
    # --- DECODE LOOP ---
    print("Starting Live Generation...")
    cv2.namedWindow("Live Generation", cv2.WINDOW_NORMAL)
    
    for t in range(args.context_len, T_total):
        start_time = time.time()
        
        action_t = actions[:, t:t+1]
        
        # Generate Frame t
        # z_gen passed for shape reference
        z_next = solve_frame_cached(dyn, action_t, z_gen, t, device, num_steps=4)
        
        z_gen[:, t:t+1] = z_next
        
        with torch.no_grad():
            _, recon_frame = dec.forward_step(z_next, start_step_idx=t, update_cache=True)
            
        gt_frame = images[0, t].float().cpu().permute(1, 2, 0).numpy()
        gen_frame = recon_frame[0, 0].float().cpu().permute(1, 2, 0).numpy()
        
        display_img = np.hstack((gt_frame, gen_frame))
        display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
        display_img = np.clip(display_img * 255, 0, 255).astype(np.uint8)
        
        cv2.imshow("Live Generation", display_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
        print(f"Frame {t}: {(time.time() - start_time)*1000:.1f}ms")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--dynamics_ckpt", type=str, required=True)
    parser.add_argument("--traj_path", type=str, required=True, help="Path to the .pt file from export_traj.py")
    parser.add_argument("--context_len", type=int, default=10)
    args = parser.parse_args()
    
    run_live_eval(args)
