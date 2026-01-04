import sys
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
MAX_INTERACTIVE_LEN = 500 # Arbitrary limit for this session

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
        seq_len=MAX_INTERACTIVE_LEN, mlp_ratio=MLP_RATIO, n_latents=N_LATENTS,
        bottleneck_dim=BOTTLENECK_D, temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    # torch.compile on MPS can sometimes be flaky or slow to warm up. 
    enc = torch.compile(enc) 
    enc.load_state_dict(ckpt_tok["enc"])

    # Decoder
    dec = DreamerV4Decoder(
        image_size=(IMG_H, IMG_W), patch_size=PATCH, d_model=D_MODEL_ENC,
        n_layers=N_LAYERS_ENC, num_heads_q=HEADS_Q_ENC, num_heads_kv_latent=HEADS_KV_LATENT_ENC,
        bottleneck_dim=BOTTLENECK_D, seq_len=MAX_INTERACTIVE_LEN, mlp_ratio=MLP_RATIO,
        n_latents=N_LATENTS, temporal_every=TEMPORAL_EVERY
    ).to(device, dtype=torch.bfloat16).eval()
    dec = torch.compile(dec)
    if "dec" in ckpt_tok:
        dec.load_state_dict(ckpt_tok["dec"])
    
    # Dynamics
    dyn = DreamerV4Dynamics(
        action_dim=action_dim, num_latents=N_LATENTS, latent_dim=BOTTLENECK_D,
        d_model=D_MODEL_DYN, num_layers=N_LAYERS_DYN, num_heads=HEADS_Q_DYN,
        num_registers=NUM_REGISTERS, seq_len=MAX_INTERACTIVE_LEN, num_tau_levels=NUM_TAU_LEVELS,
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
    
    # We only need initial images for context. Actions will be overridden by user.
    images = traj_data['images'].unsqueeze(0).to(device).to(torch.bfloat16)
    # We use the loaded actions just to get the dimension and initial context
    loaded_actions = traj_data['actions'].unsqueeze(0).to(device).to(torch.bfloat16)
    
    B, T_total_loaded, C, H, W = images.shape
    action_dim = loaded_actions.shape[-1]
    
    # Load models
    enc, dec, dyn = load_models(args.tokenizer_ckpt, args.dynamics_ckpt, action_dim, device)
    
    # Initialize Caches
    dyn.init_cache(B, device, max_seq_len=CONTEXT_T_DYN)
    dec.init_cache(B, device, max_seq_len=CONTEXT_T_DYN)

    d_min_idx = int(np.log2(NUM_TAU_LEVELS))

    # --- PREFILL ---
    print(f"Prefilling context (0 to {args.context_len})...")
    with torch.no_grad():
        _, _, z_gt = enc(images)
        
        # Take the first context_len frames
        context_z = z_gt[:, :args.context_len]
        context_actions = loaded_actions[:, :args.context_len]

        dummy_sigma = torch.zeros((B, args.context_len), dtype=torch.long, device=device)
        dummy_step = torch.full((B, args.context_len), d_min_idx, dtype=torch.long, device=device)

        # Prefill Dynamics
        dyn.forward_step(
            action=context_actions,
            noisy_z=context_z,
            sigma_idx=dummy_sigma,
            step_idx=dummy_step,
            start_step_idx=0,
            update_cache=True
        )

        # Prefill Decoder
        dec.forward_step(context_z, start_step_idx=0, update_cache=True)

        # Initialize generation state
        # We start generation from the end of context
        current_z = context_z[:, -1:] # Last known latent [B, 1, N, D]

    # --- INTERACTIVE LOOP ---
    print("Starting Interactive Generation...")
    print("Controls:")
    print("  Arrow Keys: Move X/Y (Up/Down -> X, Left/Right -> Y)")
    print("  W/S: Move Z (W -> +Z, S -> -Z)")
    print("  Space/Enter: Gripper (Space -> Open, Enter -> Close)")
    print("  Q/Esc: Quit")

    cv2.namedWindow("Interactive Generation", cv2.WINDOW_NORMAL)
    
    # We start at t = args.context_len
    t = args.context_len
    
    last_gripper_state = context_actions[0, -1, -1]
    while t < MAX_INTERACTIVE_LEN:
        # 1. Wait for User Input
        # 0 means wait indefinitely
        key = cv2.waitKey(0) 
        
        if key == -1: # specific error handling
            continue

        # Handle Quit
        if key == ord('q') or key == 27: # Esc
            break

        # 2. Update Action Vector based on Key
        # XYZ Deltas
        delta_pos = 0.01
        
        # Let's assume standard arrow keys for simplicity, you might need to debug `print(key)` on your machine.
        print(key)

        next_action = torch.zeros_like(context_actions[0, -1])
        
        # Orientation 3:6 -> 0
        next_action[3:6] = 0.0
        
        if key == 3: # Left
             next_action[1] -= 0.01
        elif key == 2: # Right
             next_action[1] += 0.01
        elif key == 0: # Up
             next_action[0] += 0.01
        elif key == 1: # Down
             next_action[0] -= 0.01
        elif key == ord('w'): # Z up
             next_action[2] += 0.01
        elif key == ord('s'): # Z down
             next_action[2] -= 0.01
        elif key == 32: # Enter
             last_gripper_state = 1.0
        elif key == 13: # Return -> Close
             last_gripper_state = -1.0

        next_action[6] = last_gripper_state

        # Prepare action tensor for model: [B, 1, D]
        action_t = next_action.unsqueeze(0).unsqueeze(0).to(device).to(torch.bfloat16)
        start_time = time.time()

        # 3. Generate Frame t
        # Use the previous latent `current_z` and the user-specified `action_t`
        # to generate the NEXT latent `z_next`
        z_next = solve_frame_cached(dyn, action_t, current_z, t, device, num_steps=4)
        
        # 4. Decode z_next
        with torch.no_grad():
            # Decode
            _, recon_frame = dec.forward_step(z_next, start_step_idx=t, update_cache=True)

        # 5. Display
        gen_frame = recon_frame[0, 0].float().cpu().permute(1, 2, 0).numpy()
        
        # Visualization: Only Generated Frame
        display_img = cv2.cvtColor(gen_frame, cv2.COLOR_RGB2BGR)
        display_img = np.clip(display_img * 255, 0, 255).astype(np.uint8)
        
        cv2.imshow("Interactive Generation", display_img)
        
        print(f"Step {t}: {(time.time() - start_time)*1000:.1f}ms | Action: {next_action[:3].tolist()}")

        # Update loop state
        current_z = z_next
        t += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--dynamics_ckpt", type=str, required=True)
    parser.add_argument("--traj_path", type=str, required=True, help="Path to the .pt file from export_traj.py")
    parser.add_argument("--context_len", type=int, default=10)
    args = parser.parse_args()
    
    run_live_eval(args)
