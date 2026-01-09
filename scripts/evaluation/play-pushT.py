import math
import time
import os
from functools import partial  # (kept in case you extend later)
import hydra
import cv2
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from dreamerv4.datasets import PushTDataset
from dreamerv4.utils.joy import XBoxController
from dreamerv4.single_stream.utils import load_tokenizer
from dreamerv4.single_stream.utils import load_denoiser

# -------------------------------------------------------------------------
# Configurable paths / constants
# -------------------------------------------------------------------------
PUSHT_EPISODE_PATH = "/home/mim-server/datasets/pushT/224/episode_0.h5"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
JOYSTICK_ID = 0
ACTION_SCALE = 0.15
NUM_SAMPLING_STEPS = 4

# -------------------------------------------------------------------------
# Latent sampling (shortcut diffusion schedule)
# -------------------------------------------------------------------------
@torch.no_grad()
def sample_video_snippet(
    denoiser: nn.Module,
    latents: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    num_steps: int,
    K_max: int,
    num_latent_tokens: int,
    latent_dim: int
) -> torch.Tensor:
    """
    Generates a video snippet (latent trajectory) using the dyadic shortcut schedule.

    Args:
        denoiser:        DreamerV4Denoiser.
        latents:         (B, T_init, N_lat, D_lat) initial latents (condition).
        actions:         (B, T, action_dim) continuous actions.
        batch_size:      B.
        seq_len:         T (total rollout length).
        device:          torch.device.
        num_steps:       number of shortcut integration steps.
        K_max:           finest grid resolution (must be power of 2).
        latent_shape:    (N_lat, D_lat).

    Returns:
        z:               (B, T, N_lat, D_lat) final latent trajectory at τ=1.
    """
    device = torch.device(device)
    B, T = batch_size, seq_len
    N_lat, D_lat = num_latent_tokens, latent_dim
    # 1. Initialize pure noise at τ=0
    z = torch.randn(
        B,
        T,
        N_lat,
        D_lat,
        device=device,
        dtype=torch.bfloat16,
    )

    # Overwrite prefix with given latents
    T_init = latents.shape[1]
    z[:, :T_init, ...] = latents.to(device=device, dtype=torch.bfloat16)

    # 2. Setup dyadic grid parameters
    step_size = 1.0 / float(num_steps)
    max_pow2 = int(math.log2(K_max))
    step_index_raw = int(math.log2(num_steps))
    current_step_index = max_pow2 - step_index_raw

    step_index_tensor = torch.full(
        (B, T),
        current_step_index,
        dtype=torch.long,
        device=device,
    )
    d_min = 1.0 / K_max

    actions = actions.to(device=device, dtype=torch.bfloat16)

    print(f"Sampling: B={B}, T={T}, steps={num_steps}, d={step_size:.4f}")

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for k in range(num_steps):
            tau_val = k * step_size

            # Integer index for tau on the finest grid
            current_tau_index_val = int(round(tau_val / d_min))
            tau_index_tensor = torch.full(
                (B, T),
                current_tau_index_val,
                dtype=torch.long,
                device=device,
            )

            print(f"  Step {k+1}/{num_steps}: τ={tau_val:.2f} -> idx={current_tau_index_val}")

            # Predict clean z_1
            with torch.no_grad():
                z_hat = denoiser(
                    z,
                    diffusion_step=tau_index_tensor,
                    shortcut_length=step_index_tensor,
                    actions=actions,
                )

            # 4. Update z_τ → z_{τ + d}
            if k == num_steps - 1:
                z = z_hat
            else:
                # v = (z_1 - z_τ) / (1 - τ)
                denom = 1.0 - tau_val
                velocity = (z_hat - z) / denom
                z = z + velocity * step_size

    return z.to(torch.float32)

# -------------------------------------------------------------------------
# Initial latent extraction from a PushT episode
# -------------------------------------------------------------------------
def get_initial_latents_from_dataset(
    tokenizer,
    seq_len: int,
    resolution: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Load one PushT episode, encode with tokenizer, and return an initial latent
    trajectory of length seq_len (or truncated).
    """
    dataset = PushTDataset(
        hd5_file_path=PUSHT_EPISODE_PATH,
        traj_len=seq_len*2,
        load_to_ram=True,
        non_overlapping=True,
    )

    # Arbitrary index into that episode
    batch = dataset[10]
    imgs = batch["observation.image"]  # (T, C, H, W)
    imgs = interpolate(imgs, resolution)  # resize to tokenizer resolution
    imgs = imgs[None, :seq_len].to(device=device, dtype=torch.bfloat16)  # (1, T, C, H, W)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        latents = tokenizer.encode(imgs)
    return latents  # (1, T, N_lat, D_lat)

# -------------------------------------------------------------------------
# Joystick control loop
# -------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="dynamics/single_stream/pushT", version_base=None)
def main(cfg: DictConfig):
    device = DEVICE
    # 1. Load models
    print("Loading tokenizer/model wrapper...")
    tokenizer = load_tokenizer(cfg, device)

    print("Loading dynamics/denoiser...")
    denoiser = load_denoiser(cfg, device)

    # 2. Initial latents from dataset
    print("Extracting initial latents from PushT episode...")
    latents_full = get_initial_latents_from_dataset(tokenizer, 
                                                    seq_len=cfg.denoiser.max_context_length, 
                                                    resolution= tuple(cfg.dataset.resolution),
                                                    device=device)

    # Construct rolling latent buffer:
    #   start from a single frame (e.g. t=10) and tile it across T-1.
    latents = latents_full[:, 10].unsqueeze(1)             # (1, 1, N_lat, D_lat)
    latents = latents.repeat(1, cfg.denoiser.max_context_length - 1, 1, 1).clone() # (1, 31, N_lat, D_lat)

    # 3. Action buffer
    actions = torch.zeros(1, cfg.denoiser.max_context_length, 2, device=device)

    # 4. Joystick
    joy = XBoxController(JOYSTICK_ID)

    print("Starting joystick-controlled DreamerV4 rollout...")
    for i in range(1000):
        tic = time.time()
        # ---- Read joystick and update action buffer ----
        with torch.no_grad():
            # Roll the buffer
            actions[:, :-1] = actions[:, 1:].clone()

            # Read joystick: right_joy is assumed to be (x, y) in [-1, 1]
            states = joy.getStates()
            cmd = states["right_joy"] * ACTION_SCALE

            # Map joystick Y to forward/back, X to lateral
            actions[0, -1, 0] = -cmd[1]  # forward/back
            actions[0, -1, 1] = cmd[0]   # sideways

            # ---- Sample one new latent trajectory ----
            sampled_latents = sample_video_snippet(
                denoiser=denoiser,
                latents=latents,
                actions=actions,
                batch_size=1,
                seq_len=cfg.denoiser.max_context_length,
                device=device,
                num_steps=NUM_SAMPLING_STEPS,
                K_max=cfg.denoiser.K_max,
                num_latent_tokens=cfg.denoiser.num_latent_tokens,
                latent_dim=cfg.denoiser.latent_dim,
            )

            # Use everything except the first step as the new latent buffer
            latents[:] = sampled_latents[:, 1:, ...]

            # ---- Decode the last latent into an image ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_last = latents[:, -1].unsqueeze(1)  # (1, 1, N_lat, D_lat)
                z_decoded = tokenizer.decoder(z_last)
                imgs_recon = tokenizer.image_head(z_decoded)
                imgs_recon = (imgs_recon + 1.0) / 2.0  # [-1,1] → [0,1]

        # ---- Display image with OpenCV ----
        img = torch.clamp(imgs_recon.squeeze(0).squeeze(0), 0.0, 1.0)
        img = (img.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.imshow("dreamerv4", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        while(time.time()-tic < 0.2):
            time.sleep(0.001)

    cv2.destroyAllWindows()

# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
