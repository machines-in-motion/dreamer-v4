import math
import time
import os
from functools import partial  # (kept in case you extend later)

import cv2
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose

from model.dynamics import *  # DreamerV4DenoiserCfg, DreamerV4Denoiser, apply_denoiser
from model.tokenizer import (
    CausalTokenizerDecoder,
    CausalTokenizerEncoder,
    CausalTokenizerConfig,
    TokensToImageHead,
    ImagePatchifier,
)
from model.utils import TokenMasker
from dataset import SingleViewSequenceDataset
from joy import XBoxController


# -------------------------------------------------------------------------
# Configurable paths / constants
# -------------------------------------------------------------------------

CONFIG_PATH = "config"

TOKENIZER_CONFIG_NAME = "tokenizer_large_cfg1.yaml"
TOKENIZER_CKPT_PATH = "/home/mim-server/projects/rooholla/dreamer-v4/checkpoints/tokenizer_ckpts/large/cfg2/19.pt"


DYNAMICS_CONFIG_NAME = "dynamics_small.yaml"
DYNAMICS_CKPT_PATH = "/home/mim-server/projects/rooholla/dreamer-v4/checkpoints/dynamics_ckpts/2025-12-04_23-37-40/checkpoints/checkpoint_step_0271149.pt"

PUSHT_EPISODE_PATH = "/home/mim-server/datasets/pushT/224/episode_0.h5"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 32          # rollout horizon in latents
NUM_SAMPLING_STEPS = 4
K_MAX = 128
LATENT_SHAPE = (256, 32)  # (N_lat, D_lat) – must match your dynamics config
JOYSTICK_ID = 0
ACTION_SCALE = 0.15


# -------------------------------------------------------------------------
# Hydra helpers
# -------------------------------------------------------------------------

def load_hydra_config(config_name: str, config_path: str = CONFIG_PATH) -> DictConfig:
    """
    Load a Hydra config from `config_path/config_name`.
    """
    # version_base=None is compatible across Hydra ≥1.1
    with initialize(config_path=config_path, version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


# -------------------------------------------------------------------------
# Tokenizer wrapper
# -------------------------------------------------------------------------

class ModelWrapper(nn.Module):
    """
    Wraps tokenizer encoder/decoder + patchifier/image head/masker into a single module.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        tokenizer_cfg = CausalTokenizerConfig(**OmegaConf.to_object(cfg.tokenizer))

        self.encoder = CausalTokenizerEncoder(tokenizer_cfg)
        self.decoder = CausalTokenizerDecoder(tokenizer_cfg)
        self.patchifier = ImagePatchifier(
            cfg.tokenizer.patch_size,
            cfg.tokenizer.model_dim,
        )
        self.image_head = TokensToImageHead(
            cfg.tokenizer.model_dim,
            cfg.dataset.resolution,
            cfg.tokenizer.patch_size,
        )
        self.masker = TokenMasker(
            cfg.tokenizer.model_dim,
            cfg.tokenizer.num_modality_tokens,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        End-to-end image reconstruction through tokenizer.

        Args:
            images: (B, T, C, H, W) in [0, 1]

        Returns:
            recon_images: (B, T, C, H, W) in [0, 1]
        """
        # 1. Normalize to [-1, 1]
        images = (images * 2.0) - 1.0

        # 2. Patchify and mask
        tokens = self.patchifier(images)
        masked_tokens = self.masker(tokens)

        # 3. Encode / decode
        z, _ = self.encoder(masked_tokens)
        z_decoded = self.decoder(z)

        # 4. Image head + clamp back to [0, 1]
        recon_images = self.image_head(z_decoded)
        recon_images = (recon_images + 1.0) / 2.0
        return torch.clamp(recon_images, 0.0, 1.0)


# -------------------------------------------------------------------------
# Checkpoint helpers
# -------------------------------------------------------------------------

def load_tokenizer_model(device: torch.device) -> ModelWrapper:
    """
    Load tokenizer (encoder+decoder) and heads from checkpoint.
    """
    cfg = load_hydra_config(TOKENIZER_CONFIG_NAME)
    model = ModelWrapper(cfg).to(device)

    state = torch.load(TOKENIZER_CKPT_PATH, map_location=device)
    sd = state["model"]

    # Clean FSDP `_orig_mod.` keys
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    model.load_state_dict(clean_sd, strict=True)
    model.eval()

    return model


def load_denoiser(device: torch.device) -> nn.Module:
    """
    Load DreamerV4 denoiser from checkpoint.
    """
    dynamics_cfg = load_hydra_config(DYNAMICS_CONFIG_NAME)
    denoiser_cfg = DreamerV4DenoiserCfg(**OmegaConf.to_object(dynamics_cfg.denoiser))
    denoiser = DreamerV4Denoiser(denoiser_cfg).to(device)

    state = torch.load(DYNAMICS_CKPT_PATH, map_location=device)
    sd = state["model"]

    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    denoiser.load_state_dict(clean_sd, strict=True)
    denoiser.eval()

    return denoiser


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
    num_steps: int = NUM_SAMPLING_STEPS,
    K_max: int = K_MAX,
    latent_shape: tuple[int, int] = LATENT_SHAPE,
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
    N_lat, D_lat = latent_shape

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
            z_hat = apply_denoiser(
                denoiser,
                x_tau=z,
                tau_index=tau_index_tensor,
                step_index=step_index_tensor,
                no_grad=True,
                continuous_actions=actions,
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

def get_initial_latents_from_pusht(
    model: ModelWrapper,
    device: torch.device,
    seq_len: int = SEQ_LEN,
) -> torch.Tensor:
    """
    Load one PushT episode, encode with tokenizer, and return an initial latent
    trajectory of length seq_len (or truncated).
    """
    dataset = SingleViewSequenceDataset(
        hd5_file_path=PUSHT_EPISODE_PATH,
        traj_len=64,
        load_to_ram=True,
        non_overlapping=True,
    )

    # Arbitrary index into that episode
    batch = dataset[10]
    imgs = batch["observation.image"]  # (T, C, H, W)
    imgs = interpolate(imgs, (256, 256))  # resize to tokenizer resolution
    imgs = imgs[None, :seq_len].to(device=device, dtype=torch.bfloat16)  # (1, T, C, H, W)

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        patches = model.patchifier((imgs * 2.0) - 1.0)
        latents, _ = model.encoder(patches)

    return latents  # (1, T, N_lat, D_lat)


# -------------------------------------------------------------------------
# Joystick control loop
# -------------------------------------------------------------------------

def run_joystick_control_loop():
    device = DEVICE
    print(f"Using device: {device}")

    # 1. Load models
    print("Loading tokenizer/model wrapper...")
    model = load_tokenizer_model(device)

    print("Loading dynamics/denoiser...")
    denoiser = load_denoiser(device)

    # 2. Initial latents from dataset
    print("Extracting initial latents from PushT episode...")
    latents_full = get_initial_latents_from_pusht(model, device, seq_len=SEQ_LEN)

    # Construct rolling latent buffer:
    #   start from a single frame (e.g. t=10) and tile it across T-1.
    latents = latents_full[:, 10].unsqueeze(1)             # (1, 1, N_lat, D_lat)
    latents = latents.repeat(1, SEQ_LEN - 1, 1, 1).clone() # (1, 31, N_lat, D_lat)

    # 3. Action buffer
    actions = torch.zeros(1, SEQ_LEN, 2, device=device)

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
                seq_len=SEQ_LEN,
                device=device,
                num_steps=NUM_SAMPLING_STEPS,
                K_max=K_MAX,
                latent_shape=LATENT_SHAPE,
            )

            # Use everything except the first step as the new latent buffer
            latents[:] = sampled_latents[:, 1:, ...]

            # ---- Decode the last latent into an image ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                z_last = latents[:, -1].unsqueeze(1)  # (1, 1, N_lat, D_lat)
                z_decoded = model.decoder(z_last)
                imgs_recon = model.image_head(z_decoded)
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
    run_joystick_control_loop()
