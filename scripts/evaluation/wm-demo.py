import math
import time
import os

import cv2
import zerorpc
import numpy as np

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

SEQ_LEN = 32             # rollout horizon in latents (T)
NUM_SAMPLING_STEPS = 4
K_MAX = 128
LATENT_SHAPE = (256, 32)  # (N_lat, D_lat) – must match your dynamics config
JOYSTICK_ID = 0
ACTION_SCALE_WORLD = 0.12    # scaling for world model actions
ACTION_SCALE_ROBOT = 0.12    # scaling for robot relative motion
CONTROL_DT = 0.2             # control step (seconds)

CAM_RES = (256, 256)         # (H, W) for camera & world model display


# -------------------------------------------------------------------------
# Hydra helpers
# -------------------------------------------------------------------------

def load_hydra_config(config_name: str, config_path: str = CONFIG_PATH) -> DictConfig:
    """
    Load a Hydra config from `config_path/config_name`.
    """
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

    if device.type == "cuda":
        autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        # No autocast on CPU
        class DummyCtx:
            def __enter__(self): return None
            def __exit__(self, exc_type, exc, tb): return False
        autocast_ctx = DummyCtx()

    with autocast_ctx:
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
                denom = 1.0 - tau_val
                velocity = (z_hat - z) / denom
                z = z + velocity * step_size

    return z.to(torch.float32)


# -------------------------------------------------------------------------
# Initial latent extraction (dataset or camera)
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

    batch = dataset[10]
    imgs = batch["observation.image"]  # (T, C, H, W)
    imgs = interpolate(imgs, (256, 256))
    imgs = imgs[None, :seq_len].to(device=device, dtype=torch.bfloat16)  # (1, T, C, H, W)

    with torch.no_grad():
        if device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class DummyCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            autocast_ctx = DummyCtx()

        with autocast_ctx:
            patches = model.patchifier((imgs * 2.0) - 1.0)
            latents, _ = model.encoder(patches)

    return latents  # (1, T, N_lat, D_lat)


def encode_camera_image_to_latent_prefix(
    image_tensor: torch.Tensor,
    model: ModelWrapper,
    device: torch.device,
    seq_len: int = SEQ_LEN,
) -> torch.Tensor:
    """
    Encode a single camera image into a latent sequence prefix.

    Args:
        image_tensor: (C, H, W) in [0, 1]
        model:        ModelWrapper
        device:       torch.device
        seq_len:      total rollout length T

    Returns:
        latents: (1, T_init, N_lat, D_lat) with T_init = seq_len - 1
    """
    with torch.no_grad():
        img = image_tensor.to(device=device, dtype=torch.bfloat16)  # (C,H,W)
        imgs = img.unsqueeze(0).unsqueeze(0)  # (1,1,C,H,W)

        if device.type == "cuda":
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            class DummyCtx:
                def __enter__(self): return None
                def __exit__(self, exc_type, exc, tb): return False
            autocast_ctx = DummyCtx()

        with autocast_ctx:
            patches = model.patchifier((imgs * 2.0) - 1.0)
            latents, _ = model.encoder(patches)  # (1,1,N,D)

    # Repeat the single latent frame across T-1 steps
    single_latent = latents[:, 0].unsqueeze(1)  # (1,1,N,D)
    latents_prefix = single_latent.repeat(1, seq_len - 1, 1, 1).clone()  # (1,T-1,N,D)
    return latents_prefix


# -------------------------------------------------------------------------
# Robot RPC + camera
# -------------------------------------------------------------------------

class RPCRobot:
    def __init__(self, host="tcp://127.0.0.1:5000"):
        self.host = host
        self.robot = zerorpc.Client()
        self.robot.connect(self.host)

    def setRelativeAction(self, action):
        """
        Set a relative movement: 6D vector.
        """
        assert action.shape == (6,), "Action must be a 6D vector."
        self.robot.setRelativeAction(action.tolist())

    def goHome(self):
        self.robot.goHome()

    def getState(self):
        return self.robot.getRobotState()

    def start(self):
        self.robot.start()


def send_robot_action(robot: RPCRobot, cmd_xy: np.ndarray,
                      action_scale=ACTION_SCALE_ROBOT, dt=1.0):
    """
    Map joystick XY command to relative 6D action for the robot.
    cmd_xy: np.array([x, y]) in roughly [-1, 1].
    """
    cmd_xy = np.asarray(cmd_xy, dtype=np.float32)
    # Scale and clamp
    delta = np.clip(cmd_xy * dt, -action_scale, action_scale)
    a = np.zeros(6, dtype=np.float32)
    # Map X,Y to two translational axes (adjust to your convention)
    a[0] = delta[0]
    a[1] = delta[1]
    robot.setRelativeAction(a)


# Camera client (Femto cam)
femto_cam = zerorpc.Client()
# NOTE: This is exactly as in your script; adjust if needed.
femto_cam.connect("tcp://127.0.0.0:4243")


def get_camera_image(resolution=(256, 256)) -> torch.Tensor:
    """
    Fetch image from camera RPC, resize, and return as torch tensor (C,H,W) in [0,1].

    The incoming image is assumed RGB; we convert to BGR to match cv2 default.
    """
    img1 = np.array(femto_cam.get_image())           # (H,W,3) in [0,255], RGB
    img1 = (img1.astype(np.float32) / 255.0)
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)     # now BGR
    img_tensor = torch.from_numpy(img1).permute(2, 0, 1)[None]  # (1,3,H,W)
    img_tensor = interpolate(img_tensor, size=resolution).squeeze(0)  # (3,H,W)
    return img_tensor  # [0,1], BGR


from scipy.interpolate import CubicSpline
def ScalarSpline(x_points):
    """
    Create a cubic spline with zero derivative at both ends.
    
    Parameters:
        x_points (array-like): 1D sorted array of x coordinates.
        y_points (array-like): Corresponding y coordinates.
        
    Returns:
        spline (callable): Function to evaluate the spline.
        spline_derivative (callable): Function to evaluate the derivative.
    """
    y_points = np.asarray(x_points)
    x_points = np.linspace(0,1,x_points.shape[0])
    # Create cubic spline with clamped boundary conditions (derivative=0 at ends)
    spline = CubicSpline(x_points, y_points, bc_type=((1, 0.0), (1, 0.0)))
    # First derivative of the spline
    spline_derivative = spline.derivative()
    return spline, spline_derivative

def VectorSpline(x, n):
    assert len(x.shape)==2, 'X should be of shape NxD'
    T,D = x.shape
    phis = []
    dPhis = []
    t = np.linspace(0, 1, n)
    for d in range(D):
        phi, dPhi = ScalarSpline(x[:, d])
        phis.append(phi(t))
        dPhis.append(dPhi(t))
    return np.vstack(phis).T, np.vstack(dPhis).T
    
def interpolateActions(action_chunk, action_dt=0.2, desired_dt = 0.02):
    plan_time = action_chunk.shape[0]*action_dt
    a = torch.vstack([torch.zeros(2).cuda(), action_chunk])
    chunk = torch.cumsum(a, dim=0).cpu()*action_dt
    phi, dphi = VectorSpline(chunk, int(plan_time//desired_dt))
    deltas = phi[1:]-phi[:-1]
    return deltas

# -------------------------------------------------------------------------
# Combined joystick / robot / world-model demo
# -------------------------------------------------------------------------

def run_combined_demo(timeout=600):
    device = DEVICE
    print(f"Using device: {device}")

    # 1. Load models
    print("Loading tokenizer/model wrapper...")
    model = load_tokenizer_model(device)

    print("Loading dynamics/denoiser...")
    denoiser = load_denoiser(device)

    # 2. Initial latents from PushT episode (used before first camera reset)
    print("Extracting initial latents from PushT episode...")
    latents_full = get_initial_latents_from_pusht(model, device, seq_len=SEQ_LEN)
    # Construct rolling latent buffer: take some frame and tile across T-1
    latents = latents_full[:, 10].unsqueeze(1)               # (1,1,N,D)
    latents = latents.repeat(1, SEQ_LEN - 1, 1, 1).clone()   # (1,T-1,N,D)

    # 3. Action buffer for world model
    actions = torch.zeros(1, SEQ_LEN, 2, device=device)

    # 4. Joystick
    joy = XBoxController(JOYSTICK_ID)

    # 5. Robot
    robot = RPCRobot("tcp://127.0.0.1:5000")
    # robot.start()  # Uncomment if your RPC server expects this

    # 6. OpenCV window & mouse callback
    window_name = "Robot (left)  |  World Model (right)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    control_mode = "robot"     # "robot" or "world"
    wm_reset_pending = False   # set True when user clicks on world-model image
    single_width = CAM_RES[1]  # width of each image (256 by default)

    def mouse_callback(event, x, y, flags, param):
        nonlocal control_mode, wm_reset_pending
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < param:
                control_mode = "robot"
                print("[MODE] Switched to REAL ROBOT control.")
            else:
                control_mode = "world"
                wm_reset_pending = True
                print("[MODE] Switched to WORLD MODEL control (reset from camera).")

    cv2.setMouseCallback(window_name, mouse_callback, param=single_width)

    # 7. Main loop
    start_time = time.time()
    last_cam_tensor = torch.zeros(3, CAM_RES[0], CAM_RES[1])  # placeholder
    world_img_vis = np.zeros((CAM_RES[0], CAM_RES[1], 3), dtype=np.uint8)

    print("Starting combined robot + world model demo...")
    while time.time() - start_time < timeout:
        tic = time.time()

        # ---- 1) Get latest camera image ----
        with torch.no_grad():
            cam_tensor = get_camera_image(resolution=CAM_RES)  # (3,H,W) in [0,1]
        last_cam_tensor = cam_tensor.clone()

        # Convert camera tensor to displayable uint8 (BGR)
        cam_vis = (cam_tensor.permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()

        # ---- 2) Read joystick and update action buffer ----
        with torch.no_grad():
            # Roll the buffer
            actions[:, :-1] = actions[:, 1:].clone()

            states = joy.getStates()
            # Joystick command in [-1,1]^2 (assumed)
            cmd = states["right_joy"] * ACTION_SCALE_WORLD

            # Map joystick Y to forward/back, X to lateral
            
            actions[0, -1, 0] = -cmd[1]  # forward/back
            actions[0, -1, 1] = cmd[0]   # sideways
            # While in robot control mode, the actions should not control the world model
            if control_mode == 'robot':
                actions[0, -1, 0] = 0.
                actions[0, -1, 1] = 0.


        # ---- 3) Apply robot control if in robot mode ----
        if control_mode == "robot":
            # interpolated_act = interpolateActions(torch.tensor([-cmd[1], cmd[0]]).cuda())
            # for i in range(interpolated_act.shape[0]):
            #     tic_intr = time.time()
            #     send_robot_action(robot, cmd_xy=interpolated_act[i],
            #                     action_scale=ACTION_SCALE_ROBOT, dt=1.)
            #     while time.time()-tic_intr < 0.02:
            #         time.sleep(0.0001)
            send_robot_action(robot, cmd_xy=np.array([-cmd[1], cmd[0]]),
                                action_scale=ACTION_SCALE_ROBOT, dt=0.2)
                

        # ---- 4) If world-model reset requested, re-encode camera image ----
        if wm_reset_pending:
            print("Resetting world model state from latest camera image...")
            latents = encode_camera_image_to_latent_prefix(
                image_tensor=last_cam_tensor,
                model=model,
                device=device,
                seq_len=SEQ_LEN,
            )
            actions.zero_()
            wm_reset_pending = False

        # ---- 5) World model rollout & decode last frame ----
        with torch.no_grad():
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

            # Use everything except the first step as new latent buffer
            latents = sampled_latents[:, 1:, ...]

            # Decode the last latent into an image
            if device.type == "cuda":
                autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            else:
                class DummyCtx:
                    def __enter__(self): return None
                    def __exit__(self, exc_type, exc, tb): return False
                autocast_ctx = DummyCtx()

            with autocast_ctx:
                z_last = latents[:, -1].unsqueeze(1)  # (1,1,N_lat,D_lat)
                z_decoded = model.decoder(z_last)
                imgs_recon = model.image_head(z_decoded)
                imgs_recon = (imgs_recon + 1.0) / 2.0  # [-1,1] → [0,1]

        img_wm = torch.clamp(imgs_recon.squeeze(0).squeeze(0), 0.0, 1.0)
        world_img_vis = (img_wm.permute(1, 2, 0) * 255.0).to(torch.uint8).cpu().numpy()

        # ---- 6) Compose side-by-side image ----
        cam_vis_copy = cv2.resize(cv2.cvtColor(cam_vis.copy(), cv2.COLOR_BGR2RGB), (512, 512))
        world_vis_copy = cv2.resize(cv2.cvtColor(world_img_vis.copy(), cv2.COLOR_BGR2RGB), (512, 512))

        # Overlay labels
        cv2.putText(
            cam_vis_copy,
            "REAL ROBOT",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            world_vis_copy,
            "WORLD MODEL",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if control_mode == "robot":
            cv2.putText(
                cam_vis_copy,
                "[ACTIVE]",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                world_vis_copy,
                "[ACTIVE]",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        combined = np.concatenate([cam_vis_copy, world_vis_copy], axis=1)

        # ---- 7) Show and handle keyboard ----
        cv2.imshow(window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Exiting demo...")
            break

        # ---- 8) Maintain ~CONTROL_DT rate ----
        elapsed = time.time() - tic
        if elapsed < CONTROL_DT:
            time.sleep(CONTROL_DT - elapsed)

    cv2.destroyAllWindows()


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    run_combined_demo()
