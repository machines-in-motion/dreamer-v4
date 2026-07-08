"""Interactive PushT world-model rollout.

Seeds the causal tokenizer + action-conditioned dynamics denoiser with a short
context clip, then rolls the world model forward autoregressively while an
Xbox joystick supplies the 2-D action each frame. Uses KV-caching so each frame
costs a single denoising pass plus one cache commit (real-time on a single GPU).

Checkpoints default to the paths in `scripts/config/dynamics/pushT.yaml`
(`dynamics_ckpt` / `tokenizer_ckpt`), so the plain command just works:

    python scripts/play-pushT.py

Override on the CLI if needed, e.g.:

    python scripts/play-pushT.py \
        dynamics_ckpt=/path/to/dynamics.pt tokenizer_ckpt=/path/to/tokenizer.pt

Runs without hardware too: if no joystick is found the action follows an
autonomous circular pattern, and if no display is available (headless/SSH) the
window is skipped — pass RECORD_MP4=True to still capture the rollout to a file.
"""
import contextlib
import math
import os
import time

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.nn.functional import interpolate
from tqdm import tqdm

from dreamerv4.datasets import PushTDataset
from dreamerv4.models.utils import load_denoiser, load_tokenizer
from dreamerv4.sampling import AutoRegressiveForwardDynamics
from dreamerv4.utils.joy import XBoxController

# -------------------------------------------------------------------------
# Configurable constants (Hydra overrides for these will land in a follow-up)
# -------------------------------------------------------------------------
PUSHT_EPISODE_PATH = "/home/mim-server/datasets/pushT/224/episode_0.h5"
DATASET_START_IDX = 10        # which window of the seed episode to prime from
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16
JOYSTICK_ID = 0
ACTION_SCALE = 0.1            # joystick unit → action units (matches training scale)
NUM_INIT_FRAMES = 8          # context frames used to prime the caches
CONTEXT_LEN = 32             # temporal KV-cache window
DENOISING_STEPS = 4          # shortcut Euler steps per generated frame
CONTEXT_COND_TAU = 0.9       # how "clean" cached context frames are kept (→1 = cleaner)
NUM_FORWARD_STEPS = 1000     # number of frames to imagine
TARGET_FPS = 5.0             # PushT was trained at dt=0.2 (5 Hz)
RECORD_MP4 = False           # set True to also dump the rollout to an .mp4
RECORD_DIR = "."


# -------------------------------------------------------------------------
# Initial context extraction from a PushT episode
# -------------------------------------------------------------------------
def get_initial_context(resolution, device):
    """Load a short clip from a PushT episode and return (imgs, actions).

    imgs    : (1, T, C, H, W) in [0, 1], resized to the tokenizer resolution.
    actions : (1, T, n_act)
    """
    dataset = PushTDataset(
        hd5_file_path=PUSHT_EPISODE_PATH,
        traj_len=64,
        load_to_ram=True,
        non_overlapping=True,
    )
    batch = dataset[DATASET_START_IDX]
    imgs = batch["observation.image"]                       # (T, C, H, W)
    actions = batch["action"]                               # (T, n_act)
    imgs = interpolate(imgs, resolution).to(device=device)[None]   # (1, T, C, H, W)
    return imgs, actions[None].to(device)                          # (1, T, n_act)


# -------------------------------------------------------------------------
# Joystick with graceful fallback to an autonomous action pattern
# -------------------------------------------------------------------------
class ActionSource:
    def __init__(self, n_act):
        self.n_act = n_act
        self.joy = None
        try:
            self.joy = XBoxController(JOYSTICK_ID)
            print("[play-pushT] joystick connected — right stick drives the action.")
        except Exception as e:  # no controller / no pygame video, etc.
            print(f"[play-pushT] no joystick ({e}); using autonomous circular action. "
                  "Connect an Xbox controller to steer.")

    def get(self, step, out):
        """Fill `out` (1, 1, n_act) in place and report whether to quit."""
        out.zero_()
        if self.joy is not None:
            states = self.joy.getStates()
            cmd = states["right_joy"]
            if self.n_act >= 1:
                out[..., 0] = -cmd[1] * ACTION_SCALE
            if self.n_act >= 2:
                out[..., 1] = cmd[0] * ACTION_SCALE
            return bool(states["options_right"])
        # autonomous: slow circle
        if self.n_act >= 1:
            out[..., 0] = ACTION_SCALE * math.cos(step / 15.0)
        if self.n_act >= 2:
            out[..., 1] = ACTION_SCALE * math.sin(step / 15.0)
        return False

    def close(self):
        if self.joy is not None:
            try:
                self.joy.close()
            except Exception:
                pass


def _try_open_window(name):
    try:
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        return True
    except cv2.error as e:
        print(f"[play-pushT] no display ({e}); running headless.")
        return False


# -------------------------------------------------------------------------
# Main loop
# -------------------------------------------------------------------------
@hydra.main(config_path="config", config_name="dynamics/pushT", version_base=None)
def main(cfg: DictConfig):
    n_act = int(cfg.denoiser.n_actions)

    # RoPE / temporal tables must span every absolute frame position we will
    # reach: the seed frames plus every imagined frame (+ margin). Undersizing
    # this crashes late in a long rollout when the position index runs past the
    # table.
    rope_len = NUM_INIT_FRAMES + NUM_FORWARD_STEPS + 8

    print("[play-pushT] loading models ...")
    denoiser = load_denoiser(cfg, DEVICE, max_num_forward_steps=rope_len).eval()
    tokenizer = load_tokenizer(cfg, DEVICE, max_num_forward_steps=rope_len).eval()

    print("[play-pushT] extracting initial context from dataset ...")
    imgs, actions = get_initial_context(tuple(cfg.dataset.resolution), device=DEVICE)

    world = AutoRegressiveForwardDynamics(
        denoiser, tokenizer,
        context_length=CONTEXT_LEN,
        max_forward_steps=rope_len,
        context_cond_tau=CONTEXT_COND_TAU,
        denoising_step_count=DENOISING_STEPS,
        device=DEVICE,
        dtype=DTYPE,
    )

    use_cuda = (DEVICE.type == "cuda")
    autocast = torch.autocast("cuda", dtype=DTYPE) if use_cuda else contextlib.nullcontext()

    with autocast:
        world.reset(imgs[:, :NUM_INIT_FRAMES], actions[:, :NUM_INIT_FRAMES])

    src = ActionSource(n_act)
    action_t = torch.zeros(1, 1, n_act, device=DEVICE, dtype=DTYPE)

    win = "dreamerv4-pushT"
    display_ok = _try_open_window(win)

    writer = None
    if RECORD_MP4:
        os.makedirs(RECORD_DIR, exist_ok=True)
        H, W = tuple(cfg.dataset.resolution)
        path = os.path.join(RECORD_DIR, time.strftime("play-pushT-%Y-%m-%d_%H-%M-%S.mp4"))
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(TARGET_FPS), (W, H))
        writer = writer if writer.isOpened() else None
        print(f"[play-pushT] recording → {path}" if writer else
              "[play-pushT] WARN: could not open VideoWriter; recording disabled.")

    target_dt = 1.0 / TARGET_FPS
    print("[play-pushT] starting rollout — press 'q' (window) or Options (joystick) to quit.")
    try:
        for step in tqdm(range(NUM_FORWARD_STEPS)):
            tic = time.time()
            quit_requested = src.get(step, action_t)

            with autocast:
                img = world.step(action_t)                 # (1, C, H, W) in [0, 1]

            frame_rgb = (img[0].clamp(0, 1).permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)  # tokenizer emits RGB; cv2 wants BGR

            if writer is not None:
                writer.write(frame_bgr)

            if display_ok:
                cv2.imshow(win, frame_bgr)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            if quit_requested:
                break

            # Real-time pacing at TARGET_FPS.
            while time.time() - tic < target_dt:
                time.sleep(0.001)
    finally:
        src.close()
        if writer is not None:
            writer.release()
        if display_ok:
            cv2.destroyAllWindows()
    print("[play-pushT] done.")


if __name__ == "__main__":
    main()
