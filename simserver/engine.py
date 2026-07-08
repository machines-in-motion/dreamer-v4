"""SimEngine — one loaded pushT world model, one live rollout session.

The denoiser/tokenizer KV caches live on the shared model modules, so exactly
ONE rollout session is active at a time (matches the single-env design). Every
GPU call is serialized under a lock, so the FastAPI threadpool and the
WebSocket handler can both drive it safely.
"""
import base64
import contextlib
import io
import threading
import time
import uuid

import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate

from dreamerv4.datasets import PushTDataset
from dreamerv4.models.utils import load_denoiser, load_tokenizer
from dreamerv4.sampling import AutoRegressiveForwardDynamics


class SessionError(RuntimeError):
    """Raised when a request targets a session that is not the active one."""


class SimEngine:
    def __init__(
        self,
        cfg,
        device,
        *,
        seed_episode,
        seed_window: int = 64,
        seed_clip_idx: int = 10,
        num_init_frames: int = 8,
        context_len: int = 32,
        denoising_steps: int = 4,
        context_cond_tau: float = 0.9,
        action_scale: float = 0.1,
        max_steps: int = 1000,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.cfg = cfg
        self.device = torch.device(device)
        self.dtype = dtype if self.device.type == "cuda" else torch.float32
        self.num_init_frames = int(num_init_frames)
        self.context_len = int(context_len)
        self.denoising_steps = int(denoising_steps)
        self.context_cond_tau = float(context_cond_tau)
        self.action_scale = float(action_scale)
        self.max_steps = int(max_steps)
        self.n_act = int(cfg.denoiser.n_actions)
        self.resolution = tuple(int(x) for x in cfg.dataset.resolution)  # (H, W)

        # RoPE / temporal tables must span seed frames + every imagined frame.
        self._rope_len = self.num_init_frames + self.max_steps + 8
        self.denoiser = load_denoiser(cfg, self.device, max_num_forward_steps=self._rope_len).eval()
        self.tokenizer = load_tokenizer(cfg, self.device, max_num_forward_steps=self._rope_len).eval()

        # Cache a small pool of real seed frames on-GPU (avoids holding the whole
        # multi-GB episode in RAM).
        ds = PushTDataset(hd5_file_path=seed_episode, traj_len=seed_window,
                          load_to_ram=False, non_overlapping=True)
        batch = ds[min(seed_clip_idx, len(ds) - 1)]
        imgs = interpolate(batch["observation.image"], self.resolution)   # (T,C,H,W) in [0,1]
        acts = batch["action"][:, : self.n_act]                           # (T, n_act)
        self.seed_imgs = imgs.to(self.device)
        self.seed_acts = acts.to(self.device)
        self.seed_window = int(self.seed_imgs.shape[0])

        self._lock = threading.Lock()
        self.session_id = None
        self.world = None
        self.t = 0
        self._last_latent = None                 # (1,1,N,D)
        self._last_image = None                  # (C,H,W) float in [0,1], on device

    # ---------------------------------------------------------------- helpers
    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=self.dtype)
        return contextlib.nullcontext()

    def info(self) -> dict:
        return {
            "env": "pushT-neural-sim",
            "action_dim": self.n_act,
            "action_low": [-1.0] * self.n_act,
            "action_high": [1.0] * self.n_act,
            "action_scale": self.action_scale,
            "image_shape": [self.resolution[0], self.resolution[1], 3],
            "image_dtype": "uint8",
            "latent_shape": [int(self.cfg.denoiser.num_latent_tokens),
                             int(self.cfg.denoiser.latent_dim)],
            "latent_dtype": "float16",
            "dt": float(self.cfg.dataset.get("dt", 0.2)),
            "max_steps": self.max_steps,
            "num_denoising_steps": self.denoising_steps,
            "context_length": self.context_len,
            "reward": "none — the pushT dynamics has no reward head; reward is always 0.0",
            "device": str(self.device),
        }

    # ------------------------------------------------------------------- api
    def reset(self, seed=None, start_idx: int = 0) -> str:
        with self._lock:
            if seed is not None:
                torch.manual_seed(int(seed))
            n0 = self.num_init_frames
            span = max(1, self.seed_window - n0 + 1)
            s = int(start_idx) % span
            imgs = self.seed_imgs[s:s + n0].unsqueeze(0)                    # (1,n0,C,H,W)
            acts = self.seed_acts[s:s + n0].unsqueeze(0).to(self.dtype)     # (1,n0,n_act)

            world = AutoRegressiveForwardDynamics(
                self.denoiser, self.tokenizer,
                context_length=self.context_len,
                max_forward_steps=self._rope_len,
                context_cond_tau=self.context_cond_tau,
                denoising_step_count=self.denoising_steps,
                device=self.device, dtype=self.dtype,
            )
            with torch.no_grad(), self._autocast():
                world.reset(imgs, acts)
                init_img = self.tokenizer.decode(world.current_z)[:, 0]     # (1,C,H,W)

            self.world = world
            self.session_id = uuid.uuid4().hex
            self.t = 0
            self._last_latent = world.current_z
            self._last_image = init_img[0].float().clamp(0.0, 1.0)
            return self.session_id

    def step(self, session_id: str, action) -> dict:
        with self._lock:
            if self.world is None or session_id != self.session_id:
                raise SessionError("no active session for this session_id; call reset()")
            a = np.asarray(action, dtype=np.float32).reshape(-1)
            if a.shape[0] < self.n_act:
                a = np.pad(a, (0, self.n_act - a.shape[0]))
            a = np.clip(a[: self.n_act], -1.0, 1.0) * self.action_scale
            at = torch.tensor(a, device=self.device, dtype=self.dtype).view(1, 1, self.n_act)

            t0 = time.perf_counter()
            with torch.no_grad(), self._autocast():
                img = self.world.step(at)                                  # (1,C,H,W)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            step_ms = (time.perf_counter() - t0) * 1000.0

            self.t += 1
            self._last_latent = self.world.current_z
            self._last_image = img[0].float().clamp(0.0, 1.0)
            return {"t": self.t, "reward": 0.0, "terminated": False,
                    "truncated": self.t >= self.max_steps, "step_ms": step_ms}

    def close(self, session_id: str) -> bool:
        with self._lock:
            if session_id == self.session_id:
                self.world = None
                self.session_id = None
                self.t = 0
                return True
            return False

    # -------------------------------------------------------------- encoding
    def encode_obs(self, obs_type: str = "image", encoding: str = "raw_u8") -> dict:
        if self._last_image is None:
            raise SessionError("no observation yet; call reset()")
        if obs_type == "latent":
            lat = self._last_latent[0, 0].detach().to(torch.float16).cpu().numpy()  # (N,D)
            return {"encoding": "raw_f16", "shape": list(lat.shape), "dtype": "float16",
                    "data": base64.b64encode(lat.tobytes()).decode()}
        img = (self._last_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()  # (H,W,3)
        if encoding == "jpeg":
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, "JPEG", quality=88)
            return {"encoding": "jpeg", "shape": list(img.shape), "dtype": "uint8",
                    "data": base64.b64encode(buf.getvalue()).decode()}
        return {"encoding": "raw_u8", "shape": list(img.shape), "dtype": "uint8",
                "data": base64.b64encode(img.tobytes()).decode()}

    def last_jpeg_bytes(self, quality: int = 85) -> bytes:
        img = (self._last_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=quality)
        return buf.getvalue()
