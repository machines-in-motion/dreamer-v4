"""SimEngine — one loaded pushT world model, one live rollout session.

The denoiser/tokenizer KV caches live on the shared model modules, so exactly
ONE rollout session is active at a time (single-env by design). To make this
airtight under concurrent clients (a policy over REST *and* the browser over a
WebSocket at the same time), **all** GPU work runs on a single dedicated worker
thread fed by a job queue. Endpoints submit a job and block on its result, so
there is never concurrent CUDA access to the shared caches — no locks, no
use-after-free on reallocated cache buffers.
"""
import base64
import contextlib
import io
import queue
import threading
import time
import uuid

import h5py
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import interpolate

from dreamerv4.models.utils import load_denoiser, load_tokenizer
from dreamerv4.sampling import AutoRegressiveForwardDynamics


class SessionError(RuntimeError):
    """Raised when a request targets a session that is not the active one."""


class EpisodeEnded(SessionError):
    """Raised when stepping a session that has already reached ``max_steps``.

    The rollout advances an *absolute* temporal position each step; running past
    ``max_steps`` would index past the RoPE table (a CUDA device-side assert), so
    the horizon is a hard stop — call ``reset()`` to start a new episode.
    """


class SimEngine:
    def __init__(
        self,
        cfg,
        device,
        *,
        seed_episode,
        seed_pool_size: int = 128,
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

        # Build a pool of seed windows sampled from ACROSS the whole episode, so
        # each reset can start from a different real configuration. The pool is
        # sampled with a fixed RNG so the set of possible starts is stable (and a
        # given `seed` always maps to the same start); the per-reset choice is
        # random. Cached on-GPU (~1-2 GB for the default pool).
        n0 = self.num_init_frames
        pool_rng = np.random.default_rng(0)
        with h5py.File(seed_episode, "r") as f:
            cam, act = f["cam1"], f["actions"]        # cam1: (T,3,H,W) float in [0,1]
            total = int(cam.shape[0])
            k = int(min(seed_pool_size, max(1, total - n0)))
            starts = np.sort(pool_rng.choice(total - n0, size=k, replace=False))
            imgs = np.stack([cam[s:s + n0] for s in starts])          # (K, n0, 3, H0, W0)
            acts = np.stack([act[s:s + n0, : self.n_act] for s in starts])  # (K, n0, n_act)
        imgs = torch.from_numpy(imgs).float()
        K, W = imgs.shape[0], imgs.shape[1]
        imgs = interpolate(imgs.view(K * W, *imgs.shape[2:]), self.resolution)
        self.seed_pool_imgs = imgs.view(K, W, 3, *self.resolution).to(self.device)   # (K,n0,3,H,W)
        self.seed_pool_acts = torch.from_numpy(acts).float().to(self.device)         # (K,n0,n_act)
        self.pool_size = int(K)

        # session state (only ever touched by the worker thread)
        self.session_id = None
        self.world = None
        self.t = 0
        self._last_latent = None                 # (1,1,N,D)
        self._last_image = None                  # (C,H,W) float in [0,1], on device

        # single GPU worker
        self._jobs: "queue.Queue" = queue.Queue()
        self._worker = threading.Thread(target=self._run, name="sim-gpu-worker", daemon=True)
        self._worker.start()

    # ---------------------------------------------------- worker plumbing
    def _run(self):
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        while True:
            fn, args, box = self._jobs.get()
            if fn is None:            # shutdown sentinel
                box["event"].set()
                return
            try:
                box["result"] = fn(*args)
            except BaseException as e:  # noqa: BLE001 — propagate to the caller
                box["error"] = e
            finally:
                box["event"].set()

    def _submit(self, fn, *args):
        box = {"event": threading.Event(), "result": None, "error": None}
        self._jobs.put((fn, args, box))
        box["event"].wait()
        if box["error"] is not None:
            raise box["error"]
        return box["result"]

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------- helpers
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
            "seed_pool_size": self.pool_size,
            "num_denoising_steps": self.denoising_steps,
            "context_length": self.context_len,
            "reward": "none — the pushT dynamics has no reward head; reward is always 0.0",
            "device": str(self.device),
        }

    # ------------------------------------------------- public API (submits)
    def reset(self, seed=None, start_idx: int = 0) -> str:
        return self._submit(self._reset_impl, seed, start_idx)

    def step(self, session_id: str, action) -> dict:
        return self._submit(self._step_impl, session_id, action)

    def close(self, session_id: str) -> bool:
        return self._submit(self._close_impl, session_id)

    def encode_obs(self, obs_type: str = "image", encoding: str = "raw_u8") -> dict:
        return self._submit(self._encode_impl, obs_type, encoding)

    def last_jpeg_bytes(self, quality: int = 85) -> bytes:
        return self._submit(self._jpeg_impl, quality)

    # ------------------------------------------- implementations (worker only)
    def _reset_impl(self, seed, start_idx) -> str:
        # Choose which real seed window to start from:
        #   seed given      -> reproducible config + noise (seed picks the window)
        #   start_idx given -> that specific window (override)
        #   neither         -> a random config each reset
        if seed is not None:
            torch.manual_seed(int(seed))
            idx = int(seed) % self.pool_size
        elif start_idx is not None:
            idx = int(start_idx) % self.pool_size
        else:
            idx = int(torch.randint(self.pool_size, (1,)).item())
        imgs = self.seed_pool_imgs[idx].unsqueeze(0)                    # (1,n0,C,H,W)
        acts = self.seed_pool_acts[idx].unsqueeze(0).to(self.dtype)    # (1,n0,n_act)
        self._start_idx = idx

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
        self._sync()
        return self.session_id

    def _step_impl(self, session_id, action) -> dict:
        if self.world is None or session_id != self.session_id:
            raise SessionError("no active session for this session_id; call reset()")
        if self.t >= self.max_steps:
            raise EpisodeEnded(f"episode reached max_steps={self.max_steps}; call reset()")

        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape[0] < self.n_act:
            a = np.pad(a, (0, self.n_act - a.shape[0]))
        a = np.clip(a[: self.n_act], -1.0, 1.0) * self.action_scale
        at = torch.tensor(a, device=self.device, dtype=self.dtype).view(1, 1, self.n_act)

        t0 = time.perf_counter()
        with torch.no_grad(), self._autocast():
            img = self.world.step(at)                                  # (1,C,H,W)
        self._sync()
        step_ms = (time.perf_counter() - t0) * 1000.0

        self.t += 1
        self._last_latent = self.world.current_z
        self._last_image = img[0].float().clamp(0.0, 1.0)
        return {"t": self.t, "reward": 0.0, "terminated": False,
                "truncated": self.t >= self.max_steps, "step_ms": step_ms}

    def _close_impl(self, session_id) -> bool:
        if session_id == self.session_id:
            self.world = None
            self.session_id = None
            self.t = 0
            return True
        return False

    def _encode_impl(self, obs_type, encoding) -> dict:
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

    def _jpeg_impl(self, quality) -> bytes:
        if self._last_image is None:
            raise SessionError("no observation yet; call reset()")
        img = (self._last_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, "JPEG", quality=quality)
        return buf.getvalue()
