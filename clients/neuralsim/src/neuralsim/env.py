"""NeuralSimEnv — Gymnasium-compatible client over the simulator's REST API.

Follows the Gymnasium API surface (``reset() -> (obs, info)``,
``step(a) -> (obs, reward, terminated, truncated, info)``) without *requiring*
gymnasium. If gymnasium is installed the spaces are real ``gymnasium.spaces.Box``;
otherwise a minimal drop-in ``Box`` is used. The only hard deps are
``requests`` and ``numpy``.
"""
import base64

import numpy as np
import requests


class _Box:
    """Minimal stand-in for gymnasium.spaces.Box (used when gymnasium is absent)."""

    def __init__(self, low, high, shape, dtype):
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

    def sample(self):
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return np.random.uniform(lo, hi).astype(self.dtype)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == self.shape and bool((x >= self.low).all() and (x <= self.high).all())

    def __repr__(self):
        return f"Box({self.low.min()}, {self.high.max()}, {self.shape}, {self.dtype})"


def _make_box(low, high, shape, dtype):
    try:
        from gymnasium.spaces import Box
        return Box(low=low, high=high, shape=tuple(shape), dtype=dtype)
    except Exception:
        return _Box(low, high, shape, dtype)


def _decode_obs(obs: dict) -> np.ndarray:
    raw = base64.b64decode(obs["data"])
    enc = obs["encoding"]
    if enc == "raw_u8":
        return np.frombuffer(raw, dtype=np.uint8).reshape(obs["shape"]).copy()
    if enc == "raw_f16":
        return np.frombuffer(raw, dtype=np.float16).reshape(obs["shape"]).copy()
    if enc == "jpeg":
        try:
            import io
            from PIL import Image
        except ImportError as e:
            raise RuntimeError(
                "jpeg-encoded observations need Pillow: `pip install neuralsim[jpeg]`, "
                "or request obs with the default raw encoding."
            ) from e
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    raise ValueError(f"unknown obs encoding: {enc!r}")


class NeuralSimEnv:
    """Gym-style handle to one remote pushT world-model session.

    Parameters
    ----------
    base_url : str        URL of the running server, e.g. "http://192.168.1.4:8000".
    obs_type : str        "image" (H,W,3 uint8) or "latent" (N,D float16).
    encoding : str        image transport: "raw_u8" (numpy-only, default) or "jpeg" (needs Pillow).
    timeout  : float      per-request timeout, seconds.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, base_url="http://localhost:8000", obs_type="image",
                 encoding="raw_u8", timeout=30.0):
        self.base_url = base_url.rstrip("/")
        self.obs_type = obs_type
        self.encoding = "raw_f16" if obs_type == "latent" else encoding
        self.timeout = float(timeout)
        self._http = requests.Session()

        self.info_spec = self._http.get(self.base_url + "/info", timeout=self.timeout).json()
        ad = int(self.info_spec["action_dim"])
        self.action_space = _make_box(-1.0, 1.0, (ad,), np.float32)
        if obs_type == "latent":
            n, d = self.info_spec["latent_shape"]
            self.observation_space = _make_box(-np.inf, np.inf, (n, d), np.float16)
        else:
            h, w, c = self.info_spec["image_shape"]
            self.observation_space = _make_box(0, 255, (h, w, c), np.uint8)

        self.session_id = None
        self._last_obs = None

    # ------------------------------------------------------------------ gym
    def reset(self, *, seed=None, options=None):
        body = {"obs_type": self.obs_type, "encoding": self.encoding}
        if seed is not None:
            body["seed"] = int(seed)
        if options and "start_idx" in options:
            body["start_idx"] = int(options["start_idx"])
        r = self._http.post(self.base_url + "/reset", json=body, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        self.session_id = data["session_id"]
        self._last_obs = _decode_obs(data["obs"])
        return self._last_obs, dict(data.get("info", {}))

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1).tolist()
        body = {"session_id": self.session_id, "action": a,
                "obs_type": self.obs_type, "encoding": self.encoding}
        r = self._http.post(self.base_url + "/step", json=body, timeout=self.timeout)
        if r.status_code == 409:
            raise RuntimeError(
                "The server has no active session for this client — another client may have "
                "reset it (the simulator is single-env). Call reset() again."
            )
        r.raise_for_status()
        data = r.json()
        self._last_obs = _decode_obs(data["obs"])
        return (self._last_obs, float(data["reward"]),
                bool(data["terminated"]), bool(data["truncated"]), dict(data.get("info", {})))

    def render(self):
        """Return the most recent observation (rgb_array style)."""
        return self._last_obs

    def close(self):
        if self.session_id:
            try:
                self._http.post(self.base_url + "/close",
                                json={"session_id": self.session_id}, timeout=self.timeout)
            except Exception:
                pass
            self.session_id = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def make(base_url="http://localhost:8000", **kwargs) -> NeuralSimEnv:
    """Convenience constructor: ``neuralsim.make("http://host:8000")``."""
    return NeuralSimEnv(base_url, **kwargs)
