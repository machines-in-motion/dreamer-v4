"""Launch the pushT neural-simulator server.

    python scripts/serve.py                      # 0.0.0.0:8000, pushT config
    SERVE_PORT=9000 python scripts/serve.py      # pick a port
    python scripts/serve.py denoiser.n_actions=2 # any Hydra override

Checkpoint paths from the config are resolved to absolute, so it runs from any
working directory. Browser UI: http://<host>:<port>/ · REST docs: /docs
"""
import os
import sys
from pathlib import Path

import hydra
import torch
import uvicorn
from omegaconf import DictConfig, open_dict

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from simserver import SimEngine, create_app  # noqa: E402

SEED_EPISODE = os.environ.get(
    "SEED_EPISODE", "/home/mim-server/datasets/pushT/224/episode_0.h5")


@hydra.main(config_path="config", config_name="dynamics/pushT", version_base=None)
def main(cfg: DictConfig):
    # Make checkpoint paths cwd-independent.
    with open_dict(cfg):
        for key in ("tokenizer_ckpt", "dynamics_ckpt"):
            p = cfg.get(key)
            if p and not os.path.isabs(p):
                cfg[key] = str(_REPO / p)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[serve] loading pushT world model on {device} ...")
    engine = SimEngine(
        cfg, device,
        seed_episode=SEED_EPISODE,
        seed_pool_size=int(os.environ.get("SEED_POOL_SIZE", 128)),
        num_init_frames=int(os.environ.get("SEED_LEN", 8)),
        context_len=int(os.environ.get("CONTEXT_LEN", 32)),
        denoising_steps=int(os.environ.get("DENOISING_STEPS", 4)),
        action_scale=float(os.environ.get("ACTION_SCALE", 0.1)),
        max_steps=int(os.environ.get("MAX_STEPS", 1000)),
    )
    app = create_app(engine)

    host = os.environ.get("SERVE_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVE_PORT", 8000))
    print(f"[serve] ready → browser  http://{host}:{port}/   |   REST  http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
