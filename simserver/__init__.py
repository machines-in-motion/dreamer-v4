"""pushT neural-simulator server (single-environment).

Wraps the DreamerV4 pushT world model behind a small FastAPI service:
  * REST  (/reset, /step, /close, /info, /health) — for the Gym-style client.
  * WebSocket (/ws) + a static page (/) — for live in-browser control.
"""
from .engine import SimEngine, SessionError, EpisodeEnded
from .app import create_app

__all__ = ["SimEngine", "SessionError", "EpisodeEnded", "create_app"]
