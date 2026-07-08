"""FastAPI app: REST for the Gym client, WebSocket + static page for the browser."""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .engine import SimEngine, SessionError

_STATIC = Path(__file__).parent / "static"


class ResetReq(BaseModel):
    seed: Optional[int] = None
    start_idx: int = 0
    obs_type: str = "image"      # "image" | "latent"
    encoding: str = "raw_u8"     # "raw_u8" | "jpeg"  (image); latents are "raw_f16"


class StepReq(BaseModel):
    session_id: str
    action: List[float]
    obs_type: str = "image"
    encoding: str = "raw_u8"


class CloseReq(BaseModel):
    session_id: str


def create_app(engine: SimEngine) -> FastAPI:
    app = FastAPI(title="pushT Neural Simulator", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "env": "pushT-neural-sim",
                "active_session": engine.session_id is not None}

    @app.get("/info")
    def info():
        return engine.info()

    @app.post("/reset")
    async def reset(req: ResetReq):
        sid = await run_in_threadpool(engine.reset, req.seed, req.start_idx)
        obs = engine.encode_obs(req.obs_type, req.encoding)
        return {"session_id": sid, "obs": obs, "t": 0,
                "info": {"max_steps": engine.max_steps}}

    @app.post("/step")
    async def step(req: StepReq):
        try:
            res = await run_in_threadpool(engine.step, req.session_id, req.action)
        except SessionError as e:
            raise HTTPException(status_code=409, detail=str(e))
        obs = engine.encode_obs(req.obs_type, req.encoding)
        return {"obs": obs, "reward": res["reward"], "terminated": res["terminated"],
                "truncated": res["truncated"], "t": res["t"],
                "info": {"step_ms": round(res["step_ms"], 1)}}

    @app.post("/close")
    async def close(req: CloseReq):
        return {"ok": await run_in_threadpool(engine.close, req.session_id)}

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = _STATIC / "index.html"
        if not page.exists():
            return HTMLResponse("<h1>pushT neural simulator</h1><p>Browser page not found.</p>")
        return HTMLResponse(page.read_text())

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        # A browser connection owns the single env: it resets on connect and on
        # request, and drives one step per message (request/response paced).
        await sock.accept()
        sid = await run_in_threadpool(engine.reset, None, 0)
        await sock.send_bytes(await run_in_threadpool(engine.last_jpeg_bytes))
        try:
            while True:
                msg = await sock.receive_json()
                if msg.get("type") == "reset":
                    sid = await run_in_threadpool(engine.reset, msg.get("seed"), int(msg.get("start_idx", 0)))
                    await sock.send_json({"t": 0})
                    await sock.send_bytes(await run_in_threadpool(engine.last_jpeg_bytes))
                    continue
                action = msg.get("action", [0.0] * engine.n_act)
                try:
                    res = await run_in_threadpool(engine.step, sid, action)
                except SessionError:
                    sid = await run_in_threadpool(engine.reset, None, 0)
                    res = {"t": 0, "step_ms": 0.0}
                await sock.send_json({"t": res["t"], "step_ms": round(res.get("step_ms", 0.0), 1)})
                await sock.send_bytes(await run_in_threadpool(engine.last_jpeg_bytes))
        except WebSocketDisconnect:
            pass

    return app
