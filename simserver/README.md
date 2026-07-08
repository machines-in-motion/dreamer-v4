# pushT neural-simulator server

Serves the DreamerV4 pushT world model as a **single-environment** neural
simulator: a Gym-style REST API for policies, plus a WebSocket + browser page
for live human control.

## Launch (on the GPU host)

```bash
conda activate dreamerv4          # env with torch + the deps below
cd /path/to/dreamer-v4
pip install fastapi "uvicorn[standard]" pillow    # one-time, if not present
python scripts/serve.py                            # → 0.0.0.0:8000
```

- Browser control: **http://<host>:8000/**
- REST docs (OpenAPI): **http://<host>:8000/docs**
- Checkpoints come from `scripts/config/dynamics/pushT.yaml` (already wired to
  the local pushT checkpoints). Paths are resolved to absolute, so you can
  launch from anywhere.

Environment-variable knobs (all optional):

| var | default | meaning |
|-----|---------|---------|
| `SERVE_HOST` / `SERVE_PORT` | `0.0.0.0` / `8000` | bind address |
| `DENOISING_STEPS` | `4` | Euler steps/frame (↑ = sharper, slower) |
| `ACTION_SCALE` | `0.1` | maps client action ∈[-1,1] to training range |
| `MAX_STEPS` | `1000` | episode horizon (`truncated` fires here) |
| `SEED_LEN` / `CONTEXT_LEN` | `8` / `32` | context frames / KV window |
| `SEED_EPISODE` | pushT `episode_0.h5` | clip used to seed a reset |

## Sharing it with a friend

- **Tailscale** (recommended): both machines join your tailnet; give him
  `http://<your-tailscale-ip>:8000`. Private, direct, low-latency.
- **cloudflared / ngrok**: `cloudflared tunnel --url http://localhost:8000`
  gives a public HTTPS URL to hand over (add auth if you care who connects).

He then `pip install neuralsim` (see `clients/neuralsim/`) and points it at the URL.

## REST API

| method | path | body | returns |
|--------|------|------|---------|
| GET  | `/health` | – | liveness |
| GET  | `/info`   | – | action/obs spec, `dt`, `max_steps`, … |
| POST | `/reset`  | `{seed?, start_idx?, obs_type, encoding}` | `{session_id, obs, t}` |
| POST | `/step`   | `{session_id, action:[ax,ay], obs_type, encoding}` | `{obs, reward, terminated, truncated, t, info}` |
| POST | `/close`  | `{session_id}` | `{ok}` |
| WS   | `/ws`     | `{action:[..]}` / `{type:"reset"}` | binary JPEG frames + JSON meta |

`obs_type`: `"image"` (H,W,3 uint8) or `"latent"` (N,D float16).
`encoding` (image only): `"raw_u8"` (numpy-only client) or `"jpeg"` (needs Pillow).

**Reward is always 0.0** (the pushT dynamics has no reward head). It's a
world-model sandbox: it imagines plausible consequences of actions, it does not
score them.

## Design (single env)

The KV caches live on the shared model, so exactly one rollout session is
active at a time — whoever `reset`s last owns it. All GPU calls are serialized
under a lock, so REST clients and the browser can't corrupt each other's cache
(they take turns). For concurrent independent envs you'd replicate the model or
add batched inference — deliberately out of scope here.
