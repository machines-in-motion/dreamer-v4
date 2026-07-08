# neuralsim — client for the pushT neural simulator

A tiny, **Gymnasium-style** client for a remote DreamerV4 pushT world model.
Talks to the server over plain HTTP. Hard dependencies: **`requests` + `numpy`**.

## Install

```bash
pip install neuralsim                 # from the wheel you were given
# or from source:
pip install ./neuralsim               # the folder containing pyproject.toml
```

Optional extras: `pip install "neuralsim[gym]"` (real `gymnasium` spaces),
`pip install "neuralsim[jpeg]"` (decode JPEG-encoded frames).

## Use

```python
import neuralsim

env = neuralsim.make("http://SERVER_HOST:8000")   # ask your host for the URL
obs, info = env.reset(seed=0)                      # obs: (256, 256, 3) uint8

for t in range(300):
    action = my_policy(obs)                        # action in [-1, 1]^2  ([up/down, left/right])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```

The API matches Gymnasium: `reset() -> (obs, info)` and
`step(a) -> (obs, reward, terminated, truncated, info)`. It works with or
without `gymnasium` installed (spaces fall back to a minimal `Box`).

### Latent observations (bandwidth-light)

```python
env = neuralsim.make("http://SERVER_HOST:8000", obs_type="latent")
obs, _ = env.reset()      # obs: (256, 32) float16 — the model's latent, no image decode
```

## Notes / limits

- **Reward is always `0.0`.** This checkpoint's dynamics has no reward head, so
  `reward`/`terminated` carry no task signal — compute any reward you need
  client-side from the image. `truncated` becomes `True` at the server's
  `max_steps`.
- **Single environment.** The server hosts one live rollout; if another client
  (or the browser page) resets it, your next `step()` raises with a clear
  message — just call `reset()` again.
- **Action space** is `Box(-1, 1, (2,))`; values are scaled server-side to the
  training range. `env.info_spec` has the full spec (`action_scale`, `dt`,
  `max_steps`, shapes, …).
- **Rate** is ~5 Hz (the model is compute-bound); `step()` blocks ~200 ms.
