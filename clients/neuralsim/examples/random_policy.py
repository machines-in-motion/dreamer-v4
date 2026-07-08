"""Minimal example: drive the pushT neural simulator with a scripted policy.

    python random_policy.py --url http://SERVER_HOST:8000 --steps 120

Saves the frames it receives to rollout.gif so you can see what the policy did.
Only needs: neuralsim, numpy (+ imageio for the gif).
"""
import argparse
import math

import numpy as np
import neuralsim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--gif", default="rollout.gif")
    args = ap.parse_args()

    env = neuralsim.make(args.url)
    print("env spec:", env.info_spec)

    obs, info = env.reset(seed=0)
    frames = [obs]
    ret = 0.0
    for t in range(args.steps):
        # a slow circle in action space — replace with your policy(obs)
        action = np.array([math.cos(t / 12.0), math.sin(t / 12.0)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        ret += reward
        frames.append(obs)
        if terminated or truncated:
            print(f"episode ended at t={t + 1} (truncated={truncated})")
            break
    env.close()
    print(f"ran {len(frames) - 1} steps, return={ret:.3f}, last step {info.get('step_ms')} ms")

    try:
        import imageio.v2 as imageio
        imageio.mimsave(args.gif, frames, duration=0.12, loop=0)
        print(f"wrote {args.gif}")
    except Exception as e:
        print(f"(skipped gif: {e})")


if __name__ == "__main__":
    main()
