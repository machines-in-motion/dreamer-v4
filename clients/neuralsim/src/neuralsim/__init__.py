"""neuralsim — a tiny Gym-style client for the pushT neural simulator.

    import neuralsim
    env = neuralsim.make("http://HOST:8000")     # deps: requests + numpy
    obs, info = env.reset(seed=0)
    for _ in range(200):
        action = policy(obs)                       # your code; action in [-1, 1]^2
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()
"""
from .env import NeuralSimEnv, make

__all__ = ["NeuralSimEnv", "make"]
__version__ = "0.1.0"
