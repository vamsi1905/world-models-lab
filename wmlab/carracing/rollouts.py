"""Step 1 of Ha & Schmidhuber (2018): collect rollouts from the real world.

Each rollout is saved as one compressed .npz holding 64x64 frames, actions,
rewards and done flags. A later pass can collect with a trained controller
(`policy=`), which is the paper's iterative-training loop in miniature.
"""
from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np

ZOOM_FRAMES = 50        # CarRacing opens with a camera zoom; skip it


def preprocess(frame: np.ndarray) -> np.ndarray:
    """96x96x3 -> 64x64x3 uint8. Area resize keeps thin track edges visible."""
    from PIL import Image
    return np.asarray(Image.fromarray(frame).resize((64, 64), Image.BILINEAR))


class BrownianPolicy:
    """Uniform random actions make the car twitch in place. A slowly drifting
    random action drives it round corners and off the track, which is the
    variety the VAE and MDN-RNN need to see."""

    def __init__(self, rng: np.random.Generator, dt: float = 0.1):
        self.rng, self.dt = rng, dt
        self.a = np.array([0.0, 0.5, 0.0])

    def __call__(self, _obs=None) -> np.ndarray:
        self.a = self.a + self.rng.normal(0, 1, 3) * np.sqrt(self.dt) * np.array([1.0, 0.5, 0.3])
        self.a = np.clip(self.a, [-1, 0, 0], [1, 1, 1])
        a = self.a.copy()
        a[2] = a[2] if a[2] > 0.6 else 0.0          # brake rarely
        return a.astype(np.float32)


def run_episode(env, policy, max_steps: int, seed: int):
    obs, _ = env.reset(seed=seed)
    for _ in range(ZOOM_FRAMES):
        obs, *_ = env.step(np.zeros(3, dtype=np.float32))
    frames, acts, rews, dones = [], [], [], []
    for _ in range(max_steps):
        f = preprocess(obs)
        a = policy(f)
        obs, r, term, trunc, _ = env.step(a)
        frames.append(f)
        acts.append(a)
        rews.append(r)
        dones.append(term or trunc)
        if term or trunc:
            break
    return (np.stack(frames), np.stack(acts).astype(np.float32),
            np.array(rews, dtype=np.float32), np.array(dones, dtype=bool))


def _collect(args):
    out_dir, start, n, max_steps, seed = args
    import gymnasium as gym
    env = gym.make("CarRacing-v3")
    rng = np.random.default_rng(seed)
    for i in range(start, start + n):
        path = os.path.join(out_dir, f"rollout_{i:05d}.npz")
        if os.path.exists(path):
            continue                               # resumable after a Colab disconnect
        obs, act, rew, done = run_episode(env, BrownianPolicy(rng), max_steps, seed=int(seed * 100_000 + i))
        np.savez_compressed(path, obs=obs, act=act, rew=rew, done=done)
    env.close()


def collect_random(out_dir: str, n_rollouts: int = 300, max_steps: int = 600,
                   workers: int = 2, seed: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    per = (n_rollouts + workers - 1) // workers
    jobs = [(out_dir, w * per, min(per, n_rollouts - w * per), max_steps, seed + w)
            for w in range(workers) if w * per < n_rollouts]
    with Pool(len(jobs)) as p:
        p.map(_collect, jobs)


def load_rollouts(data_dir: str, limit: int | None = None):
    """List of dicts with obs/act/rew/done arrays, sorted by filename."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))[:limit]
    out = []
    for f in files:
        with np.load(os.path.join(data_dir, f)) as d:
            out.append({k: d[k] for k in ("obs", "act", "rew", "done")})
    return out
