"""C: the controller, and the two worlds it can practise in.

C is a single linear layer from [z, h] to three actions: under a thousand
parameters. All the intelligence lives in V and M; C only has to read it.
CMA-ES evolves C by scoring whole populations of candidate weight vectors.
In the dream, the whole population runs as one batched tensor on the GPU,
which is why dream training is so cheap.
"""
from __future__ import annotations

import numpy as np
import torch

from .mdnrnn import sample_mdn
from .rollouts import ZOOM_FRAMES, preprocess
from .vae import Z_DIM, to_tensor

HIDDEN = 256
N_IN = Z_DIM + HIDDEN
N_PARAMS = N_IN * 3 + 3


def act(params: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """params (P, N_PARAMS), z (P, 32), h (P, 256) -> actions (P, 3)."""
    W = params[:, :N_IN * 3].view(-1, 3, N_IN)
    b = params[:, N_IN * 3:]
    raw = torch.tanh(torch.einsum("pai,pi->pa", W, torch.cat([z, h], -1)) + b)
    steer = raw[:, 0]
    gas = (raw[:, 1] + 1) / 2
    brake = raw[:, 2].clamp(min=0)
    return torch.stack([steer, gas, brake], -1)


@torch.no_grad()
def dream_rollout(mdnrnn, params, z0_pool, steps=200, temperature=1.15, repeats=4,
                  use_done=True, return_traj=False):
    """Score each candidate by the reward M *predicts* over `steps` imagined
    frames, averaged over `repeats` dreams from random real starting frames.
    No environment, no pixels: only latents and an LSTM."""
    device = params.device
    P = params.shape[0]
    p = params.repeat_interleave(repeats, 0)
    B = p.shape[0]
    z = z0_pool[torch.randint(len(z0_pool), (B,), device=device)]
    h = torch.zeros(1, B, mdnrnn.hidden, device=device)
    c = torch.zeros_like(h)
    alive = torch.ones(B, device=device)
    total = torch.zeros(B, device=device)
    traj = []
    for _ in range(steps):
        a = act(p, z, h[0])
        log_pi, mu, ls, r_hat, d_hat, _, (h, c) = mdnrnn(z[:, None], a[:, None], (h, c))
        total += alive * r_hat[:, 0]
        if use_done:
            alive = alive * (torch.rand(B, device=device) > torch.sigmoid(d_hat[:, 0])).float()
        z = sample_mdn(log_pi[:, 0], mu[:, 0], ls[:, 0], temperature)
        if return_traj:
            traj.append(z[0].clone())
    score = total.view(P, repeats).mean(1)
    return (score, torch.stack(traj)) if return_traj else score


@torch.no_grad()
def real_rollout(vae, mdnrnn, params, seed=0, max_steps=1000, record=False, device="cuda"):
    """Run one controller in the real CarRacing environment. V sees each real
    frame, M keeps its memory updated, C acts. Returns (reward, frames)."""
    import gymnasium as gym
    env = gym.make("CarRacing-v3")
    obs, _ = env.reset(seed=seed)
    for _ in range(ZOOM_FRAMES):
        obs, *_ = env.step(np.zeros(3, dtype=np.float32))
    p = torch.as_tensor(params, dtype=torch.float32, device=device)[None]
    state = (torch.zeros(1, 1, mdnrnn.hidden, device=device),) * 2
    total, frames = 0.0, []
    for _ in range(max_steps):
        f = preprocess(obs)
        if record:
            frames.append(f)
        z, _ = vae.encode(to_tensor(f[None], device))
        a = act(p, z, state[0][0])
        *_, state = mdnrnn(z[:, None], a[:, None], state)
        obs, r, term, trunc, _ = env.step(a[0].cpu().numpy())
        total += r
        if term or trunc:
            break
    env.close()
    return total, frames


def train_in_dream(vae, mdnrnn, z0_pool, generations=100, popsize=64, sigma0=0.1,
                   dream_steps=200, temperature=1.15, repeats=4, real_every=10,
                   real_steps=None, device="cuda", seed=0):
    """CMA-ES on the controller, scored only in the dream. Every `real_every`
    generations the current mean is also driven for real, purely to *measure*
    the dream-to-reality gap; those scores never reach the optimiser."""
    import cma
    real_steps = real_steps or dream_steps
    es = cma.CMAEvolutionStrategy(np.zeros(N_PARAMS), sigma0,
                                  {"popsize": popsize, "seed": seed + 1, "verbose": -9})
    log = []
    for g in range(generations):
        sols = es.ask()
        P = torch.tensor(np.array(sols), dtype=torch.float32, device=device)
        scores = dream_rollout(mdnrnn, P, z0_pool, dream_steps, temperature, repeats).cpu().numpy()
        es.tell(sols, (-scores).tolist())                  # CMA-ES minimises
        row = dict(gen=g, dream_mean=float(scores.mean()), dream_best=float(scores.max()))
        if real_every and (g % real_every == 0 or g == generations - 1):
            row["real_of_mean"] = real_rollout(vae, mdnrnn, es.mean, seed=10_000 + g,
                                               max_steps=real_steps, device=device)[0]
        log.append(row)
        print({k: round(v, 1) if isinstance(v, float) else v for k, v in row.items()})
    return es, log


@torch.no_grad()
def dream_video_frames(vae, mdnrnn, params, z0, steps=200, temperature=1.0, device="cuda"):
    """Decode one dream back to pixels so it can sit beside the real drive."""
    p = torch.as_tensor(params, dtype=torch.float32, device=device)[None]
    _, traj = dream_rollout(mdnrnn, p, z0[None], steps, temperature, repeats=1,
                            use_done=False, return_traj=True)
    imgs = vae.decode(traj).permute(0, 2, 3, 1).clamp(0, 1).cpu().numpy()
    return (imgs * 255).astype(np.uint8)


def side_by_side_video(dream_frames, real_frames, path, fps=30, scale=4):
    """Write dream | real as one mp4. Short side is padded with its last frame."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    n = max(len(dream_frames), len(real_frames))
    pad = lambda fr: list(fr) + [fr[-1]] * (n - len(fr))
    w = imageio.get_writer(path, fps=fps, macro_block_size=1)
    for d, r in zip(pad(dream_frames), pad(real_frames)):
        pair = np.concatenate([d, np.full((64, 2, 3), 255, np.uint8), r], 1)
        img = Image.fromarray(pair).resize((pair.shape[1] * scale, 64 * scale), Image.NEAREST)
        draw = ImageDraw.Draw(img)
        draw.text((6, 4), "DREAM", fill=(255, 255, 0))
        draw.text((66 * scale + 6, 4), "REAL", fill=(255, 255, 0))
        w.append_data(np.asarray(img))
    w.close()
    return path
