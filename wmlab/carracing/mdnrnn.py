"""M: the memory model. An LSTM whose output is a mixture density network
(MDN): a probability distribution over the next latent z, not a single guess.

Ha & Schmidhuber trained M on CarRacing only to predict z. To train the
controller *inside the dream* we also need the dream to hand out rewards, so
this M carries two extra heads: expected reward and probability of episode end.
(The paper added the same done head for its VizDoom dream experiment.)
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vae import Z_DIM

A_DIM = 3
LOG_2PI = math.log(2 * math.pi)


class MDNRNN(nn.Module):
    def __init__(self, z_dim=Z_DIM, a_dim=A_DIM, hidden=256, n_mix=5):
        super().__init__()
        self.z_dim, self.hidden, self.n_mix = z_dim, hidden, n_mix
        self.lstm = nn.LSTM(z_dim + a_dim, hidden, batch_first=True)
        self.mdn = nn.Linear(hidden, 3 * n_mix * z_dim)       # log_pi, mu, log_sigma per dim
        self.reward = nn.Linear(hidden, 1)
        self.done = nn.Linear(hidden, 1)

    def heads(self, h):
        B, T, _ = h.shape
        out = self.mdn(h).view(B, T, 3, self.z_dim, self.n_mix)
        log_pi = F.log_softmax(out[:, :, 0], dim=-1)
        mu, log_sigma = out[:, :, 1], out[:, :, 2].clamp(-7, 3)
        return log_pi, mu, log_sigma, self.reward(h).squeeze(-1), self.done(h).squeeze(-1)

    def forward(self, z, a, state=None):
        h, state = self.lstm(torch.cat([z, a], -1), state)
        return (*self.heads(h), h, state)


def mdn_nll(z_next, log_pi, mu, log_sigma):
    """Negative log-likelihood of z_next under the per-dimension mixture,
    summed over the 32 dimensions and averaged over time and batch."""
    z = z_next.unsqueeze(-1)
    log_n = -0.5 * ((z - mu) / log_sigma.exp()).pow(2) - log_sigma - 0.5 * LOG_2PI
    return -torch.logsumexp(log_pi + log_n, dim=-1).sum(-1).mean()


def sample_mdn(log_pi, mu, log_sigma, temperature=1.0):
    """Draw one z. Temperature > 1 widens the dream: mixture weights flatten
    and every Gaussian grows. Ha & Schmidhuber used this to make the dream
    harder to cheat; a controller that only works at tau=1.0 has usually
    found a hole in the model, not a way to drive."""
    logits = log_pi / temperature
    k = torch.distributions.Categorical(logits=logits).sample().unsqueeze(-1)
    m = mu.gather(-1, k).squeeze(-1)
    s = log_sigma.gather(-1, k).squeeze(-1).exp() * math.sqrt(temperature)
    return m + s * torch.randn_like(m)


def make_windows(rollouts, seq_len):
    """Cut every rollout into non-overlapping windows of seq_len+1 steps."""
    wins = []
    for ri, r in enumerate(rollouts):
        T = len(r["act"])
        for s in range(0, T - seq_len - 1, seq_len):
            wins.append((ri, s))
    return wins


def train_mdnrnn(rollouts, epochs=20, seq_len=64, batch_size=64, lr=1e-3, device="cuda",
                 reward_w=1.0, done_w=1.0, log_every=100):
    """Rollouts must already carry 'mu' and 'logvar' from the VAE. Each batch
    re-samples z from N(mu, sigma), a cheap form of data augmentation."""
    model = MDNRNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    wins, step, history = make_windows(rollouts, seq_len), 0, []
    for ep in range(epochs):
        np.random.shuffle(wins)
        for i in range(0, len(wins) - batch_size + 1, batch_size):
            b = wins[i:i + batch_size]
            sl = lambda k: np.stack([rollouts[ri][k][s:s + seq_len + 1] for ri, s in b])
            mu, lv = torch.from_numpy(sl("mu")).to(device), torch.from_numpy(sl("logvar")).to(device)
            z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
            a = torch.from_numpy(sl("act")).to(device)
            r = torch.from_numpy(sl("rew")).to(device)
            d = torch.from_numpy(sl("done").astype(np.float32)).to(device)
            log_pi, m, ls, r_hat, d_hat, _, _ = model(z[:, :-1], a[:, :-1])
            nll = mdn_nll(z[:, 1:], log_pi, m, ls)
            r_loss = F.mse_loss(r_hat, r[:, :-1])
            d_loss = F.binary_cross_entropy_with_logits(d_hat, d[:, :-1])
            loss = nll + reward_w * r_loss + done_w * d_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            history.append((loss.item(), nll.item(), r_loss.item(), d_loss.item()))
            if step % log_every == 0:
                print(f"epoch {ep} step {step} nll {nll.item():.2f} reward_mse {r_loss.item():.3f} done_bce {d_loss.item():.4f}")
            step += 1
    return model, history
