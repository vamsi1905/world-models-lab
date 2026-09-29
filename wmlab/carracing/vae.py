"""V: the vision model. A convolutional VAE that squeezes a 64x64 frame into a
32-number latent z. Architecture follows Ha & Schmidhuber (2018) exactly."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM = 32


class ConvVAE(nn.Module):
    def __init__(self, z_dim: int = Z_DIM):
        super().__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(                           # 64 -> 31 -> 14 -> 6 -> 2
            nn.Conv2d(3, 32, 4, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2), nn.ReLU(), nn.Flatten())
        self.mu = nn.Linear(1024, z_dim)
        self.logvar = nn.Linear(1024, z_dim)
        self.fc = nn.Linear(z_dim, 1024)
        self.dec = nn.Sequential(                           # 1 -> 5 -> 13 -> 30 -> 64
            nn.ConvTranspose2d(1024, 128, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 5, 2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 6, 2), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 6, 2), nn.Sigmoid())

    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def decode(self, z):
        return self.dec(self.fc(z).view(-1, 1024, 1, 1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()      # reparameterisation trick
        return self.decode(z), mu, logvar


def to_tensor(frames_uint8, device):
    """(N,64,64,3) uint8 numpy -> (N,3,64,64) float in [0,1]."""
    return torch.from_numpy(np.array(frames_uint8, copy=True)).to(device).permute(0, 3, 1, 2).float() / 255.0


def vae_loss(recon, x, mu, logvar, kl_tolerance=0.5):
    """Sum-of-squares reconstruction plus KL, with Ha's 'free bits': the KL
    term stops pushing once it falls below kl_tolerance nats per dimension,
    so the latent is not squeezed into uselessness."""
    rec = (recon - x).pow(2).sum(dim=(1, 2, 3)).mean()
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
    kl = torch.clamp(kl, min=kl_tolerance * mu.shape[1])
    return rec + kl, rec, kl


def train_vae(frames: np.ndarray, epochs=10, batch_size=128, lr=1e-3, device="cuda", log_every=200):
    model = ConvVAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n, step, history = len(frames), 0, []
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            x = to_tensor(frames[np.sort(perm[i:i + batch_size])], device)
            recon, mu, logvar = model(x)
            loss, rec, kl = vae_loss(recon, x, mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            history.append((loss.item(), rec.item(), kl.item()))
            if step % log_every == 0:
                print(f"epoch {ep} step {step} loss {loss.item():.1f} rec {rec.item():.1f} kl {kl.item():.1f}")
            step += 1
    return model, history


@torch.no_grad()
def encode_rollouts(model, rollouts, device="cuda", batch=1000):
    """Adds 'mu' and 'logvar' (T, 32) to each rollout dict, in place."""
    model.eval()
    for r in rollouts:
        mus, lvs = [], []
        for i in range(0, len(r["obs"]), batch):
            mu, lv = model.encode(to_tensor(r["obs"][i:i + batch], device))
            mus.append(mu.cpu())
            lvs.append(lv.cpu())
        r["mu"] = torch.cat(mus).numpy()
        r["logvar"] = torch.cat(lvs).numpy()
    return rollouts
