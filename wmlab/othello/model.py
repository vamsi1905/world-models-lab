"""A minimal GPT for Othello move sequences, written to be probed.

`forward` can return the residual stream after every block, and accepts an
`edit` callback that rewrites the residual stream at a chosen layer. Those two
hooks are all the interpretability machinery the notebook needs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import VOCAB, MAX_LEN, TOK_TO_SQ, replay


@dataclass
class GPTConfig:
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    ctx: int = MAX_LEN - 1       # predict move t+1 from moves 1..t
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head,
                                          dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class OthelloGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(VOCAB, cfg.d_model)
        self.pos = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB, bias=False)
        mask = torch.triu(torch.ones(cfg.ctx, cfg.ctx, dtype=torch.bool), 1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, idx, return_resid=False, edit=None):
        """idx: (B, T) tokens. `edit(layer, resid) -> resid` is called after
        each block, so an intervention can rewrite the model's 'board'."""
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        resids = [x]
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.mask[:T, :T])
            if edit is not None:
                x = edit(i + 1, x)
            resids.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, resids) if return_resid else logits


def make_batches(games: np.ndarray, batch_size: int, rng: np.random.Generator):
    idx = rng.permutation(len(games))
    for i in range(0, len(idx) - batch_size + 1, batch_size):
        g = torch.from_numpy(games[idx[i:i + batch_size]].astype(np.int64))
        yield g[:, :-1], g[:, 1:]            # inputs, next-move targets


def train(model, games, epochs=2, batch_size=256, lr=5e-4, device="cuda", log_every=200):
    """Plain next-token training. Padding (token 0) is ignored in the loss."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    steps = epochs * (len(games) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.05)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng, step, history = np.random.default_rng(0), 0, []
    for ep in range(epochs):
        for x, y in make_batches(games, batch_size, rng):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, VOCAB).float(), y.reshape(-1), ignore_index=0)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            history.append(loss.item())
            if step % log_every == 0:
                print(f"epoch {ep} step {step}/{steps} loss {loss.item():.3f}")
            step += 1
    return history


@torch.no_grad()
def legal_move_rate(model, games, n=500, device="cuda"):
    """Share of positions where the model's top-1 move is legal. The random-
    play distribution means ~100% is achievable; loss alone hides this."""
    model.eval()
    x = torch.from_numpy(games[:n, :-1].astype(np.int64)).to(device)
    pred = model(x).argmax(-1).cpu().numpy()
    ok = tot = 0
    for g in range(n):
        _, _, legals = replay(games[g])
        for t, legal in enumerate(legals[:-1]):
            if not legal:
                break
            tot += 1
            ok += TOK_TO_SQ.get(int(pred[g, t]), -1) in legal
    return ok / max(tot, 1)
