"""Linear probes and causal interventions for Othello-GPT.

Label convention (Nanda, 2023): each square is EMPTY, MINE or THEIRS relative
to the player about to move. Li et al. found black/white boards needed a
non-linear probe; relative to the mover, a linear probe suffices. That change
of basis is itself one of the more instructive results in the field.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .game import Othello, TOK_TO_SQ, SQ_TO_TOK, replay

EMPTY, MINE, THEIRS = 0, 1, 2
STATE_NAMES = ["empty", "mine", "theirs"]


def relative_board(board: np.ndarray, player: int) -> np.ndarray:
    lab = np.zeros(64, dtype=np.int64)
    flat = board.reshape(-1)
    lab[flat == player] = MINE
    lab[flat == -player] = THEIRS
    return lab


@torch.no_grad()
def collect(model, games: np.ndarray, device="cuda", batch=256):
    """Residual-stream activations at every layer, plus board labels.
    Returns acts: (L+1, N, d) float16 on CPU, labels: (N, 64)."""
    model.eval()
    labels, keep = [], []
    for g in games:
        boards, nexts, legals = replay(g)
        for t in range(min(len(boards), model.cfg.ctx)):
            labels.append(relative_board(boards[t], nexts[t]))
        keep.append(min(len(boards), model.cfg.ctx))
    acts = []
    for i in range(0, len(games), batch):
        x = torch.from_numpy(games[i:i + batch, :-1].astype(np.int64)).to(device)
        _, resids = model(x, return_resid=True)
        r = torch.stack(resids).half().cpu()                 # (L+1, B, T, d)
        for j in range(r.shape[1]):
            acts.append(r[:, j, :keep[i + j]])
    return torch.cat(acts, dim=1), torch.from_numpy(np.stack(labels))


def train_probe(X, Y, epochs=8, lr=1e-3, device="cuda", val_frac=0.1, seed=0):
    """Fit a linear map d -> 64 x 3. Returns (probe, per-square val accuracy)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(X), generator=g)
    n_val = int(len(X) * val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    probe = nn.Linear(X.shape[1], 64 * 3).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-2)
    for _ in range(epochs):
        for i in range(0, len(tr), 4096):
            b = tr[i:i + 4096]
            x, y = X[b].float().to(device), Y[b].to(device)
            loss = F.cross_entropy(probe(x).view(-1, 3), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = torch.cat([probe(X[va[i:i + 8192]].float().to(device)).view(-1, 64, 3).argmax(-1).cpu()
                          for i in range(0, n_val, 8192)])
        acc = (pred == Y[va]).float().mean(0).numpy()
    return probe, acc


# ---------------------------------------------------------------- intervention
def make_edit(probes: dict, pos: int, square: int, new_state: int, layers, margin=2.0):
    """Build an `edit(layer, resid)` hook that pushes the probed state of one
    square toward `new_state` at position `pos`, in each of `layers`.

    The push is along (w_new - w_old) for that square's probe, sized so the
    probe's new-state logit beats the best alternative by `margin`."""
    def edit(layer, resid):
        if layer not in layers:
            return resid
        probe = probes[layer]
        W = probe.weight.view(64, 3, -1)[square]                  # (3, d)
        b = probe.bias.view(64, 3)[square]
        v = resid[:, pos].float()                                  # (B, d)
        logits = v @ W.T + b
        old = logits.clone()
        old[:, new_state] = -1e9
        other = old.argmax(-1)
        direction = W[new_state] - W[other]                        # (B, d)
        gap = (logits.gather(1, other[:, None]) - logits[:, new_state:new_state + 1]).squeeze(1)
        alpha = ((gap + margin) / (direction.pow(2).sum(-1) + 1e-6)).clamp(min=0)
        resid = resid.clone()
        resid[:, pos] = (v + alpha[:, None] * direction).to(resid.dtype)
        return resid
    return edit


def flipped_board_legal(tokens, t: int, square: int):
    """Flip one occupied square's colour after move t; return the new legal
    moves for the side to move, or None if the square was empty."""
    g = Othello()
    for tok in tokens[:t + 1]:
        g.play(TOK_TO_SQ[int(tok)])
    r, c = divmod(square, 8)
    if g.board[r, c] == 0:
        return None
    g.board[r, c] *= -1
    return g.legal_moves()


def topn_error(logits: torch.Tensor, legal: list[int]) -> int:
    """Li et al.'s metric: take the model's top-N moves, N = |legal|, and count
    how many are not in the legal set. 0 = perfect."""
    n = len(legal)
    if n == 0:
        return 0
    top = logits.topk(n).indices.tolist()
    return sum(TOK_TO_SQ.get(t, -1) not in legal for t in top)


@torch.no_grad()
def run_interventions(model, probes, games, layers, n_trials=200, device="cuda", seed=0):
    """Randomly flip one tile mid-game via the activations, and check whether
    the model's predictions track the *edited* board or the original one."""
    rng = np.random.default_rng(seed)
    model.eval()
    rows = []
    tries = 0
    while len(rows) < n_trials and tries < n_trials * 20:
        tries += 1
        g = games[rng.integers(len(games))]
        boards, nexts, legals = replay(g)
        t = int(rng.integers(5, min(len(boards), model.cfg.ctx) - 1))
        lab = relative_board(boards[t], nexts[t])
        occupied = np.flatnonzero(lab != EMPTY)
        sq = int(rng.choice(occupied))
        new_legal = flipped_board_legal(g, t, sq)
        if new_legal is None or sorted(new_legal) == sorted(legals[t]) or not new_legal:
            continue                                   # only informative flips
        new_state = MINE if lab[sq] == THEIRS else THEIRS
        x = torch.from_numpy(g[None, :-1].astype(np.int64)).to(device)
        base = model(x)[0, t].float().cpu()
        edited = model(x, edit=make_edit(probes, t, sq, new_state, layers))[0, t].float().cpu()
        rows.append(dict(t=t, square=sq,
                         base_vs_old=topn_error(base, legals[t]),
                         base_vs_new=topn_error(base, new_legal),
                         edit_vs_new=topn_error(edited, new_legal),
                         edit_vs_old=topn_error(edited, legals[t])))
    return rows


# ----------------------------------------------------------------- plotting
def plot_board(ax, labels, title="", mark=None, mark_color="tab:red"):
    """Draw a 64-square relative board (EMPTY/MINE/THEIRS). `mark` is a list
    of squares to ring, e.g. legal moves or the model's top predictions."""
    import matplotlib.patches as mpatches
    ax.set_facecolor("#2e7d32")
    for s, v in enumerate(np.asarray(labels).reshape(-1)):
        r, c = divmod(s, 8)
        if v != EMPTY:
            ax.add_patch(mpatches.Circle((c, r), 0.4, color="black" if v == MINE else "white", ec="k"))
    for s in (mark or []):
        r, c = divmod(s, 8)
        ax.add_patch(mpatches.Rectangle((c - .45, r - .45), .9, .9, fill=False, ec=mark_color, lw=2))
    ax.set_xlim(-.5, 7.5)
    ax.set_ylim(7.5, -.5)
    ax.set_xticks(range(8), list("12345678"))
    ax.set_yticks(range(8), list("ABCDEFGH"))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
