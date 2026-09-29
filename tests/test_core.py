"""Fast checks: run with `pytest -q` (about 20 seconds on a laptop)."""
import numpy as np
import torch

from wmlab.othello.game import Othello, generate_games, random_game, replay, _bb_legal
from wmlab.othello.model import OthelloGPT, GPTConfig
from wmlab.othello.probe import make_edit, train_probe
from wmlab.carracing.vae import ConvVAE
from wmlab.carracing.mdnrnn import MDNRNN, mdn_nll, sample_mdn
from wmlab.carracing.controller import act, dream_rollout, N_PARAMS


def test_opening_moves():
    assert sorted(Othello().legal_moves()) == [19, 26, 37, 44]


def test_bitboard_matches_array_engine():
    rng = np.random.default_rng(0)
    for _ in range(50):
        moves = random_game(rng)
        g = Othello()
        for m in moves:
            p = sum(1 << i for i in range(64) if g.board.flat[i] == g.player)
            o = sum(1 << i for i in range(64) if g.board.flat[i] == -g.player)
            bb = [i for i in range(64) if _bb_legal(p, o) >> i & 1]
            assert sorted(bb) == sorted(g.legal_moves())
            g.play(m)                      # raises if the generator made an illegal move
        assert g.game_over()


def test_replay_lengths():
    games = generate_games(20, workers=1)
    for g in games:
        boards, nexts, legals = replay(g)
        assert len(boards) == (g > 0).sum()


def test_gpt_shapes_and_edit():
    m = OthelloGPT(GPTConfig(n_layer=2, n_head=2, d_model=32))
    x = torch.randint(1, 61, (3, 59))
    logits, resids = m(x, return_resid=True)
    assert logits.shape == (3, 59, 61) and len(resids) == 3
    X = resids[1].reshape(-1, 32).detach()
    probe, _ = train_probe(X, torch.randint(0, 3, (len(X), 64)), epochs=1, device="cpu")
    edited = m(x, edit=make_edit({1: probe}, pos=10, square=5, new_state=1, layers={1}))
    assert torch.allclose(edited[:, :10], logits[:, :10], atol=1e-5)   # causal: past untouched
    assert not torch.allclose(edited[:, 10], logits[:, 10])


def test_vae_shapes():
    v = ConvVAE()
    recon, mu, lv = v(torch.rand(2, 3, 64, 64))
    assert recon.shape == (2, 3, 64, 64) and mu.shape == (2, 32)


def test_mdn_and_dream():
    m = MDNRNN()
    z, a = torch.randn(2, 5, 32), torch.rand(2, 5, 3)
    log_pi, mu, ls, r, d, h, _ = m(z, a)
    assert mdn_nll(z, log_pi, mu, ls).isfinite()
    assert sample_mdn(log_pi[:, 0], mu[:, 0], ls[:, 0]).shape == (2, 32)
    P = torch.randn(4, N_PARAMS) * 0.1
    a = act(P, torch.randn(4, 32), torch.randn(4, 256))
    assert a.shape == (4, 3) and (a[:, 1] >= 0).all() and (a[:, 2] >= 0).all()
    assert dream_rollout(m, P, torch.randn(10, 32), steps=5, repeats=2).shape == (4,)
