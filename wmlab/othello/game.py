"""A small, dependency-free Othello engine plus a synthetic game generator.

The Othello-GPT experiment (Li et al., 2023) trains a transformer on move
sequences alone. The model never sees a board. If a linear probe can later read
the board out of its activations, the model built that board for itself.

Board encoding: 8x8 numpy array, 0 = empty, 1 = black, -1 = white.
Black moves first. Square index = row * 8 + col (0..63).
Token vocabulary: the 60 non-centre squares map to tokens 1..60; 0 is padding.
"""
from __future__ import annotations

import numpy as np
from multiprocessing import Pool

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CENTRE = {27, 28, 35, 36}
SQUARES = [s for s in range(64) if s not in CENTRE]          # 60 playable squares
SQ_TO_TOK = {s: i + 1 for i, s in enumerate(SQUARES)}        # 1..60
TOK_TO_SQ = {t: s for s, t in SQ_TO_TOK.items()}
VOCAB = 61                                                     # pad + 60 squares
MAX_LEN = 60


def sq_name(s: int) -> str:
    return "ABCDEFGH"[s // 8] + str(s % 8 + 1)


class Othello:
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=np.int8)
        self.board[3, 3] = self.board[4, 4] = -1
        self.board[3, 4] = self.board[4, 3] = 1
        self.player = 1  # player to move

    def copy(self) -> "Othello":
        g = Othello.__new__(Othello)
        g.board = self.board.copy()
        g.player = self.player
        return g

    def _flips(self, r: int, c: int, player: int) -> list[tuple[int, int]]:
        if self.board[r, c] != 0:
            return []
        out = []
        b = self.board
        for dr, dc in DIRS:
            rr, cc, line = r + dr, c + dc, []
            while 0 <= rr < 8 and 0 <= cc < 8 and b[rr, cc] == -player:
                line.append((rr, cc))
                rr += dr
                cc += dc
            if line and 0 <= rr < 8 and 0 <= cc < 8 and b[rr, cc] == player:
                out.extend(line)
        return out

    def legal_moves(self, player: int | None = None) -> list[int]:
        p = self.player if player is None else player
        return [r * 8 + c for r in range(8) for c in range(8)
                if self.board[r, c] == 0 and self._flips(r, c, p)]

    def play(self, sq: int) -> None:
        """Play a move for the side to move, then hand over the turn,
        passing automatically if the opponent has no legal reply."""
        r, c = divmod(sq, 8)
        flips = self._flips(r, c, self.player)
        if not flips:
            raise ValueError(f"illegal move {sq_name(sq)}")
        self.board[r, c] = self.player
        for rr, cc in flips:
            self.board[rr, cc] = self.player
        self.player = -self.player
        if not self.legal_moves():          # opponent must pass
            self.player = -self.player      # (if we also have none, game over)

    def game_over(self) -> bool:
        return not self.legal_moves()


# --- Fast bitboard move generator, used only for synthetic data. ----------
# Pure-Python ints as 64-bit boards: roughly 20x faster than the array engine.
_FULL = (1 << 64) - 1
_NOT_C0 = sum(1 << (r * 8 + c) for r in range(8) for c in range(8) if c != 0)
_NOT_C7 = sum(1 << (r * 8 + c) for r in range(8) for c in range(8) if c != 7)
_SHIFTS = [
    lambda x: (x << 1) & _NOT_C0,           # east
    lambda x: (x >> 1) & _NOT_C7,           # west
    lambda x: (x << 8) & _FULL,             # south
    lambda x: x >> 8,                       # north
    lambda x: (x << 9) & _NOT_C0 & _FULL,   # south-east
    lambda x: (x << 7) & _NOT_C7 & _FULL,   # south-west
    lambda x: (x >> 7) & _NOT_C0,           # north-east
    lambda x: (x >> 9) & _NOT_C7,           # north-west
]


def _bb_legal(p: int, o: int) -> int:
    empty, moves = ~(p | o) & _FULL, 0
    for sh in _SHIFTS:
        t = sh(p) & o
        for _ in range(5):
            t |= sh(t) & o
        moves |= sh(t) & empty
    return moves


def _bb_flips(p: int, o: int, m: int) -> int:
    flips = 0
    for sh in _SHIFTS:
        line, x = 0, sh(m)
        while x & o:
            line |= x
            x = sh(x)
        if x & p:
            flips |= line
    return flips


def random_game(rng: np.random.Generator) -> list[int]:
    """One uniformly random legal game, as a list of square indices."""
    black = (1 << 28) | (1 << 35)          # D5, E4
    white = (1 << 27) | (1 << 36)          # D4, E5
    p, o, moves = black, white, []
    while True:
        legal = _bb_legal(p, o)
        if not legal:
            p, o = o, p                     # pass
            legal = _bb_legal(p, o)
            if not legal:
                return moves
        sqs = [i for i in range(64) if legal >> i & 1]
        s = sqs[rng.integers(len(sqs))]
        m = 1 << s
        f = _bb_flips(p, o, m)
        p |= m | f
        o &= ~f
        moves.append(s)
        p, o = o, p


def _worker(args):
    seed, n = args
    rng = np.random.default_rng(seed)
    out = np.zeros((n, MAX_LEN), dtype=np.int8)
    for i in range(n):
        toks = [SQ_TO_TOK[m] for m in random_game(rng)]
        out[i, :len(toks)] = toks
    return out


def generate_games(n_games: int, seed: int = 0, workers: int = 2) -> np.ndarray:
    """(n_games, 60) int8 array of tokens, zero-padded for short games."""
    chunk = 2000
    jobs = [(seed * 1_000_003 + i, min(chunk, n_games - i * chunk))
            for i in range((n_games + chunk - 1) // chunk)]
    with Pool(workers) as p:
        parts = p.map(_worker, jobs)
    return np.concatenate(parts)


def replay(tokens) -> tuple[list[np.ndarray], list[int], list[list[int]]]:
    """Replay a token sequence. Returns, for each move t (after it is played):
    the board, the player to move next, and that player's legal squares."""
    g = Othello()
    boards, nexts, legals = [], [], []
    for t in tokens:
        t = int(t)
        if t == 0:
            break
        g.play(TOK_TO_SQ[t])
        boards.append(g.board.copy())
        nexts.append(g.player)
        legals.append(g.legal_moves())
    return boards, nexts, legals
