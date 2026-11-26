"""Conformal skip gate (plan §6.2, C1).

k = floor(alpha * (n+ + 1));  t_alpha = s_(k) (k-th smallest score of the n+ vulnerable functions in B)
skip x  <=>  s(x) <= t_alpha;  guarantee P(s(X) <= t_alpha | Y = 1) <= k / (n+ + 1) <= alpha (exchangeable, continuous)
"""
from __future__ import annotations

import math

import numpy as np


def threshold(positive_scores_b: np.ndarray, alpha: float) -> tuple[float | None, int]:
    scores = np.sort(np.asarray(positive_scores_b, dtype=float))
    k = int(math.floor(alpha * (len(scores) + 1)))
    if k < 1 or len(scores) == 0:
        return None, 0          # alpha below the smallest achievable level: skip nothing
    return float(scores[k - 1]), k


def alpha_min(n_pos: int) -> float:
    return 1.0 / (n_pos + 1) if n_pos > 0 else float("nan")


def skip_mask(scores: np.ndarray, t_alpha: float | None) -> np.ndarray:
    if t_alpha is None:
        return np.zeros(len(scores), dtype=bool)
    return np.asarray(scores) <= t_alpha
