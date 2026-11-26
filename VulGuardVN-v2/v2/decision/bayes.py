"""Bayes rule for the prefilter branch (plan §6.3): theta* = c_FP / (c_FP + c_FN), d_pf = 1[p_hat >= theta*]."""
from __future__ import annotations

import numpy as np


def theta_star(cost_ratio: float, c_fp: float = 1.0) -> float:
    c_fn = cost_ratio * c_fp
    return c_fp / (c_fp + c_fn)


def d_pf(p_hat: np.ndarray, theta: float) -> np.ndarray:
    return (np.asarray(p_hat) >= theta).astype(int)


def cost(d: np.ndarray, y: np.ndarray, cost_ratio: float, c_fp: float = 1.0) -> np.ndarray:
    d, y = np.asarray(d), np.asarray(y)
    return cost_ratio * c_fp * ((d == 0) & (y == 1)) + c_fp * ((d == 1) & (y == 0))
