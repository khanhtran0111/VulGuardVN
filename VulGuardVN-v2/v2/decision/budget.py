"""Budgeted routing (plan P6 step 5): greedy by U_hat / c_hat until B is used, plus baseline routers.
B is a fraction of all test functions (plan: "theo tỷ lệ hàm"); USD is reported alongside."""
from __future__ import annotations

import numpy as np


def _take(order: np.ndarray, eligible: np.ndarray, n_budget: int) -> np.ndarray:
    chosen = np.zeros(len(eligible), dtype=bool)
    picked = [i for i in order if eligible[i]][:n_budget]
    chosen[picked] = True
    return chosen


def route_learned(u_hat: np.ndarray, c_hat: np.ndarray, eligible: np.ndarray, n_budget: int) -> np.ndarray:
    ratio = u_hat / np.clip(c_hat, 1e-12, None)
    return _take(np.argsort(-ratio, kind="stable"), eligible & (u_hat > 0), n_budget)


def route_oracle(delta: np.ndarray, cost: np.ndarray, eligible: np.ndarray, n_budget: int) -> np.ndarray:
    ratio = delta / np.clip(cost, 1e-12, None)
    return _take(np.argsort(-ratio, kind="stable"), eligible & (delta > 0), n_budget)


def route_uncertainty(p_hat: np.ndarray, theta: float, eligible: np.ndarray, n_budget: int) -> np.ndarray:
    return _take(np.argsort(np.abs(p_hat - theta), kind="stable"), eligible, n_budget)


def route_random(eligible: np.ndarray, n_budget: int, seed: int) -> np.ndarray:
    return _take(np.random.default_rng(seed).permutation(len(eligible)), eligible, n_budget)


def two_threshold_band(p_hat_a: np.ndarray, theta: float, fraction: float) -> tuple[float, float]:
    """A5: fixed band [lo, hi] around theta* whose coverage on A equals the budget fraction."""
    if fraction <= 0:
        return theta, theta
    if fraction >= 1:
        return -np.inf, np.inf
    distance = np.sort(np.abs(p_hat_a - theta))
    radius = distance[min(len(distance) - 1, int(np.ceil(fraction * len(distance))) - 1)]
    return theta - radius, theta + radius


def route_two_threshold(p_hat: np.ndarray, band: tuple[float, float], eligible: np.ndarray) -> np.ndarray:
    return eligible & (p_hat >= band[0]) & (p_hat <= band[1])
