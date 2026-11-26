"""Weighted calibrators for p_hat (beta, temperature, isotonic; plan P3 step 4). Fitted on A with the
importance weights that undo the oversampling of positives (§3.2). Ported/extended from v1 metrics.py."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def weighted_nll(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    p = _clip(p)
    w = np.ones_like(p) if w is None else w
    return float(-np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / np.sum(w))


def fit_beta(scores: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    p = _clip(_sigmoid(scores))
    if len(np.unique(y)) < 2:
        return {"a": 1.0, "b": 0.0, "c": 0.0}
    features = np.column_stack([np.log(p), -np.log1p(-p)])
    model = LogisticRegression(C=1e4, max_iter=5000).fit(features, y, sample_weight=w)
    return {"a": float(model.coef_[0][0]), "b": float(model.coef_[0][1]), "c": float(model.intercept_[0])}


def apply_beta(scores: np.ndarray, params: dict) -> np.ndarray:
    p = _clip(_sigmoid(scores))
    return _sigmoid(params["a"] * np.log(p) - params["b"] * np.log1p(-p) + params["c"])


def fit_temp(scores: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Temperature + bias on the logit (bias is needed when A has a different class balance)."""
    if len(np.unique(y)) < 2:
        return {"t": 1.0, "b": 0.0}
    best = (np.inf, 1.0, 0.0)
    for t in np.linspace(0.25, 8.0, 63):
        for b in np.linspace(-8, 4, 97):
            nll = weighted_nll(y, _sigmoid(scores / t + b), w)
            if nll < best[0]:
                best = (nll, float(t), float(b))
    return {"t": best[1], "b": best[2]}


def apply_temp(scores: np.ndarray, params: dict) -> np.ndarray:
    return _sigmoid(np.asarray(scores) / params["t"] + params["b"])


def fit_iso(scores: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, y, sample_weight=w)
    return {"x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist()}


def apply_iso(scores: np.ndarray, params: dict) -> np.ndarray:
    return np.interp(scores, params["x"], params["y"])


FITTERS = {"beta": (fit_beta, apply_beta), "temp": (fit_temp, apply_temp), "iso": (fit_iso, apply_iso)}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None, bins: int = 15) -> float:
    w = np.ones_like(p) if w is None else w
    edges = np.linspace(0, 1, bins + 1)
    total, err = np.sum(w), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            ww = w[mask]
            err += abs(np.average(y[mask], weights=ww) - np.average(p[mask], weights=ww)) * ww.sum() / total
    return float(err)


def brier(y: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones_like(p) if w is None else w
    return float(np.sum(w * (p - y) ** 2) / np.sum(w))
