"""Metrics (plan §7.2): detection, pair-wise, risk control, cost."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denominator if denominator else 0.0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall,
            "specificity": specificity, "f1": f1, "mcc": mcc, "balanced_accuracy": (recall + specificity) / 2,
            "accuracy": (tp + tn) / max(len(y), 1), "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "fnr": fn / (fn + tp) if fn + tp else 0.0}


def expected_cost(y: np.ndarray, pred: np.ndarray, cost_ratio: float) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    return float((cost_ratio * ((pred == 0) & (y == 1)) + ((pred == 1) & (y == 0))).mean())


def vd_score(y: np.ndarray, scores: np.ndarray, fpr: float = 0.005) -> float:
    """VD-S (PrimeVul): FNR at the threshold where FPR <= fpr (lowest FNR subject to the FPR constraint)."""
    y, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=float)
    neg = np.sort(scores[y == 0])[::-1]
    pos = scores[y == 1]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allowed = int(math.floor(fpr * len(neg)))          # number of negatives allowed above the threshold
    threshold = neg[allowed] if allowed < len(neg) else -np.inf
    return float((pos <= threshold).mean())


def pair_metrics(frame: pd.DataFrame, pred_col: str = "pred") -> dict:
    """P-C (both correct), P-V (both predicted vulnerable), P-B (both benign), P-R (reversed)."""
    part = frame[frame["pair_id"].notna()]
    if part.empty:
        return {"pairs": 0}
    wide = part.pivot_table(index="pair_id", columns="pair_role", values=pred_col, aggfunc="first").dropna()
    if wide.empty or "vulnerable" not in wide or "patched" not in wide:
        return {"pairs": 0}
    v, p = wide["vulnerable"].astype(int), wide["patched"].astype(int)
    n = len(wide)
    return {"pairs": n, "P-C": float(((v == 1) & (p == 0)).mean()), "P-V": float(((v == 1) & (p == 1)).mean()),
            "P-B": float(((v == 0) & (p == 0)).mean()), "P-R": float(((v == 0) & (p == 1)).mean())}


def skip_risk(y: np.ndarray, skipped: np.ndarray) -> dict:
    y, skipped = np.asarray(y).astype(int), np.asarray(skipped).astype(bool)
    return {"skip_miss_rate": float(skipped[y == 1].mean()) if (y == 1).any() else float("nan"),
            "coverage": float(skipped.mean())}
