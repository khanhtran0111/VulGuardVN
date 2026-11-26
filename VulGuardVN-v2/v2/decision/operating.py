"""Operating points of the prefilter branch (2026-11-25).

An operating point fixes the threshold theta of d_pf = 1[p_hat >= theta] and the cost ratio r = c_FN / c_FP used as the
router target (delta = cost(d_pf) - cost(d_llm)) and for the LLM mode choice (refute iff p_hat >= theta):
  cr{r}   Bayes rule of plan §6.3: theta = 1 / (1 + r)                                       (fixed, a priori)
  f1B     theta chosen on B (calibration split, never test) to maximise the objective (F1) of the prefilter decision;
          its implied cost ratio r = (1 - theta) / theta keeps the router and the LLM mode Bayes-consistent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .bayes import theta_star


def fixed_op(ratio: float) -> str:
    return f"cr{float(ratio):g}"


def _objective(y: np.ndarray, pred: np.ndarray, name: str) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())
    if name == "f1":
        return 2 * tp / max(2 * tp + fp + fn, 1.0)
    if name == "mcc":
        denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0
    raise ValueError(f"unknown operating-point objective: {name}")


def theta_on_b(p_hat: np.ndarray, y: np.ndarray, objective: str, grid: list[float]) -> tuple[float, float]:
    """Threshold on the grid [lo, hi, step] that maximises `objective` on B (first maximum on ties)."""
    lo, hi, step = (float(v) for v in grid)
    thetas = np.round(np.arange(lo, hi + step / 2, step), 6)
    values = [_objective(y, (p_hat >= t).astype(int), objective) for t in thetas]
    best = int(np.argmax(values))
    return float(thetas[best]), float(values[best])


def resolve(frame_b: pd.DataFrame, dcfg: dict) -> dict[str, dict]:
    """All operating points for one prefilter rep. frame_b = B rows (in_main) with p_hat and label."""
    ops = {fixed_op(r): {"cost_ratio": float(r), "theta": theta_star(float(r)), "kind": "bayes"} for r in dcfg["cost_ratios"]}
    for name, spec in (dcfg.get("operating_points") or {}).items():
        theta, value = theta_on_b(frame_b["p_hat"].to_numpy(), frame_b["label"].to_numpy(), spec["objective"], spec["grid"])
        ops[name] = {"cost_ratio": (1 - theta) / theta, "theta": theta, "kind": f"max {spec['objective']} on B",
                     "objective_on_b": value, "n_b": int(len(frame_b))}
    return ops
