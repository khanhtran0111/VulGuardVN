"""Stacked fusion of the fine-tuned view (2026-11-26, prefilter.yaml `model.fusion: stacked`).

    s(x) = c + a * s_base(x) + b * ft(x)

s_base = ensemble-mean fusion logit of the multi-view network trained WITHOUT the fine-tuned view, ft = the raw
cross-fitted fine-tuned UniXcoder logit. (a, b, c) is a logistic regression fitted on train_dev, which is neither A
(calibrators, router), B (conformal gate, operating point) nor test; ft is out-of-fold there.
Why: trained jointly, the network gave the fine-tuned view a weight of 0.73 (it looks stronger on train, out-of-fold AUC
0.728, than on test, 0.717) while the best linear weight is ~0.45-0.5. Devign: test AUC 0.719 (joint) -> 0.730 (stacked,
weights fitted on B in the diagnostic), +0.008 on A.
`fusion: joint` keeps the 2026-11-23 behaviour (fine-tuned logit as an input of the network; ablation A14).
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def stacked(cfg: dict) -> bool:
    model = cfg["model"]
    return model.get("fusion", "joint") == "stacked" and "finetune" in model.get("views", [])


def network_views(cfg: dict, drop: str | None = None) -> tuple[str, ...]:
    """Views of the multi-view network: without `finetune` when stacked, and without `drop` (A1)."""
    views = [v for v in cfg["model"]["views"] if v != drop]
    if stacked(cfg):
        views = [v for v in views if v != "finetune"]
    return tuple(views)


def main_tag(rep: int | str, cfg: dict) -> str:
    """Model directory of main rep r: base-r{r} (stacked: network without the fine-tuned view) or main-r{r} (joint)."""
    return f"base-r{rep}" if stacked(cfg) else f"main-r{rep}"


def main_prefix(cfg: dict) -> str:
    return "base-r" if stacked(cfg) else "main-r"


def fit(s_base: np.ndarray, ft: np.ndarray, y: np.ndarray) -> dict:
    model = LogisticRegression(C=1e4, max_iter=5000).fit(np.column_stack([s_base, ft]), y)
    return {"a": float(model.coef_[0][0]), "b": float(model.coef_[0][1]), "c": float(model.intercept_[0]),
            "fitted_on": "train_dev", "n": int(len(y))}


def apply(s_base: np.ndarray, ft: np.ndarray, params: dict) -> np.ndarray:
    return params["c"] + params["a"] * np.asarray(s_base, dtype=float) + params["b"] * np.asarray(ft, dtype=float)
