"""Offline simulation of the v2 cascade (plan P7 step 1, D4): every (alpha, cost ratio, B, router) point is
computed from the cached LLM decisions - no API calls.

  skip   if s(x) <= t_alpha                       -> y = 0
  route  if selected by the router within budget B -> y = d_llm(x, m(x))   (fallback d_pf)
  else                                             -> y = d_pf(x) = 1[p_hat >= theta*]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..decision.budget import (route_learned, route_oracle, route_random, route_two_threshold, route_uncertainty,
                               two_threshold_band)
from .metrics import binary_metrics, expected_cost, pair_metrics, skip_risk

ROUTERS = ("learned", "oracle", "uncertainty", "random", "two_threshold")


def simulate(test: pd.DataFrame, *, t_alpha: float | None, budget: float, router: str, theta: float,
             band: tuple[float, float] | None = None, seed: int = 0, use_gate: bool = True) -> dict:
    s = test["s"].to_numpy()
    skip = (s <= t_alpha) if (use_gate and t_alpha is not None) else np.zeros(len(test), dtype=bool)
    eligible = ~skip & test["llm_available"].to_numpy(dtype=bool)
    n_main = int(test["in_main"].sum())
    n_budget = int(round(budget * n_main))
    if budget <= 0:
        routed = np.zeros(len(test), dtype=bool)
    elif router == "learned":
        routed = route_learned(test["U_hat"].to_numpy(), test["c_hat"].to_numpy(), eligible, n_budget)
    elif router == "oracle":
        routed = route_oracle(test["delta"].to_numpy(), np.clip(test["cost_usd"].to_numpy(), 1e-9, None), eligible, n_budget)
    elif router == "uncertainty":
        routed = route_uncertainty(test["p_hat"].to_numpy(), theta, eligible, n_budget)
    elif router == "random":
        routed = route_random(eligible, n_budget, seed)
    elif router == "two_threshold":
        routed = route_two_threshold(test["p_hat"].to_numpy(), band or (theta, theta), eligible)
    elif router == "all":
        routed = eligible
    else:
        raise ValueError(router)
    pred = np.where(skip, 0, np.where(routed, test["d_llm"].to_numpy(), test["d_pf"].to_numpy())).astype(int)
    return {"pred": pred, "skip": skip, "routed": routed}


def evaluate(test: pd.DataFrame, result: dict, cost_ratio: float) -> dict:
    main = test["in_main"].to_numpy(dtype=bool)
    y = test["label"].to_numpy()
    pred, skip, routed = result["pred"], result["skip"], result["routed"]
    out = binary_metrics(y[main], pred[main])
    out["expected_cost"] = expected_cost(y[main], pred[main], cost_ratio)
    out.update(skip_risk(y[main], skip[main]))
    frame = test[["pair_id", "pair_role"]].assign(pred=pred)
    out.update({k: v for k, v in pair_metrics(frame).items()})
    out["routed_fraction"] = float(routed[main].mean())
    out["llm_functions"] = int(routed.sum())
    out["usd"] = float(test["cost_usd"].to_numpy()[routed].sum())
    out["llm_calls"] = float(test.get("llm_calls", pd.Series(np.zeros(len(test)))).fillna(0).to_numpy()[routed].sum()) \
        if "llm_calls" in test else float(routed.sum() * 3)
    # error by branch (§7.2)
    yb = y[main]
    for name, mask in {"skip": skip[main], "bayes": (~skip & ~routed)[main], "llm": routed[main]}.items():
        out[f"{name}_fn"] = int((mask & (yb == 1) & (pred[main] == 0)).sum())
        out[f"{name}_fp"] = int((mask & (yb == 0) & (pred[main] == 1)).sum())
    return out


def band_for(a_frame: pd.DataFrame, theta: float, budget: float) -> tuple[float, float]:
    return two_threshold_band(a_frame["p_hat"].to_numpy(), theta, budget)
