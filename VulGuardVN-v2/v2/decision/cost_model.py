"""Cost model c_hat(x) (plan P6 step 4): regression of the LLM-layer cost (tokens in + out + reasoning, USD)
on function length and slice size, fitted on the A-pool logs."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURES = ["log_tokens", "log_lines", "n_candidate_sinks", "has_hyp"]


def _design(frame: pd.DataFrame) -> np.ndarray:
    x = pd.DataFrame({
        "log_tokens": np.log1p(frame["n_code_tokens"].fillna(0).astype(float)),
        "log_lines": np.log1p(frame["n_lines"].fillna(0).astype(float)),
        "n_candidate_sinks": frame["n_candidate_sinks"].fillna(0).astype(float),
        "has_hyp": (frame["has_hyp"] if "has_hyp" in frame else frame.get("has_hyp_neutral", pd.Series(0.0, index=frame.index)))
        .astype(float).fillna(0.0),
    })
    return x[FEATURES].to_numpy()


class CostModel:
    def __init__(self) -> None:
        self.model = LinearRegression()
        self.r2 = None
        self.fallback = 0.0

    def fit(self, frame: pd.DataFrame, target: str = "cost_usd") -> "CostModel":
        y = frame[target].astype(float).to_numpy()
        self.fallback = float(np.mean(y)) if len(y) else 0.0
        if len(frame) >= 10:
            x = _design(frame)
            self.model.fit(x, y)
            self.r2 = float(self.model.score(x, y))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.r2 is None:
            return np.full(len(frame), max(self.fallback, 1e-9))
        floor = max(self.fallback * 0.05, 1e-9)
        return np.clip(self.model.predict(_design(frame)), floor, None)
