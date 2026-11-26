"""Learned router (plan §6.3, C2): LightGBM regressor of the benefit of sending x to the LLM layer.

Label (renamed from u(x) to avoid the clash with the ensemble variance u(x)):
    delta(x) = cost(d_pf(x), y) - cost(d_llm(x, m(x)), y),   m(x) = refute if p_hat >= theta* else neutral
Features: s, p_sem, p_gra, |p_sem - p_gra|, u, n_tokens, n_cpg_nodes, cyclomatic, n_candidate_sinks, max_sim_to_demo_bank
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

FEATURES = ["s", "p_sem", "p_gra", "sem_gra_gap", "u", "n_tokens", "n_cpg_nodes", "cyclomatic", "n_candidate_sinks",
            "max_sim_to_demo_bank"]


def design(frame: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    x["s"] = frame["s"]
    x["p_sem"] = frame["p_sem"]
    x["p_gra"] = frame["p_gra"]
    x["sem_gra_gap"] = (frame["p_sem"] - frame["p_gra"]).abs()
    x["u"] = frame["u"]
    x["n_tokens"] = frame["n_code_tokens"]
    for name in ["n_cpg_nodes", "cyclomatic", "n_candidate_sinks", "max_sim_to_demo_bank"]:
        x[name] = frame[name] if name in frame else np.nan
    return x[FEATURES].astype(float)


class Router:
    def __init__(self, cfg: dict, seed: int) -> None:
        self.cfg = cfg
        self.seed = seed
        self.model = None
        self.best_iteration = None

    def _params(self) -> dict:
        return {"objective": "regression", "num_leaves": int(self.cfg["num_leaves"]),
                "learning_rate": float(self.cfg["learning_rate"]), "min_child_samples": int(self.cfg["min_child_samples"]),
                "subsample": 0.9, "subsample_freq": 1, "colsample_bytree": 0.9, "seed": self.seed, "verbose": -1,
                "deterministic": True, "force_row_wise": True}

    def cross_validate(self, x: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        """5-fold CV on A: picks n_estimators by early stopping and returns out-of-fold predictions."""
        oof = np.zeros(len(y))
        iterations = []
        folds = KFold(int(self.cfg["cv_folds"]), shuffle=True, random_state=self.seed)
        for train, valid in folds.split(x):
            model = lgb.train(self._params(), lgb.Dataset(x.iloc[train], y[train], weight=w[train]),
                              num_boost_round=int(self.cfg["n_estimators"]),
                              valid_sets=[lgb.Dataset(x.iloc[valid], y[valid], weight=w[valid])],
                              callbacks=[lgb.early_stopping(int(self.cfg["early_stopping_rounds"]), verbose=False)])
            iterations.append(model.best_iteration or int(self.cfg["n_estimators"]))
            oof[valid] = model.predict(x.iloc[valid], num_iteration=model.best_iteration)
        self.best_iteration = max(1, int(np.mean(iterations)))
        return oof

    def fit(self, x: pd.DataFrame, y: np.ndarray, w: np.ndarray) -> "Router":
        if self.best_iteration is None:
            self.cross_validate(x, y, w)
        self.model = lgb.train(self._params(), lgb.Dataset(x, y, weight=w), num_boost_round=self.best_iteration)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        return self.model.predict(x)

    def importance(self) -> dict:
        gains = self.model.feature_importance(importance_type="gain")
        return {name: float(g) for name, g in zip(FEATURES, gains)}
