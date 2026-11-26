"""Joins prefilter scores, LLM-layer decisions (P5), Joern stats and retrieval similarity into one table per
(dataset, prefilter rep) for the A-pool and test functions. Used by P6 and P7."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import dataset_dir
from ..data.io import load_records
from ..scripts.run_p5_verify import decisions_path
from .bayes import d_pf

DECISION_COLS = ["d", "fallback", "d_raw", "usd", "tokens_in", "tokens_out", "reasoning_tokens", "llm_calls", "has_hyp", "reason"]


def load_scores(dataset: str) -> pd.DataFrame:
    path = dataset_dir(dataset, "prefilter") / "prefilter_scores.parquet"
    if not path.exists():
        raise SystemExit(f"[{dataset}] prefilter scores missing - run run_p3_prefilter --stage score")
    return pd.read_parquet(path)


def load_decisions(dataset: str) -> pd.DataFrame:
    path = decisions_path(dataset)
    if not path.exists():
        raise SystemExit(f"[{dataset}] {path.name} missing - run run_p5_verify")
    return pd.read_parquet(path)


def pivot_decisions(decisions: pd.DataFrame, variant: str) -> pd.DataFrame:
    part = decisions[decisions["variant"] == variant]
    if part.empty:
        return pd.DataFrame(columns=["id"])
    wide = part.pivot_table(index="id", columns="mode", values=[c for c in DECISION_COLS if c in part], aggfunc="first")
    wide.columns = [f"{value}_{mode}" for value, mode in wide.columns]
    joern = part.groupby("id")[["n_cpg_nodes", "cyclomatic", "n_candidate_sinks"]].first()
    return wide.join(joern).reset_index()


class DecisionFrame:
    """All inputs of the decision layer for one dataset; `for_rep` returns the per-rep table."""

    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        records = load_records(dataset, ["id", "label", "subset", "weight", "in_main", "pair_id", "pair_role"])
        self.records = records[records["subset"].isin(["a_pool", "b", "test"])]
        self.scores = load_scores(dataset)
        stats = pd.read_parquet(dataset_dir(dataset, "features") / "ids.parquet")
        self.base = self.records.merge(stats, on="id", how="left")
        sim_path = dataset_dir(dataset, "retrieval") / "max_sim.parquet"
        if sim_path.exists():
            self.base = self.base.merge(pd.read_parquet(sim_path), on="id", how="left")
        else:
            self.base["max_sim_to_demo_bank"] = np.nan
        self.decisions = load_decisions(dataset)
        self.variants = {v: pivot_decisions(self.decisions, v) for v in self.decisions["variant"].unique()}

    def reps(self, prefix: str | None = None) -> list[str]:
        reps = sorted(self.scores["rep"].unique(), key=str)
        if prefix is None:
            return [r for r in reps if str(r).isdigit()]
        return [r for r in reps if str(r).startswith(prefix)]

    def for_rep(self, rep: str, calibrator: str = "beta", variant: str = "main") -> pd.DataFrame:
        scores = self.scores[self.scores["rep"] == rep][["id", "s", "u", "p_sem", "p_gra", f"p_hat_{calibrator}"]]
        frame = self.base.merge(scores.rename(columns={f"p_hat_{calibrator}": "p_hat"}), on="id", how="inner")
        llm = self.variants.get(variant)
        if llm is not None and not llm.empty:
            frame = frame.merge(llm, on="id", how="left")
        for mode in ("neutral", "refute"):
            for col in ("d", "fallback", "d_raw", "usd", "tokens_in", "tokens_out", "reasoning_tokens", "llm_calls", "has_hyp"):
                if f"{col}_{mode}" not in frame:
                    frame[f"{col}_{mode}"] = np.nan
        for col in ("n_cpg_nodes", "cyclomatic", "n_candidate_sinks"):
            if col not in frame:
                frame[col] = np.nan
        frame["has_llm"] = frame["d_neutral"].notna() | frame["d_refute"].notna() | frame["fallback_neutral"].notna()
        return frame.reset_index(drop=True)


def llm_outcome(frame: pd.DataFrame, theta: float, *, no_refute: bool = False, no_cpg: bool = False,
                refute_no_hyp: int | None = None) -> dict:
    """d_llm(x, m(x)) with the mode chosen by the Bayes rule; fallback (Joern error, invalid output, missing call)
    -> d_pf(x). Returns arrays: mode_refute, d_pf, d_llm, cost_usd, tokens, fallback, llm_available."""
    p_hat = frame["p_hat"].to_numpy()
    pf = d_pf(p_hat, theta)
    refute = (p_hat >= theta) & (not no_refute)
    def pick(col):
        return np.where(refute, frame[f"{col}_refute"].to_numpy(dtype=float), frame[f"{col}_neutral"].to_numpy(dtype=float))
    d = pick("d_raw" if no_cpg else "d")
    if refute_no_hyp is not None and "reason_refute" in frame:    # A12: override the refute / no-hypothesis rule
        d = np.where(refute & (frame["reason_refute"].to_numpy() == "no_hypothesis"), float(refute_no_hyp), d)
    fallback = np.isnan(d) | (pick("fallback") == 1) if not no_cpg else np.isnan(d)
    d_llm = np.where(fallback, pf, np.nan_to_num(d)).astype(int)
    usd = np.nan_to_num(pick("usd"))
    tokens = np.nan_to_num(pick("tokens_in")) + np.nan_to_num(pick("tokens_out"))
    return {"mode_refute": refute, "d_pf": pf, "d_llm": d_llm, "cost_usd": usd, "tokens": tokens,
            "fallback": fallback.astype(int), "llm_available": ~np.isnan(pick("usd"))}
