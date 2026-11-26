"""P6 - Decision layer: conformal thresholds on B, operating points (Bayes theta* per cost ratio + theta chosen on B),
router (LightGBM, 5-fold CV on A), cost model, learned vs uncertainty router on A (CV) and the router guard.

    python -m v2.scripts.run_p6_decision --datasets all
Outputs per dataset in artifacts/{ds}/decision/: conformal.json, operating_points.json, router_scores.parquet
(one block per rep x operating point, column `op`), router_cv.json (out-of-fold benefit on A per budget + `guard`),
router_importance.json.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from ..common import dump_json
from ..config import artifacts_dir, dataset_dir, get_logger, load_config, log_run_header, parse_datasets
from ..decision.bayes import cost, theta_star
from ..decision.budget import route_learned, route_uncertainty
from ..decision.conformal import alpha_min, threshold
from ..decision.cost_model import CostModel
from ..decision.frame import DecisionFrame, llm_outcome
from ..decision.operating import resolve as resolve_ops
from ..decision.router import Router, design

log = get_logger("p6")


def conformal_table(df: DecisionFrame, reps: list[str], alphas: list[float]) -> dict:
    out = {}
    for rep in reps:
        scores = df.scores[(df.scores["rep"] == rep) & (df.scores["subset"] == "b") & (df.scores["label"] == 1)]
        n_pos = int(len(scores))
        out[rep] = {"n_pos_b": n_pos, "alpha_min": alpha_min(n_pos), "thresholds": {}}
        for alpha in alphas:
            t, k = threshold(scores["s"].to_numpy(), alpha)
            out[rep]["thresholds"][str(alpha)] = {"t_alpha": t, "k": k, "achievable": t is not None,
                                                 "bound": k / (n_pos + 1) if n_pos else None}
    return out


def fit_router_rep(df: DecisionFrame, rep: str, cost_ratio: float, cfg: dict, calibrator: str, *,
                   no_refute: bool = False, no_cpg: bool = False, refute_no_hyp: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Router + cost model for one (prefilter rep, cost ratio). no_refute (A9) / no_cpg (A10) change d_llm."""
    frame = df.for_rep(rep, calibrator)
    frame = frame[frame["subset"].isin(["a_pool", "test"])].reset_index(drop=True)
    theta = theta_star(cost_ratio)
    out = llm_outcome(frame, theta, no_refute=no_refute, no_cpg=no_cpg, refute_no_hyp=refute_no_hyp)
    y = frame["label"].to_numpy()
    delta = cost(out["d_pf"], y, cost_ratio) - cost(out["d_llm"], y, cost_ratio)
    frame = frame.assign(delta=delta, d_pf=out["d_pf"], d_llm=out["d_llm"], cost_usd=out["cost_usd"],
                         mode_refute=out["mode_refute"], fallback=out["fallback"], llm_available=out["llm_available"])
    seed = int(rep) if str(rep).isdigit() else int.from_bytes(str(rep).encode(), "little") % 1000
    a_all = frame[(frame["subset"] == "a_pool") & frame["in_main"]]
    rng = np.random.default_rng(seed)
    a_fit = a_all.iloc[np.sort(rng.choice(len(a_all), size=max(10, int(len(a_all) * float(cfg["subsample_fraction"]))),
                                          replace=False))] if len(a_all) > 10 else a_all
    router = Router(cfg, seed)
    x_fit = design(a_fit)
    oof_fit = router.cross_validate(x_fit, a_fit["delta"].to_numpy(dtype=float), a_fit["weight"].to_numpy(dtype=float))
    router.fit(x_fit, a_fit["delta"].to_numpy(dtype=float), a_fit["weight"].to_numpy(dtype=float))
    frame["U_hat"] = router.predict(design(frame))
    frame.loc[a_fit.index, "U_hat"] = oof_fit                       # out-of-fold on the rows used for training
    cost_model = CostModel().fit(a_all[a_all["llm_available"]])
    frame["c_hat"] = cost_model.predict(frame)
    info = {"theta": theta, "best_iteration": router.best_iteration, "importance": router.importance(),
            "cost_model_r2": cost_model.r2, "a_fit": int(len(a_fit))}
    frame["rep"], frame["cost_ratio"] = rep, cost_ratio
    return frame, info


def cv_benefit(frame: pd.DataFrame, budgets: list[float], theta: float) -> dict:
    """On A (out-of-fold U_hat): weighted total benefit of learned vs uncertainty routing per budget.
    guard = learned benefit > 0: the router guard routes at that budget only then (else B = 0)."""
    a = frame[(frame["subset"] == "a_pool") & frame["in_main"]].reset_index(drop=True)
    eligible = np.ones(len(a), dtype=bool)
    w, delta = a["weight"].to_numpy(), a["delta"].to_numpy()
    out = {}
    for budget in budgets:
        if budget <= 0:
            continue
        n = int(round(min(budget, 1.0) * len(a)))
        learned = route_learned(a["U_hat"].to_numpy(), a["c_hat"].to_numpy(), eligible, n)
        uncertain = route_uncertainty(a["p_hat"].to_numpy(), theta, eligible, n)
        benefit = float((w * delta)[learned].sum())
        out[str(budget)] = {"learned": benefit, "uncertainty": float((w * delta)[uncertain].sum()),
                            "routed_a": int(learned.sum()), "guard": bool(benefit > 0)}
    return out


def operating_points(df: DecisionFrame, rep: str, cfg: dict) -> dict:
    """Operating points of one rep (decision/operating.py); theta of f1B is chosen on B (in_main rows)."""
    frame = df.for_rep(rep, cfg["calibrator"])
    return resolve_ops(frame[(frame["subset"] == "b") & frame["in_main"]], cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock:
        os.environ["VG2_LLM_MOCK"] = "1"
    log_run_header("run_p6_decision", args, ["decision", "experiments", "prefilter"])
    cfg = load_config("decision")
    wins = {}
    for dataset in parse_datasets(args.datasets):
        df = DecisionFrame(dataset)
        reps = df.reps()
        out_dir = dataset_dir(dataset, "decision")
        conformal = conformal_table(df, reps, [float(a) for a in cfg["alphas"]])
        dump_json(out_dir / "conformal.json", conformal)
        frames, infos, cv, ops_all = [], {}, {}, {}
        for rep in reps:
            ops_all[rep] = operating_points(df, rep, cfg)
            for op, spec in ops_all[rep].items():
                frame, info = fit_router_rep(df, rep, float(spec["cost_ratio"]), cfg["router"], cfg["calibrator"])
                frame["op"] = op
                frames.append(frame[["id", "rep", "op", "cost_ratio", "subset", "in_main", "label", "weight", "p_hat", "s",
                                     "u", "U_hat", "c_hat", "delta", "d_pf", "d_llm", "cost_usd", "mode_refute", "fallback",
                                     "llm_available"]])
                infos[f"rep{rep}_{op}"] = info
                cv[f"rep{rep}_{op}"] = cv_benefit(frame, [float(b) for b in cfg["budgets"]], info["theta"])
        pd.concat(frames, ignore_index=True).to_parquet(out_dir / "router_scores.parquet", index=False)
        dump_json(out_dir / "operating_points.json", ops_all)
        dump_json(out_dir / "router_importance.json", infos)
        dump_json(out_dir / "router_cv.json", cv)
        main_op = str(load_config("experiments")["main"]["operating_point"])
        keys = [k for k in cv if k.endswith(f"_{main_op}")]
        better = [np.mean([v["learned"] > v["uncertainty"] for v in cv[k].values()]) for k in keys if cv[k]]
        wins[dataset] = float(np.mean(better)) if better else None
        budget = str(float(load_config("experiments")["main"]["budget"]))
        guard = {k.split("_")[0]: cv[k].get(budget, {}).get("guard") for k in keys}
        alpha_info = {rep: round(c["alpha_min"], 5) for rep, c in conformal.items()}
        theta_info = {rep: round(ops[main_op]["theta"], 3) for rep, ops in ops_all.items() if main_op in ops}
        log.info("[%s] alpha_min per rep %s | theta (%s) per rep %s", dataset, alpha_info, main_op, theta_info)
        log.info("[%s] learned>uncertainty on A (CV, %s): %s | router guard open at B=%s: %s", dataset, main_op,
                 wins[dataset], budget, guard)
    summary_path = artifacts_dir() / "p6_summary.json"
    if summary_path.exists():                     # keep datasets decided in earlier (per-dataset) runs
        import json as _json
        previous = _json.loads(summary_path.read_text()).get("learned_beats_uncertainty_fraction", {})
        wins = {**previous, **wins}
    ok = sum(1 for v in wins.values() if v is not None and v > 0.5)
    dump_json(summary_path, {"learned_beats_uncertainty_fraction": wins, "datasets_won": ok,
                             "done_criterion_met": ok >= 2})
    log.info("P6 done criterion (learned > uncertainty on >= 2 datasets): %s", ok >= 2)


if __name__ == "__main__":
    main()
