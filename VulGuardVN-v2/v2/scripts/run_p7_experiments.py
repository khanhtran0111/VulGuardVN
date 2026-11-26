"""P7 - Evaluation and experiments, all offline from the caches (plan §7).

    python -m v2.scripts.run_p7_experiments --datasets all
Outputs (artifacts/results/):
  metrics_grid.parquet      every (dataset, rep, alpha, cost ratio, B, router) point          (P7 step 1-3)
  baselines.csv / ablations.csv / summary_main.csv   mean +- SD, 95% CI over repetitions
  significance.csv          paired bootstrap (10k) + McNemar, v2 vs each baseline (rep 1)
  rq2_conformal.csv, rq3_router.csv, rq4_layers.csv, rq5_verifier.csv, tables.md, figures/*.png
  {ds}/predictions/{config}.parquet  per-record predictions of the main configuration (traceable to a config)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ..common import dump_json
from ..config import artifacts_dir, config_hash, dataset_dir, get_logger, load_config, log_run_header, parse_datasets
from ..data.io import load_records
from ..decision.conformal import threshold
from ..decision.frame import DecisionFrame, load_decisions, load_scores
from ..eval.metrics import binary_metrics, pair_metrics, vd_score
from ..eval.simulate import band_for, evaluate, simulate
from ..eval.stats import mcnemar, mean_ci, paired_bootstrap
from .run_p6_decision import cv_benefit, fit_router_rep, operating_points

log = get_logger("p7")
KEY_METRICS = ["f1", "mcc", "precision", "recall", "specificity", "balanced_accuracy", "expected_cost", "skip_miss_rate",
               "coverage", "P-C", "P-V", "P-B", "P-R", "usd", "llm_functions", "routed_fraction"]


def results_dir():
    path = artifacts_dir() / "results"
    (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def v2_name(router: str) -> str:
    return "v2 (learned router + guard)" if router == "learned_guard" else f"v2 ({router} router)"


class DatasetEval:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.dcfg = load_config("decision")
        self.ecfg = load_config("experiments")
        self.records = load_records(dataset, ["id", "pair_id", "pair_role"])
        decision_dir = dataset_dir(dataset, "decision")
        self.router = pd.read_parquet(decision_dir / "router_scores.parquet")
        if "op" not in self.router:
            raise SystemExit(f"[{dataset}] router_scores.parquet predates operating points - rerun run_p6_decision")
        self.conformal = json.loads((decision_dir / "conformal.json").read_text())
        self.ops = json.loads((decision_dir / "operating_points.json").read_text())
        self.cv = json.loads((decision_dir / "router_cv.json").read_text())
        self.main_op = str(self.ecfg["main"]["operating_point"])
        self.eval_ratio = float(self.ecfg.get("eval_cost_ratio", self.dcfg["main_cost_ratio"]))
        decisions = load_decisions(dataset)
        main = decisions[decisions["variant"] == "main"].pivot_table(index="id", columns="mode", values="llm_calls", aggfunc="first")
        self.calls = main.rename(columns=lambda c: f"llm_calls_{c}").reset_index()
        z = decisions[decisions["variant"] == "zeroshot"][["id", "d", "usd"]]
        self.zeroshot = z.rename(columns={"d": "d_zeroshot", "usd": "usd_zeroshot"})
        self.decisions = decisions

    def op_names(self) -> list[str]:
        return list(dict.fromkeys(self.router["op"]))

    def ratio(self, rep: str, op: str) -> float:
        return float(self.ops[str(rep)][op]["cost_ratio"])

    def theta(self, rep: str, op: str) -> float:
        return float(self.ops[str(rep)][op]["theta"])

    def guard_open(self, rep: str, op: str, budget: float) -> bool:
        """Router guard (decision.yaml): out-of-fold benefit of the learned router on A at this budget is > 0."""
        if budget <= 0 or not self.dcfg.get("router_guard", True):
            return True
        return bool(self.cv.get(f"rep{rep}_{op}", {}).get(str(float(budget)), {}).get("guard", True))

    def sim(self, test: pd.DataFrame, rep: str, op: str, alpha, budget: float, router: str, **kwargs) -> dict:
        """simulate() at an operating point; router 'learned_guard' = learned router, B = 0 when the guard is closed."""
        if router == "learned_guard":
            router, budget = "learned", (budget if self.guard_open(rep, op, budget) else 0.0)
        return simulate(test, t_alpha=self.t_alpha(rep, alpha), budget=budget, router=router, theta=self.theta(rep, op), **kwargs)

    def frames(self, rep: str, op: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        part = self.router[(self.router["rep"] == rep) & (self.router["op"] == op)]
        part = part.merge(self.records, on="id", how="left").merge(self.calls, on="id", how="left")
        part["llm_calls"] = np.where(part["mode_refute"], part.get("llm_calls_refute", 0), part.get("llm_calls_neutral", 0))
        part["llm_calls"] = part["llm_calls"].fillna(0)
        return part[part["subset"] == "a_pool"].reset_index(drop=True), part[part["subset"] == "test"].reset_index(drop=True)

    def t_alpha(self, rep: str, alpha: float | None) -> float | None:
        if alpha is None:
            return None
        return self.conformal[rep]["thresholds"][str(alpha)]["t_alpha"]


def _config(dataset, rep, alpha, op, budget, router, extra=None) -> dict:
    cfg = {"dataset": dataset, "rep": rep, "alpha": alpha, "op": op, "budget": budget, "router": router}
    cfg.update(extra or {})
    cfg["config_id"] = config_hash(cfg)
    return cfg


# ---------------------------------------------------------------- grid (RQ1 curves, A5/A6)
def run_grid(ev: DatasetEval) -> pd.DataFrame:
    rows = []
    alphas = [None] + [float(a) for a in ev.dcfg["alphas"]]
    for rep in sorted(ev.router["rep"].unique(), key=str):
        for op in ev.op_names():
            a, test = ev.frames(rep, op)
            theta, ratio = ev.theta(rep, op), ev.ratio(rep, op)
            for alpha in alphas:
                t_alpha = ev.t_alpha(rep, alpha)
                for budget in ev.dcfg["budgets"]:
                    for router in ev.ecfg["routers"]:
                        band = band_for(a, theta, float(budget)) if router == "two_threshold" else None
                        seed = int(rep) * 1000 + int(budget * 100) if str(rep).isdigit() else 0
                        result = ev.sim(test, rep, op, alpha, float(budget), router, band=band, seed=seed)
                        # expected cost at the operating point's own ratio (the router's objective)
                        rows.append({**_config(ev.dataset, rep, alpha, op, float(budget), router), "cost_ratio": ratio,
                                     "theta": theta, "t_alpha": t_alpha, **evaluate(test, result, ratio)})
    return pd.DataFrame(rows)


def save_main_predictions(ev: DatasetEval) -> None:
    main = ev.ecfg["main"]
    out = dataset_dir(ev.dataset, "predictions")
    for rep in sorted(ev.router["rep"].unique(), key=str):
        a, test = ev.frames(rep, ev.main_op)
        result = ev.sim(test, rep, ev.main_op, main["alpha"], float(main["budget"]), main["router"],
                        seed=int(rep) if str(rep).isdigit() else 0)
        cfg = _config(ev.dataset, rep, main["alpha"], ev.main_op, float(main["budget"]), main["router"],
                      {"theta": ev.theta(rep, ev.main_op), "guard_open": ev.guard_open(rep, ev.main_op, float(main["budget"]))})
        frame = test[["id", "label", "in_main", "pair_id", "pair_role", "s", "p_hat", "U_hat", "c_hat", "d_pf", "d_llm",
                      "mode_refute", "fallback"]].assign(pred=result["pred"], skipped=result["skip"], routed=result["routed"])
        for key, value in cfg.items():
            frame[f"cfg_{key}"] = value
        frame.to_parquet(out / f"main_rep{rep}_{cfg['config_id']}.parquet", index=False)


# ---------------------------------------------------------------- baselines (§7.3)
def run_baselines(ev: DatasetEval) -> tuple[pd.DataFrame, dict]:
    main = ev.ecfg["main"]
    op, alpha, budget = ev.main_op, main["alpha"], float(main["budget"])
    cost_ratio = ev.eval_ratio                       # expected cost reported at the same c_FN/c_FP for every method
    rows, preds = [], {}
    for rep in sorted(ev.router["rep"].unique(), key=str):
        a, test = ev.frames(rep, op)
        theta = ev.theta(rep, op)
        y = test["label"].to_numpy()
        variants = {
            v2_name(main["router"]): ev.sim(test, rep, op, alpha, budget, main["router"]),
            "B1 prefilter only (B=0)": ev.sim(test, rep, op, alpha, 0.0, "learned"),
            "B2 LLM layer on 100%": simulate(test, t_alpha=None, budget=1.0, router="all", theta=theta, use_gate=False),
            "B4 uncertainty cascade": ev.sim(test, rep, op, alpha, budget, "uncertainty"),
            "B5 random routing": ev.sim(test, rep, op, alpha, budget, "random", seed=int(rep)),
            "B6 oracle router (upper bound, uses test labels)": ev.sim(test, rep, op, alpha, budget, "oracle"),
        }
        variants["B0 all vulnerable"] = {"pred": np.ones(len(test), int), "skip": np.zeros(len(test), bool),
                                         "routed": np.zeros(len(test), bool)}
        z = test[["id"]].merge(ev.zeroshot, on="id", how="left")
        if z["d_zeroshot"].notna().any():
            variants["B3 zero-shot gpt-oss-20b"] = {"pred": z["d_zeroshot"].fillna(0).astype(int).to_numpy(),
                                                    "skip": np.zeros(len(test), bool), "routed": z["d_zeroshot"].notna().to_numpy()}
        b7 = dataset_dir(ev.dataset, "baselines") / "b7_unixcoder_ft.parquet"
        if b7.exists():
            bp = test[["id"]].merge(pd.read_parquet(b7)[["id", "pred"]], on="id", how="left")
            variants["B7 UniXcoder fine-tune"] = {"pred": bp["pred"].fillna(0).astype(int).to_numpy(),
                                                  "skip": np.zeros(len(test), bool), "routed": np.zeros(len(test), bool)}
        for name, result in variants.items():
            metrics = evaluate(test, result, cost_ratio)
            if name.startswith("B3"):
                metrics["usd"] = float(z["usd_zeroshot"].fillna(0).sum())
            rows.append({"dataset": ev.dataset, "rep": rep, "method": name, **metrics})
            preds.setdefault(name, {})[rep] = (y, result["pred"], test["in_main"].to_numpy(dtype=bool))
        rows.append({"dataset": ev.dataset, "rep": rep, "method": "prefilter VD-S (score)",
                     "vds": vd_score(y[test["in_main"]], test.loc[test["in_main"], "s"].to_numpy(), float(ev.ecfg["vds_fpr"]))})
    return pd.DataFrame(rows), preds


def significance(dataset: str, preds: dict, n_boot: int) -> list[dict]:
    rows = []
    reference = next((by_rep for name, by_rep in preds.items() if name.startswith("v2")), {})
    for rep, (y, pred_v2, main) in reference.items():
        if rep != sorted(reference, key=str)[0]:
            continue
        for name, by_rep in preds.items():
            if name.startswith("v2") or rep not in by_rep:
                continue
            _, pred_b, _ = by_rep[rep]
            for metric_name, fn in {"f1": lambda a, b: binary_metrics(a, b)["f1"], "mcc": lambda a, b: binary_metrics(a, b)["mcc"]}.items():
                boot = paired_bootstrap(y[main], pred_v2[main], pred_b[main], fn, n_samples=n_boot)
                rows.append({"dataset": dataset, "rep": rep, "baseline": name, "metric": metric_name, **boot})
            rows.append({"dataset": dataset, "rep": rep, "baseline": name, "metric": "mcnemar", **mcnemar(y[main], pred_v2[main], pred_b[main])})
    return rows


# ---------------------------------------------------------------- ablations (§7.4)
def run_ablations(ev: DatasetEval, df: DecisionFrame) -> pd.DataFrame:
    main = ev.ecfg["main"]
    op, alpha, budget = ev.main_op, main["alpha"], float(main["budget"])
    cost_ratio = ev.eval_ratio
    guarded = main["router"] == "learned_guard" and ev.dcfg.get("router_guard", True)
    rcfg = ev.dcfg["router"]
    scores = load_scores(ev.dataset)
    rows = []

    def run(name: str, rep: str, calibrator: str = "beta", guard: bool = True, **flags) -> None:
        spec = operating_points(df, rep, {**ev.dcfg, "calibrator": calibrator})[op]     # theta of this rep / calibrator
        frame, info = fit_router_rep(df, rep, float(spec["cost_ratio"]), rcfg, calibrator, **flags)
        frame = frame.merge(ev.records, on="id", how="left", suffixes=("", "_r"))
        test = frame[frame["subset"] == "test"].reset_index(drop=True)
        positives = scores[(scores["rep"] == rep) & (scores["subset"] == "b") & (scores["label"] == 1)]["s"].to_numpy()
        t_alpha, _ = threshold(positives, alpha) if alpha is not None else (None, 0)
        open_ = (not (guard and guarded)) or cv_benefit(frame, [budget], info["theta"]).get(str(budget), {}).get("guard", True)
        result = simulate(test, t_alpha=t_alpha, budget=budget if open_ else 0.0, router="learned", theta=float(spec["theta"]))
        rows.append({"dataset": ev.dataset, "ablation": name, "rep": rep, **evaluate(test, result, cost_ratio)})

    main_reps = df.reps()
    for rep in main_reps:
        run("full v2", rep)
        run("A4 isotonic instead of beta", rep, calibrator="iso")
        run("A9 no refute mode", rep, no_refute=True)
        run("A10 no CPG verification", rep, no_cpg=True)
        run("A12 plan rule: refute without hypothesis -> 1", rep, refute_no_hyp=1)
        if guarded:
            run("A13 no router guard", rep, guard=False)
    for rep in df.reps("single-r"):
        run("A2 no ensemble", rep)
    for rep in df.reps("nopair-r"):
        run("A3 no pairwise loss", rep)
    for rep in df.reps("joint-r"):
        run("A14 joint fusion (fine-tuned logit as a network input, no stacking)", rep)
    for rep in df.reps("lovo-"):
        view = rep.split("-")[1]
        run(f"A1 leave out {view}", rep)
    grid = pd.read_parquet(results_dir() / "metrics_grid.parquet")
    grid = grid[(grid["dataset"] == ev.dataset) & (grid["op"] == op) & (grid["budget"] == budget) &
                (grid["alpha"].fillna(-1) == (alpha if alpha is not None else -1))]
    for router, name in [("two_threshold", "A5 two fixed thresholds"), ("uncertainty", "A6 uncertainty-only router")]:
        for row in grid[grid["router"] == router].to_dict("records"):
            rows.append({"dataset": ev.dataset, "ablation": name, "rep": row["rep"], **{k: row[k] for k in KEY_METRICS if k in row}})
    # A11: at the same recall, assign the high branch directly (top p_hat) instead of checking with the LLM
    for rep in main_reps:
        a, test = ev.frames(rep, op)
        main_rows = test[test["in_main"]]
        result = ev.sim(test, rep, op, alpha, budget, main["router"])
        recall = binary_metrics(main_rows["label"], result["pred"][test["in_main"].to_numpy()])["recall"]
        order = np.argsort(-main_rows["p_hat"].to_numpy())
        y = main_rows["label"].to_numpy()
        tp_needed = int(np.ceil(recall * y.sum()))
        cum = np.cumsum(y[order])
        k = int(np.searchsorted(cum, tp_needed) + 1) if tp_needed else 0
        pred = np.zeros(len(y), int)
        pred[order[:k]] = 1
        rows.append({"dataset": ev.dataset, "ablation": "A11 direct high-branch assignment (same recall)", "rep": rep,
                     **binary_metrics(y, pred)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RQ2 conformal, RQ3 router, RQ4/5 LLM layer
def rq2(dataset: str) -> pd.DataFrame:
    scores = load_scores(dataset)
    rows = []
    for rep in [r for r in scores["rep"].unique() if str(r).isdigit()]:
        part = scores[scores["rep"] == rep]
        b_pos = part[(part["subset"] == "b") & (part["label"] == 1)]["s"].to_numpy()
        test = part[(part["subset"] == "test") & part["in_main"]]
        for alpha in np.unique(np.r_[load_config("decision")["alphas"], np.linspace(0.005, 0.2, 40)]):
            t, k = threshold(b_pos, float(alpha))
            skipped = (test["s"] <= t).to_numpy() if t is not None else np.zeros(len(test), bool)
            y = test["label"].to_numpy()
            rows.append({"dataset": dataset, "rep": rep, "alpha": float(alpha), "achievable": t is not None,
                         "empirical_miss": float(skipped[y == 1].mean()) if (y == 1).any() else np.nan,
                         "coverage": float(skipped.mean()), "split": "temporal" if dataset.startswith("primevul") and "random" not in dataset else "random"})
    return pd.DataFrame(rows)


def rq3(ev: DatasetEval) -> pd.DataFrame:
    grid = pd.read_parquet(results_dir() / "metrics_grid.parquet")
    main = ev.ecfg["main"]
    grid = grid[(grid["dataset"] == ev.dataset) & (grid["op"] == ev.main_op) &
                (grid["alpha"].fillna(-1) == (main["alpha"] if main["alpha"] is not None else -1))]
    rows = []
    base = grid[grid["budget"] == 0][["rep", "expected_cost"]].rename(columns={"expected_cost": "cost_b0"}).drop_duplicates("rep")
    for budget, part in grid.groupby("budget"):
        merged = part.merge(base, on="rep")
        for router, rpart in merged.groupby("router"):
            gain = (rpart["cost_b0"] - rpart["expected_cost"]).mean()
            rows.append({"dataset": ev.dataset, "budget": budget, "router": router, "cost_reduction_vs_B0": float(gain)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        oracle = frame[frame["router"] == "oracle"].set_index("budget")["cost_reduction_vs_B0"]
        frame["fraction_of_oracle"] = frame.apply(lambda r: r["cost_reduction_vs_B0"] / oracle.get(r["budget"], np.nan)
                                                  if oracle.get(r["budget"], 0) else np.nan, axis=1)
    return frame


def rq3_transfer(datasets: list[str]) -> pd.DataFrame:
    """Router trained on dataset X's A-pool, applied to dataset Y's test (rep 1, main cost ratio)."""
    from ..decision.router import Router, design

    cfg = load_config("decision")
    main = load_config("experiments")["main"]
    frames = {}
    for dataset in datasets:
        path = dataset_dir(dataset, "decision") / "operating_points.json"
        ops = json.loads(path.read_text()) if path.exists() else {}
        if "1" in ops and main["operating_point"] in ops["1"]:
            df = DecisionFrame(dataset)
            ratio = float(ops["1"][main["operating_point"]]["cost_ratio"])
            frame, _ = fit_router_rep(df, "1", ratio, cfg["router"], cfg["calibrator"])
            frames[dataset] = frame
    rows = []
    for source, sframe in frames.items():
        a = sframe[(sframe["subset"] == "a_pool") & sframe["in_main"]]
        router = Router(cfg["router"], 1)
        router.cross_validate(design(a), a["delta"].to_numpy(float), a["weight"].to_numpy(float))
        router.fit(design(a), a["delta"].to_numpy(float), a["weight"].to_numpy(float))
        for target, tframe in frames.items():
            test = tframe[tframe["subset"] == "test"].reset_index(drop=True)
            u_hat = router.predict(design(test))
            n = int(round(float(main["budget"]) * test["in_main"].sum()))
            ratio = u_hat / np.clip(test["c_hat"].to_numpy(), 1e-12, None)
            chosen = [i for i in np.argsort(-ratio) if u_hat[i] > 0][:n]
            rows.append({"source": source, "target": target, "benefit": float(test["delta"].to_numpy()[chosen].sum())})
    return pd.DataFrame(rows)


def rq4_rq5(ev: DatasetEval) -> pd.DataFrame:
    """LLM layer alone on the API subset: main vs A7 (no Pass 1), A8 (single demo), RQ5 (gpt-oss-120b),
    A9 (neutral only), A10 (no CPG verification), zero-shot."""
    theta = ev.theta("1", ev.main_op)
    scores = load_scores(ev.dataset)
    p_hat = scores[scores["rep"] == "1"].set_index("id")[f"p_hat_{ev.dcfg['calibrator']}"]
    records = load_records(ev.dataset, ["id", "label", "in_main", "pair_id", "pair_role"]).set_index("id")
    rows = []
    variants = ev.decisions["variant"].unique()
    decisions = ev.decisions
    if "a7" in variants:                       # A7 runs Pass 2 on every function of the API subset
        subset_ids = set(decisions.loc[decisions["variant"] == "a7", "id"])
    else:
        subset_ids = set(decisions.loc[(decisions["subset"] == "test") & (decisions["variant"] == "main"), "id"])
    main_rows = decisions[(decisions["variant"] == "main") & decisions["id"].isin(subset_ids)]

    def wide_of(variant: str) -> pd.DataFrame:
        part = decisions[(decisions["variant"] == variant) & decisions["id"].isin(subset_ids)]
        if variant in {"a8", "rq5"}:
            # functions without a Pass-1 hypothesis get no Pass 2: same decision as main (table §6.5)
            missing = main_rows[~main_rows["id"].isin(set(part["id"]))]
            part = pd.concat([part, missing])
        return part.pivot_table(index="id", columns="mode", values=["d", "d_raw", "fallback"], aggfunc="first")

    for variant in [v for v in ["main", "a7", "a8", "rq5"] if v in variants]:
        for label, flags in {"": {}, " / neutral only (A9)": {"neutral_only": True}, " / no CPG (A10)": {"raw": True}}.items():
            if variant != "main" and label:
                continue
            wide = wide_of(variant)
            if wide.empty:
                continue
            ids = [i for i in wide.index if i in p_hat.index]
            refute = (p_hat.loc[ids].to_numpy() >= theta) & (not flags.get("neutral_only"))
            col = "d_raw" if flags.get("raw") else "d"
            d = np.where(refute, wide.loc[ids, (col, "refute")], wide.loc[ids, (col, "neutral")]).astype(float)
            pf = (p_hat.loc[ids].to_numpy() >= theta).astype(int)
            pred = np.where(np.isnan(d), pf, d).astype(int)
            frame = records.loc[ids].assign(pred=pred)
            m = frame[frame["in_main"]]
            rows.append({"dataset": ev.dataset, "variant": {"main": "v2 LLM layer", "a7": "A7 no Pass 1", "a8": "A8 single demo",
                                                            "rq5": "RQ5 verifier gpt-oss-120b"}[variant] + label,
                         "n": len(ids), **binary_metrics(m["label"], m["pred"]), **pair_metrics(frame.reset_index())})
    z = ev.decisions[(ev.decisions["variant"] == "zeroshot") & ev.decisions["id"].isin(subset_ids)]
    if len(z):
        frame = records.loc[z["id"]].assign(pred=z["d"].fillna(0).astype(int).to_numpy())
        m = frame[frame["in_main"]]
        rows.append({"dataset": ev.dataset, "variant": "zero-shot (B3)", "n": len(z), **binary_metrics(m["label"], m["pred"]),
                     **pair_metrics(frame.reset_index())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summaries, tables, figures
def summarise(frame: pd.DataFrame, by: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(by, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        for metric in metrics:
            if metric in part:
                ci = mean_ci(part[metric].tolist())
                row[metric] = ci["mean"]
                row[f"{metric}_sd"] = ci["sd"]
                row[f"{metric}_ci"] = f"[{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
        row["n_reps"] = int(part["rep"].nunique()) if "rep" in part else 1
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(frame: pd.DataFrame, columns: list[str]) -> str:
    cols = [c for c in columns if c in frame]
    if frame.empty or not cols:
        return "_no data_\n"
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in frame[cols].itertuples(index=False):
        cells = []
        for value in row:
            cells.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def plots(grid: pd.DataFrame, rq2_frame: pd.DataFrame, baselines: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main = load_config("experiments")["main"]
    figures = results_dir() / "figures"
    colors = {"learned": "#4C78A8", "learned_guard": "#1F3A5F", "oracle": "#54A24B", "uncertainty": "#F58518", "random": "#9D9D9D", "two_threshold": "#B279A2"}
    for dataset, part in grid.groupby("dataset"):
        part = part[(part["op"] == str(main["operating_point"])) &
                    (part["alpha"].fillna(-1) == (main["alpha"] if main["alpha"] is not None else -1))]
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for router, rpart in part.groupby("router"):
            agg = rpart.groupby("budget").agg(f1=("f1", "mean"), mcc=("mcc", "mean"), cost=("expected_cost", "mean"),
                                              usd=("usd", "mean")).reset_index()
            for ax, metric in zip(axes, ["f1", "mcc", "cost"]):
                ax.plot(agg["budget"], agg[metric], marker="o", label=router, color=colors.get(router))
        for ax, title in zip(axes, ["F1", "MCC", f"expected cost (operating point {main['operating_point']})"]):
            ax.set_xlabel("budget B (fraction of functions sent to the LLM)")
            ax.set_title(title)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{dataset}: cost-effectiveness vs budget (alpha={main['alpha']})")
        fig.tight_layout()
        fig.savefig(figures / f"cost_curve_{dataset}.png", dpi=150)
        plt.close(fig)
    if not rq2_frame.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
        for dataset, part in rq2_frame.groupby("dataset"):
            agg = part.groupby("alpha").agg(miss=("empirical_miss", "mean"), coverage=("coverage", "mean")).reset_index()
            axes[0].plot(agg["alpha"], agg["miss"], marker=".", label=f"{dataset} ({part['split'].iloc[0]})")
            axes[1].plot(agg["coverage"], agg["miss"], marker=".", label=dataset)
        lim = rq2_frame["alpha"].max()
        axes[0].plot([0, lim], [0, lim], "k--", lw=1, label="target alpha")
        axes[0].set_xlabel("alpha")
        axes[0].set_ylabel("empirical miss rate of skip (test)")
        axes[1].set_xlabel("coverage (fraction skipped)")
        axes[1].set_ylabel("miss rate")
        axes[0].legend(fontsize=8)
        axes[1].set_title("risk-coverage")
        fig.tight_layout()
        fig.savefig(figures / "rq2_conformal.png", dpi=150)
        plt.close(fig)
    if not baselines.empty and "skip_fn" in baselines:
        part = baselines[baselines["method"].str.startswith("v2")].groupby("dataset")[
            ["skip_fn", "bayes_fn", "bayes_fp", "llm_fn", "llm_fp"]].mean()
        ax = part.plot(kind="bar", figsize=(8, 3.6))
        ax.set_title("error by branch (v2 main configuration, mean over reps)")
        ax.figure.tight_layout()
        ax.figure.savefig(figures / "errors_by_branch.png", dpi=150)
        plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--skip-ablations", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=None)
    args = parser.parse_args()
    if args.mock:
        os.environ["VG2_LLM_MOCK"] = "1"
    log_run_header("run_p7_experiments", args, ["decision", "experiments", "llm", "prefilter"])
    ecfg = load_config("experiments")
    n_boot = int(args.bootstrap or ecfg["bootstrap_samples"])
    requested = parse_datasets(args.datasets)
    datasets = [d for d in requested if (dataset_dir(d, "decision") / "router_scores.parquet").exists()]
    skipped = [d for d in requested if d not in datasets]
    if skipped:
        log.info("not ready (P6 not run yet), skipped: %s", skipped)
    if not datasets:
        raise SystemExit("no dataset has finished P6 yet")
    out = results_dir()
    grids, base_rows, sig_rows, abl, rq2_rows, rq3_rows, rq45 = [], [], [], [], [], [], []
    evals = {}
    for dataset in datasets:
        log.info("[%s] grid", dataset)
        ev = DatasetEval(dataset)
        evals[dataset] = ev
        grids.append(run_grid(ev))
        save_main_predictions(ev)
    grid = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    grid.to_parquet(out / "metrics_grid.parquet", index=False)
    for dataset, ev in evals.items():
        log.info("[%s] baselines / RQs", dataset)
        frame, preds = run_baselines(ev)
        base_rows.append(frame)
        sig_rows += significance(dataset, preds, n_boot)
        rq2_rows.append(rq2(dataset))
        rq3_rows.append(rq3(ev))
        rq45.append(rq4_rq5(ev))
        if not args.skip_ablations:
            log.info("[%s] ablations", dataset)
            abl.append(run_ablations(ev, DecisionFrame(dataset)))
    baselines = pd.concat(base_rows, ignore_index=True) if base_rows else pd.DataFrame()
    summary = summarise(baselines, ["dataset", "method"], KEY_METRICS + ["vds"])
    summary.to_csv(out / "baselines.csv", index=False)
    pd.DataFrame(sig_rows).to_csv(out / "significance.csv", index=False)
    rq2_frame = pd.concat(rq2_rows, ignore_index=True) if rq2_rows else pd.DataFrame()
    rq2_frame.to_csv(out / "rq2_conformal.csv", index=False)
    rq3_frame = pd.concat(rq3_rows, ignore_index=True) if rq3_rows else pd.DataFrame()
    rq3_frame.to_csv(out / "rq3_router.csv", index=False)
    if len(datasets) > 1:
        rq3_transfer(datasets).to_csv(out / "rq3_transfer.csv", index=False)
    importance = {d: json.loads((dataset_dir(d, "decision") / "router_importance.json").read_text()) for d in datasets}
    dump_json(out / "rq3_feature_importance.json", importance)
    rq45_frame = pd.concat(rq45, ignore_index=True) if rq45 else pd.DataFrame()
    rq45_frame.to_csv(out / "rq4_rq5_llm_layer.csv", index=False)
    if abl:
        ablations = summarise(pd.concat(abl, ignore_index=True), ["dataset", "ablation"], KEY_METRICS)
        ablations.to_csv(out / "ablations.csv", index=False)
    else:   # --skip-ablations: keep the previous ablation table if there is one
        previous = out / "ablations.csv"
        ablations = pd.read_csv(previous) if previous.exists() and previous.stat().st_size > 1 else pd.DataFrame()
    plots(grid, rq2_frame, baselines)
    cols = ["dataset", "method", "f1", "f1_ci", "mcc", "mcc_ci", "recall", "precision", "expected_cost", "P-C", "skip_miss_rate",
            "coverage", "routed_fraction", "usd", "vds", "n_reps"]
    rq2_tab = rq2_frame[rq2_frame["alpha"].isin(load_config("decision")["alphas"])].groupby(["dataset", "split", "alpha"]).agg(
        empirical_miss=("empirical_miss", "mean"), coverage=("coverage", "mean")).reset_index() if not rq2_frame.empty else rq2_frame
    text = ["# VulGuard v2 - results", "", f"Main configuration: `{ecfg['main']}`. Mean over prefilter repetitions; CI = 95% t-interval.", "",
            "## RQ1 - baselines (main configuration)", to_markdown(summary, cols),
            "## RQ2 - conformal skip gate", to_markdown(rq2_tab, ["dataset", "split", "alpha", "empirical_miss", "coverage"]),
            "## RQ3 - routers vs oracle", to_markdown(rq3_frame[rq3_frame["budget"] == float(ecfg["main"]["budget"])] if not rq3_frame.empty else rq3_frame,
                                                       ["dataset", "budget", "router", "cost_reduction_vs_B0", "fraction_of_oracle"]),
            "## RQ4 / RQ5 - LLM layer on the API subset", to_markdown(rq45_frame, ["dataset", "variant", "n", "f1", "recall", "precision", "fp", "P-C", "P-V", "P-B", "P-R"]),
            "## Ablations", to_markdown(ablations, ["dataset", "ablation", "f1", "mcc", "expected_cost", "P-C", "usd", "n_reps"]),
            "Figures: `figures/cost_curve_*.png`, `figures/rq2_conformal.png`, `figures/errors_by_branch.png`.", ""]
    (out / "tables.md").write_text("\n".join(text), encoding="utf-8")
    log.info("results -> %s", out)


if __name__ == "__main__":
    main()
