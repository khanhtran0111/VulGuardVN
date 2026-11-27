"""Pilot: does gpt-oss-120b carry signal the prefilter lacks, where the router would send functions?

Go / no-go check before paying for a full 120b run (see the B6 / fusion analysis of 2026-11-27: with gpt-oss-20b
the LLM layer adds no information beyond p_hat on Devign / ReVeal - fusion AUC = prefilter AUC, the LLM is right on
only ~30% of the functions the prefilter gets wrong).

    python -m v2.scripts.run_pilot_120b --datasets devign,reveal --dry-run     # selection + cost estimate, no API
    python -m v2.scripts.run_pilot_120b --datasets devign,reveal               # call the API, then report
    python -m v2.scripts.run_pilot_120b --datasets devign,reveal --report-only # recompute the report from the cache

Selection (never touches test): A-pool, in_main, with a cached gpt-oss-20b decision, the N functions with p_hat
closest to theta (main operating point, selection rep) - the band a budgeted router would route.
LLM chain: the full v2 LLM layer with the rq5 model (llm.yaml rq5_model / rq5_provider), same frozen prompts,
reasoning effort and CPG verification as the main run: Pass 1 -> Pass 2 neutral + refute. Stages are written as
pilot120_pass1 / pilot120_neutral / pilot120_refute, so the main pipeline (P5-P7) never reads them.

Criteria, averaged over the prefilter reps (both must hold for PASS):
  C1  accuracy of 120b on the selected functions the prefilter gets wrong > 0.5 (a coin flip; 20b reaches ~0.3-0.4)
  C2  repeated 5-fold CV logistic fusion [logit p_hat + 120b outputs] - AUC of p_hat alone >= min_auc_gain,
      with the 2.5% bootstrap quantile of the gain > 0
Outputs: artifacts/{ds}/pilot/pilot120_ids.json, pilot120_decisions.parquet, pilot120_report.json and
artifacts/results/pilot120_summary.md.
"""
from __future__ import annotations

import argparse
import json
import warnings

import httpx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..common import dump_json, loads_or_none
from ..config import artifacts_dir, dataset_dir, get_logger, load_config, log_run_header, parse_datasets

log = get_logger("pilot120")
PREFIX = "pilot120"
P1_STAGE = f"{PREFIX}_pass1"
P2_STAGES = {"neutral": f"{PREFIX}_neutral", "refute": f"{PREFIX}_refute"}
STAGES = [P1_STAGE, *P2_STAGES.values()]
DEFAULT_N = {"reveal": 500}          # ReVeal: ~10% positives -> few prefilter errors near theta, so a larger sample
FALLBACK_PRICE_FACTOR = 5.0          # 120b / 20b price ratio if the OpenRouter price list cannot be read


# ---------------------------------------------------------------- selection
def load_router(dataset: str, op: str) -> pd.DataFrame:
    path = dataset_dir(dataset, "decision") / "router_scores.parquet"
    if not path.exists():
        raise SystemExit(f"[{dataset}] {path} missing - run run_p6_decision first")
    frame = pd.read_parquet(path)
    frame = frame[(frame["op"] == op) & frame["rep"].astype(str).str.isdigit()].copy()
    frame["rep"] = frame["rep"].astype(str)
    return frame


def selection(dataset: str, op: str, rep: str, n: int) -> dict:
    router = load_router(dataset, op)
    ops = json.loads((dataset_dir(dataset, "decision") / "operating_points.json").read_text())
    theta = float(ops[rep][op]["theta"])
    part = router[(router["rep"] == rep) & (router["subset"] == "a_pool") & router["in_main"].astype(bool)
                  & router["llm_available"].astype(bool)].reset_index(drop=True)
    if part.empty:
        raise SystemExit(f"[{dataset}] selection rep {rep} has no A-pool rows with LLM decisions")
    order = np.argsort(np.abs(part["p_hat"].to_numpy() - theta), kind="stable")[:n]
    chosen = part.iloc[order]
    return {"ids": chosen["id"].tolist(), "theta": theta, "rep": rep, "op": op, "n_pool": int(len(part)),
            "p_hat_range": [float(chosen["p_hat"].min()), float(chosen["p_hat"].max())],
            "positives": int(chosen["label"].sum()), "router": router, "ops": ops}


# ---------------------------------------------------------------- cost
def price_120b(model: str) -> tuple[dict | None, str]:
    """Public OpenRouter price list (no key, no project data sent). Returns per-token prices."""
    try:
        reply = httpx.get("https://openrouter.ai/api/v1/models", timeout=20)
        for item in reply.json().get("data", []):
            if item.get("id") == model:
                p = item.get("pricing") or {}
                return {"input": float(p.get("prompt", 0)), "output": float(p.get("completion", 0))}, "openrouter /models"
    except Exception as exc:  # noqa: BLE001
        log.info("price lookup failed (%s) - using %sx the 20b price", str(exc)[:120], FALLBACK_PRICE_FACTOR)
    return None, f"assumed {FALLBACK_PRICE_FACTOR}x the gpt-oss-20b price"


def estimate_usd(dataset: str, ids: list[str], model: str) -> dict:
    """Token usage of the cached 20b chain on the same functions x the 120b price (output tokens doubled as a margin:
    the larger model tends to write longer answers)."""
    from ..llm.runner import load_index

    index = load_index(dataset)
    part = index[index["id"].isin(set(ids)) & index["stage"].isin(["pass1", "pass2_neutral", "pass2_refute"])]
    tokens_in, tokens_out = float(part["prompt_tokens"].sum()), float(part["completion_tokens"].sum())
    usd_20b = float(part["cost_usd"].fillna(0).sum())
    price, source = price_120b(model)
    if price:
        usd = tokens_in * price["input"] + 2 * tokens_out * price["output"]
    else:
        usd = usd_20b * FALLBACK_PRICE_FACTOR * 1.5
    return {"functions": len(ids), "tokens_in_20b": int(tokens_in), "tokens_out_20b": int(tokens_out),
            "usd_20b_same_functions": round(usd_20b, 4), "estimated_usd_120b": round(usd, 4), "price_source": source,
            "price_per_m": {k: v * 1e6 for k, v in price.items()} if price else None}


# ---------------------------------------------------------------- 120b calls + CPG decisions
def run_120b(dataset: str, ids: list[str]) -> dict:
    from ..llm.runner import LLMLayer

    cfg = load_config("llm")
    layer = LLMLayer(dataset, model=cfg["rq5_model"], provider=dict(cfg["rq5_provider"]),
                     max_tokens=cfg.get("rq5_max_tokens"),
                     response_format=cfg.get("rq5_response_format"))
    log.info("[%s] %s Pass 1 on %d functions", dataset, cfg["rq5_model"], len(ids))
    p1 = layer.pass1(ids, stage=P1_STAGE)
    n_hyp = sum(1 for e in p1.values() if (e.get("parsed") or {}).get("hypotheses"))
    log.info("[%s] Pass 1 done: %d with hypotheses, %d invalid", dataset, n_hyp,
             sum(1 for e in p1.values() if e.get("parsed") is None))
    p2 = layer.pass2(ids, p1, prefix=PREFIX)
    log.info("[%s] Pass 2 done: %s | session spend %.4f USD", dataset, {k: len(v) for k, v in p2.items()},
             layer.client.spent_session)
    return {"pass1_with_hyp": n_hyp, "spend_session_usd": layer.client.spent_session}


def decisions_120b(dataset: str, ids: list[str]) -> pd.DataFrame:
    """CPG verification (P5 decide) of the cached pilot120 outputs, both modes, + the verbalised confidence."""
    from ..cpg.joern import load_facts
    from ..data.io import load_records
    from ..llm.runner import load_index
    from ..verify.cpg_decide import decide

    refute_no_hyp = int(load_config("decision").get("refute_without_hypothesis", 1))
    index = load_index(dataset)
    stages = [P1_STAGE, *P2_STAGES.values()]
    entry = {(r.id, r.stage): r for r in index[index["stage"].isin(stages)].itertuples(index=False)}
    facts = load_facts(dataset, set(ids))
    records = load_records(dataset, ["id", "code"]).set_index("id")
    rows = []
    for record_id in ids:
        p1_row = entry.get((record_id, P1_STAGE))
        p1 = {"parsed": loads_or_none(p1_row.parsed)} if p1_row is not None else None
        n_hyp = len(((p1 or {}).get("parsed") or {}).get("hypotheses") or []) if p1 and p1["parsed"] is not None else np.nan
        usd = float(p1_row.cost_usd or 0.0) if p1_row is not None else 0.0
        for mode in ("neutral", "refute"):
            p2_row = entry.get((record_id, P2_STAGES[mode]))
            parsed = loads_or_none(p2_row.parsed) if p2_row is not None else None
            p2 = {"parsed": parsed} if parsed is not None else None
            out = decide(mode, p1, p2, facts.get(record_id), records.loc[record_id, "code"], refute_no_hyp)
            conf = (parsed or {}).get("confidence")
            rows.append({"id": record_id, **out, "confidence": float(conf) if isinstance(conf, (int, float)) else np.nan,
                         "n_hyp": n_hyp, "usd": usd + (float(p2_row.cost_usd or 0.0) if p2_row is not None else 0.0)})
    return pd.DataFrame(rows)


def wide(dec: pd.DataFrame, tag: str) -> pd.DataFrame:
    """One row per id: d / fallback / d_raw / confidence per mode, n_hyp."""
    cols = [c for c in ["d", "fallback", "d_raw", "confidence"] if c in dec]
    w = dec.pivot_table(index="id", columns="mode", values=cols, aggfunc="first")
    w.columns = [f"{tag}_{a}_{b}" for a, b in w.columns]
    if "n_hyp" in dec:
        w[f"{tag}_n_hyp"] = dec.groupby("id")["n_hyp"].first()
    return w


def decisions_20b(dataset: str, ids: list[str]) -> pd.DataFrame:
    from ..llm.runner import load_index
    from ..scripts.run_p5_verify import decisions_path

    d = pd.read_parquet(decisions_path(dataset))
    d = d[(d["variant"] == "main") & d["id"].isin(set(ids))].copy()
    index = load_index(dataset)
    conf = {}
    for mode in ("neutral", "refute"):
        part = index[(index["stage"] == f"pass2_{mode}") & index["id"].isin(set(ids))].drop_duplicates("id", keep="last")
        conf.update({(i, mode): (loads_or_none(p) or {}).get("confidence") for i, p in zip(part["id"], part["parsed"])})
    d["confidence"] = [conf.get((i, m), np.nan) for i, m in zip(d["id"], d["mode"])]
    d["confidence"] = pd.to_numeric(d["confidence"], errors="coerce")
    p1 = index[(index["stage"] == "pass1") & index["id"].isin(set(ids))].drop_duplicates("id", keep="last")
    n_hyp = {i: len((loads_or_none(p) or {}).get("hypotheses") or []) for i, p in zip(p1["id"], p1["parsed"])}
    d["n_hyp"] = d["id"].map(n_hyp)
    return d


# ---------------------------------------------------------------- evaluation
def d_llm_for(frame: pd.DataFrame, tag: str, theta: float) -> np.ndarray:
    """d_llm(x, m(x)): refute mode if p_hat >= theta else neutral; fallback / missing -> d_pf (same as llm_outcome)."""
    refute = frame["p_hat"].to_numpy() >= theta
    d = np.where(refute, frame[f"{tag}_d_refute"].to_numpy(dtype=float), frame[f"{tag}_d_neutral"].to_numpy(dtype=float))
    fb = np.where(refute, frame[f"{tag}_fallback_refute"].to_numpy(dtype=float),
                  frame[f"{tag}_fallback_neutral"].to_numpy(dtype=float))
    fallback = np.isnan(d) | (fb == 1)
    return np.where(fallback, frame["d_pf"].to_numpy(), np.nan_to_num(d)).astype(int), fallback


def feature_matrix(frame: pd.DataFrame, tag: str | None) -> np.ndarray:
    p = np.clip(frame["p_hat"].to_numpy(dtype=float), 1e-4, 1 - 1e-4)
    cols = [np.log(p / (1 - p))]
    if tag:
        for name in ["d_neutral", "d_refute", "d_raw_neutral", "d_raw_refute", "confidence_neutral",
                     "confidence_refute", "fallback_neutral", "fallback_refute", "n_hyp"]:
            col = f"{tag}_{name}"
            if col in frame:
                v = frame[col].to_numpy(dtype=float)
                cols += [np.nan_to_num(v, nan=0.0), np.isnan(v).astype(float)]
    x = np.column_stack(cols)
    keep = x.std(0) > 0
    keep[0] = True
    x = x[:, keep]
    return (x - x.mean(0)) / (x.std(0) + 1e-9)


def cv_auc(x: np.ndarray, y: np.ndarray, repeats: int, seed: int) -> np.ndarray:
    """Out-of-fold predictions averaged over `repeats` stratified 5-fold splits."""
    oof = np.zeros((repeats, len(y)))
    for r in range(repeats):
        for train, valid in StratifiedKFold(5, shuffle=True, random_state=seed + r).split(x, y):
            model = LogisticRegression(C=0.3, max_iter=2000).fit(x[train], y[train])
            oof[r, valid] = model.predict_proba(x[valid])[:, 1]
    return oof.mean(0)


def bootstrap_gain(y: np.ndarray, base: np.ndarray, fused: np.ndarray, n: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        gains.append(roc_auc_score(y[idx], fused[idx]) - roc_auc_score(y[idx], base[idx]))
    return [float(np.quantile(gains, 0.025)), float(np.quantile(gains, 0.975))]


def evaluate_rep(frame: pd.DataFrame, theta: float, cost_ratio: float, tags: list[str], cv_repeats: int,
                 n_boot: int, seed: int) -> dict:
    y = frame["label"].to_numpy().astype(int)
    pf = frame["d_pf"].to_numpy().astype(int)
    pf_wrong = pf != y
    base_auc = roc_auc_score(y, frame["p_hat"])
    out = {"n": int(len(y)), "positives": int(y.sum()), "prefilter_wrong": int(pf_wrong.sum()),
           "prefilter_accuracy": float((~pf_wrong).mean()), "auc_p_hat": float(base_auc)}
    x0 = feature_matrix(frame, None)
    base_oof = cv_auc(x0, y, cv_repeats, seed)
    for tag in tags:
        d, fallback = d_llm_for(frame, tag, theta)
        fix = pf_wrong & (d == y)
        brk = ~pf_wrong & (d != y)
        cost_pf = cost_ratio * ((pf == 0) & (y == 1)) + ((pf == 1) & (y == 0))
        cost_llm = cost_ratio * ((d == 0) & (y == 1)) + ((d == 1) & (y == 0))
        xf = feature_matrix(frame, tag)
        fused = cv_auc(xf, y, cv_repeats, seed)
        conf = np.where(frame["p_hat"].to_numpy() >= theta, frame[f"{tag}_confidence_refute"], frame[f"{tag}_confidence_neutral"])
        ok = ~np.isnan(conf.astype(float))
        out[tag] = {
            "acc_where_prefilter_wrong": float((d[pf_wrong] == y[pf_wrong]).mean()) if pf_wrong.any() else np.nan,
            "acc_where_prefilter_right": float((d[~pf_wrong] == y[~pf_wrong]).mean()) if (~pf_wrong).any() else np.nan,
            "llm_accuracy": float((d == y).mean()), "fixes": int(fix.sum()), "breaks": int(brk.sum()),
            "net_fixes": int(fix.sum() - brk.sum()), "cost_change_if_all_routed": float((cost_llm - cost_pf).sum()),
            "fallback_rate": float(fallback.mean()), "vulnerable_rate": float(d.mean()),
            "auc_confidence_alone": float(roc_auc_score(y[ok], conf[ok].astype(float))) if ok.sum() > 10 and len(set(y[ok])) > 1 else np.nan,
            "auc_p_hat_cv": float(roc_auc_score(y, base_oof)), "auc_fusion_cv": float(roc_auc_score(y, fused)),
            "auc_gain": float(roc_auc_score(y, fused) - roc_auc_score(y, base_oof)),
            "auc_gain_ci95": bootstrap_gain(y, base_oof, fused, n_boot, seed),
        }
    return out


def report(dataset: str, sel: dict, args) -> dict:
    ids = sel["ids"]
    dec120 = decisions_120b(dataset, ids)
    out_dir = dataset_dir(dataset, "pilot")
    dec120.to_parquet(out_dir / "pilot120_decisions.parquet", index=False)
    called = dec120.groupby("id")["reason"].first()
    missing = int((dec120.groupby("id")["fallback"].min() == 1).sum())
    w = wide(dec120, "b120").join(wide(decisions_20b(dataset, ids), "b20"), how="left")
    router, ops = sel["router"], sel["ops"]
    reps = sorted(router["rep"].unique(), key=lambda r: int(r))
    per_rep = {}
    for rep in reps:
        part = router[(router["rep"] == rep) & router["id"].isin(set(ids))][["id", "label", "p_hat", "d_pf"]]
        frame = part.merge(w, left_on="id", right_index=True, how="left").reset_index(drop=True)
        theta, ratio = float(ops[rep][sel["op"]]["theta"]), float(ops[rep][sel["op"]]["cost_ratio"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            per_rep[rep] = evaluate_rep(frame, theta, ratio, ["b20", "b120"], args.cv_repeats, args.bootstrap, int(rep))
    def mean(path):
        vals = []
        for r in per_rep.values():
            v = r
            for key in path:
                v = v[key]
            vals.append(v)
        return float(np.nanmean(vals))
    summary = {"prefilter_wrong": mean(["prefilter_wrong"]), "auc_p_hat": mean(["auc_p_hat"])}
    for tag in ["b20", "b120"]:
        summary[tag] = {k: mean([tag, k]) for k in ["acc_where_prefilter_wrong", "acc_where_prefilter_right",
                                                    "llm_accuracy", "fixes", "breaks", "net_fixes", "auc_confidence_alone",
                                                    "auc_p_hat_cv", "auc_fusion_cv", "auc_gain", "fallback_rate",
                                                    "vulnerable_rate"]}
        summary[tag]["auc_gain_ci95_low"] = mean([tag, "auc_gain_ci95", 0])
        summary[tag]["auc_gain_ci95_high"] = mean([tag, "auc_gain_ci95", 1])
    s = summary["b120"]
    c1 = s["acc_where_prefilter_wrong"] > 0.5
    c2 = s["auc_gain"] >= args.min_auc_gain and s["auc_gain_ci95_low"] > 0
    verdict = {"C1_acc_where_prefilter_wrong_gt_0.5": bool(c1),
               f"C2_auc_gain_ge_{args.min_auc_gain}_and_ci_low_gt_0": bool(c2), "PASS": bool(c1 and c2)}
    result = {"dataset": dataset, "model": load_config("llm")["rq5_model"], "selection": {k: v for k, v in sel.items()
              if k not in {"ids", "router", "ops"}}, "n_functions": len(ids),
              "pass2_called_functions": int((called != "no_hypothesis").sum()), "functions_all_fallback": missing,
              "usd_120b": _pilot_spend(dataset, ids), "summary_mean_over_reps": summary, "verdict": verdict,
              "per_rep": per_rep}
    dump_json(out_dir / "pilot120_report.json", result)
    return result


def _pilot_spend(dataset: str, ids: list[str]) -> float:
    from ..llm.runner import load_index

    index = load_index(dataset)
    part = index[index["id"].isin(set(ids)) & index["stage"].isin(STAGES)]
    return float(part["cost_usd"].fillna(0).sum())


def markdown(results: list[dict]) -> str:
    lines = ["# Pilot gpt-oss-120b vs gpt-oss-20b (A-pool, functions nearest theta)", "",
             "Mean over prefilter reps. C1: 120b accuracy where the prefilter is wrong > 0.5. "
             "C2: CV fusion AUC gain over p_hat >= min gain and bootstrap 2.5% quantile > 0.", "",
             "| dataset | model | n | prefilter wrong | acc where PF wrong | acc where PF right | fixes | breaks | "
             "AUC conf alone | AUC p_hat (CV) | AUC fusion (CV) | gain [95% CI] |", "|" + "---|" * 12]
    for r in results:
        s = r["summary_mean_over_reps"]
        for tag, name in [("b20", "gpt-oss-20b (cached)"), ("b120", "gpt-oss-120b")]:
            t = s[tag]
            lines.append(f"| {r['dataset']} | {name} | {r['n_functions']} | {s['prefilter_wrong']:.0f} | "
                         f"{t['acc_where_prefilter_wrong']:.3f} | {t['acc_where_prefilter_right']:.3f} | {t['fixes']:.0f} | "
                         f"{t['breaks']:.0f} | {t['auc_confidence_alone']:.3f} | {t['auc_p_hat_cv']:.3f} | "
                         f"{t['auc_fusion_cv']:.3f} | {t['auc_gain']:+.3f} [{t['auc_gain_ci95_low']:+.3f}, "
                         f"{t['auc_gain_ci95_high']:+.3f}] |")
    lines += ["", "| dataset | C1 | C2 | PASS | 120b spend (USD) |", "|---|---|---|---|---|"]
    for r in results:
        v = list(r["verdict"].values())
        lines.append(f"| {r['dataset']} | {v[0]} | {v[1]} | **{v[2]}** | {r['usd_120b']:.4f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="devign,reveal")
    parser.add_argument("--n", type=int, default=None, help="functions per dataset (default 300; ReVeal 500)")
    parser.add_argument("--rep", default="1", help="prefilter rep used to pick the functions nearest theta")
    parser.add_argument("--max-usd", type=float, default=3.0, help="refuse to call the API if the estimate exceeds this")
    parser.add_argument("--min-auc-gain", type=float, default=0.02)
    parser.add_argument("--cv-repeats", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true", help="selection + cost estimate only, no API call")
    parser.add_argument("--report-only", action="store_true", help="no API call, rebuild the report from the cache")
    args = parser.parse_args()
    log_run_header("run_pilot_120b", args, ["llm", "decision", "experiments"])
    from ..llm.runner import check_frozen

    check_frozen(["pass1", "pass2_neutral", "pass2_refute"])       # same frozen prompts as the main 20b run
    cfg = load_config("llm")
    op = str(load_config("experiments")["main"]["operating_point"])
    datasets = parse_datasets(args.datasets)
    selections, estimates = {}, {}
    for dataset in datasets:
        n = args.n or DEFAULT_N.get(dataset, 300)
        sel = selection(dataset, op, args.rep, n)
        selections[dataset] = sel
        dump_json(dataset_dir(dataset, "pilot") / "pilot120_ids.json",
                  {"ids": sel["ids"], **{k: v for k, v in sel.items() if k not in {"ids", "router", "ops"}}})
        estimates[dataset] = estimate_usd(dataset, sel["ids"], cfg["rq5_model"])
        log.info("[%s] selected %d / %d A-pool functions nearest theta=%.3f (rep %s, op %s), p_hat in [%.3f, %.3f], "
                 "%d positives | cost: %s", dataset, len(sel["ids"]), sel["n_pool"], sel["theta"], args.rep, op,
                 *sel["p_hat_range"], sel["positives"], estimates[dataset])
    total = sum(e["estimated_usd_120b"] for e in estimates.values())
    log.info("estimated total for %s: %.3f USD (max-usd %.2f)", cfg["rq5_model"], total, args.max_usd)
    if args.dry_run:
        return
    if not args.report_only:
        if total > args.max_usd:
            raise SystemExit(f"estimated {total:.3f} USD > --max-usd {args.max_usd} - raise it explicitly to continue")
        for dataset in datasets:
            run_120b(dataset, selections[dataset]["ids"])
    results = [report(d, selections[d], args) for d in datasets]
    text = markdown(results)
    path = artifacts_dir() / "results" / "pilot120_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(text)
    log.info("written %s", path)


if __name__ == "__main__":
    main()
