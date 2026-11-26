"""P5 - CPG verification of every cached Pass-2 output, both modes (offline, no API).

Writes artifacts/{ds}/llm/decisions.parquet with d_llm(x, mode) for every function of A-pool and test:
  id, variant (main | a7 | a8 | rq5 | zeroshot), mode (neutral | refute | single), d, fallback, reason, v1, v2, v3,
  verdict, guard_line, d_raw, has_hyp, tokens_in, tokens_out, reasoning_tokens, usd, llm_calls
plus verify_stats.json and manual_review_30.md (P5 step 4).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from ..common import dump_json, loads_or_none, number_code
from ..config import dataset_dir, get_logger, log_run_header, parse_datasets
from ..cpg.joern import load_facts
from ..data.io import load_records
from ..llm.client import mock_mode
from ..llm.runner import load_index
from ..verify.cpg_decide import decide, decide_without_hypothesis

log = get_logger("p5")
VARIANTS = {"main": "pass2", "a7": "a7", "a8": "a8", "rq5": "rq5"}


def decisions_path(dataset: str):
    return dataset_dir(dataset, "llm") / ("decisions.mock.parquet" if mock_mode() else "decisions.parquet")


def build(dataset: str) -> pd.DataFrame:
    records = load_records(dataset, ["id", "code", "label", "subset", "dev_set", "in_main"])
    targets = records[records["subset"].isin(["a_pool", "test"]) | records["dev_set"]]
    index = load_index(dataset)
    if index.empty:
        raise SystemExit(f"[{dataset}] no LLM outputs - run run_p4_llm first")
    entry = {(r.id, r.stage): r for r in index.itertuples(index=False)}
    facts = load_facts(dataset, set(targets["id"]))
    code_of = dict(zip(targets["id"], targets["code"]))
    from ..config import load_config
    refute_no_hyp = int(load_config("decision").get("refute_without_hypothesis", 1))
    rows = []
    for record_id in targets["id"]:
        p1_row = entry.get((record_id, "pass1"))
        if p1_row is None:
            continue
        p1 = {"parsed": loads_or_none(p1_row.parsed)}
        f = facts.get(record_id)
        joern = {"n_cpg_nodes": (f or {}).get("n_nodes"), "cyclomatic": (f or {}).get("cyclomatic"),
                 "n_candidate_sinks": len((f or {}).get("sinks") or []) if f and f.get("ok") else None}
        for variant, prefix in VARIANTS.items():
            for mode in ("neutral", "refute"):
                stage = f"pass2_{mode}" if prefix == "pass2" else f"{prefix}_{mode}"
                p2_row = entry.get((record_id, stage))
                if variant != "main" and p2_row is None:
                    continue
                p2 = {"parsed": loads_or_none(p2_row.parsed)} if p2_row is not None and loads_or_none(p2_row.parsed) is not None else None
                out = decide_without_hypothesis(mode, p2, f) if variant == "a7" else decide(mode, p1, p2, f, code_of[record_id], refute_no_hyp)
                calls = ([] if variant == "a7" else [p1_row]) + ([p2_row] if p2_row is not None else [])
                rows.append({"id": record_id, "variant": variant, **out, **joern,
                             "tokens_in": int(sum(c.prompt_tokens for c in calls)),
                             "tokens_out": int(sum(c.completion_tokens for c in calls)),
                             "reasoning_tokens": int(sum(c.reasoning_tokens for c in calls)),
                             "usd": float(sum(c.cost_usd or 0.0 for c in calls)), "llm_calls": len(calls)})
        z = entry.get((record_id, "zeroshot"))
        if z is not None:
            parsed = loads_or_none(z.parsed)
            rows.append({"id": record_id, "variant": "zeroshot", "mode": "single",
                         "d": int(bool(parsed) and parsed.get("verdict") == "vulnerable"), "fallback": int(parsed is None),
                         "reason": "zeroshot" if parsed else "invalid", "verdict": (parsed or {}).get("verdict"),
                         "d_raw": int(bool(parsed) and parsed.get("verdict") == "vulnerable"), **joern,
                         "tokens_in": int(z.prompt_tokens), "tokens_out": int(z.completion_tokens),
                         "reasoning_tokens": int(z.reasoning_tokens), "usd": float(z.cost_usd or 0.0), "llm_calls": 1})
    frame = pd.DataFrame(rows)
    frame = frame.merge(records[["id", "label", "subset", "in_main", "dev_set"]], on="id", how="left")
    frame.to_parquet(decisions_path(dataset), index=False)
    return frame


def stats(dataset: str, frame: pd.DataFrame) -> dict:
    out = {}
    main = frame[(frame["variant"] == "main") & frame["subset"].isin(["a_pool", "test"])]
    for mode, part in main.groupby("mode"):
        y = part["label"].to_numpy()
        with_cpg = part["d"].fillna(-1).to_numpy()
        raw = part["d_raw"].fillna(-1).to_numpy()
        valid = (with_cpg >= 0) & (raw >= 0)
        def fp_fn(pred):
            return {"FP": int(((pred == 1) & (y == 0) & valid).sum()), "FN": int(((pred == 0) & (y == 1) & valid).sum())}
        out[mode] = {
            "functions": int(len(part)), "joern_error_rate": float((part["reason"] == "joern_error").mean()),
            "fallback_rate": float(part["fallback"].mean()), "reasons": part["reason"].value_counts().to_dict(),
            "rejected_for_missing_evidence": float(part["reason"].isin(["cpg_rejected", "guard_rejected"]).mean()),
            "with_verification": fp_fn(with_cpg), "without_verification_A10": fp_fn(raw),
        }
    dump_json(dataset_dir(dataset, "llm") / "verify_stats.json", out)
    return out


def manual_review(dataset: str, frame: pd.DataFrame, n: int = 30) -> None:
    records = load_records(dataset, ["id", "code"]).set_index("id")
    index = load_index(dataset)
    hyp_of = {r.id: r.hyp for r in index[index["stage"] == "pass2_neutral"].itertuples()}
    rejected = frame[(frame["variant"] == "main") & frame["reason"].isin(["cpg_rejected", "guard_rejected"])]
    sample = rejected.sample(min(n, len(rejected)), random_state=0) if len(rejected) else rejected
    lines = [f"# P5 step 4 - manual review of {len(sample)} CPG rejections ({dataset})", "",
             "For each case check whether the V1-V3 query decision is right. Mark: [ ] correct  [ ] wrong", ""]
    for row in sample.itertuples():
        hyp = loads_or_none(hyp_of.get(row.id))
        code = records.loc[row.id, "code"]
        line = int((hyp or {}).get("sink_line") or 1)
        lines += [f"## {row.id} (label={row.label}, mode={row.mode}, reason={row.reason})",
                  f"- hypothesis: `{json.dumps(hyp)}`", f"- LLM verdict: {row.verdict}, guard_line: {row.guard_line}",
                  f"- V1={row.v1} V2={row.v2} V3={row.v3}", "", "```c",
                  number_code(code, range(max(1, line - 12), line + 6)), "```", "", "- [ ] correct  - [ ] wrong", ""]
    (dataset_dir(dataset, "llm") / "manual_review_30.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    if args.mock:
        os.environ["VG2_LLM_MOCK"] = "1"
    log_run_header("run_p5_verify", args, ["llm", "cpg"])
    for dataset in parse_datasets(args.datasets):
        frame = build(dataset)
        summary = stats(dataset, frame)
        manual_review(dataset, frame)
        log.info("[%s] decisions: %d rows | %s", dataset, len(frame),
                 {m: {"joern_err": round(s["joern_error_rate"], 4), "fallback": round(s["fallback_rate"], 4)} for m, s in summary.items()})


if __name__ == "__main__":
    main()
