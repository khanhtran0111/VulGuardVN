"""P4 - LLM layer (OpenRouter, gpt-oss-20b via the fixed provider). Every call is cached; re-running is free.

    python -m v2.scripts.run_p4_llm probe                              # P0 step 4: one test request
    python -m v2.scripts.run_p4_llm dev                                # P4 step 4: prompt loop on the dev set (300)
    python -m v2.scripts.run_p4_llm freeze                             # P4 step 5: freeze prompt hashes (D6)
    python -m v2.scripts.run_p4_llm run --datasets all --subset a_pool # then --subset test (includes zero-shot B3)
    python -m v2.scripts.run_p4_llm ablations --datasets primevul      # A7, A8 on the API subset
    python -m v2.scripts.run_p4_llm rq5 --datasets primevul            # verifier gpt-oss-120b on the API subset
    python -m v2.scripts.run_p4_llm uncertainty --datasets primevul    # P4 step 7: 200 x 3 repeated calls
    python -m v2.scripts.run_p4_llm status                             # tokens / cost per dataset and stage
Add --mock for an offline dry run (separate cache), --limit N to cap the number of functions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

import numpy as np
import pandas as pd

from ..common import dump_json, loads_or_none
from ..config import CONFIG_DIR, artifacts_dir, dataset_dir, get_logger, load_config, log_run_header, parse_datasets

log = get_logger("p4")


def _ids(records: pd.DataFrame, subset: str, limit: int | None) -> list[str]:
    if subset == "dev":
        ids = records.loc[records["dev_set"], "id"]
    elif subset == "test":
        ids = records.loc[records["subset"] == "test", "id"]            # includes pair partners (pair metrics)
    else:
        ids = records.loc[records["subset"] == subset, "id"]
    ids = ids.tolist()
    return ids[:limit] if limit else ids


def api_subset(records: pd.DataFrame, n_random: int, seed: int = 2026) -> list[str]:
    """PrimeVul paired test (both members) + n random test functions (§7.4)."""
    test = records[records["subset"] == "test"]
    paired = test.loc[test["pair_id"].notna(), "id"].tolist()
    rest = test.loc[test["pair_id"].isna() & test["in_main"], "id"].to_numpy()
    rng = np.random.default_rng(seed)
    extra = rng.choice(rest, size=min(n_random, len(rest)), replace=False).tolist()
    return paired + extra


class PreflightError(SystemExit):
    pass


# cost of a command relative to one full v2 LLM layer call chain (Pass 1 + 2 x Pass 2) per function
COST_FACTOR = {"pass1": 1.0, "a7_neutral": 4 / 3, "rq5_neutral": 2 / 3 * 3.0, "repeat1_neutral": 0.0}


def preflight(dataset: str, ids: list[str], *, with_zeroshot: bool, allow_unfrozen: bool, mock: bool,
              stages: list[str], check_stage: str = "pass1", need_features: bool = True) -> dict:
    """Checks before spending money: CPG facts, feature store, frozen prompts, API key; prints a cost estimate."""
    from ..cpg.joern import facts_path, load_done_ids
    from ..llm.client import total_spend
    from ..llm.runner import FrozenPromptError, check_frozen, load_index

    problems = []
    done = load_done_ids(facts_path(dataset))
    coverage = sum(1 for i in ids if i in done) / max(len(ids), 1)
    if coverage < 0.99:
        problems.append(f"Joern facts cover only {coverage:.1%} of the {len(ids)} functions - run P2 first "
                        f"(`DATASETS={dataset} ./run_all.sh local` or run_p2_cpg --datasets {dataset})")
    if need_features and not (dataset_dir(dataset, "features") / "unixcoder.npy").exists():
        problems.append("UniXcoder feature store missing (needed for retrieval / max_sim) - run run_p3_prefilter --stage features")
    if not allow_unfrozen:
        try:
            check_frozen(stages)
        except FrozenPromptError as exc:
            problems.append(str(exc))
    cfg = load_config("llm")
    if not mock and not os.getenv(cfg.get("api_key_env", "OPENROUTER_API_KEY"), "").strip():
        problems.append("OPENROUTER_API_KEY is empty in VulGuardVN-v2/.env (save the file after editing)")
    index = load_index(dataset)
    have = set(index.loc[index["stage"] == check_stage, "id"]) if not index.empty else set()
    todo = [i for i in ids if i not in have]
    per_fn = 0.00077                       # plan §8 with the CoreWeave price (3 calls, ~11k in / 3.4k out)
    source = "plan §8 assumption"
    report = dataset_dir(load_config("datasets")["dev_set"]["dataset"], "llm") / "dev_report.json"
    if report.exists():
        measured = json.loads(report.read_text()).get("per_function_llm_layer", {}).get("usd")
        if measured:
            per_fn, source = float(measured), "measured on the dev set"
    zeroshot = 0.00017 if with_zeroshot else 0.0
    if check_stage == "repeat1_neutral":            # uncertainty: 200 functions x 3 x (Pass 1 + Pass 2 neutral)
        estimate = min(len(todo), int(cfg["subsets"]["uncertainty_n"])) * int(cfg["subsets"]["uncertainty_repeats"]) * per_fn * 2 / 3
    else:
        estimate = len(todo) * (per_fn * COST_FACTOR.get(check_stage, 1.0) + zeroshot)
    spent = 0.0 if mock else total_spend()
    info = {"functions": len(ids), "not_yet_called": len(todo), "estimated_usd": round(estimate, 3),
            "per_function_usd": per_fn, "estimate_source": source, "spent_so_far_usd": round(spent, 3),
            "spend_cap_usd": cfg.get("spend_cap_usd"), "cpg_coverage": round(coverage, 4)}
    log.info("[%s] preflight: %s", dataset, info)
    if spent + estimate > float(cfg.get("spend_cap_usd", 95)):
        problems.append(f"estimated total {spent + estimate:.2f} USD exceeds spend_cap_usd={cfg.get('spend_cap_usd')}")
    if problems:
        for problem in problems:
            log.info("[%s] PREFLIGHT FAILED: %s", dataset, problem)
        raise PreflightError(2)
    return info


def cmd_freeze(if_ok: bool = False, mock: bool = False) -> None:
    from ..llm.prompts import current_hashes

    if if_ok:   # P4 step 4 decision point: only freeze if v2 beats zero-shot on the dev set
        dataset = load_config("datasets")["dev_set"]["dataset"]
        report = dataset_dir(dataset, "llm") / ("dev_report.mock.json" if mock else "dev_report.json")
        if not report.exists():
            log.info("FREEZE REFUSED: %s missing - run the dev loop first", report)
            raise SystemExit(3)
        decision = json.loads(report.read_text())["decision_point_step4"]
        if not decision.get("ok_to_freeze"):
            log.info("FREEZE REFUSED: v2 pair accuracy %.3f <= zero-shot %.3f on the dev set (P4 step 4) - revise the "
                     "prompts / verification and rerun `dev`", decision["v2_pair_accuracy"], decision["zeroshot_pair_accuracy"])
            raise SystemExit(3)
        log.info("dev set: v2 %.3f > zero-shot %.3f -> freezing", decision["v2_pair_accuracy"], decision["zeroshot_pair_accuracy"])

    hashes = current_hashes()
    path = CONFIG_DIR / "llm.yaml"
    text = path.read_text(encoding="utf-8")
    line = "prompt_hashes: {" + ", ".join(f"{k}: {v}" for k, v in hashes.items()) + "}"
    text = re.sub(r"^prompt_hashes:.*$", line, text, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")
    log.info("frozen prompt hashes -> %s: %s", path, hashes)


def cmd_probe() -> None:
    from ..llm.client import LLMClient, run_requests
    from ..llm.prompts import prompt_hash, schema

    client = LLMClient("probe")
    messages = [{"role": "system", "content": "Answer with JSON only."},
                {"role": "user", "content": "Is `strcpy(buf, argv[1]);` with `char buf[8];` vulnerable? "
                                            'Return {"verdict": ..., "evidence_lines": [1], "confidence": ...}'}]
    job = {"record_id": "probe", "stage": "zeroshot", "prompt_hash": prompt_hash("zeroshot"), "messages": messages,
           "schema": schema("zeroshot")}
    entry = asyncio.run(run_requests(client, [job], 1))[0]
    log.info("probe: status=%s provider=%s usage=%s cost=%.6f latency=%.1fs", entry.get("status"), entry.get("provider"),
             entry.get("usage"), entry.get("cost_usd") or 0.0, entry.get("latency_s") or 0.0)
    log.info("content: %s", (entry.get("content") or "")[:300])
    log.info("cached in %s", client.cache.path)


def dev_report(layer, ids: list[str]) -> dict:
    from ..llm.runner import load_index
    from ..verify.cpg_decide import decide

    index = load_index(layer.dataset)
    index = index[index["id"].isin(ids)]
    parsed = {(r.id, r.stage): {"parsed": loads_or_none(r.parsed)} for r in index.itertuples()}
    facts = layer.facts(ids)
    records = layer.records.loc[ids]
    preds: dict[str, dict[str, int]] = {"zeroshot": {}, "v2_neutral": {}, "v2_refute": {}, "v2_neutral_no_cpg": {}}
    for record_id in ids:
        z = parsed.get((record_id, "zeroshot"), {}).get("parsed")
        preds["zeroshot"][record_id] = int(bool(z) and z.get("verdict") == "vulnerable")
        p1 = parsed.get((record_id, "pass1"))
        for mode in ("neutral", "refute"):
            out = decide(mode, p1, parsed.get((record_id, f"pass2_{mode}")), facts.get(record_id),
                         records.loc[record_id, "code"], int(load_config("decision").get("refute_without_hypothesis", 1)))
            value = out["d"] if out["d"] is not None else (out["d_raw"] if out["d_raw"] is not None else 0)
            preds[f"v2_{mode}"][record_id] = int(value)
            if mode == "neutral":
                preds["v2_neutral_no_cpg"][record_id] = int(out["d_raw"] or 0)
    pairs = records[records["pair_id"].notna()].groupby("pair_id")
    report = {"n_functions": len(ids), "n_pairs": int(records["pair_id"].nunique()), "methods": {}}
    for name, pred in preds.items():
        pc = pv = pb = pr = 0
        for _, group in pairs:
            roles = dict(zip(group["pair_role"], group["id"]))
            if len(roles) < 2:
                continue
            a, b = pred[roles["vulnerable"]], pred[roles["patched"]]
            pc += a == 1 and b == 0
            pv += a == 1 and b == 1
            pb += a == 0 and b == 0
            pr += a == 0 and b == 1
        n = max(pc + pv + pb + pr, 1)
        y = records["label"].to_numpy()
        p = np.array([pred[i] for i in records.index])
        report["methods"][name] = {"P-C": pc / n, "P-V": pv / n, "P-B": pb / n, "P-R": pr / n,
                                   "accuracy": float((p == y).mean()), "recall": float(p[y == 1].mean()) if (y == 1).any() else None}
    validity, tokens = {}, {}
    for stage, part in index.groupby("stage"):
        validity[stage] = float((part["status"] == "ok").mean())
        tokens[stage] = {"calls": int(len(part)), "prompt": float(part["prompt_tokens"].mean()),
                         "completion": float(part["completion_tokens"].mean()),
                         "reasoning": float(part["reasoning_tokens"].mean()), "usd_total": float(part["cost_usd"].sum())}
    report["json_valid_rate"] = validity
    report["tokens_per_call"] = tokens
    per_fn = index[index["stage"].isin(["pass1", "pass2_neutral", "pass2_refute"])].groupby("id")
    report["per_function_llm_layer"] = {
        "tokens_in": float(per_fn["prompt_tokens"].sum().mean()) if len(per_fn) else 0.0,
        "tokens_out": float((per_fn["completion_tokens"].sum()).mean()) if len(per_fn) else 0.0,
        "usd": float(per_fn["cost_usd"].sum().mean()) if len(per_fn) else 0.0}
    v2_best = max(report["methods"]["v2_neutral"]["P-C"], report["methods"]["v2_refute"]["P-C"])
    report["decision_point_step4"] = {"v2_pair_accuracy": v2_best, "zeroshot_pair_accuracy": report["methods"]["zeroshot"]["P-C"],
                                      "ok_to_freeze": v2_best > report["methods"]["zeroshot"]["P-C"]}
    return report


def cmd_status() -> None:
    from ..llm.runner import load_index

    rows = []
    for path in sorted(artifacts_dir().glob("*/llm/index*.parquet")):
        dataset = path.parts[-3]
        index = pd.read_parquet(path)
        for stage, part in index.groupby("stage"):
            rows.append({"dataset": dataset, "file": path.name, "stage": stage, "calls": len(part),
                         "valid": float((part["status"] == "ok").mean()), "prompt_tokens": int(part["prompt_tokens"].sum()),
                         "completion_tokens": int(part["completion_tokens"].sum()),
                         "reasoning_tokens": int(part["reasoning_tokens"].sum()), "usd": float(part["cost_usd"].sum())})
    frame = pd.DataFrame(rows)
    if frame.empty:
        log.info("no LLM calls yet")
        return
    frame.to_csv(artifacts_dir() / "llm_cost_summary.csv", index=False)
    print(frame.to_string(index=False))
    print(f"\nTOTAL USD (real calls): {frame.loc[~frame['file'].str.contains('mock'), 'usd'].sum():.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["probe", "dev", "freeze", "run", "ablations", "rq5", "uncertainty", "status"])
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--subset", default="a_pool", choices=["a_pool", "test", "dev"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true", help="offline deterministic LLM (separate cache)")
    parser.add_argument("--allow-unfrozen", action="store_true", help="smoke tests only: skip the D6 freeze check")
    parser.add_argument("--dry-run", action="store_true", help="only run the preflight checks and print the cost estimate")
    parser.add_argument("--if-ok", action="store_true", help="freeze: only if the dev report says ok_to_freeze")
    args = parser.parse_args()
    if args.mock:
        os.environ["VG2_LLM_MOCK"] = "1"
    log_run_header("run_p4_llm", args, ["llm", "cpg", "datasets"])
    if args.command == "freeze":
        return cmd_freeze(args.if_ok, args.mock)
    if args.command == "probe":
        return cmd_probe()
    if args.command == "status":
        return cmd_status()

    from ..llm.runner import LLMLayer, check_frozen, load_index

    from ..data.io import load_records

    cfg = load_config("llm")
    if args.command == "dev":
        dataset = load_config("datasets")["dev_set"]["dataset"]
        records = load_records(dataset, ["id", "subset", "dev_set", "pair_id", "in_main"])
        ids = _ids(records, "dev", args.limit)
        preflight(dataset, ids, with_zeroshot=True, allow_unfrozen=True, mock=args.mock, stages=[],
                  need_features=False)
        if args.dry_run:
            return
        layer = LLMLayer(dataset)
        layer.run_main(ids, with_zeroshot=True)
        report = dev_report(layer, ids)
        dump_json(dataset_dir(dataset, "llm") / ("dev_report.mock.json" if args.mock else "dev_report.json"), report)
        print(json.dumps(report, indent=2))
        return

    for dataset in parse_datasets(args.datasets):
        if args.command == "run":
            stages = ["pass1", "pass2_neutral", "pass2_refute"] + (["zeroshot"] if args.subset == "test" else [])
            records = load_records(dataset, ["id", "subset", "dev_set", "pair_id", "in_main"])
            ids = _ids(records, args.subset, args.limit)
            preflight(dataset, ids, with_zeroshot=args.subset == "test", allow_unfrozen=args.allow_unfrozen,
                      mock=args.mock, stages=stages)
            if args.dry_run:
                continue
            layer = LLMLayer(dataset)
            summary = layer.run_main(ids, with_zeroshot=args.subset == "test")
            targets = layer.records.loc[layer.records["subset"].isin(["a_pool", "test"]), "id"].tolist()
            layer.max_similarity(targets)
            log.info("[%s] %s done: %s", dataset, args.subset, summary)
        elif args.command in {"ablations", "rq5", "uncertainty"}:
            records = load_records(dataset, ["id", "subset", "dev_set", "pair_id", "in_main"])
            ids = api_subset(records, int(cfg["subsets"]["api_subset_random"]))
            if args.limit:
                ids = ids[: args.limit]
            index = load_index(dataset)
            have_p1 = set(index.loc[index["stage"] == "pass1", "id"]) if not index.empty else set()
            if sum(i in have_p1 for i in ids) < len(ids):
                log.info("[%s] PREFLIGHT FAILED: run `run_p4_llm run --subset test` first (%s needs the main Pass 1)",
                         dataset, args.command)
                raise SystemExit(2)
            check = {"ablations": "a7_neutral", "rq5": "rq5_neutral", "uncertainty": "repeat1_neutral"}[args.command]
            preflight(dataset, ids, with_zeroshot=False, allow_unfrozen=args.allow_unfrozen, mock=args.mock,
                      stages=["pass1", "pass2_neutral", "pass2_refute"], check_stage=check)
            if args.dry_run:
                continue
            layer = LLMLayer(dataset)
            p1 = layer.pass1_results(ids)
            if args.command == "ablations":
                layer.pass2(ids, p1, prefix="a7", use_hypothesis=False)          # A7: no Pass 1
                layer.pass2(ids, p1, prefix="a8", single_demo=True)              # A8: single demo instead of a pair
            elif args.command == "rq5":
                big = LLMLayer(dataset, model=cfg["rq5_model"], provider=dict(cfg["rq5_provider"]))
                big._facts, big._contexts = layer._facts, layer._contexts
                big.pass2(ids, p1, prefix="rq5")
            else:
                rng = np.random.default_rng(7)
                with_hyp = [i for i in ids if (p1.get(i, {}).get("parsed") or {}).get("hypotheses")]
                sample = rng.choice(with_hyp, size=min(int(cfg["subsets"]["uncertainty_n"]), len(with_hyp)), replace=False).tolist()
                flips = uncertainty(layer, sample, p1, int(cfg["subsets"]["uncertainty_repeats"]))
                dump_json(dataset_dir(dataset, "llm") / "uncertainty_report.json", flips)
                print(json.dumps(flips, indent=2))


def uncertainty(layer, ids: list[str], p1: dict, repeats: int) -> dict:
    """P4 step 7: repeat Pass 1 and Pass 2 (neutral) and report how often the conclusion changes."""
    base = layer.pass2(ids, p1, modes=("neutral",))["pass2_neutral"]
    verdicts = {i: [(base.get(i, {}).get("parsed") or {}).get("verdict")] for i in ids}
    sinks = {i: [((p1[i].get("parsed") or {}).get("hypotheses") or [{}])[0].get("sink_line")] for i in ids}
    for k in range(1, repeats + 1):
        rp1 = layer.pass1(ids, stage=f"repeat{k}_pass1", extra_params={"repeat": k})
        for i in ids:
            sinks[i].append((((rp1.get(i) or {}).get("parsed") or {}).get("hypotheses") or [{}])[0].get("sink_line"))
        rp2 = layer.pass2(ids, p1, modes=("neutral",), prefix=f"repeat{k}", extra_params={"repeat": k})
        for i in ids:
            verdicts[i].append(((rp2.get(f"repeat{k}_neutral", {}).get(i) or {}).get("parsed") or {}).get("verdict"))
    changed_verdict = sum(1 for v in verdicts.values() if len(set(v)) > 1)
    changed_sink = sum(1 for v in sinks.values() if len(set(v)) > 1)
    return {"n": len(ids), "repeats": repeats, "pass2_verdict_changed_rate": changed_verdict / max(len(ids), 1),
            "pass1_top_sink_changed_rate": changed_sink / max(len(ids), 1)}


if __name__ == "__main__":
    main()
