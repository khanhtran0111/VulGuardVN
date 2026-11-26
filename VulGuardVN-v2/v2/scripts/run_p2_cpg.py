"""P2 - Joern / CPG for test + A-pool + dev set (D7). Resumable.

    python -m v2.scripts.run_p2_cpg --datasets all
    python -m v2.scripts.run_p2_cpg --datasets primevul --benchmark 2000     # P2 step 2: measure speed
    python -m v2.scripts.run_p2_cpg --datasets primevul --subset demo_bank   # optional: demo-bank slices
"""
from __future__ import annotations

import argparse
import json

from ..common import dump_json
from ..config import dataset_dir, get_logger, load_config, log_run_header, parse_datasets
from ..cpg.joern import facts_path, joern_version, load_facts, run_cpg
from ..data.io import llm_targets, load_records

log = get_logger("p2")


def stats(dataset: str) -> dict:
    records = load_records(dataset, ["id", "subset", "dev_set", "in_main"])
    targets = records[llm_targets(records)]
    facts = load_facts(dataset)
    rows = {}
    for name, mask in {"test": targets["subset"] == "test", "a_pool": targets["subset"] == "a_pool",
                       "dev_set": targets["dev_set"]}.items():
        ids = targets.loc[mask, "id"]
        have = [facts[i] for i in ids if i in facts]
        ok = sum(1 for f in have if f.get("ok"))
        rows[name] = {"targets": int(len(ids)), "processed": len(have), "ok": ok,
                      "parse_rate": ok / len(have) if have else None,
                      "with_sinks": sum(1 for f in have if f.get("ok") and f.get("sinks"))}
    errors: dict[str, int] = {}
    for payload in facts.values():
        if not payload.get("ok"):
            key = str(payload.get("error", "?")).split(":")[0][:40]
            errors[key] = errors.get(key, 0) + 1
    summary = {"dataset": dataset, "joern_version": joern_version(load_config("cpg")), "subsets": rows, "errors": errors}
    dump_json(dataset_dir(dataset, "cpg_cache") / "stats.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--benchmark", type=int, default=0, help="only run N functions and report speed")
    parser.add_argument("--subset", default="llm", choices=["llm", "dev", "demo_bank"],
                        help="llm = test + A-pool + dev set (default); dev = dev set only (prompt loop)")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    log_run_header("run_p2_cpg", args, ["cpg", "datasets"])
    cfg = load_config("cpg")
    if args.workers:
        cfg["workers"] = args.workers
    log.info("Joern version %s", joern_version(cfg))
    for dataset in parse_datasets(args.datasets):
        records = load_records(dataset, ["id", "code", "lang", "subset", "dev_set", "demo_bank"])
        mask = {"llm": llm_targets(records), "dev": records["dev_set"], "demo_bank": records["demo_bank"]}[args.subset]
        targets = records[mask]
        # dev set first (prompt development can start early), then A-pool, then test
        order = targets["dev_set"].map({True: 0, False: 1}) * 10 + targets["subset"].map({"a_pool": 1, "test": 2}).fillna(3)
        targets = targets.assign(_order=order).sort_values("_order", kind="stable").drop(columns="_order")
        result = run_cpg(dataset, targets, cfg, limit=args.benchmark or None)
        if args.benchmark:
            rate = result["ran"] / max(result["seconds"], 1e-6)
            log.info("[%s] benchmark: %d functions in %.0fs = %.2f fn/s -> %.1f h for %d targets", dataset,
                     result["ran"], result["seconds"], rate, len(targets) / max(rate, 1e-6) / 3600, len(targets))
        log.info("[%s] stats: %s", dataset, json.dumps(stats(dataset)["subsets"]))
        log.info("[%s] facts -> %s", dataset, facts_path(dataset))


if __name__ == "__main__":
    main()
