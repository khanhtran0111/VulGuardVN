"""P1 - Data: load, de-duplicate, split, A-pool/B, dev set, demo bank, data_card.md.

    python -m v2.scripts.run_p1_data --datasets all
    python -m v2.scripts.run_p1_data --datasets primevul,devign --smoke      # small subset for testing
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from ..common import atomic_to_parquet, dump_json
from ..config import artifacts_dir, dataset_dir, get_logger, load_config, log_run_header, parse_datasets, resolve_project_path
from ..data.io import RECORD_COLUMNS
from ..data.loaders import load_dataset
from ..data.splits import (assign_demo_bank, assign_dev_set, assign_subsets, dedup_and_split, leakage_check,
                           split_table, subsample_for_smoke)

log = get_logger("p1")
SMOKE_SIZES = {"train": 1500, "valid": 700, "test": 400}


def build(name: str, global_cfg: dict, smoke: bool) -> dict:
    cfg = global_cfg["datasets"][name]
    data_root = resolve_project_path(global_cfg["data_root"])
    log.info("[%s] loading raw data from %s", name, data_root)
    records = list(load_dataset(name, cfg, data_root))
    frame, card = dedup_and_split(records, name, cfg, global_cfg)
    del records
    if smoke:
        frame = subsample_for_smoke(frame, SMOKE_SIZES, int(global_cfg["split_seed"]))
        card["smoke"] = SMOKE_SIZES
    frame = assign_subsets(frame, cfg, global_cfg, card)
    dev_cfg = global_cfg.get("dev_set", {})
    frame["dev_set"] = False
    if dev_cfg.get("dataset") == name:
        frame["dev_set"] = assign_dev_set(frame, dev_cfg, int(global_cfg["split_seed"]))
    frame["demo_bank"] = assign_demo_bank(frame, bool(cfg.get("has_pairs")))
    frame["commit_date"] = None   # not available in the released data (see data_card notes)
    frame["dataset"] = name
    leaks = leakage_check(frame)
    card.update({
        "splits": split_table(frame),
        "leakage_hashes_across_splits": leaks,
        "dev_set": {"functions": int(frame["dev_set"].sum()), "pairs": int(frame.loc[frame["dev_set"], "pair_id"].nunique())},
        "demo_bank": {"records": int(frame["demo_bank"].sum()), "pairs": int(frame.loc[frame["demo_bank"], "pair_id"].nunique())},
        "subsets": frame["subset"].value_counts().to_dict(),
        "with_cwe_vulnerable": int(frame.loc[frame["label"] == 1, "cwe"].notna().sum()),
    })
    if leaks:
        raise RuntimeError(f"[{name}] leakage check failed: {leaks} hashes appear in more than one split")
    out = dataset_dir(name)
    atomic_to_parquet(frame[RECORD_COLUMNS], out / "records.parquet")
    dump_json(out / "data_card.json", card)
    log.info("[%s] saved %d records -> %s", name, len(frame), out / "records.parquet")
    return card


def render_card(cards: list[dict]) -> str:
    lines = [f"# Data card - VulGuard v2", "", f"Generated {dt.date.today().isoformat()}.", "",
             "Counts are after de-duplication (comments/whitespace-insensitive hash; conflicting labels dropped).",
             "`pairs` = vulnerable/patched pairs with both members in the split. Aux records are pair partners that",
             "are not part of the main split (used only for pair metrics / training pairs).", ""]
    for card in cards:
        lines += [f"## {card['dataset']}", "", "| Split | Functions | Vulnerable | Rate | Pairs | Aux |", "|---|---|---|---|---|---|"]
        for row in card["splits"]:
            lines.append(f"| {row['split']} | {row['functions']:,} | {row['vulnerable']:,} | {row['rate']:.2%} | "
                         f"{row['pairs']:,} | {row['aux_records']:,} |")
        a, b = card["a_pool"], card["b"]
        alpha = f"{b['alpha_min']:.4%}" if b["alpha_min"] else "n/a"
        lines += ["", f"- A-pool: {a['size']:,} functions, {a['n_pos']:,} vulnerable (sample rate {a['rate_a']:.2%}, "
                  f"true rate {a['rate_valid']:.2%}; weights pos={a['weight_pos']:.3f}, neg={a['weight_neg']:.3f})",
                  f"- B (conformal calibration): {b['size']:,} functions, {b['n_pos']:,} vulnerable -> smallest alpha = {alpha}",
                  f"- Dev set: {card['dev_set']['functions']} functions ({card['dev_set']['pairs']} pairs); "
                  f"demo bank: {card['demo_bank']['records']:,} records ({card['demo_bank']['pairs']:,} pairs)",
                  f"- Dropped: too long {card['dropped_too_long']}, empty {card['dropped_empty']}; dedup: `{card['dedup']}`",
                  f"- Leakage check (hashes in >1 split): **{card['leakage_hashes_across_splits']}**",
                  f"- Vulnerable functions with CWE: {card['with_cwe_vulnerable']:,}", ""]
    lines += ["Note: `commit_date` is not present in the released PrimeVul/Big-Vul/Devign/ReVeal files; the PrimeVul",
              "split itself is chronological, so RQ2 compares it against a random re-split instead of dates.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--smoke", action="store_true", help="small stratified subset for smoke tests")
    parser.add_argument("--force", action="store_true", help="rebuild records.parquet even if it exists "
                        "(invalidates feature stores, CPG facts and models built on the old records)")
    args = parser.parse_args()
    log_run_header("run_p1_data", args, ["datasets"])
    global_cfg = load_config("datasets")
    cards = []
    for name in parse_datasets(args.datasets):
        if (dataset_dir(name) / "records.parquet").exists() and not args.force:
            log.info("[%s] records.parquet exists - skip (use --force to rebuild)", name)
            continue
        cards.append(build(name, global_cfg, args.smoke))
    existing = {}
    for path in sorted(artifacts_dir().glob("*/data_card.json")):
        import json
        payload = json.loads(path.read_text(encoding="utf-8"))
        existing[payload["dataset"]] = payload
    for card in cards:
        existing[card["dataset"]] = card
    order = [name for name in load_config("datasets")["datasets"] if name in existing]
    (artifacts_dir() / "data_card.md").write_text(render_card([existing[n] for n in order]), encoding="utf-8")
    log.info("data card -> %s", artifacts_dir() / "data_card.md")


if __name__ == "__main__":
    main()
