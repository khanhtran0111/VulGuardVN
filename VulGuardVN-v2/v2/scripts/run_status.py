"""Status of the pipeline per dataset: what is done, what is missing, and the next command to run.

    python -m v2.scripts.run_status --datasets all
"""
from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from ..config import artifacts_dir, load_config, parse_datasets
from ..llm.prompts import current_hashes

OK, PART, MISS = "OK  ", "PART", "--  "


def _mark(done: bool, partial: bool = False) -> str:
    return OK if done else (PART if partial else MISS)


def dataset_status(dataset: str, mock: bool) -> tuple[list[str], str | None]:
    lines, next_step = [], None
    root = artifacts_dir() / dataset
    rec_path = root / "records.parquet"
    if not rec_path.exists():
        return [f"  {MISS} P1 records"], f"./run_dataset.sh {dataset} prep"
    records = pd.read_parquet(rec_path, columns=["id", "subset", "dev_set", "in_main", "label"])
    targets = records[records["subset"].isin(["test", "a_pool"]) | records["dev_set"]]
    counts = records["subset"].value_counts().to_dict()
    lines.append(f"  {OK} P1 records: {len(records):,} (test {counts.get('test', 0):,}, A {counts.get('a_pool', 0):,}, "
                 f"B {counts.get('b', 0):,}, train_fit {counts.get('train_fit', 0):,})")

    facts = root / "cpg_cache" / "facts.jsonl"
    latest = {}
    if facts.exists():
        with facts.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    payload = json.loads(line)
                    latest[payload["id"]] = bool(payload.get("ok"))
                except Exception:
                    continue
    covered = sum(1 for i in targets["id"] if i in latest)
    parsed = sum(1 for i in targets["id"] if latest.get(i))
    cov = covered / max(len(targets), 1)
    lines.append(f"  {_mark(cov >= 0.999, covered > 0)} P2 Joern: {covered:,}/{len(targets):,} functions processed, "
                 f"parse rate {parsed / max(covered, 1):.1%}")
    if cov < 0.999 and next_step is None:
        next_step = f"./run_dataset.sh {dataset} prep"

    fdir = root / "features"
    heuristic = all((fdir / n).exists() for n in ["tokens.npy", "ast.npy", "numeric.npy", "ids.parquet", "vocab.json", "feature_names.json"])
    unix = (fdir / "unixcoder.npy").exists()
    chunks = len(list((fdir / "unixcoder_chunks").glob("*.npy"))) if (fdir / "unixcoder_chunks").exists() else 0
    lines.append(f"  {_mark(heuristic and unix, heuristic or chunks > 0)} P3 features: heuristic={'yes' if heuristic else 'no'}, "
                 f"UniXcoder={'yes' if unix else f'no ({chunks} chunks done)'}")
    if not (heuristic and unix) and next_step is None:
        next_step = f"./run_dataset.sh {dataset} prep"
    views = load_config("prefilter")["model"].get("views", [])
    if "finetune" in views:
        ft_ready = (fdir / "finetune_logit.npy").exists()
        folds_done = len(list((root / "finetune").glob("fold*/pred.npz"))) if (root / "finetune").exists() else 0
        k = int(load_config("prefilter")["finetune"]["folds"])
        b7 = root / "baselines" / "b7_summary.json"
        b7_text = f", B7 test F1 {json.loads(b7.read_text())['test_f1']:.4f}" if b7.exists() else ""
        lines.append(f"  {_mark(ft_ready, folds_done > 0)} P3 fine-tune (view + B7): {folds_done}/{k} folds{b7_text}")
        if not ft_ready and next_step is None:
            next_step = f"./run_dataset.sh {dataset} prep   (resumes the fine-tuning)"

    seeds_per_rep = int(load_config("prefilter")["train"]["seeds_per_rep"])
    mdir = root / "prefilter" / "models"
    from ..prefilter.stacking import main_prefix
    prefix = main_prefix(load_config("prefilter"))           # base-r (stacked fusion) or main-r (joint)
    reps = {}
    for tag in sorted(mdir.glob(f"{prefix}*")) if mdir.exists() else []:
        rep = tag.name[len(prefix):]
        if rep.isdigit():
            reps[rep] = len(list(tag.glob("seed*/logits.npz")))
    complete = sorted((r for r, n in reps.items() if n >= seeds_per_rep), key=int)
    partial = {r: n for r, n in reps.items() if n < seeds_per_rep}
    ablation_tags = sorted({p.name.split("-r")[0] if p.name.startswith("nopair") else p.name
                            for p in mdir.glob("*") if not p.name.startswith(prefix)}) if mdir.exists() else []
    lines.append(f"  {_mark(len(complete) >= 5, bool(reps))} P3 prefilter: complete reps {complete or 'none'}"
                 + (f", partial {partial}" if partial else "") + f"; ablation models: {ablation_tags or 'none'}")
    scores = root / "prefilter" / "prefilter_scores.parquet"
    scored = []
    if scores.exists():
        scored = sorted((r for r in pd.read_parquet(scores, columns=["rep"])["rep"].unique() if str(r).isdigit()), key=int)
        metrics = json.loads((root / "prefilter" / "prefilter_metrics.json").read_text())
        aucs = [metrics[r]["roc_auc"] for r in scored if metrics.get(r, {}).get("roc_auc") is not None]
        auc = f", test ROC-AUC mean {sum(aucs) / len(aucs):.4f}" if aucs else ""
        lines.append(f"  {_mark(set(scored) >= set(complete) and bool(scored))} P3 scores: reps {scored}{auc}")
    else:
        lines.append(f"  {MISS} P3 scores")
    if next_step is None and (not complete or partial or not set(map(str, complete)) <= set(map(str, scored))):
        # unfinished / interrupted reps resume where they stopped; the same command also (re)scores them
        next_step = (f"./run_dataset.sh {dataset} train   (re-run with the same --reps as before, default 1-5: "
                     f"finished seeds are skipped, interrupted ones restart)") if (complete or partial) else \
            f"./run_dataset.sh {dataset} train"

    index_path = root / "llm" / ("index.mock.parquet" if mock else "index.parquet")
    if index_path.exists():
        index = pd.read_parquet(index_path, columns=["id", "stage", "status", "cost_usd"])
        for subset in ("a_pool", "test"):
            ids = set(records.loc[records["subset"] == subset, "id"])
            p1 = index[(index["stage"] == "pass1") & index["id"].isin(ids)]
            z = index[(index["stage"] == "zeroshot") & index["id"].isin(ids)]
            extra = f", zero-shot {len(z):,}" if subset == "test" else ""
            lines.append(f"  {_mark(len(p1) >= len(ids), len(p1) > 0)} P4 LLM {subset}: Pass 1 {len(p1):,}/{len(ids):,}{extra}, "
                         f"invalid {(p1['status'] != 'ok').mean() if len(p1) else 0:.1%}")
        other = sorted(s for s in index["stage"].unique() if s.startswith(("a7", "a8", "rq5", "repeat")))
        lines.append(f"       API ablations: {other or 'none'}; spend {index['cost_usd'].sum():.3f} USD")
        llm_done = all(len(index[(index['stage'] == 'pass1') & index['id'].isin(set(records.loc[records['subset'] == s, 'id']))])
                       >= (records['subset'] == s).sum() for s in ("a_pool", "test"))
    else:
        lines.append(f"  {MISS} P4 LLM (no calls yet)")
        llm_done = False
    if complete and not llm_done and next_step is None:
        cfg = load_config("llm")
        frozen = all((cfg.get("prompt_hashes") or {}).get(s) == h for s, h in current_hashes().items())
        next_step = f"./run_dataset.sh {dataset} llm" if frozen else \
            f"prompts not frozen: iterate ./run_dataset.sh {dataset} dev until ok_to_freeze, then ./run_dataset.sh {dataset} freeze"

    decisions = root / "llm" / ("decisions.mock.parquet" if mock else "decisions.parquet")
    stale = decisions.exists() and index_path.exists() and decisions.stat().st_mtime < index_path.stat().st_mtime
    lines.append(f"  {_mark(decisions.exists() and not stale, decisions.exists())} P5 decisions" + (" (older than the LLM index - rerun)" if stale else ""))
    router = root / "decision" / "router_scores.parquet"
    router_reps = sorted(pd.read_parquet(router, columns=["rep"])["rep"].unique(), key=str) if router.exists() else []
    lines.append(f"  {_mark(router.exists() and set(map(str, router_reps)) >= set(scored), router.exists())} P6 decision layer: reps {router_reps or 'none'}")
    grid = artifacts_dir() / "results" / "metrics_grid.parquet"
    in_results = grid.exists() and dataset in set(pd.read_parquet(grid, columns=["dataset"])["dataset"])
    lines.append(f"  {_mark(in_results)} P7 results/tables.md")
    if llm_done and (not router.exists() or stale or not in_results) and next_step is None:
        next_step = f"./run_dataset.sh {dataset} offline"
    return lines, next_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()
    mock = args.mock or os.getenv("VG2_LLM_MOCK") == "1"
    cfg = load_config("llm")
    frozen = cfg.get("prompt_hashes") or {}
    current = current_hashes()
    key = bool(os.getenv(cfg.get("api_key_env", "OPENROUTER_API_KEY"), "").strip())
    print(f"artifacts: {artifacts_dir()}{'  (mock LLM)' if mock else ''}")
    print(f"API key set: {'yes' if key else 'NO - fill OPENROUTER_API_KEY in .env and save'} | provider {cfg['provider']['order']}")
    state = "frozen" if all(frozen.get(s) == h for s, h in current.items()) else \
        ("changed after freezing!" if any(frozen.values()) else "not frozen (run dev loop, then freeze)")
    print(f"prompts: {state}")
    dev = artifacts_dir() / load_config("datasets")["dev_set"]["dataset"] / "llm" / ("dev_report.mock.json" if mock else "dev_report.json")
    if dev.exists():
        d = json.loads(dev.read_text())["decision_point_step4"]
        print(f"dev loop: v2 pair acc {d['v2_pair_accuracy']:.3f} vs zero-shot {d['zeroshot_pair_accuracy']:.3f} -> ok_to_freeze={d['ok_to_freeze']}")
    else:
        print("dev loop: not run (./run_all.sh dev)")
    for dataset in parse_datasets(args.datasets):
        lines, next_step = dataset_status(dataset, mock)
        print(f"\n[{dataset}]")
        print("\n".join(lines))
        print(f"  next: {next_step or 'nothing - dataset complete'}")


if __name__ == "__main__":
    main()
