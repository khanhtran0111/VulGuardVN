"""P3 - Prefilter: feature store, 5 reps x 5-seed ensemble, calibrators on A, scores, histograms.
A1 (leave-one-view-out) and A3 (no pairwise loss) retrain variants are stage `ablations`.

    python -m v2.scripts.run_p3_prefilter --datasets all --stage features     # heuristic + UniXcoder (slow, resumable)
    python -m v2.scripts.run_p3_prefilter --datasets all --stage finetune     # cross-fitted fine-tuned UniXcoder (+ B7)
    python -m v2.scripts.run_p3_prefilter --datasets all --stage train
    python -m v2.scripts.run_p3_prefilter --datasets all --stage score
    python -m v2.scripts.run_p3_prefilter --datasets all --stage ablations
    python -m v2.scripts.run_p3_prefilter --datasets all --stage all          # features + train + score
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from ..common import dump_json
from ..config import artifacts_dir, dataset_dir, get_logger, load_config, log_run_header, parse_datasets
from ..features.store import build_heuristic_views, build_unixcoder_view
from ..prefilter import stacking
from ..prefilter.finetune import crossfit, logits_path
from ..prefilter.model import ALL_VIEWS
from ..prefilter.train import DatasetTensors, evaluate_scores, load_logits, models_dir, run_tag, score_set

log = get_logger("p3")


def main_seeds(rep: int, n: int) -> list[int]:
    return [10 * rep + k for k in range(n)]


def stage_train(dataset: str, cfg: dict, reps: list[int]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DatasetTensors(dataset, device)
    log.info("[%s] train_fit=%d train_dev=%d pairs=%d device=%s", dataset, len(data.train_idx), len(data.dev_idx),
             len(data.pairs), device)
    n = int(cfg["train"]["seeds_per_rep"])
    views = stacking.network_views(cfg)
    log.info("[%s] fusion=%s, network views=%s", dataset, cfg["model"].get("fusion", "joint"), ",".join(views))
    for rep in reps:
        run_tag(dataset, data, cfg, stacking.main_tag(rep, cfg), main_seeds(rep, n), views=views)
    del data
    torch.cuda.empty_cache()


def stage_ablations(dataset: str, cfg: dict) -> None:
    exp = load_config("experiments")["ablations"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DatasetTensors(dataset, device)
    all_views = tuple(cfg["model"].get("views", ALL_VIEWS))
    n = int(cfg["train"]["seeds_per_rep"])
    for view in all_views:                                       # A1: leave-one-view-out, single models
        if _lovo_from_base(cfg, view, exp):
            continue                                             # = the base network of rep 1 (scored from base-r1)
        run_tag(dataset, data, cfg, f"lovo-{view}", list(exp["lovo_seeds"]), views=stacking.network_views(cfg, drop=view))
    if len(data.pairs):                                          # A3: no pairwise loss
        for rep in exp["nopair_reps"]:
            run_tag(dataset, data, cfg, f"nopair-r{rep}", main_seeds(int(rep), n), views=stacking.network_views(cfg),
                    pair_loss=False)
    if stacking.stacked(cfg):                                    # A14: fine-tuned logit as a network input (joint)
        for rep in exp.get("joint_reps", [1]):
            run_tag(dataset, data, cfg, f"main-r{rep}", main_seeds(int(rep), n), views=all_views)
    del data
    torch.cuda.empty_cache()


def _lovo_from_base(cfg: dict, view: str, exp: dict) -> bool:
    """Stacked fusion: leaving out `finetune` gives exactly the base network; reuse base-r1 when the seeds match."""
    return stacking.stacked(cfg) and view == "finetune" and set(map(int, exp["lovo_seeds"])) <= set(main_seeds(1, int(cfg["train"]["seeds_per_rep"])))


def _available(dataset: str, tag: str, seeds: list[int]) -> bool:
    return all((models_dir(dataset) / tag / f"seed{s}" / "logits.npz").exists() for s in seeds)


def trained_reps(dataset: str, cfg: dict | None = None) -> list[int]:
    """Every main-rep directory on disk (base-r{rep} when stacked, main-r{rep} when joint; any order)."""
    prefix = stacking.main_prefix(cfg or load_config("prefilter"))
    root = models_dir(dataset)
    reps = [int(path.name[len(prefix):]) for path in root.glob(f"{prefix}*") if path.name[len(prefix):].isdigit()]
    return sorted(reps)


def stage_score(dataset: str, cfg: dict) -> pd.DataFrame:
    from ..data.io import load_records

    records = load_records(dataset, ["id", "split", "subset", "label", "weight", "dev_set", "in_main"])
    calibrators = list(cfg["calibrators"])
    n = int(cfg["train"]["seeds_per_rep"])
    exp = load_config("experiments")["ablations"]
    frames, params, metrics = [], {}, {}
    stacked = stacking.stacked(cfg)
    finetune = np.load(logits_path(dataset)) if stacked else None

    def add(rep_name: str, tag: str, seeds: list[int], stack: bool = stacked) -> None:
        if not _available(dataset, tag, seeds):
            return
        frame, p = score_set(records, [load_logits(dataset, tag, s) for s in seeds], rep_name, calibrators,
                             finetune if stack else None)
        frames.append(frame)
        params[rep_name] = p
        metrics[rep_name] = evaluate_scores(frame, calibrators)

    for rep in trained_reps(dataset, cfg):
        seeds = main_seeds(rep, n)
        tag = stacking.main_tag(rep, cfg)
        if not _available(dataset, tag, seeds):
            done = [s for s in seeds if _available(dataset, tag, [s])]
            log.info("[%s] rep %d incomplete (%d/%d seeds trained: %s) - not scored yet", dataset, rep, len(done), n, done)
            continue
        add(str(rep), tag, seeds)
        add(f"single-r{rep}", tag, seeds[:1])                                # A2: no ensemble
    for rep in exp["nopair_reps"]:
        add(f"nopair-r{rep}", f"nopair-r{rep}", main_seeds(int(rep), n))
    for view in cfg["model"].get("views", ALL_VIEWS):
        for seed in exp["lovo_seeds"]:
            if _lovo_from_base(cfg, view, exp):                              # A1 without finetune: base net, no stack
                add(f"lovo-{view}-s{seed}", stacking.main_tag(1, cfg), [int(seed)], stack=False)
            else:
                add(f"lovo-{view}-s{seed}", f"lovo-{view}", [int(seed)])
    if stacked:
        for rep in exp.get("joint_reps", [1]):                               # A14: joint 5-view network
            add(f"joint-r{rep}", f"main-r{rep}", main_seeds(int(rep), n), stack=False)
    if not frames:
        raise SystemExit(f"[{dataset}] no trained models found - run --stage train first")
    scores = pd.concat(frames, ignore_index=True)
    scores.insert(1, "dataset", dataset)
    out = dataset_dir(dataset, "prefilter")
    scores.to_parquet(out / "prefilter_scores.parquet", index=False)
    dump_json(out / "calibrators.json", params)
    dump_json(out / "prefilter_metrics.json", metrics)
    _histograms(dataset, scores)
    main_reps = [m for k, m in metrics.items() if k.isdigit()]
    if main_reps:
        auc = [m["roc_auc"] for m in main_reps if m["roc_auc"] is not None]
        log.info("[%s] test ROC-AUC %.4f +- %.4f | PR-AUC %.4f (n_reps=%d)", dataset, np.mean(auc), np.std(auc),
                 np.mean([m["pr_auc"] for m in main_reps]), len(main_reps))
        if dataset == "primevul" and np.mean(auc) <= 0.6:
            log.info("DECISION POINT (P3): PrimeVul ROC-AUC <= 0.6 - strengthen the prefilter before P6")
    return scores


def _histograms(dataset: str, scores: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    main_reps = sorted((r for r in scores["rep"].unique() if str(r).isdigit()), key=int)
    if not main_reps:
        return
    first = main_reps[0]
    test = scores[(scores["rep"] == first) & (scores["subset"] == "test") & scores["in_main"]]
    if test.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for label, color, name in [(0, "#4C78A8", "not vulnerable"), (1, "#E45756", "vulnerable")]:
        part = test[test["label"] == label]
        axes[0].hist(part["s"], bins=60, alpha=0.6, density=True, color=color, label=f"{name} (n={len(part)})")
        axes[1].hist(part["p_hat_beta"], bins=60, alpha=0.6, density=True, color=color, label=name)
    axes[0].set_xlabel("ensemble logit s(x)")
    axes[1].set_xlabel("calibrated p_hat (beta)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: prefilter score by class (test, rep {first})")
    fig.tight_layout()
    fig.savefig(dataset_dir(dataset, "prefilter") / "hist_scores.png", dpi=150)
    plt.close(fig)


def summary_table(datasets: list[str]) -> None:
    rows = []
    for dataset in datasets:
        path = dataset_dir(dataset, "prefilter") / "prefilter_metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text())
        for rep, m in metrics.items():
            rows.append({"dataset": dataset, "rep": rep, **{k: v for k, v in m.items()}})
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(artifacts_dir() / "prefilter_metrics.csv", index=False)
        log.info("prefilter metrics -> %s", artifacts_dir() / "prefilter_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--stage", default="all", choices=["features", "heuristic", "unixcoder", "finetune", "train", "score",
                                                            "ablations", "all"])
    parser.add_argument("--reps", default=None, help="comma list of reps to train, e.g. 1,2 (default: 1..reps in "
                        "prefilter.yaml). Rep r = ensemble seeds {10r..10r+4}; reps are independent and resumable")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quick", action="store_true", help="smoke test: 2 reps x 2 seeds x 2 epochs")
    args = parser.parse_args()
    log_run_header("run_p3_prefilter", args, ["prefilter", "datasets", "experiments"])
    cfg = load_config("prefilter")
    if args.quick:
        cfg["train"].update({"reps": 2, "seeds_per_rep": 2, "max_epochs": 2})
        cfg["finetune"].update({"folds": 2, "epochs": 1, "max_len": 128, "max_neg": 300})
    reps = [int(r) for r in args.reps.split(",")] if args.reps else list(range(1, int(cfg["train"]["reps"]) + 1))
    datasets = parse_datasets(args.datasets)
    for dataset in datasets:
        if args.stage in {"features", "heuristic", "all"}:
            build_heuristic_views(dataset, cfg["features"], args.force)
        if args.stage in {"features", "unixcoder", "all"}:
            build_unixcoder_view(dataset, cfg["features"], args.force)
        if args.stage in {"finetune", "all"} and "finetune" in cfg["model"].get("views", ALL_VIEWS):
            crossfit(dataset, cfg["finetune"], args.force)
        if args.stage in {"train", "all"}:
            stage_train(dataset, cfg, reps)
        if args.stage == "ablations":
            stage_ablations(dataset, cfg)
        if args.stage in {"score", "all", "ablations"}:
            stage_score(dataset, cfg)
    summary_table(parse_datasets("all"))       # merged table over every dataset scored so far


if __name__ == "__main__":
    main()
