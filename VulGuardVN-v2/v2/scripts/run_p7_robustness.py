"""P7 step 5 / RQ5 - robustness of the skip branch under semantics-preserving transformations.

2,000 test functions (all vulnerable test functions first, the rest random) are transformed (variable renaming,
dead-code insertion), re-scored by the prefilter ensembles and passed through the conformal gate; we report the
skip miss rate on vulnerable functions before vs after the transformation.

    python -m v2.scripts.run_p7_robustness --datasets primevul
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from ..common import dump_json
from ..config import artifacts_dir, dataset_dir, get_logger, load_config, log_run_header, parse_datasets
from ..data.io import load_records
from ..decision.frame import load_scores
from ..experiments.transforms import TRANSFORMS
from ..prefilter import stacking
from ..prefilter.infer import ensemble_logits, featurize_codes
from ..prefilter.train import models_dir

log = get_logger("p7.robust")


def sample_ids(records: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    test = records[(records["subset"] == "test") & records["in_main"]]
    rng = np.random.default_rng(seed)
    pos = test[test["label"] == 1]
    pos = pos.iloc[rng.permutation(len(pos))[: n // 2]]
    neg = test[test["label"] == 0]
    neg = neg.iloc[rng.permutation(len(neg))[: n - len(pos)]]
    return pd.concat([pos, neg]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="primevul")
    args = parser.parse_args()
    log_run_header("run_p7_robustness", args, ["experiments", "prefilter", "decision"])
    ecfg = load_config("experiments")
    pcfg = load_config("prefilter")
    alpha = ecfg["main"]["alpha"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for dataset in parse_datasets(args.datasets):
        records = load_records(dataset, ["id", "code", "label", "subset", "in_main"])
        sample = sample_ids(records, int(ecfg["robustness"]["n_functions"]), int(ecfg["robustness"]["seed"]))
        conformal = json.loads((dataset_dir(dataset, "decision") / "conformal.json").read_text())
        scores = load_scores(dataset)
        calibration = json.loads((dataset_dir(dataset, "prefilter") / "calibrators.json").read_text())
        out_dir = dataset_dir(dataset, "robustness")
        views = {"original": None}
        codes = {"original": sample["code"].tolist()}
        for name, fn in TRANSFORMS.items():
            codes[name] = [fn(code) for code in codes["original"]]
        pd.DataFrame({"id": sample["id"], **{f"code_{k}": v for k, v in codes.items() if k != "original"}}).to_parquet(
            out_dir / "transformed_code.parquet", index=False)
        for name in TRANSFORMS:
            views[name] = featurize_codes(dataset, codes[name], device)
        for rep_key in sorted((k for k in conformal if k.isdigit()), key=int):
            rep = int(rep_key)
            t_alpha = conformal[rep_key]["thresholds"][str(alpha)]["t_alpha"]
            original = scores[(scores["rep"] == rep_key)].set_index("id").loc[sample["id"], "s"].to_numpy()
            tag = stacking.main_tag(rep, pcfg)
            seeds = [10 * rep + k for k in range(int(pcfg["train"]["seeds_per_rep"]))
                     if (models_dir(dataset) / tag / f"seed{10 * rep + k}" / "model.pt").exists()]
            if not seeds:
                continue
            stack = calibration.get(rep_key, {}).get("stack")
            y = sample["label"].to_numpy()

            def rescore(v: dict) -> np.ndarray:
                s_net = ensemble_logits(dataset, tag, seeds, v, device)
                return stacking.apply(s_net, v["finetune"], stack) if stack else s_net

            for name in ["original", *TRANSFORMS]:
                s = original if name == "original" else rescore(views[name])
                skipped = s <= t_alpha if t_alpha is not None else np.zeros(len(s), bool)
                rows.append({"dataset": dataset, "rep": rep, "transform": name, "alpha": alpha,
                             "skip_miss_rate": float(skipped[y == 1].mean()), "coverage": float(skipped.mean()),
                             "mean_abs_score_shift": float(np.abs(s - original).mean())})
        log.info("[%s] robustness done", dataset)
    frame = pd.DataFrame(rows)
    out = artifacts_dir() / "results"
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "rq5_robustness.csv", index=False)
    summary = frame.groupby(["dataset", "transform"])[["skip_miss_rate", "coverage", "mean_abs_score_shift"]].agg(["mean", "std"])
    print(summary.to_string())
    dump_json(out / "rq5_robustness_summary.json", json.loads(summary.reset_index().to_json(orient="records")))


if __name__ == "__main__":
    main()
