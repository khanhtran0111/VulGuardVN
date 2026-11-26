"""B7 baseline - fine-tuned UniXcoder (local GPU). Since 2026-11-23 it is produced by the same cross-fitted
fine-tuning that feeds the prefilter view `finetune` (P3 stage `finetune`), so the two can never disagree:
K fold models, test logit = their mean, threshold = max F1 on train_dev (out-of-fold).

    python -m v2.scripts.run_b7_unixcoder_ft --datasets devign
Output: artifacts/{ds}/baselines/b7_unixcoder_ft.parquet (+ b7_summary.json), picked up by run_p7_experiments.
"""
from __future__ import annotations

import argparse

from ..config import load_config, log_run_header, parse_datasets
from ..prefilter.finetune import crossfit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="devign")
    parser.add_argument("--force", action="store_true", help="re-run the fine-tuning (also rewrites the prefilter view)")
    args = parser.parse_args()
    log_run_header("run_b7_unixcoder_ft", args, ["prefilter"])
    for dataset in parse_datasets(args.datasets):
        crossfit(dataset, load_config("prefilter")["finetune"], args.force)


if __name__ == "__main__":
    main()
