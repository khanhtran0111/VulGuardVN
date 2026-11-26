"""Analysis-only calibration tables and figures from an E01 calibration.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis_plots import histogram, line_plot


def _roc_points(labels: np.ndarray, probabilities: np.ndarray) -> tuple[list[float], list[float]]:
    thresholds = [float("inf"), *sorted({float(value) for value in probabilities}, reverse=True), float("-inf")]
    positives, negatives = max(int(np.sum(labels == 1)), 1), max(int(np.sum(labels == 0)), 1)
    fpr, tpr = [], []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        tpr.append(float(np.sum(predictions & (labels == 1)) / positives))
        fpr.append(float(np.sum(predictions & (labels == 0)) / negatives))
    return fpr, tpr


def _pr_points(labels: np.ndarray, probabilities: np.ndarray) -> tuple[list[float], list[float]]:
    thresholds = [float("inf"), *sorted({float(value) for value in probabilities}, reverse=True), float("-inf")]
    total_positive = max(int(np.sum(labels == 1)), 1); recall, precision = [], []
    for threshold in thresholds:
        predictions = probabilities >= threshold
        tp = int(np.sum(predictions & (labels == 1))); fp = int(np.sum(predictions & (labels == 0)))
        recall.append(float(tp / total_positive)); precision.append(float(tp / max(tp + fp, 1)))
    return recall, precision


def analyze(calibration_path: Path, output_dir: Path, dataset: str, seed: int) -> dict:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if payload.get("threshold_selection_split") != "validation":
        raise RuntimeError(
            f"Threshold provenance violation: {payload.get('threshold_selection_split')!r}; expected 'validation'"
        )
    records = payload.get("probability_records") or []
    if not records:
        raise RuntimeError("calibration.json does not contain probability_records")
    labels = np.asarray([int(row["ground_truth"]) for row in records], dtype=int)
    probabilities = np.asarray([float(row["calibrated_probability"]) for row in records], dtype=float)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "probability_records.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["record_id", "ground_truth", "raw_fusion_score", "calibrated_probability", "calibration_method"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(records)

    tau_low, tau_high = payload.get("tau_low"), payload.get("tau_high")
    histogram(
        output_dir / "probability_histogram.png",
        probabilities[labels == 0], probabilities[labels == 1],
        tau_low=tau_low, tau_high=tau_high, title=f"{dataset}: calibrated probability distribution",
    )
    bin_ids = np.minimum((probabilities * 10).astype(int), 9)
    predicted, observed = [], []
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if np.any(mask):
            predicted.append(float(np.mean(probabilities[mask]))); observed.append(float(np.mean(labels[mask])))
    line_plot(
        output_dir / "reliability_diagram.png",
        [(predicted, observed, "model"), ([0, 1], [0, 1], "perfect")],
        title=f"{dataset}: reliability", xlabel="Mean predicted probability", ylabel="Observed positive rate",
    )
    fpr, tpr = _roc_points(labels, probabilities)
    line_plot(output_dir / "roc_curve.png", [(fpr, tpr, "ROC")], title=f"{dataset}: ROC", xlabel="False-positive rate", ylabel="True-positive rate")
    recall, precision = _pr_points(labels, probabilities)
    line_plot(output_dir / "pr_curve.png", [(recall, precision, "PR")], title=f"{dataset}: precision-recall", xlabel="Recall", ylabel="Precision")

    metrics = payload.get("calibration_metrics") or {}
    calibrator = payload.get("calibrator") or {}
    summary = {
        "dataset": dataset,
        "training_seed": int(seed),
        "calibration_method": payload.get("calibration_method") or calibrator.get("method"),
        "calibrator_parameters": calibrator.get("parameters"),
        "tau_low": tau_low,
        "tau_high": tau_high,
        "threshold_selection_split": payload.get("threshold_selection_split"),
        "threshold_selection_strategy": payload.get("threshold_selection_strategy"),
        "ece": metrics.get("ece"),
        "brier": metrics.get("brier"),
        "nll": metrics.get("nll"),
        "record_count": len(records),
        "vulnerable_count": int(np.sum(labels == 1)),
        "non_vulnerable_count": int(np.sum(labels == 0)),
        "vulnerable_probability_mean": float(np.mean(probabilities[labels == 1])) if np.any(labels == 1) else None,
        "non_vulnerable_probability_mean": float(np.mean(probabilities[labels == 0])) if np.any(labels == 0) else None,
        "source_calibration_path": str(calibration_path),
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "calibration_path": str(args.calibration_path), "output_path": str(args.output_path)}, indent=2)); return
    if not args.calibration_path.is_file(): raise FileNotFoundError(args.calibration_path)
    print(json.dumps(analyze(args.calibration_path, args.output_path, args.dataset, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
