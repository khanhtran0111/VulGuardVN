"""Create descriptive routing-policy tables/curves without selecting test thresholds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from analysis_plots import line_plot


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "baseline" / "baseline2"))
from metrics import compute_binary_metrics  # noqa: E402


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_predictions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze(run_dir: Path, output_dir: Path, dataset: str, seed: int, baseline_run: Path | None = None) -> dict:
    predictions_path = run_dir / "predictions.jsonl"
    for name in ("config.json", "metrics.json", "calibration.json", "predictions.jsonl"):
        if not (run_dir / name).is_file(): raise FileNotFoundError(run_dir / name)
    config = _read_json(run_dir / "config.json"); metrics = _read_json(run_dir / "metrics.json"); calibration = _read_json(run_dir / "calibration.json")
    rows = _read_predictions(predictions_path)
    labels = np.asarray([int(row["ground_truth"]) for row in rows], dtype=int)
    probabilities = np.asarray([float(row["calibrated_probability"]) for row in rows], dtype=float)
    final_predictions = np.asarray([int(row["prediction"]) for row in rows], dtype=int)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_metrics = {
        "dataset": dataset, "training_seed": int(seed), "configuration": config.get("configuration"),
        "call_llm_for_inspect": config.get("call_llm_for_inspect"),
        "call_llm_for_high": config.get("call_llm_for_high"),
        "delta_high": config.get("delta_high"),
        "force_inspect_all": config.get("force_inspect_all"),
        "tau_low": calibration.get("tau_low"), "tau_high": calibration.get("tau_high"),
        "threshold_selection_split": calibration.get("threshold_selection_split"),
        "metrics": {key: metrics.get(key) for key in ("samples", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "llm_calls", "llm_call_ratio")},
        "operating_point_note": "Primary tau_low/tau_high were selected on validation. Test-grid points below are descriptive only.",
    }
    (output_dir / "routing_policy_metrics.json").write_text(json.dumps(policy_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    points = []
    for width in np.linspace(0.0, 0.49, 26):
        tau_low, tau_high = 0.5 - width, 0.5 + width
        inspect = (probabilities > tau_low) & (probabilities < tau_high)
        proxy = np.where(probabilities <= tau_low, 0, np.where(probabilities >= tau_high, 1, probabilities >= 0.5)).astype(int)
        point_metrics = compute_binary_metrics(labels, proxy, probabilities)
        covered = ~inspect
        risk = float(np.mean(proxy[covered] != labels[covered])) if np.any(covered) else None
        points.append({
            "tau_low": float(tau_low), "tau_high": float(tau_high),
            "coverage": float(np.mean(covered)), "risk": risk,
            "llm_call_ratio": float(np.mean(inspect)), "recall": point_metrics.get("recall"), "f1": point_metrics.get("f1"),
            "selection_role": "descriptive_test_curve_only",
        })
    with (output_dir / "routing_operating_points.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0])); writer.writeheader(); writer.writerows(points)
    line_plot(output_dir / "risk_coverage_curve.png", [([p["coverage"] for p in points], [p["risk"] or 0.0 for p in points], "risk")], title=f"{dataset}: risk-coverage", xlabel="Coverage", ylabel="Risk")
    line_plot(output_dir / "recall_llm_call_curve.png", [([p["llm_call_ratio"] for p in points], [p["recall"] for p in points], "recall")], title=f"{dataset}: recall vs LLM-call ratio", xlabel="LLM-call ratio", ylabel="Recall")
    line_plot(output_dir / "f1_llm_call_curve.png", [([p["llm_call_ratio"] for p in points], [p["f1"] for p in points], "F1")], title=f"{dataset}: F1 vs LLM-call ratio", xlabel="LLM-call ratio", ylabel="F1")

    matched_rows = []
    if baseline_run is not None and (baseline_run / "metrics.json").is_file():
        baseline_metrics = _read_json(baseline_run / "metrics.json"); baseline_recall = baseline_metrics.get("recall")
        if baseline_recall is not None:
            matched = min(points, key=lambda point: abs(float(point["recall"]) - float(baseline_recall)))
            matched_rows.append({
                "baseline_recall": baseline_recall, "matched_recall": matched["recall"],
                "recall_gap": float(matched["recall"] - baseline_recall), "llm_call_ratio": matched["llm_call_ratio"],
                "f1": matched["f1"], "tau_low": matched["tau_low"], "tau_high": matched["tau_high"],
                "selection_role": "descriptive_matched_recall_not_primary_threshold",
            })
    fields = ["baseline_recall", "matched_recall", "recall_gap", "llm_call_ratio", "f1", "tau_low", "tau_high", "selection_role"]
    with (output_dir / "matched_recall.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(matched_rows)
    return {**policy_metrics, "operating_points": len(points), "matched_recall_available": bool(matched_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset", required=True); parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--baseline-run", type=Path); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "run_dir": str(args.run_dir), "output_path": str(args.output_path), "baseline_run": str(args.baseline_run) if args.baseline_run else None}, indent=2)); return
    print(json.dumps(analyze(args.run_dir, args.output_path, args.dataset, args.seed, args.baseline_run), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
