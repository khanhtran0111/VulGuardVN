import json

from common import METRICS_DIR, PREDICTIONS_DIR, dump_json
from metrics import bootstrap_f1_interval, compute_binary_metrics, mcnemar_exact


DATASET_NAME = "devign"
PREDICTIONS_PATH = PREDICTIONS_DIR / DATASET_NAME / "grace_prefilter_predictions.jsonl"
BASELINE_COMPARE_PATH = None


def _load_predictions(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_PATH}")
    rows = _load_predictions(PREDICTIONS_PATH)
    labels = [int(row["ground_truth"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    probabilities = [float(row["calibrated_probability"]) for row in rows]
    metrics = compute_binary_metrics(labels, predictions, probabilities)
    metrics["dataset"] = DATASET_NAME
    metrics["samples"] = len(rows)
    metrics["llm_calls"] = sum(1 for row in rows if row.get("llm_called"))
    metrics["llm_call_ratio"] = metrics["llm_calls"] / max(len(rows), 1)
    metrics["routing"] = {
        "skip": sum(1 for row in rows if row.get("risk_band") == "skip"),
        "uncertain": sum(1 for row in rows if row.get("risk_band") == "uncertain"),
        "high": sum(1 for row in rows if row.get("risk_band") == "high"),
    }
    metrics["bootstrap_f1"] = bootstrap_f1_interval(labels, predictions, iterations=1000)
    if BASELINE_COMPARE_PATH:
        baseline_rows = {row["record_id"]: row for row in _load_predictions(BASELINE_COMPARE_PATH)}
        aligned = [row for row in rows if row["record_id"] in baseline_rows]
        if aligned:
            base_predictions = [int(baseline_rows[row["record_id"]]["prediction"]) for row in aligned]
            aligned_labels = [int(row["ground_truth"]) for row in aligned]
            aligned_predictions = [int(row["prediction"]) for row in aligned]
            metrics["comparison"] = {
                "mcnemar": mcnemar_exact(aligned_labels, base_predictions, aligned_predictions),
                "baseline_f1": compute_binary_metrics(aligned_labels, base_predictions)["f1"],
                "current_f1": compute_binary_metrics(aligned_labels, aligned_predictions)["f1"],
            }
    output_path = METRICS_DIR / DATASET_NAME / "evaluation_summary.json"
    dump_json(output_path, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
