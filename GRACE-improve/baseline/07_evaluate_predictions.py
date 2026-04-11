import json

from common import METRICS_DIR, PREDICTIONS_DIR, dump_json
from metrics import bootstrap_f1_interval, compute_binary_metrics, mcnemar_exact


DATASET_NAME = "devign"
PREDICTIONS_PATH = PREDICTIONS_DIR / DATASET_NAME / "grace_prefilter_predictions.jsonl"
RUN_STATE_PATH = PREDICTIONS_DIR / DATASET_NAME / "grace_prefilter_run_state.json"
BASELINE_COMPARE_PATH = None
EXPECTED_SCHEMA_VERSION = 4


def _load_predictions(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_predictions(rows, run_state):
    if run_state.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise RuntimeError(
            f"Prediction schema mismatch. Expected {EXPECTED_SCHEMA_VERSION}, got {run_state.get('schema_version')}."
        )
    if not run_state.get("complete"):
        raise RuntimeError(
            f"Run is incomplete: resolved_samples={run_state.get('resolved_samples')} / target_samples={run_state.get('target_samples')}."
        )
    if not run_state.get("evaluation_ready"):
        raise RuntimeError(
            "Run is not evaluation-ready. Resolve the issue in run_state and rerun script 06 before evaluating."
        )
    if len(rows) != int(run_state.get("target_samples", -1)):
        raise RuntimeError(
            f"Predictions count mismatch. Found {len(rows)} rows but target_samples={run_state.get('target_samples')}."
        )
    record_ids = set()
    for row in rows:
        if row.get("schema_version") != EXPECTED_SCHEMA_VERSION:
            raise RuntimeError(f"Found incompatible prediction row for record {row.get('record_id')}.")
        if row.get("resolution_status") != "resolved":
            raise RuntimeError(f"Found unresolved prediction row for record {row.get('record_id')}.")
        record_id = row.get("record_id")
        if record_id in record_ids:
            raise RuntimeError(f"Duplicate prediction row for record {record_id}.")
        record_ids.add(record_id)


def main() -> None:
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"Missing predictions file: {PREDICTIONS_PATH}")
    if not RUN_STATE_PATH.exists():
        raise FileNotFoundError(f"Missing run state file: {RUN_STATE_PATH}")

    rows = _load_predictions(PREDICTIONS_PATH)
    run_state = _load_json(RUN_STATE_PATH)
    _validate_predictions(rows, run_state)

    labels = [int(row["ground_truth"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    probabilities = [float(row["calibrated_probability"]) for row in rows]

    metrics = compute_binary_metrics(labels, predictions, probabilities)
    metrics["dataset"] = DATASET_NAME
    metrics["model_name"] = run_state.get("model_name")
    metrics["experiment_mode"] = run_state.get("experiment_mode")
    metrics["schema_version"] = EXPECTED_SCHEMA_VERSION
    metrics["samples"] = len(rows)
    metrics["llm_calls"] = sum(1 for row in rows if row.get("llm_called"))
    metrics["api_requests_made"] = sum(1 for row in rows if row.get("api_request_made"))
    metrics["llm_cache_hits"] = sum(1 for row in rows if row.get("llm_cache_hit"))
    metrics["llm_call_ratio"] = metrics["llm_calls"] / max(len(rows), 1)
    metrics["routing"] = {
        "skip": sum(1 for row in rows if row.get("risk_band") == "skip"),
        "uncertain": sum(1 for row in rows if row.get("risk_band") == "uncertain"),
        "high": sum(1 for row in rows if row.get("risk_band") == "high"),
    }
    metrics["decision_sources"] = {
        "prefilter": sum(1 for row in rows if row.get("decision_source") == "prefilter"),
        "llm": sum(1 for row in rows if row.get("decision_source") == "llm"),
    }
    metrics["retrieval_backend"] = run_state.get("retrieval_backend")
    metrics["graph_backend_requested"] = run_state.get("graph_backend_requested")
    metrics["graph_backend_counts"] = run_state.get("graph_backend_counts")
    metrics["timing_ms"] = run_state.get("timing_ms")
    metrics["bootstrap_f1"] = bootstrap_f1_interval(labels, predictions, iterations=1000)
    metrics["config"] = run_state.get("config")
    metrics["run_state_path"] = str(RUN_STATE_PATH)

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
