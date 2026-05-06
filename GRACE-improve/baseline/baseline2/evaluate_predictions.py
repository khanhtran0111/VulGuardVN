import json
from pathlib import Path
from typing import Any

from common import METRICS_DIR, PREDICTIONS_DIR, dump_json
from metrics import bootstrap_f1_interval, compute_binary_metrics, mcnemar_exact


DEFAULT_DATASET_NAME = "devign"
DEFAULT_PREDICTIONS_FILENAME = "grace_hybrid_predictions.jsonl"
DEFAULT_RUN_STATE_FILENAME = "grace_hybrid_run_state.json"
EXPECTED_SCHEMA_VERSION = 1


def default_prediction_paths(dataset_name: str = DEFAULT_DATASET_NAME) -> tuple[Path, Path]:
    dataset_dir = PREDICTIONS_DIR / dataset_name
    return (
        dataset_dir / DEFAULT_PREDICTIONS_FILENAME,
        dataset_dir / DEFAULT_RUN_STATE_FILENAME,
    )


def load_predictions(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _count_by_field(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def validate_predictions(
    rows: list[dict[str, Any]],
    run_state: dict[str, Any],
    expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
) -> None:
    if run_state.get("schema_version") != expected_schema_version:
        raise RuntimeError(
            f"Prediction schema mismatch. Expected {expected_schema_version}, got {run_state.get('schema_version')}."
        )
    if not run_state.get("complete"):
        raise RuntimeError(
            f"Run is incomplete: resolved_samples={run_state.get('resolved_samples')} / target_samples={run_state.get('target_samples')}."
        )
    if not run_state.get("evaluation_ready"):
        raise RuntimeError(
            "Run is not evaluation-ready. Resolve the issue in run_state and rerun script 07 before evaluating."
        )
    if len(rows) != int(run_state.get("target_samples", -1)):
        raise RuntimeError(
            f"Predictions count mismatch. Found {len(rows)} rows but target_samples={run_state.get('target_samples')}."
        )
    record_ids = set()
    for row in rows:
        if row.get("schema_version") != expected_schema_version:
            raise RuntimeError(f"Found incompatible prediction row for record {row.get('record_id')}.")
        if row.get("resolution_status") != "resolved":
            raise RuntimeError(f"Found unresolved prediction row for record {row.get('record_id')}.")
        record_id = row.get("record_id")
        if record_id in record_ids:
            raise RuntimeError(f"Duplicate prediction row for record {record_id}.")
        record_ids.add(record_id)


def build_evaluation_metrics(
    rows: list[dict[str, Any]],
    run_state: dict[str, Any],
    *,
    dataset_name: str | None = None,
    baseline_compare_path: Path | None = None,
    expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
    bootstrap_iterations: int = 1000,
    run_state_path: Path | None = None,
) -> dict[str, Any]:
    resolved_dataset = dataset_name or run_state.get("dataset") or DEFAULT_DATASET_NAME

    labels = [int(row["ground_truth"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    probabilities = [float(row["calibrated_probability"]) for row in rows]

    metrics = compute_binary_metrics(labels, predictions, probabilities)
    metrics["dataset"] = resolved_dataset
    metrics["model_name"] = run_state.get("model_name")
    metrics["experiment_mode"] = run_state.get("experiment_mode")
    metrics["schema_version"] = expected_schema_version
    metrics["samples"] = len(rows)
    metrics["llm_calls"] = sum(1 for row in rows if row.get("llm_called"))
    metrics["api_requests_made"] = sum(1 for row in rows if row.get("api_request_made"))
    metrics["llm_cache_hits"] = sum(1 for row in rows if row.get("llm_cache_hit"))
    metrics["llm_call_ratio"] = metrics["llm_calls"] / max(len(rows), 1)
    metrics["routing"] = _count_by_field(rows, "risk_band")
    metrics["decision_sources"] = _count_by_field(rows, "decision_source")
    metrics["retrieval_backend"] = run_state.get("retrieval_backend")
    metrics["graph_backend_requested"] = run_state.get("graph_backend_requested")
    metrics["graph_backend_counts"] = run_state.get("graph_backend_counts")
    metrics["timing_ms"] = run_state.get("timing_ms")
    metrics["run_signature"] = run_state.get("run_signature")
    metrics["bootstrap_f1"] = bootstrap_f1_interval(labels, predictions, iterations=bootstrap_iterations)
    metrics["config"] = run_state.get("config")
    metrics["predictions_path"] = run_state.get("predictions_path")
    if run_state_path is not None:
        metrics["run_state_path"] = str(run_state_path)

    if baseline_compare_path:
        baseline_rows = {row["record_id"]: row for row in load_predictions(baseline_compare_path)}
        aligned = [row for row in rows if row["record_id"] in baseline_rows]
        if aligned:
            base_predictions = [int(baseline_rows[row["record_id"]]["prediction"]) for row in aligned]
            aligned_labels = [int(row["ground_truth"]) for row in aligned]
            aligned_predictions = [int(row["prediction"]) for row in aligned]
            metrics["comparison"] = {
                "mcnemar": mcnemar_exact(aligned_labels, base_predictions, aligned_predictions),
                "baseline_f1": compute_binary_metrics(aligned_labels, base_predictions)["f1"],
                "current_f1": compute_binary_metrics(aligned_labels, aligned_predictions)["f1"],
                "aligned_samples": len(aligned),
                "baseline_compare_path": str(baseline_compare_path),
            }

    return metrics


def evaluate_prediction_artifacts(
    predictions_path: Path,
    run_state_path: Path,
    *,
    dataset_name: str | None = None,
    baseline_compare_path: Path | None = None,
    expected_schema_version: int = EXPECTED_SCHEMA_VERSION,
    bootstrap_iterations: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {predictions_path}")
    if not run_state_path.exists():
        raise FileNotFoundError(f"Missing run state file: {run_state_path}")

    rows = load_predictions(predictions_path)
    run_state = load_json(run_state_path)
    validate_predictions(rows, run_state, expected_schema_version=expected_schema_version)
    metrics = build_evaluation_metrics(
        rows,
        run_state,
        dataset_name=dataset_name,
        baseline_compare_path=baseline_compare_path,
        expected_schema_version=expected_schema_version,
        bootstrap_iterations=bootstrap_iterations,
        run_state_path=run_state_path,
    )
    return rows, run_state, metrics


def write_evaluation_summary(
    metrics: dict[str, Any],
    *,
    dataset_name: str | None = None,
    filename: str = "evaluation_summary.json",
) -> Path:
    resolved_dataset = dataset_name or metrics.get("dataset")
    if not resolved_dataset:
        raise ValueError("dataset_name is required when metrics do not include a dataset field.")
    output_path = METRICS_DIR / resolved_dataset / filename
    dump_json(output_path, metrics)
    return output_path
