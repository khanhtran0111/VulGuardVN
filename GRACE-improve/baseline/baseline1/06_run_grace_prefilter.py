import json
import os
import time
from pathlib import Path

import numpy as np

from common import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, RETRIEVAL_DIR, SPLITS_DIR, dump_json, ensure_dir, get_record_code, iter_jsonl, tokenize_code
from graphs import get_graph_features
from local_llm_client import CACHE_SCHEMA_VERSION, DEFAULT_MODEL_REPO_ID, LocalVulnLLMClassifier, build_detection_prompt, default_local_model_dir
from metrics import apply_platt_scaler, bootstrap_f1_interval, compute_binary_metrics
from retrieval import DEMO_BANK_SCHEMA_VERSION, load_demo_bank, retrieve_examples
from train_prefilter import load_prefilter_model


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    return int(lowered)


DATASET_NAME = os.getenv("GRACE_DATASET", "devign")
MODEL_NAME = os.getenv("GRACE_LOCAL_MODEL_ID", DEFAULT_MODEL_REPO_ID)
LOCAL_MODEL_DIR = default_local_model_dir(MODEL_NAME)
MAX_TEST_SAMPLES = _env_int("GRACE_MAX_TEST_SAMPLES", None)
BATCH_SIZE = 256
UNCERTAIN_DEMOS = 2
HIGH_RISK_DEMOS = 4
MAX_NEW_TOKENS = 128
LOAD_IN_4BIT = _env_flag("GRACE_LOAD_IN_4BIT", True)
AUTO_DOWNLOAD_MODEL_IF_MISSING = _env_flag("GRACE_AUTO_DOWNLOAD_MODEL", False)
CALL_LLM_FOR_UNCERTAIN = _env_flag("GRACE_CALL_LLM_FOR_UNCERTAIN", True)
CALL_LLM_FOR_HIGH_RISK = _env_flag("GRACE_CALL_LLM_FOR_HIGH_RISK", False)
RESUME_FROM_EXISTING = _env_flag("GRACE_RESUME", True)
GRAPH_BACKEND = os.getenv("GRACE_GRAPH_BACKEND", "auto")

PREDICTION_SCHEMA_VERSION = 4


def _risk_band(probability: float, tau_low: float, tau_high: float) -> str:
    if probability <= tau_low:
        return "skip"
    if probability >= tau_high:
        return "high"
    return "uncertain"


def _should_call_llm(risk_band: str) -> bool:
    if risk_band == "uncertain":
        return CALL_LLM_FOR_UNCERTAIN
    if risk_band == "high":
        return CALL_LLM_FOR_HIGH_RISK
    return False


def _experiment_mode() -> str:
    if CALL_LLM_FOR_UNCERTAIN and CALL_LLM_FOR_HIGH_RISK:
        return "full_prefilter_plus_local_llm"
    if CALL_LLM_FOR_UNCERTAIN and not CALL_LLM_FOR_HIGH_RISK:
        return "local_uncertain_only"
    if not CALL_LLM_FOR_UNCERTAIN and CALL_LLM_FOR_HIGH_RISK:
        return "local_high_risk_only"
    return "prefilter_only"


def _build_stop_reason(record_id: str, exc: Exception) -> tuple[str, str]:
    lowered = str(exc).lower()
    if "missing local llm dependencies" in lowered:
        return (
            "missing_dependencies",
            "Local LLM dependencies are missing. Install torch, transformers, accelerate, bitsandbytes, and huggingface_hub, then rerun script 06.",
        )
    if "local model directory is missing or incomplete" in lowered:
        return (
            "model_missing",
            "Local model weights are missing. Run `python GRACE-improve/baseline/download_vulnllm_r_7b.py` first, then rerun script 06.",
        )
    if "out of memory" in lowered or "cuda out of memory" in lowered:
        return (
            "cuda_oom",
            (
                f"CUDA OOM while resolving record {record_id}. Close GPU-heavy apps, reduce MAX_NEW_TOKENS or demo count, "
                "then rerun script 06."
            ),
        )
    return (
        "local_llm_failed",
        (
            f"Local LLM failed to resolve record {record_id}. "
            "Check grace_prefilter_last_issue.json, fix the runtime issue, and rerun script 06."
        ),
    )


def _iter_records_with_probabilities(model, split_path, batch_size: int, limit: int | None = None):
    batch_records = []
    batch_texts = []
    seen = 0
    for record in iter_jsonl(split_path):
        code = get_record_code(record)
        if not code:
            continue
        canonical = dict(record)
        canonical["code"] = code
        batch_records.append(canonical)
        batch_texts.append(" ".join(tokenize_code(code)))
        seen += 1
        if len(batch_texts) >= batch_size:
            probs = model.predict(np.asarray(batch_texts, dtype=object), batch_size=batch_size, verbose=0).reshape(-1)
            for item, prob in zip(batch_records, probs.tolist()):
                yield item, float(prob)
            batch_records = []
            batch_texts = []
        if limit is not None and seen >= limit:
            break
    if batch_texts:
        probs = model.predict(np.asarray(batch_texts, dtype=object), batch_size=batch_size, verbose=0).reshape(-1)
        for item, prob in zip(batch_records, probs.tolist()):
            yield item, float(prob)


def _count_records(split_path: Path, limit: int | None = None) -> int:
    total = 0
    for record in iter_jsonl(split_path):
        if not get_record_code(record):
            continue
        total += 1
        if limit is not None and total >= limit:
            break
    return total


def _load_predictions(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_predictions(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _is_compatible_prediction_row(row: dict) -> bool:
    required_fields = {
        "schema_version",
        "record_id",
        "ground_truth",
        "prediction",
        "risk_band",
        "resolution_status",
        "decision_source",
        "calibrated_probability",
    }
    return (
        isinstance(row, dict)
        and row.get("schema_version") == PREDICTION_SCHEMA_VERSION
        and row.get("resolution_status") == "resolved"
        and required_fields.issubset(row.keys())
    )


def _prepare_predictions_file(predictions_path: Path) -> list[dict]:
    if not RESUME_FROM_EXISTING or not predictions_path.exists():
        return []
    existing_rows = _load_predictions(predictions_path)
    if not existing_rows:
        return []
    if not all(_is_compatible_prediction_row(row) for row in existing_rows):
        backup_path = predictions_path.with_name(f"{predictions_path.stem}.legacy_{int(time.time())}{predictions_path.suffix}")
        predictions_path.replace(backup_path)
        print(f"[reset] moved incompatible predictions to {backup_path}")
        return []
    deduped_rows = list({row["record_id"]: row for row in existing_rows}.values())
    if len(deduped_rows) != len(existing_rows):
        _write_predictions(predictions_path, deduped_rows)
    return deduped_rows


def _build_direct_result(band: str, probability: float) -> tuple[int, dict]:
    if band == "skip":
        return 0, {
            "label": "Non-vulnerable",
            "label_int": 0,
            "confidence": max(0.0, 1.0 - float(probability)),
            "reason": "prefilter_skip",
        }
    return 1, {
        "label": "Vulnerable",
        "label_int": 1,
        "confidence": float(probability),
        "reason": "prefilter_high_risk_accept",
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _summarize_predictions(
    predictions_path: Path,
    total_target_records: int,
    output_dir: Path,
    stopped_reason: str | None,
    api_requests_made_this_run: int,
    last_issue_path: Path,
    bank: dict,
) -> dict:
    rows = _load_predictions(predictions_path)
    routing = {"skip": 0, "uncertain": 0, "high": 0}
    decision_sources = {"prefilter": 0, "llm": 0}
    llm_cache_hits = 0
    graph_backend_counts = {}
    for row in rows:
        routing[row["risk_band"]] = routing.get(row["risk_band"], 0) + 1
        decision_sources[row["decision_source"]] = decision_sources.get(row["decision_source"], 0) + 1
        if row.get("llm_cache_hit"):
            llm_cache_hits += 1
        graph_backend = row.get("graph_backend_used")
        if graph_backend:
            graph_backend_counts[graph_backend] = graph_backend_counts.get(graph_backend, 0) + 1
    complete = len(rows) >= total_target_records
    graph_times = [float(row.get("graph_latency_ms") or 0.0) for row in rows]
    retrieval_times = [float(row.get("retrieval_latency_ms") or 0.0) for row in rows]
    llm_times = [float(row.get("llm_latency_ms") or 0.0) for row in rows]
    total_times = [float(row.get("record_runtime_ms") or 0.0) for row in rows]
    summary = {
        "dataset": DATASET_NAME,
        "model_name": MODEL_NAME,
        "backend": "local_transformers",
        "experiment_mode": _experiment_mode(),
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "resolved_samples": len(rows),
        "target_samples": total_target_records,
        "complete": complete,
        "evaluation_ready": complete and stopped_reason is None,
        "llm_calls": sum(1 for row in rows if row.get("llm_called")),
        "api_requests_made": sum(1 for row in rows if row.get("api_request_made")),
        "api_requests_made_this_run": api_requests_made_this_run,
        "llm_cache_hits": llm_cache_hits,
        "llm_call_ratio": float(sum(1 for row in rows if row.get("llm_called")) / max(len(rows), 1)),
        "routing": routing,
        "decision_sources": decision_sources,
        "retrieval_backend": bank.get("semantic_backend"),
        "retrieval_notice": bank.get("semantic_notice"),
        "graph_backend_requested": bank.get("graph_backend_requested"),
        "graph_backend_counts": graph_backend_counts,
        "timing_ms": {
            "graph_total": float(sum(graph_times)),
            "graph_mean": _mean(graph_times),
            "retrieval_total": float(sum(retrieval_times)),
            "retrieval_mean": _mean(retrieval_times),
            "llm_total": float(sum(llm_times)),
            "llm_mean": _mean(llm_times),
            "record_total": float(sum(total_times)),
            "record_mean": _mean(total_times),
        },
        "predictions_path": str(predictions_path),
        "stopped_reason": stopped_reason,
        "last_issue_path": str(last_issue_path) if last_issue_path.exists() else None,
        "config": {
            "local_model_dir": str(LOCAL_MODEL_DIR),
            "retrieval_backend": bank.get("semantic_backend"),
            "graph_backend": GRAPH_BACKEND,
            "max_test_samples": MAX_TEST_SAMPLES,
            "call_llm_for_uncertain": CALL_LLM_FOR_UNCERTAIN,
            "call_llm_for_high_risk": CALL_LLM_FOR_HIGH_RISK,
            "uncertain_demos": UNCERTAIN_DEMOS,
            "high_risk_demos": HIGH_RISK_DEMOS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "load_in_4bit": LOAD_IN_4BIT,
            "auto_download_model_if_missing": AUTO_DOWNLOAD_MODEL_IF_MISSING,
        },
    }
    if rows:
        labels = [int(row["ground_truth"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        probabilities = [float(row["calibrated_probability"]) for row in rows]
        resolved_metrics = compute_binary_metrics(labels, predictions, probabilities)
        summary["resolved_metrics"] = resolved_metrics
        if complete:
            summary.update(resolved_metrics)
            summary["bootstrap_f1"] = bootstrap_f1_interval(labels, predictions, iterations=500)
        else:
            summary["bootstrap_f1"] = None
    else:
        summary["resolved_metrics"] = None
        summary["bootstrap_f1"] = None
    dump_json(METRICS_DIR / DATASET_NAME / "grace_prefilter_metrics.json", summary)
    dump_json(output_dir / "grace_prefilter_run_state.json", summary)
    return summary


def main() -> None:
    model_path = MODELS_DIR / DATASET_NAME / "prefilter_cnn_model"
    calibration_path = MODELS_DIR / DATASET_NAME / "calibration.json"
    bank_path = RETRIEVAL_DIR / DATASET_NAME / "demo_bank.joblib"
    test_path = SPLITS_DIR / DATASET_NAME / "test.jsonl"
    if not model_path.exists() or not calibration_path.exists() or not bank_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing prefilter model, calibration, demo bank, or test split.")

    model = load_prefilter_model(model_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    tau_low = float(calibration["tau_low"])
    tau_high = float(calibration["tau_high"])
    bank = load_demo_bank(bank_path)
    if bank.get("schema_version") != DEMO_BANK_SCHEMA_VERSION:
        raise RuntimeError(
            "Demo bank schema mismatch. Rebuild retrieval artifacts with `python GRACE-improve/baseline/05_build_demo_bank.py`."
        )

    client = None
    client_info = None
    if CALL_LLM_FOR_UNCERTAIN or CALL_LLM_FOR_HIGH_RISK:
        client = LocalVulnLLMClassifier(
            model_name=MODEL_NAME,
            model_dir=LOCAL_MODEL_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            load_in_4bit=LOAD_IN_4BIT,
            auto_download=AUTO_DOWNLOAD_MODEL_IF_MISSING,
        )
        client_info = client.prepare()

    output_dir = ensure_dir(PREDICTIONS_DIR / DATASET_NAME)
    predictions_path = output_dir / "grace_prefilter_predictions.jsonl"
    last_issue_path = output_dir / "grace_prefilter_last_issue.json"
    if last_issue_path.exists():
        last_issue_path.unlink()
    total_target_records = _count_records(test_path, limit=MAX_TEST_SAMPLES)

    existing_rows = _prepare_predictions_file(predictions_path)
    processed_ids = {row["record_id"] for row in existing_rows}
    if existing_rows:
        print(f"[resume] found {len(existing_rows)} compatible resolved predictions in {predictions_path}")
    else:
        print(f"[start] writing predictions to {predictions_path}")

    if len(processed_ids) >= total_target_records:
        summary = _summarize_predictions(
            predictions_path,
            total_target_records,
            output_dir,
            stopped_reason=None,
            api_requests_made_this_run=0,
            last_issue_path=last_issue_path,
            bank=bank,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    file_mode = "a" if predictions_path.exists() and processed_ids else "w"
    processed_this_run = 0
    api_requests_made_this_run = 0
    stopped_reason = None

    print(
        f"[config] dataset={DATASET_NAME}"
        f" | model={MODEL_NAME}"
        f" | backend=local_transformers"
        f" | experiment_mode={_experiment_mode()}"
        f" | total_target_records={total_target_records}"
        f" | max_new_tokens={MAX_NEW_TOKENS}"
        f" | load_in_4bit={LOAD_IN_4BIT}"
        f" | call_llm_uncertain={CALL_LLM_FOR_UNCERTAIN}"
        f" | call_llm_high={CALL_LLM_FOR_HIGH_RISK}"
        f" | retrieval_backend={bank.get('semantic_backend')}"
        f" | graph_backend={GRAPH_BACKEND}"
        f" | device={client_info['device'] if client_info else 'n/a'}"
    )

    with predictions_path.open(file_mode, encoding="utf-8") as handle:
        for record, raw_probability in _iter_records_with_probabilities(model, test_path, batch_size=BATCH_SIZE, limit=MAX_TEST_SAMPLES):
            if record["record_id"] in processed_ids:
                continue

            record_started = time.perf_counter()
            probability = float(apply_platt_scaler([raw_probability], calibration["platt_scaler"])[0])
            band = _risk_band(probability, tau_low, tau_high)
            call_llm = _should_call_llm(band)
            examples = []
            graph_features = None
            graph_latency_ms = 0.0
            retrieval_latency_ms = 0.0
            llm_latency_ms = 0.0

            if not call_llm:
                final_label, result = _build_direct_result(band, probability)
                payload = {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "resolution_status": "resolved",
                    "decision_source": "prefilter",
                    "record_id": record["record_id"],
                    "dataset": record["dataset"],
                    "project": record.get("project", ""),
                    "ground_truth": int(record["label"]),
                    "prefilter_probability": float(raw_probability),
                    "calibrated_probability": probability,
                    "risk_band": band,
                    "llm_called": False,
                    "api_request_made": False,
                    "llm_cache_hit": False,
                    "llm_finish_reason": None,
                    "llm_backend": "prefilter",
                    "llm_device": None,
                    "llm_prompt_tokens": None,
                    "llm_generated_tokens": None,
                    "graph_backend_used": None,
                    "graph_nodes": 0,
                    "graph_edges": 0,
                    "graph_latency_ms": graph_latency_ms,
                    "retrieval_backend": bank.get("semantic_backend"),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "record_runtime_ms": float(round((time.perf_counter() - record_started) * 1000.0, 3)),
                    "retrieved_examples": [],
                    "prediction": final_label,
                    "llm_label": result["label"],
                    "llm_confidence": float(result["confidence"]),
                    "reason": result["reason"],
                }
            else:
                total_k = HIGH_RISK_DEMOS if band == "high" else UNCERTAIN_DEMOS
                graph_started = time.perf_counter()
                graph_features = get_graph_features(record, graph_backend=GRAPH_BACKEND)
                graph_latency_ms = float(round((time.perf_counter() - graph_started) * 1000.0, 3))
                retrieval_started = time.perf_counter()
                examples = retrieve_examples(
                    record["code"],
                    bank,
                    total_k=total_k,
                    calibrated_probability=probability,
                    query_record=record,
                    graph_backend=GRAPH_BACKEND,
                    query_graph_features=graph_features,
                )
                retrieval_latency_ms = float(round((time.perf_counter() - retrieval_started) * 1000.0, 3))
                prompt = build_detection_prompt(record, examples, probability, band, graph_features=graph_features)
                prompt_hash = client.prompt_hash(prompt)
                cached_before_call = client.is_cached(prompt)
                api_request_made = not cached_before_call
                if api_request_made:
                    api_requests_made_this_run += 1
                try:
                    llm_started = time.perf_counter()
                    result = client.classify(prompt)
                    llm_latency_ms = float(round((time.perf_counter() - llm_started) * 1000.0, 3))
                except Exception as exc:
                    error_kind, stopped_reason = _build_stop_reason(record["record_id"], exc)
                    dump_json(
                        last_issue_path,
                        {
                            "record_id": record["record_id"],
                            "risk_band": band,
                            "calibrated_probability": probability,
                            "prompt_hash": prompt_hash,
                            "retrieved_examples": [example["record_id"] for example in examples],
                            "graph_backend_used": graph_features.get("backend") if graph_features else None,
                            "graph_latency_ms": graph_latency_ms,
                            "retrieval_latency_ms": retrieval_latency_ms,
                            "api_request_attempted": api_request_made,
                            "cached_before_call": cached_before_call,
                            "error_kind": error_kind,
                            "error": str(exc),
                        },
                    )
                    print(f"[stop] {stopped_reason}")
                    break

                usage_metadata = result.get("usage_metadata") or {}
                payload = {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "resolution_status": "resolved",
                    "decision_source": "llm",
                    "record_id": record["record_id"],
                    "dataset": record["dataset"],
                    "project": record.get("project", ""),
                    "ground_truth": int(record["label"]),
                    "prefilter_probability": float(raw_probability),
                    "calibrated_probability": probability,
                    "risk_band": band,
                    "llm_called": True,
                    "api_request_made": api_request_made,
                    "llm_cache_hit": bool(result.get("cached")),
                    "llm_finish_reason": result.get("finish_reason"),
                    "llm_backend": "local_transformers",
                    "llm_device": result.get("device"),
                    "llm_prompt_tokens": usage_metadata.get("prompt_tokens"),
                    "llm_generated_tokens": usage_metadata.get("generated_tokens"),
                    "graph_backend_used": graph_features.get("backend"),
                    "graph_nodes": int(graph_features.get("graph_summary", {}).get("nodes", 0)),
                    "graph_edges": int(graph_features.get("graph_summary", {}).get("edges", 0)),
                    "graph_latency_ms": graph_latency_ms,
                    "retrieval_backend": bank.get("semantic_backend"),
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "record_runtime_ms": float(round((time.perf_counter() - record_started) * 1000.0, 3)),
                    "retrieved_examples": [example["record_id"] for example in examples],
                    "prediction": int(result["label_int"]),
                    "llm_label": result["label"],
                    "llm_confidence": float(result["confidence"]),
                    "reason": result["reason"],
                }

            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()

            processed_ids.add(record["record_id"])
            processed_this_run += 1
            total_done = len(processed_ids)
            decision = payload["decision_source"]
            print(
                f"[progress] {total_done}/{total_target_records}"
                f" | record_id={record['record_id']}"
                f" | band={band}"
                f" | decision={decision}"
                f" | retrieval={bank.get('semantic_backend')}"
                f" | graph={payload.get('graph_backend_used') or 'n/a'}"
                f" | calibrated_probability={probability:.4f}"
            )

    summary = _summarize_predictions(
        predictions_path,
        total_target_records,
        output_dir,
        stopped_reason=stopped_reason,
        api_requests_made_this_run=api_requests_made_this_run,
        last_issue_path=last_issue_path,
        bank=bank,
    )
    summary["processed_this_run"] = processed_this_run
    summary["remaining_samples"] = max(0, total_target_records - summary["resolved_samples"])
    if not summary["evaluation_ready"]:
        if stopped_reason and "download_vulnllm_r_7b.py" in stopped_reason:
            summary["next_action"] = "Download the local model, then rerun script 06 until evaluation_ready=true, then run script 07."
        elif stopped_reason and "dependencies" in stopped_reason.lower():
            summary["next_action"] = "Install local LLM dependencies, then rerun script 06 until evaluation_ready=true, then run script 07."
        elif stopped_reason and "cuda oom" in stopped_reason.lower():
            summary["next_action"] = "Free GPU memory or reduce generation settings, then rerun script 06 until evaluation_ready=true, then run script 07."
        else:
            summary["next_action"] = "Fix the local LLM runtime issue, rerun script 06 until evaluation_ready=true, then run script 07."
    dump_json(METRICS_DIR / DATASET_NAME / "grace_prefilter_metrics.json", summary)
    dump_json(output_dir / "grace_prefilter_run_state.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
