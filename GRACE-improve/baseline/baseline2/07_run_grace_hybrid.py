import json
import os
import time
from pathlib import Path

import numpy as np

from common import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, RETRIEVAL_DIR, SPLITS_DIR, dump_json, ensure_dir, get_record_code, iter_jsonl
from graphs import get_graph_features
from hybrid_prefilter import DEFAULT_PREFILTER_MODEL_NAME, predict_feature_store
from local_llm_client import CACHE_SCHEMA_VERSION, DEFAULT_MODEL_REPO_ID, LocalVulnLLMClassifier, build_detection_prompt, default_local_model_dir
from localizer import locate_suspicious_slices
from metrics import apply_calibrator, apply_platt_scaler, bootstrap_f1_interval, compute_binary_metrics
from retrieval import DEMO_BANK_SCHEMA_VERSION, load_demo_bank, retrieve_examples


PREDICTION_SCHEMA_VERSION = 1


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
PREFILTER_MODEL_NAME = os.getenv("GRACE_PREFILTER_MODEL_NAME", DEFAULT_PREFILTER_MODEL_NAME)
LLM_MODEL_NAME = os.getenv("GRACE_LOCAL_MODEL_ID", DEFAULT_MODEL_REPO_ID)
LOCAL_MODEL_DIR = default_local_model_dir(LLM_MODEL_NAME)
GRAPH_BACKEND = os.getenv("GRACE_GRAPH_BACKEND", "auto")
MAX_TEST_SAMPLES = _env_int("GRACE_MAX_TEST_SAMPLES", None)
TEST_CHUNK_SIZE = _env_int("GRACE_TEST_CHUNK_SIZE", None)
TEST_CHUNK_INDEX = _env_int("GRACE_TEST_CHUNK_INDEX", None)
INSPECT_DEMOS = int(os.getenv("GRACE_INSPECT_DEMOS", "3"))
HIGH_RISK_DEMOS = int(os.getenv("GRACE_HIGH_RISK_DEMOS", "5"))
DEMO_CHAR_LIMIT = int(os.getenv("GRACE_DEMO_CHAR_LIMIT", "800"))
MAX_NEW_TOKENS = int(os.getenv("GRACE_MAX_NEW_TOKENS", "160"))
LOAD_IN_4BIT = _env_flag("GRACE_LOAD_IN_4BIT", True)
AUTO_DOWNLOAD_MODEL = _env_flag("GRACE_AUTO_DOWNLOAD_MODEL", False)
CALL_LLM_FOR_INSPECT = _env_flag("GRACE_CALL_LLM_FOR_INSPECT", True)
CALL_LLM_FOR_HIGH = _env_flag("GRACE_CALL_LLM_FOR_HIGH", False)
RESUME = _env_flag("GRACE_RESUME", True)
VARIANT_SUFFIX = os.getenv("GRACE_VARIANT_OUTPUT_SUFFIX", "").strip().lower()
PREDICTION_FILE_STEM = os.getenv("GRACE_PREDICTION_FILE_STEM") or (f"grace_hybrid_predictions_{VARIANT_SUFFIX}" if VARIANT_SUFFIX else "grace_hybrid_predictions")
RUN_STATE_FILE_STEM = os.getenv("GRACE_RUN_STATE_FILE_STEM") or (f"grace_hybrid_run_state_{VARIANT_SUFFIX}" if VARIANT_SUFFIX else "grace_hybrid_run_state")
METRICS_FILE_STEM = os.getenv("GRACE_EVALUATION_FILE_STEM") or (f"grace_hybrid_evaluation_summary_{VARIANT_SUFFIX}" if VARIANT_SUFFIX else "grace_hybrid_evaluation_summary")
EVIDENCE_AWARE_VERIFIER = _env_flag("GRACE_EVIDENCE_AWARE_VERIFIER", False)


def _risk_band(probability: float, tau_low: float, tau_high: float) -> str:
    if probability <= tau_low:
        return "skip"
    if probability >= tau_high:
        return "high"
    return "inspect"


def _should_call_llm(risk_band: str) -> bool:
    if risk_band == "inspect":
        return CALL_LLM_FOR_INSPECT
    if risk_band == "high":
        return CALL_LLM_FOR_HIGH
    return False


def _direct_prefilter_prediction(risk_band: str) -> int:
    if risk_band == "high":
        return 1
    return 0


def _apply_saved_calibrator(score: float, calibration: dict) -> float:
    if "calibrator" in calibration:
        calibrated = apply_calibrator([score], calibration["calibrator"])
        return float(calibrated[0])
    if "platt_scaler" in calibration:
        calibrated = apply_platt_scaler([score], calibration["platt_scaler"])
        return float(calibrated[0])
    return float(score)


def _has_positive_evidence(result: dict) -> bool:
    vulnerable_lines = result.get("vulnerable_lines") or []
    if not vulnerable_lines:
        return False
    sink_or_api = str(result.get("sink_or_api") or "").strip()
    missing_guard = str(result.get("missing_guard") or "").strip()
    return bool(sink_or_api or missing_guard)


def _verified_llm_prediction(result: dict) -> tuple[int, str, str]:
    prediction = int(result.get("label_int", 0))
    reason = str(result.get("reason") or "").strip()
    if not EVIDENCE_AWARE_VERIFIER:
        return prediction, "disabled", reason
    if prediction == 1 and not _has_positive_evidence(result):
        return 0, "rejected_missing_evidence", f"evidence_verifier_rejected_positive: {reason}" if reason else "evidence_verifier_rejected_positive"
    return prediction, "accepted", reason


def _count_records(split_path: Path, limit: int | None = None) -> int:
    total = 0
    for record in iter_jsonl(split_path):
        if not get_record_code(record):
            continue
        total += 1
        if limit is not None and total >= limit:
            break
    return total


def _resolve_chunk_bounds(total_records: int) -> tuple[int, int] | None:
    if TEST_CHUNK_SIZE is None or TEST_CHUNK_INDEX is None:
        return None
    if TEST_CHUNK_SIZE <= 0:
        raise ValueError("GRACE_TEST_CHUNK_SIZE must be a positive integer.")
    if TEST_CHUNK_INDEX < 0:
        raise ValueError("GRACE_TEST_CHUNK_INDEX must be a non-negative integer.")
    start = TEST_CHUNK_INDEX * TEST_CHUNK_SIZE
    end = min(total_records, start + TEST_CHUNK_SIZE)
    return start, end


def _select_target_record_ids(split_path: Path, *, limit: int | None = None) -> tuple[set[str], int, dict | None]:
    total_records = _count_records(split_path, limit=limit)
    bounds = _resolve_chunk_bounds(total_records)
    if bounds is None:
        selected = set()
        seen = 0
        for record in iter_jsonl(split_path):
            if not get_record_code(record):
                continue
            selected.add(str(record["record_id"]))
            seen += 1
            if limit is not None and seen >= limit:
                break
        return selected, total_records, None

    start, end = bounds
    selected = set()
    seen = 0
    for record in iter_jsonl(split_path):
        if not get_record_code(record):
            continue
        if seen >= end:
            break
        if seen >= start:
            selected.add(str(record["record_id"]))
        seen += 1
        if limit is not None and seen >= limit:
            break
    chunk_context = {
        "chunk_index": TEST_CHUNK_INDEX,
        "chunk_size": TEST_CHUNK_SIZE,
        "start_offset": start,
        "end_offset_exclusive": end,
        "target_records_in_chunk": len(selected),
    }
    return selected, total_records, chunk_context


def _load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _prepare_prediction_file(predictions_path: Path) -> list[dict]:
    if not RESUME or not predictions_path.exists():
        return []
    rows = _load_predictions(predictions_path)
    return [row for row in rows if row.get("schema_version") == PREDICTION_SCHEMA_VERSION]


def _build_run_signature(calibration: dict, bank: dict) -> dict:
    return {
        "dataset": DATASET_NAME,
        "variant_suffix": VARIANT_SUFFIX,
        "prefilter_model_name": PREFILTER_MODEL_NAME,
        "llm_model_name": LLM_MODEL_NAME,
        "graph_backend": GRAPH_BACKEND,
        "retrieval_backend": bank.get("semantic_backend"),
        "tau_low": float(calibration["tau_low"]),
        "tau_high": float(calibration["tau_high"]),
        "calibration_method": calibration.get("calibration_method"),
        "routing_mode": calibration.get("routing_mode"),
        "inspect_demos": INSPECT_DEMOS,
        "high_risk_demos": HIGH_RISK_DEMOS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "load_in_4bit": LOAD_IN_4BIT,
        "call_llm_for_inspect": CALL_LLM_FOR_INSPECT,
        "call_llm_for_high": CALL_LLM_FOR_HIGH,
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def _summarize_predictions(
    predictions_path: Path,
    total_target_records: int,
    output_dir: Path,
    bank: dict,
    *,
    run_signature: dict,
    chunk_context: dict | None = None,
) -> dict:
    rows = _load_predictions(predictions_path)
    routing = {"skip": 0, "inspect": 0, "high": 0}
    decision_sources = {"prefilter": 0, "llm": 0}
    llm_cache_hits = 0
    llm_calls = 0
    for row in rows:
        routing[row["risk_band"]] = routing.get(row["risk_band"], 0) + 1
        decision_sources[row["decision_source"]] = decision_sources.get(row["decision_source"], 0) + 1
        if row.get("llm_cache_hit"):
            llm_cache_hits += 1
        if row.get("llm_called"):
            llm_calls += 1
    summary = {
        "dataset": DATASET_NAME,
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "resolved_samples": len(rows),
        "target_samples": total_target_records,
        "complete": len(rows) == total_target_records,
        "evaluation_ready": len(rows) == total_target_records,
        "llm_calls": llm_calls,
        "llm_cache_hits": llm_cache_hits,
        "llm_call_ratio": float(llm_calls / max(len(rows), 1)),
        "routing": routing,
        "decision_sources": decision_sources,
        "retrieval_backend": bank.get("semantic_backend"),
        "graph_backend_requested": GRAPH_BACKEND,
        "predictions_path": str(predictions_path),
        "chunking": chunk_context,
        "run_signature": run_signature,
        "config": {
            "prefilter_model_name": PREFILTER_MODEL_NAME,
            "llm_model_name": LLM_MODEL_NAME,
            "inspect_demos": INSPECT_DEMOS,
            "high_risk_demos": HIGH_RISK_DEMOS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "load_in_4bit": LOAD_IN_4BIT,
            "call_llm_for_inspect": CALL_LLM_FOR_INSPECT,
            "call_llm_for_high": CALL_LLM_FOR_HIGH,
        },
    }
    if rows:
        labels = [int(row["ground_truth"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        probabilities = [float(row["calibrated_probability"]) for row in rows]
        graph_times = [float(row.get("graph_latency_ms") or 0.0) for row in rows]
        retrieval_times = [float(row.get("retrieval_latency_ms") or 0.0) for row in rows]
        llm_times = [float(row.get("llm_latency_ms") or 0.0) for row in rows]
        total_times = [float(row.get("record_runtime_ms") or 0.0) for row in rows]
        summary["timing_ms"] = {
            "graph_total": float(sum(graph_times)),
            "graph_mean": _mean(graph_times),
            "retrieval_total": float(sum(retrieval_times)),
            "retrieval_mean": _mean(retrieval_times),
            "llm_total": float(sum(llm_times)),
            "llm_mean": _mean(llm_times),
            "record_total": float(sum(total_times)),
            "record_mean": _mean(total_times),
        }
        summary.update(compute_binary_metrics(labels, predictions, probabilities))
        summary["bootstrap_f1"] = bootstrap_f1_interval(labels, predictions, iterations=500)
    dump_json(output_dir / f"{RUN_STATE_FILE_STEM}.json", summary)
    dump_json(METRICS_DIR / DATASET_NAME / f"{METRICS_FILE_STEM}.json", summary)
    return summary


def main() -> None:
    calibration_path = MODELS_DIR / DATASET_NAME / f"calibration.{PREFILTER_MODEL_NAME}.json"
    bank_path = RETRIEVAL_DIR / DATASET_NAME / "demo_bank.joblib"
    test_path = SPLITS_DIR / DATASET_NAME / "test.jsonl"
    if not calibration_path.exists():
        raise FileNotFoundError(f"Missing calibration file: {calibration_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"Missing demo bank: {bank_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test split: {test_path}")

    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    tau_low = float(calibration["tau_low"])
    tau_high = float(calibration["tau_high"])
    bank = load_demo_bank(bank_path)
    if bank.get("schema_version") != DEMO_BANK_SCHEMA_VERSION:
        raise RuntimeError("Demo bank schema mismatch. Rebuild `06_build_demo_bank.py`.")
    run_signature = _build_run_signature(calibration, bank)

    test_predictions = predict_feature_store(DATASET_NAME, "test", model_name=PREFILTER_MODEL_NAME)
    score_map = {
        record_id: {
            "fusion_score": float(fusion),
            "semantic_score": float(semantic),
            "graph_score": float(graph),
        }
        for record_id, fusion, semantic, graph in zip(
            test_predictions["record_ids"],
            test_predictions["fusion_score"],
            test_predictions["semantic_score"],
            test_predictions["graph_score"],
        )
    }

    client = None
    if CALL_LLM_FOR_INSPECT or CALL_LLM_FOR_HIGH:
        client = LocalVulnLLMClassifier(
            model_name=LLM_MODEL_NAME,
            model_dir=LOCAL_MODEL_DIR,
            max_new_tokens=MAX_NEW_TOKENS,
            load_in_4bit=LOAD_IN_4BIT,
            auto_download=AUTO_DOWNLOAD_MODEL,
        )
        client.prepare()

    output_dir = ensure_dir(PREDICTIONS_DIR / DATASET_NAME)
    predictions_path = output_dir / f"{PREDICTION_FILE_STEM}.jsonl"
    existing_rows = _prepare_prediction_file(predictions_path)
    processed_ids = {row["record_id"] for row in existing_rows}
    target_record_ids, total_target_records, chunk_context = _select_target_record_ids(test_path, limit=MAX_TEST_SAMPLES)
    chunk_target_records = len(target_record_ids)
    if not target_record_ids:
        print(
            json.dumps(
                {
                    "dataset": DATASET_NAME,
                    "message": "No records selected for this chunk configuration.",
                    "chunking": chunk_context,
                    "total_target_records": total_target_records,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        summary = _summarize_predictions(
            predictions_path,
            total_target_records,
            output_dir,
            bank,
            run_signature=run_signature,
            chunk_context=chunk_context,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    run_state_path = output_dir / f"{RUN_STATE_FILE_STEM}.json"
    if RESUME and predictions_path.exists() and run_state_path.exists():
        try:
            previous_run_state = json.loads(run_state_path.read_text(encoding="utf-8"))
        except Exception:
            previous_run_state = {}
        previous_signature = previous_run_state.get("run_signature")
        if previous_signature and previous_signature != run_signature:
            print(
                json.dumps(
                    {
                        "message": "Existing predictions were produced by a different run signature. Starting a fresh prediction file.",
                        "previous_run_signature": previous_signature,
                        "current_run_signature": run_signature,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            existing_rows = []
            processed_ids = set()

    already_done_in_chunk = len([record_id for record_id in target_record_ids if record_id in processed_ids])
    print(
        f"[config] dataset={DATASET_NAME} | model={PREFILTER_MODEL_NAME} | llm={LLM_MODEL_NAME} "
        f"| total_target_records={total_target_records} | chunk_target_records={chunk_target_records} "
        f"| chunk={chunk_context if chunk_context is not None else 'full'} "
        f"| graph={GRAPH_BACKEND} | retrieval={bank.get('semantic_backend')}"
    )

    file_mode = "a" if predictions_path.exists() and processed_ids else "w"
    with predictions_path.open(file_mode, encoding="utf-8") as handle:
        seen = 0
        chunk_processed = already_done_in_chunk
        for record in iter_jsonl(test_path):
            code = get_record_code(record)
            if not code:
                continue
            seen += 1
            if MAX_TEST_SAMPLES is not None and seen > MAX_TEST_SAMPLES:
                break
            if record["record_id"] not in target_record_ids:
                continue
            if record["record_id"] in processed_ids:
                continue

            record_started = time.perf_counter()
            scores = score_map[record["record_id"]]
            calibrated_probability = _apply_saved_calibrator(scores["fusion_score"], calibration)
            band = _risk_band(calibrated_probability, tau_low, tau_high)
            call_llm = _should_call_llm(band)
            direct_prediction = _direct_prefilter_prediction(band)

            graph_latency_ms = 0.0
            retrieval_latency_ms = 0.0
            llm_latency_ms = 0.0
            graph_features = None
            suspicious_context = None
            retrieved_examples = []

            if not call_llm:
                prefilter_reason = "prefilter_direct_positive" if direct_prediction == 1 else "prefilter_skip"
                payload = {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "resolution_status": "resolved",
                    "record_id": record["record_id"],
                    "dataset": record.get("dataset", DATASET_NAME),
                    "ground_truth": int(record["label"]),
                    "prediction": direct_prediction,
                    "decision_source": "prefilter",
                    "risk_band": band,
                    "llm_called": False,
                    "llm_cache_hit": False,
                    "prefilter_fusion_score": float(scores["fusion_score"]),
                    "prefilter_semantic_score": float(scores["semantic_score"]),
                    "prefilter_graph_score": float(scores["graph_score"]),
                    "calibrated_probability": calibrated_probability,
                    "retrieved_examples": [],
                    "graph_backend_used": None,
                    "graph_latency_ms": graph_latency_ms,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "record_runtime_ms": float(round((time.perf_counter() - record_started) * 1000.0, 3)),
                    "reason": prefilter_reason,
                }
            else:
                graph_started = time.perf_counter()
                graph_features = get_graph_features({**record, "code": code}, graph_backend=GRAPH_BACKEND)
                graph_latency_ms = float(round((time.perf_counter() - graph_started) * 1000.0, 3))

                suspicious_context = locate_suspicious_slices(
                    code,
                    semantic_score=scores["semantic_score"],
                    graph_score=scores["graph_score"],
                    fusion_score=calibrated_probability,
                    risk_band=band,
                )
                retrieval_started = time.perf_counter()
                retrieved_examples = retrieve_examples(
                    code,
                    bank,
                    total_k=HIGH_RISK_DEMOS if band == "high" else INSPECT_DEMOS,
                    calibrated_probability=calibrated_probability,
                    demo_char_limit=DEMO_CHAR_LIMIT,
                    query_record={**record, "code": code},
                    graph_backend=GRAPH_BACKEND,
                    query_graph_features=graph_features,
                )
                retrieval_latency_ms = float(round((time.perf_counter() - retrieval_started) * 1000.0, 3))
                prompt = build_detection_prompt(
                    {**record, "code": code},
                    retrieved_examples,
                    calibrated_probability,
                    band,
                    graph_features=graph_features,
                    suspicious_context=suspicious_context,
                    semantic_score=scores["semantic_score"],
                    graph_score=scores["graph_score"],
                    fusion_score=scores["fusion_score"],
                )
                llm_started = time.perf_counter()
                result = client.classify(prompt)
                llm_latency_ms = float(round((time.perf_counter() - llm_started) * 1000.0, 3))
                verified_prediction, evidence_status, verified_reason = _verified_llm_prediction(result)
                payload = {
                    "schema_version": PREDICTION_SCHEMA_VERSION,
                    "resolution_status": "resolved",
                    "record_id": record["record_id"],
                    "dataset": record.get("dataset", DATASET_NAME),
                    "ground_truth": int(record["label"]),
                    "prediction": int(verified_prediction),
                    "decision_source": "llm",
                    "risk_band": band,
                    "llm_called": True,
                    "llm_cache_hit": bool(result.get("cached")),
                    "prefilter_fusion_score": float(scores["fusion_score"]),
                    "prefilter_semantic_score": float(scores["semantic_score"]),
                    "prefilter_graph_score": float(scores["graph_score"]),
                    "calibrated_probability": calibrated_probability,
                    "retrieved_examples": [row["record_id"] for row in retrieved_examples],
                    "graph_backend_used": graph_features.get("backend"),
                    "graph_latency_ms": graph_latency_ms,
                    "retrieval_latency_ms": retrieval_latency_ms,
                    "llm_latency_ms": llm_latency_ms,
                    "record_runtime_ms": float(round((time.perf_counter() - record_started) * 1000.0, 3)),
                    "reason": verified_reason or result["reason"],
                    "llm_label": result["label"],
                    "llm_confidence": float(result["confidence"]),
                    "llm_evidence_status": evidence_status,
                    "llm_cwe_family": result.get("cwe_family"),
                    "llm_vulnerable_lines": result.get("vulnerable_lines"),
                    "llm_sink_or_api": result.get("sink_or_api"),
                    "llm_missing_guard": result.get("missing_guard"),
                    "suspicious_top_lines": suspicious_context.get("top_lines"),
                }

            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
            processed_ids.add(record["record_id"])
            chunk_processed += 1
            print(
                f"[progress] global={len(processed_ids)}/{total_target_records} "
                f"| chunk={chunk_processed}/{chunk_target_records} | record_id={record['record_id']} "
                f"| band={band} | decision={payload['decision_source']} | calibrated={calibrated_probability:.4f}"
            )

    summary = _summarize_predictions(
        predictions_path,
        total_target_records,
        output_dir,
        bank,
        run_signature=run_signature,
        chunk_context=chunk_context,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
