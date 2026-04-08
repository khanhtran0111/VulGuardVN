import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from common import METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR, RETRIEVAL_DIR, SPLITS_DIR, dump_json, ensure_dir, get_record_code, iter_jsonl, tokenize_code
from gemini_client import GeminiClassifier, build_detection_prompt
from metrics import apply_platt_scaler, bootstrap_f1_interval, compute_binary_metrics
from retrieval import load_demo_bank, retrieve_examples
from train_prefilter import load_prefilter_model


DATASET_NAME = "devign"
MODEL_NAME = "gemini-2.5-flash"
MAX_TEST_SAMPLES = 200
BATCH_SIZE = 256
UNCERTAIN_DEMOS = 2
HIGH_RISK_DEMOS = 4

# Free-tier friendly defaults.
CALL_LLM_FOR_UNCERTAIN = True
CALL_LLM_FOR_HIGH_RISK = False
REQUESTS_PER_MINUTE_LIMIT = 5
MIN_SECONDS_BETWEEN_REQUESTS = 15.0
MAX_API_CALLS_PER_RUN = 18
RESUME_FROM_EXISTING = True


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


def _summarize_predictions(predictions_path: Path, total_target_records: int, output_dir: Path, stopped_reason: str | None, api_requests_made_this_run: int) -> dict:
    rows = _load_predictions(predictions_path)
    routing = {"skip": 0, "uncertain": 0, "high": 0}
    for row in rows:
        routing[row["risk_band"]] = routing.get(row["risk_band"], 0) + 1
    summary = {
        "dataset": DATASET_NAME,
        "model_name": MODEL_NAME,
        "samples": len(rows),
        "target_samples": total_target_records,
        "complete": len(rows) >= total_target_records,
        "llm_calls": sum(1 for row in rows if row.get("llm_called")),
        "api_requests_made": sum(1 for row in rows if row.get("api_request_made", row.get("llm_called", False))),
        "api_requests_made_this_run": api_requests_made_this_run,
        "llm_call_ratio": float(sum(1 for row in rows if row.get("llm_called")) / max(len(rows), 1)),
        "routing": routing,
        "predictions_path": str(predictions_path),
        "stopped_reason": stopped_reason,
        "config": {
            "max_test_samples": MAX_TEST_SAMPLES,
            "call_llm_for_uncertain": CALL_LLM_FOR_UNCERTAIN,
            "call_llm_for_high_risk": CALL_LLM_FOR_HIGH_RISK,
            "requests_per_minute_limit": REQUESTS_PER_MINUTE_LIMIT,
            "min_seconds_between_requests": MIN_SECONDS_BETWEEN_REQUESTS,
            "max_api_calls_per_run": MAX_API_CALLS_PER_RUN,
            "uncertain_demos": UNCERTAIN_DEMOS,
            "high_risk_demos": HIGH_RISK_DEMOS,
        },
    }
    if rows:
        labels = [int(row["ground_truth"]) for row in rows]
        predictions = [int(row["prediction"]) for row in rows]
        probabilities = [float(row["calibrated_probability"]) for row in rows]
        summary.update(compute_binary_metrics(labels, predictions, probabilities))
        summary["bootstrap_f1"] = bootstrap_f1_interval(labels, predictions, iterations=500)
    else:
        summary.update(
            {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "pr_auc": None,
                "tp": 0,
                "tn": 0,
                "fp": 0,
                "fn": 0,
                "bootstrap_f1": {"mean": 0.0, "low": 0.0, "high": 0.0},
            }
        )
    dump_json(METRICS_DIR / DATASET_NAME / "grace_prefilter_metrics.json", summary)
    dump_json(output_dir / "grace_prefilter_run_state.json", summary)
    return summary


class RequestRateLimiter:
    def __init__(self, requests_per_minute: int, min_seconds_between_requests: float) -> None:
        self.requests_per_minute = requests_per_minute
        self.min_seconds_between_requests = min_seconds_between_requests
        self.request_timestamps: deque[float] = deque()

    def wait_for_slot(self) -> None:
        if self.requests_per_minute <= 0:
            return
        now = time.time()
        while self.request_timestamps and (now - self.request_timestamps[0]) >= 60.0:
            self.request_timestamps.popleft()
        sleep_seconds = 0.0
        if self.request_timestamps:
            sleep_seconds = max(sleep_seconds, self.min_seconds_between_requests - (now - self.request_timestamps[-1]))
        if len(self.request_timestamps) >= self.requests_per_minute:
            sleep_seconds = max(sleep_seconds, 60.0 - (now - self.request_timestamps[0]) + 1.0)
        if sleep_seconds > 0:
            print(f"[rate-limit] sleeping {sleep_seconds:.1f}s before next Gemini request")
            time.sleep(sleep_seconds)
        self.request_timestamps.append(time.time())


def main() -> None:
    model_path = MODELS_DIR / DATASET_NAME / "prefilter_cnn_model"
    calibration_path = MODELS_DIR / DATASET_NAME / "calibration.json"
    bank_path = RETRIEVAL_DIR / DATASET_NAME / "demo_bank.joblib"
    test_path = SPLITS_DIR / DATASET_NAME / "test.jsonl"
    if not model_path.exists() or not calibration_path.exists() or not bank_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing model, calibration, demo bank, or test split.")

    model = load_prefilter_model(model_path)
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    tau_low = float(calibration["tau_low"])
    tau_high = float(calibration["tau_high"])
    bank = load_demo_bank(bank_path)
    client = GeminiClassifier(model_name=MODEL_NAME)
    rate_limiter = RequestRateLimiter(
        requests_per_minute=REQUESTS_PER_MINUTE_LIMIT,
        min_seconds_between_requests=MIN_SECONDS_BETWEEN_REQUESTS,
    )

    output_dir = ensure_dir(PREDICTIONS_DIR / DATASET_NAME)
    predictions_path = output_dir / "grace_prefilter_predictions.jsonl"
    total_target_records = _count_records(test_path, limit=MAX_TEST_SAMPLES)

    existing_rows = _load_predictions(predictions_path) if RESUME_FROM_EXISTING else []
    processed_ids = {row["record_id"] for row in existing_rows}
    if existing_rows:
        print(f"[resume] found {len(existing_rows)} existing predictions in {predictions_path}")
    else:
        print(f"[start] writing predictions to {predictions_path}")

    if len(processed_ids) >= total_target_records:
        summary = _summarize_predictions(predictions_path, total_target_records, output_dir, stopped_reason=None, api_requests_made_this_run=0)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    file_mode = "a" if RESUME_FROM_EXISTING and predictions_path.exists() else "w"
    processed_this_run = 0
    api_requests_made_this_run = 0
    stopped_reason = None

    print(
        f"[config] dataset={DATASET_NAME}"
        f" | model={MODEL_NAME}"
        f" | total_target_records={total_target_records}"
        f" | max_api_calls_per_run={MAX_API_CALLS_PER_RUN}"
        f" | rpm_limit={REQUESTS_PER_MINUTE_LIMIT}"
        f" | min_spacing={MIN_SECONDS_BETWEEN_REQUESTS:.1f}s"
        f" | call_llm_uncertain={CALL_LLM_FOR_UNCERTAIN}"
        f" | call_llm_high={CALL_LLM_FOR_HIGH_RISK}"
    )

    with predictions_path.open(file_mode, encoding="utf-8") as handle:
        for record, raw_probability in _iter_records_with_probabilities(model, test_path, batch_size=BATCH_SIZE, limit=MAX_TEST_SAMPLES):
            if record["record_id"] in processed_ids:
                continue
            probability = float(apply_platt_scaler([raw_probability], calibration["platt_scaler"])[0])
            band = _risk_band(probability, tau_low, tau_high)
            call_llm = _should_call_llm(band)
            examples = []
            api_request_made = False

            if not call_llm:
                final_label, result = _build_direct_result(band, probability)
            else:
                total_k = HIGH_RISK_DEMOS if band == "high" else UNCERTAIN_DEMOS
                examples = retrieve_examples(record["code"], bank, total_k=total_k, calibrated_probability=probability)
                prompt = build_detection_prompt(record, examples, probability, band)
                cached = client.is_cached(prompt)
                if not cached and api_requests_made_this_run >= MAX_API_CALLS_PER_RUN:
                    stopped_reason = (
                        f"Reached MAX_API_CALLS_PER_RUN={MAX_API_CALLS_PER_RUN}. "
                        "Resume tomorrow after the daily quota resets."
                    )
                    print(f"[stop] {stopped_reason}")
                    break
                if not cached:
                    rate_limiter.wait_for_slot()
                    api_requests_made = True
                    api_requests_made_this_run += 1
                try:
                    result = client.classify(prompt)
                except Exception as exc:
                    fallback_label = 1 if band == "high" else int(probability >= 0.5)
                    result = {
                        "label": "Vulnerable" if fallback_label == 1 else "Non-vulnerable",
                        "label_int": fallback_label,
                        "confidence": float(probability),
                        "reason": f"fallback_after_error: {exc}",
                    }
                final_label = int(result["label_int"])

            payload = {
                "record_id": record["record_id"],
                "dataset": record["dataset"],
                "project": record.get("project", ""),
                "ground_truth": int(record["label"]),
                "prefilter_probability": float(raw_probability),
                "calibrated_probability": probability,
                "risk_band": band,
                "llm_called": call_llm,
                "api_request_made": api_request_made,
                "retrieved_examples": [example["record_id"] for example in examples],
                "prediction": final_label,
                "llm_label": result["label"],
                "llm_confidence": float(result["confidence"]),
                "reason": result["reason"],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()

            processed_ids.add(record["record_id"])
            processed_this_run += 1
            total_done = len(processed_ids)
            decision = "llm" if call_llm else "direct"
            print(
                f"[progress] {total_done}/{total_target_records}"
                f" | record_id={record['record_id']}"
                f" | band={band}"
                f" | decision={decision}"
                f" | calibrated_probability={probability:.4f}"
            )

    summary = _summarize_predictions(
        predictions_path,
        total_target_records,
        output_dir,
        stopped_reason=stopped_reason,
        api_requests_made_this_run=api_requests_made_this_run,
    )
    summary["processed_this_run"] = processed_this_run
    summary["remaining_samples"] = max(0, total_target_records - summary["samples"])
    if not summary["complete"]:
        summary["next_action"] = "Run script 06 again after quota reset, then run script 07 only when complete=true."
    dump_json(METRICS_DIR / DATASET_NAME / "grace_prefilter_metrics.json", summary)
    dump_json(output_dir / "grace_prefilter_run_state.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
