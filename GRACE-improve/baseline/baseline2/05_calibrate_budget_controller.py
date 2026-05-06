import json
import os

import numpy as np

from common import MODELS_DIR, dump_json
from hybrid_prefilter import DEFAULT_PREFILTER_MODEL_NAME, predict_feature_store
from metrics import (
    apply_calibrator,
    apply_platt_scaler,
    choose_best_f1_threshold,
    choose_low_threshold,
    compute_binary_metrics,
    fit_calibrator,
)


DATASET_NAME = os.getenv("GRACE_DATASET", "devign")
MODEL_NAME = os.getenv("GRACE_PREFILTER_MODEL_NAME", DEFAULT_PREFILTER_MODEL_NAME)
CALIBRATION_METHOD = os.getenv("GRACE_CALIBRATION_METHOD", "auto").strip().lower()
TARGET_RECALL = float(os.getenv("GRACE_TARGET_RECALL", "0.995"))
ROUTING_MODE = os.getenv("GRACE_ROUTING_MODE", "baseline").strip().lower()
ROUTING_OBJECTIVE = os.getenv("GRACE_ROUTING_OBJECTIVE", "f1").strip().lower()
ROUTING_INSPECT_PROXY = os.getenv("GRACE_ROUTING_INSPECT_PROXY", "probability").strip().lower()
ROUTING_RECALL_FLOOR = float(os.getenv("GRACE_ROUTING_RECALL_FLOOR", str(TARGET_RECALL)))
LLM_BUDGET = float(os.getenv("GRACE_LLM_BUDGET", "0.15"))
HIGH_RISK_TARGET_PRECISION = float(os.getenv("GRACE_HIGH_RISK_TARGET_PRECISION", "0.70"))
DIRECT_ACCEPT_MIN_PROBABILITY = float(os.getenv("GRACE_DIRECT_ACCEPT_MIN_PROBABILITY", "0.20"))
HIGH_RISK_THRESHOLD_STRATEGY = os.getenv("GRACE_HIGH_RISK_THRESHOLD_STRATEGY", "f1").strip().lower()
TAU_NEG_MIN = float(os.getenv("GRACE_TAU_NEG_MIN", "0.02"))
TAU_NEG_MAX = float(os.getenv("GRACE_TAU_NEG_MAX", "0.30"))
TAU_NEG_STEPS = int(os.getenv("GRACE_TAU_NEG_STEPS", "15"))
TAU_POS_MIN = float(os.getenv("GRACE_TAU_POS_MIN", "0.45"))
TAU_POS_MAX = float(os.getenv("GRACE_TAU_POS_MAX", "0.90"))
TAU_POS_STEPS = int(os.getenv("GRACE_TAU_POS_STEPS", "19"))


def _candidate_grid(low: float, high: float, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.asarray([float(low)], dtype=np.float32)
    return np.unique(np.round(np.linspace(low, high, steps), 6))


def _choose_high_threshold(probabilities: np.ndarray, labels: np.ndarray, tau_low: float, target_precision: float, minimum: float) -> tuple[float, str]:
    best_precision_threshold = None
    for threshold in np.unique(np.round(np.sort(probabilities), 6)):
        if threshold <= max(tau_low, minimum):
            continue
        predictions = (probabilities >= threshold).astype(int)
        metrics = compute_binary_metrics(labels, predictions, probabilities)
        precision = float(metrics["precision"])
        coverage = float(np.mean(probabilities >= threshold))
        if precision >= target_precision and coverage > 0:
            best_precision_threshold = float(threshold)
            break
    if best_precision_threshold is not None:
        return best_precision_threshold, "target_precision"

    best_threshold = max(tau_low + 0.05, minimum)
    best_score = -1.0
    for threshold in np.unique(np.round(np.sort(probabilities), 6)):
        if threshold <= max(tau_low, minimum):
            continue
        predictions = (probabilities >= threshold).astype(int)
        metrics = compute_binary_metrics(labels, predictions, probabilities)
        precision = float(metrics["precision"])
        recall = float(metrics["recall"])
        beta_sq = 0.5 * 0.5
        denominator = beta_sq * precision + recall
        score = 0.0 if denominator == 0 else (1 + beta_sq) * precision * recall / denominator
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold, "f0_5_fallback"


def _calibration_payload(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    calibrator = fit_calibrator(probabilities, labels, method=CALIBRATION_METHOD)
    calibrated = apply_calibrator(probabilities, calibrator)
    return {
        "calibration_method_requested": CALIBRATION_METHOD,
        "calibrator": calibrator,
        "calibrated": calibrated,
        "calibration_metrics": {
            "brier": float(compute_binary_metrics(labels, (calibrated >= 0.5).astype(int), calibrated).get("brier") or 0.0),
            "nll": float(compute_binary_metrics(labels, (calibrated >= 0.5).astype(int), calibrated).get("nll") or 0.0),
            "ece": float(compute_binary_metrics(labels, (calibrated >= 0.5).astype(int), calibrated).get("ece") or 0.0),
        },
    }


def _routing_proxy_predictions(probabilities: np.ndarray, tau_low: float, tau_high: float) -> np.ndarray:
    if ROUTING_INSPECT_PROXY == "positive":
        inspect_pred = np.ones_like(probabilities, dtype=np.int32)
    elif ROUTING_INSPECT_PROXY == "negative":
        inspect_pred = np.zeros_like(probabilities, dtype=np.int32)
    else:
        inspect_pred = (probabilities >= 0.5).astype(np.int32)
    return np.where(probabilities <= tau_low, 0, np.where(probabilities >= tau_high, 1, inspect_pred)).astype(np.int32)


def _routing_stats(probabilities: np.ndarray, labels: np.ndarray, tau_low: float, tau_high: float) -> dict:
    proxy_predictions = _routing_proxy_predictions(probabilities, tau_low, tau_high)
    metrics = compute_binary_metrics(labels, proxy_predictions, probabilities)
    inspect_mask = (probabilities > tau_low) & (probabilities < tau_high)
    metrics["llm_call_rate"] = float(np.mean(inspect_mask))
    metrics["auto_positive_rate"] = float(np.mean(probabilities >= tau_high))
    metrics["auto_negative_rate"] = float(np.mean(probabilities <= tau_low))
    metrics["inspect_rate"] = float(np.mean(inspect_mask))
    return metrics


def _choose_routing_thresholds(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float, str, dict]:
    neg_grid = _candidate_grid(TAU_NEG_MIN, TAU_NEG_MAX, TAU_NEG_STEPS)
    pos_grid = _candidate_grid(TAU_POS_MIN, TAU_POS_MAX, TAU_POS_STEPS)
    best = None
    best_key = None
    best_metrics = None
    for tau_low in neg_grid:
        for tau_high in pos_grid:
            if float(tau_high) <= float(tau_low):
                continue
            metrics = _routing_stats(probabilities, labels, float(tau_low), float(tau_high))
            recall = float(metrics["recall"])
            llm_rate = float(metrics["llm_call_rate"])
            if recall >= ROUTING_RECALL_FLOOR and llm_rate <= LLM_BUDGET:
                objective = float(metrics.get(ROUTING_OBJECTIVE) or 0.0)
                key = (objective, float(metrics["precision"]), float(metrics["accuracy"]), -llm_rate)
                if best_key is None or key > best_key:
                    best_key = key
                    best = (float(tau_low), float(tau_high))
                    best_metrics = metrics
    if best is not None:
        return best[0], best[1], "constrained_search", best_metrics

    fallback_low = float(choose_low_threshold(probabilities, labels, ROUTING_RECALL_FLOOR))
    fallback_high, tau_high_best_f1 = choose_best_f1_threshold(probabilities, labels, minimum=max(fallback_low, DIRECT_ACCEPT_MIN_PROBABILITY))
    fallback_metrics = _routing_stats(probabilities, labels, fallback_low, fallback_high)
    fallback_metrics["tau_high_best_f1"] = float(tau_high_best_f1)
    return fallback_low, float(fallback_high), "fallback_f1", fallback_metrics


def main() -> None:
    predictions = predict_feature_store(DATASET_NAME, "val", model_name=MODEL_NAME)
    labels = np.asarray(predictions["labels"], dtype=np.int32)
    fusion_scores = np.asarray(predictions["fusion_score"], dtype=np.float32)
    semantic_scores = np.asarray(predictions["semantic_score"], dtype=np.float32)
    graph_scores = np.asarray(predictions["graph_score"], dtype=np.float32)

    calibration = _calibration_payload(fusion_scores, labels)
    calibrated = np.asarray(calibration["calibrated"], dtype=np.float32)
    if ROUTING_MODE == "constrained":
        tau_low, tau_high, tau_strategy, routing_metrics = _choose_routing_thresholds(calibrated, labels)
    else:
        tau_low = choose_low_threshold(calibrated, labels, TARGET_RECALL)
        tau_high_minimum = max(float(tau_low), DIRECT_ACCEPT_MIN_PROBABILITY)
        if HIGH_RISK_THRESHOLD_STRATEGY == "precision":
            tau_high, tau_strategy = _choose_high_threshold(
                calibrated,
                labels,
                tau_low=tau_low,
                target_precision=HIGH_RISK_TARGET_PRECISION,
                minimum=tau_high_minimum,
            )
            tau_high_best_f1 = None
        else:
            tau_high, tau_high_best_f1 = choose_best_f1_threshold(
                calibrated,
                labels,
                minimum=tau_high_minimum,
            )
            tau_strategy = "max_f1"
        routing_metrics = _routing_stats(calibrated, labels, tau_low, tau_high)
        routing_metrics["tau_high_best_f1"] = float(tau_high_best_f1) if tau_high_best_f1 is not None else None

    low_predictions = (calibrated > tau_low).astype(int)
    high_predictions = (calibrated >= tau_high).astype(int)
    routing_proxy_predictions = _routing_proxy_predictions(calibrated, tau_low, tau_high)
    summary = {
        "dataset": DATASET_NAME,
        "model_name": MODEL_NAME,
        "target_recall": TARGET_RECALL,
        "routing_mode": ROUTING_MODE,
        "routing_objective": ROUTING_OBJECTIVE,
        "routing_inspect_proxy": ROUTING_INSPECT_PROXY,
        "routing_recall_floor": ROUTING_RECALL_FLOOR,
        "llm_budget": LLM_BUDGET,
        "calibration_method_requested": CALIBRATION_METHOD,
        "calibration_method": calibration["calibrator"]["method"],
        "high_risk_target_precision": HIGH_RISK_TARGET_PRECISION,
        "high_risk_threshold_strategy": HIGH_RISK_THRESHOLD_STRATEGY,
        "direct_accept_min_probability": DIRECT_ACCEPT_MIN_PROBABILITY,
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "tau_high_strategy": tau_strategy,
        "tau_high_best_f1": routing_metrics.get("tau_high_best_f1"),
        "calibrator": calibration["calibrator"],
        "calibration_metrics": calibration["calibration_metrics"],
        "val_metrics_uncalibrated": compute_binary_metrics(labels, (fusion_scores >= 0.5).astype(int), fusion_scores),
        "val_metrics_keep_for_llm": compute_binary_metrics(labels, low_predictions, calibrated),
        "val_metrics_high_risk": compute_binary_metrics(labels, high_predictions, calibrated),
        "val_metrics_direct_accept": compute_binary_metrics(labels, high_predictions, calibrated),
        "val_metrics_routing_proxy": compute_binary_metrics(labels, routing_proxy_predictions, calibrated),
        "branch_means": {
            "fusion_score_mean": float(np.mean(fusion_scores)),
            "semantic_score_mean": float(np.mean(semantic_scores)),
            "graph_score_mean": float(np.mean(graph_scores)),
        },
        "llm_budget_estimate": {
            "keep_ratio": float(np.mean(calibrated > tau_low)),
            "high_ratio": float(np.mean(calibrated >= tau_high)),
            "inspect_ratio": float(np.mean((calibrated > tau_low) & (calibrated < tau_high))),
        },
        "routing_metrics": routing_metrics,
    }
    output_path = MODELS_DIR / DATASET_NAME / f"calibration.{MODEL_NAME}.json"
    dump_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
