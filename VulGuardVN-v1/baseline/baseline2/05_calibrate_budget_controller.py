import json
import os
from pathlib import Path

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
TAU_LOW_OVERRIDE = float(os.getenv("GRACE_TAU_LOW")) if os.getenv("GRACE_TAU_LOW") else None
TAU_HIGH_OVERRIDE = float(os.getenv("GRACE_TAU_HIGH")) if os.getenv("GRACE_TAU_HIGH") else None
CALIBRATION_OUTPUT_PATH = Path(os.getenv("GRACE_CALIBRATION_OUTPUT_PATH")) if os.getenv("GRACE_CALIBRATION_OUTPUT_PATH") else None
FIGURES_DIR = Path(os.getenv("GRACE_FIGURES_DIR")) if os.getenv("GRACE_FIGURES_DIR") else None


def _candidate_grid(low: float, high: float, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.asarray([float(low)], dtype=np.float32)
    return np.unique(np.round(np.linspace(low, high, steps), 6))


def _validate_threshold_overrides(tau_low: float | None, tau_high: float | None) -> bool:
    configured = tau_low is not None or tau_high is not None
    if configured and (
        tau_low is None
        or tau_high is None
        or not 0.0 <= tau_low < tau_high <= 1.0
    ):
        raise ValueError(
            "GRACE_TAU_LOW and GRACE_TAU_HIGH must both be set with "
            "0 <= tau_low < tau_high <= 1"
        )
    return configured


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

    tau_low = float(choose_low_threshold(probabilities, labels, ROUTING_RECALL_FLOOR))
    tau_high, tau_high_best_f1 = choose_best_f1_threshold(probabilities, labels, minimum=max(tau_low, DIRECT_ACCEPT_MIN_PROBABILITY))
    if tau_high <= tau_low:
        tau_high = min(1.0, tau_low + 1e-6)
    if tau_high <= tau_low:
        raise RuntimeError("Automatic threshold selection could not satisfy tau_low < tau_high")
    fallback_metrics = _routing_stats(probabilities, labels, tau_low, tau_high)
    fallback_metrics["tau_high_best_f1"] = float(tau_high_best_f1)
    return tau_low, float(tau_high), "fallback_f1", fallback_metrics


def _save_calibration_figures(
    raw_scores: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    tau_low: float,
    tau_high: float,
    figures_dir: Path,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception as exc:
        print(f"[calibration] figure generation skipped: {exc}")
        return []

    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(probabilities[labels == 0], bins=30, alpha=0.65, label="non-vulnerable")
    ax.hist(probabilities[labels == 1], bins=30, alpha=0.65, label="vulnerable")
    ax.axvline(tau_low, color="tab:blue", linestyle="--", label="tau_low")
    ax.axvline(tau_high, color="tab:red", linestyle="--", label="tau_high")
    ax.set(xlabel="Calibrated probability", ylabel="Count", title=f"{DATASET_NAME}: probability distribution")
    ax.legend()
    path = figures_dir / "probability_histogram.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    fraction_positive, mean_predicted = calibration_curve(labels, probabilities, n_bins=10, strategy="uniform")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
    ax.plot(mean_predicted, fraction_positive, marker="o", label="model")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive rate", title=f"{DATASET_NAME}: reliability")
    ax.legend()
    path = figures_dir / "reliability_diagram.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    fpr, tpr, _ = roc_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(fpr, tpr); ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f"{DATASET_NAME}: ROC")
    path = figures_dir / "roc_curve.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))

    precision, recall, _ = precision_recall_curve(labels, probabilities)
    fig, ax = plt.subplots(figsize=(5, 5)); ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title=f"{DATASET_NAME}: precision-recall")
    path = figures_dir / "pr_curve.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); paths.append(str(path))
    return paths


def main() -> None:
    predictions = predict_feature_store(DATASET_NAME, "val", model_name=MODEL_NAME)
    labels = np.asarray(predictions["labels"], dtype=np.int32)
    record_ids = [str(value) for value in predictions["record_ids"]]
    fusion_scores = np.asarray(predictions["fusion_score"], dtype=np.float32)
    semantic_scores = np.asarray(predictions.get("semantic_score", np.full(len(labels), np.nan)), dtype=np.float32)
    graph_scores = np.asarray(predictions.get("graph_score", np.full(len(labels), np.nan)), dtype=np.float32)

    calibration = _calibration_payload(fusion_scores, labels)
    calibrated = np.asarray(calibration["calibrated"], dtype=np.float32)
    if _validate_threshold_overrides(TAU_LOW_OVERRIDE, TAU_HIGH_OVERRIDE):
        tau_low, tau_high, tau_strategy = TAU_LOW_OVERRIDE, TAU_HIGH_OVERRIDE, "configured_validation_operating_point"
        routing_metrics = _routing_stats(calibrated, labels, tau_low, tau_high)
    elif ROUTING_MODE == "constrained":
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
        if tau_high <= tau_low:
            tau_high = min(1.0, tau_low + 1e-6)
        if tau_high <= tau_low:
            raise RuntimeError("Automatic threshold selection could not satisfy tau_low < tau_high")
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
        "threshold_selection_strategy": tau_strategy,
        "threshold_selection_split": "validation",
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
            "semantic_score_mean": None if np.all(np.isnan(semantic_scores)) else float(np.nanmean(semantic_scores)),
            "graph_score_mean": None if np.all(np.isnan(graph_scores)) else float(np.nanmean(graph_scores)),
        },
        "llm_budget_estimate": {
            "keep_ratio": float(np.mean(calibrated > tau_low)),
            "high_ratio": float(np.mean(calibrated >= tau_high)),
            "inspect_ratio": float(np.mean((calibrated > tau_low) & (calibrated < tau_high))),
        },
        "routing_metrics": routing_metrics,
        "probability_records": [
            {
                "record_id": record_id,
                "ground_truth": int(label),
                "raw_fusion_score": float(raw),
                "calibrated_probability": float(probability),
                "calibration_method": calibration["calibrator"]["method"],
            }
            for record_id, label, raw, probability in zip(record_ids, labels, fusion_scores, calibrated)
        ],
    }
    output_path = CALIBRATION_OUTPUT_PATH or (MODELS_DIR / DATASET_NAME / f"calibration.{MODEL_NAME}.json")
    figures_dir = FIGURES_DIR or (output_path.parent / "figures")
    summary["figures"] = _save_calibration_figures(fusion_scores, calibrated, labels, tau_low, tau_high, figures_dir)
    dump_json(output_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
