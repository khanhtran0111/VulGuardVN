from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

try:
    from scipy.stats import binomtest
except Exception:
    binomtest = None


def _safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def _safe_probabilities(probabilities: list[float] | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def brier_score(labels: list[int] | np.ndarray, probabilities: list[float] | np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = _safe_probabilities(probabilities)
    return float(np.mean((p - y) ** 2))


def negative_log_likelihood(labels: list[int] | np.ndarray, probabilities: list[float] | np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    p = _safe_probabilities(probabilities)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error(labels: list[int] | np.ndarray, probabilities: list[float] | np.ndarray, *, bins: int = 10) -> float:
    y = np.asarray(labels, dtype=int)
    p = _safe_probabilities(probabilities)
    if len(y) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y))
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        if high < 1.0:
            mask = (p >= low) & (p < high)
        else:
            mask = (p >= low) & (p <= high)
        if not np.any(mask):
            continue
        accuracy = float(np.mean(y[mask]))
        confidence = float(np.mean(p[mask]))
        error += abs(accuracy - confidence) * (float(np.sum(mask)) / total)
    return float(error)


def _calibration_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "brier": brier_score(labels, probabilities),
        "nll": negative_log_likelihood(labels, probabilities),
        "ece": expected_calibration_error(labels, probabilities),
    }


def fit_platt_scaler(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    x = _safe_logit(np.asarray(probabilities, dtype=float)).reshape(-1, 1)
    if len(np.unique(y)) < 2:
        return {"coef": 1.0, "intercept": 0.0}
    model = LogisticRegression(max_iter=2000, solver="lbfgs")
    model.fit(x, y)
    return {
        "coef": float(model.coef_[0][0]),
        "intercept": float(model.intercept_[0]),
    }


def apply_platt_scaler(probabilities: list[float] | np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    logits = _safe_logit(np.asarray(probabilities, dtype=float))
    calibrated = calibration["coef"] * logits + calibration["intercept"]
    return _sigmoid(calibrated)


def fit_temperature_scaler(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> dict[str, float]:
    probs = _safe_probabilities(probabilities)
    y = np.asarray(labels, dtype=int)
    logits = _safe_logit(probs)
    if len(np.unique(y)) < 2:
        return {"temperature": 1.0}
    best_temperature = 1.0
    best_nll = float("inf")
    for temperature in np.linspace(0.5, 5.0, 91):
        calibrated = _sigmoid(logits / float(temperature))
        score = negative_log_likelihood(y, calibrated)
        if score < best_nll:
            best_nll = score
            best_temperature = float(temperature)
    return {"temperature": best_temperature}


def apply_temperature_scaler(probabilities: list[float] | np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    logits = _safe_logit(np.asarray(probabilities, dtype=float))
    temperature = float(calibration.get("temperature", 1.0))
    return _sigmoid(logits / max(temperature, 1e-6))


def fit_beta_calibration(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> dict[str, float]:
    probs = _safe_probabilities(probabilities)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return {"coef_pos": 1.0, "coef_neg": 0.0, "intercept": 0.0}
    features = np.column_stack([np.log(probs), np.log1p(-probs)])
    model = LogisticRegression(max_iter=4000, solver="lbfgs")
    model.fit(features, y)
    return {
        "coef_pos": float(model.coef_[0][0]),
        "coef_neg": float(model.coef_[0][1]),
        "intercept": float(model.intercept_[0]),
    }


def apply_beta_calibration(probabilities: list[float] | np.ndarray, calibration: dict[str, float]) -> np.ndarray:
    probs = _safe_probabilities(probabilities)
    features = np.column_stack([np.log(probs), np.log1p(-probs)])
    logits = (
        float(calibration.get("coef_pos", 1.0)) * features[:, 0]
        + float(calibration.get("coef_neg", 0.0)) * features[:, 1]
        + float(calibration.get("intercept", 0.0))
    )
    return _sigmoid(logits)


def fit_isotonic_calibration(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray) -> dict[str, Any]:
    probs = _safe_probabilities(probabilities)
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) < 2:
        return {"thresholds": [0.0, 1.0], "values": [0.0, 1.0]}
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(probs, y)
    return {
        "thresholds": [float(value) for value in model.X_thresholds_.tolist()],
        "values": [float(value) for value in model.y_thresholds_.tolist()],
    }


def apply_isotonic_calibration(probabilities: list[float] | np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    probs = _safe_probabilities(probabilities)
    thresholds = np.asarray(calibration.get("thresholds", [0.0, 1.0]), dtype=float)
    values = np.asarray(calibration.get("values", [0.0, 1.0]), dtype=float)
    if len(thresholds) == 0 or len(values) == 0:
        return probs
    if len(thresholds) == 1:
        return np.full_like(probs, float(values[0]), dtype=float)
    return np.interp(probs, thresholds, values, left=float(values[0]), right=float(values[-1]))


def fit_calibrator(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray, method: str = "auto") -> dict[str, Any]:
    requested = (method or "platt").strip().lower()
    probs = _safe_probabilities(probabilities)
    y = np.asarray(labels, dtype=int)
    if requested == "platt":
        return {"method": "platt", "parameters": fit_platt_scaler(probs, y)}
    if requested == "temperature":
        return {"method": "temperature", "parameters": fit_temperature_scaler(probs, y)}
    if requested == "beta":
        return {"method": "beta", "parameters": fit_beta_calibration(probs, y)}
    if requested == "isotonic":
        return {"method": "isotonic", "parameters": fit_isotonic_calibration(probs, y)}
    if requested != "auto":
        raise ValueError(f"Unsupported calibration method: {method}")

    candidates = [
        {"method": "platt", "parameters": fit_platt_scaler(probs, y)},
        {"method": "temperature", "parameters": fit_temperature_scaler(probs, y)},
        {"method": "isotonic", "parameters": fit_isotonic_calibration(probs, y)},
        {"method": "beta", "parameters": fit_beta_calibration(probs, y)},
    ]
    scored = []
    for candidate in candidates:
        calibrated = apply_calibrator(probs, candidate)
        metrics = _calibration_metrics(y, calibrated)
        scored_candidate = dict(candidate)
        scored_candidate["metrics"] = metrics
        scored.append(scored_candidate)
    best = min(scored, key=lambda item: (float(item["metrics"]["nll"]), float(item["metrics"]["ece"]), float(item["metrics"]["brier"])))
    best["method_requested"] = "auto"
    best["candidate_metrics"] = {item["method"]: item["metrics"] for item in scored}
    return best


def apply_calibrator(probabilities: list[float] | np.ndarray, calibration: dict[str, Any]) -> np.ndarray:
    method = str(calibration.get("method", "platt")).strip().lower()
    params = calibration.get("parameters", calibration)
    if method == "platt":
        return apply_platt_scaler(probabilities, params)
    if method == "temperature":
        return apply_temperature_scaler(probabilities, params)
    if method == "isotonic":
        return apply_isotonic_calibration(probabilities, params)
    if method == "beta":
        return apply_beta_calibration(probabilities, params)
    if method == "identity":
        return np.asarray(probabilities, dtype=float)
    raise ValueError(f"Unsupported calibration method: {method}")


def choose_low_threshold(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray, target_recall: float) -> float:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    positives = probs[y == 1]
    if len(positives) == 0:
        return 0.0
    best = 0.0
    for threshold in np.unique(np.round(np.sort(probs), 6)):
        recall = float(np.mean(positives > threshold))
        if recall >= target_recall:
            best = float(threshold)
        else:
            break
    return best


def choose_high_threshold(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray, minimum: float) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    candidates = np.unique(np.round(np.sort(probs), 6))
    best_threshold = max(minimum + 0.05, 0.5)
    best_f1 = -1.0
    for threshold in candidates:
        if threshold <= minimum:
            continue
        predictions = (probs >= threshold).astype(int)
        score = f1_score(y, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    if best_f1 < 0:
        predictions = (probs >= 0.5).astype(int)
        best_threshold = max(minimum + 0.05, 0.5)
        best_f1 = float(f1_score(y, predictions, zero_division=0))
    return best_threshold, best_f1


def choose_best_f1_threshold(
    probabilities: list[float] | np.ndarray,
    labels: list[int] | np.ndarray,
    minimum: float = 0.0,
) -> tuple[float, float]:
    probs = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    candidates = np.unique(np.round(np.sort(probs), 6))
    best_threshold = float(minimum)
    best_f1 = -1.0
    for threshold in candidates:
        if threshold < minimum:
            continue
        predictions = (probs >= threshold).astype(int)
        score = f1_score(y, predictions, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    if best_f1 < 0:
        predictions = (probs >= minimum).astype(int)
        best_f1 = float(f1_score(y, predictions, zero_division=0))
    return best_threshold, best_f1


def compute_binary_metrics(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray, probabilities: list[float] | np.ndarray | None = None) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }
    if probabilities is not None:
        probs = np.asarray(probabilities, dtype=float)
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        except Exception:
            metrics["roc_auc"] = None
        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, probs))
        except Exception:
            metrics["pr_auc"] = None
        metrics["brier"] = brier_score(y_true, probs)
        metrics["nll"] = negative_log_likelihood(y_true, probs)
        metrics["ece"] = expected_calibration_error(y_true, probs)
    return metrics


def bootstrap_f1_interval(labels: list[int] | np.ndarray, predictions: list[int] | np.ndarray, iterations: int = 1000, seed: int = 42) -> dict[str, float]:
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    if len(y_true) == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(iterations):
        sample = rng.integers(0, len(y_true), size=len(y_true))
        scores.append(f1_score(y_true[sample], y_pred[sample], zero_division=0))
    low, high = np.percentile(scores, [2.5, 97.5]).tolist()
    return {
        "mean": float(np.mean(scores)),
        "low": float(low),
        "high": float(high),
    }


def mcnemar_exact(labels: list[int] | np.ndarray, predictions_a: list[int] | np.ndarray, predictions_b: list[int] | np.ndarray) -> dict[str, float | int | None]:
    y_true = np.asarray(labels, dtype=int)
    a = np.asarray(predictions_a, dtype=int)
    b = np.asarray(predictions_b, dtype=int)
    a_correct = a == y_true
    b_correct = b == y_true
    n01 = int(np.sum(a_correct & ~b_correct))
    n10 = int(np.sum(~a_correct & b_correct))
    total = n01 + n10
    if total == 0:
        return {"n01": n01, "n10": n10, "p_value": 1.0}
    if binomtest is not None:
        p_value = float(binomtest(min(n01, n10), total, 0.5, alternative="two-sided").pvalue)
    else:
        p_value = None
    return {"n01": n01, "n10": n10, "p_value": p_value}
