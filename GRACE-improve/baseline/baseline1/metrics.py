from typing import Any

import numpy as np
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
    return 1.0 / (1.0 + np.exp(-calibrated))


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
