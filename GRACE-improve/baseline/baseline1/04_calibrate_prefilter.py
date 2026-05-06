import json

import numpy as np

from common import MODELS_DIR, SPLITS_DIR, dump_json, get_record_code, iter_jsonl, tokenize_code
from metrics import apply_platt_scaler, choose_high_threshold, choose_low_threshold, compute_binary_metrics, fit_platt_scaler
from train_prefilter import load_prefilter_model


DATASET_NAME = "devign"
TARGET_RECALL = 0.99
BATCH_SIZE = 256


def _predict_split(model, split_path, batch_size: int):
    labels = []
    probabilities = []
    batch_texts = []
    batch_labels = []
    for record in iter_jsonl(split_path):
        code = get_record_code(record)
        if not code:
            continue
        batch_texts.append(" ".join(tokenize_code(code)))
        batch_labels.append(int(record["label"]))
        if len(batch_texts) >= batch_size:
            probs = model.predict(np.asarray(batch_texts, dtype=object), batch_size=batch_size, verbose=0).reshape(-1)
            probabilities.extend(probs.tolist())
            labels.extend(batch_labels)
            batch_texts = []
            batch_labels = []
    if batch_texts:
        probs = model.predict(np.asarray(batch_texts, dtype=object), batch_size=batch_size, verbose=0).reshape(-1)
        probabilities.extend(probs.tolist())
        labels.extend(batch_labels)
    return np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=float)


def main() -> None:
    model_path = MODELS_DIR / DATASET_NAME / "prefilter_cnn_model"
    val_path = SPLITS_DIR / DATASET_NAME / "val.jsonl"
    if not model_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing model or validation split for {DATASET_NAME}.")
    model = load_prefilter_model(model_path)
    labels, raw_probabilities = _predict_split(model, val_path, batch_size=BATCH_SIZE)
    calibration = fit_platt_scaler(raw_probabilities, labels)
    calibrated = apply_platt_scaler(raw_probabilities, calibration)
    tau_low = choose_low_threshold(calibrated, labels, TARGET_RECALL)
    tau_high, best_f1 = choose_high_threshold(calibrated, labels, tau_low)
    summary = {
        "dataset": DATASET_NAME,
        "target_recall": TARGET_RECALL,
        "tau_low": float(tau_low),
        "tau_high": float(tau_high),
        "high_threshold_f1": float(best_f1),
        "platt_scaler": calibration,
        "val_metrics_uncalibrated": compute_binary_metrics(labels, (raw_probabilities >= 0.5).astype(int), raw_probabilities),
        "val_metrics_calibrated": compute_binary_metrics(labels, (calibrated >= tau_high).astype(int), calibrated),
    }
    dump_json(MODELS_DIR / DATASET_NAME / "calibration.json", summary)
    print(f"Saved calibration for {DATASET_NAME} to {MODELS_DIR / DATASET_NAME / 'calibration.json'}")


if __name__ == "__main__":
    main()
