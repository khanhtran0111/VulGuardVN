from pathlib import Path
import math
import shutil
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from common import MODELS_DIR, SPLITS_DIR, dump_json, ensure_dir, get_record_code, iter_jsonl, load_json, tokenize_code


def _build_model(max_tokens: int, sequence_length: int, embedding_dim: int, filters: int, dense_units: int, dropout_rate: float) -> keras.Model:
    vectorizer = keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=sequence_length,
        max_tokens=max_tokens,
    )
    inputs = keras.Input(shape=(), dtype=tf.string, name="code_tokens")
    x = vectorizer(inputs)
    x = keras.layers.Embedding(max_tokens, embedding_dim)(x)
    x = keras.layers.SpatialDropout1D(dropout_rate)(x)
    x = keras.layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
    x = keras.layers.Conv1D(filters, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalMaxPooling1D()(x)
    x = keras.layers.Dense(dense_units, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.vectorizer = vectorizer
    return model


def _iter_tokenized_examples(split_path: Path, limit: int | None = None):
    seen = 0
    for record in iter_jsonl(split_path):
        code = get_record_code(record)
        if not code:
            continue
        yield " ".join(tokenize_code(code)), float(record["label"])
        seen += 1
        if limit is not None and seen >= limit:
            break


def _iter_tokenized_texts(split_path: Path, limit: int | None = None):
    for text, _ in _iter_tokenized_examples(split_path, limit=limit):
        yield text


def _count_labels(split_path: Path, limit: int | None = None) -> tuple[int, int]:
    total = 0
    positives = 0
    seen = 0
    for record in iter_jsonl(split_path):
        code = get_record_code(record)
        if not code:
            continue
        label = int(record["label"])
        total += 1
        positives += label
        seen += 1
        if limit is not None and seen >= limit:
            break
    return total, positives


def _build_dataset(
    split_path: Path,
    batch_size: int,
    *,
    shuffle: bool,
    positive_weight: float = 1.0,
    limit: int | None = None,
    shuffle_seed: int = 42,
    repeat: bool = False,
) -> tf.data.Dataset:
    def generator():
        for text, label in _iter_tokenized_examples(split_path, limit=limit):
            weight = positive_weight if label == 1.0 else 1.0
            yield text, np.float32(label), np.float32(weight)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    if shuffle:
        dataset = dataset.shuffle(10000, seed=shuffle_seed, reshuffle_each_iteration=True)
    if repeat:
        dataset = dataset.repeat()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _build_text_dataset(split_path: Path, batch_size: int, limit: int | None = None) -> tf.data.Dataset:
    dataset = tf.data.Dataset.from_generator(
        lambda: _iter_tokenized_texts(split_path, limit=limit),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string),
    )
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {seconds:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes}m {seconds:.1f}s"


def _format_metrics(logs: dict | None) -> str:
    if not logs:
        return "no metrics"
    metric_order = (
        "loss",
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "val_loss",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_roc_auc",
        "val_pr_auc",
    )
    formatted: list[str] = []
    for key in metric_order:
        value = logs.get(key)
        if value is not None:
            formatted.append(f"{key}={float(value):.4f}")
    return " | ".join(formatted) if formatted else "no metrics"


def _get_learning_rate(optimizer: keras.optimizers.Optimizer) -> float | None:
    try:
        return float(tf.keras.backend.get_value(optimizer.learning_rate))
    except Exception:
        return None


class ProgressLogger(keras.callbacks.Callback):
    def __init__(self, *, steps_per_epoch: int, validation_steps: int, batch_log_interval: int | None = None) -> None:
        super().__init__()
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.batch_log_interval = batch_log_interval
        self.current_epoch = 0
        self.total_epochs = "?"
        self.epoch_start_time = 0.0

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get("epochs", "?")
        if self.batch_log_interval is None:
            self.batch_log_interval = max(1, self.steps_per_epoch // 10)
        print(
            "[train] started"
            f" | epochs={self.total_epochs}"
            f" | steps_per_epoch={self.steps_per_epoch}"
            f" | validation_steps={self.validation_steps}"
            f" | batch_log_interval={self.batch_log_interval}"
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        self.epoch_start_time = time.time()
        print(f"[epoch {self.current_epoch}/{self.total_epochs}] started")

    def on_train_batch_end(self, batch, logs=None):
        if not self.batch_log_interval or self.batch_log_interval < 1:
            return
        batch_index = batch + 1
        if batch_index % self.batch_log_interval != 0 and batch_index != self.steps_per_epoch:
            return
        print(
            f"[epoch {self.current_epoch}/{self.total_epochs}]"
            f" batch {batch_index}/{self.steps_per_epoch}"
            f" | {_format_metrics(logs)}"
        )

    def on_epoch_end(self, epoch, logs=None):
        duration = _format_duration(time.time() - self.epoch_start_time)
        learning_rate = _get_learning_rate(self.model.optimizer)
        learning_rate_text = f"{learning_rate:.2e}" if learning_rate is not None else "n/a"
        print(
            f"[epoch {epoch + 1}/{self.total_epochs}] finished"
            f" | duration={duration}"
            f" | {_format_metrics(logs)}"
            f" | lr={learning_rate_text}"
        )

    def on_train_end(self, logs=None):
        print("[train] finished")


def _save_prefilter_bundle(
    model: keras.Model,
    artifact_dir: Path,
    *,
    max_tokens: int,
    sequence_length: int,
    embedding_dim: int,
    filters: int,
    dense_units: int,
    dropout_rate: float,
) -> None:
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    ensure_dir(artifact_dir)
    vocabulary = model.vectorizer.get_vocabulary()
    (artifact_dir / "vocabulary.txt").write_text("\n".join(vocabulary), encoding="utf-8")
    dump_json(
        artifact_dir / "config.json",
        {
            "max_tokens": max_tokens,
            "sequence_length": sequence_length,
            "embedding_dim": embedding_dim,
            "filters": filters,
            "dense_units": dense_units,
            "dropout_rate": dropout_rate,
        },
    )
    model.save_weights(artifact_dir / "weights.weights.h5")


def load_prefilter_model(artifact_dir: Path) -> keras.Model:
    config = load_json(artifact_dir / "config.json")
    vocabulary = (artifact_dir / "vocabulary.txt").read_text(encoding="utf-8").splitlines()
    model = _build_model(
        max_tokens=int(config["max_tokens"]),
        sequence_length=int(config["sequence_length"]),
        embedding_dim=int(config["embedding_dim"]),
        filters=int(config["filters"]),
        dense_units=int(config["dense_units"]),
        dropout_rate=float(config["dropout_rate"]),
    )
    model.vectorizer.set_vocabulary(vocabulary)
    model.load_weights(artifact_dir / "weights.weights.h5")
    return model


def train_prefilter(
    dataset_name: str,
    batch_size: int = 128,
    max_tokens: int = 40000,
    sequence_length: int = 512,
    embedding_dim: int = 128,
    filters: int = 128,
    dense_units: int = 128,
    dropout_rate: float = 0.2,
    epochs: int = 12,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
    adapt_text_limit: int | None = None,
    train_limit: int | None = None,
    val_limit: int | None = None,
    verbose: int = 1,
    log_progress: bool = False,
    batch_log_interval: int | None = None,
) -> dict:
    tf.keras.utils.set_random_seed(random_seed)
    train_path = SPLITS_DIR / dataset_name / "train.jsonl"
    val_path = SPLITS_DIR / dataset_name / "val.jsonl"
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Missing split files for {dataset_name}. Run 02_create_splits.py first.")
    if adapt_text_limit is None and dataset_name == "bigvul":
        adapt_text_limit = 100000
    train_size, train_positive = _count_labels(train_path, limit=train_limit)
    val_size, val_positive = _count_labels(val_path, limit=val_limit)
    model = _build_model(max_tokens, sequence_length, embedding_dim, filters, dense_units, dropout_rate)
    if log_progress:
        print(
            f"[data] dataset={dataset_name}"
            f" | train={train_size} (pos={train_positive}, neg={train_size - train_positive})"
            f" | val={val_size} (pos={val_positive}, neg={val_size - val_positive})"
        )
        print(
            "[config]"
            f" batch_size={batch_size}"
            f" | epochs={epochs}"
            f" | learning_rate={learning_rate:.2e}"
            f" | max_tokens={max_tokens}"
            f" | sequence_length={sequence_length}"
            f" | embedding_dim={embedding_dim}"
            f" | filters={filters}"
            f" | dense_units={dense_units}"
            f" | dropout_rate={dropout_rate}"
        )
        limit_text = adapt_text_limit if adapt_text_limit is not None else "full train split"
        print(f"[vectorizer] adapting vocabulary on {limit_text}")
    model.vectorizer.adapt(_build_text_dataset(train_path, batch_size=batch_size, limit=adapt_text_limit))
    if log_progress:
        print(f"[vectorizer] ready | vocabulary_size={len(model.vectorizer.get_vocabulary())}")
    negative = float(train_size - train_positive)
    positive = float(train_positive)
    positive_weight = max(1.0, negative / max(positive, 1.0))
    if log_progress:
        print(f"[weights] positive_class_weight={positive_weight:.4f}")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_pr_auc", mode="max", factor=0.5, patience=1, min_lr=1e-5),
    ]
    steps_per_epoch = max(1, math.ceil(train_size / batch_size))
    validation_steps = max(1, math.ceil(val_size / batch_size))
    if log_progress:
        callbacks.append(
            ProgressLogger(
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps,
                batch_log_interval=batch_log_interval,
            )
        )
    train_dataset = _build_dataset(
        train_path,
        batch_size=batch_size,
        shuffle=True,
        positive_weight=positive_weight,
        limit=train_limit,
        shuffle_seed=random_seed,
        repeat=True,
    )
    val_dataset = _build_dataset(
        val_path,
        batch_size=batch_size,
        shuffle=False,
        positive_weight=1.0,
        limit=val_limit,
        repeat=True,
    )
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )
    output_dir = ensure_dir(MODELS_DIR / dataset_name)
    model_path = output_dir / "prefilter_cnn_model"
    _save_prefilter_bundle(
        model,
        model_path,
        max_tokens=max_tokens,
        sequence_length=sequence_length,
        embedding_dim=embedding_dim,
        filters=filters,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )
    best_val_pr_auc = float(max(history.history.get("val_pr_auc", [0.0])))
    best_val_recall = float(max(history.history.get("val_recall", [0.0])))
    summary = {
        "dataset": dataset_name,
        "model_path": str(model_path),
        "train_size": int(train_size),
        "val_size": int(val_size),
        "adapt_text_limit": adapt_text_limit,
        "train_limit": train_limit,
        "val_limit": val_limit,
        "positive_class_weight": float(positive_weight),
        "best_val_pr_auc": best_val_pr_auc,
        "best_val_recall": best_val_recall,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    summary_path = output_dir / "training_summary.json"
    dump_json(summary_path, summary)
    if log_progress:
        print(f"[train] saved model bundle to {model_path}")
        print(f"[train] wrote training summary to {summary_path}")
        print(f"[train] best_val_pr_auc={best_val_pr_auc:.4f} | best_val_recall={best_val_recall:.4f}")
    return summary
