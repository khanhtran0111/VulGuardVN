import math
import os
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras

from common import (
    FEATURES_DIR,
    MODELS_DIR,
    RISKY_APIS,
    SPLITS_DIR,
    build_skeleton,
    dump_json,
    ensure_dir,
    extract_calls,
    get_record_code,
    iter_jsonl,
    normalize_code,
    read_jsonl,
    tokenize_code,
)
from graphs import get_graph_features
from retrieval import DEFAULT_RETRIEVAL_MODEL_REPO_ID, SemanticRetrievalEncoder, default_retrieval_model_dir


FEATURE_STORE_SCHEMA_VERSION = 1
PREFILTER_MODEL_SCHEMA_VERSION = 1
DEFAULT_PREFILTER_MODEL_NAME = "hybrid_multiview_prefilter"
DEFAULT_FEATURE_PROGRESS_EVERY = 256

NUMERIC_FEATURE_NAMES = [
    "log_token_count",
    "unique_token_ratio",
    "log_line_count",
    "parameter_count",
    "log_call_count",
    "risky_call_count",
    "risky_call_ratio",
    "control_density",
    "pointer_density",
    "array_access_density",
    "numeric_literal_density",
    "memory_ops_count",
    "backend_is_joern",
    "log_graph_nodes",
    "log_graph_edges",
    "graph_avg_degree",
    "graph_call_ratio",
    "graph_control_ratio",
    "graph_expression_ratio",
    "graph_cfg_ratio",
    "graph_ast_ratio",
    "graph_reaches_ratio",
    "graph_max_out_degree",
    "log_ast_token_count",
]

MEMORY_KEYWORDS = {
    "malloc",
    "calloc",
    "realloc",
    "free",
    "new",
    "delete",
    "memcpy",
    "memmove",
    "memset",
}
CONTROL_TOKENS = {"if", "else", "switch", "case", "for", "while", "do", "goto", "return", "break", "continue"}


def _feature_store_suffix() -> str:
    suffix = os.getenv("GRACE_FEATURE_STORE_SUFFIX", "").strip()
    if not suffix:
        return ""
    return suffix if suffix.startswith("_") else f"_{suffix}"


def feature_store_path(dataset_name: str, split_name: str) -> Path:
    return FEATURES_DIR / dataset_name / f"{split_name}_features{_feature_store_suffix()}.joblib"


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _count_numeric_literals(code: str) -> int:
    total = 0
    for token in tokenize_code(code):
        if token == "num_lit":
            total += 1
    return total


def _compute_numeric_features(code: str, graph_features: dict) -> np.ndarray:
    text = normalize_code(code)
    tokens = tokenize_code(text)
    token_count = max(len(tokens), 1)
    unique_token_ratio = _safe_div(len(set(tokens)), token_count)
    line_count = len([line for line in text.splitlines() if line.strip()])
    calls = extract_calls(text)
    risky_call_count = sum(1 for call in calls if call.lower() in RISKY_APIS)
    control_count = sum(tokens.count(token) for token in CONTROL_TOKENS)
    pointer_ops = text.count("->") + text.count("*") + text.count("&")
    array_accesses = text.count("[")
    numeric_literals = _count_numeric_literals(text)
    memory_ops_count = sum(1 for token in tokens if token in MEMORY_KEYWORDS)

    summary = graph_features.get("graph_summary", {})
    node_types = summary.get("node_types", {})
    edge_types = summary.get("edge_types", {})
    node_count = int(summary.get("nodes", 0))
    edge_count = int(summary.get("edges", 0))
    graph_call_ratio = _safe_div(float(node_types.get("CALL", 0)), node_count)
    graph_control_ratio = _safe_div(float(node_types.get("CONTROL_STRUCTURE", 0)), node_count)
    graph_expression_ratio = _safe_div(float(node_types.get("EXPRESSION", 0)), node_count)
    graph_cfg_ratio = _safe_div(float(edge_types.get("FLOWS_TO", 0)), edge_count)
    graph_ast_ratio = _safe_div(float(edge_types.get("IS_AST_PARENT", 0)), edge_count)
    graph_reaches_ratio = _safe_div(float(edge_types.get("REACHES", 0)), edge_count)
    ast_token_count = len((graph_features.get("ast_sequence") or "").split())

    out_degree: dict[str, int] = {}
    for edge in graph_features.get("edge_rows", []):
        source = str(edge.get("source"))
        out_degree[source] = out_degree.get(source, 0) + 1
    graph_max_out_degree = max(out_degree.values()) if out_degree else 0
    graph_avg_degree = _safe_div(float(sum(out_degree.values())), max(len(out_degree), 1))

    values = [
        math.log1p(token_count),
        unique_token_ratio,
        math.log1p(line_count),
        float(_estimate_parameter_count(text)),
        math.log1p(len(calls)),
        float(risky_call_count),
        _safe_div(float(risky_call_count), max(len(calls), 1)),
        _safe_div(float(control_count), token_count),
        _safe_div(float(pointer_ops), max(line_count, 1)),
        _safe_div(float(array_accesses), max(line_count, 1)),
        _safe_div(float(numeric_literals), token_count),
        float(memory_ops_count),
        1.0 if graph_features.get("backend") == "joern" else 0.0,
        math.log1p(node_count),
        math.log1p(edge_count),
        graph_avg_degree,
        graph_call_ratio,
        graph_control_ratio,
        graph_expression_ratio,
        graph_cfg_ratio,
        graph_ast_ratio,
        graph_reaches_ratio,
        float(graph_max_out_degree),
        math.log1p(ast_token_count),
    ]
    return np.asarray(values, dtype=np.float32)


def _estimate_parameter_count(code: str) -> int:
    signature_head = normalize_code(code)[:1000]
    start = signature_head.find("(")
    end = signature_head.find(")", start + 1)
    if start < 0 or end < 0 or end <= start:
        return 0
    inside = signature_head[start + 1 : end].strip()
    if not inside or inside == "void":
        return 0
    return len([part for part in inside.split(",") if part.strip()])


def _iter_records(split_path: Path, limit: int | None = None) -> Iterable[dict]:
    seen = 0
    for record in iter_jsonl(split_path):
        code = get_record_code(record)
        if not code:
            continue
        payload = dict(record)
        payload["code"] = code
        yield payload
        seen += 1
        if limit is not None and seen >= limit:
            break


def build_feature_store(
    dataset_name: str,
    split_name: str,
    *,
    semantic_model_name: str = DEFAULT_RETRIEVAL_MODEL_REPO_ID,
    semantic_model_dir: Path | None = None,
    graph_backend: str = "auto",
    force_rebuild: bool = False,
    auto_download_semantic_model: bool = False,
    batch_size: int = 16,
    limit: int | None = None,
    progress_every: int = DEFAULT_FEATURE_PROGRESS_EVERY,
) -> dict:
    output_path = feature_store_path(dataset_name, split_name)
    if output_path.exists() and not force_rebuild:
        payload = joblib.load(output_path)
        if payload.get("schema_version") == FEATURE_STORE_SCHEMA_VERSION:
            return payload

    split_path = SPLITS_DIR / dataset_name / f"{split_name}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    encoder = SemanticRetrievalEncoder(
        model_name=semantic_model_name,
        model_dir=semantic_model_dir or default_retrieval_model_dir(semantic_model_name),
        batch_size=batch_size,
        auto_download=auto_download_semantic_model,
    )

    rows = list(_iter_records(split_path, limit=limit))
    if not rows:
        raise RuntimeError(f"No records found for dataset={dataset_name} split={split_name}")

    print(
        f"[feature-store] dataset={dataset_name} split={split_name} "
        f"| rows={len(rows)} | semantic_model={semantic_model_name} | graph={graph_backend}"
    )
    started = time.perf_counter()
    token_texts: list[str] = []
    ast_texts: list[str] = []
    labels: list[int] = []
    record_ids: list[str] = []
    code_hashes: list[str] = []
    graph_backends: list[str] = []
    numeric_features: list[np.ndarray] = []
    codes: list[str] = []
    semantic_inputs: list[str] = []

    for index, record in enumerate(rows, start=1):
        graph_features = get_graph_features(record, graph_backend=graph_backend, force_rebuild=force_rebuild)
        code = record["code"]
        token_texts.append(" ".join(tokenize_code(code)))
        ast_text = graph_features.get("ast_sequence") or build_skeleton(code)
        ast_texts.append(ast_text)
        numeric_features.append(_compute_numeric_features(code, graph_features))
        labels.append(int(record["label"]))
        record_ids.append(str(record["record_id"]))
        code_hashes.append(str(record.get("code_hash") or ""))
        graph_backends.append(str(graph_features.get("backend") or "unknown"))
        semantic_inputs.append(code)
        codes.append(code)
        if progress_every and (index % progress_every == 0 or index == len(rows)):
            elapsed = time.perf_counter() - started
            print(f"[feature-store] prepared graph view {index}/{len(rows)} in {elapsed:.1f}s")

    semantic_embeddings = encoder.encode_texts(semantic_inputs)
    payload = {
        "schema_version": FEATURE_STORE_SCHEMA_VERSION,
        "dataset": dataset_name,
        "split": split_name,
        "semantic_config": encoder.export_config(),
        "graph_backend_requested": graph_backend,
        "record_ids": record_ids,
        "code_hashes": code_hashes,
        "graph_backends": graph_backends,
        "labels": np.asarray(labels, dtype=np.int32),
        "token_texts": token_texts,
        "ast_texts": ast_texts,
        "numeric_features": np.asarray(numeric_features, dtype=np.float32),
        "semantic_embeddings": np.asarray(semantic_embeddings, dtype=np.float32),
        "codes": codes,
        "feature_names": list(NUMERIC_FEATURE_NAMES),
    }
    ensure_dir(output_path.parent)
    joblib.dump(payload, output_path)
    print(f"[feature-store] saved to {output_path}")
    return payload


def load_feature_store(dataset_name: str, split_name: str) -> dict:
    path = feature_store_path(dataset_name, split_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing feature store: {path}")
    payload = joblib.load(path)
    if payload.get("schema_version") != FEATURE_STORE_SCHEMA_VERSION:
        raise RuntimeError(f"Incompatible feature store schema for {path}")
    return payload


def _build_prefilter_model(
    *,
    token_max_tokens: int,
    token_sequence_length: int,
    token_embedding_dim: int,
    token_filters: int,
    ast_max_tokens: int,
    ast_sequence_length: int,
    ast_embedding_dim: int,
    ast_filters: int,
    semantic_dim: int,
    numeric_dim: int,
    projection_dim: int,
    dense_units: int,
    dropout_rate: float,
) -> keras.Model:
    token_vectorizer = keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=token_sequence_length,
        max_tokens=token_max_tokens,
        name="token_vectorizer",
    )
    ast_vectorizer = keras.layers.TextVectorization(
        standardize=None,
        split="whitespace",
        output_mode="int",
        output_sequence_length=ast_sequence_length,
        max_tokens=ast_max_tokens,
        name="ast_vectorizer",
    )

    token_input = keras.Input(shape=(), dtype=tf.string, name="token_text")
    ast_input = keras.Input(shape=(), dtype=tf.string, name="ast_text")
    semantic_input = keras.Input(shape=(semantic_dim,), dtype=tf.float32, name="semantic_embedding")
    numeric_input = keras.Input(shape=(numeric_dim,), dtype=tf.float32, name="numeric_features")

    token_branch = token_vectorizer(token_input)
    token_branch = keras.layers.Embedding(token_max_tokens, token_embedding_dim, name="token_embedding")(token_branch)
    token_branch = keras.layers.SpatialDropout1D(dropout_rate)(token_branch)
    token_branch = keras.layers.Conv1D(token_filters, 5, padding="same", activation="relu")(token_branch)
    token_branch = keras.layers.Conv1D(token_filters, 3, padding="same", activation="relu")(token_branch)
    token_branch = keras.layers.GlobalMaxPooling1D()(token_branch)

    ast_branch = ast_vectorizer(ast_input)
    ast_branch = keras.layers.Embedding(ast_max_tokens, ast_embedding_dim, name="ast_embedding")(ast_branch)
    ast_branch = keras.layers.SpatialDropout1D(dropout_rate)(ast_branch)
    ast_branch = keras.layers.Conv1D(ast_filters, 5, padding="same", activation="relu")(ast_branch)
    ast_branch = keras.layers.Conv1D(ast_filters, 3, padding="same", activation="relu")(ast_branch)
    ast_branch = keras.layers.GlobalMaxPooling1D()(ast_branch)

    semantic_branch = keras.layers.Dense(projection_dim, activation="relu")(semantic_input)
    semantic_branch = keras.layers.Dropout(dropout_rate)(semantic_branch)
    semantic_hidden = keras.layers.Dense(max(32, projection_dim // 2), activation="relu")(semantic_branch)
    semantic_score = keras.layers.Dense(1, activation="sigmoid", name="semantic_score")(semantic_hidden)

    graph_branch = keras.layers.Concatenate(name="graph_concat")([ast_branch, numeric_input])
    graph_branch = keras.layers.Dense(projection_dim, activation="relu")(graph_branch)
    graph_branch = keras.layers.Dropout(dropout_rate)(graph_branch)
    graph_hidden = keras.layers.Dense(max(32, projection_dim // 2), activation="relu")(graph_branch)
    graph_score = keras.layers.Dense(1, activation="sigmoid", name="graph_score")(graph_hidden)

    fusion_branch = keras.layers.Concatenate(name="fusion_concat")(
        [token_branch, ast_branch, semantic_branch, numeric_input, semantic_score, graph_score]
    )
    fusion_branch = keras.layers.Dense(dense_units, activation="relu")(fusion_branch)
    fusion_branch = keras.layers.Dropout(dropout_rate)(fusion_branch)
    fusion_branch = keras.layers.Dense(max(64, dense_units // 2), activation="relu")(fusion_branch)
    fusion_score = keras.layers.Dense(1, activation="sigmoid", name="fusion_score")(fusion_branch)

    model = keras.Model(
        inputs={
            "token_text": token_input,
            "ast_text": ast_input,
            "semantic_embedding": semantic_input,
            "numeric_features": numeric_input,
        },
        outputs={
            "fusion_score": fusion_score,
            "semantic_score": semantic_score,
            "graph_score": graph_score,
        },
    )
    model.token_vectorizer = token_vectorizer
    model.ast_vectorizer = ast_vectorizer
    return model


def _prepare_scaled_inputs(payload: dict, numeric_mean: np.ndarray, numeric_std: np.ndarray) -> dict[str, np.ndarray]:
    scaled_numeric = (payload["numeric_features"] - numeric_mean) / numeric_std
    return {
        "token_text": np.asarray(payload["token_texts"], dtype=object),
        "ast_text": np.asarray(payload["ast_texts"], dtype=object),
        "semantic_embedding": np.asarray(payload["semantic_embeddings"], dtype=np.float32),
        "numeric_features": np.asarray(scaled_numeric, dtype=np.float32),
    }


def _targets(labels: np.ndarray) -> dict[str, np.ndarray]:
    values = labels.astype(np.float32).reshape(-1, 1)
    return {
        "fusion_score": values,
        "semantic_score": values,
        "graph_score": values,
    }


def _sample_weights_from_array(weights: np.ndarray) -> dict[str, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float32)
    return {
        "fusion_score": weights,
        "semantic_score": weights,
        "graph_score": weights,
    }


def _sample_weights(labels: np.ndarray, positive_weight: float, extra_weights: np.ndarray | None = None) -> dict[str, np.ndarray]:
    weights = np.where(labels.astype(int) == 1, positive_weight, 1.0).astype(np.float32)
    if extra_weights is not None:
        weights = weights * np.asarray(extra_weights, dtype=np.float32)
    return _sample_weights_from_array(weights)


def _format_metrics(logs: dict | None) -> str:
    if not logs:
        return "no metrics"
    keys = [
        "loss",
        "fusion_score_loss",
        "fusion_score_pr_auc",
        "fusion_score_recall",
        "val_loss",
        "val_fusion_score_loss",
        "val_fusion_score_pr_auc",
        "val_fusion_score_recall",
    ]
    parts = []
    for key in keys:
        value = logs.get(key)
        if value is not None:
            parts.append(f"{key}={float(value):.4f}")
    return " | ".join(parts) if parts else "no metrics"


class ProgressLogger(keras.callbacks.Callback):
    def __init__(self) -> None:
        super().__init__()
        self.total_epochs = "?"

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get("epochs", "?")
        print(f"[train] started | epochs={self.total_epochs}")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started = time.time()
        print(f"[epoch {epoch + 1}/{self.total_epochs}] started")

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_started
        print(f"[epoch {epoch + 1}/{self.total_epochs}] finished | duration={duration:.1f}s | {_format_metrics(logs)}")

    def on_train_end(self, logs=None):
        print("[train] finished")


def _binary_focal_loss(gamma: float = 2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-6, 1.0 - 1e-6)
        pt = tf.where(tf.equal(y_true, 1.0), y_pred, 1.0 - y_pred)
        ce = keras.backend.binary_crossentropy(y_true, y_pred)
        return tf.pow(1.0 - pt, gamma) * ce

    return loss


def _build_binary_loss(loss_name: str, focal_gamma: float) -> keras.losses.Loss | Any:
    requested = (loss_name or "bce").strip().lower()
    if requested in {"bce", "binary_crossentropy", "weighted_bce"}:
        return keras.losses.BinaryCrossentropy()
    if requested == "focal":
        return _binary_focal_loss(focal_gamma)
    raise ValueError(f"Unsupported loss: {loss_name}")


def train_hybrid_prefilter(
    dataset_name: str,
    *,
    model_name: str = DEFAULT_PREFILTER_MODEL_NAME,
    semantic_model_name: str = DEFAULT_RETRIEVAL_MODEL_REPO_ID,
    token_max_tokens: int = 32000,
    token_sequence_length: int = 384,
    token_embedding_dim: int = 96,
    token_filters: int = 96,
    ast_max_tokens: int = 8000,
    ast_sequence_length: int = 196,
    ast_embedding_dim: int = 64,
    ast_filters: int = 64,
    projection_dim: int = 192,
    dense_units: int = 192,
    dropout_rate: float = 0.25,
    batch_size: int = 128,
    epochs: int = 10,
    learning_rate: float = 7e-4,
    random_seed: int = 42,
    log_progress: bool = True,
    loss_name: str = "bce",
    focal_gamma: float = 2.0,
    hard_negative_mining: bool = False,
    hard_negative_quantile: float = 0.85,
    hard_negative_weight: float = 2.5,
    hard_negative_epochs: int = 2,
) -> dict:
    tf.keras.utils.set_random_seed(random_seed)
    train_payload = load_feature_store(dataset_name, "train")
    val_payload = load_feature_store(dataset_name, "val")

    semantic_dim = int(train_payload["semantic_embeddings"].shape[1])
    numeric_dim = int(train_payload["numeric_features"].shape[1])
    numeric_mean = train_payload["numeric_features"].mean(axis=0).astype(np.float32)
    numeric_std = train_payload["numeric_features"].std(axis=0).astype(np.float32)
    numeric_std = np.where(numeric_std < 1e-6, 1.0, numeric_std)

    model = _build_prefilter_model(
        token_max_tokens=token_max_tokens,
        token_sequence_length=token_sequence_length,
        token_embedding_dim=token_embedding_dim,
        token_filters=token_filters,
        ast_max_tokens=ast_max_tokens,
        ast_sequence_length=ast_sequence_length,
        ast_embedding_dim=ast_embedding_dim,
        ast_filters=ast_filters,
        semantic_dim=semantic_dim,
        numeric_dim=numeric_dim,
        projection_dim=projection_dim,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
    )
    model.token_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_payload["token_texts"]).batch(batch_size))
    model.ast_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_payload["ast_texts"]).batch(batch_size))

    train_inputs = _prepare_scaled_inputs(train_payload, numeric_mean, numeric_std)
    val_inputs = _prepare_scaled_inputs(val_payload, numeric_mean, numeric_std)
    train_labels = np.asarray(train_payload["labels"], dtype=np.int32)
    val_labels = np.asarray(val_payload["labels"], dtype=np.int32)
    positive_weight = max(1.0, float(np.sum(train_labels == 0) / max(np.sum(train_labels == 1), 1)))

    if log_progress:
        print(
            f"[train] dataset={dataset_name} | train={len(train_labels)} | val={len(val_labels)} "
            f"| positive_weight={positive_weight:.4f} | semantic_model={semantic_model_name}"
        )

    binary_loss = _build_binary_loss(loss_name, focal_gamma)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss={
            "fusion_score": binary_loss,
            "semantic_score": binary_loss,
            "graph_score": binary_loss,
        },
        loss_weights={"fusion_score": 1.0, "semantic_score": 0.25, "graph_score": 0.25},
        metrics={
            "fusion_score": [
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="roc_auc"),
                keras.metrics.AUC(name="pr_auc", curve="PR"),
            ]
        },
    )

    callbacks: list[keras.callbacks.Callback] = [
        keras.callbacks.EarlyStopping(monitor="val_fusion_score_pr_auc", mode="max", patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_fusion_score_pr_auc", mode="max", factor=0.5, patience=1, min_lr=1e-5),
    ]
    if log_progress:
        callbacks.append(ProgressLogger())

    history = model.fit(
        train_inputs,
        _targets(train_labels),
        validation_data=(val_inputs, _targets(val_labels), _sample_weights(val_labels, 1.0)),
        sample_weight=_sample_weights(train_labels, positive_weight),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )

    hard_negative_summary = {
        "enabled": bool(hard_negative_mining),
        "applied": False,
        "quantile": float(hard_negative_quantile),
        "weight": float(hard_negative_weight),
        "epochs": int(hard_negative_epochs),
        "cutoff": None,
        "selected": 0,
    }
    if hard_negative_mining:
        train_outputs = model.predict(train_inputs, batch_size=batch_size, verbose=0)
        train_scores = np.asarray(train_outputs["fusion_score"], dtype=np.float32).reshape(-1)
        negative_scores = train_scores[train_labels == 0]
        if len(negative_scores) > 0:
            cutoff = float(np.quantile(negative_scores, np.clip(hard_negative_quantile, 0.5, 0.99)))
            hard_negative_mask = (train_labels == 0) & (train_scores >= cutoff)
            hard_negative_weights = np.ones_like(train_labels, dtype=np.float32)
            hard_negative_weights[hard_negative_mask] = float(hard_negative_weight)
            hard_negative_summary = {
                "enabled": True,
                "applied": True,
                "quantile": float(hard_negative_quantile),
                "weight": float(hard_negative_weight),
                "epochs": int(hard_negative_epochs),
                "cutoff": float(cutoff),
                "selected": int(np.sum(hard_negative_mask)),
            }
            if log_progress:
                print(
                    f"[train] hard-negative mining | selected={hard_negative_summary['selected']} | "
                    f"cutoff={hard_negative_summary['cutoff']:.4f} | weight={hard_negative_weight:.2f}"
                )
            hard_callbacks: list[keras.callbacks.Callback] = [ProgressLogger()] if log_progress else []
            hard_history = model.fit(
                train_inputs,
                _targets(train_labels),
                validation_data=(val_inputs, _targets(val_labels), _sample_weights(val_labels, 1.0)),
                sample_weight=_sample_weights(train_labels, positive_weight, extra_weights=hard_negative_weights),
                epochs=hard_negative_epochs,
                batch_size=batch_size,
                callbacks=hard_callbacks,
                verbose=0,
            )
            history.history["hard_negative_loss"] = [float(value) for value in hard_history.history.get("loss", [])]

    output_dir = ensure_dir(MODELS_DIR / dataset_name / model_name)
    model.save_weights(output_dir / "weights.weights.h5")
    (output_dir / "token_vocabulary.txt").write_text("\n".join(model.token_vectorizer.get_vocabulary()), encoding="utf-8")
    (output_dir / "ast_vocabulary.txt").write_text("\n".join(model.ast_vectorizer.get_vocabulary()), encoding="utf-8")

    config = {
        "schema_version": PREFILTER_MODEL_SCHEMA_VERSION,
        "architecture": "hybrid_multiview_prefilter",
        "token_max_tokens": token_max_tokens,
        "token_sequence_length": token_sequence_length,
        "token_embedding_dim": token_embedding_dim,
        "token_filters": token_filters,
        "ast_max_tokens": ast_max_tokens,
        "ast_sequence_length": ast_sequence_length,
        "ast_embedding_dim": ast_embedding_dim,
        "ast_filters": ast_filters,
        "projection_dim": projection_dim,
        "dense_units": dense_units,
        "dropout_rate": dropout_rate,
        "semantic_dim": semantic_dim,
        "numeric_dim": numeric_dim,
        "numeric_feature_names": NUMERIC_FEATURE_NAMES,
        "numeric_mean": numeric_mean.tolist(),
        "numeric_std": numeric_std.tolist(),
        "semantic_model_name": semantic_model_name,
        "loss_name": loss_name,
        "focal_gamma": float(focal_gamma),
        "hard_negative_mining": bool(hard_negative_mining),
        "hard_negative_quantile": float(hard_negative_quantile),
        "hard_negative_weight": float(hard_negative_weight),
        "hard_negative_epochs": int(hard_negative_epochs),
    }
    dump_json(output_dir / "config.json", config)

    summary = {
        "dataset": dataset_name,
        "model_name": model_name,
        "model_path": str(output_dir),
        "train_size": int(len(train_labels)),
        "val_size": int(len(val_labels)),
        "positive_class_weight": float(positive_weight),
        "best_val_pr_auc": float(max(history.history.get("val_fusion_score_pr_auc", [0.0]))),
        "best_val_recall": float(max(history.history.get("val_fusion_score_recall", [0.0]))),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "feature_names": list(NUMERIC_FEATURE_NAMES),
        "semantic_model_name": semantic_model_name,
        "loss_name": loss_name,
        "focal_gamma": float(focal_gamma),
        "hard_negative_mining": bool(hard_negative_mining),
        "hard_negative_quantile": float(hard_negative_quantile),
        "hard_negative_weight": float(hard_negative_weight),
        "hard_negative_epochs": int(hard_negative_epochs),
        "hard_negative_summary": hard_negative_summary,
    }
    dump_json(MODELS_DIR / dataset_name / f"training_summary.{model_name}.json", summary)
    return summary


class HybridPrefilterBundle:
    def __init__(self, artifact_dir: Path) -> None:
        self.artifact_dir = Path(artifact_dir)
        config_path = self.artifact_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing prefilter config: {config_path}")
        self.config = _load_json(config_path)
        self.model = _build_prefilter_model(
            token_max_tokens=int(self.config["token_max_tokens"]),
            token_sequence_length=int(self.config["token_sequence_length"]),
            token_embedding_dim=int(self.config["token_embedding_dim"]),
            token_filters=int(self.config["token_filters"]),
            ast_max_tokens=int(self.config["ast_max_tokens"]),
            ast_sequence_length=int(self.config["ast_sequence_length"]),
            ast_embedding_dim=int(self.config["ast_embedding_dim"]),
            ast_filters=int(self.config["ast_filters"]),
            semantic_dim=int(self.config["semantic_dim"]),
            numeric_dim=int(self.config["numeric_dim"]),
            projection_dim=int(self.config["projection_dim"]),
            dense_units=int(self.config["dense_units"]),
            dropout_rate=float(self.config["dropout_rate"]),
        )
        token_vocabulary = (self.artifact_dir / "token_vocabulary.txt").read_text(encoding="utf-8").splitlines()
        ast_vocabulary = (self.artifact_dir / "ast_vocabulary.txt").read_text(encoding="utf-8").splitlines()
        self.model.token_vectorizer.set_vocabulary(token_vocabulary)
        self.model.ast_vectorizer.set_vocabulary(ast_vocabulary)
        self.model.load_weights(self.artifact_dir / "weights.weights.h5")
        self.numeric_mean = np.asarray(self.config["numeric_mean"], dtype=np.float32)
        self.numeric_std = np.asarray(self.config["numeric_std"], dtype=np.float32)

    def predict_payload(self, payload: dict, batch_size: int = 128) -> dict[str, np.ndarray]:
        inputs = _prepare_scaled_inputs(payload, self.numeric_mean, self.numeric_std)
        outputs = self.model.predict(inputs, batch_size=batch_size, verbose=0)
        return {
            "fusion_score": outputs["fusion_score"].reshape(-1),
            "semantic_score": outputs["semantic_score"].reshape(-1),
            "graph_score": outputs["graph_score"].reshape(-1),
        }


def load_hybrid_prefilter_bundle(dataset_name: str, model_name: str = DEFAULT_PREFILTER_MODEL_NAME) -> HybridPrefilterBundle:
    return HybridPrefilterBundle(MODELS_DIR / dataset_name / model_name)


def predict_feature_store(
    dataset_name: str,
    split_name: str,
    *,
    model_name: str = DEFAULT_PREFILTER_MODEL_NAME,
    batch_size: int = 128,
) -> dict:
    bundle = load_hybrid_prefilter_bundle(dataset_name, model_name=model_name)
    payload = load_feature_store(dataset_name, split_name)
    predictions = bundle.predict_payload(payload, batch_size=batch_size)
    return {
        "record_ids": payload["record_ids"],
        "labels": payload["labels"],
        **predictions,
    }


def build_single_record_feature_payload(
    record: dict,
    *,
    semantic_encoder: SemanticRetrievalEncoder,
    graph_backend: str = "auto",
) -> dict:
    code = get_record_code(record)
    graph_features = get_graph_features({**record, "code": code}, graph_backend=graph_backend)
    payload = {
        "record_ids": [str(record.get("record_id", "record-0"))],
        "labels": np.asarray([int(record.get("label", 0))], dtype=np.int32),
        "token_texts": [" ".join(tokenize_code(code))],
        "ast_texts": [graph_features.get("ast_sequence") or build_skeleton(code)],
        "numeric_features": np.asarray([_compute_numeric_features(code, graph_features)], dtype=np.float32),
        "semantic_embeddings": np.asarray(semantic_encoder.encode_texts([code]), dtype=np.float32),
    }
    return payload


def _load_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


__all__ = [
    "DEFAULT_PREFILTER_MODEL_NAME",
    "FEATURE_STORE_SCHEMA_VERSION",
    "NUMERIC_FEATURE_NAMES",
    "HybridPrefilterBundle",
    "build_feature_store",
    "build_single_record_feature_payload",
    "feature_store_path",
    "load_feature_store",
    "load_hybrid_prefilter_bundle",
    "predict_feature_store",
    "train_hybrid_prefilter",
]
