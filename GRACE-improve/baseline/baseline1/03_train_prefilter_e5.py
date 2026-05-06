import json
import os

from train_prefilter import train_ensemble_prefilter


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    return int(lowered)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None and value.strip() else float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DATASET_NAME = os.getenv("GRACE_DATASET", "devign")
OUTPUT_MODEL_NAME = os.getenv("GRACE_PREFILTER_MODEL_NAME", "prefilter_e5_model")
MEMBER_PREFIX = os.getenv("GRACE_PREFILTER_MEMBER_PREFIX", OUTPUT_MODEL_NAME)
BATCH_SIZE = int(os.getenv("GRACE_PREFILTER_BATCH_SIZE", "128"))
EPOCHS = int(os.getenv("GRACE_PREFILTER_EPOCHS", "12"))
LEARNING_RATE = _env_float("GRACE_PREFILTER_LEARNING_RATE", 1e-3)
RANDOM_SEED = int(os.getenv("GRACE_PREFILTER_RANDOM_SEED", "42"))
ADAPT_TEXT_LIMIT = _env_int("GRACE_PREFILTER_ADAPT_LIMIT", None)
TRAIN_LIMIT = _env_int("GRACE_PREFILTER_TRAIN_LIMIT", None)
VAL_LIMIT = _env_int("GRACE_PREFILTER_VAL_LIMIT", None)
VERBOSE = int(os.getenv("GRACE_PREFILTER_VERBOSE", "1"))
LOG_PROGRESS = _env_bool("GRACE_PREFILTER_LOG_PROGRESS", True)
BATCH_LOG_INTERVAL = _env_int("GRACE_PREFILTER_BATCH_LOG_INTERVAL", None)

CNN_MAX_TOKENS = int(os.getenv("GRACE_E5_CNN_MAX_TOKENS", "40000"))
CNN_SEQUENCE_LENGTH = int(os.getenv("GRACE_E5_CNN_SEQUENCE_LENGTH", "512"))
CNN_EMBEDDING_DIM = int(os.getenv("GRACE_E5_CNN_EMBEDDING_DIM", "128"))
CNN_FILTERS = int(os.getenv("GRACE_E5_CNN_FILTERS", "128"))
CNN_DENSE_UNITS = int(os.getenv("GRACE_E5_CNN_DENSE_UNITS", "128"))
CNN_DROPOUT_RATE = _env_float("GRACE_E5_CNN_DROPOUT", 0.25)

LSTM_MAX_TOKENS = int(os.getenv("GRACE_E5_LSTM_MAX_TOKENS", "40000"))
LSTM_SEQUENCE_LENGTH = int(os.getenv("GRACE_E5_LSTM_SEQUENCE_LENGTH", "512"))
LSTM_EMBEDDING_DIM = int(os.getenv("GRACE_E5_LSTM_EMBEDDING_DIM", "128"))
LSTM_RECURRENT_UNITS = int(os.getenv("GRACE_E5_LSTM_RECURRENT_UNITS", "128"))
LSTM_DENSE_UNITS = int(os.getenv("GRACE_E5_LSTM_DENSE_UNITS", "128"))
LSTM_DROPOUT_RATE = _env_float("GRACE_E5_LSTM_DROPOUT", 0.25)
LSTM_BIDIRECTIONAL = _env_bool("GRACE_E5_LSTM_BIDIRECTIONAL", True)

ENSEMBLE_CNN_WEIGHT = _env_float("GRACE_E5_CNN_WEIGHT", 1.0)
ENSEMBLE_LSTM_WEIGHT = _env_float("GRACE_E5_LSTM_WEIGHT", 1.0)


def main() -> None:
    summary = train_ensemble_prefilter(
        dataset_name=DATASET_NAME,
        output_model_name=OUTPUT_MODEL_NAME,
        member_prefix=MEMBER_PREFIX,
        member_weights=(ENSEMBLE_CNN_WEIGHT, ENSEMBLE_LSTM_WEIGHT),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        random_seed=RANDOM_SEED,
        adapt_text_limit=ADAPT_TEXT_LIMIT,
        train_limit=TRAIN_LIMIT,
        val_limit=VAL_LIMIT,
        verbose=VERBOSE,
        log_progress=LOG_PROGRESS,
        batch_log_interval=BATCH_LOG_INTERVAL,
        cnn_max_tokens=CNN_MAX_TOKENS,
        cnn_sequence_length=CNN_SEQUENCE_LENGTH,
        cnn_embedding_dim=CNN_EMBEDDING_DIM,
        cnn_filters=CNN_FILTERS,
        cnn_dense_units=CNN_DENSE_UNITS,
        cnn_dropout_rate=CNN_DROPOUT_RATE,
        lstm_max_tokens=LSTM_MAX_TOKENS,
        lstm_sequence_length=LSTM_SEQUENCE_LENGTH,
        lstm_embedding_dim=LSTM_EMBEDDING_DIM,
        lstm_recurrent_units=LSTM_RECURRENT_UNITS,
        lstm_dense_units=LSTM_DENSE_UNITS,
        lstm_dropout_rate=LSTM_DROPOUT_RATE,
        lstm_bidirectional=LSTM_BIDIRECTIONAL,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
