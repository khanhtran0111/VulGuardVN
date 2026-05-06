import json
import os

from hybrid_prefilter import DEFAULT_PREFILTER_MODEL_NAME, train_hybrid_prefilter
from retrieval import DEFAULT_RETRIEVAL_MODEL_REPO_ID


DATASET_NAME = os.getenv("GRACE_DATASET", "devign")
MODEL_NAME = os.getenv("GRACE_PREFILTER_MODEL_NAME", DEFAULT_PREFILTER_MODEL_NAME)
SEMANTIC_MODEL_NAME = os.getenv("GRACE_RETRIEVAL_MODEL_ID", DEFAULT_RETRIEVAL_MODEL_REPO_ID)
BATCH_SIZE = int(os.getenv("GRACE_PREFILTER_BATCH_SIZE", "128"))
EPOCHS = int(os.getenv("GRACE_PREFILTER_EPOCHS", "10"))
LEARNING_RATE = float(os.getenv("GRACE_PREFILTER_LEARNING_RATE", "7e-4"))
RANDOM_SEED = int(os.getenv("GRACE_PREFILTER_RANDOM_SEED", "42"))
LOSS_NAME = os.getenv("GRACE_PREFILTER_LOSS", "bce")
FOCAL_GAMMA = float(os.getenv("GRACE_PREFILTER_FOCAL_GAMMA", "2.0"))
HARD_NEGATIVE_MINING = os.getenv("GRACE_HARD_NEGATIVE_MINING", "0").strip().lower() in {"1", "true", "yes", "on"}
HARD_NEGATIVE_QUANTILE = float(os.getenv("GRACE_HARD_NEGATIVE_QUANTILE", "0.85"))
HARD_NEGATIVE_WEIGHT = float(os.getenv("GRACE_HARD_NEGATIVE_WEIGHT", "2.5"))
HARD_NEGATIVE_EPOCHS = int(os.getenv("GRACE_HARD_NEGATIVE_EPOCHS", "2"))
TOKEN_MAX_TOKENS = int(os.getenv("GRACE_TOKEN_MAX_TOKENS", "32000"))
TOKEN_SEQUENCE_LENGTH = int(os.getenv("GRACE_TOKEN_SEQUENCE_LENGTH", "384"))
TOKEN_EMBEDDING_DIM = int(os.getenv("GRACE_TOKEN_EMBEDDING_DIM", "96"))
TOKEN_FILTERS = int(os.getenv("GRACE_TOKEN_FILTERS", "96"))
AST_MAX_TOKENS = int(os.getenv("GRACE_AST_MAX_TOKENS", "8000"))
AST_SEQUENCE_LENGTH = int(os.getenv("GRACE_AST_SEQUENCE_LENGTH", "196"))
AST_EMBEDDING_DIM = int(os.getenv("GRACE_AST_EMBEDDING_DIM", "64"))
AST_FILTERS = int(os.getenv("GRACE_AST_FILTERS", "64"))
PROJECTION_DIM = int(os.getenv("GRACE_PREFILTER_PROJECTION_DIM", "192"))
DENSE_UNITS = int(os.getenv("GRACE_PREFILTER_DENSE_UNITS", "192"))
DROPOUT_RATE = float(os.getenv("GRACE_PREFILTER_DROPOUT", "0.25"))


def main() -> None:
    summary = train_hybrid_prefilter(
        DATASET_NAME,
        model_name=MODEL_NAME,
        semantic_model_name=SEMANTIC_MODEL_NAME,
        token_max_tokens=TOKEN_MAX_TOKENS,
        token_sequence_length=TOKEN_SEQUENCE_LENGTH,
        token_embedding_dim=TOKEN_EMBEDDING_DIM,
        token_filters=TOKEN_FILTERS,
        ast_max_tokens=AST_MAX_TOKENS,
        ast_sequence_length=AST_SEQUENCE_LENGTH,
        ast_embedding_dim=AST_EMBEDDING_DIM,
        ast_filters=AST_FILTERS,
        projection_dim=PROJECTION_DIM,
        dense_units=DENSE_UNITS,
        dropout_rate=DROPOUT_RATE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        random_seed=RANDOM_SEED,
        log_progress=True,
        loss_name=LOSS_NAME,
        focal_gamma=FOCAL_GAMMA,
        hard_negative_mining=HARD_NEGATIVE_MINING,
        hard_negative_quantile=HARD_NEGATIVE_QUANTILE,
        hard_negative_weight=HARD_NEGATIVE_WEIGHT,
        hard_negative_epochs=HARD_NEGATIVE_EPOCHS,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
