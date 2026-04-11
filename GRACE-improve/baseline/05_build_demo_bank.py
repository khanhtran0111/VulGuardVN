import random
import os

from common import RETRIEVAL_DIR, SPLITS_DIR, dump_json, ensure_dir, get_record_code, iter_jsonl
from retrieval import (
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MAX_LENGTH,
    DEFAULT_RETRIEVAL_MODEL_REPO_ID,
    build_demo_bank,
    default_retrieval_model_dir,
    save_demo_bank,
)


DATASET_NAME = os.getenv("GRACE_DATASET", "devign")
MAX_EXAMPLES_PER_LABEL = int(os.getenv("GRACE_MAX_EXAMPLES_PER_LABEL", "4000"))
MAX_FEATURES = int(os.getenv("GRACE_TFIDF_MAX_FEATURES", "50000"))
SEED = 42
SEMANTIC_BACKEND = os.getenv("GRACE_RETRIEVAL_BACKEND", "auto")
SEMANTIC_MODEL_NAME = os.getenv("GRACE_RETRIEVAL_MODEL_ID", DEFAULT_RETRIEVAL_MODEL_REPO_ID)
SEMANTIC_MODEL_DIR = default_retrieval_model_dir(SEMANTIC_MODEL_NAME)
SEMANTIC_BATCH_SIZE = int(os.getenv("GRACE_RETRIEVAL_BATCH_SIZE", str(DEFAULT_EMBEDDING_BATCH_SIZE)))
SEMANTIC_MAX_LENGTH = int(os.getenv("GRACE_RETRIEVAL_MAX_LENGTH", str(DEFAULT_EMBEDDING_MAX_LENGTH)))
AUTO_DOWNLOAD_SEMANTIC_MODEL = os.getenv("GRACE_AUTO_DOWNLOAD_RETRIEVAL_MODEL", "").strip().lower() in {"1", "true", "yes", "on"}
GRAPH_BACKEND = os.getenv("GRACE_GRAPH_BACKEND", "auto")


def _sample_records_by_label(split_path, max_examples_per_label: int, seed: int):
    rng = random.Random(seed)
    reservoirs = {0: [], 1: []}
    seen = {0: 0, 1: 0}
    for record in iter_jsonl(split_path):
        code = get_record_code(record)
        if not code:
            continue
        label = int(record["label"])
        canonical = {
            "record_id": record["record_id"],
            "dataset": record.get("dataset", DATASET_NAME),
            "label": label,
            "project": record.get("project", ""),
            "code": code,
            "code_hash": record.get("code_hash"),
        }
        seen[label] += 1
        bucket = reservoirs[label]
        if len(bucket) < max_examples_per_label:
            bucket.append(canonical)
            continue
        replacement_index = rng.randint(0, seen[label] - 1)
        if replacement_index < max_examples_per_label:
            bucket[replacement_index] = canonical
    return reservoirs[0] + reservoirs[1]


def main() -> None:
    train_path = SPLITS_DIR / DATASET_NAME / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split for {DATASET_NAME}. Run 02_create_splits.py first.")
    records = _sample_records_by_label(train_path, max_examples_per_label=MAX_EXAMPLES_PER_LABEL, seed=SEED)
    bank = build_demo_bank(
        records=records,
        max_examples_per_label=MAX_EXAMPLES_PER_LABEL,
        max_features=MAX_FEATURES,
        random_seed=SEED,
        semantic_backend=SEMANTIC_BACKEND,
        semantic_model_name=SEMANTIC_MODEL_NAME,
        semantic_model_dir=SEMANTIC_MODEL_DIR,
        semantic_batch_size=SEMANTIC_BATCH_SIZE,
        semantic_max_length=SEMANTIC_MAX_LENGTH,
        auto_download_semantic_model=AUTO_DOWNLOAD_SEMANTIC_MODEL,
        graph_backend=GRAPH_BACKEND,
    )
    output_dir = ensure_dir(RETRIEVAL_DIR / DATASET_NAME)
    bank_path = output_dir / "demo_bank.joblib"
    save_demo_bank(bank_path, bank)
    summary = {
        "dataset": DATASET_NAME,
        "bank_path": str(bank_path),
        "total_examples": len(bank["records"]),
        "negative_examples": sum(1 for row in bank["records"] if row["label"] == 0),
        "positive_examples": sum(1 for row in bank["records"] if row["label"] == 1),
        "semantic_backend": bank.get("semantic_backend"),
        "semantic_notice": bank.get("semantic_notice"),
        "semantic_config": bank.get("semantic_config"),
        "graph_backend_requested": bank.get("graph_backend_requested"),
        "graph_backend_counts": bank.get("graph_backend_counts"),
    }
    dump_json(output_dir / "summary.json", summary)
    print(f"Saved demo bank for {DATASET_NAME} to {bank_path}")


if __name__ == "__main__":
    main()
