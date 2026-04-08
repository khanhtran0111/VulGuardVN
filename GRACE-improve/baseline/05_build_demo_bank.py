import random

from common import RETRIEVAL_DIR, SPLITS_DIR, dump_json, ensure_dir, get_record_code, iter_jsonl
from retrieval import build_demo_bank, save_demo_bank


DATASET_NAME = "devign"
MAX_EXAMPLES_PER_LABEL = 4000
MAX_FEATURES = 50000
SEED = 42


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
            "label": label,
            "project": record.get("project", ""),
            "code": code,
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
    }
    dump_json(output_dir / "summary.json", summary)
    print(f"Saved demo bank for {DATASET_NAME} to {bank_path}")


if __name__ == "__main__":
    main()
