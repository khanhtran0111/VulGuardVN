import csv
import os
from collections import Counter

from common import PROCESSED_DIR, dump_json, ensure_dir
from datasets import discover_reveal_root, get_dataset_iterator


TARGET_DATASETS = [name.strip() for name in os.getenv("GRACE_DATASETS", os.getenv("GRACE_DATASET", "devign")).split(",") if name.strip()]


def prepare_dataset(dataset_name: str) -> None:
    if dataset_name == "reveal" and discover_reveal_root() is None:
        print("Skipping reveal because no raw files were found in data/reveal_raw or similar folders.")
        return
    output_dir = ensure_dir(PROCESSED_DIR / dataset_name)
    stale_records_path = output_dir / "records.jsonl"
    if stale_records_path.exists():
        stale_records_path.unlink()
    index_path = output_dir / "index.csv"
    stats = Counter()
    label_counter = Counter()
    project_counter = Counter()
    iterator = get_dataset_iterator(dataset_name)
    with index_path.open("w", encoding="utf-8", newline="") as index_handle:
        writer = csv.DictWriter(
            index_handle,
            fieldnames=["record_id", "dataset", "project", "label", "commit_id", "cwe_id", "code_hash", "source_path", "split"],
        )
        writer.writeheader()
        for record in iterator():
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "dataset": record["dataset"],
                    "project": record["project"],
                    "label": record["label"],
                    "commit_id": record["commit_id"],
                    "cwe_id": record["cwe_id"],
                    "code_hash": record["code_hash"],
                    "source_path": record["source_path"],
                    "split": record.get("split", ""),
                }
            )
            stats["records"] += 1
            label_counter[int(record["label"])] += 1
            project_counter[record["project"] or "unknown"] += 1
    payload = {
        "dataset": dataset_name,
        "records": int(stats["records"]),
        "positive": int(label_counter[1]),
        "negative": int(label_counter[0]),
        "positive_ratio": float(label_counter[1] / max(stats["records"], 1)),
        "top_projects": project_counter.most_common(10),
        "index_path": str(index_path),
        "materialization": "index_only",
    }
    dump_json(output_dir / "stats.json", payload)
    print(f"{dataset_name}: {payload['records']} rows indexed at {index_path}")


def main() -> None:
    for dataset_name in TARGET_DATASETS:
        prepare_dataset(dataset_name)


if __name__ == "__main__":
    main()
