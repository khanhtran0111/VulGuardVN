import json
from collections import Counter

import pandas as pd

from common import DATA_DIR, dump_json, ensure_dir, normalize_code, stable_hash


RAW_DIR = DATA_DIR / "reveal_raw"
OUTPUT_DIR = DATA_DIR / "reveal"
SPLIT_FILES = {
    "train": RAW_DIR / "train-00000-of-00001.parquet",
    "val": RAW_DIR / "validation-00000-of-00001.parquet",
    "test": RAW_DIR / "test-00000-of-00001.parquet",
}
OVERWRITE = False


def _load_parquet(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing ReVeal parquet file: {path}")
    return pd.read_parquet(path)


def _convert_row(row: dict, split: str, index: int) -> dict | None:
    code = normalize_code(str(row.get("functionSource", "") or ""))
    if not code:
        return None
    label = row.get("label")
    if label is None:
        return None
    label = int(label)
    project = str(row.get("project", "") or "")
    hash_value = row.get("hash")
    size_value = row.get("size")
    return {
        "record_id": f"reveal-{split}-{index}",
        "dataset": "reveal",
        "split": split,
        "project": project,
        "label": label,
        "code": code,
        "func": code,
        "functionSource": code,
        "hash": int(hash_value) if hash_value is not None else None,
        "size": int(size_value) if size_value is not None else None,
        "code_hash": stable_hash(code),
        "source_format": "parquet",
    }


def convert_split(split: str, parquet_path, output_path):
    frame = _load_parquet(parquet_path)
    rows = frame.to_dict(orient="records")
    stats = Counter()
    projects = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(rows):
            converted = _convert_row(row, split, index)
            if converted is None:
                continue
            handle.write(json.dumps(converted, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats[f"label_{converted['label']}"] += 1
            projects[converted["project"] or "unknown"] += 1
    return {
        "split": split,
        "rows": int(stats["rows"]),
        "positive": int(stats["label_1"]),
        "negative": int(stats["label_0"]),
        "top_projects": projects.most_common(10),
        "output_path": str(output_path),
    }


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not OVERWRITE:
        print(f"ReVeal processed data already exists at {OUTPUT_DIR}")
        return
    ensure_dir(OUTPUT_DIR)
    summaries = {}
    for split, parquet_path in SPLIT_FILES.items():
        output_path = OUTPUT_DIR / f"{split}.jsonl"
        print(f"Converting {parquet_path.name} -> {output_path.name}")
        summaries[split] = convert_split(split, parquet_path, output_path)
    dump_json(OUTPUT_DIR / "stats.json", {"dataset": "reveal", "splits": summaries})
    print(f"Done. Processed ReVeal files are under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
