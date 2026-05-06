import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from common import PROCESSED_DIR, SPLITS_DIR, dump_json, ensure_dir
from datasets import (
    get_dataset_iterator,
    has_reveal_official_splits,
    iter_reveal_split_records,
)


TARGET_DATASETS = [name.strip() for name in os.getenv("GRACE_DATASETS", os.getenv("GRACE_DATASET", "devign")).split(",") if name.strip()]
OUTER_SPLITS = 10
INNER_SPLITS = 9
SEED = 42


def _assign_splits(frame: pd.DataFrame) -> pd.DataFrame:
    x = np.zeros(len(frame))
    y = frame["label"].to_numpy()
    groups = frame["code_hash"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=SEED)
    train_val_idx, test_idx = next(outer.split(x, y, groups))
    assignments = np.array([""] * len(frame), dtype=object)
    assignments[test_idx] = "test"
    inner_frame = frame.iloc[train_val_idx].reset_index(drop=True)
    inner_x = np.zeros(len(inner_frame))
    inner_y = inner_frame["label"].to_numpy()
    inner_groups = inner_frame["code_hash"].to_numpy()
    inner = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED + 1)
    train_idx, val_idx = next(inner.split(inner_x, inner_y, inner_groups))
    assignments[train_val_idx[train_idx]] = "train"
    assignments[train_val_idx[val_idx]] = "val"
    result = frame.copy()
    result["split"] = assignments
    return result


def _write_records_from_assignment(dataset_name: str, assigned: pd.DataFrame, output_dir) -> None:
    split_index_path = output_dir / "split_index.csv"
    assigned.to_csv(split_index_path, index=False)
    record_to_split = dict(zip(assigned["record_id"], assigned["split"]))
    handles = {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    iterator = get_dataset_iterator(dataset_name)
    try:
        for record in iterator():
            split = record_to_split.get(record["record_id"])
            if split in handles:
                handles[split].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()


def _summarize_assigned(dataset_name: str, assigned: pd.DataFrame, split_index_path) -> dict:
    summary = {}
    for split in ["train", "val", "test"]:
        subset = assigned[assigned["split"] == split]
        summary[split] = {
            "rows": int(len(subset)),
            "positive": int(subset["label"].sum()),
            "negative": int((1 - subset["label"]).sum()),
            "groups": int(subset["code_hash"].nunique()),
        }
    summary["dataset"] = dataset_name
    summary["split_index_path"] = str(split_index_path)
    return summary


def _create_random_group_splits(dataset_name: str) -> None:
    index_path = PROCESSED_DIR / dataset_name / "index.csv"
    if not index_path.exists():
        print(f"Skipping {dataset_name}: run 01_prepare_datasets.py first.")
        return
    frame = pd.read_csv(index_path)
    if frame.empty:
        print(f"Skipping {dataset_name}: empty index.")
        return
    frame["label"] = frame["label"].astype(int)
    assigned = _assign_splits(frame)
    output_dir = ensure_dir(SPLITS_DIR / dataset_name)
    _write_records_from_assignment(dataset_name, assigned, output_dir)
    split_index_path = output_dir / "split_index.csv"
    summary = _summarize_assigned(dataset_name, assigned, split_index_path)
    dump_json(output_dir / "split_summary.json", summary)
    print(f"{dataset_name}: split files written to {output_dir}")


def _create_reveal_official_splits() -> None:
    output_dir = ensure_dir(SPLITS_DIR / "reveal")
    split_rows = []
    stats = {}
    for split_name in ["train", "val", "test"]:
        rows = list(iter_reveal_split_records(split_name))
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in rows:
            split_rows.append(
                {
                    "record_id": row["record_id"],
                    "dataset": row["dataset"],
                    "project": row["project"],
                    "label": row["label"],
                    "commit_id": row.get("commit_id", ""),
                    "cwe_id": row.get("cwe_id", ""),
                    "code_hash": row["code_hash"],
                    "source_path": row["source_path"],
                    "split": split_name,
                }
            )
        stats[split_name] = {
            "rows": len(rows),
            "positive": sum(int(row["label"]) for row in rows),
            "negative": sum(1 - int(row["label"]) for row in rows),
            "groups": len({row["code_hash"] for row in rows}),
        }
    split_index = pd.DataFrame(split_rows)
    split_index_path = output_dir / "split_index.csv"
    split_index.to_csv(split_index_path, index=False)
    dump_json(
        output_dir / "split_summary.json",
        {
            "dataset": "reveal",
            "strategy": "official",
            "split_index_path": str(split_index_path),
            **stats,
        },
    )
    print(f"reveal: official split files written to {output_dir}")


def _create_hint_based_splits(dataset_name: str) -> bool:
    index_path = PROCESSED_DIR / dataset_name / "index.csv"
    if not index_path.exists():
        return False
    frame = pd.read_csv(index_path)
    if frame.empty or "split" not in frame.columns:
        return False
    frame["split"] = frame["split"].fillna("").astype(str)
    valid = frame["split"].isin(["train", "val", "test"]).all()
    if not valid:
        return False
    output_dir = ensure_dir(SPLITS_DIR / dataset_name)
    _write_records_from_assignment(dataset_name, frame, output_dir)
    split_index_path = output_dir / "split_index.csv"
    summary = _summarize_assigned(dataset_name, frame, split_index_path)
    summary["strategy"] = "source_split_hint"
    dump_json(output_dir / "split_summary.json", summary)
    print(f"{dataset_name}: split files written from source split hints to {output_dir}")
    return True


def create_dataset_splits(dataset_name: str) -> None:
    if dataset_name == "reveal" and has_reveal_official_splits():
        _create_reveal_official_splits()
        return
    if dataset_name == "reveal" and _create_hint_based_splits(dataset_name):
        return
    _create_random_group_splits(dataset_name)


def main() -> None:
    for dataset_name in TARGET_DATASETS:
        create_dataset_splits(dataset_name)


if __name__ == "__main__":
    main()
