from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

from pipeline_common import ensure_dir, read_jsonl, write_jsonl


DEFAULT_MODEL = "microsoft/graphcodebert-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C: tokenize for GraphCodeBERT")
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/features"))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def build_features(rows: List[Dict], tokenizer: AutoTokenizer, max_length: int) -> Dict[str, np.ndarray]:
    texts = [row["code"] for row in rows]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    labels = np.array([row["label"] for row in rows], dtype=np.int64)

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    for split in ["train", "valid", "test"]:
        in_path = args.dataset_dir / f"{split}.jsonl"
        rows = read_jsonl(in_path)
        feats = build_features(rows, tokenizer, args.max_length)

        np.save(args.out_dir / f"{split}.input_ids.npy", feats["input_ids"])
        np.save(args.out_dir / f"{split}.attention_mask.npy", feats["attention_mask"])
        np.save(args.out_dir / f"{split}.labels.npy", feats["labels"])

        meta_rows = [
            {
                "sample_id": row["sample_id"],
                "pair_id": row["pair_id"],
                "project_slug": row["project_slug"],
                "cve_id": row["cve_id"],
                "cwe_id": row["cwe_id"],
                "signature": row["signature"],
                "file": row["file"],
                "method": row["method"],
                "label": row["label"],
                "split": row["split"],
            }
            for row in rows
        ]
        write_jsonl(args.out_dir / f"{split}.meta.jsonl", meta_rows)

    with (args.out_dir / "feature_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "max_length": args.max_length,
                "splits": ["train", "valid", "test"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Phase C done.")
    print(f"Output dir: {args.out_dir}")


if __name__ == "__main__":
    main()
