from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from transformers import AutoTokenizer

DEFAULT_MODEL = "microsoft/graphcodebert-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase C (improved): tokenize for GraphCodeBERT")
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/artifacts/dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/features"))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--input-field", type=str, default="code")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_features(rows: List[Dict], tokenizer: AutoTokenizer, max_length: int, input_field: str) -> Dict[str, np.ndarray]:
    texts = [str(row[input_field]) for row in rows]
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    labels = np.array([int(row["label"]) for row in rows], dtype=np.int64)
    lengths = encoded["attention_mask"].sum(axis=1).astype(np.int64)
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "lengths": lengths,
    }


def main() -> None:
    args = parse_args()
    ensure_dir(args.out_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    split_stats: Dict[str, Dict] = {}

    for split in ["train", "valid", "test"]:
        in_path = args.dataset_dir / f"{split}.jsonl"
        rows = read_jsonl(in_path)
        feats = build_features(rows, tokenizer, args.max_length, args.input_field)

        np.save(args.out_dir / f"{split}.input_ids.npy", feats["input_ids"])
        np.save(args.out_dir / f"{split}.attention_mask.npy", feats["attention_mask"])
        np.save(args.out_dir / f"{split}.labels.npy", feats["labels"])
        np.save(args.out_dir / f"{split}.lengths.npy", feats["lengths"])

        trunc_flags = [int(length >= args.max_length) for length in feats["lengths"]]
        meta_rows = []
        for row, tok_len, truncated in zip(rows, feats["lengths"].tolist(), trunc_flags):
            meta_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "pair_id": row["pair_id"],
                    "project_slug": row["project_slug"],
                    "cve_id": row["cve_id"],
                    "cwe_id": row["cwe_id"],
                    "file": row["file"],
                    "class": row.get("class", ""),
                    "method": row.get("method", ""),
                    "signature": row.get("signature", ""),
                    "unit_kind": row.get("unit_kind", ""),
                    "label": int(row["label"]),
                    "split": row["split"],
                    "token_length": int(tok_len),
                    "was_truncated": int(truncated),
                }
            )
        write_jsonl(args.out_dir / f"{split}.meta.jsonl", meta_rows)

        split_stats[split] = {
            "num_samples": int(len(rows)),
            "avg_token_length": float(np.mean(feats["lengths"])) if len(rows) else 0.0,
            "max_token_length": int(np.max(feats["lengths"])) if len(rows) else 0,
            "num_hit_max_length": int(sum(trunc_flags)),
            "truncation_rate": float(np.mean(trunc_flags)) if len(rows) else 0.0,
        }

    with (args.out_dir / "feature_config_improved.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "input_field": args.input_field,
                "max_length": args.max_length,
                "splits": ["train", "valid", "test"],
                "split_stats": split_stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Phase C (improved) done.")
    print(json.dumps(split_stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
