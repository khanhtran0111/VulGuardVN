from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline_common import read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase F: classification + retrieval evaluation")
    parser.add_argument("--features-dir", type=Path, default=Path("data/artifacts/features"))
    parser.add_argument("--model-dir", type=Path, default=Path("model"))
    parser.add_argument("--index-dir", type=Path, default=Path("index"))
    parser.add_argument("--out-dir", type=Path, default=Path("eval"))
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def evaluate_binary(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict:
    result = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
        "classification_report": classification_report(labels, preds, zero_division=0, output_dict=True),
    }
    if len(np.unique(labels)) > 1:
        result["roc_auc"] = float(roc_auc_score(labels, probs))
    else:
        result["roc_auc"] = 0.0
    return result


def evaluate_retrieval(test_meta: List[Dict], test_emb: np.ndarray, topk: int) -> Dict:
    meta = test_meta
    labels = np.array([row["label"] for row in meta], dtype=np.int64)

    vuln_idx = np.where(labels == 1)[0]
    fixed_idx = np.where(labels == 0)[0]

    if len(vuln_idx) == 0 or len(fixed_idx) == 0:
        return {"error": "Need both vulnerable and fixed samples in test split."}

    fixed_emb = test_emb[fixed_idx].astype(np.float32)
    faiss.normalize_L2(fixed_emb)

    index = faiss.IndexFlatIP(fixed_emb.shape[1])
    index.add(fixed_emb)

    recalls = []
    rr_scores = []

    fixed_pair_ids = [meta[i]["pair_id"] for i in fixed_idx]

    for q in vuln_idx:
        q_vec = test_emb[q : q + 1].astype(np.float32)
        faiss.normalize_L2(q_vec)
        _, nn = index.search(q_vec, topk)

        gt_pair = meta[q]["pair_id"]
        retrieved_pairs = [fixed_pair_ids[r] for r in nn[0] if 0 <= r < len(fixed_pair_ids)]

        hit = gt_pair in retrieved_pairs
        recalls.append(1.0 if hit else 0.0)

        rr = 0.0
        for rank, pair_id in enumerate(retrieved_pairs, start=1):
            if pair_id == gt_pair:
                rr = 1.0 / rank
                break
        rr_scores.append(rr)

    return {
        "num_queries": len(vuln_idx),
        "num_candidates": len(fixed_idx),
        f"recall@{topk}": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(rr_scores)) if rr_scores else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(args.model_dir / "test.labels.npy")
    preds = np.load(args.model_dir / "test.preds.npy")
    probs = np.load(args.model_dir / "test.probs.npy")

    binary_metrics = evaluate_binary(labels=labels, preds=preds, probs=probs)

    test_meta = read_jsonl(args.features_dir / "test.meta.jsonl")
    test_emb = np.load(args.index_dir / "test.emb.npy")
    retrieval_metrics = evaluate_retrieval(test_meta=test_meta, test_emb=test_emb, topk=args.topk)

    with (args.out_dir / "metrics_binary.json").open("w", encoding="utf-8") as f:
        json.dump(binary_metrics, f, ensure_ascii=False, indent=2)

    with (args.out_dir / "metrics_retrieval.json").open("w", encoding="utf-8") as f:
        json.dump(retrieval_metrics, f, ensure_ascii=False, indent=2)

    print("Phase F done.")
    print(json.dumps({"binary": binary_metrics, "retrieval": retrieval_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
