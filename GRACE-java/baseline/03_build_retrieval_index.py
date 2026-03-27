from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from common import WORK_DIR, ensure_dir, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build retrieval index: GraphCodeBERT/CodeT5 embedding + whitening + FAISS.")
    p.add_argument("--input", default=str(WORK_DIR / "train_ast.jsonl"), help="Train JSONL with code and ast_seq.")
    p.add_argument("--out-dir", default=str(WORK_DIR / "retrieval"))
    p.add_argument("--model-name", default="microsoft/graphcodebert-base")
    p.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--whiten-dim", type=int, default=256)
    p.add_argument("--index-type", choices=["flat", "ivf"], default="flat")
    p.add_argument("--nlist", type=int, default=100)
    return p.parse_args()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


@torch.no_grad()
def encode_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    batch_size: int,
    max_length: int,
    pooling: str,
    device: torch.device,
) -> np.ndarray:
    vectors: list[np.ndarray] = []
    is_seq2seq_encoder = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        if is_seq2seq_encoder:
            outputs = model.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = outputs.last_hidden_state
        else:
            outputs = model(**enc)
            hidden = outputs.last_hidden_state

        if pooling == "cls":
            pooled = cls_pool(hidden)
        else:
            pooled = mean_pool(hidden, enc["attention_mask"])
        vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0)


def fit_whitening(x: np.ndarray, out_dim: int) -> tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    xc = x - mu
    cov = np.cov(xc.T)
    u, s, _ = np.linalg.svd(cov)
    kernel = (u / np.sqrt(s + 1e-12))[:, :out_dim]
    bias = -mu
    return kernel.astype(np.float32), bias.astype(np.float32)


def transform_and_normalize(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    y = (x + bias) @ kernel
    norms = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    return (y / norms).astype(np.float32)


def build_faiss(vecs: np.ndarray, index_type: str, nlist: int) -> faiss.Index:
    dim = vecs.shape[1]
    if index_type == "flat":
        index = faiss.IndexFlatIP(dim)
        index.add(vecs)
        return index

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, len(vecs)), faiss.METRIC_INNER_PRODUCT)
    index.train(vecs)
    index.add(vecs)
    index.nprobe = min(16, min(nlist, len(vecs)))
    return index


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if not rows:
        raise ValueError("No train rows found.")

    texts = [r["code"] for r in rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    raw_vecs = encode_texts(texts, tokenizer, model, args.batch_size, args.max_length, args.pooling, device)
    kernel, bias = fit_whitening(raw_vecs, args.whiten_dim)
    whitened_vecs = transform_and_normalize(raw_vecs, kernel, bias)
    index = build_faiss(whitened_vecs, args.index_type, args.nlist)

    out_dir = ensure_dir(args.out_dir)
    faiss.write_index(index, str(out_dir / "retrieval.index"))

    meta = {
        "model_name": args.model_name,
        "pooling": args.pooling,
        "max_length": args.max_length,
        "whiten_dim": args.whiten_dim,
        "index_type": args.index_type,
        "rows": rows,
        "kernel": kernel,
        "bias": bias,
    }
    with (out_dir / "metadata.pkl").open("wb") as f:
        pickle.dump(meta, f)

    with (out_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "num_rows": len(rows),
                "embedding_dim": int(raw_vecs.shape[1]),
                "whiten_dim": int(whitened_vecs.shape[1]),
                "index_type": args.index_type,
                "model_name": args.model_name,
                "pooling": args.pooling,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved retrieval index to {out_dir}")


if __name__ == "__main__":
    main()
