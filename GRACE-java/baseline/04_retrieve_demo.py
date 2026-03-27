from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from difflib import SequenceMatcher

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

try:
    import Levenshtein
except ImportError:  # pragma: no cover - optional dependency
    Levenshtein = None

from common import WORK_DIR, load_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieve GRACE-style demonstrations for query rows.")
    p.add_argument("--input", default=str(WORK_DIR / "test_ast.jsonl"), help="Val/test JSONL with code and ast_seq.")
    p.add_argument("--index-dir", default=str(WORK_DIR / "retrieval"))
    p.add_argument("--output", default=str(WORK_DIR / "test_with_demo.jsonl"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.7, help="Lexical weight in rerank score.")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def jaccard_tokens(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def ast_seq_similarity(a_tokens: list[str], b_tokens: list[str]) -> float:
    if Levenshtein is not None:
        return float(Levenshtein.seqratio(a_tokens, b_tokens))
    return float(SequenceMatcher(None, a_tokens, b_tokens).ratio())


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def cls_pool(last_hidden_state: torch.Tensor) -> torch.Tensor:
    return last_hidden_state[:, 0, :]


@torch.no_grad()
def encode_texts(texts, tokenizer, model, batch_size, max_length, pooling, device):
    vecs = []
    is_seq2seq_encoder = bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if is_seq2seq_encoder:
            out = model.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            hidden = out.last_hidden_state
        else:
            out = model(**enc)
            hidden = out.last_hidden_state

        if pooling == "cls":
            pooled = cls_pool(hidden)
        else:
            pooled = mean_pool(hidden, enc["attention_mask"])
        vecs.append(pooled.cpu().numpy())
    return np.concatenate(vecs, axis=0)


def transform_and_normalize(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    y = (x + bias) @ kernel
    norms = np.linalg.norm(y, axis=1, keepdims=True) + 1e-12
    return (y / norms).astype(np.float32)


def main() -> None:
    args = parse_args()
    query_rows = load_jsonl(args.input)
    with open(Path(args.index_dir) / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    train_rows = meta["rows"]
    kernel = meta["kernel"]
    bias = meta["bias"]
    pooling = meta.get("pooling", "mean")
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])
    model = AutoModel.from_pretrained(meta["model_name"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    index = faiss.read_index(str(Path(args.index_dir) / "retrieval.index"))

    query_vecs = encode_texts(
        [r["code"] for r in query_rows],
        tokenizer,
        model,
        args.batch_size,
        meta["max_length"],
        pooling,
        device,
    )
    query_vecs = transform_and_normalize(query_vecs, kernel, bias)
    _, idxs = index.search(query_vecs, args.top_k)

    enriched = []
    for qrow, cand_ids in zip(query_rows, idxs.tolist()):
        best = None
        for cid in cand_ids:
            crow = train_rows[cid]
            lex = jaccard_tokens(qrow["code"], crow["code"])
            syn = ast_seq_similarity(qrow.get("ast_seq", "").split(), crow.get("ast_seq", "").split())
            mixed = args.alpha * lex + (1.0 - args.alpha) * syn
            item = {
                "candidate_sample_id": crow.get("sample_id"),
                "label": crow.get("label"),
                "lexical_similarity": lex,
                "syntactic_similarity": syn,
                "mixed_score": mixed,
            }
            if best is None or item["mixed_score"] > best["mixed_score"]:
                best = item | {
                    "example_code": crow["code"],
                    "example_ast_seq": crow.get("ast_seq", ""),
                    "example_cwe": crow.get("cwe"),
                    "example_project": crow.get("project"),
                }

        out = dict(qrow)
        out["retrieval_topk_ids"] = cand_ids
        out["retrieved_demo"] = best
        out["example"] = best["example_code"]
        out["example_label"] = best["label"]
        enriched.append(out)

    write_jsonl(args.output, enriched)
    print(f"Saved {len(enriched)} retrieved queries to {args.output}")


if __name__ == "__main__":
    main()
