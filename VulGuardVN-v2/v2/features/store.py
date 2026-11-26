"""Feature store per dataset (plan P3 step 1): token ids, AST-like ids, 24 numeric, UniXcoder embedding.

Layout `artifacts/{dataset}/features/`:
  ids.parquet         row order (= records.parquet order), n_code_tokens, n_lines
  vocab.json          token / AST vocabularies built on train_fit only
  tokens.npy          int32 [N, token_seq_len]      ast.npy  int32 [N, ast_seq_len]
  numeric.npy         float32 [N, 24]               unixcoder.npy float16 [N, 768]
"""
from __future__ import annotations

import json
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import atomic_save_npy, atomic_to_parquet, atomic_write_text
from ..config import dataset_dir, get_logger
from ..data.io import load_records
from .heuristic import NUMERIC_FEATURE_NAMES, featurize

log = get_logger("features")
PAD, UNK = 0, 1
_VOCAB: dict = {}


def _count_chunk(args: tuple[list[str], int, int]) -> tuple[Counter, Counter]:
    codes, token_len, ast_len = args
    tok, ast = Counter(), Counter()
    for code in codes:
        tokens, seq, *_ = featurize(code)
        tok.update(tokens[:token_len])
        ast.update(seq.split()[:ast_len])
    return tok, ast


def _init_worker(vocab: dict) -> None:
    _VOCAB.clear()
    _VOCAB.update(vocab)


def _encode(tokens: list[str], mapping: dict, length: int) -> np.ndarray:
    out = np.zeros(length, dtype=np.int32)
    ids = [mapping.get(token, UNK) for token in tokens[:length]]
    out[: len(ids)] = ids
    return out


def _encode_chunk(codes: list[str]):
    token_map, ast_map = _VOCAB["token"], _VOCAB["ast"]
    token_len, ast_len = _VOCAB["token_len"], _VOCAB["ast_len"]
    tok = np.zeros((len(codes), token_len), dtype=np.int32)
    ast = np.zeros((len(codes), ast_len), dtype=np.int32)
    num = np.zeros((len(codes), len(NUMERIC_FEATURE_NAMES)), dtype=np.float32)
    stats = np.zeros((len(codes), 2), dtype=np.int32)
    for row, code in enumerate(codes):
        tokens, seq, numeric, n_tokens, n_lines = featurize(code)
        tok[row] = _encode(tokens, token_map, token_len)
        ast[row] = _encode(seq.split(), ast_map, ast_len)
        num[row] = numeric
        stats[row] = (n_tokens, n_lines)
    return tok, ast, num, stats


def _chunks(items: list, size: int) -> list[list]:
    return [items[i: i + size] for i in range(0, len(items), size)]


def build_vocab(counter: Counter, size: int, min_freq: int) -> dict[str, int]:
    mapping = {"<pad>": PAD, "<unk>": UNK}
    for token, count in counter.most_common():
        if count < min_freq or len(mapping) >= size:
            break
        mapping[token] = len(mapping)
    return mapping


def encode_codes(codes: list[str], vocab: dict, workers: int) -> tuple[np.ndarray, ...]:
    with Pool(workers, initializer=_init_worker, initargs=(vocab,)) as pool:
        parts = pool.map(_encode_chunk, _chunks(codes, 512))
    return tuple(np.concatenate([part[i] for part in parts]) for i in range(4))


def build_heuristic_views(dataset: str, cfg: dict, force: bool = False) -> None:
    fdir = dataset_dir(dataset, "features")
    if heuristic_ready(dataset) and not force:
        log.info("[%s] heuristic/token/AST views exist - skip", dataset)
        return
    records = load_records(dataset, ["id", "code", "subset"])
    codes = records["code"].tolist()
    workers = int(cfg.get("workers", 8))
    token_len, ast_len = int(cfg["token_seq_len"]), int(cfg["ast_seq_len"])
    started = time.perf_counter()
    train_codes = records.loc[records["subset"] == "train_fit", "code"].tolist()
    with Pool(workers) as pool:
        counts = pool.map(_count_chunk, [(chunk, token_len, ast_len) for chunk in _chunks(train_codes, 512)])
    token_counter, ast_counter = Counter(), Counter()
    for tok, ast in counts:
        token_counter.update(tok)
        ast_counter.update(ast)
    vocab = {"token": build_vocab(token_counter, int(cfg["token_vocab_size"]), int(cfg["min_freq"])),
             "ast": build_vocab(ast_counter, int(cfg["ast_vocab_size"]), int(cfg["min_freq"])),
             "token_len": token_len, "ast_len": ast_len}
    log.info("[%s] vocab token=%d ast=%d (train_fit=%d) in %.0fs", dataset, len(vocab["token"]), len(vocab["ast"]),
             len(train_codes), time.perf_counter() - started)
    tok, ast, num, stats = encode_codes(codes, vocab, workers)
    (fdir / "feature_names.json").unlink(missing_ok=True)          # completion marker, written last
    atomic_save_npy(fdir / "tokens.npy", tok)
    atomic_save_npy(fdir / "ast.npy", ast)
    atomic_save_npy(fdir / "numeric.npy", num)
    atomic_to_parquet(pd.DataFrame({"id": records["id"], "n_code_tokens": stats[:, 0], "n_lines": stats[:, 1]}),
                      fdir / "ids.parquet")
    atomic_write_text(fdir / "vocab.json", json.dumps(vocab))
    atomic_write_text(fdir / "feature_names.json", json.dumps(NUMERIC_FEATURE_NAMES))
    log.info("[%s] heuristic views for %d functions in %.0fs", dataset, len(codes), time.perf_counter() - started)


def build_unixcoder_view(dataset: str, cfg: dict, force: bool = False) -> None:
    from .unixcoder import get_encoder

    fdir = dataset_dir(dataset, "features")
    final = fdir / "unixcoder.npy"
    if final.exists() and not force:
        log.info("[%s] UniXcoder view exists - skip", dataset)
        return
    ucfg = cfg["unixcoder"]
    records = load_records(dataset, ["id", "code"])
    codes = records["code"].tolist()
    chunk_dir = dataset_dir(dataset, "features", "unixcoder_chunks")
    chunk_size = int(ucfg.get("chunk_size", 16384))
    encoder = get_encoder(ucfg)
    n_chunks = (len(codes) + chunk_size - 1) // chunk_size
    started = time.perf_counter()
    for index in range(n_chunks):
        path = chunk_dir / f"{index:04d}.npy"
        if path.exists() and not force:
            continue
        chunk = codes[index * chunk_size: (index + 1) * chunk_size]
        t0 = time.perf_counter()
        emb = encoder.encode(chunk, progress_every=4096)
        atomic_save_npy(path, emb.astype(np.float16))
        log.info("[%s] UniXcoder chunk %d/%d (%d fn) %.1f fn/s", dataset, index + 1, n_chunks, len(chunk),
                 len(chunk) / max(time.perf_counter() - t0, 1e-6))
    parts = [np.load(chunk_dir / f"{index:04d}.npy") for index in range(n_chunks)]
    merged = np.concatenate(parts).astype(np.float16)
    if len(merged) != len(codes):
        raise RuntimeError(f"[{dataset}] UniXcoder chunks cover {len(merged)} rows, expected {len(codes)} - delete "
                           f"{chunk_dir} and rerun")
    atomic_write_text(fdir / "unixcoder_config.json", json.dumps(encoder.export_config()))
    atomic_save_npy(final, merged)
    for index in range(n_chunks):
        (chunk_dir / f"{index:04d}.npy").unlink(missing_ok=True)
    log.info("[%s] UniXcoder view for %d functions in %.0fs", dataset, len(codes), time.perf_counter() - started)


def load_store(dataset: str, mmap: bool = True) -> dict:
    fdir = dataset_dir(dataset, "features")
    mode = "r" if mmap else None
    store = {
        "ids": pd.read_parquet(fdir / "ids.parquet"),
        "tokens": np.load(fdir / "tokens.npy", mmap_mode=mode),
        "ast": np.load(fdir / "ast.npy", mmap_mode=mode),
        "numeric": np.load(fdir / "numeric.npy", mmap_mode=mode),
        "vocab": json.loads((fdir / "vocab.json").read_text(encoding="utf-8")),
    }
    emb_path = fdir / "unixcoder.npy"
    store["unixcoder"] = np.load(emb_path, mmap_mode=mode) if emb_path.exists() else None
    return store


def heuristic_ready(dataset: str) -> bool:
    fdir = dataset_dir(dataset, "features")
    return all((fdir / name).exists() for name in ["tokens.npy", "ast.npy", "numeric.npy", "ids.parquet", "vocab.json",
                                                    "feature_names.json"])


def store_ready(dataset: str) -> bool:
    fdir = dataset_dir(dataset, "features")
    return all((fdir / name).exists() for name in ["numeric.npy", "tokens.npy", "ast.npy", "unixcoder.npy"])
