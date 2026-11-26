"""Contrastive retrieval (plan P4 step 3, §6.4): key = (CWE, sink API, slice embedding) -> 1-2 demo pairs.

Demo bank (train only, never the dev set):
  * datasets with pairs (PrimeVul, Big-Vul): one entry per (vulnerable, patched) pair. The demo text is the diff
    region (changed lines +- context) of each version, so no Joern run is needed for the bank; its embedding is the
    UniXcoder embedding of the vulnerable region.
  * datasets without pairs (Devign, ReVeal): train functions; 1 vulnerable + 1 non-vulnerable nearest demo, using
    the feature-store function embeddings.
"""
from __future__ import annotations

import difflib
import json

import numpy as np
import pandas as pd

from ..common import atomic_save_npy, atomic_to_parquet, extract_calls
from ..config import dataset_dir, get_logger, load_config
from ..data.io import load_records
from ..features.store import load_store
from ..localizer import suspicious_lines

log = get_logger("p4.retrieval")
BANK_COLUMNS = ["demo_id", "kind", "label", "cwe", "vulnerable", "patched", "calls", "hashes", "pair_id"]


def _region(lines: list[str], start: int, end: int, context: int, cap: int) -> str:
    lo, hi = max(0, start - context), min(len(lines), max(end, start + 1) + context)
    chunk = lines[lo:hi]
    if len(chunk) > cap:
        chunk = chunk[:cap] + ["    ..."]
    prefix = ["    ..."] if lo > 0 else []
    suffix = ["    ..."] if hi < len(lines) and len(chunk) <= cap else []
    return "\n".join(prefix + chunk + suffix)


def diff_regions(vulnerable: str, patched: str, context: int, cap: int) -> tuple[str, str]:
    a, b = vulnerable.split("\n"), patched.split("\n")
    ops = [op for op in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes() if op[0] != "equal"]
    if not ops:
        return _region(a, 0, cap, 0, cap), _region(b, 0, cap, 0, cap)
    i1, i2 = ops[0][1], ops[-1][2]
    j1, j2 = ops[0][3], ops[-1][4]
    return _region(a, i1, i2, context, cap), _region(b, j1, j2, context, cap)


def _encoder_device() -> str:
    import torch

    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > 2.0 * 1024 ** 3:
            return "cuda"
    return "cpu"


def get_text_encoder():
    from ..features.unixcoder import UniXcoderEncoder

    ucfg = load_config("prefilter")["features"]["unixcoder"]
    device = _encoder_device()
    log.info("retrieval encoder on %s", device)
    return UniXcoderEncoder(ucfg["model_dir"], 256, 4096, "float32", device=device)


def bank_dir(dataset: str):
    return dataset_dir(dataset, "retrieval")


def build_demo_bank(dataset: str, force: bool = False) -> pd.DataFrame:
    out = bank_dir(dataset)
    if (out / "demo_bank.parquet").exists() and not force:
        return pd.read_parquet(out / "demo_bank.parquet")
    cfg = load_config("llm")["retrieval"]
    has_pairs = bool(load_config("datasets")["datasets"][dataset].get("has_pairs"))
    records = load_records(dataset, ["id", "code", "label", "cwe", "pair_id", "pair_role", "code_hash", "demo_bank"])
    bank = records[records["demo_bank"]]
    if has_pairs:
        rows = []
        for pair_id, group in bank.groupby("pair_id"):
            roles = {row.pair_role: row for row in group.itertuples(index=False)}
            if "vulnerable" not in roles or "patched" not in roles:
                continue
            vul, pat = roles["vulnerable"], roles["patched"]
            vul_text, pat_text = diff_regions(vul.code, pat.code, int(cfg["context_lines"]), int(cfg["max_demo_lines"]))
            rows.append({"demo_id": pair_id, "kind": "pair", "label": 1, "cwe": vul.cwe, "vulnerable": vul_text,
                         "patched": pat_text, "calls": json.dumps(extract_calls(vul_text)),
                         "hashes": json.dumps([vul.code_hash, pat.code_hash]), "pair_id": pair_id})
        frame = pd.DataFrame(rows, columns=BANK_COLUMNS)
        emb = get_text_encoder().encode(frame["vulnerable"].tolist(), progress_every=2000).astype(np.float32) \
            if len(frame) else np.zeros((0, 768), np.float32)
    else:
        store = load_store(dataset)
        row_of = {rid: i for i, rid in enumerate(store["ids"]["id"])}
        rows = []
        for row in bank.itertuples(index=False):
            lines = row.code.split("\n")
            top = suspicious_lines(row.code, 1)
            center = top[0]["line"] - 1 if top else 0
            excerpt = _region(lines, center, center + 1, 12, int(cfg["max_demo_lines"]))
            rows.append({"demo_id": row.id, "kind": "single", "label": int(row.label), "cwe": row.cwe,
                         "vulnerable": excerpt, "patched": None, "calls": json.dumps(extract_calls(excerpt)),
                         "hashes": json.dumps([row.code_hash]), "pair_id": None, "_row": row_of[row.id]})
        frame = pd.DataFrame(rows)
        emb = np.asarray(store["unixcoder"][frame["_row"].to_numpy()], dtype=np.float32)
        frame = frame.drop(columns="_row")
    atomic_save_npy(out / "demo_emb.npy", emb)
    atomic_to_parquet(frame, out / "demo_bank.parquet")                 # marker, written last
    log.info("[%s] demo bank: %d entries (%s)", dataset, len(frame), "pairs" if has_pairs else "single functions")
    return frame


class Retriever:
    def __init__(self, dataset: str) -> None:
        self.dataset = dataset
        self.cfg = load_config("llm")["retrieval"]
        self.bank = build_demo_bank(dataset)
        emb = np.load(bank_dir(dataset) / "demo_emb.npy").astype(np.float32)
        self.emb = emb if len(self.bank) else np.zeros((0, 768), np.float32)
        if len(self.bank) == 0:
            log.info("[%s] demo bank is empty - Pass 2 runs without demos", dataset)
        norms = np.linalg.norm(self.emb, axis=1, keepdims=True)
        self.emb /= np.clip(norms, 1e-6, None)
        self.cwe = self.bank["cwe"].fillna("").str.upper().to_numpy()
        self.calls = [set(json.loads(c)) for c in self.bank["calls"]]
        self.hashes = [set(json.loads(h)) for h in self.bank["hashes"]]
        self.pair_ids = self.bank["pair_id"].to_numpy()
        self.labels = self.bank["label"].to_numpy()
        self.has_pairs = bool((self.bank["kind"] == "pair").any())

    def max_similarity(self, query_emb: np.ndarray) -> np.ndarray:
        if len(self.bank) == 0:
            return np.zeros(len(query_emb))
        q = query_emb / np.clip(np.linalg.norm(query_emb, axis=1, keepdims=True), 1e-6, None)
        sims = []
        for start in range(0, len(q), 4096):
            sims.append((q[start: start + 4096] @ self.emb.T).max(axis=1))
        return np.concatenate(sims) if sims else np.zeros(0)

    def query(self, query_emb: np.ndarray, *, cwe: str | None, sink: str | None, exclude_hashes: set[str],
              exclude_pair: str | None, single_demo: bool = False) -> tuple[list[dict], float]:
        if len(self.bank) == 0:
            return [], 0.0
        q = query_emb / max(np.linalg.norm(query_emb), 1e-6)
        sims = self.emb @ q
        score = sims.copy()
        if cwe:
            score += float(self.cfg["cwe_bonus"]) * np.array([cwe.upper() in c for c in self.cwe])
        if sink:
            score += float(self.cfg["sink_bonus"]) * np.array([sink in calls for calls in self.calls])
        for index in np.argsort(-score)[:64]:
            if self.hashes[index] & exclude_hashes or (exclude_pair and self.pair_ids[index] == exclude_pair):
                score[index] = -np.inf
        demos: list[dict] = []
        if self.has_pairs:
            k = 1 if single_demo else int(self.cfg["top_k_pairs"])
            for index in np.argsort(-score)[:k]:
                row = self.bank.iloc[index]
                demos.append({"demo_id": row["demo_id"], "cwe": row["cwe"], "vulnerable": row["vulnerable"],
                              "patched": None if single_demo else row["patched"], "label_text": "vulnerable",
                              "score": float(score[index])})
        else:
            for label, text in [(1, "vulnerable"), (0, "not vulnerable")]:
                if single_demo and label == 0:
                    continue
                masked = np.where(self.labels == label, score, -np.inf)
                index = int(np.argmax(masked))
                if np.isfinite(masked[index]):
                    row = self.bank.iloc[index]
                    demos.append({"demo_id": row["demo_id"], "cwe": row["cwe"], "vulnerable": row["vulnerable"],
                                  "patched": None, "label_text": text, "score": float(masked[index])})
        return demos, float(np.max(sims)) if len(sims) else 0.0
