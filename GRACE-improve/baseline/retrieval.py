from difflib import SequenceMatcher

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import build_skeleton, build_structure_summary, ensure_dir, tokenize_code, truncate_text


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _interleave(primary: list[dict], secondary: list[dict], total_k: int) -> list[dict]:
    results = []
    while len(results) < total_k and (primary or secondary):
        if primary:
            results.append(primary.pop(0))
        if len(results) >= total_k:
            break
        if secondary:
            results.append(secondary.pop(0))
    while len(results) < total_k and primary:
        results.append(primary.pop(0))
    while len(results) < total_k and secondary:
        results.append(secondary.pop(0))
    return results


def build_demo_bank(records: list[dict], max_examples_per_label: int, max_features: int, random_seed: int) -> dict:
    rng = np.random.default_rng(random_seed)
    grouped = {0: [], 1: []}
    for record in records:
        grouped[int(record["label"])].append(record)
    sampled = []
    for label, items in grouped.items():
        if len(items) > max_examples_per_label:
            indices = rng.choice(len(items), size=max_examples_per_label, replace=False)
            items = [items[index] for index in sorted(indices)]
        for record in items:
            tokens = tokenize_code(record["code"])
            sampled.append(
                {
                    "record_id": record["record_id"],
                    "label": int(record["label"]),
                    "project": record.get("project", ""),
                    "code": record["code"],
                    "tokenized_text": " ".join(tokens),
                    "tokens": tokens,
                    "skeleton": build_skeleton(record["code"]),
                    "structure_summary": build_structure_summary(record["code"]),
                }
            )
    vectorizer = TfidfVectorizer(
        tokenizer=str.split,
        preprocessor=None,
        token_pattern=None,
        lowercase=False,
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([row["tokenized_text"] for row in sampled])
    label_indices = {
        0: [index for index, row in enumerate(sampled) if row["label"] == 0],
        1: [index for index, row in enumerate(sampled) if row["label"] == 1],
    }
    return {
        "records": sampled,
        "vectorizer": vectorizer,
        "matrix": matrix,
        "label_indices": label_indices,
    }


def save_demo_bank(path, bank: dict) -> None:
    ensure_dir(path.parent)
    joblib.dump(bank, path)


def load_demo_bank(path) -> dict:
    return joblib.load(path)


def retrieve_examples(
    query_code: str,
    bank: dict,
    total_k: int,
    calibrated_probability: float,
    candidate_pool_size: int = 24,
    demo_char_limit: int = 1800,
) -> list[dict]:
    vectorizer = bank["vectorizer"]
    matrix = bank["matrix"]
    records = bank["records"]
    label_indices = bank["label_indices"]
    query_tokens = tokenize_code(query_code)
    query_skeleton = build_skeleton(query_code)
    query_vector = vectorizer.transform([" ".join(query_tokens)])
    semantic = linear_kernel(query_vector, matrix).ravel()
    vulnerable_ratio = 0.5 if total_k <= 2 else max(0.5, min(0.75, calibrated_probability))
    vulnerable_k = min(total_k, max(1, int(round(total_k * vulnerable_ratio))))
    benign_k = max(0, total_k - vulnerable_k)

    def rank_label(label: int, needed: int) -> list[dict]:
        if needed <= 0 or not label_indices[label]:
            return []
        candidates = label_indices[label]
        candidate_scores = semantic[candidates]
        top_local = np.argsort(candidate_scores)[-candidate_pool_size:][::-1]
        scored = []
        for local_index in top_local:
            bank_index = candidates[int(local_index)]
            record = records[bank_index]
            lexical = _jaccard(query_tokens, record["tokens"])
            syntactic = SequenceMatcher(None, query_skeleton, record["skeleton"]).ratio()
            score = 0.7 * lexical + 0.3 * syntactic
            scored.append((score, semantic[bank_index], record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = []
        for _, _, record in scored[:needed]:
            selected.append(
                {
                    "record_id": record["record_id"],
                    "label": record["label"],
                    "project": record["project"],
                    "structure_summary": record["structure_summary"],
                    "code": truncate_text(record["code"], demo_char_limit),
                }
            )
        return selected

    vulnerable = rank_label(1, vulnerable_k)
    benign = rank_label(0, benign_k)
    results = _interleave(vulnerable, benign, total_k)
    if len(results) < total_k:
        spill = rank_label(1, total_k - len(results)) + rank_label(0, total_k - len(results))
        for record in spill:
            if all(existing["record_id"] != record["record_id"] for existing in results):
                results.append(record)
            if len(results) >= total_k:
                break
    return results
