import json
import os
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from common import MODELS_DIR, ensure_dir, tokenize_code, truncate_text
from graphs import get_graph_features, resolve_graph_backend_with_notice


DEMO_BANK_SCHEMA_VERSION = 2
DEFAULT_RETRIEVAL_MODEL_REPO_ID = "Salesforce/codet5-base"
DEFAULT_EMBEDDING_MAX_LENGTH = 512
DEFAULT_EMBEDDING_BATCH_SIZE = 16
DEFAULT_LEXICAL_WEIGHT = 0.7
DEFAULT_SYNTACTIC_WEIGHT = 0.3
AST_SIMILARITY_MAX_TOKENS = 192
DEFAULT_BUILD_PROGRESS_EVERY = 250

_ENCODER_CACHE: dict[tuple[str, str], "SemanticRetrievalEncoder"] = {}


def default_retrieval_model_dir(repo_id: str = DEFAULT_RETRIEVAL_MODEL_REPO_ID) -> Path:
    return MODELS_DIR / "retrieval" / repo_id.replace("/", "--")


def is_retrieval_model_downloaded(model_dir: Path) -> bool:
    required = ["config.json"]
    tokenizers = ["tokenizer.json", "spiece.model", "vocab.json"]
    weights = list(model_dir.glob("*.safetensors")) or list(model_dir.glob("pytorch_model*.bin"))
    return all((model_dir / name).exists() for name in required) and any((model_dir / name).exists() for name in tokenizers) and bool(weights)


def download_retrieval_model_snapshot(repo_id: str, local_dir: Path | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `huggingface_hub`. Install it before downloading the retrieval model."
        ) from exc
    target_dir = Path(local_dir or default_retrieval_model_dir(repo_id))
    ensure_dir(target_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
        token=_resolve_hf_token(),
        allow_patterns=[
            "*.json",
            "*.txt",
            "*.model",
            "tokenizer*",
            "spiece.model",
            "*.safetensors",
            "pytorch_model*.bin",
        ],
    )
    return target_dir


class SemanticRetrievalEncoder:
    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RETRIEVAL_MODEL_REPO_ID,
        model_dir: Path | None = None,
        max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        auto_download: bool = False,
    ) -> None:
        self.model_name = model_name
        self.model_dir = Path(model_dir or default_retrieval_model_dir(model_name))
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.auto_download = auto_download
        self._runtime: dict[str, Any] | None = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def export_config(self) -> dict[str, Any]:
        return {
            "semantic_backend": "codet5",
            "model_name": self.model_name,
            "model_dir": str(self.model_dir),
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "auto_download": self.auto_download,
        }

    def prepare(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        if self.auto_download and not is_retrieval_model_downloaded(self.model_dir):
            download_retrieval_model_snapshot(self.model_name, self.model_dir)
        if not is_retrieval_model_downloaded(self.model_dir):
            raise FileNotFoundError(
                f"Retrieval model directory is missing or incomplete: {self.model_dir}. "
                "Download the CodeT5 checkpoint or enable auto-download."
            )
        runtime = self._load_runtime()
        torch = runtime["torch"]
        AutoModel = runtime["AutoModel"]
        AutoTokenizer = runtime["AutoTokenizer"]
        RobertaTokenizer = runtime["RobertaTokenizer"]
        T5EncoderModel = runtime["T5EncoderModel"]
        tokenizer = None
        vocab_path = self.model_dir / "vocab.json"
        merges_path = self.model_dir / "merges.txt"
        if vocab_path.exists() and merges_path.exists():
            tokenizer = RobertaTokenizer(
                vocab_file=str(vocab_path),
                merges_file=str(merges_path),
                errors="replace",
                bos_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                cls_token="<s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                add_prefix_space=False,
            )
            tokenizer.model_max_length = self.max_length
            tokenizer.additional_special_tokens = [f"<extra_id_{index}>" for index in range(99, -1, -1)]
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_fast=False,
                )
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_dir,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_fast=True,
                )
        config_payload = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        if str(config_payload.get("model_type", "")).strip().lower() == "t5":
            model = T5EncoderModel.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        else:
            model = AutoModel.from_pretrained(self.model_dir, local_files_only=True, trust_remote_code=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._model = model
        self._device = device

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        self.prepare()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        torch = self._runtime["torch"]
        embeddings: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(self._device) for name, tensor in encoded.items()}
            with torch.inference_mode():
                outputs = self._model(**encoded)
            hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1.0)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(normalized.detach().cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer, RobertaTokenizer, T5EncoderModel
        except Exception as exc:
            raise RuntimeError(
                "Missing retrieval dependencies. Install `torch`, `transformers`, and tokenizer dependencies "
                "such as `sentencepiece` before using CodeT5 retrieval."
            ) from exc
        self._runtime = {
            "torch": torch,
            "AutoModel": AutoModel,
            "AutoTokenizer": AutoTokenizer,
            "RobertaTokenizer": RobertaTokenizer,
            "T5EncoderModel": T5EncoderModel,
        }
        return self._runtime


def build_demo_bank(
    records: list[dict],
    max_examples_per_label: int,
    max_features: int,
    random_seed: int,
    *,
    semantic_backend: str = "auto",
    semantic_model_name: str = DEFAULT_RETRIEVAL_MODEL_REPO_ID,
    semantic_model_dir: Path | None = None,
    semantic_batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    semantic_max_length: int = DEFAULT_EMBEDDING_MAX_LENGTH,
    auto_download_semantic_model: bool = False,
    graph_backend: str = "auto",
    progress_every: int = DEFAULT_BUILD_PROGRESS_EVERY,
) -> dict:
    rng = np.random.default_rng(random_seed)
    grouped = {0: [], 1: []}
    for record in records:
        grouped[int(record["label"])].append(record)

    sampled = []
    graph_backend_counts = {}
    graph_backend_resolved, graph_backend_notice = resolve_graph_backend_with_notice(graph_backend)
    total_records = sum(min(len(items), max_examples_per_label) for items in grouped.values())
    started = time.perf_counter()
    print(
        f"[demo-bank] Building graph features for {total_records} records "
        f"(graph={graph_backend_resolved}, semantic={semantic_backend})"
    )
    if graph_backend_notice:
        print(f"[demo-bank] Graph backend notice: {graph_backend_notice}")
    processed = 0
    for label, items in grouped.items():
        if len(items) > max_examples_per_label:
            indices = rng.choice(len(items), size=max_examples_per_label, replace=False)
            items = [items[index] for index in sorted(indices)]
        for record in items:
            graph_features = get_graph_features(record, graph_backend=graph_backend)
            graph_backend_counts[graph_features["backend"]] = graph_backend_counts.get(graph_features["backend"], 0) + 1
            tokens = tokenize_code(record["code"])
            ast_sequence = graph_features.get("ast_sequence", "")
            sampled.append(
                {
                    "record_id": record["record_id"],
                    "label": int(record["label"]),
                    "project": record.get("project", ""),
                    "dataset": record.get("dataset", ""),
                    "code": record["code"],
                    "tokenized_text": " ".join(tokens),
                    "tokens": tokens,
                    "ast_sequence": ast_sequence,
                    "ast_tokens": ast_sequence.split(),
                    "graph_backend": graph_features["backend"],
                }
            )
            processed += 1
            if progress_every and (processed % progress_every == 0 or processed == total_records):
                elapsed = time.perf_counter() - started
                print(f"[demo-bank] Graph features ready: {processed}/{total_records} in {elapsed:.1f}s")

    print(f"[demo-bank] Building semantic store with backend request={semantic_backend}")
    semantic_backend_used, semantic_payload, semantic_notice = _build_semantic_store(
        sampled,
        semantic_backend=semantic_backend,
        semantic_model_name=semantic_model_name,
        semantic_model_dir=semantic_model_dir,
        semantic_batch_size=semantic_batch_size,
        semantic_max_length=semantic_max_length,
        max_features=max_features,
        auto_download_semantic_model=auto_download_semantic_model,
    )
    print(f"[demo-bank] Semantic store ready with backend={semantic_backend_used}")
    label_indices = {
        0: [index for index, row in enumerate(sampled) if row["label"] == 0],
        1: [index for index, row in enumerate(sampled) if row["label"] == 1],
    }
    return {
        "schema_version": DEMO_BANK_SCHEMA_VERSION,
        "records": sampled,
        "label_indices": label_indices,
        "semantic_backend": semantic_backend_used,
        "semantic_notice": semantic_notice,
        "semantic_config": semantic_payload["config"],
        "semantic_store": semantic_payload["store"],
        "graph_backend_requested": graph_backend,
        "graph_backend_resolved": graph_backend_resolved,
        "graph_backend_notice": graph_backend_notice,
        "graph_backend_counts": graph_backend_counts,
        "rerank": {
            "lexical_weight": DEFAULT_LEXICAL_WEIGHT,
            "syntactic_weight": DEFAULT_SYNTACTIC_WEIGHT,
        },
    }


def _build_semantic_store(
    sampled: list[dict],
    *,
    semantic_backend: str,
    semantic_model_name: str,
    semantic_model_dir: Path | None,
    semantic_batch_size: int,
    semantic_max_length: int,
    max_features: int,
    auto_download_semantic_model: bool,
) -> tuple[str, dict, str | None]:
    requested = (semantic_backend or "auto").strip().lower()
    if requested not in {"auto", "codet5", "tfidf"}:
        raise ValueError(f"Unsupported retrieval backend: {semantic_backend}")
    if requested in {"auto", "codet5"}:
        try:
            encoder = SemanticRetrievalEncoder(
                model_name=semantic_model_name,
                model_dir=semantic_model_dir,
                max_length=semantic_max_length,
                batch_size=semantic_batch_size,
                auto_download=auto_download_semantic_model,
            )
            embeddings = encoder.encode_texts([row["code"] for row in sampled])
            return "codet5", {"config": encoder.export_config(), "store": {"embeddings": embeddings}}, None
        except Exception as exc:
            if requested == "codet5":
                raise
            semantic_notice = f"Falling back to TF-IDF retrieval because CodeT5 could not be loaded: {exc}"
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
            return "tfidf", {"config": {"semantic_backend": "tfidf", "max_features": max_features}, "store": {"vectorizer": vectorizer, "matrix": matrix}}, semantic_notice
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
    return "tfidf", {"config": {"semantic_backend": "tfidf", "max_features": max_features}, "store": {"vectorizer": vectorizer, "matrix": matrix}}, None


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
    *,
    query_record: dict | None = None,
    graph_backend: str | None = None,
    query_graph_features: dict | None = None,
) -> list[dict]:
    query_record = dict(query_record or {})
    query_record["code"] = query_code
    graph_features = query_graph_features or get_graph_features(
        query_record,
        dataset_name=query_record.get("dataset"),
        graph_backend=graph_backend or bank.get("graph_backend_requested", "auto"),
    )
    query_tokens = tokenize_code(query_code)
    query_ast_tokens = graph_features.get("ast_sequence", "").split()
    semantic = _semantic_scores(query_code, query_tokens, bank)

    label_indices = bank["label_indices"]
    records = bank["records"]
    rerank = bank.get("rerank", {})
    lexical_weight = float(rerank.get("lexical_weight", DEFAULT_LEXICAL_WEIGHT))
    syntactic_weight = float(rerank.get("syntactic_weight", DEFAULT_SYNTACTIC_WEIGHT))

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
            syntactic = _syntactic_similarity(query_ast_tokens, record["ast_tokens"])
            mixed = lexical_weight * lexical + syntactic_weight * syntactic
            scored.append((mixed, float(semantic[bank_index]), lexical, syntactic, record))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = []
        for mixed, semantic_score, lexical, syntactic, record in scored[:needed]:
            selected.append(
                {
                    "record_id": record["record_id"],
                    "label": record["label"],
                    "project": record["project"],
                    "code": truncate_text(record["code"], demo_char_limit),
                    "mixed_score": mixed,
                    "semantic_score": semantic_score,
                    "lexical_score": lexical,
                    "syntactic_score": syntactic,
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


def _semantic_scores(query_code: str, query_tokens: list[str], bank: dict) -> np.ndarray:
    backend = bank.get("semantic_backend", "tfidf")
    store = bank.get("semantic_store", {})
    if backend == "codet5":
        encoder = _get_encoder(bank.get("semantic_config", {}))
        query_embedding = encoder.encode_texts([query_code])
        return np.dot(store["embeddings"], query_embedding[0])
    vectorizer = store["vectorizer"]
    matrix = store["matrix"]
    query_vector = vectorizer.transform([" ".join(query_tokens)])
    return linear_kernel(query_vector, matrix).ravel()


def _get_encoder(config: dict) -> SemanticRetrievalEncoder:
    model_name = config.get("model_name", DEFAULT_RETRIEVAL_MODEL_REPO_ID)
    model_dir = Path(config.get("model_dir") or default_retrieval_model_dir(model_name))
    cache_key = (model_name, str(model_dir))
    encoder = _ENCODER_CACHE.get(cache_key)
    if encoder is None:
        encoder = SemanticRetrievalEncoder(
            model_name=model_name,
            model_dir=model_dir,
            max_length=int(config.get("max_length", DEFAULT_EMBEDDING_MAX_LENGTH)),
            batch_size=int(config.get("batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)),
            auto_download=bool(config.get("auto_download", False)),
        )
        _ENCODER_CACHE[cache_key] = encoder
    return encoder


def _jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def _syntactic_similarity(left: list[str], right: list[str]) -> float:
    left_seq = left[:AST_SIMILARITY_MAX_TOKENS]
    right_seq = right[:AST_SIMILARITY_MAX_TOKENS]
    if not left_seq and not right_seq:
        return 1.0
    denominator = len(left_seq) + len(right_seq)
    if denominator == 0:
        return 0.0
    distance = _levenshtein_distance(left_seq, right_seq)
    return max(0.0, (denominator - distance) / denominator)


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row_index, left_token in enumerate(left, start=1):
        current = [row_index]
        for col_index, right_token in enumerate(right, start=1):
            insert_cost = current[col_index - 1] + 1
            delete_cost = previous[col_index] + 1
            substitute_cost = previous[col_index - 1] + (0 if left_token == right_token else 1)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


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


def _resolve_hf_token() -> str | None:
    for env_name in ["HUGGINGFACE_HUB_TOKEN", "HF_TOKEN"]:
        value = os.getenv(env_name)
        if value:
            return value.strip().strip('"').strip("'")
    return None
