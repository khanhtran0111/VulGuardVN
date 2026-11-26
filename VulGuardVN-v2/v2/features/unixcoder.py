"""UniXcoder encoder (encoder-only mode, mean pooling, L2-normalised), length-sorted dynamic batching."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import get_logger, resolve_project_path

log = get_logger("unixcoder")


class UniXcoderEncoder:
    def __init__(self, model_dir: str | Path, max_length: int = 512, token_budget: int = 8192,
                 dtype: str = "float32", device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.model_dir = resolve_project_path(model_dir)
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"UniXcoder model not found in {self.model_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModel.from_pretrained(self.model_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = torch.float16 if dtype == "float16" and self.device.type == "cuda" else torch.float32
        self.model.to(self.device, dtype=self.dtype).eval()
        self.max_length = int(max_length)
        self.token_budget = int(token_budget)
        vocab = self.tokenizer.get_vocab()
        self.mode_id = vocab.get("<encoder-only>")
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id
        self.pad_id = self.tokenizer.pad_token_id
        self.body_max = self.max_length - (4 if self.mode_id is not None else 2)

    def export_config(self) -> dict:
        return {"model_dir": str(self.model_dir), "max_length": self.max_length, "encoder_only_prefix": self.mode_id is not None,
                "dtype": str(self.dtype), "pooling": "mean", "normalize": True}

    def _ids(self, texts: list[str]) -> list[list[int]]:
        body = self.tokenizer(texts, add_special_tokens=False, truncation=True, max_length=self.body_max)["input_ids"]
        prefix = [self.cls_id, self.mode_id, self.sep_id] if self.mode_id is not None else [self.cls_id]
        return [prefix + ids + [self.sep_id] for ids in body]

    def encode(self, texts: list[str], progress_every: int = 0) -> np.ndarray:
        torch = self.torch
        if not texts:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)
        ids = self._ids(texts)
        order = np.argsort([len(item) for item in ids], kind="stable")
        out = np.zeros((len(texts), self.model.config.hidden_size), dtype=np.float32)
        position, started, done = 0, time.perf_counter(), 0
        while position < len(order):
            longest = len(ids[order[min(position, len(order) - 1)]])
            batch_indices = []
            while position < len(order):
                candidate_len = max(longest, len(ids[order[position]]))
                if batch_indices and candidate_len * (len(batch_indices) + 1) > self.token_budget:
                    break
                longest = candidate_len
                batch_indices.append(order[position])
                position += 1
            width = max(len(ids[i]) for i in batch_indices)
            input_ids = torch.full((len(batch_indices), width), self.pad_id, dtype=torch.long)
            mask = torch.zeros((len(batch_indices), width), dtype=torch.long)
            for row, index in enumerate(batch_indices):
                seq = ids[index]
                input_ids[row, : len(seq)] = torch.tensor(seq)
                mask[row, : len(seq)] = 1
            input_ids, mask = input_ids.to(self.device), mask.to(self.device)
            with torch.inference_mode():
                hidden = self.model(input_ids=input_ids, attention_mask=mask).last_hidden_state.float()
            weights = mask.unsqueeze(-1).float()
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out[batch_indices] = pooled.cpu().numpy()
            done += len(batch_indices)
            if progress_every and (done // progress_every) != ((done - len(batch_indices)) // progress_every):
                rate = done / max(time.perf_counter() - started, 1e-6)
                log.info("encoded %d/%d (%.1f fn/s)", done, len(texts), rate)
        return out


_ENCODER: UniXcoderEncoder | None = None


def get_encoder(cfg: dict) -> UniXcoderEncoder:
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = UniXcoderEncoder(cfg["model_dir"], cfg.get("max_length", 512), cfg.get("token_budget", 8192),
                                    cfg.get("dtype", "float32"))
    return _ENCODER
