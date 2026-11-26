"""Re-scoring arbitrary code with the trained prefilter ensembles (robustness experiment, P7 step 5)."""
from __future__ import annotations

import json

import numpy as np
import torch

from ..config import dataset_dir, load_config
from ..features.store import encode_codes, load_store
from .model import MultiViewPrefilter
from .train import models_dir


def featurize_codes(dataset: str, codes: list[str], device: torch.device) -> dict:
    """Token/AST ids with the dataset vocabulary, numeric features and UniXcoder embeddings for new code."""
    from ..features.unixcoder import UniXcoderEncoder

    store = load_store(dataset)
    vocab = store["vocab"]
    workers = int(load_config("prefilter")["features"].get("workers", 8))
    tokens, ast, numeric, _ = encode_codes(codes, vocab, workers)
    ucfg = load_config("prefilter")["features"]["unixcoder"]
    encoder = UniXcoderEncoder(ucfg["model_dir"], ucfg.get("max_length", 512), ucfg.get("token_budget", 8192),
                               "float32", device=str(device))
    semantic = encoder.encode(codes, progress_every=1000).astype(np.float16)
    del encoder
    torch.cuda.empty_cache()
    views = {"tokens": tokens, "ast": ast, "numeric": numeric, "semantic": semantic}
    if "finetune" in load_config("prefilter")["model"].get("views", []):
        from .finetune import logits_for_codes

        views["finetune"] = logits_for_codes(dataset, codes, device)
    return views


@torch.no_grad()
def ensemble_logits(dataset: str, tag: str, seeds: list[int], views: dict, device: torch.device) -> np.ndarray:
    """Mean fusion logit s(x) of an ensemble for pre-computed views."""
    cfg = load_config("prefilter")
    outputs = []
    for seed in seeds:
        payload = torch.load(models_dir(dataset) / tag / f"seed{seed}" / "model.pt", map_location=device, weights_only=False)
        stats = payload["numeric_stats"]
        model = MultiViewPrefilter(payload["token_vocab"], payload["ast_vocab"], views["semantic"].shape[1],
                                   views["numeric"].shape[1], payload.get("model_cfg", cfg["model"]), tuple(payload["views"]))
        model.load_state_dict(payload["state"])
        model.to(device).eval()
        numeric = np.clip((views["numeric"] - np.asarray(stats["mean"])) / np.asarray(stats["std"]), -10, 10).astype(np.float32)
        ft_stats = payload.get("finetune_stats") or {"mean": 0.0, "std": 1.0}
        raw_ft = views.get("finetune", np.zeros(len(numeric), np.float32))
        finetune = ((raw_ft - ft_stats["mean"]) / ft_stats["std"]).astype(np.float32)
        logits = []
        for start in range(0, len(numeric), 1024):
            sl = slice(start, start + 1024)
            out = model(torch.from_numpy(views["tokens"][sl]).long().to(device), torch.from_numpy(views["ast"][sl]).long().to(device),
                        torch.from_numpy(views["semantic"][sl]).float().to(device), torch.from_numpy(numeric[sl]).to(device),
                        torch.from_numpy(finetune[sl]).to(device))
            logits.append(out["fus"].float().cpu().numpy())
        outputs.append(np.concatenate(logits))
    return np.mean(outputs, axis=0)
