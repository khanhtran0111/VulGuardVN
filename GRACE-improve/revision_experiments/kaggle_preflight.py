"""Kaggle input preparation and fail-fast checks for model notebooks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


SEMANTIC_MODEL_ID = "microsoft/unixcoder-base-nine"
QWEN_MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"


def _weights_exist(model_dir: Path) -> bool:
    return any(model_dir.glob("*.safetensors")) or any(model_dir.glob("pytorch_model*.bin"))


def semantic_model_ready(model_dir: str | Path) -> bool:
    model = Path(model_dir)
    tokenizer_ready = any((model / name).is_file() for name in ("tokenizer.json", "spiece.model", "vocab.json"))
    return (model / "config.json").is_file() and tokenizer_ready and _weights_exist(model)


def llm_model_ready(model_dir: str | Path) -> bool:
    model = Path(model_dir)
    return (model / "config.json").is_file() and _weights_exist(model)


def _find_input_file(input_path: str | Path | None, filename: str) -> Path | None:
    if input_path in (None, ""):
        return None
    source = Path(input_path)
    if source.is_file():
        return source if source.name == filename else None
    if not source.is_dir():
        return None
    direct = source / filename
    if direct.is_file():
        return direct
    candidates = sorted(path for path in source.rglob(filename) if path.is_file())
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple {filename} candidates found under {source}: {candidates}")
    return candidates[0] if candidates else None


def _find_model_directory(input_path: str | Path | None, readiness) -> Path | None:
    if input_path in (None, ""):
        return None
    source = Path(input_path)
    if source.is_dir() and readiness(source):
        return source
    if not source.is_dir():
        return None
    candidates = sorted({path.parent for path in source.rglob("config.json") if readiness(path.parent)})
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple complete model directories found under {source}: {candidates}")
    return candidates[0] if candidates else None


def _validate_devign(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Devign dataset is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Devign dataset must be a non-empty JSON list: {path}")
    usable = sum(
        1 for row in payload
        if isinstance(row, dict) and str(row.get("func") or "").strip() and row.get("target") is not None
    )
    if usable == 0:
        raise RuntimeError(f"Devign dataset contains no usable rows with `func` and `target`: {path}")
    return usable


def prepare_devign_input(devign_input: str | Path | None, repository_root: str | Path) -> Path:
    """Copy Devign into the exact location imported by baseline2/datasets.py."""
    destination = Path(repository_root) / "GRACE-improve" / "data" / "function.json"
    source = _find_input_file(devign_input, "function.json")
    if source is not None:
        _validate_devign(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    if not destination.is_file():
        raise FileNotFoundError(
            "Devign dataset was not found. Attach a Kaggle Dataset containing `function.json` "
            f"and set DEVIGN_INPUT. Expected pipeline path: {destination}"
        )
    usable = _validate_devign(destination)
    print(f"[preflight] Devign ready: {destination} ({usable} usable rows)")
    return destination


def prepare_semantic_model(
    semantic_model_input: str | Path | None,
    repository_root: str | Path,
    *,
    auto_download: bool,
    model_id: str = SEMANTIC_MODEL_ID,
) -> Path:
    mounted = _find_model_directory(semantic_model_input, semantic_model_ready)
    if mounted is not None:
        print(f"[preflight] Semantic encoder mounted: {mounted}")
        return mounted
    default_dir = (
        Path(repository_root) / "GRACE-improve" / "baseline" / "artifacts" / "models"
        / "retrieval" / model_id.replace("/", "--")
    )
    if semantic_model_ready(default_dir):
        print(f"[preflight] Semantic encoder ready: {default_dir}")
        return default_dir
    if not auto_download:
        raise FileNotFoundError(
            "Semantic encoder is missing. Attach a Kaggle Dataset containing the UniXcoder snapshot "
            "and set SEMANTIC_MODEL_INPUT, or enable Internet and set AUTO_DOWNLOAD_SEMANTIC_MODEL=True."
        )
    try:
        from huggingface_hub import snapshot_download
        default_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(default_dir),
            allow_patterns=[
                "*.json", "*.txt", "*.model", "tokenizer*", "spiece.model",
                "merges.txt", "vocab.*", "*.safetensors", "pytorch_model*.bin",
            ],
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not download semantic encoder {model_id}. Confirm Kaggle Internet/HF access: {exc}"
        ) from exc
    if not semantic_model_ready(default_dir):
        raise RuntimeError(f"Downloaded semantic encoder is incomplete: {default_dir}")
    print(f"[preflight] Downloaded semantic encoder: {default_dir}")
    return default_dir


def prepare_qwen_model(
    qwen_model_input: str | Path | None,
    repository_root: str | Path,
    *,
    auto_download: bool,
    required: bool,
    model_id: str = QWEN_MODEL_ID,
) -> Path | None:
    if not required:
        print("[preflight] Qwen is not required for this run mode; model preparation skipped.")
        return None
    mounted = _find_model_directory(qwen_model_input, llm_model_ready)
    if mounted is not None:
        print(f"[preflight] Qwen mounted: {mounted}")
        return mounted
    default_dir = (
        Path(repository_root) / "GRACE-improve" / "baseline" / "artifacts" / "models"
        / "local_llm" / model_id.replace("/", "--")
    )
    if llm_model_ready(default_dir):
        print(f"[preflight] Qwen ready: {default_dir}")
        return default_dir
    if not auto_download:
        raise FileNotFoundError(
            "Qwen is required but missing. Attach a Kaggle Dataset containing the Qwen snapshot and "
            "set QWEN_MODEL_INPUT, or enable Internet and set AUTO_DOWNLOAD_MODEL=True."
        )
    try:
        from huggingface_hub import snapshot_download
        default_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=model_id,
            local_dir=str(default_dir),
            allow_patterns=[
                "*.json", "*.txt", "*.model", "tokenizer*", "merges.txt",
                "vocab.*", "*.safetensors", "pytorch_model*.bin",
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Could not download Qwen model {model_id}. Confirm Kaggle Internet/HF access: {exc}") from exc
    if not llm_model_ready(default_dir):
        raise RuntimeError(f"Downloaded Qwen model is incomplete: {default_dir}")
    print(f"[preflight] Downloaded Qwen: {default_dir}")
    return default_dir


__all__ = [
    "QWEN_MODEL_ID",
    "SEMANTIC_MODEL_ID",
    "llm_model_ready",
    "prepare_devign_input",
    "prepare_qwen_model",
    "prepare_semantic_model",
    "semantic_model_ready",
]
