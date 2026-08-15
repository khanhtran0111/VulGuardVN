"""Download, cache, validate, and materialize Kaggle assets for model runs."""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


SEMANTIC_MODEL_ID = "microsoft/unixcoder-base-nine"
QWEN_MODEL_ID = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
DEFAULT_TIMEOUT_SECONDS = 120
DEFAULT_RETRIES = 3


@dataclass(frozen=True)
class PreparedAssets:
    devign_path: Path
    retrieval_model_dir: Path
    local_llm_model_dir: Path | None
    devign_source: str
    retrieval_model_source: str
    local_llm_model_source: str | None
    devign_usable_records: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "devign_path": str(self.devign_path),
            "retrieval_model_dir": str(self.retrieval_model_dir),
            "local_llm_model_dir": str(self.local_llm_model_dir) if self.local_llm_model_dir else None,
            "devign_source": self.devign_source,
            "retrieval_model_source": self.retrieval_model_source,
            "local_llm_model_source": self.local_llm_model_source,
            "devign_usable_records": self.devign_usable_records,
        }


def validate_devign_file(path: str | Path) -> int:
    """Return usable record count or raise for HTML, invalid JSON, or bad schema."""
    source = Path(path)
    if not source.is_file() or source.stat().st_size < 16:
        raise RuntimeError(f"Devign file is missing or too small: {source}")
    prefix = source.read_bytes()[:256].lstrip().lower()
    if prefix.startswith((b"<!doctype html", b"<html")):
        raise RuntimeError(f"Devign download returned HTML instead of JSON: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Devign dataset is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"Devign dataset must be a non-empty JSON list: {source}")
    usable = sum(
        1
        for row in payload
        if isinstance(row, dict)
        and str(row.get("func") or "").strip()
        and row.get("target") is not None
    )
    if usable == 0:
        raise RuntimeError(f"Devign has no usable records containing `func` and `target`: {source}")
    return usable


def _weight_files(model_dir: Path) -> list[Path]:
    return sorted([*model_dir.glob("*.safetensors"), *model_dir.glob("pytorch_model*.bin")])


def validate_hf_model_directory(model_dir: str | Path) -> Path:
    """Validate a complete Transformers snapshot, including sharded-weight index."""
    model = Path(model_dir)
    if not model.is_dir() or not (model / "config.json").is_file():
        raise RuntimeError(f"Hugging Face model is missing config.json: {model}")
    tokenizer_names = (
        "tokenizer.json", "tokenizer_config.json", "spiece.model", "vocab.json", "vocab.txt",
    )
    if not any((model / name).is_file() for name in tokenizer_names):
        raise RuntimeError(f"Hugging Face model is missing tokenizer assets: {model}")
    weights = _weight_files(model)
    if not weights:
        raise RuntimeError(f"Hugging Face model is missing weights: {model}")
    safetensor_shards = [path for path in weights if path.name.startswith("model-") and "-of-" in path.name]
    bin_shards = [path for path in weights if path.name.startswith("pytorch_model-") and "-of-" in path.name]
    if safetensor_shards and not (model / "model.safetensors.index.json").is_file():
        raise RuntimeError(f"Sharded safetensors model is missing model.safetensors.index.json: {model}")
    if bin_shards and not (model / "pytorch_model.bin.index.json").is_file():
        raise RuntimeError(f"Sharded PyTorch model is missing pytorch_model.bin.index.json: {model}")
    return model


def _valid_devign(path: Path) -> int | None:
    try:
        return validate_devign_file(path)
    except Exception:
        return None


def _valid_model(path: Path) -> bool:
    try:
        validate_hf_model_directory(path)
        return True
    except Exception:
        return False


def _find_file(source: str | Path | None, filename: str) -> Path | None:
    if source in (None, ""):
        return None
    root = Path(source)
    if root.is_file():
        return root if root.name == filename else None
    if not root.is_dir():
        return None
    direct = root / filename
    if direct.is_file():
        return direct
    candidates = sorted(path for path in root.rglob(filename) if path.is_file())
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple {filename} candidates found under {root}: {candidates}")
    return candidates[0] if candidates else None


def _find_model(source: str | Path | None) -> Path | None:
    if source in (None, ""):
        return None
    root = Path(source)
    if _valid_model(root):
        return root
    if not root.is_dir():
        return None
    candidates = sorted({path.parent for path in root.rglob("config.json") if _valid_model(path.parent)})
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple complete model snapshots found under {root}: {candidates}")
    return candidates[0] if candidates else None


def _atomic_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_remove_asset(path: Path, asset_root: Path) -> None:
    resolved = Path(os.path.abspath(path))
    root = asset_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove path outside asset root: {resolved}")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _atomic_copy_model(source: Path, destination: Path, model_root: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        shutil.copytree(source, temporary / "snapshot", dirs_exist_ok=True)
        validate_hf_model_directory(temporary / "snapshot")
        if destination.exists() or destination.is_symlink():
            _safe_remove_asset(destination, model_root)
        os.replace(temporary / "snapshot", destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _download_requests(
    url: str,
    destination: Path,
    *,
    timeout: int,
    requests_get: Callable[..., Any] | None,
) -> None:
    if requests_get is None:
        import requests

        requests_get = requests.get
    response = requests_get(url, stream=True, timeout=timeout)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()
    with destination.open("wb") as handle:
        if hasattr(response, "iter_content"):
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
        else:
            handle.write(response.content)


def _download_gdrive(
    url: str,
    destination: Path,
    *,
    gdown_download: Callable[..., Any] | None,
) -> None:
    if gdown_download is None:
        import gdown

        gdown_download = gdown.download
    result = gdown_download(url=url, output=str(destination), quiet=False, fuzzy=True)
    if not result or not destination.is_file():
        raise RuntimeError(f"gdown did not create an output file for {url}")


def _archive_member_bytes(path: Path, requested_member: str) -> bytes | None:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            member = requested_member if requested_member and requested_member in names else None
            if member is None:
                matches = [name for name in names if Path(name).name == "function.json"]
                if len(matches) != 1:
                    raise RuntimeError(f"Archive must contain exactly one function.json; found {matches}")
                member = matches[0]
            return archive.read(member)
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            members = [member for member in archive.getmembers() if member.isfile()]
            selected = next((member for member in members if requested_member and member.name == requested_member), None)
            if selected is None:
                matches = [member for member in members if Path(member.name).name == "function.json"]
                if len(matches) != 1:
                    raise RuntimeError(f"Archive must contain exactly one function.json; found {[m.name for m in matches]}")
                selected = matches[0]
            extracted = archive.extractfile(selected)
            if extracted is None:
                raise RuntimeError(f"Could not read archive member {selected.name}")
            return extracted.read()
    return None


def _normalize_devign_download(downloaded: Path, archive_member: str) -> Path:
    member_bytes = _archive_member_bytes(downloaded, archive_member)
    if member_bytes is None:
        validate_devign_file(downloaded)
        return downloaded
    normalized = downloaded.with_suffix(downloaded.suffix + ".function.json")
    normalized.write_bytes(member_bytes)
    validate_devign_file(normalized)
    return normalized


def resolve_or_download_devign(
    *,
    repository_root: str | Path,
    asset_root: str | Path,
    source: str | Path | None,
    urls: Iterable[str],
    archive_member: str = "",
    auto_download: bool = True,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = DEFAULT_RETRIES,
    requests_get: Callable[..., Any] | None = None,
    gdown_download: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Path, str, int]:
    assets = Path(asset_root)
    cache = assets / "datasets" / "devign" / "function.json"
    repository_copy = Path(repository_root) / "GRACE-improve" / "data" / "function.json"
    downloads = assets / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)

    override = _find_file(source, "function.json")
    if source not in (None, "") and Path(source).exists() and override is None:
        raise RuntimeError(f"DEVIGN_SOURCE_PATH does not contain function.json: {source}")
    if override is not None:
        usable = validate_devign_file(override)
        _atomic_copy_file(override, cache)
        status = "mounted"
    else:
        repository_usable = _valid_devign(repository_copy)
        cache_usable = _valid_devign(cache)
        if repository_usable is not None:
            _atomic_copy_file(repository_copy, cache)
            usable, status = repository_usable, "cached"
        elif cache_usable is not None:
            usable, status = cache_usable, "cached"
        else:
            if cache.exists():
                cache.unlink()
            if not auto_download:
                raise FileNotFoundError(
                    "Devign is missing. Set DEVIGN_SOURCE_PATH or enable AUTO_DOWNLOAD_DATASET_IF_MISSING. "
                    f"Expected cache: {cache}"
                )
            errors: list[str] = []
            downloaded_ok = False
            for url_index, url in enumerate(urls):
                if not str(url).strip():
                    continue
                for attempt in range(1, max(int(retries), 3) + 1):
                    temporary = downloads / f"devign_{url_index}_{attempt}.part"
                    normalized: Path | None = None
                    temporary.unlink(missing_ok=True)
                    print(f"[assets] Downloading Devign ({attempt}/{max(int(retries), 3)}): {url}")
                    try:
                        if "drive.google.com" in url:
                            _download_gdrive(url, temporary, gdown_download=gdown_download)
                        else:
                            _download_requests(url, temporary, timeout=timeout, requests_get=requests_get)
                        normalized = _normalize_devign_download(temporary, archive_member)
                        usable = validate_devign_file(normalized)
                        _atomic_copy_file(normalized, cache)
                        temporary.unlink(missing_ok=True)
                        if normalized != temporary:
                            normalized.unlink(missing_ok=True)
                        status = "downloaded"
                        downloaded_ok = True
                        break
                    except Exception as exc:
                        errors.append(f"{url} attempt {attempt}: {exc}")
                        temporary.unlink(missing_ok=True)
                        if normalized is not None and normalized != temporary:
                            normalized.unlink(missing_ok=True)
                        if attempt < max(int(retries), 3):
                            sleep(min(float(attempt), 3.0))
                if downloaded_ok:
                    break
            if not downloaded_ok:
                raise RuntimeError("Unable to download a valid Devign dataset. " + " | ".join(errors))

    usable = validate_devign_file(cache)
    _atomic_copy_file(cache, repository_copy)
    validate_devign_file(repository_copy)
    print(f"[assets] Devign source: {status}")
    print(f"[assets] Devign path: {cache}")
    print(f"[assets] Devign usable records: {usable}")
    return cache, status, usable


def resolve_or_download_hf_model(
    *,
    asset_root: str | Path,
    model_id: str,
    source_dir: str | Path | None,
    token: str | None = None,
    auto_download: bool = True,
    use_symlinks_when_possible: bool = False,
    copy_models_instead_of_link: bool = True,
    snapshot_download_fn: Callable[..., Any] | None = None,
) -> tuple[Path, str]:
    assets = Path(asset_root)
    model_root = assets / "models"
    target = model_root / model_id.replace("/", "--")
    mounted = _find_model(source_dir)
    if source_dir not in (None, "") and Path(source_dir).exists() and mounted is None:
        raise RuntimeError(f"Mounted model override is incomplete or invalid: {source_dir}")
    if mounted is not None:
        if copy_models_instead_of_link:
            if mounted.resolve() != target.resolve():
                _atomic_copy_model(mounted, target, model_root)
            resolved = target
        elif use_symlinks_when_possible:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                _safe_remove_asset(target, model_root)
            target.symlink_to(mounted, target_is_directory=True)
            resolved = target
        else:
            resolved = mounted
        validate_hf_model_directory(resolved)
        print(f"[assets] {model_id} source: mounted")
        print(f"[assets] {model_id} path: {resolved}")
        return resolved, "mounted"
    if _valid_model(target):
        print(f"[assets] {model_id} source: cached")
        print(f"[assets] {model_id} path: {target}")
        return target, "cached"
    if not auto_download:
        raise FileNotFoundError(
            f"Model {model_id} is missing. Provide a mounted source directory or enable AUTO_DOWNLOAD_MISSING_MODELS."
        )
    if target.exists() or target.is_symlink():
        _safe_remove_asset(target, model_root)
    target.mkdir(parents=True, exist_ok=True)
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download
    kwargs = {
        "repo_id": model_id,
        "local_dir": str(target),
        "token": token or None,
        "local_dir_use_symlinks": False,
    }
    print(f"[assets] Downloading Hugging Face model: {model_id}")
    try:
        snapshot_download_fn(**kwargs)
    except TypeError as exc:
        if "local_dir_use_symlinks" not in str(exc):
            raise
        kwargs.pop("local_dir_use_symlinks")
        snapshot_download_fn(**kwargs)
    try:
        validate_hf_model_directory(target)
    except Exception:
        _safe_remove_asset(target, model_root)
        raise
    print(f"[assets] {model_id} source: downloaded")
    print(f"[assets] {model_id} path: {target}")
    return target, "downloaded"


def prepare_required_assets(
    *,
    repository_root: str | Path,
    asset_root: str | Path,
    devign_source: str | Path | None,
    devign_urls: Iterable[str],
    devign_archive_member: str,
    retrieval_model_source: str | Path | None,
    retrieval_model_id: str,
    local_llm_source: str | Path | None,
    local_llm_model_id: str,
    hf_token: str | None,
    auto_download_dataset: bool,
    auto_download_models: bool,
    require_llm: bool,
    use_symlinks_when_possible: bool = False,
    copy_models_instead_of_link: bool = True,
    reset_asset_root: bool = False,
    requests_get: Callable[..., Any] | None = None,
    gdown_download: Callable[..., Any] | None = None,
    snapshot_download_fn: Callable[..., Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> PreparedAssets:
    assets = Path(asset_root)
    if reset_asset_root and assets.exists():
        parent = assets.parent.resolve()
        if assets.resolve() == parent or parent not in assets.resolve().parents:
            raise RuntimeError(f"Unsafe ASSET_ROOT reset target: {assets}")
        shutil.rmtree(assets)
    assets.mkdir(parents=True, exist_ok=True)
    devign_path, devign_status, usable = resolve_or_download_devign(
        repository_root=repository_root,
        asset_root=assets,
        source=devign_source,
        urls=devign_urls,
        archive_member=devign_archive_member,
        auto_download=auto_download_dataset,
        requests_get=requests_get,
        gdown_download=gdown_download,
        sleep=sleep,
    )
    retrieval_dir, retrieval_status = resolve_or_download_hf_model(
        asset_root=assets,
        model_id=retrieval_model_id,
        source_dir=retrieval_model_source,
        token=hf_token,
        auto_download=auto_download_models,
        use_symlinks_when_possible=use_symlinks_when_possible,
        copy_models_instead_of_link=copy_models_instead_of_link,
        snapshot_download_fn=snapshot_download_fn,
    )
    local_dir: Path | None = None
    local_status: str | None = None
    if require_llm:
        local_dir, local_status = resolve_or_download_hf_model(
            asset_root=assets,
            model_id=local_llm_model_id,
            source_dir=local_llm_source,
            token=hf_token,
            auto_download=auto_download_models,
            use_symlinks_when_possible=use_symlinks_when_possible,
            copy_models_instead_of_link=copy_models_instead_of_link,
            snapshot_download_fn=snapshot_download_fn,
        )
    else:
        print("[assets] Qwen source: skipped (LLM is not required for this run mode)")
    os.environ["GRACE_RETRIEVAL_MODEL_ID"] = retrieval_model_id
    os.environ["GRACE_RETRIEVAL_MODEL_DIR"] = str(retrieval_dir)
    os.environ["GRACE_AUTO_DOWNLOAD_RETRIEVAL_MODEL"] = "false"
    os.environ["GRACE_LOCAL_MODEL_ID"] = local_llm_model_id
    os.environ["GRACE_AUTO_DOWNLOAD_MODEL"] = "false"
    if local_dir is not None:
        os.environ["GRACE_LOCAL_MODEL_DIR"] = str(local_dir)
    else:
        os.environ.pop("GRACE_LOCAL_MODEL_DIR", None)
    return PreparedAssets(
        devign_path=devign_path,
        retrieval_model_dir=retrieval_dir,
        local_llm_model_dir=local_dir,
        devign_source=devign_status,
        retrieval_model_source=retrieval_status,
        local_llm_model_source=local_status,
        devign_usable_records=usable,
    )


# Backward-compatible wrappers used by older local callers.
def prepare_devign_input(devign_input: str | Path | None, repository_root: str | Path) -> Path:
    path, _, _ = resolve_or_download_devign(
        repository_root=repository_root,
        asset_root=Path(repository_root) / ".kaggle_assets",
        source=devign_input,
        urls=(),
        auto_download=False,
    )
    return path


def prepare_semantic_model(
    semantic_model_input: str | Path | None,
    repository_root: str | Path,
    *,
    auto_download: bool,
    model_id: str = SEMANTIC_MODEL_ID,
) -> Path:
    path, _ = resolve_or_download_hf_model(
        asset_root=Path(repository_root) / ".kaggle_assets",
        model_id=model_id,
        source_dir=semantic_model_input,
        auto_download=auto_download,
    )
    return path


def prepare_qwen_model(
    qwen_model_input: str | Path | None,
    repository_root: str | Path,
    *,
    auto_download: bool,
    required: bool,
    model_id: str = QWEN_MODEL_ID,
) -> Path | None:
    if not required:
        return None
    path, _ = resolve_or_download_hf_model(
        asset_root=Path(repository_root) / ".kaggle_assets",
        model_id=model_id,
        source_dir=qwen_model_input,
        auto_download=auto_download,
    )
    return path


semantic_model_ready = _valid_model
llm_model_ready = _valid_model


__all__ = [
    "PreparedAssets",
    "QWEN_MODEL_ID",
    "SEMANTIC_MODEL_ID",
    "llm_model_ready",
    "prepare_devign_input",
    "prepare_qwen_model",
    "prepare_required_assets",
    "prepare_semantic_model",
    "resolve_or_download_devign",
    "resolve_or_download_hf_model",
    "semantic_model_ready",
    "validate_devign_file",
    "validate_hf_model_directory",
]
