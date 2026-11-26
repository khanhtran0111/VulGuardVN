"""Paths, configuration loading, config hashing and run logging for VulGuard v2."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import platform
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent              # VulGuardVN-v2/v2
PROJECT_DIR = PKG_DIR.parent                           # VulGuardVN-v2
CONFIG_DIR = PKG_DIR / "configs"
PROMPT_DIR = PKG_DIR / "llm" / "prompts"
SCHEMA_DIR = PKG_DIR / "llm" / "schemas"
CPG_SCRIPT_DIR = PKG_DIR / "cpg"

load_dotenv(PROJECT_DIR / ".env")


def artifacts_dir() -> Path:
    """Root of all generated outputs. Override with VG2_ARTIFACTS (e.g. for smoke tests)."""
    value = os.getenv("VG2_ARTIFACTS")
    path = Path(value) if value else PKG_DIR / "artifacts"
    if not path.is_absolute():
        path = PROJECT_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataset_dir(dataset: str, *parts: str) -> Path:
    path = artifacts_dir() / dataset
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_DIR / path).resolve()


def load_config(name: str) -> dict[str, Any]:
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if name == "llm":
        # experiment overrides (logged in run_log.jsonl through the config hash), e.g. a reasoning-effort study run
        # in a separate VG2_ARTIFACTS root so it never mixes with the main results
        if os.getenv("VG2_LLM_REASONING"):
            cfg["reasoning"] = {**(cfg.get("reasoning") or {}), "effort": os.environ["VG2_LLM_REASONING"]}
        if os.getenv("VG2_LLM_CONCURRENCY"):
            cfg["concurrency"] = int(os.environ["VG2_LLM_CONCURRENCY"])
    return cfg


def save_config(name: str, payload: dict[str, Any]) -> None:
    path = CONFIG_DIR / f"{name}.yaml"
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True, width=120)


def config_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def all_configs() -> dict[str, Any]:
    return {path.stem: load_config(path.stem) for path in sorted(CONFIG_DIR.glob("*.yaml"))}


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_run_header(script: str, args: Any = None, configs: list[str] | None = None) -> str:
    """Log the config hash at the start of every run (plan §4 convention, §11 checklist)."""
    names = configs or [path.stem for path in sorted(CONFIG_DIR.glob("*.yaml"))]
    payload = {name: load_config(name) for name in names}
    digest = config_hash(payload)
    record = {
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "script": script,
        "config_hash": digest,
        "configs": names,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "python": platform.python_version(),
        "artifacts": str(artifacts_dir()),
    }
    with (artifacts_dir() / "run_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    get_logger(script).info("config_hash=%s configs=%s artifacts=%s", digest, ",".join(names), artifacts_dir())
    return digest


def parse_datasets(value: str | None) -> list[str]:
    configured = list(load_config("datasets")["datasets"].keys())
    if not value or value == "all":
        return configured
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in configured]
    if unknown:
        raise SystemExit(f"Unknown dataset(s): {unknown}. Configured: {configured}")
    return names
