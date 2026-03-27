from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


BASELINE_DIR = Path(__file__).resolve().parent
GRACE_JAVA_DIR = BASELINE_DIR.parent
REPO_ROOT_DIR = GRACE_JAVA_DIR.parent
WORK_DIR = GRACE_JAVA_DIR / "work"
DEFAULT_BENCHMARK_CANDIDATES = [
    GRACE_JAVA_DIR / "data" / "benchmark.jsonl",
    BASELINE_DIR / "data" / "benchmark.jsonl",
    REPO_ROOT_DIR / "artifacts" / "dataset" / "function_dataset.jsonl",
    REPO_ROOT_DIR / "artifacts" / "dataset" / "train.jsonl",
    REPO_ROOT_DIR / "dataset" / "function.json",
    REPO_ROOT_DIR / "devign_test_processed.json",
]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_json_or_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".jsonl":
        return load_jsonl(p)

    if suffix == ".json":
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
        if isinstance(data, dict):
            for key in ("data", "rows", "samples", "items"):
                maybe_rows = data.get(key)
                if isinstance(maybe_rows, list):
                    return [x for x in maybe_rows if isinstance(x, dict)]
            return [data]
        raise ValueError(f"Unsupported JSON structure in {p}")

    raise ValueError(f"Unsupported input format: {p}. Use .jsonl or .json")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_first_existing(paths: Iterable[str | Path]) -> Path | None:
    for p in paths:
        candidate = Path(p)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def parse_binary_label(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) != 0 else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "vulnerable", "vul", "yes", "y"}:
        return 1
    if text in {"0", "false", "non-vulnerable", "non_vulnerable", "safe", "no", "n"}:
        return 0
    return 0


def normalize_row_for_baseline(row: dict[str, Any], idx: int) -> dict[str, Any]:
    sample_id = row.get("sample_id") or row.get("id") or f"sample_{idx}"
    project = row.get("project") or row.get("project_slug") or row.get("repo") or "unknown_project"
    code = row.get("code") or row.get("func") or row.get("method_code") or row.get("snippet") or ""

    label_source = row.get("label", row.get("target", row.get("is_vulnerable", 0)))
    normalized = dict(row)
    normalized["sample_id"] = str(sample_id)
    normalized["project"] = str(project)
    normalized["code"] = str(code)
    normalized["label"] = parse_binary_label(label_source)

    if "cwe" not in normalized:
        normalized["cwe"] = row.get("cwe_id") or row.get("cwe")
    return normalized


def slugify(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        text = "sample"
    return text[:max_len]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def wrap_java_method(code: str, class_name: str) -> str:
    """Wrap a Java method snippet into a compilable class for Joern/javalang.

    If the snippet already looks like a compilation unit, it is returned unchanged.
    """
    code = normalize_whitespace(code)
    lowered = code.lower()
    if any(tok in lowered for tok in [" class ", " interface ", " enum ", " record ", "package "]):
        return code
    return f"public class {class_name} {{\n{code}\n}}\n"


def truncate_text(text: str, max_chars: int) -> str:
    text = text or ""
    return text[:max_chars]
