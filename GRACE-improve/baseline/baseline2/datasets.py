import csv
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from common import DATA_DIR, get_record_code, normalize_code, stable_hash


DEVIGN_SOURCE = DATA_DIR / "function.json"
BIGVUL_SOURCE = DATA_DIR / "MSR_data_cleaned.csv"
BIGVUL_PARQUET_DIR = DATA_DIR / "bigvul_raw"
BIGVUL_PARQUET_FILES = {
    "train": BIGVUL_PARQUET_DIR / "train-00000-of-00001.parquet",
    "val": BIGVUL_PARQUET_DIR / "validation-00000-of-00001.parquet",
    "test": BIGVUL_PARQUET_DIR / "test-00000-of-00001.parquet",
}
REVEAL_SPLIT_FILES = {
    "train": DATA_DIR / "reveal" / "train.jsonl",
    "val": DATA_DIR / "reveal" / "val.jsonl",
    "test": DATA_DIR / "reveal" / "test.jsonl",
}
REVEAL_CANDIDATE_DIRS = [
    DATA_DIR / "reveal",
    DATA_DIR / "reveal_ready",
    DATA_DIR / "reveal_raw",
    DATA_DIR / "ReVeal",
    DATA_DIR / "Reveal",
]

PROJECT_FIELDS = ["project", "project_before", "repo", "repository"]
CWE_FIELDS = ["cwe_id", "cwe", "CWE ID", "cweID"]
CODE_FIELDS = ["code", "func", "functionSource", "source", "raw_code", "func_before", "before"]
LABEL_FIELDS = ["target", "label", "vul", "is_vul", "is_vulnerable"]


def list_available_datasets() -> list[str]:
    datasets = []
    if DEVIGN_SOURCE.exists():
        datasets.append("devign")
    if has_bigvul_source():
        datasets.append("bigvul")
    if discover_reveal_root():
        datasets.append("reveal")
    return datasets


def has_bigvul_source() -> bool:
    return BIGVUL_SOURCE.exists() or has_bigvul_parquet_source()


def has_bigvul_parquet_source() -> bool:
    return all(path.exists() for path in BIGVUL_PARQUET_FILES.values())


def discover_reveal_root() -> Path | None:
    if has_reveal_official_splits():
        return REVEAL_SPLIT_FILES["train"].parent
    for candidate in REVEAL_CANDIDATE_DIRS:
        if candidate.exists():
            return candidate
    return None


def has_reveal_official_splits() -> bool:
    return all(path.exists() for path in REVEAL_SPLIT_FILES.values())


def _choose_field(row: dict, names: list[str], default: str = "") -> str:
    for name in names:
        value = row.get(name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def _parse_label(value) -> int | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "vulnerable", "positive"}:
        return 1
    if text in {"0", "false", "no", "non-vulnerable", "benign", "negative"}:
        return 0
    return None


def _project_from_row(row: dict, default: str = "") -> str:
    return _choose_field(row, PROJECT_FIELDS, default=default)


def _cwe_from_row(row: dict) -> str:
    return _choose_field(row, CWE_FIELDS)


def _canonical_record(
    *,
    dataset: str,
    record_id: str,
    code: str,
    label: int,
    project: str = "",
    commit_id: str = "",
    cwe_id: str = "",
    source_path: str = "",
    split: str = "",
    extra: dict | None = None,
) -> dict:
    record = {
        "record_id": record_id,
        "dataset": dataset,
        "project": project,
        "label": int(label),
        "code": normalize_code(code),
        "commit_id": commit_id,
        "cwe_id": cwe_id,
        "source_path": source_path,
        "code_hash": stable_hash(normalize_code(code)),
    }
    if split:
        record["split"] = split
    if extra:
        record.update(extra)
    return record


def get_dataset_iterator(dataset_name: str):
    if dataset_name == "devign":
        return iter_devign_records
    if dataset_name == "bigvul":
        return iter_bigvul_records
    if dataset_name == "reveal":
        return iter_reveal_records
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def iter_devign_records() -> Iterable[dict]:
    data = json.loads(DEVIGN_SOURCE.read_text(encoding="utf-8"))
    for index, row in enumerate(data):
        code = normalize_code(str(row.get("func", "") or ""))
        label = _parse_label(row.get("target"))
        if not code or label is None:
            continue
        yield _canonical_record(
            dataset="devign",
            record_id=f"devign-{index}",
            code=code,
            label=label,
            project=str(row.get("project", "") or ""),
            commit_id=str(row.get("commit_id", "") or ""),
            source_path=str(DEVIGN_SOURCE.name),
            extra={"source_row": index},
        )


def iter_bigvul_records() -> Iterable[dict]:
    if has_bigvul_parquet_source():
        yield from iter_bigvul_parquet_records()
        return
    with BIGVUL_SOURCE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            code = normalize_code(str(row.get("func_before", "") or ""))
            label = _parse_label(row.get("vul"))
            if not code or label is None:
                continue
            yield _canonical_record(
                dataset="bigvul",
                record_id=f"bigvul-{index}",
                code=code,
                label=label,
                project=_project_from_row(row),
                commit_id=str(row.get("commit_id", "") or ""),
                cwe_id=_cwe_from_row(row),
                source_path=str(BIGVUL_SOURCE.name),
                extra={"source_row": index},
            )


def iter_bigvul_parquet_records() -> Iterable[dict]:
    if not has_bigvul_parquet_source():
        return
    row_offset = 0
    for split_name, path in BIGVUL_PARQUET_FILES.items():
        frame = pd.read_parquet(path)
        for index, row in enumerate(frame.to_dict(orient="records")):
            code = normalize_code(str(row.get("func_before", "") or ""))
            label = _parse_label(row.get("vul"))
            if not code or label is None:
                continue
            yield _canonical_record(
                dataset="bigvul",
                record_id=f"bigvul-{row_offset + index}",
                code=code,
                label=label,
                project=_project_from_row(row),
                commit_id=str(row.get("commit_id", "") or ""),
                cwe_id=_cwe_from_row(row),
                source_path=str(path.relative_to(DATA_DIR)),
                split=split_name,
                extra={"source_row": row_offset + index},
            )
        row_offset += len(frame)


def _iter_reveal_jsonl_split(split_name: str, path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            code = get_record_code(row)
            label = _parse_label(_choose_field(row, LABEL_FIELDS))
            if not code or label is None:
                continue
            project = _project_from_row(row, default=path.parent.name)
            record_id = str(row.get("record_id", "") or f"reveal-{split_name}-{index}")
            extra = {}
            for key in ["hash", "size", "source_format"]:
                if key in row and row[key] is not None:
                    extra[key] = row[key]
            yield _canonical_record(
                dataset="reveal",
                record_id=record_id,
                code=code,
                label=label,
                project=project,
                source_path=str(path.relative_to(DATA_DIR)),
                split=split_name,
                extra=extra,
            )


def iter_reveal_split_records(split_name: str) -> Iterable[dict]:
    path = REVEAL_SPLIT_FILES[split_name]
    if not path.exists():
        return
    yield from _iter_reveal_jsonl_split(split_name, path)


def _iter_json_file(path: Path, dataset_name: str) -> Iterable[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if isinstance(payload, dict):
        for key in ["data", "records", "items"]:
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        code = get_record_code(row)
        label = _parse_label(_choose_field(row, LABEL_FIELDS))
        if not code or label is None:
            continue
        split_hint = str(row.get("split", "") or "")
        yield _canonical_record(
            dataset=dataset_name,
            record_id=f"{dataset_name}-{path.stem}-{index}",
            code=code,
            label=label,
            project=_project_from_row(row, default=path.parent.name),
            commit_id=str(row.get("commit_id", "") or ""),
            cwe_id=_cwe_from_row(row),
            source_path=str(path.relative_to(DATA_DIR)),
            split=split_hint,
        )


def _iter_jsonl_file(path: Path, dataset_name: str) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            code = get_record_code(row)
            label = _parse_label(_choose_field(row, LABEL_FIELDS))
            if not code or label is None:
                continue
            split_hint = str(row.get("split", "") or "")
            record_id = str(row.get("record_id", "") or f"{dataset_name}-{path.stem}-{index}")
            extra = {}
            for key in ["hash", "size", "source_format"]:
                if key in row and row[key] is not None:
                    extra[key] = row[key]
            yield _canonical_record(
                dataset=dataset_name,
                record_id=record_id,
                code=code,
                label=label,
                project=_project_from_row(row, default=path.parent.name),
                commit_id=str(row.get("commit_id", "") or ""),
                cwe_id=_cwe_from_row(row),
                source_path=str(path.relative_to(DATA_DIR)),
                split=split_hint,
                extra=extra,
            )


def _iter_table_file(path: Path, dataset_name: str, delimiter: str) -> Iterable[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for index, row in enumerate(reader):
            code = get_record_code(row)
            label = _parse_label(_choose_field(row, LABEL_FIELDS))
            if not code or label is None:
                continue
            split_hint = str(row.get("split", "") or "")
            yield _canonical_record(
                dataset=dataset_name,
                record_id=f"{dataset_name}-{path.stem}-{index}",
                code=code,
                label=label,
                project=_project_from_row(row, default=path.parent.name),
                commit_id=str(row.get("commit_id", "") or ""),
                cwe_id=_cwe_from_row(row),
                source_path=str(path.relative_to(DATA_DIR)),
                split=split_hint,
            )


def _iter_parquet_file(path: Path, dataset_name: str) -> Iterable[dict]:
    frame = pd.read_parquet(path)
    rows = frame.to_dict(orient="records")
    split_hint = "train" if "train" in path.stem else "val" if "validation" in path.stem else "test" if "test" in path.stem else ""
    for index, row in enumerate(rows):
        code = get_record_code(row)
        label = _parse_label(_choose_field(row, LABEL_FIELDS))
        if not code or label is None:
            continue
        extra = {}
        for key in ["hash", "size"]:
            if key in row and row[key] is not None:
                extra[key] = row[key]
        yield _canonical_record(
            dataset=dataset_name,
            record_id=f"{dataset_name}-{path.stem}-{index}",
            code=code,
            label=label,
            project=_project_from_row(row, default=path.parent.name),
            source_path=str(path.relative_to(DATA_DIR)),
            split=split_hint,
            extra=extra,
        )


def _parse_label_from_path(path: Path) -> int | None:
    parts = [part.lower() for part in path.parts]
    for part in reversed(parts):
        if part in {"1", "vulnerable", "positive", "bad"}:
            return 1
        if part in {"0", "non-vulnerable", "benign", "negative", "good"}:
            return 0
    stem = path.stem.lower()
    if stem.endswith("_1") or stem.endswith("-1"):
        return 1
    if stem.endswith("_0") or stem.endswith("-0"):
        return 0
    return None


def _iter_source_files(root: Path, dataset_name: str) -> Iterable[dict]:
    allowed_suffixes = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    for path in root.rglob("*"):
        if path.suffix.lower() not in allowed_suffixes:
            continue
        label = _parse_label_from_path(path)
        if label is None:
            continue
        code = normalize_code(path.read_text(encoding="utf-8", errors="ignore"))
        if not code:
            continue
        yield _canonical_record(
            dataset=dataset_name,
            record_id=f"{dataset_name}-{stable_hash(str(path.relative_to(root)))}",
            code=code,
            label=label,
            project=path.parent.name,
            source_path=str(path.relative_to(DATA_DIR)),
        )


def iter_reveal_records() -> Iterable[dict]:
    if has_reveal_official_splits():
        for split_name in ["train", "val", "test"]:
            yield from iter_reveal_split_records(split_name)
        return
    root = discover_reveal_root()
    if root is None:
        return
    seen_ids = set()
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            iterator = _iter_json_file(path, "reveal")
        elif suffix == ".jsonl":
            iterator = _iter_jsonl_file(path, "reveal")
        elif suffix == ".csv":
            iterator = _iter_table_file(path, "reveal", ",")
        elif suffix == ".tsv":
            iterator = _iter_table_file(path, "reveal", "\t")
        elif suffix == ".parquet":
            iterator = _iter_parquet_file(path, "reveal")
        else:
            iterator = []
        for row in iterator:
            if row["record_id"] in seen_ids:
                continue
            seen_ids.add(row["record_id"])
            yield row
    for row in _iter_source_files(root, "reveal"):
        if row["record_id"] in seen_ids:
            continue
        seen_ids.add(row["record_id"])
        yield row
