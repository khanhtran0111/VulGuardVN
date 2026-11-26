"""Raw dataset loaders (plan P1 steps 1-2). Every loader yields canonical dicts:

id, dataset, split (official datasets only), code, label, project, cwe, cve, commit_id,
file_name, lang, pair_id, pair_role, in_main
"""
from __future__ import annotations

import ast
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

from ..common import normalize_code

CPP_EXTENSIONS = (".cc", ".cpp", ".cxx", ".c++", ".hpp", ".hh", ".hxx")


def _lang_from_filename(name: str | None, default: str = "c") -> str:
    lowered = (name or "").lower()
    return "cpp" if lowered.endswith(CPP_EXTENSIONS) else default


def _parse_cwe(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        items = [str(item) for item in value if item]
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "[]"}:
            return None
        if text.startswith("["):
            try:
                items = [str(item) for item in ast.literal_eval(text) if item]
            except Exception:
                items = [text.strip("[]'\" ")]
        else:
            items = [text]
    items = [item for item in items if item and item.upper().startswith("CWE")]
    return "|".join(items) if items else None


def _record(**kwargs) -> dict:
    base = {
        "id": None, "dataset": None, "split": None, "code": "", "label": 0, "project": None, "cwe": None,
        "cve": None, "commit_id": None, "file_name": None, "lang": "c", "pair_id": None, "pair_role": None,
        "in_main": True,
    }
    base.update(kwargs)
    return base


# ---------------------------------------------------------------- PrimeVul v0.1
def load_primevul(cfg: dict, data_root: Path) -> Iterable[dict]:
    """Official temporal split. Pairs come from *_paired.jsonl (consecutive vulnerable/patched lines)."""
    for split, rel in cfg["files"].items():
        pair_of_hash: dict[str, tuple[str, str]] = {}
        paired_rows: dict[str, dict] = {}
        paired_path = data_root / cfg.get("paired_files", {}).get(split, "__missing__")
        if paired_path.exists():
            rows = [json.loads(line) for line in paired_path.open(encoding="utf-8") if line.strip()]
            for index in range(0, len(rows) - 1, 2):
                vul, patched = rows[index], rows[index + 1]
                if int(vul["target"]) != 1 or int(patched["target"]) != 0:
                    continue
                pair_id = f"pv_pair_{split}_{index // 2:04d}"
                pair_of_hash[str(vul["func_hash"])] = (pair_id, "vulnerable")
                pair_of_hash[str(patched["func_hash"])] = (pair_id, "patched")
                paired_rows[str(vul["func_hash"])] = vul
                paired_rows[str(patched["func_hash"])] = patched
        seen_hashes: set[str] = set()
        with (data_root / rel).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                func_hash = str(row["func_hash"])
                seen_hashes.add(func_hash)
                pair_id, role = pair_of_hash.get(func_hash, (None, None))
                yield _primevul_record(row, split, pair_id, role, in_main=True)
        # Pair members missing from the main split file are kept as auxiliary records (pair metrics only).
        for func_hash, row in paired_rows.items():
            if func_hash not in seen_hashes:
                pair_id, role = pair_of_hash[func_hash]
                yield _primevul_record(row, split, pair_id, role, in_main=False)


def _primevul_record(row: dict, split: str, pair_id, role, *, in_main: bool) -> dict:
    suffix = "" if in_main else "_aux"
    return _record(
        id=f"primevul_{row['idx']}{suffix}", dataset="primevul", split=split,
        code=normalize_code(str(row.get("func") or "")), label=int(row["target"]),
        project=row.get("project"), cwe=_parse_cwe(row.get("cwe")), cve=row.get("cve"),
        commit_id=row.get("commit_id"), file_name=row.get("file_name"),
        lang=_lang_from_filename(row.get("file_name")), pair_id=pair_id, pair_role=role, in_main=in_main,
    )


# ---------------------------------------------------------------- Big-Vul (MSR_data_cleaned.csv)
def load_bigvul(cfg: dict, data_root: Path) -> Iterable[dict]:
    """Streams the 10.8 GB CSV. Vulnerable rows with a different func_after give a (vulnerable, patched) pair;
    the patched version is an auxiliary record (in_main=False) so Big-Vul keeps its usual size."""
    csv.field_size_limit(sys.maxsize)
    path = data_root / cfg["file"]
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        index_column = reader.fieldnames[0] if reader.fieldnames else ""
        for position, row in enumerate(reader):
            raw_index = (row.get(index_column) or "").strip() or str(position)
            code = normalize_code(row.get("func_before") or "")
            label_text = (row.get("vul") or "").strip()
            if not code or label_text not in {"0", "1"}:
                continue
            label = int(label_text)
            lang = "cpp" if (row.get("lang") or "").strip().lower() in {"c++", "cpp"} else "c"
            common = dict(
                dataset="bigvul", project=row.get("project"), cwe=_parse_cwe(row.get("CWE ID")),
                cve=row.get("CVE ID"), commit_id=row.get("commit_id"), file_name=row.get("file_name"),
                lang=lang,
            )
            pair_id = None
            after = normalize_code(row.get("func_after") or "")
            if label == 1 and after and after != code:
                pair_id = f"bv_pair_{raw_index}"
            yield _record(id=f"bigvul_{raw_index}", code=code, label=label,
                          pair_id=pair_id, pair_role="vulnerable" if pair_id else None, **common)
            if pair_id:
                yield _record(id=f"bigvul_{raw_index}_patched", code=after, label=0,
                              pair_id=pair_id, pair_role="patched", in_main=False, **common)


# ---------------------------------------------------------------- Devign (function.json)
def load_devign(cfg: dict, data_root: Path) -> Iterable[dict]:
    rows = json.loads((data_root / cfg["file"]).read_text(encoding="utf-8"))
    for index, row in enumerate(rows):
        yield _record(id=f"devign_{index:05d}", dataset="devign", code=normalize_code(row.get("func") or ""),
                      label=int(row["target"]), project=row.get("project"), commit_id=row.get("commit_id"))


# ---------------------------------------------------------------- ReVeal (HF parquet mirror, official split)
def load_reveal(cfg: dict, data_root: Path) -> Iterable[dict]:
    import pyarrow.parquet as pq

    for split, rel in cfg["files"].items():
        table = pq.read_table(data_root / rel).to_pylist()
        for index, row in enumerate(table):
            yield _record(id=f"reveal_{split}_{index:05d}", dataset="reveal", split=split,
                          code=normalize_code(row.get("functionSource") or ""), label=int(row["label"]),
                          project=row.get("project"))


LOADERS = {"primevul": load_primevul, "bigvul": load_bigvul, "devign": load_devign, "reveal": load_reveal}


def load_dataset(name: str, cfg: dict, data_root: Path) -> Iterable[dict]:
    return LOADERS[cfg["loader"]](cfg, data_root)
