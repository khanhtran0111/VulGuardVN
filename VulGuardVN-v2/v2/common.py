"""Code utilities ported from v1 (`VulGuardVN-v1/baseline/baseline2/common.py`) plus v2 helpers."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

C_KEYWORDS = {
    "auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum",
    "extern", "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return",
    "short", "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void",
    "volatile", "while", "class", "namespace", "new", "delete", "public", "private", "protected",
    "template", "typename", "try", "catch", "throw", "using", "virtual", "bool", "true", "false",
    "nullptr", "null",
}
CONTROL_KEYWORDS = ["if", "else", "switch", "case", "for", "while", "do", "goto", "return", "break", "continue"]
RISKY_APIS = {
    "strcpy", "strncpy", "strcat", "strncat", "sprintf", "snprintf", "vsprintf", "scanf", "sscanf",
    "fscanf", "gets", "memcpy", "memmove", "memset", "malloc", "calloc", "realloc", "free", "new",
    "delete", "read", "write", "recv", "send", "open", "close", "fopen", "fclose", "strtok", "system",
    "exec", "popen",
}

STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
CHAR_PATTERN = re.compile(r"'(?:\\.|[^'\\])*'", re.DOTALL)
NUMBER_PATTERN = re.compile(r"\b(?:0x[0-9a-fA-F]+|\d+\.\d+|\d+)\b")
TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|==|!=|<=|>=|->|\+\+|--|&&|\|\||[{}\[\]();,.*&|^~!<>%/\-+=?:]")
SIGNATURE_PATTERN = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)\s*\{", re.DOTALL)
CAMEL_PATTERN = re.compile(r"([a-z0-9])([A-Z])")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
LINE_COMMENT = re.compile(r"//[^\n]*")
WHITESPACE = re.compile(r"\s+")
CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_code(code: str) -> str:
    """v1 normalisation. This exact text is shown to the LLM and written to Joern input files."""
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def dedup_key(code: str) -> str:
    """Hash used for de-duplication: comments and all whitespace removed (plan P1 step 3)."""
    text = BLOCK_COMMENT.sub(" ", code or "")
    text = LINE_COMMENT.sub(" ", text)
    text = WHITESPACE.sub("", text)
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _sanitize_literals(code: str) -> str:
    text = STRING_PATTERN.sub(" STR_LIT ", code)
    text = CHAR_PATTERN.sub(" CHAR_LIT ", text)
    return NUMBER_PATTERN.sub(" NUM_LIT ", text)


def _split_identifier(token: str) -> list[str]:
    token = CAMEL_PATTERN.sub(r"\1_\2", token).replace("__", "_").strip("_")
    parts = [part.lower() for part in token.split("_") if part]
    return parts or [token.lower()]


def tokenize_code(code: str) -> list[str]:
    text = _sanitize_literals(normalize_code(code))
    results: list[str] = []
    for token in TOKEN_PATTERN.findall(text):
        if token in {"STR_LIT", "CHAR_LIT"}:
            results.append("str_lit")
        elif token == "NUM_LIT":
            results.append("num_lit")
        elif token[0].isalpha() or token[0] == "_":
            lowered = token.lower()
            if lowered in C_KEYWORDS:
                results.append(lowered)
            else:
                results.extend(_split_identifier(token))
        else:
            results.append(token)
    return results


def build_skeleton(code: str) -> str:
    text = _sanitize_literals(normalize_code(code))
    tokens = TOKEN_PATTERN.findall(text)
    skeleton: list[str] = []
    for index, token in enumerate(tokens):
        lowered = token.lower()
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        if token in {"STR_LIT", "CHAR_LIT"}:
            skeleton.append("lit")
        elif token == "NUM_LIT":
            skeleton.append("num")
        elif token[0].isalpha() or token[0] == "_":
            if lowered in C_KEYWORDS:
                skeleton.append(lowered)
            elif next_token == "(":
                skeleton.append("call")
            else:
                skeleton.append("id")
        else:
            skeleton.append(token)
    return " ".join(skeleton)


def extract_function_name(code: str) -> str:
    match = SIGNATURE_PATTERN.search(normalize_code(code)[:1200])
    if not match:
        return "unknown"
    name = match.group(1)
    return name if name.lower() not in CONTROL_KEYWORDS else "unknown"


def estimate_parameter_count(code: str) -> int:
    match = SIGNATURE_PATTERN.search(normalize_code(code)[:1200])
    if not match:
        return 0
    params = match.group(2).strip()
    if not params or params == "void":
        return 0
    return len([part for part in params.split(",") if part.strip()])


def extract_calls(code: str) -> list[str]:
    seen: set[str] = set()
    results: list[str] = []
    for call in CALL_PATTERN.findall(code or ""):
        lowered = call.lower()
        if lowered in C_KEYWORDS or lowered in CONTROL_KEYWORDS or lowered in seen:
            continue
        seen.add(lowered)
        results.append(call)
    return results


def numbered_lines(code: str) -> list[str]:
    return code.split("\n")


def number_code(code: str, line_numbers: Iterable[int] | None = None) -> str:
    lines = code.split("\n")
    if line_numbers is None:
        return "\n".join(f"{index:>4}: {line}" for index, line in enumerate(lines, start=1))
    wanted = sorted({n for n in line_numbers if 1 <= n <= len(lines)})
    out: list[str] = []
    previous = 0
    for number in wanted:
        if previous and number > previous + 1:
            out.append("     ...")
        out.append(f"{number:>4}: {lines[number - 1]}")
        previous = number
    if previous and previous < len(lines):
        out.append("     ...")
    return "\n".join(out)


def truncate_text(text: str, limit: int) -> str:
    value = text or ""
    return value if len(value) <= limit else value[: limit - 3].rstrip() + "..."


# ---------------------------------------------------------------- interruption-safe writes
# Every "this step is done" file is written to a temporary name and renamed at the end (os.replace is atomic),
# so an interruption (Ctrl+C, closed terminal, power loss) never leaves a half-written file that looks finished.
def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def atomic_save_npy(path: Path, array) -> None:
    import numpy as np

    tmp = _tmp(path)
    with tmp.open("wb") as handle:
        np.save(handle, array)
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays) -> None:
    import numpy as np

    tmp = _tmp(path)
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def atomic_to_parquet(frame, path: Path) -> None:
    tmp = _tmp(path)
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = _tmp(path)
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def loads_or_none(value):
    """JSON column of a parquet index -> object. Missing values come back as None *or* NaN (float) after pandas
    concatenation, e.g. for an LLM answer that was not valid JSON; both mean 'no parsed output'."""
    if isinstance(value, (str, bytes)) and value:
        return json.loads(value)
    return None


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path)) if path.exists() else []


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def dump_json(path: Path, payload) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_json(path: Path, default=None):
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    z = math.exp(value)
    return z / (1.0 + z)
