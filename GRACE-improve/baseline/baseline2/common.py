import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


BASELINE_DIR = Path(__file__).resolve().parent
BASELINE_ROOT_DIR = BASELINE_DIR.parent
ROOT_DIR = BASELINE_ROOT_DIR.parent
DATA_DIR = ROOT_DIR / "data"

# Shared baseline assets already prepared by previous runs live here.
SHARED_ARTIFACTS_DIR = BASELINE_ROOT_DIR / "artifacts"
SHARED_PROCESSED_DIR = SHARED_ARTIFACTS_DIR / "processed"
SHARED_SPLITS_DIR = SHARED_ARTIFACTS_DIR / "splits"
SHARED_MODELS_DIR = SHARED_ARTIFACTS_DIR / "models"
SHARED_GRAPH_DIR = SHARED_ARTIFACTS_DIR / "graphs"
SHARED_CACHE_DIR = SHARED_ARTIFACTS_DIR / "cache"
SHARED_METRICS_DIR = SHARED_ARTIFACTS_DIR / "metrics"
SHARED_PREDICTIONS_DIR = SHARED_ARTIFACTS_DIR / "predictions"
SHARED_RETRIEVAL_DIR = SHARED_ARTIFACTS_DIR / "retrieval"

# Baseline2 writes its own derived artifacts here to avoid clobbering baseline1.
ARTIFACTS_DIR = BASELINE_DIR / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
RETRIEVAL_DIR = ARTIFACTS_DIR / "retrieval"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
METRICS_DIR = ARTIFACTS_DIR / "metrics"
CACHE_DIR = ARTIFACTS_DIR / "cache"

# Keep aliases for code that expects dataset and graph assets.
PROCESSED_DIR = SHARED_PROCESSED_DIR
SPLITS_DIR = SHARED_SPLITS_DIR
GRAPH_DIR = SHARED_GRAPH_DIR
ENV_PATH = ROOT_DIR / ".env"

load_dotenv(ENV_PATH)
csv.field_size_limit(10**9)

C_KEYWORDS = {
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "inline",
    "int",
    "long",
    "register",
    "restrict",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    "class",
    "namespace",
    "new",
    "delete",
    "public",
    "private",
    "protected",
    "template",
    "typename",
    "try",
    "catch",
    "throw",
    "using",
    "virtual",
    "bool",
    "true",
    "false",
    "nullptr",
    "null",
}

CONTROL_KEYWORDS = ["if", "else", "switch", "case", "for", "while", "do", "goto", "return", "break", "continue"]
RISKY_APIS = {
    "strcpy",
    "strncpy",
    "strcat",
    "strncat",
    "sprintf",
    "snprintf",
    "vsprintf",
    "scanf",
    "sscanf",
    "fscanf",
    "gets",
    "memcpy",
    "memmove",
    "memset",
    "malloc",
    "calloc",
    "realloc",
    "free",
    "new",
    "delete",
    "read",
    "write",
    "recv",
    "send",
    "open",
    "close",
    "fopen",
    "fclose",
    "strtok",
    "system",
    "exec",
    "popen",
}

STRING_PATTERN = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
CHAR_PATTERN = re.compile(r"'(?:\\.|[^'\\])*'", re.DOTALL)
NUMBER_PATTERN = re.compile(r"\b(?:0x[0-9a-fA-F]+|\d+\.\d+|\d+)\b")
TOKEN_PATTERN = re.compile(r"[A-Za-z_]\w*|==|!=|<=|>=|->|\+\+|--|&&|\|\||[{}\[\]();,.*&|^~!<>%/\-+=?:]")
SIGNATURE_PATTERN = re.compile(r"([A-Za-z_]\w*)\s*\((.*?)\)\s*\{", re.DOTALL)
CAMEL_PATTERN = re.compile(r"([a-z0-9])([A-Z])")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def normalize_code(code: str) -> str:
    text = (code or "").replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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
    tokens = TOKEN_PATTERN.findall(text)
    results: list[str] = []
    for token in tokens:
        if token in {"STR_LIT", "CHAR_LIT"}:
            results.append("str_lit")
        elif token == "NUM_LIT":
            results.append("num_lit")
        elif re.match(r"[A-Za-z_]\w*$", token):
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
        elif re.match(r"[A-Za-z_]\w*$", token):
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
    calls = re.findall(r"\b([A-Za-z_]\w*)\s*\(", normalize_code(code))
    seen = set()
    results = []
    for call in calls:
        lowered = call.lower()
        if lowered in C_KEYWORDS or lowered in seen:
            continue
        seen.add(lowered)
        results.append(call)
    return results


def build_structure_summary(code: str) -> str:
    text = normalize_code(code)
    tokens = tokenize_code(text)
    token_set = set(tokens)
    controls = {name: tokens.count(name) for name in CONTROL_KEYWORDS if tokens.count(name)}
    risky = [call for call in extract_calls(text) if call.lower() in RISKY_APIS]
    memory_ops = [name for name in ["malloc", "calloc", "realloc", "free", "new", "delete", "memcpy", "memmove"] if name in token_set]
    lines = [
        f"function={extract_function_name(text)}",
        f"params={estimate_parameter_count(text)}",
        f"lines={len([line for line in text.splitlines() if line.strip()])}",
        f"calls={', '.join(extract_calls(text)[:8]) or 'none'}",
        f"control={json.dumps(controls, ensure_ascii=True) if controls else '{}'}",
        f"risky_apis={', '.join(risky) or 'none'}",
        f"memory_ops={', '.join(memory_ops) or 'none'}",
        f"pointer_ops={text.count('->') + text.count('*') + text.count('&')}",
        f"array_accesses={text.count('[')}",
    ]
    return "\n".join(lines)


def truncate_text(text: str, limit: int) -> str:
    value = normalize_code(text)
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def get_record_code(record: dict) -> str:
    for key in ["code", "func", "functionSource", "source", "raw_code", "func_before", "before"]:
        value = record.get(key)
        if value is None:
            continue
        text = normalize_code(str(value))
        if text:
            return text
    return ""


def read_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path, default: dict | None = None) -> dict:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_gemini_api_key() -> str:
    key = os.getenv("API_GEMINI") or os.getenv("GOOGLE_API_KEY") or os.getenv("API_KEY")
    if not key:
        raise RuntimeError(f"Missing Gemini API key. Set API_GEMINI in {ENV_PATH}.")
    return key.strip().strip('"').strip("'")


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)
