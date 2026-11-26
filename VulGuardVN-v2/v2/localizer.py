"""Suspicious-line localizer (ported from v1 `localizer.py`). Seed-independent (plan D5): the v1 branch_scale
that mixed in prefilter scores is fixed to 1, so the LLM input never depends on the prefilter run."""
from __future__ import annotations

import re

from .common import RISKY_APIS, extract_calls, tokenize_code

MEMORY_TERMS = {"malloc", "calloc", "realloc", "free", "memcpy", "memmove", "memset", "new", "delete"}
VALIDATION_TERMS = {"if", "while", "for", "assert", "check", "validate", "verify", "length", "size", "limit", "bound", "guard"}
POINTER_TOKENS = {"->", "*", "&", "[", "]"}
SINK_REGEX = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
INDEX_REGEX = re.compile(r"\b([A-Za-z_]\w*)\s*\[\s*([^\]]+)\]")
DEREF_REGEX = re.compile(r"\b([A-Za-z_]\w*)\s*->")


def line_score(line: str) -> tuple[float, list[str]]:
    tokens = tokenize_code(line)
    calls = extract_calls(line)
    reasons: list[str] = []
    score = 0.0
    risky = [call for call in calls if call.lower() in RISKY_APIS]
    if risky:
        score += 3.0 + 0.4 * len(risky)
        reasons.append("risky_api=" + ",".join(risky[:3]))
    if any(token in MEMORY_TERMS for token in tokens):
        score += 1.6
        reasons.append("memory_op")
    if any(token in POINTER_TOKENS for token in tokens) or "->" in line:
        score += 1.2
        reasons.append("pointer_or_array")
    if any(op in line for op in "+-*/%") and any(t in tokens for t in ("size", "len", "offset", "index")):
        score += 1.0
        reasons.append("index_arithmetic")
    if any(term in tokens for term in VALIDATION_TERMS):
        score += 0.8
        reasons.append("control_or_validation")
    if "=" in line and any(term in tokens for term in ("ptr", "buf", "dst", "src", "data")):
        score += 0.8
        reasons.append("buffer_assignment")
    if "return" in tokens and any(t in tokens for t in ("error", "fail", "null", "nullptr", "invalid")):
        score += 0.5
        reasons.append("error_path")
    return score, reasons


def suspicious_lines(code: str, top_k: int = 4) -> list[dict]:
    rows = []
    for number, text in enumerate(code.split("\n"), start=1):
        if not text.strip() or number == 1:
            continue
        score, reasons = line_score(text)
        if score > 0:
            rows.append({"line": number, "score": round(score, 3), "code": text.strip()[:200], "reasons": reasons})
    rows.sort(key=lambda row: (-row["score"], row["line"]))
    return rows[:top_k]


def heuristic_sinks(code: str, sink_apis: set[str], limit: int = 30) -> list[dict]:
    """Fallback candidate sinks when Joern failed on a function (regex based, same schema as the CPG sinks)."""
    out: list[dict] = []
    for number, text in enumerate(code.split("\n"), start=1):
        for name in SINK_REGEX.findall(text):
            if name in sink_apis:
                out.append({"line": number, "kind": "api", "name": name, "code": text.strip()[:200], "vars": []})
        for base, index in INDEX_REGEX.findall(text):
            out.append({"line": number, "kind": "index", "name": "[]", "code": f"{base}[{index}]"[:200], "vars": [base]})
        for base in DEREF_REGEX.findall(text):
            out.append({"line": number, "kind": "deref", "name": "->", "code": text.strip()[:200], "vars": [base]})
    priority = {"api": 0, "size_arith": 1, "index": 2, "deref": 3}
    seen, unique = set(), []
    for sink in sorted(out, key=lambda s: (priority[s["kind"]], s["line"])):
        key = (sink["line"], sink["kind"], sink["name"])
        if key not in seen:
            seen.add(key)
            unique.append(sink)
    return sorted(unique[:limit], key=lambda s: s["line"])
