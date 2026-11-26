"""Semantics-preserving transformations (plan P7 step 5). Both keep the number of lines unchanged so that
line-level tooling (Joern, the LLM prompts) still applies."""
from __future__ import annotations

import re

from ..common import C_KEYWORDS

IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
STRING_OR_CHAR = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


def _protected_spans(code: str) -> list[tuple[int, int]]:
    return [m.span() for m in STRING_OR_CHAR.finditer(code)]


def rename_variables(code: str) -> str:
    """Consistently renames local variables and parameters to v0, v1, ... Skips keywords, called names,
    struct fields (after . / ->), macros / constants (ALL_CAPS), type-like names (followed by an identifier or
    ending in _t) and anything inside string/char literals."""
    spans = _protected_spans(code)

    def protected(pos: int) -> bool:
        return any(a <= pos < b for a, b in spans)

    candidates: list[str] = []
    for match in IDENT.finditer(code):
        name, start, end = match.group(0), match.start(), match.end()
        if protected(start) or name in C_KEYWORDS or name.isupper() or name.endswith("_t"):
            continue
        before = code[max(0, start - 2):start]
        after = code[end:end + 40]
        if before.endswith(".") or before.endswith("->") or re.match(r"\s*\(", after):
            continue
        if re.match(r"\s*\*?\s*[A-Za-z_]\w*", after) and not re.match(r"\s*[A-Za-z_]\w*\s*\(", after):
            # `name other` -> name is a type (e.g. `foo bar;`); `name * other` is ambiguous, treated as a type as well
            if re.match(r"\s+[A-Za-z_]", after) or re.match(r"\s*\*\s*[A-Za-z_]\w*\s*[,;=)\[]", after):
                continue
        if name not in candidates:
            candidates.append(name)
    first_line_end = code.find("\n")
    signature = code[: first_line_end if first_line_end > 0 else len(code)]
    function_name = re.search(r"([A-Za-z_]\w*)\s*\(", signature)
    skip = {function_name.group(1)} if function_name else set()
    mapping = {name: f"v{i}" for i, name in enumerate(n for n in candidates if n not in skip)}

    def replace(match: re.Match) -> str:
        name = match.group(0)
        if protected(match.start()) or name not in mapping:
            return name
        before = code[max(0, match.start() - 2):match.start()]
        if before.endswith(".") or before.endswith("->"):
            return name
        return mapping[name]

    return IDENT.sub(replace, code)


def insert_dead_code(code: str) -> str:
    """Inserts an unreachable block right after the first opening brace (same line, line count unchanged)."""
    index = code.find("{")
    if index < 0:
        return code
    dead = " if (0) { int vg_dead_0 = 0; vg_dead_0++; } "
    return code[: index + 1] + dead + code[index + 1:]


TRANSFORMS = {"rename_variables": rename_variables, "dead_code": insert_dead_code}
