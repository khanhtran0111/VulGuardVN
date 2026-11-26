from collections import Counter

from common import RISKY_APIS, extract_calls, normalize_code, tokenize_code, truncate_text


MEMORY_TERMS = {
    "malloc",
    "calloc",
    "realloc",
    "free",
    "memcpy",
    "memmove",
    "memset",
    "new",
    "delete",
}
VALIDATION_TERMS = {
    "if",
    "while",
    "for",
    "assert",
    "check",
    "validate",
    "verify",
    "length",
    "size",
    "limit",
    "bound",
    "guard",
}
POINTER_TOKENS = {"->", "*", "&", "[", "]"}


def _line_tokens(line: str) -> list[str]:
    return tokenize_code(normalize_code(line))


def _line_score(line: str, *, branch_scale: float) -> tuple[float, list[str]]:
    text = normalize_code(line)
    tokens = _line_tokens(text)
    calls = extract_calls(text)
    reasons: list[str] = []
    score = 0.0

    risky_calls = [call for call in calls if call.lower() in RISKY_APIS]
    if risky_calls:
        score += 3.0 + 0.4 * len(risky_calls)
        reasons.append(f"risky_api={','.join(risky_calls[:3])}")
    memory_terms = [token for token in tokens if token in MEMORY_TERMS]
    if memory_terms:
        score += 1.6
        reasons.append("memory_op")
    if any(token in POINTER_TOKENS for token in tokens) or "->" in text:
        score += 1.2
        reasons.append("pointer_or_array")
    if any(op in text for op in ["+", "-", "*", "/", "%"]) and any(token in tokens for token in ["size", "len", "offset", "index"]):
        score += 1.0
        reasons.append("index_arithmetic")
    if any(term in tokens for term in VALIDATION_TERMS):
        score += 0.8
        reasons.append("control_or_validation")
    if "=" in text and any(term in tokens for term in ["ptr", "buf", "dst", "src", "data"]):
        score += 0.8
        reasons.append("buffer_assignment")
    if "return" in tokens and any(token in tokens for token in ["error", "fail", "null", "nullptr", "invalid"]):
        score += 0.5
        reasons.append("error_path")

    if text.count("(") != text.count(")") or text.count("{") != text.count("}"):
        score += 0.4
        reasons.append("unbalanced_structure")
    return score * branch_scale, reasons


def locate_suspicious_slices(
    code: str,
    *,
    semantic_score: float,
    graph_score: float,
    fusion_score: float,
    risk_band: str,
    top_k: int | None = None,
    context_radius: int = 1,
) -> dict:
    text = normalize_code(code)
    raw_lines = text.splitlines()
    numbered_lines = [(index + 1, line.rstrip()) for index, line in enumerate(raw_lines) if line.strip()]
    if not numbered_lines:
        return {
            "top_lines": [],
            "slices": [],
            "slices_text": "No suspicious slices could be extracted.",
            "token_highlights": [],
        }

    branch_scale = 0.35 + 0.3 * float(semantic_score) + 0.35 * float(graph_score) + 0.2 * float(fusion_score)
    scored_lines = []
    for line_number, line_text in numbered_lines:
        score, reasons = _line_score(line_text, branch_scale=branch_scale)
        if score <= 0:
            continue
        scored_lines.append(
            {
                "line_number": line_number,
                "score": float(round(score, 4)),
                "code": line_text,
                "reasons": reasons,
            }
        )

    if not scored_lines:
        scored_lines = [
            {
                "line_number": line_number,
                "score": float(round(0.1 * branch_scale, 4)),
                "code": line_text,
                "reasons": ["fallback_context"],
            }
            for line_number, line_text in numbered_lines[:3]
        ]

    scored_lines.sort(key=lambda item: (item["score"], -item["line_number"]), reverse=True)
    if top_k is None:
        top_k = 2 if risk_band == "inspect" else 4 if risk_band == "high" else 1
    top_lines = scored_lines[:top_k]

    selected_line_numbers = sorted({row["line_number"] for row in top_lines})
    slices: list[dict] = []
    occupied = set()
    for center in selected_line_numbers:
        start = max(1, center - context_radius)
        end = min(len(raw_lines), center + context_radius)
        if any(line in occupied for line in range(start, end + 1)):
            continue
        for line in range(start, end + 1):
            occupied.add(line)
        slice_lines = [f"{line_no:>4}: {raw_lines[line_no - 1]}" for line_no in range(start, end + 1)]
        slice_score = max((row["score"] for row in top_lines if start <= row["line_number"] <= end), default=0.0)
        slice_reasons = []
        for row in top_lines:
            if start <= row["line_number"] <= end:
                slice_reasons.extend(row["reasons"])
        slices.append(
            {
                "start_line": start,
                "end_line": end,
                "score": float(round(slice_score, 4)),
                "reasons": sorted(set(slice_reasons)),
                "text": "\n".join(slice_lines),
            }
        )

    token_counter = Counter()
    for row in top_lines:
        for token in _line_tokens(row["code"]):
            if token.isidentifier() and token not in {"if", "for", "while", "return"}:
                token_counter[token] += 1
    token_highlights = [token for token, _ in token_counter.most_common(8)]
    slices_text = "\n\n".join(
        [
            "\n".join(
                [
                    f"Slice score={item['score']:.4f} | lines {item['start_line']}-{item['end_line']} | reasons={','.join(item['reasons']) or 'n/a'}",
                    item["text"],
                ]
            )
            for item in slices
        ]
    )
    return {
        "top_lines": top_lines,
        "slices": slices,
        "slices_text": truncate_text(slices_text, 2200) if slices_text else "No suspicious slices.",
        "token_highlights": token_highlights,
    }


__all__ = ["locate_suspicious_slices"]
