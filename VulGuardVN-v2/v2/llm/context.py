"""Builds the per-function context shown to the LLM (plan §6.4): numbered code (or windows for long
functions), candidate sinks + slices from Joern (regex fallback if Joern failed), suspicious lines (localizer)."""
from __future__ import annotations

from ..common import number_code
from ..localizer import heuristic_sinks, suspicious_lines

KIND_PRIORITY = {"api": 0, "size_arith": 1, "index": 2, "deref": 3}


def build_context(record_id: str, code: str, facts: dict | None, prompt_cfg: dict, sink_apis: set[str]) -> dict:
    lines = code.split("\n")
    joern_ok = bool(facts and facts.get("ok"))
    if joern_ok:
        sinks = [dict(s, slice=facts["slices"][i] if i < len(facts.get("slices", [])) else [])
                 for i, s in enumerate(facts.get("sinks", []))]
    else:
        sinks = [dict(s, slice=list(range(max(1, s["line"] - 3), min(len(lines), s["line"] + 3) + 1)))
                 for s in heuristic_sinks(code, sink_apis)]
    ranked = sorted(sinks, key=lambda s: (KIND_PRIORITY.get(s["kind"], 9), s["line"]))
    shown = sorted(ranked[: int(prompt_cfg["max_sinks_shown"])], key=lambda s: s["line"])
    slices = []
    for sink in ranked[: int(prompt_cfg["max_slices_shown"])]:
        slice_lines = sorted(set(sink.get("slice") or []))
        if len(slice_lines) > int(prompt_cfg["max_slice_lines"]):
            # keep the lines closest to the sink
            slice_lines = sorted(sorted(slice_lines, key=lambda n: abs(n - sink["line"]))[: int(prompt_cfg["max_slice_lines"])])
        slices.append({"sink_line": sink["line"], "sink": sink["name"], "kind": sink["kind"],
                       "text": number_code(code, slice_lines)})
    suspicious = suspicious_lines(code, int(prompt_cfg["max_suspicious_lines"]))
    truncated = len(lines) > int(prompt_cfg["max_code_lines"]) or len(code) > int(prompt_cfg["max_code_chars"])
    if truncated:
        keep = set(range(1, 6))
        for sink in shown:
            keep.update(range(sink["line"] - 4, sink["line"] + 5))
        for row in suspicious:
            keep.update(range(row["line"] - 2, row["line"] + 3))
        code_view = number_code(code, keep)
    else:
        code_view = number_code(code)
    return {"id": record_id, "code_view": code_view, "truncated": truncated, "n_lines": len(lines),
            "sinks": shown, "all_sinks": sinks, "slices": slices, "suspicious": suspicious, "joern_ok": joern_ok}


def slice_text_for_line(context: dict, code: str, line: int | None, max_lines: int = 30) -> str:
    """Slice around the hypothesis sink line: the CPG slice of the nearest candidate sink, else a window."""
    sinks = context.get("all_sinks") or []
    if line and sinks:
        nearest = min(sinks, key=lambda s: abs(s["line"] - line))
        if abs(nearest["line"] - line) <= 2 and nearest.get("slice"):
            lines = sorted(set(nearest["slice"]) | {line})
            if len(lines) > max_lines:
                lines = sorted(sorted(lines, key=lambda n: abs(n - line))[:max_lines])
            return number_code(code, lines)
    if line:
        return number_code(code, range(max(1, line - 6), line + 7))
    return number_code(code, range(1, max_lines + 1))
