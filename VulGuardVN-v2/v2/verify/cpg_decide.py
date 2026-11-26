"""CPG verification (plan §6.5, C4), evaluated offline on the facts exported in P2.

V1  sink line holds a sink operation (call / deref / index) involving the named sink or `variable`
V2  data reaching the sink comes from outside the function (parameter / source via DDG, or a field / global)
V3  "missing guard": no condition that checks what `required_guard` asks for dominates/controls the sink
    "guard at line g": a condition at g dominates/controls the sink and references `variable` (or a sink
    identifier), or line g is a bounding assignment (MIN / mask / modulo / sizeof) before the sink
(v2.1 of the rules, tuned on the PrimeVul dev set only - see dev_report.json)

| mode    | situation                                      | y |
| neutral | verdict = vulnerable AND V1 AND V2 AND V3       | 1 |
| neutral | otherwise (incl. no hypothesis)                 | 0 |
| refute  | guard_line given AND V3 confirms the guard      | 0 |
| refute  | otherwise (incl. no hypothesis)                 | 1 |
| both    | Joern error (or invalid LLM output)             | d_pf(x), fallback=1 |
"""
from __future__ import annotations

import re

IDENT = re.compile(r"[A-Za-z_]\w*")
NOT_VARS = {"sizeof", "int", "char", "unsigned", "long", "size_t", "const", "struct", "NULL", "null", "nullptr", "true",
            "false", "is", "not", "and", "or", "the", "valid", "within", "bounds", "of", "be", "must", "non", "zero"}
LINE_TOLERANCE = 1     # V3 guard lines
SINK_TOLERANCE = 2     # V1/V2: multi-line statements, Joern reports the first line of a call
GENERIC_SINKS = ("index", "array", "subscript", "deref", "pointer", "[]", "->", "*", "write", "access", "assign")
RELATIONAL = re.compile(r"<=|>=|<|>")
BOUNDING = re.compile(r"\b(?:min|MIN|max|MAX|clamp|CLAMP|sizeof|ARRAY_SIZE)\b|%|&(?!&)|\?|>>")


def variable_ids(variable: str | None) -> list[str]:
    return [token for token in IDENT.findall(variable or "") if token not in NOT_VARS]


def _lines(facts: dict, line: int, tolerance: int = LINE_TOLERANCE) -> list[dict]:
    table = facts.get("lines") or {}
    return [table[str(n)] for n in range(line - tolerance, line + tolerance + 1) if str(n) in table]


def _operators(code: str) -> str:
    return code.replace("->", " ").replace("<<", " ").replace(">>", " ")


def check_v1(facts: dict, line: int, sink: str | None, variable: str | None) -> bool:
    """The sink line holds a sink operation (call / deref / index) that involves the hypothesis: the named call
    (fuzzy: `view->resize` matches `resize`, `memcpy_s` matches `memcpy`), or an access on the variable."""
    ids = set(variable_ids(variable))
    sink_text = (sink or "").lower()
    sink_ids = {s.lower() for s in IDENT.findall(sink or "")}
    generic = any(word in sink_text for word in GENERIC_SINKS)
    for lf in _lines(facts, line, SINK_TOLERANCE):
        calls = [c.lower() for c in lf.get("calls", [])]
        accesses = set(lf.get("deref", [])) | set(lf.get("index", []))
        if any(c in sink_ids or (len(c) > 3 and any(c in s or s in c for s in sink_ids if len(s) > 3)) for c in calls):
            return True
        if ids & accesses:
            return True
        if (calls or accesses) and ids & (set(lf.get("ids", [])) | set(lf.get("tainted", []))):
            return True
        if generic and accesses:
            return True
    return False


def check_v2(facts: dict, line: int, variable: str | None) -> bool:
    """Some data reaching the sink comes from outside the function: a parameter / source call (DDG), or a field /
    global access (`->`, `.`) that intraprocedural DDG cannot follow. Fails only for purely local, untainted sinks."""
    ids = set(variable_ids(variable))
    for lf in _lines(facts, line, SINK_TOLERANCE):
        tainted = set(lf.get("tainted", []))
        if tainted and (not ids or ids & tainted or tainted & set(lf.get("ids", []))):
            return True
        if lf.get("deref"):
            return True
    return "->" in (variable or "") or "." in (variable or "")


def _guards(facts: dict, line: int) -> list[dict]:
    conds = facts.get("conds") or []
    out, seen = [], set()
    for lf in _lines(facts, line):
        for index in lf.get("dom", []):
            if index not in seen and index < len(conds):
                seen.add(index)
                out.append(conds[index])
    return out


def _references(cond: dict, variable: str | None) -> bool:
    ids = variable_ids(variable)
    if not ids:
        return False
    compact_var = re.sub(r"\s+", "", variable or "")
    compact_code = re.sub(r"\s+", "", cond.get("code", ""))
    return bool(set(ids) & set(cond.get("ids", []))) or (len(compact_var) > 1 and compact_var in compact_code)


def guard_kind(required_guard: str | None) -> str:
    text = _operators(required_guard or "")
    if RELATIONAL.search(text):
        return "bound"
    if re.search(r"\bNULL\b|\bnullptr\b|!=\s*0\b|==\s*0\b|!", text) or not re.search(r"[=<>+\-*/%]", text):
        return "null"
    return "other"


def is_relevant_guard(cond: dict, variable: str | None, required_guard: str | None) -> bool:
    """A condition counts as the required guard only if it checks the same thing: it references the variable,
    covers >= 2/3 of the identifiers of `required_guard` (so `p != NULL` is not a guard for `p->len > 1`), and has
    the right kind of test (bound -> relational comparison; null -> NULL / truthiness check)."""
    if not _references(cond, variable):
        return False
    code = cond.get("code", "")
    required = {i for i in variable_ids(required_guard)} if required_guard else set()
    if required:
        tokens = set(IDENT.findall(code))
        if len(required & tokens) / len(required) < 2 / 3:
            return False
    kind = guard_kind(required_guard)
    if kind == "bound":
        return bool(RELATIONAL.search(_operators(code)))
    return True


def check_v3_missing(facts: dict, line: int, variable: str | None, required_guard: str | None = None) -> bool:
    return not any(is_relevant_guard(cond, variable, required_guard) for cond in _guards(facts, line))


def _sanitizing_assignment(code_lines: list[str] | None, guard_line: int, sink_line: int, variable: str | None) -> bool:
    """`len = MIN(len, sizeof(buf));`, `idx &= MASK;`, `n = n % SIZE;` before the sink also make it safe."""
    if not code_lines or not (1 <= guard_line <= len(code_lines)) or guard_line > sink_line:
        return False
    text = code_lines[guard_line - 1]
    ids = set(variable_ids(variable))
    assigned = re.search(r"(?<![=!<>])=(?!=)|[&%|]=", text)
    return bool(ids & set(IDENT.findall(text)) and assigned and BOUNDING.search(text))


def check_v3_guard(facts: dict, line: int, variable: str | None, guard_line: int | None,
                   code_lines: list[str] | None = None) -> bool:
    """Refute mode: the guard named by the LLM exists - a condition at guard_line (+-1) that dominates / controls
    the sink and references the variable (or an identifier used on the sink line), or a bounding assignment."""
    if guard_line is None:
        return False
    sink_ids = set()
    for lf in _lines(facts, line, SINK_TOLERANCE):
        sink_ids |= set(lf.get("deref", [])) | set(lf.get("index", [])) | set(lf.get("tainted", []))
    for cond in _guards(facts, line):
        if abs(int(cond.get("line", -10)) - int(guard_line)) <= LINE_TOLERANCE:
            if _references(cond, variable) or (sink_ids & set(cond.get("ids", []))):
                return True
    return _sanitizing_assignment(code_lines, int(guard_line), line, variable)


def guard_candidates(facts: dict | None, line: int | None) -> list[int]:
    if not facts or not facts.get("ok") or not line:
        return []
    return sorted({int(c.get("line", 0)) for c in _guards(facts, int(line))})


def decide_without_hypothesis(mode: str, pass2: dict | None, facts: dict | None) -> dict:
    """A7 (no Pass 1): Pass 2 was asked without a hypothesis, so the sink line / variable come from its own
    evidence: the first evidence line that holds the named sink (or a deref/index); variables = identifiers used
    there as pointer/index or reachable from a parameter."""
    out = {"mode": mode, "d": None, "fallback": 0, "reason": "", "v1": None, "v2": None, "v3": None,
           "verdict": None, "guard_line": None, "d_raw": None, "has_hyp": False}
    if not pass2 or pass2.get("parsed") is None:
        out.update(fallback=1, reason="pass2_invalid")
        return out
    parsed = pass2["parsed"]
    out["verdict"], out["guard_line"] = parsed.get("verdict"), parsed.get("guard_line")
    out["d_raw"] = 1 if out["verdict"] == "vulnerable" else 0
    if not facts or not facts.get("ok"):
        out.update(fallback=1, reason="joern_error")
        return out
    sink = (parsed.get("sink") or "").split("(")[0].strip()
    lines = [int(n) for n in parsed.get("evidence_lines") or [] if isinstance(n, int)]
    table = facts.get("lines") or {}
    line, variables = None, []
    for n in lines:
        lf = table.get(str(n)) or {}
        if (sink and sink in lf.get("calls", [])) or lf.get("deref") or lf.get("index"):
            line = n
            variables = sorted(set(lf.get("deref", [])) | set(lf.get("index", [])) | set(lf.get("tainted", [])))
            break
    variable = " ".join(variables)
    if mode == "neutral":
        v1 = line is not None
        v2 = bool(line) and check_v2(facts, line, variable or None)
        v3 = bool(line) and check_v3_missing(facts, line, variable or None)
        out.update(v1=v1, v2=v2, v3=v3, d=int(out["verdict"] == "vulnerable" and v1 and v2 and v3))
        out["reason"] = "verified" if out["d"] else ("llm_not_vulnerable" if out["verdict"] != "vulnerable" else "cpg_rejected")
    else:
        targets = [line] if line else lines
        v3 = any(check_v3_guard(facts, n, variable or " ".join(variable_ids(" ".join(
            (table.get(str(n)) or {}).get("ids", [])))), out["guard_line"]) for n in targets if n)
        out.update(v3=v3, d=0 if v3 else 1)
        out["reason"] = "guard_confirmed" if v3 else ("no_guard_claimed" if out["guard_line"] is None else "guard_rejected")
    return out


def decide(mode: str, pass1: dict | None, pass2: dict | None, facts: dict | None, code: str | None = None,
           refute_no_hyp: int = 1) -> dict:
    """Decision of the LLM branch for one function and one mode. d=None means: use d_pf(x) (fallback)."""
    out = {"mode": mode, "d": None, "fallback": 0, "reason": "", "v1": None, "v2": None, "v3": None,
           "verdict": None, "guard_line": None, "d_raw": None, "has_hyp": False}
    if pass2 and pass2.get("parsed"):
        out["verdict"] = pass2["parsed"].get("verdict")
        out["guard_line"] = pass2["parsed"].get("guard_line")
        # A10 (no CPG verification): the verdict of Pass 2 alone
        out["d_raw"] = 1 if out["verdict"] == "vulnerable" else 0
    if not facts or not facts.get("ok"):
        out.update(fallback=1, reason="joern_error")
        return out
    if not pass1 or pass1.get("parsed") is None:
        out.update(fallback=1, reason="pass1_invalid")
        return out
    hyps = pass1["parsed"].get("hypotheses") or []
    if not hyps:
        value = 0 if mode == "neutral" else int(refute_no_hyp)      # decision.yaml: refute_without_hypothesis
        out.update(d=value, reason="no_hypothesis", d_raw=value)
        return out
    out["has_hyp"] = True
    if not pass2 or pass2.get("parsed") is None:
        out.update(fallback=1, reason="pass2_invalid")
        return out
    hyp = hyps[0]
    line, variable = int(hyp.get("sink_line") or 0), hyp.get("variable")
    sink = pass2["parsed"].get("sink")
    if mode == "neutral":
        v1, v2 = check_v1(facts, line, sink, variable), check_v2(facts, line, variable)
        v3 = check_v3_missing(facts, line, variable, hyp.get("required_guard"))
        out.update(v1=v1, v2=v2, v3=v3)
        out["d"] = int(out["verdict"] == "vulnerable" and v1 and v2 and v3)
        out["reason"] = "verified" if out["d"] else ("llm_not_vulnerable" if out["verdict"] != "vulnerable" else "cpg_rejected")
    else:
        v3 = check_v3_guard(facts, line, variable, out["guard_line"], code.split("\n") if code else None)
        out.update(v3=v3)
        out["d"] = 0 if v3 else 1
        out["reason"] = "guard_confirmed" if v3 else ("no_guard_claimed" if out["guard_line"] is None else "guard_rejected")
    return out
