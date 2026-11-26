"""Heuristic graph backend (ported from v1 `graphs.py`) + 24 numeric features + AST-like sequence.

Used for EVERY split (plan D7) so train and test features are consistent.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from ..common import (CONTROL_KEYWORDS, RISKY_APIS, build_skeleton, estimate_parameter_count, extract_calls, extract_function_name,
                      tokenize_code)

MAX_HEURISTIC_LINES = 200       # v1 used 40; v2 keeps more of long functions
MAX_AST_TOKENS = 512
IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\b")
ASSIGNMENT_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*=(?!=)")
LINE_CALL_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
MEMORY_KEYWORDS = {"malloc", "calloc", "realloc", "free", "new", "delete", "memcpy", "memmove", "memset"}
CONTROL_TOKENS = set(CONTROL_KEYWORDS)

NUMERIC_FEATURE_NAMES = [
    "log_token_count", "unique_token_ratio", "log_line_count", "parameter_count", "log_call_count",
    "risky_call_count", "risky_call_ratio", "control_density", "pointer_density", "array_access_density",
    "numeric_literal_density", "memory_ops_count", "max_brace_depth", "log_graph_nodes", "log_graph_edges",
    "graph_avg_degree", "graph_call_ratio", "graph_control_ratio", "graph_expression_ratio", "graph_cfg_ratio",
    "graph_ast_ratio", "graph_reaches_ratio", "graph_max_out_degree", "log_ast_token_count",
]
assert len(NUMERIC_FEATURE_NAMES) == 24


def _node_type(line: str) -> str:
    lowered = line.lower()
    for keyword in CONTROL_KEYWORDS:
        if lowered.startswith(keyword) or f"{keyword} (" in lowered or f"{keyword}(" in lowered:
            return "CONTROL_STRUCTURE"
    if "=" in line:
        return "CALL" if "(" in line and ")" in line else "EXPRESSION"
    if "(" in line and ")" in line:
        return "CALL"
    if lowered.startswith("return"):
        return "RETURN"
    return "STATEMENT"


def _line_calls(line: str) -> list[str]:
    seen, out = set(), []
    for name in LINE_CALL_PATTERN.findall(line):
        lowered = name.lower()
        if lowered in CONTROL_TOKENS or lowered in seen:
            continue
        seen.add(lowered)
        out.append(name)
    return out


def heuristic_graph(code: str) -> dict:
    nodes: dict[int, str] = {}
    edges: list[tuple[int, int, str]] = []
    counter = [0]

    def add(node_type: str) -> int:
        counter[0] += 1
        nodes[counter[0]] = node_type
        return counter[0]

    method = add("METHOD")
    for _ in range(estimate_parameter_count(code)):
        edges.append((method, add("PARAM"), "IS_AST_PARENT"))
    statements: list[int] = []
    last_def: dict[str, int] = {}
    lines = [line.strip() for line in code.splitlines() if line.strip()][:MAX_HEURISTIC_LINES]
    for line in lines:
        statement = add(_node_type(line))
        statements.append(statement)
        edges.append((method, statement, "IS_AST_PARENT"))
        for _call in _line_calls(line):
            edges.append((statement, add("CALL"), "IS_AST_PARENT"))
        for identifier in IDENTIFIER_PATTERN.findall(line):
            source = last_def.get(identifier.lower())
            if source is not None and source != statement:
                edges.append((source, statement, "REACHES"))
        for target in ASSIGNMENT_PATTERN.findall(line):
            last_def[target.lower()] = statement
    edges.extend((a, b, "FLOWS_TO") for a, b in zip(statements, statements[1:]))
    edges = list(dict.fromkeys(edges))
    return {"nodes": nodes, "edges": edges}


def ast_sequence(graph: dict) -> str:
    children: dict[int, list[int]] = defaultdict(list)
    for source, target, kind in graph["edges"]:
        if kind == "IS_AST_PARENT":
            children[source].append(target)
    out: list[str] = []
    stack: list[tuple[int, bool]] = [(1, False)]
    while stack and len(out) < MAX_AST_TOKENS:
        node, closing = stack.pop()
        node_type = graph["nodes"][node]
        if closing:
            out += [")", node_type]
            continue
        out += ["(", node_type]
        stack.append((node, True))
        for child in reversed(sorted(children.get(node, []))):
            stack.append((child, False))
    return " ".join(out[:MAX_AST_TOKENS])


def _brace_depth(code: str) -> int:
    depth = best = 0
    for char in code:
        if char == "{":
            depth += 1
            best = max(best, depth)
        elif char == "}":
            depth = max(depth - 1, 0)
    return best


def numeric_features(code: str, tokens: list[str], graph: dict, ast_seq: str) -> np.ndarray:
    token_count = max(len(tokens), 1)
    lines = [line for line in code.splitlines() if line.strip()]
    line_count = max(len(lines), 1)
    calls = extract_calls(code)
    risky = sum(1 for call in calls if call.lower() in RISKY_APIS)
    counts = Counter(tokens)
    control = sum(counts[token] for token in CONTROL_TOKENS)
    pointer_ops = code.count("->") + code.count("*") + code.count("&")
    node_types = Counter(graph["nodes"].values())
    edge_types = Counter(kind for _, _, kind in graph["edges"])
    n_nodes = len(graph["nodes"])
    n_edges = len(graph["edges"])
    out_degree = Counter(source for source, _, _ in graph["edges"])

    def ratio(a: float, b: float) -> float:
        return float(a) / float(b) if b else 0.0

    values = [
        math.log1p(token_count), ratio(len(counts), token_count), math.log1p(line_count),
        float(estimate_parameter_count(code)), math.log1p(len(calls)), float(risky), ratio(risky, max(len(calls), 1)),
        ratio(control, token_count), ratio(pointer_ops, line_count), ratio(code.count("["), line_count),
        ratio(counts["num_lit"], token_count), float(sum(counts[k] for k in MEMORY_KEYWORDS)), float(_brace_depth(code)),
        math.log1p(n_nodes), math.log1p(n_edges), ratio(sum(out_degree.values()), max(len(out_degree), 1)),
        ratio(node_types["CALL"], n_nodes), ratio(node_types["CONTROL_STRUCTURE"], n_nodes),
        ratio(node_types["EXPRESSION"], n_nodes), ratio(edge_types["FLOWS_TO"], n_edges),
        ratio(edge_types["IS_AST_PARENT"], n_edges), ratio(edge_types["REACHES"], n_edges),
        float(max(out_degree.values()) if out_degree else 0), math.log1p(len(ast_seq.split())),
    ]
    return np.asarray(values, dtype=np.float32)


def featurize(code: str) -> tuple[list[str], str, np.ndarray, int, int]:
    """Returns (code tokens, AST-like sequence, numeric features, n_code_tokens, n_lines).

    The AST-like view is the syntactic skeleton of the code (keywords, operators, id/call/lit), the v1 fallback
    sequence: the heuristic-graph sequence has only ~10 distinct symbols and carries little signal. The heuristic
    graph is still used for the 24 numeric features (D7)."""
    tokens = tokenize_code(code)
    graph = heuristic_graph(code)
    graph_seq = ast_sequence(graph)
    skeleton = build_skeleton(code)
    return tokens, skeleton, numeric_features(code, tokens, graph, graph_seq), len(tokens), code.count("\n") + 1


def describe_function(code: str) -> str:
    return extract_function_name(code)
