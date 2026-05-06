import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from common import (
    CONTROL_KEYWORDS,
    GRAPH_DIR,
    ensure_dir,
    estimate_parameter_count,
    extract_function_name,
    normalize_code,
    stable_hash,
    truncate_text,
)


GRAPH_SCHEMA_VERSION = 1
DEFAULT_MAX_NODES = 72
DEFAULT_MAX_EDGES = 96
MAX_AST_TOKENS = 512
GRAPH_FILE_NAME = "graph_features.json"
JOERN_INSTALL_ROOT = GRAPH_DIR / "tools" / "joern"
JOERN_HOME_DIR = JOERN_INSTALL_ROOT / "joern-cli"

DOT_NODE_PATTERN = re.compile(r'^\s*"?(?P<node_id>[^"\s]+)"?\s*\[\s*label\s*=\s*"(?P<label>.*)"\s*\]\s*;?\s*$')
DOT_EDGE_PATTERN = re.compile(
    r'^\s*"?(?P<src>[^"\s]+)"?\s*->\s*"?(?P<dst>[^"\s]+)"?(?:\s*\[(?P<attrs>.*?)\])?\s*;?\s*$'
)
DOT_LABEL_ATTR_PATTERN = re.compile(r'label\s*=\s*"(?P<label>[^"]+)"')
IDENTIFIER_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\b")
ASSIGNMENT_PATTERN = re.compile(r"\b([A-Za-z_]\w*)\s*=")

EDGE_LABEL_ALIASES = {
    "AST": "IS_AST_PARENT",
    "CFG": "FLOWS_TO",
    "CDG": "CONTROLS",
    "DDG": "REACHES",
    "DOMINATE": "DOM",
    "POST_DOMINATE": "POST_DOM",
    "REACHING_DEF": "REACHES",
}


def default_graph_cache_dir(dataset_name: str) -> Path:
    return GRAPH_DIR / dataset_name


def graph_cache_path(dataset_name: str, code_hash: str) -> Path:
    return default_graph_cache_dir(dataset_name) / code_hash / GRAPH_FILE_NAME


def resolve_graph_backend(preferred: str = "auto") -> str:
    resolved, _ = resolve_graph_backend_with_notice(preferred)
    return resolved


def resolve_graph_backend_with_notice(preferred: str = "auto") -> tuple[str, str | None]:
    lowered = (preferred or "auto").strip().lower()
    if lowered not in {"auto", "joern", "heuristic"}:
        raise ValueError(f"Unsupported graph backend: {preferred}")
    if lowered != "auto":
        return lowered, None
    joern_ok, joern_notice = _probe_joern_backend()
    if joern_ok:
        return "joern", None
    return "heuristic", joern_notice


def default_joern_install_dir() -> Path:
    return JOERN_HOME_DIR


def resolve_joern_command(executable_name: str) -> str:
    env_name = "GRACE_JOERN_PARSE" if executable_name == "joern-parse" else "GRACE_JOERN_EXPORT"
    configured = os.getenv(env_name)
    if configured:
        return configured
    for candidate in _candidate_joern_commands(executable_name):
        if shutil.which(candidate):
            return candidate
        if Path(candidate).exists():
            return str(Path(candidate))
    return executable_name


def get_graph_features(
    record: dict,
    *,
    dataset_name: str | None = None,
    graph_backend: str = "auto",
    force_rebuild: bool = False,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
    cache_dir: Path | None = None,
) -> dict:
    dataset = dataset_name or record.get("dataset", "default")
    code = normalize_code(record.get("code", ""))
    if not code:
        raise ValueError("Graph extraction requires a non-empty `code` field.")
    code_hash = record.get("code_hash") or stable_hash(code)
    target_path = cache_dir or graph_cache_path(dataset, code_hash)
    if not force_rebuild and target_path.exists():
        cached = json.loads(target_path.read_text(encoding="utf-8"))
        if cached.get("schema_version") == GRAPH_SCHEMA_VERSION:
            return cached
    started = time.perf_counter()
    artifact = build_graph_features(
        code,
        record_id=str(record.get("record_id", "")),
        code_hash=code_hash,
        graph_backend=graph_backend,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    artifact.update(
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "dataset": dataset,
            "record_id": str(record.get("record_id", "")),
            "code_hash": code_hash,
            "cache_path": str(target_path),
            "build_seconds": round(time.perf_counter() - started, 6),
        }
    )
    ensure_dir(target_path.parent)
    target_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def build_graph_features(
    code: str,
    *,
    record_id: str = "",
    code_hash: str = "",
    graph_backend: str = "auto",
    max_nodes: int = DEFAULT_MAX_NODES,
    max_edges: int = DEFAULT_MAX_EDGES,
) -> dict:
    requested_backend = (graph_backend or "auto").strip().lower()
    resolved_backend = resolve_graph_backend(requested_backend)
    last_error = None
    if resolved_backend == "joern":
        try:
            artifact = _build_joern_graph_features(code, max_nodes=max_nodes, max_edges=max_edges)
            artifact["requested_backend"] = requested_backend
            artifact["record_id"] = record_id
            artifact["code_hash"] = code_hash
            return artifact
        except Exception as exc:
            last_error = str(exc)
            if requested_backend == "joern":
                raise
    artifact = _build_heuristic_graph_features(code, max_nodes=max_nodes, max_edges=max_edges)
    artifact["requested_backend"] = requested_backend
    artifact["record_id"] = record_id
    artifact["code_hash"] = code_hash
    if last_error:
        artifact["backend_notice"] = f"Falling back to heuristic graph extraction because Joern failed: {last_error}"
    return artifact


def _build_joern_graph_features(code: str, *, max_nodes: int, max_edges: int) -> dict:
    joern_parse = resolve_joern_command("joern-parse")
    joern_export = resolve_joern_command("joern-export")
    if not _command_exists(joern_parse):
        raise FileNotFoundError(f"Missing Joern parser executable: {joern_parse}")
    if not _command_exists(joern_export):
        raise FileNotFoundError(f"Missing Joern export executable: {joern_export}")

    temp_dir = Path(tempfile.mkdtemp(prefix="grace_joern_"))
    try:
        source_dir = ensure_dir(temp_dir / "src")
        (source_dir / "snippet.c").write_text(normalize_code(code) + "\n", encoding="utf-8")
        _run_joern_command([joern_parse, str(source_dir)], cwd=temp_dir)

        merged_nodes = {}
        merged_edges = []
        export_specs = [
            ("ast", "IS_AST_PARENT"),
            ("cfg", "FLOWS_TO"),
            ("pdg", "REACHES"),
        ]
        for representation, default_edge_type in export_specs:
            output_dir = temp_dir / representation
            try:
                _run_joern_command([joern_export, "--repr", representation, "--out", str(output_dir)], cwd=temp_dir)
            except RuntimeError:
                if representation == "pdg":
                    continue
                raise
            dot_path = _choose_dot_file(output_dir)
            if dot_path is None:
                if representation == "pdg":
                    continue
                raise FileNotFoundError(f"Joern did not export a {representation.upper()} dot file.")
            nodes, edges = _parse_dot_graph(dot_path, default_edge_type=default_edge_type)
            merged_nodes.update(nodes)
            merged_edges.extend(edges)
        if not merged_nodes:
            raise RuntimeError("Joern export returned no nodes.")
        return _finalize_graph_artifact(
            merged_nodes,
            merged_edges,
            backend="joern",
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _run_joern_command(command: list[str], *, cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(command)}"
            f" | stdout={truncate_text(result.stdout or '', 240)}"
            f" | stderr={truncate_text(result.stderr or '', 240)}"
        )


def _choose_dot_file(output_dir: Path) -> Path | None:
    dot_files = sorted(output_dir.rglob("*.dot"), key=lambda path: path.stat().st_size if path.exists() else 0, reverse=True)
    return dot_files[0] if dot_files else None


def _parse_dot_graph(dot_path: Path, *, default_edge_type: str) -> tuple[dict[str, dict], list[dict]]:
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    for line in dot_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        node_match = DOT_NODE_PATTERN.match(line)
        if node_match:
            node_id = node_match.group("node_id")
            node_type, code = _parse_dot_node_label(node_match.group("label"))
            nodes[node_id] = {
                "id": node_id,
                "type": node_type or "UNKNOWN",
                "code": truncate_text(code or node_type or "", 120),
            }
            continue
        edge_match = DOT_EDGE_PATTERN.match(line)
        if edge_match:
            edge_type = _canonicalize_edge_label(_extract_edge_label(edge_match.group("attrs")), default_edge_type)
            edges.append(
                {
                    "source": edge_match.group("src"),
                    "target": edge_match.group("dst"),
                    "type": edge_type,
                }
            )
    return nodes, edges


def _parse_dot_node_label(label: str) -> tuple[str, str]:
    inner = (label or "").strip()
    if inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1]
    first = inner.find(",")
    if first < 0:
        return inner or "UNKNOWN", ""
    second = inner.find(",", first + 1)
    if second < 0:
        return inner[:first].strip() or "UNKNOWN", inner[first + 1 :].strip()
    node_type = inner[:first].strip() or "UNKNOWN"
    primary_code = inner[first + 1 : second].strip()
    fallback_code = inner[second + 1 :].strip()
    return node_type, primary_code or fallback_code


def _extract_edge_label(attrs: str | None) -> str | None:
    if not attrs:
        return None
    match = DOT_LABEL_ATTR_PATTERN.search(attrs)
    return match.group("label") if match else None


def _canonicalize_edge_label(label: str | None, default: str) -> str:
    if not label:
        return default
    candidate = label.strip().replace("-", "_").replace(" ", "_").upper()
    candidate = candidate.split(":")[0]
    return EDGE_LABEL_ALIASES.get(candidate, candidate or default)


def _build_heuristic_graph_features(code: str, *, max_nodes: int, max_edges: int) -> dict:
    text = normalize_code(code)
    function_name = extract_function_name(text)
    param_count = estimate_parameter_count(text)
    nodes: dict[str, dict] = {}
    edges: list[dict] = []
    next_node_id = 1

    def add_node(node_type: str, snippet: str) -> str:
        nonlocal next_node_id
        node_id = str(next_node_id)
        next_node_id += 1
        nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "code": truncate_text(snippet, 120),
        }
        return node_id

    method_id = add_node("METHOD", f"{function_name}()")
    for param_index in range(param_count):
        param_id = add_node("PARAM", f"param_{param_index + 1}")
        edges.append({"source": method_id, "target": param_id, "type": "IS_AST_PARENT"})

    statement_ids: list[str] = []
    last_definition_for_name: dict[str, str] = {}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines[:40]:
        node_type = _heuristic_node_type(line)
        statement_id = add_node(node_type, line)
        statement_ids.append(statement_id)
        edges.append({"source": method_id, "target": statement_id, "type": "IS_AST_PARENT"})
        for call_name in _extract_line_calls(line):
            call_id = add_node("CALL", f"{call_name}(...)")
            edges.append({"source": statement_id, "target": call_id, "type": "IS_AST_PARENT"})
        for identifier in _extract_line_identifiers(line):
            if identifier in last_definition_for_name:
                edges.append({"source": last_definition_for_name[identifier], "target": statement_id, "type": "REACHES"})
        for assigned_name in _extract_assignment_targets(line):
            last_definition_for_name[assigned_name] = statement_id
    for left, right in zip(statement_ids, statement_ids[1:]):
        edges.append({"source": left, "target": right, "type": "FLOWS_TO"})

    return _finalize_graph_artifact(
        nodes,
        edges,
        backend="heuristic",
        max_nodes=max_nodes,
        max_edges=max_edges,
    )


def _heuristic_node_type(line: str) -> str:
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


def _extract_line_calls(line: str) -> list[str]:
    matches = re.findall(r"\b([A-Za-z_]\w*)\s*\(", line)
    results = []
    seen = set()
    for name in matches:
        lowered = name.lower()
        if lowered in CONTROL_KEYWORDS or lowered in seen:
            continue
        seen.add(lowered)
        results.append(name)
    return results


def _extract_line_identifiers(line: str) -> list[str]:
    identifiers = []
    for match in IDENTIFIER_PATTERN.findall(line):
        lowered = match.lower()
        if lowered in CONTROL_KEYWORDS:
            continue
        identifiers.append(lowered)
    return identifiers


def _extract_assignment_targets(line: str) -> list[str]:
    return [match.lower() for match in ASSIGNMENT_PATTERN.findall(line)]


def _finalize_graph_artifact(
    nodes: dict[str, dict],
    edges: list[dict],
    *,
    backend: str,
    max_nodes: int,
    max_edges: int,
) -> dict:
    deduped_edges = []
    seen_edges = set()
    for edge in edges:
        key = (str(edge["source"]), str(edge["target"]), str(edge["type"]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        deduped_edges.append({"source": key[0], "target": key[1], "type": key[2]})
    sorted_nodes = sorted(nodes.values(), key=lambda row: _node_sort_key(row["id"]))
    sorted_edges = sorted(
        deduped_edges,
        key=lambda row: (_node_sort_key(row["source"]), _node_sort_key(row["target"]), row["type"]),
    )
    ast_sequence = _build_ast_sequence(sorted_nodes, sorted_edges)
    node_rows = sorted_nodes[:max_nodes]
    edge_rows = sorted_edges[:max_edges]
    node_type_counts = Counter(row["type"] for row in sorted_nodes)
    edge_type_counts = Counter(row["type"] for row in sorted_edges)
    return {
        "backend": backend,
        "ast_sequence": ast_sequence,
        "node_rows": node_rows,
        "edge_rows": edge_rows,
        "node_info": _format_node_rows(node_rows),
        "edge_info": _format_edge_rows(edge_rows),
        "graph_summary": {
            "nodes": len(sorted_nodes),
            "edges": len(sorted_edges),
            "node_types": dict(node_type_counts.most_common(10)),
            "edge_types": dict(edge_type_counts.most_common(10)),
        },
    }


def _build_ast_sequence(node_rows: list[dict], edge_rows: list[dict]) -> str:
    nodes_by_id = {str(row["id"]): row for row in node_rows}
    ast_children: dict[str, list[str]] = defaultdict(list)
    for edge in edge_rows:
        if edge["type"] != "IS_AST_PARENT":
            continue
        if edge["source"] in nodes_by_id and edge["target"] in nodes_by_id:
            ast_children[str(edge["source"])].append(str(edge["target"]))
    for children in ast_children.values():
        children.sort(key=_node_sort_key)

    method_nodes = [row["id"] for row in node_rows if row["type"] == "METHOD"]
    if method_nodes:
        root_id = str(method_nodes[0])
    elif node_rows:
        root_id = str(node_rows[0]["id"])
    else:
        return ""

    sequence: list[str] = []
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visited or node_id not in nodes_by_id:
            return
        visited.add(node_id)
        node_type = nodes_by_id[node_id]["type"]
        sequence.append("(")
        sequence.append(node_type)
        for child_id in ast_children.get(node_id, []):
            visit(child_id)
        sequence.append(")")
        sequence.append(node_type)

    visit(root_id)
    if not sequence:
        sequence = [row["type"] for row in node_rows]
    return " ".join(sequence[:MAX_AST_TOKENS])


def _format_node_rows(rows: list[dict]) -> str:
    lines = ["NodeID\tNodeType\tCode"]
    for row in rows:
        lines.append(f"{row['id']}\t{row['type']}\t{truncate_text(row['code'], 90)}")
    return "\n".join(lines)


def _format_edge_rows(rows: list[dict]) -> str:
    lines = ["Node1\tNode2\tEdgeType"]
    for row in rows:
        lines.append(f"{row['source']}\t{row['target']}\t{row['type']}")
    return "\n".join(lines)


def _node_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):020d}") if text.isdigit() else (1, text)


def _has_joern_tools() -> bool:
    joern_parse = resolve_joern_command("joern-parse")
    joern_export = resolve_joern_command("joern-export")
    return _command_exists(joern_parse) and _command_exists(joern_export)


@lru_cache(maxsize=1)
def _probe_joern_backend() -> tuple[bool, str | None]:
    if not _has_joern_tools():
        return False, "Joern executables were not found."
    try:
        artifact = _build_joern_graph_features(
            "int grace_probe(int value) { return value + 1; }",
            max_nodes=16,
            max_edges=16,
        )
    except Exception as exc:
        return False, f"Joern health probe failed: {exc}"
    if not artifact.get("node_rows"):
        return False, "Joern health probe returned no nodes."
    return True, None


def _command_exists(command: str) -> bool:
    return shutil.which(command) is not None or Path(command).exists()


def _candidate_joern_commands(executable_name: str) -> list[str]:
    if os.name == "nt":
        suffixes = [".bat", ".cmd", ".exe", ""]
    else:
        suffixes = ["", ".sh", ".bat", ".cmd", ".exe"]
    candidates = []
    for suffix in suffixes:
        candidates.append(executable_name + suffix)
    for suffix in suffixes:
        candidates.append(str(JOERN_HOME_DIR / (executable_name + suffix)))
        candidates.append(str(JOERN_INSTALL_ROOT / (executable_name + suffix)))
        candidates.append(str(JOERN_HOME_DIR / "bin" / (executable_name + suffix)))
    return candidates
