from __future__ import annotations

import argparse
import re
from pathlib import Path

from common import WORK_DIR, load_jsonl, write_jsonl

NODE_RE = re.compile(r'^\s*"(?P<id>[^"]+)"\s*\[label\s*=\s*"(?P<label>.*)"\s*\]')
EDGE_RE = re.compile(r'^\s*"(?P<src>[^"]+)"\s*->\s*"(?P<dst>[^"]+)"')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert Joern dot outputs into prompt-friendly node/edge text.")
    p.add_argument("--input", default=str(WORK_DIR / "test_with_joern.jsonl"))
    p.add_argument("--output", default=str(WORK_DIR / "test_graph_text.jsonl"))
    p.add_argument("--max-nodes", type=int, default=80)
    p.add_argument("--max-edges", type=int, default=120)
    return p.parse_args()


def parse_dot(path: str | None, tag: str):
    nodes = {}
    edges = []
    if not path or not Path(path).exists():
        return nodes, edges
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        n = NODE_RE.match(line)
        if n:
            nodes[n.group("id")] = f"[{tag}] {n.group('label')}"
            continue
        e = EDGE_RE.match(line)
        if e:
            edges.append((e.group("src"), e.group("dst"), tag))
    return nodes, edges


def build_text(row: dict, max_nodes: int, max_edges: int):
    nodes = {}
    edges = []
    for key, tag in [("joern_ast_dot", "AST"), ("joern_cfg_dot", "CFG"), ("joern_pdg_dot", "PDG")]:
        n, e = parse_dot(row.get(key), tag)
        nodes.update(n)
        edges.extend(e)

    selected_node_ids = list(nodes.keys())[:max_nodes]
    node_lines = [f"{nid}: {nodes[nid]}" for nid in selected_node_ids]

    edge_lines = []
    for src, dst, tag in edges:
        if src in nodes and dst in nodes:
            edge_lines.append(f"[{tag}] {src} -> {dst}")
        if len(edge_lines) >= max_edges:
            break

    return "\n".join(node_lines), "\n".join(edge_lines)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    enriched = []
    for row in rows:
        node_text, edge_text = build_text(row, args.max_nodes, args.max_edges)
        out = dict(row)
        out["graph_nodes_text"] = node_text
        out["graph_edges_text"] = edge_text
        enriched.append(out)

    write_jsonl(args.output, enriched)
    print(f"Saved graph text for {len(enriched)} rows to {args.output}")


if __name__ == "__main__":
    main()
