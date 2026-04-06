from __future__ import annotations

import argparse

from common import WORK_DIR, load_jsonl, truncate_text, write_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assemble Java GRACE-style prompts.")
    p.add_argument("--input", default=str(WORK_DIR / "test_graph_text_5.jsonl"))
    p.add_argument("--output", default=str(WORK_DIR / "test_prompts_5.jsonl"))
    p.add_argument("--code-max", type=int, default=4000)
    p.add_argument("--node-max", type=int, default=2000)
    p.add_argument("--edge-max", type=int, default=2000)
    p.add_argument("--example-max", type=int, default=4000)
    return p.parse_args()


def label_to_text(label) -> str:
    return "Vulnerable" if int(label) == 1 else "Non-vulnerable"


def build_prompt(row: dict, code_max: int, node_max: int, edge_max: int, example_max: int) -> str:
    code = truncate_text(row["code"], code_max)
    node_text = truncate_text(row.get("graph_nodes_text", ""), node_max)
    edge_text = truncate_text(row.get("graph_edges_text", ""), edge_max)
    example_code = truncate_text(row.get("example", ""), example_max)
    example_label = label_to_text(row.get("example_label", 0))

    parts = [
        "You are an expert Java security analyst.",
        "You are conducting a function vulnerability detection task for Java.",
        "[Code Snippet]",
        code,
    ]
    if node_text:
        parts.extend(["[Node Information]", node_text])
    if edge_text:
        parts.extend(["[Edge Information]", edge_text])
    if example_code:
        parts.extend(
            [
                "[Demonstration]",
                example_code,
                f"[Demonstration Label] {example_label}",
            ]
        )
    parts.append("In the above Java code snippet, check for potential security vulnerabilities and output either 'Vulnerable' or 'Non-vulnerable'.")
    return "\n\n".join(parts)


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    out_rows = []
    for row in rows:
        out = dict(row)
        out["prompt"] = build_prompt(row, args.code_max, args.node_max, args.edge_max, args.example_max)
        out_rows.append(out)
    write_jsonl(args.output, out_rows)
    print(f"Saved prompts to {args.output}")


if __name__ == "__main__":
    main()
