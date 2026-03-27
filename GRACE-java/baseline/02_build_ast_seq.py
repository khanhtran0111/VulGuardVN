from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import javalang

from common import WORK_DIR, load_jsonl, write_jsonl, wrap_java_method


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lightweight SimSBT-style AST sequences for Java methods.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--code-field", default="code")
    return p.parse_args()


def iter_children(node: Any):
    if isinstance(node, javalang.ast.Node):
        for attr in node.attrs:
            child = getattr(node, attr)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                for item in child:
                    if item is not None:
                        yield item
            else:
                yield child


def sbt_tokens(node: Any) -> list[str]:
    if isinstance(node, javalang.ast.Node):
        name = type(node).__name__
        toks = [name]
        for child in iter_children(node):
            toks.extend(sbt_tokens(child))
        toks.append(f"{name}^")
        return toks
    if isinstance(node, str):
        return ["STR", "STR^"]
    if isinstance(node, (int, float, bool)):
        return ["LIT", "LIT^"]
    return []


def extract_first_method_tree(code: str, class_name: str) -> javalang.tree.MethodDeclaration | javalang.tree.ConstructorDeclaration | None:
    wrapped = wrap_java_method(code, class_name)
    tree = javalang.parse.parse(wrapped)
    for _, node in tree.filter((javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
        return node
    return None


def process_file(input_path: Path, output_path: Path, code_field: str) -> tuple[int, int]:
    rows = load_jsonl(input_path)
    enriched = []
    failed = 0

    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", idx))
        code = row.get(code_field, "")
        try:
            method_node = extract_first_method_tree(code, f"Wrap_{idx}")
            ast_seq = " ".join(sbt_tokens(method_node)) if method_node is not None else ""
            row["ast_seq"] = ast_seq
            row["ast_parse_ok"] = bool(ast_seq)
        except Exception as e:  # noqa: BLE001
            row["ast_seq"] = ""
            row["ast_parse_ok"] = False
            row["ast_error"] = str(e)
            row["ast_sample_id"] = sample_id
            failed += 1
        enriched.append(row)

    write_jsonl(output_path, enriched)
    print(f"Wrote {len(enriched)} rows to {output_path} | parse_failed={failed}")
    return len(enriched), failed


def main() -> None:
    args = parse_args()
    if args.input:
        if not args.output:
            raise ValueError("When --input is provided, --output must also be provided.")
        process_file(Path(args.input), Path(args.output), args.code_field)
        return

    if args.split == "all":
        pairs = [
            (WORK_DIR / "splits" / "train.jsonl", WORK_DIR / "train_ast.jsonl"),
            (WORK_DIR / "splits" / "val.jsonl", WORK_DIR / "val_ast.jsonl"),
            (WORK_DIR / "splits" / "test.jsonl", WORK_DIR / "test_ast.jsonl"),
        ]
    else:
        pairs = [
            (WORK_DIR / "splits" / f"{args.split}.jsonl", WORK_DIR / f"{args.split}_ast.jsonl"),
        ]

    for in_path, out_path in pairs:
        if not in_path.exists():
            raise FileNotFoundError(f"Missing input split: {in_path}. Run 01_split_dataset.py first.")
        process_file(in_path, out_path, args.code_field)


if __name__ == "__main__":
    main()
