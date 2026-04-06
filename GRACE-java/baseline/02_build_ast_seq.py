from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import javalang
from javalang.parser import JavaSyntaxError, Parser
from javalang.tokenizer import LexerError, tokenize

from common import WORK_DIR, load_jsonl, write_jsonl, wrap_java_method


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build lightweight SimSBT-style AST sequences for Java methods.")
    p.add_argument("--input", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--code-field", default="code")
    p.add_argument("--max-tokens", type=int, default=256)
    return p.parse_args()


def iter_children(node: Any):
    if isinstance(node, javalang.ast.Node):
        for attr in node.attrs:
            child = getattr(node, attr)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, javalang.ast.Node):
                        yield item
            elif isinstance(child, javalang.ast.Node):
                yield child


def normalize_token(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text.rsplit(".", 1)[-1]
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:32].lower()


def literal_kind(value: str) -> str:
    value = (value or "").strip().lower()
    if not value:
        return "empty"
    if value in {"true", "false"}:
        return "bool"
    if value == "null":
        return "null"
    if value.startswith('"') or value.startswith("'"):
        return "string"
    if value.startswith("0x") or value[0].isdigit() or value[0] in {"-", "+"}:
        return "number"
    return "literal"


def node_tag(node: javalang.ast.Node) -> str:
    parts = [type(node).__name__]

    if isinstance(node, javalang.tree.Literal):
        parts.append(f"lit_{literal_kind(getattr(node, 'value', ''))}")

    for attr in ("name", "member", "operator"):
        value = getattr(node, attr, None)
        if isinstance(value, str):
            token = normalize_token(value)
            if token:
                parts.append(f"{attr}_{token}")

    qualifier = getattr(node, "qualifier", None)
    if isinstance(qualifier, str):
        token = normalize_token(qualifier)
        if token:
            parts.append(f"qual_{token}")

    return ":".join(parts)


def sbt_tokens(node: Any) -> list[str]:
    if isinstance(node, javalang.ast.Node):
        name = node_tag(node)
        toks = [name]
        for child in iter_children(node):
            toks.extend(sbt_tokens(child))
        toks.append(f"{name}^")
        return toks
    return []


def extract_first_method_node(tree) -> javalang.tree.MethodDeclaration | javalang.tree.ConstructorDeclaration | None:
    if isinstance(tree, (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)):
        return tree
    for node_type in (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration):
        for _, node in tree.filter(node_type):
            return node
    return None


def parse_member_snippet(code: str):
    parser = Parser(tokenize(code))
    return parser.parse_member_declaration()


def extract_first_method_tree(code: str, class_name: str) -> javalang.tree.MethodDeclaration | javalang.tree.ConstructorDeclaration | None:
    wrapped = wrap_java_method(code, class_name)

    try:
        tree = javalang.parse.parse(wrapped)
        node = extract_first_method_node(tree)
        if node is not None:
            return node
    except (JavaSyntaxError, LexerError, TypeError, IndexError):
        pass

    for candidate in (code, wrapped):
        try:
            member = parse_member_snippet(candidate)
            node = extract_first_method_node(member)
            if node is not None:
                return node
        except (JavaSyntaxError, LexerError, TypeError, IndexError):
            continue
    return None


def process_file(input_path: Path, output_path: Path, code_field: str, max_tokens: int) -> tuple[int, int]:
    rows = load_jsonl(input_path)
    enriched = []
    failed = 0

    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_id", idx))
        code = row.get(code_field, "")
        try:
            method_node = extract_first_method_tree(code, f"Wrap_{idx}")
            ast_tokens = sbt_tokens(method_node) if method_node is not None else []
            ast_seq = " ".join(ast_tokens[:max_tokens])
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
        process_file(Path(args.input), Path(args.output), args.code_field, args.max_tokens)
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
        process_file(in_path, out_path, args.code_field, args.max_tokens)


if __name__ == "__main__":
    main()
