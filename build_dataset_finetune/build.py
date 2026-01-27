import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


CONFIG = {
    "input": "../VulJ.jsonl",           # Đường dẫn đến file VulJ.jsonl
    "outdir": "./data",                 # Thư mục output
    "seed": 42,                         # Random seed
    
    "drop_tests": True,                 # Bỏ các file test
    "strip_comments": True,             # Xóa comments
    "drop_truncated": True,             # Bỏ code bị cắt cụt
    "require_balanced_braces": True,    # Yêu cầu cân bằng dấu ngoặc
    "dedup": True,                      # Loại bỏ trùng lặp
    
    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,
}


JAVA_LINE_COMMENT_RE = re.compile(r"//.*?$", re.MULTILINE)
JAVA_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

JAVA_IDENTIFIER_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")

SUSPICIOUS_START_RE = re.compile(
    r"^\s*(blic|verride|synchronized\s*$|static\s*$|final\s*$|\)|\]|\}|,|;|:)",
    re.IGNORECASE,
)

def strip_java_comments(code: str) -> str:
    code = re.sub(JAVA_BLOCK_COMMENT_RE, "", code)
    code = re.sub(JAVA_LINE_COMMENT_RE, "", code)
    return code

def normalize_whitespace(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    code = re.sub(r"\n{3,}", "\n\n", code)
    code = "\n".join([ln.rstrip() for ln in code.split("\n")]).strip()
    return code

def brace_balance_ok(code: str) -> bool:
    """Check simple balance of braces/parens/brackets (ignoring strings is hard; best-effort)."""
    pairs = {"{": "}", "(": ")", "[": "]"}
    opens = []
    for ch in code:
        if ch in pairs:
            opens.append(ch)
        elif ch in pairs.values():
            if not opens:
                return False
            top = opens.pop()
            if pairs[top] != ch:
                return False
    return len(opens) == 0

def looks_truncated(code: str) -> bool:
    """Heuristic truncation detector."""
    if not code or len(code) < 20:
        return True
    if SUSPICIOUS_START_RE.search(code):
        return True
    first = code.lstrip()[:15]
    if first and first[0] in "}),;]":
        return True
    return False


def is_test_path(path: str) -> bool:
    p = (path or "").replace("\\", "/").lower()
    return "/src/test/" in p or p.endswith("test.java") or "/test/" in p


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


@dataclass
class Item:
    id: str
    vul_id: str
    file: str
    method: str
    version: str
    labels: int
    text: str
    code_hash: str
    is_test: bool

def read_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i}: {e}") from e
            

def write_jsonl(path: str, rows: List[Item]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            obj = {
                "id": r.id,
                "vul_id": r.vul_id,
                "file": r.file,
                "method": r.method,
                "version": r.version,
                "text": r.text,
                "labels": r.labels,
                "code_hash": r.code_hash,
                "is_test": r.is_test,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def process_records(
    in_path: str,
    *,
    drop_tests: bool,
    strip_comments: bool,
    drop_truncated: bool,
    require_balanced_braces: bool,
    dedup: bool,
) -> Tuple[List[Item], Dict[str, int]]:
    stats = {
        "read": 0,
        "kept": 0,
        "dropped_missing_fields": 0,
        "dropped_label_invalid": 0,
        "dropped_tests": 0,
        "dropped_truncated": 0,
        "dropped_unbalanced": 0,
        "dropped_duplicate": 0,
    }

    seen_hashes = set()
    out: List[Item] = []

    for row in read_jsonl(in_path):
        stats["read"] += 1

        vul_id = row.get("vul_id")
        file_ = row.get("file")
        method = row.get("method", "")
        version = row.get("version", "")
        label = row.get("label")
        code = row.get("code")

        if vul_id is None or file_ is None or label is None or code is None:
            stats["dropped_missing_fields"] += 1
            continue

        try:
            labels = int(label)
        except Exception:
            stats["dropped_label_invalid"] += 1
            continue

        if labels not in (0, 1):
            stats["dropped_label_invalid"] += 1
            continue

        test_flag = is_test_path(file_)
        if drop_tests and test_flag:
            stats["dropped_tests"] += 1
            continue

        text = code
        if strip_comments:
            text = strip_java_comments(text)
        text = normalize_whitespace(text)

        if drop_truncated and looks_truncated(text):
            stats["dropped_truncated"] += 1
            continue

        if require_balanced_braces and not brace_balance_ok(text):
            stats["dropped_unbalanced"] += 1
            continue

        h = sha1_text(text)
        if dedup and h in seen_hashes:
            stats["dropped_duplicate"] += 1
            continue
        seen_hashes.add(h)
        if method and not JAVA_IDENTIFIER_RE.match(method):
            method = ""

        item_id = f"{vul_id}:{h[:12]}"
        out.append(
            Item(
                id=item_id,
                vul_id=str(vul_id),
                file=str(file_),
                method=str(method),
                version=str(version),
                labels=labels,
                text=text,
                code_hash=h,
                is_test=test_flag,
            )
        )
        stats["kept"] += 1

    return out, stats

def group_split_by_vul_id(
    items: List[Item],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Item], List[Item], List[Item]]:
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e6

    rng = random.Random(seed)
    groups: Dict[str, List[Item]] ={}
    for  it in items:
        groups.setdefault(it.vul_id, []).append(it)

    vul_ids = list(groups.keys())
    rng.shuffle(vul_ids)

    n = len(vul_ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # remainder goes to test
    train_ids = set(vul_ids[:n_train])
    valid_ids = set(vul_ids[n_train : n_train + n_valid])
    test_ids = set(vul_ids[n_train + n_valid :])

    train, valid, test = [], [], []
    for vid, rows in groups.items():
        if vid in train_ids:
            train.extend(rows)
        elif vid in valid_ids:
            valid.extend(rows)
        else:
            test.extend(rows)

    return train, valid, test


def label_distribution(items: List[Item]) -> Dict[str, int]:
    pos = sum(1 for x in items if x.labels == 1)
    neg = len(items) - pos
    return {"n": len(items), "pos": pos, "neg": neg}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=CONFIG["input"], help="Path to VulJ.jsonl")
    ap.add_argument("--outdir", default=CONFIG["outdir"], help="Output directory")
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])

    ap.add_argument("--drop-tests", action="store_true", default=CONFIG["drop_tests"], help="Drop src/test/ and other test paths")
    ap.add_argument("--strip-comments", action="store_true", default=CONFIG["strip_comments"], help="Remove Java comments")
    ap.add_argument("--drop-truncated", action="store_true", default=CONFIG["drop_truncated"], help="Drop samples that look truncated")
    ap.add_argument("--require-balanced-braces", action="store_true", default=CONFIG["require_balanced_braces"], help="Drop brace-unbalanced samples")
    ap.add_argument("--dedup", action="store_true", default=CONFIG["dedup"], help="Deduplicate by code hash")

    ap.add_argument("--train", type=float, default=CONFIG["train"])
    ap.add_argument("--valid", type=float, default=CONFIG["valid"])
    ap.add_argument("--test", type=float, default=CONFIG["test"])

    args = ap.parse_args()

    items, stats = process_records(
        args.input,
        drop_tests=args.drop_tests,
        strip_comments=args.strip_comments,
        drop_truncated=args.drop_truncated,
        require_balanced_braces=args.require_balanced_braces,
        dedup=args.dedup,
    )

    train, valid, test = group_split_by_vul_id(
        items,
        train_ratio=args.train,
        valid_ratio=args.valid,
        test_ratio=args.test,
        seed=args.seed,
    )

    os.makedirs(args.outdir, exist_ok=True)
    write_jsonl(os.path.join(args.outdir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.outdir, "valid.jsonl"), valid)
    write_jsonl(os.path.join(args.outdir, "test.jsonl"), test)

    print("=== Processing stats ===")
    for k, v in stats.items():
        print(f"{k:28s}: {v}")

    print("\n=== Split label distribution ===")
    print("train:", label_distribution(train))
    print("valid:", label_distribution(valid))
    print("test :", label_distribution(test))

    print("\nWrote:")
    print(" -", os.path.join(args.outdir, "train.jsonl"))
    print(" -", os.path.join(args.outdir, "valid.jsonl"))
    print(" -", os.path.join(args.outdir, "test.jsonl"))
    print("\nEach row contains: text (code) and labels (0/1), ready for Hugging Face Datasets/Trainer.")


if __name__ == "__main__":
    main()