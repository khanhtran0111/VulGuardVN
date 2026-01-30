import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


CONFIG = {
    "input": "../VulJ.jsonl",
    "outdir": "./data",
    "seed": 42,

    "drop_tests": True,
    "strip_comments": True,
    "drop_truncated": True,
    "require_balanced_braces": True,
    "dedup": True,

    "min_chars": 80,          # NEW: bỏ snippet quá ngắn (thường là noise)

    "train": 0.8,
    "valid": 0.1,
    "test": 0.1,

    "split_stratify": True,   # NEW: stratify by label at vul_id level
    "balance_train": "oversample",  # NEW: none | oversample | undersample
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
    """Best-effort balance check (still imperfect with strings)."""
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
    min_chars: int,
) -> Tuple[List[Item], Dict[str, int]]:
    stats = {
        "read": 0,
        "kept": 0,
        "dropped_missing_fields": 0,
        "dropped_label_invalid": 0,
        "dropped_tests": 0,
        "dropped_too_short": 0,
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

        if min_chars and len(text) < int(min_chars):
            stats["dropped_too_short"] += 1
            continue

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

        if method and len(method) > 200:
            method = method[:200]

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

def label_distribution(items: List[Item]) -> Dict[str, int]:
    pos = sum(1 for x in items if x.labels == 1)
    neg = len(items) - pos
    return {"n": len(items), "pos": pos, "neg": neg}

def group_by_vul_id(items: List[Item]) -> Dict[str, List[Item]]:
    groups: Dict[str, List[Item]] = {}
    for it in items:
        groups.setdefault(it.vul_id, []).append(it)
    return groups

def vul_id_label(groups: Dict[str, List[Item]]) -> Dict[str, int]:
    """Label at vul_id-level: if any sample is positive => vul_id positive."""
    out = {}
    for vid, rows in groups.items():
        out[vid] = 1 if any(r.labels == 1 for r in rows) else 0
    return out

def stratified_group_split_by_vul_id(
    items: List[Item],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Item], List[Item], List[Item]]:
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)
    groups = group_by_vul_id(items)
    vid2lab = vul_id_label(groups)

    pos_vids = [vid for vid, lab in vid2lab.items() if lab == 1]
    neg_vids = [vid for vid, lab in vid2lab.items() if lab == 0]

    rng.shuffle(pos_vids)
    rng.shuffle(neg_vids)

    def split_vids(vids: List[str]) -> Tuple[set, set, set]:
        n = len(vids)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        train_ids = set(vids[:n_train])
        valid_ids = set(vids[n_train:n_train + n_valid])
        test_ids = set(vids[n_train + n_valid:])
        return train_ids, valid_ids, test_ids

    pos_train, pos_valid, pos_test = split_vids(pos_vids)
    neg_train, neg_valid, neg_test = split_vids(neg_vids)

    train_ids = pos_train | neg_train
    valid_ids = pos_valid | neg_valid
    test_ids = pos_test | neg_test

    train, valid, test = [], [], []
    for vid, rows in groups.items():
        if vid in train_ids:
            train.extend(rows)
        elif vid in valid_ids:
            valid.extend(rows)
        else:
            test.extend(rows)

    return train, valid, test

def plain_group_split_by_vul_id(
    items: List[Item],
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[Item], List[Item], List[Item]]:
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)
    groups = group_by_vul_id(items)
    vul_ids = list(groups.keys())
    rng.shuffle(vul_ids)

    n = len(vul_ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_ids = set(vul_ids[:n_train])
    valid_ids = set(vul_ids[n_train:n_train + n_valid])

    train, valid, test = [], [], []
    for vid, rows in groups.items():
        if vid in train_ids:
            train.extend(rows)
        elif vid in valid_ids:
            valid.extend(rows)
        else:
            test.extend(rows)
    return train, valid, test

def balance_train_items(items: List[Item], mode: str, seed: int) -> List[Item]:
    mode = (mode or "none").lower()
    if mode == "none":
        return items

    rng = random.Random(seed)
    pos = [x for x in items if x.labels == 1]
    neg = [x for x in items if x.labels == 0]
    if not pos or not neg:
        return items

    if mode == "undersample":
        k = min(len(pos), len(neg))
        rng.shuffle(pos); rng.shuffle(neg)
        out = pos[:k] + neg[:k]
        rng.shuffle(out)
        return out

    if mode == "oversample":
        # oversample minority to match majority
        if len(pos) < len(neg):
            minor, major = pos, neg
            minor_label = 1
        else:
            minor, major = neg, pos
            minor_label = 0
        need = len(major) - len(minor)
        extra = [rng.choice(minor) for _ in range(need)]
        out = major + minor + extra
        rng.shuffle(out)
        return out

    raise ValueError(f"Unknown balance mode: {mode}. Use none|oversample|undersample.")

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", default=CONFIG["input"])
    ap.add_argument("--outdir", default=CONFIG["outdir"])
    ap.add_argument("--seed", type=int, default=CONFIG["seed"])

    ap.add_argument("--drop-tests", action=argparse.BooleanOptionalAction, default=CONFIG["drop_tests"])
    ap.add_argument("--strip-comments", action=argparse.BooleanOptionalAction, default=CONFIG["strip_comments"])
    ap.add_argument("--drop-truncated", action=argparse.BooleanOptionalAction, default=CONFIG["drop_truncated"])
    ap.add_argument("--require-balanced-braces", action=argparse.BooleanOptionalAction, default=CONFIG["require_balanced_braces"])
    ap.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=CONFIG["dedup"])

    ap.add_argument("--min-chars", type=int, default=CONFIG["min_chars"])

    ap.add_argument("--train", type=float, default=CONFIG["train"])
    ap.add_argument("--valid", type=float, default=CONFIG["valid"])
    ap.add_argument("--test", type=float, default=CONFIG["test"])

    ap.add_argument("--split-stratify", action=argparse.BooleanOptionalAction, default=CONFIG["split_stratify"])
    ap.add_argument("--balance-train", choices=["none", "oversample", "undersample"], default=CONFIG["balance_train"])

    args = ap.parse_args()

    items, stats = process_records(
        args.input,
        drop_tests=args.drop_tests,
        strip_comments=args.strip_comments,
        drop_truncated=args.drop_truncated,
        require_balanced_braces=args.require_balanced_braces,
        dedup=args.dedup,
        min_chars=args.min_chars,
    )

    if args.split_stratify:
        train, valid, test = stratified_group_split_by_vul_id(
            items, train_ratio=args.train, valid_ratio=args.valid, test_ratio=args.test, seed=args.seed
        )
    else:
        train, valid, test = plain_group_split_by_vul_id(
            items, train_ratio=args.train, valid_ratio=args.valid, test_ratio=args.test, seed=args.seed
        )

    train_bal = balance_train_items(train, args.balance_train, seed=args.seed)

    os.makedirs(args.outdir, exist_ok=True)
    write_jsonl(os.path.join(args.outdir, "train.jsonl"), train_bal)
    write_jsonl(os.path.join(args.outdir, "valid.jsonl"), valid)
    write_jsonl(os.path.join(args.outdir, "test.jsonl"), test)

    print("=== Processing stats ===")
    for k, v in stats.items():
        print(f"{k:28s}: {v}")

    print("\n=== Split label distribution (BEFORE train balancing) ===")
    print("train:", label_distribution(train))
    print("valid:", label_distribution(valid))
    print("test :", label_distribution(test))

    if args.balance_train != "none":
        print("\n=== Train label distribution (AFTER balancing) ===")
        print("train_bal:", label_distribution(train_bal))

    print("\nWrote:")
    print(" -", os.path.join(args.outdir, "train.jsonl"))
    print(" -", os.path.join(args.outdir, "valid.jsonl"))
    print(" -", os.path.join(args.outdir, "test.jsonl"))

if __name__ == "__main__":
    main()
