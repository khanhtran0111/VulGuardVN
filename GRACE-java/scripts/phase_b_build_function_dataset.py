from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import requests

RAW_GITHUB_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"
NAME_RE = re.compile(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\(")
DECL_EXCLUDE_RE = re.compile(r"\b(if|for|while|switch|catch|return|new|throw|else|do|try|synchronized)\b")
CLASS_RE_TEMPLATE = r"\b(class|interface|enum|record)\s+{name}\b"
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B (improved): build function/class-level dataset")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/CWE-Bench-Java"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/github_raw"))
    parser.add_argument("--min-lines", type=int, default=5)
    parser.add_argument("--max-lines", type=int, default=500)
    parser.add_argument("--focus-context", type=int, default=8)
    parser.add_argument("--prefer-latest-touching-commit", action="store_true", default=True)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_id(parts: Sequence[object]) -> str:
    text = "||".join(str(x) for x in parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def append_jsonl(path: Path, obj: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def count_nonempty(values: Iterable[str]) -> int:
    return sum(1 for v in values if v)


def split_fix_commits(value: object) -> List[str]:
    text = normalize_text(value)
    return [x.strip() for x in text.split(";") if x.strip()]


class GitHubRawCache:
    def __init__(self, cache_dir: Path, timeout: float) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.session = requests.Session()
        ensure_dir(cache_dir)

    def fetch(self, owner: str, repo: str, commit: str, file_path: str) -> Optional[str]:
        safe_path = file_path.replace("/", "__")
        cache_path = self.cache_dir / owner / repo / commit / safe_path
        ensure_dir(cache_path.parent)
        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="ignore")

        url = RAW_GITHUB_URL.format(owner=owner, repo=repo, commit=commit, path=file_path)
        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                return None
            text = response.text
            cache_path.write_text(text, encoding="utf-8")
            return text
        except requests.RequestException:
            return None


def parse_signature(signature: str, fallback_method: str = "") -> Tuple[str, Optional[int]]:
    sig = normalize_text(signature)
    if not sig:
        return normalize_text(fallback_method), None
    m = re.search(r"([A-Za-z_$][A-Za-z0-9_$]*)\s*\((.*)\)", sig)
    if not m:
        return normalize_text(fallback_method), None
    method_name = m.group(1)
    inside = m.group(2).strip()
    if not inside:
        return method_name, 0
    depth = 0
    arity = 1
    for ch in inside:
        if ch == '<':
            depth += 1
        elif ch == '>':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            arity += 1
    return method_name, arity


def count_decl_arity(signature_text: str) -> Optional[int]:
    if "(" not in signature_text or ")" not in signature_text:
        return None
    start = signature_text.find("(")
    end = signature_text.rfind(")")
    inside = signature_text[start + 1 : end].strip()
    if not inside:
        return 0
    depth_angle = depth_paren = depth_brack = 0
    arity = 1
    for ch in inside:
        if ch == '<':
            depth_angle += 1
        elif ch == '>':
            depth_angle = max(0, depth_angle - 1)
        elif ch == '(':
            depth_paren += 1
        elif ch == ')':
            depth_paren = max(0, depth_paren - 1)
        elif ch == '[':
            depth_brack += 1
        elif ch == ']':
            depth_brack = max(0, depth_brack - 1)
        elif ch == ',' and depth_angle == depth_paren == depth_brack == 0:
            arity += 1
    return arity


def clean_decl_candidate(text: str) -> str:
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", " ", text)
    return SPACE_RE.sub(" ", text).strip()


def choose_candidate(candidates: List[Tuple[int, int, str]], hint_line: Optional[int]) -> Optional[Tuple[int, int, str]]:
    if not candidates:
        return None
    if hint_line is None:
        return candidates[0]
    return min(candidates, key=lambda item: abs(item[0] + 1 - hint_line))


def find_method_candidates(
    lines: List[str], method_name: str, arity: Optional[int], hint_line: Optional[int]
) -> List[Tuple[int, int, str]]:
    candidates: List[Tuple[int, int, str]] = []
    if not method_name:
        return candidates

    for i in range(len(lines)):
        line = lines[i]
        if method_name not in line or "(" not in line:
            continue
        window_parts = [line.rstrip("\n")]
        end_line = i
        for j in range(i + 1, min(i + 8, len(lines))):
            joined = " ".join(window_parts)
            if "{" in joined or ";" in joined:
                break
            window_parts.append(lines[j].rstrip("\n"))
            end_line = j
        decl = clean_decl_candidate(" ".join(window_parts))
        if f"{method_name}(" not in decl and f" {method_name}(" not in decl:
            continue
        if DECL_EXCLUDE_RE.search(decl):
            continue
        if ";" in decl and "{" not in decl:
            continue
        detected_arity = count_decl_arity(decl)
        if arity is not None and detected_arity is not None and detected_arity != arity:
            continue
        candidates.append((i, end_line, decl))

    candidates.sort(key=lambda item: (abs((item[0] + 1) - hint_line) if hint_line is not None else item[0], item[0]))
    return candidates


def expand_block_from_decl(lines: List[str], start_idx: int, end_hint_idx: Optional[int] = None) -> Optional[Tuple[int, int, str]]:
    brace_started = False
    depth = 0
    block_start = start_idx
    block_end = None

    for i in range(start_idx, len(lines)):
        line = lines[i]
        for ch in line:
            if ch == '{':
                if not brace_started:
                    brace_started = True
                    block_start = start_idx
                depth += 1
            elif ch == '}':
                if brace_started:
                    depth -= 1
                    if depth == 0:
                        block_end = i
                        break
        if block_end is not None:
            break
        if not brace_started and i - start_idx > 8:
            return None
    if block_end is None:
        return None
    code = "\n".join(lines[block_start : block_end + 1]).strip()
    return block_start + 1, block_end + 1, code


def extract_class_block(text: str, class_name: str, hint_line: Optional[int]) -> Optional[Tuple[int, int, str]]:
    class_name = normalize_text(class_name)
    if not class_name:
        return None
    lines = text.splitlines()
    pattern = re.compile(CLASS_RE_TEMPLATE.format(name=re.escape(class_name)))
    candidates = [i for i, line in enumerate(lines) if pattern.search(line)]
    if not candidates:
        return None
    if hint_line is not None:
        start_idx = min(candidates, key=lambda i: abs((i + 1) - hint_line))
    else:
        start_idx = candidates[0]
    return expand_block_from_decl(lines, start_idx)


def extract_method_block(
    text: str,
    signature: str,
    fallback_method: str,
    hint_line: Optional[int],
) -> Optional[Tuple[int, int, str]]:
    lines = text.splitlines()
    method_name, arity = parse_signature(signature, fallback_method=fallback_method)
    candidates = find_method_candidates(lines, method_name, arity, hint_line)
    for start_idx, _end_idx, _decl in candidates:
        block = expand_block_from_decl(lines, start_idx)
        if block is not None:
            return block
    return None


def slice_by_lines(text: str, start_line: int, end_line: int) -> Optional[Tuple[int, int, str]]:
    lines = text.splitlines()
    if start_line <= 0 or end_line <= 0 or start_line > end_line or end_line > len(lines):
        return None
    code = "\n".join(lines[start_line - 1 : end_line]).strip()
    if not code:
        return None
    return start_line, end_line, code


def trim_changed_region(buggy_code: str, fixed_code: str, context: int) -> Tuple[str, str, Dict[str, int]]:
    bug_lines = buggy_code.splitlines()
    fix_lines = fixed_code.splitlines()
    matcher = __import__("difflib").SequenceMatcher(a=bug_lines, b=fix_lines)
    bug_spans: List[Tuple[int, int]] = []
    fix_spans: List[Tuple[int, int]] = []
    for tag, a0, a1, b0, b1 in matcher.get_opcodes():
        if tag == "equal":
            continue
        bug_spans.append((max(0, a0 - context), min(len(bug_lines), a1 + context)))
        fix_spans.append((max(0, b0 - context), min(len(fix_lines), b1 + context)))
    if not bug_spans or not fix_spans:
        return buggy_code, fixed_code, {"changed_ops": 0}

    def merge(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        spans = sorted(spans)
        out: List[Tuple[int, int]] = []
        for s, e in spans:
            if not out or s > out[-1][1]:
                out.append([s, e])
            else:
                out[-1][1] = max(out[-1][1], e)
        return [(int(s), int(e)) for s, e in out]

    bug_spans = merge(bug_spans)
    fix_spans = merge(fix_spans)
    buggy_focus = "\n\n// ...\n\n".join("\n".join(bug_lines[s:e]).strip() for s, e in bug_spans).strip()
    fixed_focus = "\n\n// ...\n\n".join("\n".join(fix_lines[s:e]).strip() for s, e in fix_spans).strip()
    return buggy_focus or buggy_code, fixed_focus or fixed_code, {"changed_ops": len(bug_spans)}


def select_model_code(full_code: str, focus_code: str, max_full_lines: int = 220) -> str:
    if len(full_code.splitlines()) <= max_full_lines:
        return full_code
    return focus_code or full_code


def unit_kind(row: pd.Series) -> str:
    has_method = not (pd.isna(row.get("method")) or pd.isna(row.get("method_start")) or pd.isna(row.get("method_end")))
    return "method" if has_method else "class"


def build_unit_key(row: pd.Series) -> str:
    signature = normalize_text(row.get("signature"))
    if signature:
        suffix = f"SIG::{signature}"
    else:
        suffix = f"CLASS::{normalize_text(row.get('class'))}"
    return "::".join([normalize_text(row.get("project_slug")), normalize_text(row.get("file")), suffix])


def commit_order_map(project_row: pd.Series) -> Dict[str, int]:
    commits = split_fix_commits(project_row.get("fix_commit_ids"))
    return {c: i for i, c in enumerate(commits)}


def choose_representative_rows(project_info: pd.DataFrame, fix_info: pd.DataFrame) -> pd.DataFrame:
    project_info = project_info.copy()
    fix_info = fix_info.copy()

    project_info["project_slug"] = project_info["project_slug"].astype(str).str.strip()
    fix_info["project_slug"] = fix_info["project_slug"].astype(str).str.strip()

    project_by_slug = project_info.drop_duplicates("project_slug").set_index("project_slug", drop=False)

    missing = sorted(set(fix_info["project_slug"]) - set(project_by_slug.index))
    if missing:
        print(f"[WARN] {len(missing)} project_slug có trong fix_info nhưng không có trong project_info")
        for x in missing[:20]:
            print("  -", x)

    grouped = []
    for _, g in fix_info.groupby(fix_info.apply(build_unit_key, axis=1)):
        project_slug = g.iloc[0]["project_slug"]

        if project_slug not in project_by_slug.index:
            continue

        project_row = project_by_slug.loc[project_slug]
        order = commit_order_map(project_row)

        g = g.copy()
        g["_rank"] = g["commit"].map(lambda c: order.get(str(c).strip(), -1))
        g["_has_method"] = ~(g["method"].isna() | g["method_start"].isna() | g["method_end"].isna())
        g = g.sort_values(["_rank", "_has_method", "class_end", "method_end"],
                          ascending=[False, False, False, False])

        chosen = g.iloc[0].copy()
        chosen["all_touching_commits"] = ";".join(sorted({str(x).strip() for x in g["commit"].tolist() if str(x).strip()}))
        chosen["num_rows_merged"] = int(len(g))
        grouped.append(chosen)

    return pd.DataFrame(grouped).reset_index(drop=True)


def extract_unit_text(text: str, row: pd.Series, snapshot_type: str) -> Optional[Tuple[int, int, str, str]]:
    method_hint = None if pd.isna(row.get("method_start")) else int(row.get("method_start"))
    class_hint = None if pd.isna(row.get("class_start")) else int(row.get("class_start"))
    block = None
    kind = unit_kind(row)

    if kind == "method":
        block = extract_method_block(
            text=text,
            signature=normalize_text(row.get("signature")),
            fallback_method=normalize_text(row.get("method")),
            hint_line=method_hint,
        )
        if block is None and method_hint is not None and not pd.isna(row.get("method_end")):
            block = slice_by_lines(text, int(row.get("method_start")), int(row.get("method_end")))
    if block is None:
        block = extract_class_block(text, normalize_text(row.get("class")), class_hint)
        kind = "class"
    if block is None and class_hint is not None and not pd.isna(row.get("class_end")):
        block = slice_by_lines(text, int(row.get("class_start")), int(row.get("class_end")))
        kind = "class"
    if block is None:
        return None
    start_line, end_line, code = block
    return start_line, end_line, code, kind


def project_stratified_split(projects_df: pd.DataFrame, seed: int, train_ratio: float, valid_ratio: float, test_ratio: float) -> Dict[str, str]:
    if not math.isclose(train_ratio + valid_ratio + test_ratio, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("train/valid/test ratios must sum to 1")

    rng = random.Random(seed)
    split_map: Dict[str, str] = {}

    for cwe_id, group in projects_df.groupby("cwe_id"):
        slugs = group["project_slug"].tolist()
        rng.shuffle(slugs)
        n = len(slugs)
        n_test = max(1, round(n * test_ratio)) if n >= 3 else max(0, int(n >= 2))
        n_valid = max(1, round(n * valid_ratio)) if n - n_test >= 2 else max(0, int(n - n_test >= 2))
        if n_test + n_valid >= n:
            if n >= 3:
                n_test = 1
                n_valid = 1
            elif n == 2:
                n_test = 1
                n_valid = 0
            else:
                n_test = 0
                n_valid = 0
        test_slugs = set(slugs[:n_test])
        valid_slugs = set(slugs[n_test : n_test + n_valid])
        for slug in slugs:
            if slug in test_slugs:
                split_map[slug] = "test"
            elif slug in valid_slugs:
                split_map[slug] = "valid"
            else:
                split_map[slug] = "train"

    return split_map


def terminal_progress(processed: int, total: int, kept: int, filtered: int, interval: int, force: bool = False) -> None:
    if total <= 0:
        return
    if not force and processed % interval != 0 and processed != total:
        return
    pct = 100.0 * processed / total
    print(f"[Phase B improved] {processed}/{total} ({pct:.1f}%) | kept_units={kept} | filtered={filtered}", flush=True)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    ensure_dir(args.out_dir)
    ensure_dir(args.cache_dir)

    raw_data_dir = args.raw_root / "raw_data"
    project_info = pd.read_csv(raw_data_dir / "project_info.csv")
    fix_info = pd.read_csv(raw_data_dir / "fix_info.csv")
    project_info = project_info.set_index("project_slug", drop=False)

    filtered_log_path = args.out_dir / "filtered_samples_improved.jsonl"
    if filtered_log_path.exists():
        filtered_log_path.unlink()

    fix_info = choose_representative_rows(project_info.reset_index(drop=True), fix_info)

    fetcher = GitHubRawCache(args.cache_dir, timeout=args.timeout)
    total_rows = int(len(fix_info))
    progress_interval = max(1, total_rows // 20)
    kept_units = 0
    filtered = 0
    records: List[Dict] = []

    def log_filtered(payload: Dict) -> None:
        nonlocal filtered
        filtered += 1
        append_jsonl(filtered_log_path, payload)

    for row_idx, row in fix_info.iterrows():
        project_slug = normalize_text(row.get("project_slug"))
        owner = normalize_text(row.get("github_username"))
        repo = normalize_text(row.get("github_repository_name"))
        file_path = normalize_text(row.get("file"))
        project_row = project_info.loc[project_slug] if project_slug in project_info.index else None

        if project_row is None:
            log_filtered({"reason": "missing_project_row", "project_slug": project_slug, "row_idx": int(row_idx)})
            terminal_progress(row_idx + 1, total_rows, kept_units, filtered, progress_interval)
            continue

        buggy_commit = normalize_text(project_row.get("buggy_commit_id"))
        fixed_commit = normalize_text(row.get("commit"))
        if not buggy_commit or not fixed_commit:
            log_filtered({
                "reason": "missing_buggy_or_fixed_commit",
                "project_slug": project_slug,
                "buggy_commit": buggy_commit,
                "fixed_commit": fixed_commit,
                "row_idx": int(row_idx),
            })
            terminal_progress(row_idx + 1, total_rows, kept_units, filtered, progress_interval)
            continue

        buggy_text = fetcher.fetch(owner, repo, buggy_commit, file_path)
        fixed_text = fetcher.fetch(owner, repo, fixed_commit, file_path)
        if not buggy_text or not fixed_text:
            log_filtered({
                "reason": "missing_buggy_or_fixed_file",
                "project_slug": project_slug,
                "row_idx": int(row_idx),
                "buggy_ok": bool(buggy_text),
                "fixed_ok": bool(fixed_text),
                "file": file_path,
            })
            terminal_progress(row_idx + 1, total_rows, kept_units, filtered, progress_interval)
            continue

        buggy_extract = extract_unit_text(buggy_text, row, snapshot_type="buggy")
        fixed_extract = extract_unit_text(fixed_text, row, snapshot_type="fixed")
        if buggy_extract is None or fixed_extract is None:
            log_filtered({
                "reason": "cannot_extract_unit",
                "project_slug": project_slug,
                "row_idx": int(row_idx),
                "file": file_path,
                "signature": normalize_text(row.get("signature")),
                "class": normalize_text(row.get("class")),
                "method": normalize_text(row.get("method")),
            })
            terminal_progress(row_idx + 1, total_rows, kept_units, filtered, progress_interval)
            continue

        bug_start, bug_end, buggy_code, actual_kind_bug = buggy_extract
        fix_start, fix_end, fixed_code, actual_kind_fix = fixed_extract
        actual_kind = "method" if actual_kind_bug == actual_kind_fix == "method" else "class"

        buggy_lines = len(buggy_code.splitlines())
        fixed_lines = len(fixed_code.splitlines())
        if min(buggy_lines, fixed_lines) < args.min_lines or max(buggy_lines, fixed_lines) > args.max_lines:
            log_filtered({
                "reason": "unit_lines_out_of_range",
                "project_slug": project_slug,
                "row_idx": int(row_idx),
                "buggy_lines": buggy_lines,
                "fixed_lines": fixed_lines,
                "kind": actual_kind,
            })
            terminal_progress(row_idx + 1, total_rows, kept_units, filtered, progress_interval)
            continue

        if buggy_code.strip() == fixed_code.strip():
            log_filtered({
                "reason": "no_change_after_extraction",
                "project_slug": project_slug,
                "row_idx": int(row_idx),
                "file": file_path,
                "signature": normalize_text(row.get("signature")),
            })
            terminal_progress(row_idx + 1, total_rows, kept_units, filtered, progress_interval)
            continue

        buggy_focus, fixed_focus, diff_meta = trim_changed_region(buggy_code, fixed_code, context=args.focus_context)
        buggy_model_code = select_model_code(buggy_code, buggy_focus)
        fixed_model_code = select_model_code(fixed_code, fixed_focus)

        pair_id = stable_id([project_slug, file_path, normalize_text(row.get("signature")) or normalize_text(row.get("class"))])
        base = {
            "pair_id": pair_id,
            "project_slug": project_slug,
            "cve_id": normalize_text(row.get("cve_id")),
            "cwe_id": normalize_text(project_row.get("cwe_id")),
            "github_username": owner,
            "github_repository_name": repo,
            "file": file_path,
            "class": normalize_text(row.get("class")),
            "method": normalize_text(row.get("method")),
            "signature": normalize_text(row.get("signature")),
            "buggy_commit": buggy_commit,
            "fixed_commit": fixed_commit,
            "all_touching_commits": normalize_text(row.get("all_touching_commits")),
            "num_rows_merged": int(row.get("num_rows_merged", 1)),
            "unit_kind": actual_kind,
            "buggy_start_line": int(bug_start),
            "buggy_end_line": int(bug_end),
            "fixed_start_line": int(fix_start),
            "fixed_end_line": int(fix_end),
            "buggy_full_code": buggy_code,
            "fixed_full_code": fixed_code,
            "buggy_focus_code": buggy_focus,
            "fixed_focus_code": fixed_focus,
            "changed_ops": int(diff_meta.get("changed_ops", 0)),
            "source": "CWE-Bench-Java(project_info.buggy_commit_id + fix_info.commit)",
        }

        records.append({**base, "sample_id": f"{pair_id}-vuln", "label": 1, "code": buggy_model_code})
        records.append({**base, "sample_id": f"{pair_id}-fixed", "label": 0, "code": fixed_model_code})
        kept_units += 1
        terminal_progress(row_idx + 1, total_rows, kept_units, filtered, progress_interval)

    terminal_progress(total_rows, total_rows, kept_units, filtered, progress_interval, force=True)
    time.sleep(0.01)

    if not records:
        raise RuntimeError("No valid records produced in Phase B improved.")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

    projects_df = df[["project_slug", "cwe_id"]].drop_duplicates().reset_index(drop=True)
    split_map = project_stratified_split(projects_df, seed=args.seed, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio, test_ratio=args.test_ratio)
    df["split"] = df["project_slug"].map(split_map)

    ordered_cols = [
        "sample_id",
        "pair_id",
        "label",
        "split",
        "project_slug",
        "cve_id",
        "cwe_id",
        "github_username",
        "github_repository_name",
        "file",
        "class",
        "method",
        "signature",
        "unit_kind",
        "buggy_commit",
        "fixed_commit",
        "all_touching_commits",
        "num_rows_merged",
        "buggy_start_line",
        "buggy_end_line",
        "fixed_start_line",
        "fixed_end_line",
        "changed_ops",
        "code",
        "buggy_focus_code",
        "fixed_focus_code",
        "buggy_full_code",
        "fixed_full_code",
        "source",
    ]
    df = df[ordered_cols]

    write_jsonl(args.out_dir / "function_dataset_improved.jsonl", df.to_dict(orient="records"))
    for split_name in ["train", "valid", "test"]:
        split_df = df[df["split"] == split_name].copy()
        write_jsonl(args.out_dir / f"{split_name}.jsonl", split_df.to_dict(orient="records"))

    summary = {
        "num_samples": int(len(df)),
        "num_pairs": int(df["pair_id"].nunique()),
        "num_projects": int(df["project_slug"].nunique()),
        "num_cves": int(df["cve_id"].nunique()),
        "num_cwes": int(df["cwe_id"].nunique()),
        "label_distribution": df["label"].value_counts().sort_index().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "split_projects": df.groupby("split")["project_slug"].nunique().to_dict(),
        "split_cwes": df.groupby("split")["cwe_id"].nunique().to_dict(),
        "unit_kind_distribution": df["unit_kind"].value_counts().to_dict(),
        "avg_lines_by_label": {
            "vuln": float(df[df["label"] == 1]["code"].map(lambda s: len(str(s).splitlines())).mean()),
            "fixed": float(df[df["label"] == 0]["code"].map(lambda s: len(str(s).splitlines())).mean()),
        },
    }
    with (args.out_dir / "summary_improved.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Phase B (improved) done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
