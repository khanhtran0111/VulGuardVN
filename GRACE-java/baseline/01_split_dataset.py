from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import random
import re
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sklearn.model_selection import GroupShuffleSplit

from common import (
    DEFAULT_BENCHMARK_CANDIDATES,
    GRACE_JAVA_DIR,
    WORK_DIR,
    ensure_dir,
    load_json_or_jsonl,
    normalize_row_for_baseline,
    resolve_first_existing,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group-aware split for Java vulnerability benchmark.")
    p.add_argument("--input", default=None, help="Standardized benchmark JSON/JSONL.")
    p.add_argument(
        "--source",
        choices=["auto", "cwe-bench", "candidates"],
        default="auto",
        help="Input source policy. `auto` prefers CWE-Bench-Java CSV if available.",
    )
    p.add_argument("--out-dir", default=str(WORK_DIR / "splits"))
    p.add_argument("--group-field", default="project", help="Field used to avoid leakage across splits.")
    p.add_argument("--data-dir", default=str(GRACE_JAVA_DIR / "data" / "CWE-Bench-Java" / "data"))
    p.add_argument("--max-samples", type=int, default=0, help="Limit loaded samples for quick dry runs. 0 means all.")
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_line_int(value: str | None) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    if not text.isdigit():
        return None
    num = int(text)
    return num if num > 0 else None


def fetch_raw_file_text(
    github_username: str,
    github_repository_name: str,
    commit_id: str,
    file_path: str,
    cache_dir: Path,
) -> str | None:
    cache_file = cache_dir / github_username / github_repository_name / commit_id / file_path
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8", errors="ignore")

    url = f"https://raw.githubusercontent.com/{github_username}/{github_repository_name}/{commit_id}/{file_path}"
    req = Request(url, headers={"User-Agent": "VulGuardVN-Baseline/1.0"})
    try:
        with urlopen(req, timeout=25) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
    except (HTTPError, URLError, TimeoutError):
        return None

    ensure_dir(cache_file.parent)
    cache_file.write_text(content, encoding="utf-8")
    return content


def extract_code_snippet(row: dict[str, str], full_text: str) -> str:
    lines = full_text.splitlines()
    method_start = parse_line_int(row.get("method_start"))
    method_end = parse_line_int(row.get("method_end"))
    class_start = parse_line_int(row.get("class_start"))
    class_end = parse_line_int(row.get("class_end"))

    method_name = (row.get("method") or "").strip()
    if method_name:
        snippet = find_named_block(lines, method_name, kind="method", anchor_line=method_start)
        if snippet:
            return snippet
        snippet = extract_hint_block(lines, method_start, method_end, kind="method", name=method_name)
        if snippet:
            return snippet

    class_name = (row.get("class") or "").strip()
    if class_name:
        snippet = find_named_block(lines, class_name, kind="class", anchor_line=class_start)
        if snippet:
            return snippet
        snippet = extract_hint_block(lines, class_start, class_end, kind="class", name=class_name)
        if snippet:
            return snippet

    return "\n".join(lines[: min(300, len(lines))]).strip()


def _find_balanced_block(lines: list[str], start_idx: int, min_end_idx: int | None = None) -> tuple[int, int] | None:
    brace_depth = 0
    seen_open = False
    for end_idx in range(start_idx, len(lines)):
        line = lines[end_idx]
        brace_depth += line.count("{")
        if "{" in line:
            seen_open = True
        brace_depth -= line.count("}")
        if seen_open and brace_depth <= 0 and (min_end_idx is None or end_idx + 1 >= min_end_idx):
            return start_idx, end_idx + 1
    return None


def _pattern_for_block(name: str, kind: str) -> re.Pattern[str]:
    escaped = re.escape(name)
    if kind == "method":
        return re.compile(rf"\b{escaped}\s*\(")
    return re.compile(rf"\b(class|interface|enum|record)\s+{escaped}\b")


def extract_hint_block(
    lines: list[str],
    start: int | None,
    end: int | None,
    kind: str,
    name: str,
) -> str:
    if not start or not end or end < start:
        return ""

    start_idx = max(0, start - 1)
    min_end_idx = min(len(lines), end)
    pattern = _pattern_for_block(name, kind) if name else None

    if pattern is not None:
        search_start = max(0, start_idx - 8)
        search_end = min(len(lines), start_idx + 8)
        for idx in range(search_start, search_end):
            line = lines[idx]
            if pattern.search(line):
                if kind == "method" and line.strip().endswith(";"):
                    continue
                start_idx = idx
                break

    span = _find_balanced_block(lines, start_idx, min_end_idx=min_end_idx)
    if span is None:
        return ""
    block_start, block_end = span
    snippet = "\n".join(lines[block_start:block_end]).strip()
    return snippet


def find_named_block(lines: list[str], name: str, kind: str, anchor_line: int | None = None) -> str:
    if not name:
        return ""

    pattern = _pattern_for_block(name, kind)
    candidates: list[tuple[int, str]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not pattern.search(line):
            continue
        if kind == "method" and stripped.endswith(";"):
            continue
        span = _find_balanced_block(lines, idx)
        if span is None:
            continue
        start_idx, end_idx = span
        snippet = "\n".join(lines[start_idx:end_idx]).strip()
        if snippet:
            candidates.append((idx, snippet))

    if not candidates:
        return ""
    if anchor_line:
        anchor_idx = max(0, anchor_line - 1)
        candidates.sort(key=lambda item: abs(item[0] - anchor_idx))
        return candidates[0][1]
    return candidates[0][1]


def dedupe_key(row: dict[str, str], commit_id: str, label: int) -> tuple[str, ...]:
    return (
        (row.get("project_slug") or "").strip(),
        commit_id.strip(),
        str(label),
        (row.get("file") or "").strip(),
        (row.get("class") or "").strip(),
        (row.get("method") or "").strip(),
        (row.get("signature") or "").strip(),
    )


def build_variant_row(
    row: dict[str, str],
    meta: dict[str, str | set[str]],
    cache_dir: Path,
    idx: int,
    commit_id: str,
    label: int,
    variant: str,
    source_fix_commit_id: str,
) -> dict | None:
    github_username = (row.get("github_username") or "").strip()
    github_repository_name = (row.get("github_repository_name") or "").strip()
    file_path = (row.get("file") or "").strip()
    if not github_username or not github_repository_name or not file_path:
        return None

    full_text = fetch_raw_file_text(github_username, github_repository_name, commit_id, file_path, cache_dir)
    if full_text is None:
        return None

    code = extract_code_snippet(row, full_text)
    if not code:
        return None

    slug = (row.get("project_slug") or "").strip()
    return {
        "sample_id": f"{slug}__{variant}__{commit_id[:12]}__{idx}",
        "project": slug,
        "label": label,
        "variant": variant,
        "cwe": (row.get("cwe_id") or meta["cwe_id"] or "").strip(),
        "code": code,
        "file_path": file_path,
        "class_name": (row.get("class") or "").strip(),
        "method_name": (row.get("method") or "").strip(),
        "signature": (row.get("signature") or "").strip(),
        "commit_id": commit_id,
        "source_fix_commit_id": source_fix_commit_id,
        "project_slug": slug,
        "cve_id": (row.get("cve_id") or "").strip(),
        "source_dataset": "CWE-Bench-Java",
    }


def build_rows_from_cwe_bench_data(data_dir: Path, work_dir: Path, max_samples: int) -> tuple[list[dict], dict[str, int]]:
    fix_info_path = data_dir / "fix_info.csv"
    project_info_path = data_dir / "project_info.csv"
    if not fix_info_path.exists() or not project_info_path.exists():
        raise FileNotFoundError(f"Missing fix_info.csv/project_info.csv in {data_dir}")

    project_rows = read_csv_rows(project_info_path)
    fix_rows = read_csv_rows(fix_info_path)

    project_meta: dict[str, dict] = {}
    for pr in project_rows:
        slug = (pr.get("project_slug") or "").strip()
        if not slug:
            continue
        fix_commits = {x.strip() for x in (pr.get("fix_commit_ids") or "").split(";") if x.strip()}
        project_meta[slug] = {
            "buggy_commit_id": (pr.get("buggy_commit_id") or "").strip(),
            "fix_commit_ids": fix_commits,
            "cwe_id": (pr.get("cwe_id") or "").strip(),
        }

    cache_dir = ensure_dir(work_dir / "raw_cache")
    out_rows: list[dict] = []
    stats = {
        "total_fix_rows": len(fix_rows),
        "skipped_no_meta": 0,
        "skipped_fetch_fail": 0,
        "skipped_empty_code": 0,
        "skipped_missing_buggy_commit": 0,
        "skipped_duplicate": 0,
        "emitted_buggy": 0,
        "emitted_fixed": 0,
    }
    seen_keys: set[tuple[str, ...]] = set()

    for idx, fr in enumerate(fix_rows):
        slug = (fr.get("project_slug") or "").strip()
        meta = project_meta.get(slug)
        if meta is None:
            stats["skipped_no_meta"] += 1
            continue

        fixed_commit = (fr.get("commit") or "").strip()
        buggy_commit = str(meta["buggy_commit_id"]).strip()
        if not buggy_commit:
            stats["skipped_missing_buggy_commit"] += 1
            continue

        for variant, commit_id, label in [
            ("buggy", buggy_commit, 1),
            ("fixed", fixed_commit, 0),
        ]:
            key = dedupe_key(fr, commit_id, label)
            if key in seen_keys:
                stats["skipped_duplicate"] += 1
                continue

            row = build_variant_row(
                row=fr,
                meta=meta,
                cache_dir=cache_dir,
                idx=idx,
                commit_id=commit_id,
                label=label,
                variant=variant,
                source_fix_commit_id=fixed_commit,
            )
            if row is None:
                full_text = fetch_raw_file_text(
                    (fr.get("github_username") or "").strip(),
                    (fr.get("github_repository_name") or "").strip(),
                    commit_id,
                    (fr.get("file") or "").strip(),
                    cache_dir,
                )
                if full_text is None:
                    stats["skipped_fetch_fail"] += 1
                else:
                    stats["skipped_empty_code"] += 1
                continue

            out_rows.append(row)
            seen_keys.add(key)
            stats[f"emitted_{variant}"] += 1

            if max_samples > 0 and len(out_rows) >= max_samples:
                break
        if max_samples > 0 and len(out_rows) >= max_samples:
            break

    return out_rows, stats


def fallback_random_split(n_items: int, test_size: float, seed: int) -> tuple[list[int], list[int]]:
    """Fallback split by sample index when group-aware split is impossible."""
    if n_items <= 1:
        return list(range(n_items)), []

    test_count = int(round(n_items * test_size))
    test_count = max(1, test_count)
    test_count = min(test_count, n_items - 1)

    idxs = list(range(n_items))
    random.Random(seed).shuffle(idxs)
    test_idx = idxs[:test_count]
    train_idx = idxs[test_count:]
    return train_idx, test_idx


def safe_group_split(groups: list[str], test_size: float, seed: int) -> tuple[list[int], list[int], str]:
    """Try group-aware split first, then fallback to random split for tiny/single-group data."""
    idxs = list(range(len(groups)))
    unique_groups = len(set(groups))

    if len(idxs) <= 1:
        return idxs, [], "single-sample"

    if unique_groups <= 1:
        train_idx, test_idx = fallback_random_split(len(idxs), test_size, seed)
        return train_idx, test_idx, "fallback-random-single-group"

    try:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train_idx, test_idx = next(splitter.split(idxs, groups=groups))
        return list(train_idx), list(test_idx), "group"
    except ValueError:
        train_idx, test_idx = fallback_random_split(len(idxs), test_size, seed)
        return train_idx, test_idx, "fallback-random-error"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input) if args.input else None
    input_desc = ""

    data_dir = Path(args.data_dir)
    cwe_csv_ready = (data_dir / "fix_info.csv").exists() and (data_dir / "project_info.csv").exists()

    use_cwe_bench = False
    if args.input:
        use_cwe_bench = False
    elif args.source == "cwe-bench":
        use_cwe_bench = True
    elif args.source == "auto":
        use_cwe_bench = cwe_csv_ready
    elif args.source == "candidates":
        use_cwe_bench = False

    if not use_cwe_bench and input_path is None:
        input_path = resolve_first_existing(DEFAULT_BENCHMARK_CANDIDATES)

    if use_cwe_bench:
        raw_rows, cwe_stats = build_rows_from_cwe_bench_data(data_dir, WORK_DIR, args.max_samples)
        input_desc = str(data_dir)
        print(
            "cwe_bench_build_stats="
            f"{cwe_stats}"
        )
    elif input_path is not None and input_path.is_file():
        raw_rows = load_json_or_jsonl(input_path)
        input_desc = str(input_path)
    else:
        data_dir = Path(args.data_dir)
        raw_rows, cwe_stats = build_rows_from_cwe_bench_data(data_dir, WORK_DIR, args.max_samples)
        input_desc = str(data_dir)
        print(
            "cwe_bench_build_stats="
            f"{cwe_stats}"
        )

    rows = [normalize_row_for_baseline(row, idx) for idx, row in enumerate(raw_rows)]
    rows = [r for r in rows if r.get("code", "").strip()]
    if not rows:
        raise ValueError("Input dataset is empty or has no valid `code` field after normalization.")

    for idx, row in enumerate(rows):
        row.setdefault("sample_id", f"sample_{idx}")
        row.setdefault(args.group_field, row["sample_id"])

    groups = [str(r.get(args.group_field) or r["sample_id"]) for r in rows]
    train_val_idx, test_idx, outer_mode = safe_group_split(groups, args.test_size, args.seed)

    train_val_rows = [rows[i] for i in train_val_idx]
    test_rows = [rows[i] for i in test_idx]

    inner_groups = [str(r.get(args.group_field) or r["sample_id"]) for r in train_val_rows]
    adjusted_val_size = args.val_size / max(1e-8, 1.0 - args.test_size)
    train_idx_rel, val_idx_rel, inner_mode = safe_group_split(inner_groups, adjusted_val_size, args.seed)

    train_rows = [train_val_rows[i] for i in train_idx_rel]
    val_rows = [train_val_rows[i] for i in val_idx_rel]

    for row in train_rows:
        row["split"] = "train"
    for row in val_rows:
        row["split"] = "val"
    for row in test_rows:
        row["split"] = "test"

    out_dir = ensure_dir(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    write_jsonl(out_dir / "test.jsonl", test_rows)
    write_jsonl(out_dir / "all_with_split.jsonl", train_rows + val_rows + test_rows)

    for name, subset in [("train", train_rows), ("val", val_rows), ("test", test_rows)]:
        cnt = Counter(int(x.get("label", 0)) for x in subset)
        print(f"{name}: n={len(subset)} label_dist={dict(cnt)}")
    print(f"split_mode_outer={outer_mode}")
    print(f"split_mode_inner={inner_mode}")
    print(f"input={input_desc}")
    print(f"out_dir={out_dir}")


if __name__ == "__main__":
    main()
