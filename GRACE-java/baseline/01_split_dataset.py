from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import random
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

    if method_start and method_end and method_end >= method_start:
        start, end = method_start, method_end
    elif class_start and class_end and class_end >= class_start:
        start, end = class_start, class_end
    else:
        start, end = 1, min(300, len(lines))

    start_idx = max(1, start) - 1
    end_idx = min(len(lines), end)
    return "\n".join(lines[start_idx:end_idx]).strip()


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
        "skipped_unknown_label": 0,
        "skipped_fetch_fail": 0,
        "skipped_empty_code": 0,
    }

    for idx, fr in enumerate(fix_rows):
        slug = (fr.get("project_slug") or "").strip()
        meta = project_meta.get(slug)
        if meta is None:
            stats["skipped_no_meta"] += 1
            continue

        commit_id = (fr.get("commit") or "").strip()
        buggy_commit = meta["buggy_commit_id"]
        fix_commits: set[str] = meta["fix_commit_ids"]
        if commit_id == buggy_commit:
            label = 1
        elif commit_id in fix_commits:
            label = 0
        else:
            stats["skipped_unknown_label"] += 1
            continue

        github_username = (fr.get("github_username") or "").strip()
        github_repository_name = (fr.get("github_repository_name") or "").strip()
        file_path = (fr.get("file") or "").strip()
        if not github_username or not github_repository_name or not file_path:
            stats["skipped_fetch_fail"] += 1
            continue

        full_text = fetch_raw_file_text(github_username, github_repository_name, commit_id, file_path, cache_dir)
        if full_text is None:
            stats["skipped_fetch_fail"] += 1
            continue

        code = extract_code_snippet(fr, full_text)
        if not code:
            stats["skipped_empty_code"] += 1
            continue

        row = {
            "sample_id": f"{slug}__{commit_id[:12]}__{idx}",
            "project": slug,
            "label": label,
            "cwe": (fr.get("cwe_id") or meta["cwe_id"] or "").strip(),
            "code": code,
            "file_path": file_path,
            "method_name": (fr.get("method") or "").strip(),
            "signature": (fr.get("signature") or "").strip(),
            "commit_id": commit_id,
            "project_slug": slug,
            "cve_id": (fr.get("cve_id") or "").strip(),
            "source_dataset": "CWE-Bench-Java",
        }
        out_rows.append(row)

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
