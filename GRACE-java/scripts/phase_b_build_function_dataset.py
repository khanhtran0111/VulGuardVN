from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit

from pipeline_common import append_log, ensure_dir, set_seed, stable_id, write_jsonl


RAW_GITHUB_URL = "https://raw.githubusercontent.com/{owner}/{repo}/{commit}/{path}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase B: build function-level dataset")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/CWE-Bench-Java"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/dataset"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--valid-size", type=float, default=0.1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--min-loc", type=int, default=3)
    parser.add_argument("--max-loc", type=int, default=400)
    return parser.parse_args()


def fetch_file_text(owner: str, repo: str, commit: str, file_path: str, timeout: float) -> Optional[str]:
    url = RAW_GITHUB_URL.format(owner=owner, repo=repo, commit=commit, path=file_path)
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            return None
        return response.text
    except requests.RequestException:
        return None


def slice_by_lines(text: str, start: int, end: int) -> Optional[str]:
    if start <= 0 or end <= 0 or end < start:
        return None
    lines = text.splitlines()
    if end > len(lines):
        return None
    snippet = "\n".join(lines[start - 1 : end]).strip()
    return snippet if snippet else None


def choose_fixed_commit(project_row: pd.Series) -> Optional[str]:
    value = str(project_row.get("fix_commit_ids", "")).strip()
    if not value:
        return None
    parts = [x.strip() for x in value.split(";") if x.strip()]
    return parts[0] if parts else None


def split_by_project(df: pd.DataFrame, seed: int, test_size: float, valid_size: float) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Dataset is empty, cannot split.")

    unique_projects = df[["project_slug"]].drop_duplicates()
    projects = unique_projects["project_slug"].values

    if len(projects) < 3:
        raise ValueError("Need at least 3 projects for train/valid/test split.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_train_valid, idx_test = next(splitter.split(projects, groups=projects))
    projects_train_valid = set(projects[idx_train_valid].tolist())
    projects_test = set(projects[idx_test].tolist())

    train_valid_df = df[df["project_slug"].isin(projects_train_valid)].copy()

    valid_ratio_within_tv = valid_size / (1.0 - test_size)
    splitter_valid = GroupShuffleSplit(
        n_splits=1, test_size=valid_ratio_within_tv, random_state=seed + 1
    )
    tv_projects = train_valid_df[["project_slug"]].drop_duplicates()["project_slug"].values
    idx_train, idx_valid = next(splitter_valid.split(tv_projects, groups=tv_projects))
    projects_train = set(tv_projects[idx_train].tolist())
    projects_valid = set(tv_projects[idx_valid].tolist())

    def split_name(project_slug: str) -> str:
        if project_slug in projects_test:
            return "test"
        if project_slug in projects_valid:
            return "valid"
        if project_slug in projects_train:
            return "train"
        raise RuntimeError("Unknown project in split assignment")

    df["split"] = df["project_slug"].map(split_name)
    return df


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    raw_data_dir = args.raw_root / "raw_data"
    project_info = pd.read_csv(raw_data_dir / "project_info.csv")
    fix_info = pd.read_csv(raw_data_dir / "fix_info.csv")

    project_info = project_info.set_index("project_slug", drop=False)

    filtered_log_path = args.out_dir / "filtered_samples.jsonl"
    if filtered_log_path.exists():
        filtered_log_path.unlink()

    records: List[Dict] = []

    for row_idx, row in fix_info.iterrows():
        project_slug = str(row["project_slug"])
        owner = str(row["github_username"])
        repo = str(row["github_repository_name"])
        file_path = str(row["file"])
        signature = str(row.get("signature", ""))

        method_name = row.get("method")
        method_start = row.get("method_start")
        method_end = row.get("method_end")

        if pd.isna(method_name) or pd.isna(method_start) or pd.isna(method_end):
            append_log(
                filtered_log_path,
                {
                    "reason": "missing_method_metadata",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                    "file": file_path,
                },
            )
            continue

        method_start = int(method_start)
        method_end = int(method_end)
        loc = method_end - method_start + 1
        if loc < args.min_loc or loc > args.max_loc:
            append_log(
                filtered_log_path,
                {
                    "reason": "loc_out_of_range",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                    "loc": int(loc),
                },
            )
            continue

        project_row = project_info.loc[project_slug] if project_slug in project_info.index else None
        if project_row is None:
            append_log(
                filtered_log_path,
                {
                    "reason": "missing_project_row",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                },
            )
            continue

        buggy_commit = str(row["commit"])
        fixed_commit = choose_fixed_commit(project_row)
        if not fixed_commit:
            append_log(
                filtered_log_path,
                {
                    "reason": "missing_fixed_commit",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                },
            )
            continue

        buggy_text = fetch_file_text(owner, repo, buggy_commit, file_path, timeout=args.timeout)
        if not buggy_text:
            append_log(
                filtered_log_path,
                {
                    "reason": "missing_buggy_file",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                    "commit": buggy_commit,
                    "file": file_path,
                },
            )
            continue

        fixed_text = fetch_file_text(owner, repo, fixed_commit, file_path, timeout=args.timeout)
        if not fixed_text:
            append_log(
                filtered_log_path,
                {
                    "reason": "missing_fixed_file",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                    "commit": fixed_commit,
                    "file": file_path,
                },
            )
            continue

        buggy_method = slice_by_lines(buggy_text, method_start, method_end)
        fixed_method = slice_by_lines(fixed_text, method_start, method_end)
        if not buggy_method or not fixed_method:
            append_log(
                filtered_log_path,
                {
                    "reason": "cannot_slice_method",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                    "file": file_path,
                    "method_start": method_start,
                    "method_end": method_end,
                },
            )
            continue

        if buggy_method == fixed_method:
            append_log(
                filtered_log_path,
                {
                    "reason": "no_change_after_fix",
                    "row_idx": int(row_idx),
                    "project_slug": project_slug,
                    "file": file_path,
                },
            )
            continue

        pair_id = stable_id(
            [project_slug, str(row.get("cve_id", "")), file_path, str(method_name), signature, str(row_idx)]
        )

        base_meta = {
            "pair_id": pair_id,
            "project_slug": project_slug,
            "cve_id": str(row.get("cve_id", "")),
            "cwe_id": str(project_row.get("cwe_id", "")),
            "signature": signature,
            "file": file_path,
            "method": str(method_name),
            "method_start": method_start,
            "method_end": method_end,
            "buggy_commit": buggy_commit,
            "fixed_commit": fixed_commit,
            "github_username": owner,
            "github_repository_name": repo,
            "source": "fix_info+github_raw",
        }

        records.append({**base_meta, "sample_id": f"{pair_id}-vuln", "label": 1, "code": buggy_method})
        records.append({**base_meta, "sample_id": f"{pair_id}-fixed", "label": 0, "code": fixed_method})

        if row_idx % 100 == 0 and row_idx > 0:
            print(f"Processed {row_idx} fix rows...")
            time.sleep(0.01)

    if not records:
        raise RuntimeError("No valid records produced in Phase B.")

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["project_slug", "signature", "label", "code"]).reset_index(drop=True)
    df = split_by_project(df, seed=args.seed, test_size=args.test_size, valid_size=args.valid_size)

    all_path = args.out_dir / "function_dataset.jsonl"
    write_jsonl(all_path, df.to_dict(orient="records"))

    for split_name in ["train", "valid", "test"]:
        split_df = df[df["split"] == split_name].copy()
        write_jsonl(args.out_dir / f"{split_name}.jsonl", split_df.to_dict(orient="records"))

    summary = {
        "num_samples": int(len(df)),
        "num_pairs": int(df["pair_id"].nunique()),
        "num_projects": int(df["project_slug"].nunique()),
        "num_cves": int(df["cve_id"].nunique()),
        "label_distribution": {
            "vulnerable_1": int((df["label"] == 1).sum()),
            "non_vulnerable_0": int((df["label"] == 0).sum()),
        },
        "split_counts": df["split"].value_counts().to_dict(),
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Phase B done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
