from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

EXPECTED_PROJECT_COLUMNS = {
    "project_slug",
    "cve_id",
    "cwe_id",
    "github_username",
    "github_repository_name",
    "buggy_commit_id",
    "fix_commit_ids",
}

EXPECTED_FIX_COLUMNS = {
    "project_slug",
    "cve_id",
    "github_username",
    "github_repository_name",
    "commit",
    "file",
    "class",
    "class_start",
    "class_end",
    "method",
    "method_start",
    "method_end",
    "signature",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A (improved): inspect CWE-Bench-Java raw data")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/CWE-Bench-Java"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/phase_a"))
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_fix_commits(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    return len([x for x in text.split(";") if x.strip()])


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def main() -> None:
    args = parse_args()
    raw_data_dir = args.raw_root / "raw_data"
    out_dir = args.out_dir
    ensure_dir(out_dir)

    project_info = pd.read_csv(raw_data_dir / "project_info.csv")
    fix_info = pd.read_csv(raw_data_dir / "fix_info.csv")
    build_info_path = raw_data_dir / "build_info.csv"
    build_info = pd.read_csv(build_info_path) if build_info_path.exists() else None

    project_missing = sorted(EXPECTED_PROJECT_COLUMNS - set(project_info.columns))
    fix_missing = sorted(EXPECTED_FIX_COLUMNS - set(fix_info.columns))

    project_info = project_info.copy()
    fix_info = fix_info.copy()

    project_info["num_fix_commits"] = project_info["fix_commit_ids"].map(count_fix_commits)
    fix_info["has_method"] = ~(fix_info["method"].isna() | fix_info["method_start"].isna() | fix_info["method_end"].isna())
    fix_info["has_class_only"] = (~fix_info["has_method"]) & (~fix_info["class"].isna())
    fix_info["unit_key"] = fix_info.apply(
        lambda r: "::".join(
            [
                normalize_text(r.get("project_slug")),
                normalize_text(r.get("file")),
                normalize_text(r.get("signature")) or f"CLASS::{normalize_text(r.get('class'))}",
            ]
        ),
        axis=1,
    )

    dup_units = (
        fix_info.groupby("unit_key").size().rename("rows").reset_index().sort_values("rows", ascending=False)
    )
    dup_units = dup_units[dup_units["rows"] > 1]

    cwe_project_distribution = (
        project_info.groupby("cwe_id")["project_slug"].nunique().sort_values(ascending=False).rename("num_projects")
    )
    cwe_fix_distribution = (
        fix_info.groupby(project_info.set_index("project_slug").reindex(fix_info["project_slug"]).reset_index(drop=True)["cwe_id"])
        .size()
        .sort_values(ascending=False)
        .rename("num_fix_rows")
    )

    summary = {
        "project_info_columns": project_info.columns.tolist(),
        "fix_info_columns": fix_info.columns.tolist(),
        "missing_project_columns": project_missing,
        "missing_fix_columns": fix_missing,
        "project_count": int(project_info["project_slug"].nunique()),
        "cve_count": int(project_info["cve_id"].nunique()),
        "cwe_count": int(project_info["cwe_id"].nunique()),
        "fix_rows": int(len(fix_info)),
        "fix_projects": int(fix_info["project_slug"].nunique()),
        "rows_with_method": int(fix_info["has_method"].sum()),
        "rows_class_only": int(fix_info["has_class_only"].sum()),
        "rows_missing_method_and_class": int((~fix_info["has_method"] & fix_info["class"].isna()).sum()),
        "projects_with_multi_fix_commits": int((project_info["num_fix_commits"] > 1).sum()),
        "max_fix_commits_for_a_project": int(project_info["num_fix_commits"].max()),
        "duplicated_units_gt1": int(len(dup_units)),
        "top_duplicated_unit_rows": int(dup_units["rows"].max()) if not dup_units.empty else 0,
    }

    if build_info is not None and "status" in build_info.columns:
        summary["build_success_projects"] = int((build_info["status"] == "success").sum())
        summary["build_failure_projects"] = int((build_info["status"] == "failure").sum())

    risks = []
    if summary["rows_with_method"] < 0.8 * summary["fix_rows"]:
        risks.append("Method-level metadata covers less than 80% of fix rows; class fallback is needed.")
    if summary["projects_with_multi_fix_commits"] > 0:
        risks.append("Many projects use multiple sequential fix commits; per-row commit handling must be careful.")
    if summary["duplicated_units_gt1"] > 0:
        risks.append("The same function/class appears in multiple fix rows; dedup/grouping is required before splitting.")
    if project_missing or fix_missing:
        risks.append("The CSV schema differs from the expected schema; extraction code should validate columns explicitly.")
    summary["risks"] = risks

    cwe_project_distribution.to_csv(out_dir / "cwe_project_distribution.csv", header=True)
    cwe_fix_distribution.to_csv(out_dir / "cwe_fix_distribution.csv", header=True)
    if not dup_units.empty:
        dup_units.head(500).to_csv(out_dir / "duplicated_units_top500.csv", index=False)

    project_audit = project_info[["project_slug", "cve_id", "cwe_id", "buggy_commit_id", "fix_commit_ids", "num_fix_commits"]].copy()
    project_audit.to_csv(out_dir / "project_audit.csv", index=False)

    fix_audit = fix_info[
        [
            "project_slug",
            "cve_id",
            "commit",
            "file",
            "class",
            "method",
            "method_start",
            "method_end",
            "signature",
            "has_method",
            "has_class_only",
            "unit_key",
        ]
    ].copy()
    fix_audit.to_csv(out_dir / "fix_audit.csv", index=False)

    with (out_dir / "raw_schema_summary_improved.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Phase A (improved) done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
