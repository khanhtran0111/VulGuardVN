from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline_common import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase A: inspect CWE-Bench-Java raw data")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw/CWE-Bench-Java"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/artifacts/phase_a"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_data_dir = args.raw_root / "raw_data"
    advisory_dir = raw_data_dir / "advisory"

    ensure_dir(args.out_dir)

    project_info = pd.read_csv(raw_data_dir / "project_info.csv")
    fix_info = pd.read_csv(raw_data_dir / "fix_info.csv")
    build_info = pd.read_csv(raw_data_dir / "build_info.csv")

    advisory_files = sorted(advisory_dir.glob("*.json"))

    schema = {
        "project_info_columns": project_info.columns.tolist(),
        "fix_info_columns": fix_info.columns.tolist(),
        "build_info_columns": build_info.columns.tolist(),
        "project_count": int(project_info["project_slug"].nunique()),
        "cve_count": int(project_info["cve_id"].nunique()),
        "cwe_count": int(project_info["cwe_id"].nunique()),
        "fix_rows": int(len(fix_info)),
        "fix_projects": int(fix_info["project_slug"].nunique()),
        "advisory_json_count": len(advisory_files),
        "rows_missing_method": int(fix_info["method"].isna().sum()),
        "rows_missing_method_bounds": int(
            fix_info["method_start"].isna().sum() + fix_info["method_end"].isna().sum()
        ),
    }

    cwe_distribution = (
        project_info.groupby("cwe_id")["project_slug"].nunique().sort_values(ascending=False)
    )
    cwe_distribution.to_csv(args.out_dir / "cwe_distribution.csv", header=["num_projects"])

    fix_per_project = fix_info.groupby("project_slug").size().sort_values(ascending=False)
    fix_per_project.to_csv(args.out_dir / "fix_rows_per_project.csv", header=["num_fix_rows"])

    with (args.out_dir / "raw_schema_summary.json").open("w", encoding="utf-8") as f:
        json.dump(schema, f, ensure_ascii=False, indent=2)

    print("Phase A done.")
    print(f"Output dir: {args.out_dir}")
    print(f"Projects: {schema['project_count']}, CVEs: {schema['cve_count']}, CWEs: {schema['cwe_count']}")
    print(f"Fix rows: {schema['fix_rows']}, advisory json: {schema['advisory_json_count']}")


if __name__ == "__main__":
    main()
