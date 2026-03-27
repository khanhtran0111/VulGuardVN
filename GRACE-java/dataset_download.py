#!/usr/bin/env python3
# download_cwe_bench_java.py

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests


GITHUB_ZIP_URL = "https://github.com/iris-sast/cwe-bench-java/archive/refs/heads/master.zip"


def download_file(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0

        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                if total > 0:
                    pct = downloaded * 100 / total
                    print(f"\rDownloading: {downloaded / 1e6:.1f}MB / {total / 1e6:.1f}MB ({pct:.1f}%)", end="")
                else:
                    print(f"\rDownloading: {downloaded / 1e6:.1f}MB", end="")

    print()


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # GitHub zip thường giải nén thành thư mục cwe-bench-java-master
    extracted_dirs = [p for p in extract_to.iterdir() if p.is_dir()]
    if len(extracted_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly 1 extracted directory, found {len(extracted_dirs)}: {extracted_dirs}"
        )

    return extracted_dirs[0]


def validate_dataset(repo_dir: Path) -> None:
    required_paths = [
        repo_dir / "data" / "project_info.csv",
        repo_dir / "data" / "build_info.csv",
        repo_dir / "data" / "fix_info.csv",
    ]

    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Dataset downloaded but missing expected files:\n" + "\n".join(missing)
        )

    print("\nFound required files:")
    for p in required_paths:
        print(f"  - {p}")

    advisory_dir = repo_dir / "advisory"
    patches_dir = repo_dir / "patches"

    if advisory_dir.exists():
        print(f"  - advisory/: {len(list(advisory_dir.glob('*.json')))} json files")
    if patches_dir.exists():
        print(f"  - patches/: {len(list(patches_dir.glob('*.patch')))} patch files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CWE-Bench-Java from GitHub")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets") / "cwe_bench_java",
        help="Directory to store the downloaded dataset",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip file after extraction",
    )
    args = parser.parse_args()

    output_dir: Path = args.output_dir.resolve()
    tmp_dir = output_dir.parent / f"{output_dir.name}_tmp"
    zip_path = tmp_dir / "cwe_bench_java_master.zip"

    if output_dir.exists():
        print(f"[INFO] Output directory already exists: {output_dir}")
        print("[INFO] Remove it first if you want a clean re-download.")
        sys.exit(0)

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[INFO] Downloading CWE-Bench-Java to: {output_dir}")
        download_file(GITHUB_ZIP_URL, zip_path)

        print("[INFO] Extracting zip...")
        extracted_root = extract_zip(zip_path, tmp_dir)

        print("[INFO] Moving extracted repo to final location...")
        shutil.move(str(extracted_root), str(output_dir))

        validate_dataset(output_dir)

        print("\n[SUCCESS] CWE-Bench-Java downloaded successfully.")
        print(f"[PATH] {output_dir}")

    finally:
        if not args.keep_zip:
            if zip_path.exists():
                zip_path.unlink(missing_ok=True)
        # dọn tmp dir nếu rỗng
        if tmp_dir.exists():
            try:
                tmp_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()