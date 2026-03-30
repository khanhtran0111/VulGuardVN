from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest
from urllib.parse import urlparse
from zipfile import ZipFile


DEFAULT_VERSION = "latest"
DEFAULT_REPO = "https://github.com/joernio/joern"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and prepare Joern CLI for GRACE-java baseline (Windows-friendly)."
    )
    p.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="Joern version tag without leading 'v' (or 'latest').",
    )
    p.add_argument(
        "--install-dir",
        default=str((Path(__file__).resolve().parent.parent / "tools" / "joern").resolve()),
        help="Installation directory. Default: GRACE-java/tools/joern",
    )
    p.add_argument(
        "--url",
        default=None,
        help="Optional direct URL to Joern zip. If omitted, uses GitHub release URL.",
    )
    p.add_argument("--force", action="store_true", help="Reinstall even if target already exists.")
    return p.parse_args()


def build_download_url(version: str) -> str:
    return f"{DEFAULT_REPO}/releases/download/v{version}/joern-cli.zip"


def resolve_latest_version() -> str:
    # GitHub redirects /releases/latest to /releases/tag/vX.Y.Z.
    req = urlrequest.Request(f"{DEFAULT_REPO}/releases/latest", method="GET")
    with urlrequest.urlopen(req) as resp:  # noqa: S310
        final_url = resp.geturl()

    path = urlparse(final_url).path
    match = re.search(r"/releases/tag/v([^/]+)$", path)
    if not match:
        raise RuntimeError(f"Could not resolve latest Joern version from URL: {final_url}")
    return match.group(1)


def download_file(url: str, dest: Path) -> None:
    print(f"Downloading: {url}")

    def _hook(blocks: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = blocks * block_size
        pct = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(f"\rProgress: {pct:6.2f}%")
        sys.stdout.flush()

    urlrequest.urlretrieve(url, dest, reporthook=_hook)  # noqa: S310
    sys.stdout.write("\n")


def find_joern_home(install_dir: Path) -> Path:
    # Preferred candidate
    direct = install_dir / "joern-cli"
    if direct.exists():
        return direct

    # Fallback: search for a directory that contains bin/joern-parse(.bat)
    for p in install_dir.rglob("*"):
        if not p.is_dir():
            continue
        if (p / "bin" / "joern-parse").exists() or (p / "bin" / "joern-parse.bat").exists():
            return p

    raise FileNotFoundError("Could not locate joern-cli after extraction.")


def verify_binaries(joern_home: Path) -> tuple[Path, Path]:
    bin_dir = joern_home / "bin"

    parse_candidates = [
        bin_dir / "joern-parse.bat",
        bin_dir / "joern-parse.cmd",
        bin_dir / "joern-parse.exe",
        bin_dir / "joern-parse",
    ]
    export_candidates = [
        bin_dir / "joern-export.bat",
        bin_dir / "joern-export.cmd",
        bin_dir / "joern-export.exe",
        bin_dir / "joern-export",
    ]

    joern_parse = next((p for p in parse_candidates if p.exists()), None)
    joern_export = next((p for p in export_candidates if p.exists()), None)

    if not joern_parse or not joern_export:
        raise FileNotFoundError(
            f"Joern binaries not found in {bin_dir}.\n"
            "Expected joern-parse and joern-export files in the bin directory."
        )

    return joern_parse, joern_export


def main() -> None:
    args = parse_args()
    install_dir = Path(args.install_dir).resolve()

    requested_version = args.version.strip().lower()
    if args.url:
        resolved_version = args.version
        url = args.url
    else:
        resolved_version = resolve_latest_version() if requested_version == "latest" else args.version
        url = build_download_url(resolved_version)

    if install_dir.exists() and args.force:
        print(f"Removing existing install dir: {install_dir}")
        shutil.rmtree(install_dir)

    install_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="joern_download_") as tmp:
        zip_path = Path(tmp) / "joern-cli.zip"
        try:
            download_file(url, zip_path)
        except urlerror.HTTPError as e:
            # Common case: pinned version no longer exists.
            if e.code != 404 or args.url:
                raise

            latest_version = resolve_latest_version()
            fallback_url = build_download_url(latest_version)
            print(
                f"Version '{args.version}' not found (404). "
                f"Falling back to latest: v{latest_version}"
            )
            resolved_version = latest_version
            download_file(fallback_url, zip_path)

        print(f"Extracting to: {install_dir}")
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(install_dir)

    joern_home = find_joern_home(install_dir)
    joern_parse, joern_export = verify_binaries(joern_home)

    print("\nSetup complete.")
    print(f"Version: v{resolved_version}")
    print(f"JOERN_HOME: {joern_home}")
    print(f"joern-parse: {joern_parse}")
    print(f"joern-export: {joern_export}")

    baseline_dir = Path(__file__).resolve().parent
    input_path = (baseline_dir.parent / "work" / "test_with_demo.jsonl").resolve()
    output_path = (baseline_dir.parent / "work" / "test_with_joern.jsonl").resolve()

    print("\nRun step 05 with explicit paths:")
    print(
        "python 05_run_joern_per_sample.py "
        f"--input \"{input_path}\" --output \"{output_path}\" "
        f"--joern-parse \"{joern_parse}\" --joern-export \"{joern_export}\""
    )

    print("\nOptional (PowerShell): set JOERN_HOME for future sessions")
    print(f"setx JOERN_HOME \"{joern_home}\"")


if __name__ == "__main__":
    main()
