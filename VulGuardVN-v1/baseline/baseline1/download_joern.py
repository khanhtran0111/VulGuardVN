import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from graphs import JOERN_INSTALL_ROOT, default_joern_install_dir, ensure_dir, resolve_joern_command


LATEST_RELEASE_API = "https://api.github.com/repos/joernio/joern/releases/latest"
FALLBACK_ARCHIVE_URL = "https://github.com/joernio/joern/releases/latest/download/joern-cli.zip"
USER_AGENT = "VulGuardVN-GRACE-Baseline/1.0"
ARCHIVE_NAME = "joern-cli.zip"


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def _load_latest_release_metadata() -> dict:
    request = urllib.request.Request(
        LATEST_RELEASE_API,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _resolve_archive_info() -> tuple[str, str]:
    try:
        release = _load_latest_release_metadata()
        tag_name = str(release.get("tag_name") or "").strip() or "latest"
        for asset in release.get("assets", []):
            if str(asset.get("name", "")).strip().lower() == ARCHIVE_NAME:
                return tag_name, str(asset["browser_download_url"])
        return tag_name, FALLBACK_ARCHIVE_URL
    except Exception:
        return "latest", FALLBACK_ARCHIVE_URL


def _download_file(url: str, destination: Path) -> None:
    headers = {"User-Agent": USER_AGENT}
    mode = "wb"
    if destination.exists():
        existing_size = destination.stat().st_size
        if existing_size > 0:
            headers["Range"] = f"bytes={existing_size}-"
            mode = "ab"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=600) as response:
        if mode == "ab" and getattr(response, "status", None) != 206:
            mode = "wb"
        with destination.open(mode) as handle:
            shutil.copyfileobj(response, handle)


def _extract_archive(archive_path: Path, destination_dir: Path) -> Path:
    ensure_dir(destination_dir)
    expected_root = destination_dir / "joern-cli"
    if expected_root.exists():
        shutil.rmtree(expected_root, ignore_errors=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(destination_dir)
    return expected_root if expected_root.exists() else destination_dir


def _discover_executable(root_dir: Path, base_name: str) -> Path | None:
    for suffix in ["", ".bat", ".cmd", ".exe"]:
        matches = list(root_dir.rglob(base_name + suffix))
        if matches:
            return matches[0]
    return None


def main() -> None:
    tag_name, archive_url = _resolve_archive_info()
    install_root = JOERN_INSTALL_ROOT
    ensure_dir(install_root)
    archive_path = install_root.parent / ARCHIVE_NAME
    if not archive_path.exists() or archive_path.stat().st_size == 0:
        partial_archive = archive_path.with_suffix(archive_path.suffix + ".part")
        _download_file(archive_url, partial_archive)
        if archive_path.exists():
            archive_path.unlink()
        partial_archive.replace(archive_path)
    install_dir = _extract_archive(archive_path, install_root)
    joern_parse = _discover_executable(install_dir, "joern-parse")
    joern_export = _discover_executable(install_dir, "joern-export")
    summary = {
        "release": tag_name,
        "archive_url": archive_url,
        "archive_path": str(archive_path),
        "install_dir": str(install_dir),
        "default_joern_home": str(default_joern_install_dir()),
        "joern_parse": str(joern_parse) if joern_parse else None,
        "joern_export": str(joern_export) if joern_export else None,
        "resolver_joern_parse": resolve_joern_command("joern-parse"),
        "resolver_joern_export": resolve_joern_command("joern-export"),
        "size_gb": round(_directory_size_bytes(install_root) / (1024**3), 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
