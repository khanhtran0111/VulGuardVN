from pathlib import Path
import importlib.util
import re
import zipfile

import requests

from common import DATA_DIR, ensure_dir


FILE_ID = "1Mn0jLaZWiPFQ8ejzlz_zXnx_TcSzbwu1"
ARCHIVE_PATH = DATA_DIR / "reveal_replication.zip"
OUTPUT_DIR = DATA_DIR / "reveal_raw"
OVERWRITE = False
CHUNK_SIZE = 1024 * 1024
GOOGLE_DRIVE_URLS = [
    "https://drive.google.com/uc?export=download",
    "https://docs.google.com/uc?export=download",
]
HF_MIRROR_FILES = {
    "train-00000-of-00001.parquet": "https://huggingface.co/datasets/claudios/ReVeal/resolve/main/data/train-00000-of-00001.parquet?download=true",
    "validation-00000-of-00001.parquet": "https://huggingface.co/datasets/claudios/ReVeal/resolve/main/data/validation-00000-of-00001.parquet?download=true",
    "test-00000-of-00001.parquet": "https://huggingface.co/datasets/claudios/ReVeal/resolve/main/data/test-00000-of-00001.parquet?download=true",
}
MANUAL_LINKS = [
    "https://github.com/VulDetProject/ReVeal",
    "https://github.com/VulDetProject/ReVeal/blob/master/data/get_data.sh",
    f"https://drive.google.com/file/d/{FILE_ID}/view",
    "https://huggingface.co/datasets/claudios/ReVeal/tree/main/data",
]


def _extract_confirm_token(response_text: str, session: requests.Session) -> str | None:
    for cookie_name, cookie_value in session.cookies.items():
        if cookie_name.startswith("download_warning") and cookie_value:
            return cookie_value
    patterns = [
        r"confirm=([0-9A-Za-z_-]+)",
        r'name="confirm"\s+value="([0-9A-Za-z_-]+)"',
        r'"confirm"\s*:\s*"([0-9A-Za-z_-]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return match.group(1)
    return None


def _is_zip_response(response: requests.Response) -> bool:
    content_type = (response.headers.get("Content-Type") or "").lower()
    content_disposition = (response.headers.get("Content-Disposition") or "").lower()
    if "application/zip" in content_type or "application/octet-stream" in content_type:
        return True
    if ".zip" in content_disposition or "attachment" in content_disposition:
        return True
    return False


def _write_response_to_file(response: requests.Response, destination: Path) -> None:
    with destination.open("wb") as handle:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                handle.write(chunk)


def _download_via_gdown(file_id: str, destination: Path) -> bool:
    if not importlib.util.find_spec("gdown"):
        return False
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    result = gdown.download(url=url, output=str(destination), quiet=False, fuzzy=True)
    return bool(result and destination.exists() and destination.stat().st_size > 0)


def _download_file(url: str, destination: Path) -> None:
    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
        )
        response = session.get(url, stream=True, timeout=120, allow_redirects=True)
        response.raise_for_status()
        _write_response_to_file(response, destination)


def download_from_huggingface(output_dir: Path) -> None:
    ensure_dir(output_dir)
    for file_name, url in HF_MIRROR_FILES.items():
        destination = output_dir / file_name
        print(f"Downloading {file_name} from Hugging Face mirror")
        _download_file(url, destination)
        if not destination.exists() or destination.stat().st_size == 0:
            raise RuntimeError(f"Failed to download mirror file: {file_name}")


def download_from_google_drive(file_id: str, destination: Path) -> None:
    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            }
        )
        last_error = None
        for url in GOOGLE_DRIVE_URLS:
            try:
                response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
                if response.ok and _is_zip_response(response):
                    _write_response_to_file(response, destination)
                    return
                response_text = response.text if response.ok else ""
                token = _extract_confirm_token(response_text, session)
                response.close()
                if not token:
                    last_error = RuntimeError(f"No confirm token from {url}; status={response.status_code}")
                    continue
                confirmed = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
                if confirmed.ok and _is_zip_response(confirmed):
                    _write_response_to_file(confirmed, destination)
                    return
                confirmed_text = confirmed.text if confirmed.ok else ""
                confirmed.close()
                last_error = RuntimeError(
                    f"Google Drive did not return the archive from {url}. "
                    f"status={confirmed.status_code}, token_found={bool(token)}, "
                    f"body_hint={confirmed_text[:160]!r}"
                )
            except Exception as exc:
                last_error = exc
        if _download_via_gdown(file_id, destination):
            return
        manual_links = "\n".join(MANUAL_LINKS)
        raise RuntimeError(
            "Failed to download ReVeal automatically from Google Drive.\n"
            f"Last error: {last_error}\n"
            "You can open one of these official links and download the archive manually:\n"
            f"{manual_links}\n"
            f"Then put the zip at {destination} and rerun this script."
        )


def extract_archive(archive_path: Path, output_dir: Path) -> None:
    ensure_dir(output_dir)
    with zipfile.ZipFile(archive_path, "r") as zip_handle:
        zip_handle.extractall(output_dir)


def main() -> None:
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()) and not OVERWRITE:
        print(f"ReVeal data already exists at {OUTPUT_DIR}")
        return
    ensure_dir(DATA_DIR)
    drive_error = None
    try:
        if ARCHIVE_PATH.exists() and zipfile.is_zipfile(ARCHIVE_PATH):
            print(f"Using existing archive at {ARCHIVE_PATH}")
        else:
            if ARCHIVE_PATH.exists():
                ARCHIVE_PATH.unlink()
            print(f"Downloading ReVeal replication archive to {ARCHIVE_PATH}")
            download_from_google_drive(FILE_ID, ARCHIVE_PATH)
        if not zipfile.is_zipfile(ARCHIVE_PATH):
            raise RuntimeError(f"Downloaded file is not a valid zip archive: {ARCHIVE_PATH}")
        print(f"Extracting archive into {OUTPUT_DIR}")
        extract_archive(ARCHIVE_PATH, OUTPUT_DIR)
        if ARCHIVE_PATH.exists():
            ARCHIVE_PATH.unlink()
        print(f"Done. ReVeal raw files are under {OUTPUT_DIR}")
        return
    except Exception as exc:
        drive_error = exc
        if ARCHIVE_PATH.exists():
            ARCHIVE_PATH.unlink()
        print(f"Google Drive download failed: {exc}")
        print("Trying Hugging Face mirror instead")
    try:
        download_from_huggingface(OUTPUT_DIR)
        print(f"Done. ReVeal mirror files are under {OUTPUT_DIR}")
        return
    except Exception as mirror_error:
        raise RuntimeError(
            "Failed to download ReVeal from both Google Drive and Hugging Face mirror.\n"
            f"Google Drive error: {drive_error}\n"
            f"Hugging Face error: {mirror_error}\n"
            "You can open one of these official links and download the dataset manually:\n"
            + "\n".join(MANUAL_LINKS)
        ) from mirror_error


if __name__ == "__main__":
    main()
