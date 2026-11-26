import json
from pathlib import Path

from retrieval import DEFAULT_RETRIEVAL_MODEL_REPO_ID, default_retrieval_model_dir, download_retrieval_model_snapshot


MODEL_REPO_ID = DEFAULT_RETRIEVAL_MODEL_REPO_ID
MODEL_DIR = default_retrieval_model_dir(MODEL_REPO_ID)


def _directory_size_bytes(path: Path) -> int:
    total = 0
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total += file_path.stat().st_size
    return total


def main() -> None:
    model_dir = download_retrieval_model_snapshot(MODEL_REPO_ID, MODEL_DIR)
    summary = {
        "model_repo_id": MODEL_REPO_ID,
        "model_dir": str(model_dir),
        "size_gb": round(_directory_size_bytes(model_dir) / (1024**3), 3),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
