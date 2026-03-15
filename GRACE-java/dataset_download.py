from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "iris-sast/CWE-Bench-Java"
PROJECT_ROOT = Path(__file__).resolve().parent
SAVE_DIR = PROJECT_ROOT / "data" / "raw" / "CWE-Bench-Java"

def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=str(SAVE_DIR),
    )

    print("Tải xong dataset.")
    print(f"Repo: {REPO_ID}")
    print(f"Lưu tại: {local_path}")

if __name__ == "__main__":
    main()