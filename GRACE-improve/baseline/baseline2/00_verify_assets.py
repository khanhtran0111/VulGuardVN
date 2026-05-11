import json
import os

from common import SHARED_SPLITS_DIR
from graphs import resolve_graph_backend_with_notice
from local_llm_client import DEFAULT_MODEL_REPO_ID, default_local_model_dir, download_model_snapshot, is_model_downloaded
from retrieval import DEFAULT_RETRIEVAL_MODEL_REPO_ID, default_retrieval_model_dir, download_retrieval_model_snapshot, is_retrieval_model_downloaded


TARGET_DATASETS = [name.strip() for name in os.getenv("GRACE_DATASETS", os.getenv("GRACE_DATASET", "devign")).split(",") if name.strip()]
AUTO_DOWNLOAD = os.getenv("GRACE_AUTO_DOWNLOAD_MISSING", "1").strip().lower() in {"1", "true", "yes", "on"}
SEMANTIC_MODEL_ID = os.getenv("GRACE_RETRIEVAL_MODEL_ID", DEFAULT_RETRIEVAL_MODEL_REPO_ID)
LLM_MODEL_ID = os.getenv("GRACE_LOCAL_MODEL_ID", DEFAULT_MODEL_REPO_ID)


def _split_status(dataset_name: str) -> dict:
    split_dir = SHARED_SPLITS_DIR / dataset_name
    files = {name: split_dir / f"{name}.jsonl" for name in ["train", "val", "test"]}
    return {
        "split_dir": str(split_dir),
        "available": all(path.exists() for path in files.values()),
        "files": {name: str(path) for name, path in files.items()},
    }


def main() -> None:
    semantic_dir = default_retrieval_model_dir(SEMANTIC_MODEL_ID)
    llm_dir = default_local_model_dir(LLM_MODEL_ID)
    semantic_ready = is_retrieval_model_downloaded(semantic_dir)
    llm_ready = is_model_downloaded(llm_dir)

    actions = []
    if AUTO_DOWNLOAD and not semantic_ready:
        download_retrieval_model_snapshot(SEMANTIC_MODEL_ID, semantic_dir)
        semantic_ready = is_retrieval_model_downloaded(semantic_dir)
        actions.append(f"downloaded semantic model: {SEMANTIC_MODEL_ID}")
    if AUTO_DOWNLOAD and not llm_ready:
        download_model_snapshot(LLM_MODEL_ID, llm_dir)
        llm_ready = is_model_downloaded(llm_dir)
        actions.append(f"downloaded llm model: {LLM_MODEL_ID}")

    graph_backend, graph_notice = resolve_graph_backend_with_notice("auto")
    payload = {
        "datasets": TARGET_DATASETS,
        "auto_download_missing": AUTO_DOWNLOAD,
        "actions": actions,
        "splits": {dataset_name: _split_status(dataset_name) for dataset_name in TARGET_DATASETS},
        "semantic_model": {
            "repo_id": SEMANTIC_MODEL_ID,
            "path": str(semantic_dir),
            "ready": semantic_ready,
        },
        "local_llm": {
            "repo_id": LLM_MODEL_ID,
            "path": str(llm_dir),
            "ready": llm_ready,
        },
        "graph_backend_auto": graph_backend,
        "graph_notice": graph_notice,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
