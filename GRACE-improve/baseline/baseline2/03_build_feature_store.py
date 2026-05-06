import json
import os

from hybrid_prefilter import build_feature_store, feature_store_path
from retrieval import DEFAULT_RETRIEVAL_MODEL_REPO_ID, default_retrieval_model_dir


DATASET_NAME = os.getenv("GRACE_DATASET", "devign")
GRAPH_BACKEND = os.getenv("GRACE_GRAPH_BACKEND", "auto")
SEMANTIC_MODEL_ID = os.getenv("GRACE_RETRIEVAL_MODEL_ID", DEFAULT_RETRIEVAL_MODEL_REPO_ID)
SEMANTIC_MODEL_DIR = default_retrieval_model_dir(SEMANTIC_MODEL_ID)
AUTO_DOWNLOAD = os.getenv("GRACE_AUTO_DOWNLOAD_RETRIEVAL_MODEL", "0").strip().lower() in {"1", "true", "yes", "on"}
FORCE_REBUILD = os.getenv("GRACE_FORCE_REBUILD_FEATURES", "0").strip().lower() in {"1", "true", "yes", "on"}
BATCH_SIZE = int(os.getenv("GRACE_FEATURE_BATCH_SIZE", "16"))
PROGRESS_EVERY = int(os.getenv("GRACE_FEATURE_PROGRESS_EVERY", "256"))


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    return int(lowered)


def main() -> None:
    summary = {"dataset": DATASET_NAME, "splits": {}}
    for split_name in ["train", "val", "test"]:
        split_limit = _env_int(f"GRACE_FEATURE_LIMIT_{split_name.upper()}") or _env_int("GRACE_FEATURE_LIMIT")
        payload = build_feature_store(
            DATASET_NAME,
            split_name,
            semantic_model_name=SEMANTIC_MODEL_ID,
            semantic_model_dir=SEMANTIC_MODEL_DIR,
            graph_backend=GRAPH_BACKEND,
            force_rebuild=FORCE_REBUILD,
            auto_download_semantic_model=AUTO_DOWNLOAD,
            batch_size=BATCH_SIZE,
            limit=split_limit,
            progress_every=PROGRESS_EVERY,
        )
        summary["splits"][split_name] = {
            "path": str(feature_store_path(DATASET_NAME, split_name)),
            "rows": len(payload["record_ids"]),
            "semantic_dim": int(payload["semantic_embeddings"].shape[1]),
            "numeric_dim": int(payload["numeric_features"].shape[1]),
            "graph_backends": sorted(set(payload["graph_backends"])),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
