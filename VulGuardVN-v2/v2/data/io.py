"""Access to the processed records of a dataset (`artifacts/{dataset}/records.parquet`)."""
from __future__ import annotations

import pandas as pd

from ..config import dataset_dir

RECORD_COLUMNS = ["id", "dataset", "split", "subset", "code", "label", "code_hash", "project", "cwe", "cve",
                  "commit_id", "commit_date", "file_name", "lang", "pair_id", "pair_role", "in_main", "weight",
                  "dev_set", "demo_bank"]


def load_records(dataset: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = dataset_dir(dataset) / "records.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m v2.scripts.run_p1_data` first")
    return pd.read_parquet(path, columns=columns)


def llm_targets(records: pd.DataFrame) -> pd.Series:
    """Functions that go through Joern + the LLM layer: test (incl. pair partners), A-pool and dev set (D7)."""
    return (records["subset"].isin(["test", "a_pool"])) | records["dev_set"]
