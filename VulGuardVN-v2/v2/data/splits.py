"""De-duplication, splits, A-pool/B, dev set and demo bank (plan §3.2, P1 steps 3-4)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..common import dedup_key
from ..config import get_logger

log = get_logger("p1.splits")
SPLITS = ("train", "valid", "test")


def _drop_conflicts_and_duplicates(frame: pd.DataFrame, policy: str) -> tuple[pd.DataFrame, dict]:
    """Within one pool of records: drop hash groups with conflicting labels (policy=drop), then keep one
    representative per hash (prefer main records that belong to a pair)."""
    stats = {"input": int(len(frame))}
    label_span = frame.groupby("code_hash")["label"].agg(["min", "max"])
    conflicting = label_span.index[label_span["min"] != label_span["max"]]
    if policy == "drop" and len(conflicting):
        frame = frame[~frame["code_hash"].isin(conflicting)]
    stats["conflicting_hashes"] = int(len(conflicting))
    frame = frame.assign(_has_pair=frame["pair_id"].notna(), _main=frame["in_main"])
    frame = frame.sort_values(["_main", "_has_pair"], ascending=False, kind="stable")
    before = len(frame)
    frame = frame.drop_duplicates("code_hash", keep="first").drop(columns=["_has_pair", "_main"])
    stats["duplicates_removed"] = int(before - len(frame))
    stats["output"] = int(len(frame))
    return frame, stats


def _repair_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """A pair is kept only if both members survive and sit in the same split."""
    paired = frame[frame["pair_id"].notna()]
    grouped = paired.groupby("pair_id").agg(n=("id", "size"), roles=("pair_role", lambda s: set(s)),
                                            splits=("split", lambda s: set(s)))
    valid = grouped.index[(grouped["n"] == 2) & grouped["roles"].apply(lambda r: r == {"vulnerable", "patched"})
                          & grouped["splits"].apply(len).eq(1)]
    broken = frame["pair_id"].notna() & ~frame["pair_id"].isin(valid)
    frame.loc[broken, ["pair_id", "pair_role"]] = None
    # auxiliary records only exist to complete a pair
    frame = frame[frame["in_main"] | frame["pair_id"].notna()]
    return frame


def _stratified_assign(frame: pd.DataFrame, ratios: dict, seed: int) -> pd.Series:
    """Random stratified split of the main records (dedup makes code-hash groups unique)."""
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=frame.index, dtype=object)
    for label in (0, 1):
        index = frame.index[frame["label"] == label].to_numpy()
        index = rng.permutation(index)
        n = len(index)
        n_test = int(round(n * ratios["test"]))
        n_valid = int(round(n * ratios["valid"]))
        assignment.loc[index[:n_test]] = "test"
        assignment.loc[index[n_test:n_test + n_valid]] = "valid"
        assignment.loc[index[n_test + n_valid:]] = "train"
    return assignment


def dedup_and_split(records: list[dict], name: str, cfg: dict, global_cfg: dict) -> tuple[pd.DataFrame, dict]:
    frame = pd.DataFrame.from_records(records)
    card: dict = {"dataset": name, "loaded": int(len(frame)), "loaded_main": int(frame["in_main"].sum())}
    max_chars = int(global_cfg.get("max_code_chars", 60000))
    too_long = frame["code"].str.len() > max_chars
    empty = frame["code"].str.strip().eq("")
    frame = frame[~too_long & ~empty].copy()
    card["dropped_too_long"] = int(too_long.sum())
    card["dropped_empty"] = int(empty.sum())
    frame["code_hash"] = frame["code"].map(dedup_key)
    policy = global_cfg.get("conflict_policy", "drop")
    seed = int(global_cfg.get("split_seed", 42))

    if cfg["split"] == "official":
        parts, stats = [], {}
        for split in SPLITS:
            part, stats[split] = _drop_conflicts_and_duplicates(frame[frame["split"] == split], policy)
            parts.append(part)
        frame = pd.concat(parts)
        # cross-split leakage: test wins over valid, valid wins over train (official test stays intact)
        test_hashes = set(frame.loc[frame["split"] == "test", "code_hash"])
        valid_hashes = set(frame.loc[frame["split"] == "valid", "code_hash"])
        leak_valid = (frame["split"] == "valid") & frame["code_hash"].isin(test_hashes)
        leak_train = (frame["split"] == "train") & frame["code_hash"].isin(test_hashes | valid_hashes)
        stats["cross_split_removed"] = {"valid": int(leak_valid.sum()), "train": int(leak_train.sum())}
        frame = frame[~leak_valid & ~leak_train]
        card["dedup"] = stats
    else:
        main = frame[frame["in_main"]]
        aux = frame[~frame["in_main"]]
        main, stats_main = _drop_conflicts_and_duplicates(main, policy)
        main = main.copy()
        main["split"] = _stratified_assign(main, cfg["ratios"], seed)
        partner_split = main.dropna(subset=["pair_id"]).set_index("pair_id")["split"]
        aux = aux[aux["pair_id"].isin(partner_split.index) & ~aux["code_hash"].isin(set(main["code_hash"]))]
        aux, stats_aux = _drop_conflicts_and_duplicates(aux, policy)
        aux = aux.copy()
        aux["split"] = aux["pair_id"].map(partner_split)
        frame = pd.concat([main, aux])
        card["dedup"] = {"main": stats_main, "aux": stats_aux}

    frame = _repair_pairs(frame)
    frame = frame.reset_index(drop=True)
    return frame, card


# ---------------------------------------------------------------- subsets
def _train_fit_dev(frame: pd.DataFrame, fraction: float, seed: int) -> pd.Series:
    """train -> train_fit / train_dev, keeping both members of a pair together, stratified by label."""
    rng = np.random.default_rng(seed + 11)
    train = frame[frame["split"] == "train"]
    group = train["pair_id"].fillna(train["id"])
    group_label = train.groupby(group)["label"].max()
    subset = pd.Series("train_fit", index=train.index, dtype=object)
    dev_groups: list = []
    for label in (0, 1):
        groups = group_label.index[group_label == label].to_numpy()
        groups = rng.permutation(groups)
        dev_groups.extend(groups[: int(round(len(groups) * fraction))])
    subset[group.isin(set(dev_groups)).to_numpy()] = "train_dev"
    return subset


def _a_pool(frame: pd.DataFrame, a_cfg: dict, seed: int) -> tuple[pd.Index, dict]:
    """A-pool from the main validation records. Oversampled positives get importance weights (§3.2)."""
    rng = np.random.default_rng(seed + 23)
    valid = frame[(frame["split"] == "valid") & frame["in_main"]]
    pos = valid.index[valid["label"] == 1].to_numpy()
    neg = valid.index[valid["label"] == 0].to_numpy()
    pos = rng.permutation(pos)
    neg = rng.permutation(neg)
    n_valid = len(valid)
    if "fraction" in a_cfg:
        size = int(round(n_valid * float(a_cfg["fraction"])))
        n_pos = int(round(size * len(pos) / max(n_valid, 1)))
    else:
        size = min(int(a_cfg["size"]), n_valid // 2)
        n_pos = a_cfg.get("n_pos")
        n_pos = int(round(size * len(pos) / max(n_valid, 1))) if n_pos is None else min(int(n_pos), int(len(pos) * 0.6))
    n_neg = min(size - n_pos, len(neg))
    chosen = np.concatenate([pos[:n_pos], neg[:n_neg]])
    rate_valid = len(pos) / max(n_valid, 1)
    rate_a = n_pos / max(n_pos + n_neg, 1)
    weights = {
        1: rate_valid / rate_a if rate_a > 0 else 1.0,
        0: (1 - rate_valid) / (1 - rate_a) if rate_a < 1 else 1.0,
    }
    info = {"size": int(len(chosen)), "n_pos": int(n_pos), "rate_valid": rate_valid, "rate_a": rate_a,
            "weight_pos": weights[1], "weight_neg": weights[0]}
    return pd.Index(chosen), info


def assign_subsets(frame: pd.DataFrame, cfg: dict, global_cfg: dict, card: dict) -> pd.DataFrame:
    seed = int(global_cfg.get("split_seed", 42))
    frame["subset"] = None
    frame["weight"] = 1.0
    frame.loc[frame["split"] == "test", "subset"] = "test"
    train_subset = _train_fit_dev(frame, float(global_cfg.get("train_dev_fraction", 0.1)), seed)
    frame.loc[train_subset.index, "subset"] = train_subset
    a_index, a_info = _a_pool(frame, cfg["a_pool"], seed)
    valid_main = (frame["split"] == "valid") & frame["in_main"]
    frame.loc[valid_main, "subset"] = "b"
    frame.loc[a_index, "subset"] = "a_pool"
    frame.loc[(frame["split"] == "valid") & ~frame["in_main"], "subset"] = "valid_aux"
    frame.loc[a_index, "weight"] = np.where(frame.loc[a_index, "label"] == 1, a_info["weight_pos"], a_info["weight_neg"])
    b = frame[frame["subset"] == "b"]
    n_pos_b = int(b["label"].sum())
    card["a_pool"] = a_info
    card["b"] = {"size": int(len(b)), "n_pos": n_pos_b, "alpha_min": 1.0 / (n_pos_b + 1) if n_pos_b else None}
    return frame


def assign_dev_set(frame: pd.DataFrame, dev_cfg: dict, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed + 37)
    dev = pd.Series(False, index=frame.index)
    train = frame[frame["split"] == "train"]
    pair_ids = train["pair_id"].dropna().unique()
    pair_ids = rng.permutation(pair_ids)
    # never more than half of the train pairs: the rest must stay available for the demo bank
    chosen_pairs = set(pair_ids[: min(int(dev_cfg.get("n_pairs", 100)), len(pair_ids) // 2)])
    dev[frame["pair_id"].isin(chosen_pairs)] = True
    singles = train.index[train["pair_id"].isna() & train["in_main"]].to_numpy()
    singles = rng.permutation(singles)
    dev[singles[: int(dev_cfg.get("n_random", 100))]] = True
    return dev


def assign_demo_bank(frame: pd.DataFrame, has_pairs: bool) -> pd.Series:
    train = (frame["split"] == "train") & ~frame["dev_set"]
    if has_pairs:
        return train & frame["pair_id"].notna()
    return train & frame["in_main"]


def leakage_check(frame: pd.DataFrame) -> int:
    span = frame.groupby("code_hash")["split"].nunique()
    return int((span > 1).sum())


def split_table(frame: pd.DataFrame) -> list[dict]:
    rows = []
    for split in SPLITS:
        part = frame[frame["split"] == split]
        main = part[part["in_main"]]
        n_pairs = int(part["pair_id"].dropna().nunique())
        rows.append({"split": split, "functions": int(len(main)), "vulnerable": int(main["label"].sum()),
                     "rate": float(main["label"].mean()) if len(main) else 0.0, "pairs": n_pairs,
                     "aux_records": int((~part["in_main"]).sum())})
    return rows


def subsample_for_smoke(frame: pd.DataFrame, per_split: dict, seed: int) -> pd.DataFrame:
    """Small stratified subset (keeps pairs, guarantees positives) for smoke tests."""
    rng = np.random.default_rng(seed)
    keep: list = []
    for split, n in per_split.items():
        part = frame[frame["split"] == split]
        pair_ids = part["pair_id"].dropna().unique()
        pair_ids = rng.permutation(pair_ids)
        chosen_pairs = set(pair_ids[: max(4, n // 20)])
        keep.extend(part.index[part["pair_id"].isin(chosen_pairs)])
        singles = part[part["pair_id"].isna() & part["in_main"]]
        pos = singles.index[singles["label"] == 1].to_numpy()
        neg = singles.index[singles["label"] == 0].to_numpy()
        pos = rng.permutation(pos)
        neg = rng.permutation(neg)
        n_pos = max(n // 5, 10)
        keep.extend(pos[:n_pos])
        keep.extend(neg[: max(n - n_pos, 0)])
    return frame.loc[sorted(set(keep))].reset_index(drop=True)


def records_path(dataset_dir: Path) -> Path:
    return dataset_dir / "records.parquet"
