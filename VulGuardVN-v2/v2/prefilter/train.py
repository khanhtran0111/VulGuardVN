"""Prefilter training (plan P3 steps 2-5): 5 repetitions x ensemble of 5 seeds, pairwise hinge loss,
early stopping on train-dev, calibrators on A, `prefilter_scores.parquet`.

Run tags (one directory per tag under artifacts/{ds}/prefilter/models/):
  base-r{r}         rep r, seeds {10r..10r+4}, network views (all but `finetune`), pair loss, then stacked with the
                    fine-tuned logit (stacking.py, `model.fusion: stacked`)  -> score reps "1".."5"
  main-r{r}         rep r with the fine-tuned logit as a network input (`fusion: joint`; with `stacked` = A14 "joint-r{r}")
  single-r{r}       (derived, A2) first seed of the main tag, no ensemble  -> score reps "single-r{r}"
  nopair-r{r}       A3: retrained without the pairwise loss                -> "nopair-r{r}"
  lovo-{view}       A1: single models without one view, one score set per seed -> "lovo-{view}-s{seed}"
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from ..common import atomic_savez, atomic_write_text
from ..config import dataset_dir, get_logger
from ..data.io import load_records
from ..features.store import load_store
from .calibrate import FITTERS, brier, expected_calibration_error
from .model import ALL_VIEWS, MultiViewPrefilter

log = get_logger("p3.train")
SCORED_SUBSETS = ("a_pool", "b", "test", "valid_aux", "train_dev")   # train_dev: fits the stacking weights only


class DatasetTensors:
    """All views of one dataset on the training device (shared by the 25 models)."""

    def __init__(self, dataset: str, device: torch.device) -> None:
        records = load_records(dataset, ["id", "split", "subset", "label", "pair_id", "pair_role", "weight", "dev_set", "in_main"])
        store = load_store(dataset, mmap=True)
        if store["unixcoder"] is None:
            raise FileNotFoundError(f"[{dataset}] UniXcoder view missing - run run_p3_prefilter --stage features")
        if not (store["ids"]["id"].to_numpy() == records["id"].to_numpy()).all():
            raise RuntimeError(f"[{dataset}] feature store rows do not match records.parquet - rebuild features")
        self.records = records
        self.device = device
        train_fit = (records["subset"] == "train_fit").to_numpy()
        numeric = np.asarray(store["numeric"], dtype=np.float32)
        mean = numeric[train_fit].mean(0)
        std = numeric[train_fit].std(0)
        std[std < 1e-6] = 1.0
        numeric = np.clip((numeric - mean) / std, -10, 10)
        self.tokens = torch.from_numpy(np.array(store["tokens"])).to(device)
        self.ast = torch.from_numpy(np.array(store["ast"])).to(device)
        self.semantic = torch.from_numpy(np.array(store["unixcoder"])).to(device)          # float16 on device
        self.numeric = torch.from_numpy(numeric).to(device)
        from .finetune import logits_path
        ft_path = logits_path(dataset)
        self.has_finetune = ft_path.exists()
        ft = np.load(ft_path).astype(np.float32) if self.has_finetune else np.zeros(len(records), np.float32)
        ft_mean, ft_std = float(ft[train_fit].mean()), float(ft[train_fit].std() or 1.0)
        self.finetune = torch.from_numpy((ft - ft_mean) / max(ft_std, 1e-6)).to(device)
        self.finetune_stats = {"mean": ft_mean, "std": max(ft_std, 1e-6)}
        self.labels = torch.from_numpy(records["label"].to_numpy().astype(np.float32)).to(device)
        self.token_vocab = len(store["vocab"]["token"])
        self.ast_vocab = len(store["vocab"]["ast"])
        self.semantic_dim = int(self.semantic.shape[1])
        self.numeric_dim = int(self.numeric.shape[1])
        self.train_idx = np.flatnonzero(train_fit)
        self.dev_idx = np.flatnonzero((records["subset"] == "train_dev").to_numpy())
        self.score_idx = np.flatnonzero((records["subset"].isin(SCORED_SUBSETS) | records["dev_set"]).to_numpy())
        pairs = records[train_fit & records["pair_id"].notna().to_numpy()]
        by_pair = pairs.groupby("pair_id")
        index_of = {row_id: i for i, row_id in enumerate(records["id"])}
        pair_list = []
        for pair_id, group in by_pair:
            roles = dict(zip(group["pair_role"], group["id"]))
            if "vulnerable" in roles and "patched" in roles:
                pair_list.append((index_of[roles["vulnerable"]], index_of[roles["patched"]]))
        self.pairs = np.asarray(pair_list, dtype=np.int64).reshape(-1, 2)
        self.numeric_stats = {"mean": mean.tolist(), "std": std.tolist()}

    def batch(self, index: np.ndarray):
        idx = torch.from_numpy(index).to(self.device)
        return (self.tokens[idx].long(), self.ast[idx].long(), self.semantic[idx].float(), self.numeric[idx],
                self.finetune[idx], self.labels[idx])


def _set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def predict(model: MultiViewPrefilter, data: DatasetTensors, index: np.ndarray, batch_size: int = 1024) -> dict:
    model.eval()
    out = {"fus": [], "sem": [], "gra": []}
    for start in range(0, len(index), batch_size):
        tokens, ast, semantic, numeric, finetune, _ = data.batch(index[start: start + batch_size])
        logits = model(tokens, ast, semantic, numeric, finetune)
        for key in out:
            value = logits.get(key)
            out[key].append(value.float().cpu().numpy() if value is not None else np.full(len(tokens), np.nan, np.float32))
    return {key: np.concatenate(values) if values else np.zeros(0, np.float32) for key, values in out.items()}


def train_one(data: DatasetTensors, cfg: dict, seed: int, views: tuple[str, ...], pair_loss: bool, out_dir: Path) -> dict:
    """Train one model; saves weights + logits for the scored subsets. Skips if already done."""
    done_marker = out_dir / "logits.npz"
    if done_marker.exists() and (out_dir / "train_log.json").exists():
        return json.loads((out_dir / "train_log.json").read_text(encoding="utf-8"))
    if "finetune" in views and not data.has_finetune:
        raise FileNotFoundError("view `finetune` needs the fine-tuned logits - run run_p3_prefilter --stage finetune "
                                "(or remove `finetune` from model.views in prefilter.yaml)")
    out_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    tcfg = cfg["train"]
    model = MultiViewPrefilter(data.token_vocab, data.ast_vocab, data.semantic_dim, data.numeric_dim, cfg["model"],
                               views).to(data.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(tcfg["lr"]), weight_decay=float(tcfg["weight_decay"]))
    train_labels = data.labels[torch.from_numpy(data.train_idx).to(data.device)]
    n_pos = float(train_labels.sum().item())
    pos_weight = min(max((len(data.train_idx) - n_pos) / max(n_pos, 1.0), 1.0), float(tcfg["pos_weight_cap"]))
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=data.device))
    use_pairs = pair_loss and len(data.pairs) > 0 and float(tcfg["pair_mu"]) > 0
    rng = np.random.default_rng(seed)
    batch_size, n_pairs = int(tcfg["batch_size"]), int(tcfg["pairs_per_batch"])
    best, best_state, bad_epochs, history = -1.0, None, 0, []
    started = time.perf_counter()
    for epoch in range(int(tcfg["max_epochs"])):
        model.train()
        order = rng.permutation(data.train_idx)
        total_loss, steps = 0.0, 0
        for start in range(0, len(order), batch_size):
            index = order[start: start + batch_size]
            pair_rows = None
            if use_pairs:
                pair_rows = data.pairs[rng.integers(0, len(data.pairs), size=n_pairs)]
                index = np.concatenate([index, pair_rows.reshape(-1)])
            tokens, ast, semantic, numeric, finetune, labels = data.batch(index)
            logits = model(tokens, ast, semantic, numeric, finetune)
            loss = bce(logits["fus"], labels)
            aux = [bce(logits[key], labels) for key in ("sem", "gra") if key in logits]
            if aux:
                loss = loss + float(tcfg["lambda_aux"]) * sum(aux)
            if pair_rows is not None:
                scores = torch.sigmoid(logits["fus"][-2 * n_pairs:]).view(n_pairs, 2)   # (vulnerable, patched)
                hinge = torch.relu(float(tcfg["pair_margin"]) - (scores[:, 0] - scores[:, 1])).mean()
                loss = loss + float(tcfg["pair_mu"]) * hinge
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            steps += 1
        dev = predict(model, data, data.dev_idx)
        dev_y = data.records["label"].to_numpy()[data.dev_idx]
        pr_auc = float(average_precision_score(dev_y, dev["fus"])) if dev_y.any() else 0.0
        history.append({"epoch": epoch + 1, "loss": total_loss / max(steps, 1), "dev_pr_auc": pr_auc})
        if pr_auc > best:
            best, bad_epochs = pr_auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= int(tcfg["patience"]):
                break
    model.load_state_dict(best_state)
    logits = predict(model, data, data.score_idx)
    torch.save({"state": best_state, "views": list(model.views), "numeric_stats": data.numeric_stats,
                "finetune_stats": data.finetune_stats,
                "token_vocab": data.token_vocab, "ast_vocab": data.ast_vocab, "model_cfg": cfg["model"]},
               out_dir / "model.pt")
    info = {"seed": seed, "views": list(model.views), "pair_loss": use_pairs, "pos_weight": pos_weight,
            "best_dev_pr_auc": best, "epochs": len(history), "seconds": time.perf_counter() - started, "history": history}
    atomic_write_text(out_dir / "train_log.json", json.dumps(info, indent=2))
    atomic_savez(done_marker, index=data.score_idx, **logits)          # done marker, written last
    return info


def models_dir(dataset: str) -> Path:
    return dataset_dir(dataset, "prefilter", "models")


def run_tag(dataset: str, data: DatasetTensors, cfg: dict, tag: str, seeds: list[int], views=ALL_VIEWS,
            pair_loss: bool = True) -> None:
    for seed in seeds:
        out_dir = models_dir(dataset) / tag / f"seed{seed}"
        info = train_one(data, cfg, seed, tuple(views), pair_loss, out_dir)
        log.info("[%s] %s seed=%d dev PR-AUC=%.4f epochs=%d (%.0fs)", dataset, tag, seed, info["best_dev_pr_auc"],
                 info["epochs"], info.get("seconds", 0))


def load_logits(dataset: str, tag: str, seed: int) -> dict:
    payload = np.load(models_dir(dataset) / tag / f"seed{seed}" / "logits.npz")
    return {key: payload[key] for key in payload.files}


# ---------------------------------------------------------------- aggregation + calibration
def score_set(records: pd.DataFrame, logits_list: list[dict], rep: str, calibrators: list[str],
              finetune: np.ndarray | None = None) -> tuple[pd.DataFrame, dict]:
    """Ensemble score s(x), variance u(x) and calibrated p_hat. With `finetune` (raw fine-tuned logits aligned with
    records), s is the stacked score fitted on train_dev (stacking.py); train_dev rows are dropped afterwards."""
    from . import stacking

    index = logits_list[0]["index"]
    fus = np.stack([l["fus"] for l in logits_list])
    sem = np.stack([l["sem"] for l in logits_list])
    gra = np.stack([l["gra"] for l in logits_list])
    sub = records.iloc[index]
    s, u = fus.mean(0), (fus.var(0) if len(logits_list) > 1 else np.zeros(len(index)))
    params = {}
    if finetune is not None:
        dev = (sub["subset"] == "train_dev").to_numpy()
        if not dev.any():
            raise RuntimeError(f"rep {rep}: no train_dev logits to fit the stacking weights - retrain these models")
        ft = np.asarray(finetune, dtype=float)[index]
        params["stack"] = stacking.fit(s[dev], ft[dev], sub["label"].to_numpy()[dev])
        s, u = stacking.apply(s, ft, params["stack"]), u * params["stack"]["a"] ** 2
    frame = pd.DataFrame({
        "id": sub["id"].to_numpy(), "split": sub["split"].to_numpy(), "subset": sub["subset"].to_numpy(),
        "dev_set": sub["dev_set"].to_numpy(), "label": sub["label"].to_numpy(), "weight": sub["weight"].to_numpy(),
        "in_main": sub["in_main"].to_numpy(),
        "rep": rep, "s": s, "u": u,
        "p_sem": (1 / (1 + np.exp(-sem))).mean(0), "p_gra": (1 / (1 + np.exp(-gra))).mean(0),
    })
    frame = frame[(frame["subset"] != "train_dev") | frame["dev_set"]].reset_index(drop=True)
    a = frame[frame["subset"] == "a_pool"]
    for name in calibrators:
        fit, apply = FITTERS[name]
        params[name] = fit(a["s"].to_numpy(), a["label"].to_numpy(), a["weight"].to_numpy())
        frame[f"p_hat_{name}"] = apply(frame["s"].to_numpy(), params[name])
    return frame, params


def evaluate_scores(frame: pd.DataFrame, calibrators: list[str]) -> dict:
    main = frame[(frame["subset"] == "test") & frame["in_main"]]
    y, s = main["label"].to_numpy(), main["s"].to_numpy()
    out = {"n_test": int(len(main)), "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
           "pr_auc": float(average_precision_score(y, s)) if y.any() else None}
    a = frame[frame["subset"] == "a_pool"]
    for name in calibrators:
        p = main[f"p_hat_{name}"].to_numpy()
        out[f"ece_test_{name}"] = expected_calibration_error(y, p)
        out[f"brier_test_{name}"] = brier(y, p)
        out[f"ece_a_{name}"] = expected_calibration_error(a["label"].to_numpy(), a[f"p_hat_{name}"].to_numpy(), a["weight"].to_numpy())
    return out
