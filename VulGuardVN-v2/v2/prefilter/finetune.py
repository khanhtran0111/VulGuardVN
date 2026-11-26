"""Fine-tuned UniXcoder, cross-fitted (5th prefilter view `finetune` + baseline B7), trained on the LOCAL GPU.

K-fold cross-fitting on the train split (train_fit + train_dev, folds grouped by pair and stratified by label):
  fold k: fine-tune on the other K-1 folds -> logits for fold k (out-of-fold) and for every non-train record.
  train records get their out-of-fold logit (the prefilter never sees a score the encoder was trained on);
  A-pool / B / test / aux get the mean logit of the K fold models.
Outputs (artifacts/{ds}/):
  features/finetune_logit.npy           float32 [N] aligned with records.parquet (the `finetune` view)
  finetune/fold{k}/pred.npz, model.pt   per-fold predictions / weights (resumable per epoch: ckpt.pt)
  baselines/b7_unixcoder_ft.parquet     B7: K-model mean on test, threshold = max F1 on train_dev (out-of-fold)
Quadro P2000 (Pascal, 4 GB): fp32, gradient checkpointing, batch 8 x grad-accum 4 (peak 2.4 GB at max_len 512).
2026-11-25: max_len 320 -> 512 (52% of Devign functions exceed 320 tokens), 2 -> 3 epochs with a linear LR schedule
(warm-up), length-bucketed batches (sort by length inside random chunks, shuffle the batches): ~1.1x the cost per epoch
of max_len 320 with random batches instead of ~1.8x.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..common import atomic_save_npy, atomic_savez, atomic_to_parquet, atomic_write_text
from ..config import dataset_dir, get_logger, load_config, resolve_project_path
from ..data.io import load_records

log = get_logger("p3.finetune")
TRAIN_SUBSETS = ("train_fit", "train_dev")


class Classifier(torch.nn.Module):
    def __init__(self, model_dir: Path, checkpointing: bool = True) -> None:
        super().__init__()
        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_dir)
        if checkpointing:
            self.encoder.gradient_checkpointing_enable()
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(self.encoder.config.hidden_size, 1))

    def forward(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        weights = mask.unsqueeze(-1).float()
        pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(1)


def finetune_dir(dataset: str) -> Path:
    return dataset_dir(dataset, "finetune")


def logits_path(dataset: str) -> Path:
    return dataset_dir(dataset, "features") / "finetune_logit.npy"


def finetune_ready(dataset: str) -> bool:
    return logits_path(dataset).exists()


def _assign_folds(records: pd.DataFrame, train_mask: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Fold id per record (-1 outside train). Pairs stay in one fold; folds stratified by label."""
    rng = np.random.default_rng(seed)
    folds = np.full(len(records), -1, dtype=int)
    train = records[train_mask]
    group = train["pair_id"].fillna(train["id"])
    group_label = train.groupby(group)["label"].max()
    fold_of_group: dict = {}
    for label in (0, 1):
        groups = rng.permutation(group_label.index[group_label == label].to_numpy())
        for i, g in enumerate(groups):
            fold_of_group[g] = i % k
    folds[np.flatnonzero(train_mask)] = group.map(fold_of_group).to_numpy()
    return folds


@torch.no_grad()
def _predict(model, tokenizer, codes: list[str], index: np.ndarray, cfg: dict, device) -> np.ndarray:
    """Logits for codes[index], in the order of `index` (batched by length to limit padding)."""
    model.eval()
    order = np.argsort([len(codes[i]) for i in index], kind="stable")
    out = np.zeros(len(index), dtype=np.float32)
    for start in range(0, len(order), 32):
        positions = order[start: start + 32]
        enc = tokenizer([codes[i] for i in index[positions]], truncation=True, max_length=int(cfg["max_len"]),
                        padding=True, return_tensors="pt")
        out[positions] = model(enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu().numpy()
    return out


def _epoch_batches(order: np.ndarray, lengths: np.ndarray, batch_size: int, rng, bucketing: bool,
                   chunk_batches: int = 50) -> list[np.ndarray]:
    """Batches of one epoch. Bucketing: sort by length inside chunks of `chunk_batches` batches of the random order,
    then shuffle the batches, so padding is small and the batch order stays random."""
    if not bucketing:
        return [order[i: i + batch_size] for i in range(0, len(order), batch_size)]
    batches = []
    chunk = batch_size * chunk_batches
    for start in range(0, len(order), chunk):
        part = order[start: start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i: i + batch_size] for i in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def _train_fold(fold: int, train_idx: np.ndarray, labels: np.ndarray, codes: list[str], cfg: dict, model_dir: Path,
                fold_dir: Path, device, tokenizer) -> Classifier:
    seed = int(cfg["seed"]) + fold
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Classifier(model_dir).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    n_pos = int(labels[train_idx].sum())
    pos_weight = torch.tensor(min((len(train_idx) - n_pos) / max(n_pos, 1), float(cfg["pos_weight_cap"])), device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    batch_size, accum, epochs = int(cfg["batch_size"]), int(cfg["grad_accum"]), int(cfg["epochs"])
    bucketing = bool(cfg.get("length_bucketing", False))
    lengths = np.array([len(code) for code in codes])       # characters: a cheap proxy of the token length
    n_batches = len(_epoch_batches(train_idx, lengths, batch_size, np.random.default_rng(0), bucketing))
    total_steps = epochs * (n_batches // accum)
    scheduler = None
    if cfg.get("lr_schedule", "constant") == "linear":
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(optimizer, int(float(cfg.get("warmup_ratio", 0.0)) * total_steps),
                                                    total_steps)
    start_epoch = 0
    ckpt = fold_dir / "ckpt.pt"
    if ckpt.exists():                                       # resume after an interruption (epoch granularity)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)     # CPU: a GPU copy would stay resident
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])                         # moves the state to the params' device
        if scheduler is not None and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch"])
        del state
        rng = np.random.default_rng(seed + 1000 * start_epoch)
        log.info("fold %d: resumed after epoch %d", fold, start_epoch)
    target = torch.tensor(labels, dtype=torch.float32)
    for epoch in range(start_epoch, epochs):
        model.train()
        started, step, total = time.perf_counter(), 0, 0.0
        batches = _epoch_batches(rng.permutation(train_idx), lengths, batch_size, rng, bucketing)
        for index in batches:
            enc = tokenizer([codes[i] for i in index], truncation=True, max_length=int(cfg["max_len"]), padding=True,
                            return_tensors="pt")
            logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = loss_fn(logits, target[index].to(device)) / accum
            loss.backward()
            total += float(loss.item()) * accum
            step += 1
            if step % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                rate = step * batch_size / (time.perf_counter() - started)
                eta = (len(batches) - step) * batch_size / max(rate, 1e-6) / 60
                log.info("fold %d epoch %d/%d step %d/%d loss %.4f lr %.2e | %.1f samples/s | epoch ETA %.0f min", fold,
                         epoch + 1, epochs, step, len(batches), total / step, optimizer.param_groups[0]["lr"], rate, eta)
        optimizer.zero_grad(set_to_none=True)
        tmp = ckpt.with_name(".ckpt.pt.tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                    "scheduler": scheduler.state_dict() if scheduler is not None else None}, tmp)
        tmp.replace(ckpt)
        log.info("fold %d epoch %d done in %.1f min", fold, epoch + 1, (time.perf_counter() - started) / 60)
    return model


def crossfit(dataset: str, cfg: dict | None = None, force: bool = False) -> None:
    cfg = cfg or load_config("prefilter")["finetune"]
    if finetune_ready(dataset) and not force:
        log.info("[%s] fine-tuned view exists - skip", dataset)
        return
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = resolve_project_path(load_config("prefilter")["features"]["unixcoder"]["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    records = load_records(dataset, ["id", "code", "label", "subset", "pair_id", "in_main"])
    codes, labels = records["code"].tolist(), records["label"].to_numpy()
    train_mask = records["subset"].isin(TRAIN_SUBSETS).to_numpy()
    k = int(cfg["folds"])
    folds = _assign_folds(records, train_mask, k, int(cfg["seed"]))
    other_idx = np.flatnonzero(~train_mask)
    root = finetune_dir(dataset)
    oof = np.full(len(records), np.nan, dtype=np.float32)
    other = np.zeros((k, len(other_idx)), dtype=np.float32)
    log.info("[%s] fine-tune cross-fitting: K=%d, train=%d, other=%d, device=%s", dataset, k, int(train_mask.sum()),
             len(other_idx), device)
    for fold in range(k):
        fold_dir = root / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pred_path = fold_dir / "pred.npz"
        holdout = np.flatnonzero(folds == fold)
        if pred_path.exists() and not force:
            saved = np.load(pred_path)
            oof[saved["holdout"]] = saved["holdout_logit"]
            other[fold] = saved["other_logit"]
            log.info("[%s] fold %d already done - loaded", dataset, fold)
            continue
        fit = np.flatnonzero(train_mask & (folds != fold))
        rng = np.random.default_rng(int(cfg["seed"]) + fold)
        pos, neg = fit[labels[fit] == 1], fit[labels[fit] == 0]
        if len(neg) > int(cfg["max_neg"]):
            neg = rng.choice(neg, size=int(cfg["max_neg"]), replace=False)
        fit = np.concatenate([pos, neg])
        log.info("[%s] fold %d/%d: fine-tune on %d (%d positive), predict %d held-out + %d other", dataset, fold + 1, k,
                 len(fit), len(pos), len(holdout), len(other_idx))
        model = _train_fold(fold, fit, labels, codes, cfg, model_dir, fold_dir, device, tokenizer)
        holdout_logit = _predict(model, tokenizer, codes, holdout, cfg, device)
        other_logit = _predict(model, tokenizer, codes, other_idx, cfg, device)
        if cfg.get("save_models", True):
            torch.save({k_: v.half() for k_, v in model.state_dict().items()}, fold_dir / "model.pt")
        atomic_savez(pred_path, holdout=holdout, holdout_logit=holdout_logit, other_logit=other_logit)
        (fold_dir / "ckpt.pt").unlink(missing_ok=True)          # optimizer state no longer needed
        oof[holdout] = holdout_logit
        other[fold] = other_logit
        del model
        torch.cuda.empty_cache()
    final = oof.copy()
    final[other_idx] = other.mean(axis=0)
    if np.isnan(final).any():
        raise RuntimeError(f"[{dataset}] {int(np.isnan(final).sum())} records without a fine-tuned logit")
    _write_b7(dataset, records, final)
    atomic_write_text(root / "meta.json", json.dumps({"folds": k, "cfg": cfg, "n_train": int(train_mask.sum())}, indent=2))
    atomic_save_npy(logits_path(dataset), final.astype(np.float32))      # completion marker, written last
    log.info("[%s] fine-tuned view saved -> %s", dataset, logits_path(dataset))


def _write_b7(dataset: str, records: pd.DataFrame, logits: np.ndarray) -> None:
    """Baseline B7: the cross-fitted fine-tuned model alone. Threshold = max F1 on train_dev (out-of-fold logits)."""
    from sklearn.metrics import f1_score, roc_auc_score

    dev = (records["subset"] == "train_dev").to_numpy()
    y_dev, s_dev = records["label"].to_numpy()[dev], logits[dev]
    candidates = np.unique(np.quantile(s_dev, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y_dev, (s_dev >= c).astype(int)) for c in candidates]
    threshold = float(candidates[int(np.argmax(f1s))])
    test = (records["subset"] == "test").to_numpy()
    out = pd.DataFrame({"id": records.loc[test, "id"].to_numpy(), "logit": logits[test],
                        "prob": 1 / (1 + np.exp(-logits[test])), "pred": (logits[test] >= threshold).astype(int)})
    main = records.loc[test, "in_main"].to_numpy()
    y = records.loc[test, "label"].to_numpy()
    info = {"threshold": threshold, "test_f1": float(f1_score(y[main], out["pred"].to_numpy()[main])),
            "test_roc_auc": float(roc_auc_score(y[main], out["logit"].to_numpy()[main])) if 0 < y[main].sum() < main.sum() else None}
    atomic_to_parquet(out, dataset_dir(dataset, "baselines") / "b7_unixcoder_ft.parquet")
    atomic_write_text(dataset_dir(dataset, "baselines") / "b7_summary.json", json.dumps(info, indent=2))
    log.info("[%s] B7 (fine-tuned UniXcoder, K-fold ensemble): test F1 %.4f, ROC-AUC %s", dataset, info["test_f1"],
             f"{info['test_roc_auc']:.4f}" if info["test_roc_auc"] else "n/a")


@torch.no_grad()
def logits_for_codes(dataset: str, codes: list[str], device) -> np.ndarray:
    """Mean logit of the K saved fold models for new code (P7 robustness re-scoring)."""
    from transformers import AutoTokenizer

    cfg = load_config("prefilter")["finetune"]
    model_dir = resolve_project_path(load_config("prefilter")["features"]["unixcoder"]["model_dir"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    paths = sorted(finetune_dir(dataset).glob("fold*/model.pt"))
    if not paths:
        raise FileNotFoundError(f"[{dataset}] no saved fold models (finetune.save_models must be true)")
    total = np.zeros(len(codes), dtype=np.float32)
    for path in paths:
        model = Classifier(model_dir, checkpointing=False)
        model.load_state_dict({k: v.float() for k, v in torch.load(path, map_location="cpu").items()})
        model.to(device)
        total += _predict(model, tokenizer, codes, np.arange(len(codes)), cfg, device)
        del model
        torch.cuda.empty_cache()
    return total / len(paths)
