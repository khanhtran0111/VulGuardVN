from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from pipeline_common import ensure_dir, set_seed


DEFAULT_MODEL = "microsoft/graphcodebert-base"


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, labels: np.ndarray):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase D: train GraphCodeBERT classifier")
    parser.add_argument("--features-dir", type=Path, default=Path("data/artifacts/features"))
    parser.add_argument("--out-dir", type=Path, default=Path("model"))
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    return parser.parse_args()


def load_split(features_dir: Path, split: str) -> NumpyDataset:
    input_ids = np.load(features_dir / f"{split}.input_ids.npy")
    attention_mask = np.load(features_dir / f"{split}.attention_mask.npy")
    labels = np.load(features_dir / f"{split}.labels.npy")
    return NumpyDataset(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)

    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
    }

    if len(np.unique(labels)) > 1:
        out["roc_auc"] = roc_auc_score(labels, probs)
    else:
        out["roc_auc"] = 0.0
    return out


def save_predictions(trainer: Trainer, dataset: NumpyDataset, split: str, out_dir: Path) -> None:
    pred_output = trainer.predict(dataset)
    logits = pred_output.predictions
    labels = pred_output.label_ids
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = np.argmax(logits, axis=-1)

    np.save(out_dir / f"{split}.logits.npy", logits)
    np.save(out_dir / f"{split}.labels.npy", labels)
    np.save(out_dir / f"{split}.probs.npy", probs)
    np.save(out_dir / f"{split}.preds.npy", preds)


def build_training_args(args: argparse.Namespace) -> TrainingArguments:
    init_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    kwargs = {
        "output_dir": str(args.out_dir / "checkpoints"),
        "seed": args.seed,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "logging_dir": str(args.out_dir / "logs"),
        "logging_steps": 20,
        "report_to": "none",
    }

    if "overwrite_output_dir" in init_params:
        kwargs["overwrite_output_dir"] = True

    if "evaluation_strategy" in init_params:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in init_params:
        kwargs["eval_strategy"] = "epoch"

    if "save_strategy" in init_params:
        kwargs["save_strategy"] = "epoch"

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in init_params}
    return TrainingArguments(**filtered_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    train_ds = load_split(args.features_dir, "train")
    valid_ds = load_split(args.features_dir, "valid")
    test_ds = load_split(args.features_dir, "test")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_args = build_training_args(args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    final_model_dir = args.out_dir / "final_model"
    trainer.save_model(str(final_model_dir))

    valid_metrics = trainer.evaluate(valid_ds)
    test_metrics = trainer.evaluate(test_ds)

    save_predictions(trainer, valid_ds, "valid", args.out_dir)
    save_predictions(trainer, test_ds, "test", args.out_dir)

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "valid": valid_metrics,
                "test": test_metrics,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("Phase D done.")
    print(json.dumps({"valid": valid_metrics, "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
