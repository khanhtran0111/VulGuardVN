from __future__ import annotations

import argparse
import json

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from common import WORK_DIR, load_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate binary vulnerability predictions.")
    p.add_argument("--input", default=str(WORK_DIR / "test_predictions.jsonl"))
    p.add_argument("--label-field", default="label")
    p.add_argument("--pred-field", default="pred_label")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    rows = [r for r in rows if int(r.get(args.pred_field, -1)) in (0, 1)]
    y_true = [int(r[args.label_field]) for r in rows]
    y_pred = [int(r[args.pred_field]) for r in rows]

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    print(
        json.dumps(
            {
                "num_rows": len(rows),
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
