"""Analysis-only branch evaluation for an existing predictions.jsonl."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASELINE2 = HERE.parent / "baseline" / "baseline2"
sys.path.insert(0, str(BASELINE2))
from metrics import compute_branch_metrics  # noqa: E402


def read_predictions(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Duplicate record_id found in predictions")
    return rows


def analyze(predictions_path: Path, output_dir: Path, dataset: str, seed: int) -> dict:
    rows = read_predictions(predictions_path)
    metrics = compute_branch_metrics(rows)
    metrics.update({"dataset": dataset, "training_seed": int(seed), "predictions_path": str(predictions_path)})
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "branch_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    error_rows = []
    for row in rows:
        label, prediction = int(row["ground_truth"]), int(row["prediction"])
        if label == prediction:
            continue
        error_rows.append({
            "record_id": row["record_id"],
            "risk_band": row["risk_band"],
            "ground_truth": label,
            "prediction": prediction,
            "error_type": "false_negative" if label == 1 else "false_positive",
            "calibrated_probability": row.get("calibrated_probability"),
            "decision_source": row.get("decision_source"),
            "llm_called": row.get("llm_called"),
        })
    with (output_dir / "branch_errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ["record_id", "risk_band", "ground_truth", "prediction", "error_type", "calibrated_probability", "decision_source", "llm_called"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(error_rows)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True, help="Output directory for branch_metrics.json and branch_errors.csv")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "predictions_path": str(args.predictions_path), "output_path": str(args.output_path)}, indent=2))
        return
    if not args.predictions_path.is_file():
        raise FileNotFoundError(args.predictions_path)
    print(json.dumps(analyze(args.predictions_path, args.output_path, args.dataset, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
