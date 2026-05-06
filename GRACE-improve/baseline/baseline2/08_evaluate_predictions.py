import json
import os
from pathlib import Path

from evaluate_predictions import evaluate_prediction_artifacts, write_evaluation_summary


DATASET_NAME = os.getenv("GRACE_DATASET", "devign")
VARIANT_SUFFIX = os.getenv("GRACE_VARIANT_OUTPUT_SUFFIX", "").strip().lower()
PREDICTION_FILE_STEM = os.getenv("GRACE_PREDICTION_FILE_STEM") or (f"grace_hybrid_predictions_{VARIANT_SUFFIX}" if VARIANT_SUFFIX else "grace_hybrid_predictions")
RUN_STATE_FILE_STEM = os.getenv("GRACE_RUN_STATE_FILE_STEM") or (f"grace_hybrid_run_state_{VARIANT_SUFFIX}" if VARIANT_SUFFIX else "grace_hybrid_run_state")
METRICS_FILE_STEM = os.getenv("GRACE_EVALUATION_FILE_STEM") or (f"grace_hybrid_evaluation_summary_{VARIANT_SUFFIX}" if VARIANT_SUFFIX else "grace_hybrid_evaluation_summary")
PREDICTIONS_PATH = Path(os.getenv("GRACE_PREDICTIONS_PATH", f"GRACE-improve/baseline/baseline2/artifacts/predictions/{DATASET_NAME}/{PREDICTION_FILE_STEM}.jsonl"))
RUN_STATE_PATH = Path(os.getenv("GRACE_RUN_STATE_PATH", f"GRACE-improve/baseline/baseline2/artifacts/predictions/{DATASET_NAME}/{RUN_STATE_FILE_STEM}.json"))
BASELINE_COMPARE_PATH = Path(os.getenv("GRACE_BASELINE_COMPARE_PATH")) if os.getenv("GRACE_BASELINE_COMPARE_PATH") else None


def main() -> None:
    _, _, metrics = evaluate_prediction_artifacts(
        PREDICTIONS_PATH,
        RUN_STATE_PATH,
        dataset_name=DATASET_NAME,
        baseline_compare_path=BASELINE_COMPARE_PATH,
        expected_schema_version=1,
    )
    output_path = write_evaluation_summary(metrics, dataset_name=DATASET_NAME, filename=f"{METRICS_FILE_STEM}.json")
    payload = {
        "output_path": str(output_path),
        "metrics": metrics,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
