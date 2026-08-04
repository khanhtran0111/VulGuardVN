"""Analysis-only runtime/cost comparison for existing proposed and baseline runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from analysis_plots import bar_plot


RUNTIME_FIELDS = (
    "dataset_preprocessing_time_ms",
    "token_ast_feature_extraction_time_ms",
    "unixcoder_embedding_time_ms",
    "numeric_graph_feature_extraction_time_ms",
    "prefilter_inference_time_ms",
    "calibration_routing_time_ms",
    "graph_extraction_time_ms",
    "retrieval_time_ms",
    "llm_inference_time_ms",
    "total_runtime_ms",
)
MEAN_FIELDS = tuple(field.replace("_time_ms", "_mean_ms") for field in RUNTIME_FIELDS[:-1])


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(run_dir: Path, role: str) -> dict[str, Any]:
    required = ("runtime.json", "run_metadata.json", "metrics.json")
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing: raise FileNotFoundError(f"{role} run is missing {missing}: {run_dir}")
    return {"runtime": _read(run_dir / "runtime.json"), "metadata": _read(run_dir / "run_metadata.json"), "metrics": _read(run_dir / "metrics.json")}


def analyze(proposed_run: Path, baseline_run: Path, output_dir: Path, dataset: str, seed: int) -> dict:
    proposed = _load_run(proposed_run, "proposed"); baseline = _load_run(baseline_run, "baseline")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for role, source in (("proposed", proposed), ("reproduced_baseline", baseline)):
        runtime, metrics = source["runtime"], source["metrics"]
        row = {"role": role, "dataset": dataset, "training_seed": int(seed)}
        for field in (*RUNTIME_FIELDS, *MEAN_FIELDS): row[field] = runtime.get(field)
        row.update({
            "peak_gpu_memory_mb": runtime.get("peak_gpu_memory_mb"),
            "llm_call_count": runtime.get("llm_call_count", metrics.get("llm_calls")),
            "llm_call_ratio": runtime.get("llm_call_ratio", metrics.get("llm_call_ratio")),
        })
        rows.append(row)
    fields = list(rows[0])
    with (output_dir / "runtime_breakdown.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)

    proposed_total = proposed["runtime"].get("total_runtime_ms"); baseline_total = baseline["runtime"].get("total_runtime_ms")
    speedup = None
    if proposed_total is not None and baseline_total is not None and float(proposed_total) > 0:
        speedup = float(baseline_total) / float(proposed_total)
    comparison = [{
        "dataset": dataset, "training_seed": int(seed),
        "proposed_total_runtime_ms": proposed_total,
        "baseline_total_runtime_ms": baseline_total,
        "speedup_vs_reproduced_baseline": speedup,
        "proposed_llm_calls": rows[0]["llm_call_count"], "baseline_llm_calls": rows[1]["llm_call_count"],
        "proposed_llm_call_ratio": rows[0]["llm_call_ratio"], "baseline_llm_call_ratio": rows[1]["llm_call_ratio"],
    }]
    with (output_dir / "cost_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0])); writer.writeheader(); writer.writerows(comparison)

    plot_labels, plot_values = [], []
    for field in RUNTIME_FIELDS[:-1]:
        value = proposed["runtime"].get(field)
        if value is not None:
            plot_labels.append(field.removesuffix("_time_ms")); plot_values.append(float(value))
    bar_plot(output_dir / "runtime_breakdown.png", plot_labels, plot_values, title=f"{dataset}: proposed runtime breakdown", ylabel="milliseconds")
    summary = {
        "dataset": dataset, "training_seed": int(seed),
        "proposed_run": str(proposed_run), "baseline_run": str(baseline_run),
        "proposed": rows[0], "reproduced_baseline": rows[1],
        "speedup_vs_reproduced_baseline": speedup,
        "measurement_note": "Missing measurements remain null and were not inferred.",
    }
    (output_dir / "runtime_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposed-run", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"mode": "dry-run", "proposed_run": str(args.proposed_run), "baseline_run": str(args.baseline_run), "output_path": str(args.output_path)}, indent=2)); return
    print(json.dumps(analyze(args.proposed_run, args.baseline_run, args.output_path, args.dataset, args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
