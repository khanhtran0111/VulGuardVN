"""Aggregate isolated revision runs into CSV tables, figures, and Markdown."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
BASELINE2 = HERE.parent / "baseline" / "baseline2"
sys.path.insert(0, str(BASELINE2))
from metrics import compute_binary_metrics, paired_metric_deltas  # noqa: E402


METRIC_NAMES = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "llm_call_ratio")


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def ci95(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if len(values) < 2:
        value = float(values[0]) if values else float("nan")
        return value, value
    mean = statistics.mean(values)
    sem = statistics.stdev(values) / math.sqrt(len(values))
    try:
        from scipy.stats import t
        margin = float(t.ppf(0.975, len(values) - 1)) * sem
    except Exception:
        margin = 1.96 * sem
    return mean - margin, mean + margin


def discover_runs(results_root: Path) -> list[dict[str, Any]]:
    runs = []
    for config_path in results_root.glob("E*/**/seed_*/config.json"):
        run_dir = config_path.parent
        config = read_json(config_path)
        metadata = read_json(run_dir / "run_metadata.json")
        metrics = read_json(run_dir / "metrics.json")
        calibration = read_json(run_dir / "calibration.json")
        branch = read_json(run_dir / "branch_metrics.json")
        runtime = read_json(run_dir / "runtime.json")
        runs.append({
            "run_dir": run_dir,
            "config": config,
            "metadata": metadata,
            "metrics": metrics,
            "calibration": calibration,
            "branch": branch,
            "runtime": runtime,
            "predictions": read_jsonl(run_dir / "predictions.jsonl"),
        })
    return runs


def validate_test_split_fingerprints(runs: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int], set[str]] = defaultdict(set)
    for run in runs:
        fingerprint = run["metadata"].get("test_split_fingerprint")
        if fingerprint:
            groups[(run["config"].get("dataset_name"), int(run["config"].get("split_seed", 42)))].add(str(fingerprint))
    changed = {key: values for key, values in groups.items() if len(values) > 1}
    if changed:
        raise RuntimeError(f"Fixed-test-split invariant failed across runs: {changed}")


def summary_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        config, metrics = run["config"], run["metrics"]
        rows.append({
            "experiment": config.get("experiment_name"),
            "dataset": config.get("dataset_name"),
            "configuration": config.get("configuration"),
            "split_seed": config.get("split_seed"),
            "training_seed": config.get("training_seed"),
            "demo_seed": config.get("demo_seed"),
            "bootstrap_seed": config.get("bootstrap_seed"),
            "enabled_views": ",".join(config.get("enabled_views", [])),
            **{name: metrics.get(name) for name in METRIC_NAMES},
            "llm_calls": metrics.get("llm_calls"),
            "samples": metrics.get("samples"),
            "tau_low": run["calibration"].get("tau_low"),
            "tau_high": run["calibration"].get("tau_high"),
            "total_runtime_ms": run["runtime"].get("total_runtime_ms"),
            "run_dir": str(run["run_dir"]),
        })
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["dataset"], row["configuration"])].append(row)
    output = []
    for (experiment, dataset, configuration), group in sorted(groups.items()):
        aggregate: dict[str, Any] = {"experiment": experiment, "dataset": dataset, "configuration": configuration, "runs": len(group)}
        for name in METRIC_NAMES:
            values = [float(row[name]) for row in group if row.get(name) is not None]
            if not values:
                continue
            low, high = ci95(values)
            aggregate[f"{name}_mean"] = statistics.mean(values)
            aggregate[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate[f"{name}_ci95_low"] = low
            aggregate[f"{name}_ci95_high"] = high
            aggregate[f"{name}_mean_std"] = f"{statistics.mean(values):.4f} ± {(statistics.stdev(values) if len(values) > 1 else 0.0):.4f}"
        output.append(aggregate)
    return output


def paired_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (run["config"].get("experiment_name"), run["config"].get("dataset_name"), run["config"].get("training_seed"), run["config"].get("configuration")): run
        for run in runs
    }
    rows = []
    for key, run in indexed.items():
        experiment, dataset, seed, configuration = key
        if configuration == "reproduced_baseline":
            continue
        baseline = indexed.get((experiment, dataset, seed, "reproduced_baseline"))
        if not baseline or not baseline["predictions"] or not run["predictions"]:
            continue
        comparison = paired_metric_deltas(baseline["predictions"], run["predictions"], require_identical_records=True)
        rows.append({
            "experiment": experiment, "dataset": dataset, "configuration": configuration, "training_seed": seed,
            **{f"delta_{name}": value for name, value in comparison["paired_deltas"].items()},
            "llm_call_reduction": comparison["llm_call_reduction"],
            "aligned_samples": comparison["aligned_samples"],
        })
    return rows


def aggregate_paired_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["experiment"], row["dataset"], row["configuration"])].append(row)
    output = []
    metric_names = ("delta_accuracy", "delta_precision", "delta_recall", "delta_f1", "delta_roc_auc", "delta_pr_auc", "llm_call_reduction")
    for key, group in sorted(groups.items()):
        aggregate: dict[str, Any] = {"experiment": key[0], "dataset": key[1], "configuration": key[2], "paired_runs": len(group)}
        for name in metric_names:
            values = [float(row[name]) for row in group if row.get(name) is not None]
            if not values:
                continue
            low, high = ci95(values)
            aggregate[f"{name}_mean"] = statistics.mean(values)
            aggregate[f"{name}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            aggregate[f"{name}_ci95_low"] = low
            aggregate[f"{name}_ci95_high"] = high
        output.append(aggregate)
    return output


def ablation_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full = {(row["dataset"]): row for row in aggregates if row["experiment"] == "E02_leave_one_view_out" and row["configuration"] == "full"}
    output = []
    for row in aggregates:
        if row["experiment"] != "E02_leave_one_view_out":
            continue
        reference = full.get(row["dataset"], {})
        output.append({
            "dataset": row["dataset"], "configuration": row["configuration"],
            **{name: row.get(f"{name}_mean") for name in METRIC_NAMES},
            "delta_f1_vs_full": None if not reference else row.get("f1_mean", 0.0) - reference.get("f1_mean", 0.0),
            "delta_pr_auc_vs_full": None if not reference else row.get("pr_auc_mean", 0.0) - reference.get("pr_auc_mean", 0.0),
            "delta_llm_call_ratio_vs_full": None if not reference else row.get("llm_call_ratio_mean", 0.0) - reference.get("llm_call_ratio_mean", 0.0),
        })
    return output


def branch_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        config = run["config"]
        for name, metrics in run["branch"].get("branches", {}).items():
            output.append({
                "experiment": config.get("experiment_name"), "dataset": config.get("dataset_name"),
                "configuration": config.get("configuration"), "training_seed": config.get("training_seed"),
                "branch": name, **metrics,
                "skip_false_negatives": run["branch"].get("skip_false_negatives") if name == "skip" else None,
                "skip_false_negative_rate": run["branch"].get("skip_false_negative_rate") if name == "skip" else None,
                "vulnerable_samples_skipped": run["branch"].get("vulnerable_samples_skipped") if name == "skip" else None,
                "high_false_positives": run["branch"].get("high_false_positives") if name == "high" else None,
                "high_precision": run["branch"].get("high_precision") if name == "high" else None,
            })
    return output


def threshold_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        config, calibration, metrics = run["config"], run["calibration"], run["metrics"]
        output.append({
            "experiment": config.get("experiment_name"), "dataset": config.get("dataset_name"),
            "configuration": config.get("configuration"), "training_seed": config.get("training_seed"),
            "tau_low": calibration.get("tau_low"), "tau_high": calibration.get("tau_high"),
            "selection_strategy": calibration.get("threshold_selection_strategy"),
            "selection_split": calibration.get("threshold_selection_split"),
            "recall": metrics.get("recall"), "f1": metrics.get("f1"), "llm_call_ratio": metrics.get("llm_call_ratio"),
        })
    return output


def runtime_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for run in runs:
        config = run["config"]
        output.append({
            "experiment": config.get("experiment_name"), "dataset": config.get("dataset_name"),
            "configuration": config.get("configuration"), "training_seed": config.get("training_seed"),
            **run["runtime"],
        })
    return output


def operating_points(run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = run["predictions"]
    if not rows:
        return []
    labels = np.asarray([int(row["ground_truth"]) for row in rows])
    probabilities = np.asarray([float(row["calibrated_probability"]) for row in rows])
    points = []
    for width in np.linspace(0.0, 0.49, 26):
        tau_low, tau_high = 0.5 - width, 0.5 + width
        inspect = (probabilities > tau_low) & (probabilities < tau_high)
        predictions = np.where(probabilities <= tau_low, 0, np.where(probabilities >= tau_high, 1, probabilities >= 0.5)).astype(int)
        metrics = compute_binary_metrics(labels, predictions, probabilities)
        covered = ~inspect
        risk = float(np.mean(predictions[covered] != labels[covered])) if np.any(covered) else 0.0
        points.append({
            "experiment": run["config"].get("experiment_name"), "dataset": run["config"].get("dataset_name"),
            "configuration": run["config"].get("configuration"), "training_seed": run["config"].get("training_seed"),
            "tau_low": float(tau_low), "tau_high": float(tau_high), "coverage": float(np.mean(covered)),
            "risk": risk, "llm_call_ratio": float(np.mean(inspect)), "recall": metrics["recall"], "f1": metrics["f1"],
        })
    return points


def matched_recall_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        (run["config"].get("dataset_name"), run["config"].get("training_seed")): run
        for run in runs if run["config"].get("configuration") == "reproduced_baseline"
    }
    output = []
    for run in runs:
        config = run["config"]
        if config.get("experiment_name") != "E05_routing_policy":
            continue
        baseline = baselines.get((config.get("dataset_name"), config.get("training_seed")))
        if not baseline:
            continue
        target_recall = baseline["metrics"].get("recall")
        points = operating_points(run)
        if target_recall is None or not points:
            continue
        matched = min(points, key=lambda point: abs(float(point["recall"]) - float(target_recall)))
        output.append({
            "dataset": config.get("dataset_name"), "configuration": config.get("configuration"),
            "training_seed": config.get("training_seed"), "baseline_recall": target_recall,
            "matched_recall": matched["recall"], "recall_gap": float(matched["recall"] - target_recall),
            "llm_call_ratio": matched["llm_call_ratio"], "f1": matched["f1"],
            "tau_low": matched["tau_low"], "tau_high": matched["tau_high"],
        })
    return output


def make_figures(runs: list[dict[str, Any]], output_dir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception as exc:
        print(f"[report] figure generation skipped: {exc}")
        return []
    figure_dir = output_dir / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in runs:
        config = run["config"]
        if not run["predictions"]:
            continue
        labels = np.asarray([int(row["ground_truth"]) for row in run["predictions"]])
        probabilities = np.asarray([float(row["calibrated_probability"]) for row in run["predictions"]])
        stem = f"{config.get('experiment_name')}_{config.get('dataset_name')}_{config.get('configuration')}_seed_{config.get('training_seed')}"
        for kind in ("probability", "roc_pr"):
            fig, ax = plt.subplots(figsize=(6, 4.5))
            if kind == "probability":
                ax.hist(probabilities[labels == 0], bins=25, alpha=.65, label="non-vulnerable")
                ax.hist(probabilities[labels == 1], bins=25, alpha=.65, label="vulnerable")
                calibration = run["calibration"]
                if calibration.get("tau_low") is not None: ax.axvline(calibration["tau_low"], linestyle="--")
                if calibration.get("tau_high") is not None: ax.axvline(calibration["tau_high"], linestyle="--")
                ax.set(xlabel="Calibrated probability", ylabel="Count")
            else:
                fpr, tpr, _ = roc_curve(labels, probabilities); precision, recall, _ = precision_recall_curve(labels, probabilities)
                ax.plot(fpr, tpr, label="ROC"); ax.plot(recall, precision, label="PR")
                ax.set(xlabel="FPR / Recall", ylabel="TPR / Precision")
            ax.legend(); ax.set_title(stem)
            path = figure_dir / f"{stem}_{kind}.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))

        points = operating_points(run)
        if points:
            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].plot([p["coverage"] for p in points], [p["risk"] for p in points]); axes[0].set(xlabel="Coverage", ylabel="Risk")
            axes[1].plot([p["llm_call_ratio"] for p in points], [p["recall"] for p in points]); axes[1].set(xlabel="LLM-call ratio", ylabel="Recall")
            axes[2].plot([p["llm_call_ratio"] for p in points], [p["f1"] for p in points]); axes[2].set(xlabel="LLM-call ratio", ylabel="F1")
            path = figure_dir / f"{stem}_routing_curves.png"; fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig); paths.append(str(path))
    return paths


def markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    rows = rows[:limit] if limit else rows
    if not rows: return "_No completed data available._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=HERE.parent / "revision_results")
    parser.add_argument("--output-dir", type=Path, default=HERE.parent / "revision_results" / "report")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.results_root)
    validate_test_split_fingerprints(runs)
    summaries = summary_rows(runs); aggregates = aggregate_rows(summaries); paired = paired_rows(runs)
    paired_aggregates = aggregate_paired_rows(paired)
    ablations = ablation_rows(aggregates); branches = branch_rows(runs); thresholds = threshold_rows(runs); runtimes = runtime_rows(runs)
    points = [point for run in runs for point in operating_points(run)]
    matched_recall = matched_recall_rows(runs)
    tables = {
        "revision_summary.csv": summaries, "mean_std_ci95.csv": aggregates, "paired_comparison.csv": paired,
        "paired_mean_std_ci95.csv": paired_aggregates,
        "ablation.csv": ablations, "branch_metrics.csv": branches, "thresholds.csv": thresholds,
        "runtime.csv": runtimes, "routing_operating_points.csv": points,
        "matched_recall.csv": matched_recall,
    }
    for filename, rows in tables.items(): write_csv(args.output_dir / filename, rows)
    figures = make_figures(runs, args.output_dir)
    report = [
        "# APSIPA ASC 2026 revision experiment report",
        "",
        f"Discovered runs: {len(runs)}. Missing artifacts are left blank rather than inferred.",
        "",
        "## Mean ± standard deviation and 95% confidence intervals",
        "",
        markdown_table(aggregates, ["experiment", "dataset", "configuration", "runs", "accuracy_mean_std", "f1_mean_std", "pr_auc_mean_std", "llm_call_ratio_mean_std"]),
        "",
        "## Paired baseline comparison",
        "",
        markdown_table(paired, ["experiment", "dataset", "configuration", "training_seed", "delta_accuracy", "delta_precision", "delta_recall", "delta_f1", "delta_roc_auc", "delta_pr_auc", "llm_call_reduction"]),
        "",
        "### Paired deltas across seeds (mean, standard deviation, 95% CI)",
        "",
        markdown_table(paired_aggregates, ["experiment", "dataset", "configuration", "paired_runs", "delta_accuracy_mean", "delta_accuracy_std", "delta_accuracy_ci95_low", "delta_accuracy_ci95_high", "delta_f1_mean", "delta_pr_auc_mean", "llm_call_reduction_mean"]),
        "",
        "## Leave-one-view-out ablation",
        "",
        markdown_table(ablations, ["dataset", "configuration", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "llm_call_ratio", "delta_f1_vs_full", "delta_pr_auc_vs_full", "delta_llm_call_ratio_vs_full"]),
        "",
        "## Branch-level metrics",
        "",
        markdown_table(branches, ["dataset", "configuration", "training_seed", "branch", "sample_count", "positive_count", "negative_count", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "error_rate", "skip_false_negative_rate", "vulnerable_samples_skipped", "high_false_positives"]),
        "",
        "## Threshold operating points",
        "",
        markdown_table(thresholds, ["experiment", "dataset", "configuration", "training_seed", "tau_low", "tau_high", "selection_strategy", "selection_split", "recall", "f1", "llm_call_ratio"]),
        "",
        "## Matched-recall comparison",
        "",
        markdown_table(matched_recall, ["dataset", "configuration", "training_seed", "baseline_recall", "matched_recall", "recall_gap", "llm_call_ratio", "f1", "tau_low", "tau_high"]),
        "",
        "## Runtime and cost",
        "",
        markdown_table(runtimes, ["experiment", "dataset", "configuration", "training_seed", "dataset_preprocessing_mean_ms", "token_ast_feature_extraction_mean_ms", "unixcoder_embedding_mean_ms", "numeric_graph_feature_extraction_mean_ms", "prefilter_inference_mean_ms", "calibration_routing_mean_ms", "graph_extraction_mean_ms", "retrieval_mean_ms", "llm_inference_mean_ms", "total_runtime_ms", "llm_call_count", "llm_call_ratio", "peak_gpu_memory_mb", "speedup_vs_non_selective_baseline"]),
        "",
        f"Generated figures: {len(figures)}.",
    ]
    (args.output_dir / "revision_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(runs), "output_dir": str(args.output_dir), "tables": list(tables), "figures": len(figures)}, indent=2))


if __name__ == "__main__":
    main()
