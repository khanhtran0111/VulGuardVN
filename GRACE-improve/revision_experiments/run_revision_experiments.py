"""Run isolated APSIPA revision experiments through the shared baseline2 code.

This file intentionally orchestrates the existing numbered pipeline scripts;
it does not copy their implementation into experiment directories.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiment_config import (
    DATASETS,
    EXPERIMENTS,
    TRAINING_SEEDS,
    ExperimentConfig,
    config_for_configuration,
    configurations_for,
    load_config,
    validate_split_integrity,
)


HERE = Path(__file__).resolve().parent
GRACE_ROOT = HERE.parent
REPOSITORY_ROOT = GRACE_ROOT.parent
BASELINE2 = GRACE_ROOT / "baseline" / "baseline2"
STAGES = (
    ("preprocess", "01_prepare_datasets.py"),
    ("splits", "02_create_splits.py"),
    ("features", "03_build_feature_store.py"),
    ("train", "04_train_hybrid_prefilter.py"),
    ("calibrate", "05_calibrate_budget_controller.py"),
    ("demo_bank", "06_build_demo_bank.py"),
    ("inference", "07_run_grace_hybrid.py"),
    ("evaluate", "08_evaluate_predictions.py"),
)
REQUIRED_OUTPUTS = (
    "config.json",
    "run_metadata.json",
    "metrics.json",
    "predictions.jsonl",
    "calibration.json",
    "branch_metrics.json",
    "runtime.json",
)


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _git_sha() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "tensorflow", "torch", "transformers", "joblib", "matplotlib")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _hardware() -> dict[str, Any]:
    gpu: dict[str, Any] = {"available": False}
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu = {"available": True, "devices": result.stdout.strip().splitlines()}
    except OSError:
        pass
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "gpu": gpu,
    }


def build_metadata(config: ExperimentConfig, status: str, **extra: Any) -> dict[str, Any]:
    calibration = _read_json(config.run_directory / "calibration.json", {})
    run_state = _read_json(config.run_directory / "_pipeline" / "run_state.json", {})
    graph_counts = run_state.get("graph_backend_counts") or {}
    stage_state = _read_json(config.run_directory / "_pipeline" / "stage_state.json", {})
    retrieval_summary = _read_json(
        config.run_directory / "_pipeline" / "retrieval" / config.dataset_name / "summary.json", {}
    )
    resolved_graph = retrieval_summary.get("graph_backend_resolved") or (sorted(graph_counts) if graph_counts else None)
    return {
        "dataset": config.dataset_name,
        "experiment": config.experiment_name,
        "configuration": config.configuration,
        "split_seed": config.split_seed,
        "training_seed": config.training_seed,
        "demo_seed": config.demo_seed,
        "bootstrap_seed": config.bootstrap_seed,
        "commit_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "hardware_information": _hardware(),
        "requested_graph_backend": config.graph_backend,
        "resolved_graph_backend": resolved_graph,
        "retrieval_backend": run_state.get("retrieval_backend") or retrieval_summary.get("semantic_backend") or config.retrieval_backend,
        "model_id": (run_state.get("config") or {}).get("llm_model_name", config.model_name),
        "prefilter_model_id": config.model_name,
        "thresholds": {"tau_low": calibration.get("tau_low", config.tau_low), "tau_high": calibration.get("tau_high", config.tau_high)},
        "calibration_method": calibration.get("calibration_method", config.calibration_method),
        "threshold_selection_split": calibration.get("threshold_selection_split", "validation"),
        "test_split_fingerprint": stage_state.get("test_split_fingerprint"),
        "status": status,
        **extra,
    }


def is_complete(run_dir: Path) -> bool:
    metadata = _read_json(run_dir / "run_metadata.json", {})
    return metadata.get("status") == "complete" and all((run_dir / name).exists() for name in REQUIRED_OUTPUTS)


def _baseline_compare_path(config: ExperimentConfig) -> Path | None:
    if config.configuration == "reproduced_baseline":
        return None
    candidate = (
        Path(config.output_directory) / config.experiment_name / config.dataset_name
        / "reproduced_baseline" / f"seed_{config.training_seed}" / "predictions.jsonl"
    )
    return candidate if candidate.exists() else None


def _stage_environment(config: ExperimentConfig, smoke: bool, resume: bool) -> dict[str, str]:
    env = os.environ.copy()
    env.update(config.to_environment())
    env["PYTHONHASHSEED"] = str(config.training_seed)
    env["GRACE_RESUME"] = str(resume).lower()
    compare_path = _baseline_compare_path(config)
    if compare_path:
        env["GRACE_BASELINE_COMPARE_PATH"] = str(compare_path)
    if smoke:
        env.update({
            "GRACE_PREPARE_LIMIT": "500",
            "GRACE_FEATURE_LIMIT": "24",
            "GRACE_MAX_TEST_SAMPLES": "24",
            "GRACE_MAX_EXAMPLES_PER_LABEL": "8",
            "GRACE_PREFILTER_EPOCHS": "1",
            "GRACE_CALL_LLM_FOR_INSPECT": "false",
            "GRACE_CALL_LLM_FOR_HIGH": "false",
        })
    return env


def _validate_split_manifest(config: ExperimentConfig) -> str:
    path = config.run_directory / "_pipeline" / "splits" / config.dataset_name / "split_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split stage did not create {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    validate_split_integrity(rows)
    test_rows = sorted(
        (str(row["record_id"]), str(row["code_hash"])) for row in rows if str(row.get("split")) == "test"
    )
    return hashlib.sha256(json.dumps(test_rows, separators=(",", ":")).encode("utf-8")).hexdigest()


def _collect_runtime(config: ExperimentConfig, stage_timings: dict[str, float]) -> dict[str, Any]:
    run_state = _read_json(config.run_directory / "_pipeline" / "run_state.json", {})
    metrics = _read_json(config.run_directory / "metrics.json", {})
    timing = run_state.get("timing_ms") or metrics.get("timing_ms") or {}
    payload = {
        "dataset": config.dataset_name,
        "experiment": config.experiment_name,
        "configuration": config.configuration,
        "training_seed": config.training_seed,
        "stage_runtime_ms": stage_timings,
        "dataset_preprocessing_time_ms": stage_timings.get("preprocess"),
        "dataset_preprocessing_mean_ms": None,
        "token_ast_feature_extraction_time_ms": None,
        "token_ast_feature_extraction_mean_ms": None,
        "unixcoder_embedding_time_ms": None,
        "unixcoder_embedding_mean_ms": None,
        "numeric_graph_feature_extraction_time_ms": None,
        "numeric_graph_feature_extraction_mean_ms": None,
        "prefilter_inference_time_ms": timing.get("prefilter_inference_total"),
        "prefilter_inference_mean_ms": timing.get("prefilter_inference_mean"),
        "calibration_routing_time_ms": timing.get("calibration_routing_total"),
        "calibration_routing_mean_ms": timing.get("calibration_routing_mean"),
        "graph_extraction_time_ms": timing.get("graph_total"),
        "graph_extraction_mean_ms": timing.get("graph_mean"),
        "retrieval_time_ms": timing.get("retrieval_total"),
        "retrieval_mean_ms": timing.get("retrieval_mean"),
        "llm_inference_time_ms": timing.get("llm_total"),
        "llm_inference_mean_ms": timing.get("llm_mean"),
        "total_runtime_ms": float(sum(stage_timings.values())),
        "llm_call_count": metrics.get("llm_calls", run_state.get("llm_calls")),
        "llm_call_ratio": metrics.get("llm_call_ratio", run_state.get("llm_call_ratio")),
        "peak_gpu_memory_mb": run_state.get("peak_gpu_memory_mb"),
        "speedup_vs_non_selective_baseline": None,
        "measurement_note": "Unavailable fields remain null; they must not be inferred.",
    }
    try:
        import joblib
        feature_root = config.run_directory / "_pipeline" / "features" / config.dataset_name
        split_payloads = []
        for path in feature_root.glob("*_features*.joblib"):
            split_payloads.append(joblib.load(path))
        if split_payloads:
            total_records = sum(len(row.get("record_ids", [])) for row in split_payloads)
            mapping = {
                "token_ast_feature_extraction_time_ms": "token_ast_feature_extraction_total",
                "unixcoder_embedding_time_ms": "unixcoder_embedding_total",
                "numeric_graph_feature_extraction_time_ms": "numeric_graph_feature_extraction_total",
            }
            for output_name, source_name in mapping.items():
                total = float(sum(float(row.get("timing_ms", {}).get(source_name, 0.0)) for row in split_payloads))
                payload[output_name] = total
                payload[output_name.replace("_time_ms", "_mean_ms")] = total / max(total_records, 1)
            if payload["dataset_preprocessing_time_ms"] is not None:
                payload["dataset_preprocessing_mean_ms"] = float(payload["dataset_preprocessing_time_ms"]) / max(total_records, 1)
    except Exception as exc:
        payload["feature_timing_notice"] = str(exc)
    return payload


def run_one(
    config: ExperimentConfig,
    *,
    dry_run: bool,
    resume: bool,
    smoke: bool,
    selected_stages: set[str] | None,
    session_budget_hours: float | None = None,
    min_remaining_minutes: float = 20.0,
    session_start_epoch: float | None = None,
) -> str:
    runner_started = time.perf_counter()
    def session_elapsed_seconds() -> float:
        if session_start_epoch is not None:
            return max(time.time() - session_start_epoch, 0.0)
        return time.perf_counter() - runner_started
    run_dir = config.run_directory
    if is_complete(run_dir):
        print(f"[skip] complete run: {run_dir}")
        return "skipped"
    commands = [(name, [sys.executable, str(BASELINE2 / script)]) for name, script in STAGES]
    if selected_stages:
        commands = [(name, command) for name, command in commands if name in selected_stages]
    print(f"[run] {config.experiment_name}/{config.dataset_name}/{config.configuration}/seed_{config.training_seed}")
    if dry_run:
        for name, command in commands:
            print(f"  [{name}] {' '.join(command)}")
        print(f"  output={run_dir}")
        return "dry-run"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(exist_ok=True)
    _write_json(run_dir / "config.json", config.to_dict())
    stage_state_path = run_dir / "_pipeline" / "stage_state.json"
    stage_state = _read_json(stage_state_path, {"completed": [], "runtime_ms": {}})
    env = _stage_environment(config, smoke, resume)

    _write_json(run_dir / "run_metadata.json", build_metadata(config, "running"))
    for stage_index, (name, command) in enumerate(commands):
        if resume and name in stage_state.get("completed", []):
            print(f"[resume] stage already complete: {name}")
            continue
        elapsed_seconds = session_elapsed_seconds()
        remaining_seconds = None if session_budget_hours is None else session_budget_hours * 3600.0 - elapsed_seconds
        previous_duration_ms = (stage_state.get("runtime_ms") or {}).get(name)
        safety_seconds = min_remaining_minutes * 60.0
        insufficient_buffer = remaining_seconds is not None and remaining_seconds <= safety_seconds
        insufficient_estimate = (
            remaining_seconds is not None
            and previous_duration_ms is not None
            and float(previous_duration_ms) / 1000.0 > max(remaining_seconds - safety_seconds, 0.0)
        )
        if insufficient_buffer or insufficient_estimate:
            stage_state["stopped_before_stage"] = name
            stage_state["partial_reason"] = "kaggle_session_time_guard"
            _write_json(stage_state_path, stage_state)
            runtime = _collect_runtime(config, stage_state.get("runtime_ms", {}))
            _write_json(run_dir / "runtime.json", runtime)
            _write_json(
                run_dir / "run_metadata.json",
                build_metadata(
                    config,
                    "partial",
                    stopped_before_stage=name,
                    elapsed_seconds=float(elapsed_seconds),
                    remaining_seconds=None if remaining_seconds is None else float(max(remaining_seconds, 0.0)),
                ),
            )
            print(f"[partial] time guard stopped before stage={name}; artifacts are resumable at {run_dir}")
            return "partial"
        print(f"[stage:{name}] {' '.join(command)}")
        started = time.perf_counter()
        result = subprocess.run(command, cwd=REPOSITORY_ROOT, env=env, check=False)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stage_state.setdefault("runtime_ms", {})[name] = float(elapsed_ms)
        if result.returncode != 0:
            _write_json(stage_state_path, stage_state)
            _write_json(run_dir / "run_metadata.json", build_metadata(config, "failed", failed_stage=name, return_code=result.returncode))
            raise RuntimeError(f"Stage {name} failed with exit code {result.returncode}")
        stage_state.setdefault("completed", []).append(name)
        if name == "splits":
            stage_state["test_split_fingerprint"] = _validate_split_manifest(config)
        _write_json(stage_state_path, stage_state)
        elapsed_seconds = session_elapsed_seconds()
        remaining_seconds = None if session_budget_hours is None else max(session_budget_hours * 3600.0 - elapsed_seconds, 0.0)
        next_name = commands[stage_index + 1][0] if stage_index + 1 < len(commands) else None
        next_estimate = (stage_state.get("runtime_ms") or {}).get(next_name) if next_name else None
        print(
            f"[time] elapsed={elapsed_seconds / 60.0:.1f}min"
            + (f" | remaining={remaining_seconds / 60.0:.1f}min" if remaining_seconds is not None else "")
            + (f" | next={next_name} | prior_estimate={float(next_estimate) / 60000.0:.1f}min" if next_estimate is not None else "")
        )

    runtime = _collect_runtime(config, stage_state.get("runtime_ms", {}))
    baseline_runtime_path = (
        Path(config.output_directory) / config.experiment_name / config.dataset_name
        / "reproduced_baseline" / f"seed_{config.training_seed}" / "runtime.json"
    )
    if config.configuration != "reproduced_baseline" and baseline_runtime_path.exists():
        baseline_runtime = _read_json(baseline_runtime_path, {})
        proposed_total = float(runtime.get("total_runtime_ms") or 0.0)
        baseline_total = float(baseline_runtime.get("total_runtime_ms") or 0.0)
        runtime["speedup_vs_non_selective_baseline"] = baseline_total / proposed_total if proposed_total > 0 else None
    _write_json(run_dir / "runtime.json", runtime)
    complete = all((run_dir / name).exists() for name in REQUIRED_OUTPUTS if name != "run_metadata.json")
    status = "complete" if complete else "incomplete"
    missing = [name for name in REQUIRED_OUTPUTS if name != "run_metadata.json" and not (run_dir / name).exists()]
    _write_json(run_dir / "run_metadata.json", build_metadata(config, status, missing_outputs=missing))
    if not complete:
        raise RuntimeError(f"Run finished but required outputs are missing: {missing}")
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", choices=DATASETS, required=True)
    parser.add_argument("--experiment", action="append", choices=EXPERIMENTS, required=True)
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--configuration", action="append")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--demo-seed", type=int)
    parser.add_argument("--bootstrap-seed", type=int)
    parser.add_argument("--output-directory")
    parser.add_argument("--stage", action="append", choices=[name for name, _ in STAGES])
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny no-LLM development run; not a paper result.")
    parser.add_argument("--session-budget-hours", type=float, help="Stop before a new stage when the session budget is nearly exhausted.")
    parser.add_argument("--min-remaining-minutes", type=float, default=20.0)
    parser.add_argument("--session-start-epoch", type=float, help="UTC epoch captured near notebook start for whole-session accounting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(
        args.config,
        split_seed=args.split_seed,
        demo_seed=args.demo_seed,
        bootstrap_seed=args.bootstrap_seed,
        output_directory=args.output_directory,
    ) if args.config else ExperimentConfig(
        split_seed=args.split_seed if args.split_seed is not None else 42,
        demo_seed=args.demo_seed if args.demo_seed is not None else 31415,
        bootstrap_seed=args.bootstrap_seed if args.bootstrap_seed is not None else 27182,
        output_directory=args.output_directory or "GRACE-improve/revision_results",
    )

    statuses = []
    for experiment in args.experiment:
        default_seeds = TRAINING_SEEDS if experiment == "E01_multiseed" else (42,)
        for dataset in dict.fromkeys(args.dataset):
            for seed in dict.fromkeys(args.seed or default_seeds):
                config = base.with_updates(dataset_name=dataset, experiment_name=experiment, training_seed=seed)
                requested_configs = args.configuration or configurations_for(experiment)
                for configuration in requested_configs:
                    if configuration not in configurations_for(experiment):
                        raise ValueError(f"Configuration {configuration!r} is invalid for {experiment}")
                    configured = config_for_configuration(config, configuration)
                    statuses.append(run_one(
                        configured, dry_run=args.dry_run, resume=args.resume,
                        smoke=args.smoke, selected_stages=set(args.stage) if args.stage else None,
                        session_budget_hours=args.session_budget_hours,
                        min_remaining_minutes=args.min_remaining_minutes,
                        session_start_epoch=args.session_start_epoch,
                    ))
    print(json.dumps({"runs": len(statuses), "statuses": statuses}, indent=2))


if __name__ == "__main__":
    main()
