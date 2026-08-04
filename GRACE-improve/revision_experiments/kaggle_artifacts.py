"""Safe helpers for moving revision artifacts between Kaggle sessions."""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REQUIRED = ("config.json", "run_metadata.json", "metrics.json")
PORTABLE_ARTIFACTS = (
    "config.json",
    "run_metadata.json",
    "metrics.json",
    "predictions.jsonl",
    "calibration.json",
    "branch_metrics.json",
    "runtime.json",
    "branch_errors.csv",
    "routing_policy_metrics.json",
    "routing_operating_points.csv",
    "matched_recall.csv",
    "calibration_summary.json",
    "probability_records.csv",
    "runtime_summary.json",
    "runtime_breakdown.csv",
    "cost_comparison.csv",
    "probability_histogram.png",
    "reliability_diagram.png",
    "roc_curve.png",
    "pr_curve.png",
    "risk_coverage_curve.png",
    "recall_llm_call_curve.png",
    "f1_llm_call_curve.png",
    "runtime_breakdown.png",
)
CHECKPOINT_MANIFEST = "checkpoint_manifest.json"


def read_json(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON artifact: {resolved}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in {resolved}")
    return payload


def run_relative_path(experiment: str, dataset: str, configuration: str, seed: int) -> Path:
    return Path(experiment) / dataset / configuration / f"seed_{int(seed)}"


def find_run(
    results_root: str | Path,
    experiment: str,
    dataset: str,
    configuration: str,
    seed: int,
) -> Path | None:
    candidate = Path(results_root) / run_relative_path(experiment, dataset, configuration, seed)
    return candidate if candidate.is_dir() else None


def validate_run(
    run_dir: str | Path,
    *,
    required: Iterable[str] = DEFAULT_REQUIRED,
    require_complete: bool = False,
) -> dict[str, Any]:
    resolved = Path(run_dir)
    missing = [name for name in required if not (resolved / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run {resolved} is missing artifacts: {missing}")
    json_payloads = {}
    for name in required:
        if name.endswith(".json"):
            json_payloads[name] = read_json(resolved / name)
    metadata = json_payloads.get("run_metadata.json") or read_json(resolved / "run_metadata.json")
    if require_complete and metadata.get("status") != "complete":
        raise RuntimeError(f"Run is not complete: status={metadata.get('status')!r}, path={resolved}")
    return {"run_dir": str(resolved), "missing": missing, "json": json_payloads, "status": metadata.get("status")}


def _safe_extract(zip_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP member in {zip_path}: {member.filename}")
        archive.extractall(destination)


def _copy_tree_read_only_source(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def materialize_input_artifacts(
    input_path: str | Path,
    working_root: str | Path,
) -> list[Path]:
    """Copy/extract input artifacts without ever writing under /kaggle/input."""
    source = Path(input_path)
    destination = Path(working_root)
    if not source.exists():
        return []
    destination.mkdir(parents=True, exist_ok=True)
    materialized: list[Path] = []
    if source.is_file() and source.suffix.lower() == ".zip":
        extract_dir = destination / source.stem
        _safe_extract(source, extract_dir)
        materialized.append(extract_dir)
        return materialized
    if source.is_dir():
        for archive in source.rglob("*.zip"):
            extract_dir = destination / archive.stem
            _safe_extract(archive, extract_dir)
            materialized.append(extract_dir)
    return materialized


def locate_or_materialize_run(
    *,
    results_root: str | Path,
    experiment: str,
    dataset: str,
    configuration: str,
    seed: int,
    input_path: str | Path | None = None,
    staging_root: str | Path | None = None,
) -> Path:
    existing = find_run(results_root, experiment, dataset, configuration, seed)
    if existing:
        return existing
    if input_path is None:
        raise FileNotFoundError(
            f"Missing run {run_relative_path(experiment, dataset, configuration, seed)} under {results_root}"
        )
    staging = Path(staging_root or (Path(results_root).parent / "imported_artifacts"))
    materialized = materialize_input_artifacts(input_path, staging)
    relative = run_relative_path(experiment, dataset, configuration, seed)
    search_roots = [Path(input_path), staging, *materialized]
    for root in search_roots:
        if not root.exists() or root.is_file():
            continue
        direct = root / relative
        candidates = [direct] if direct.is_dir() else []
        candidates.extend(path for path in root.rglob(f"seed_{int(seed)}") if path.is_dir())
        for candidate in candidates:
            normalized = candidate.as_posix()
            if f"/{experiment}/{dataset}/{configuration}/seed_{int(seed)}" not in f"/{normalized}":
                continue
            destination = Path(results_root) / relative
            _copy_tree_read_only_source(candidate, destination)
            return destination
    raise FileNotFoundError(
        f"Could not locate {relative} in {input_path}. Upload the exported E01/E05 ZIP as a Kaggle Dataset."
    )


def copy_run_artifacts(source_run: str | Path, destination_run: str | Path) -> Path:
    """Copy only portable result artifacts; omit models, caches, and raw data."""
    source = Path(source_run)
    destination = Path(destination_run)
    destination.mkdir(parents=True, exist_ok=True)
    for name in PORTABLE_ARTIFACTS:
        path = source / name
        if path.is_file():
            shutil.copy2(path, destination / name)
    figures = source / "figures"
    if figures.is_dir():
        _copy_tree_read_only_source(figures, destination / "figures")
    return destination


def initialize_analysis_run(
    destination_run: str | Path,
    *,
    source_run: str | Path,
    experiment: str,
    dataset: str,
    configuration: str,
    seed: int,
    commit_sha: str | None,
    copy_artifacts: Iterable[str] = ("metrics.json",),
) -> Path:
    destination = Path(destination_run); source = Path(source_run)
    destination.mkdir(parents=True, exist_ok=True)
    for name in copy_artifacts:
        if (source / name).is_file(): shutil.copy2(source / name, destination / name)
    config = {
        "dataset_name": dataset, "experiment_name": experiment, "configuration": configuration,
        "training_seed": int(seed), "analysis_only": True, "source_run": str(source),
    }
    metadata = {
        "dataset": dataset, "experiment": experiment, "configuration": configuration,
        "training_seed": int(seed), "commit_sha": commit_sha, "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "running", "analysis_only": True, "source_run": str(source),
    }
    (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def finalize_analysis_run(destination_run: str | Path, *, status: str = "complete") -> dict[str, Any]:
    destination = Path(destination_run); metadata_path = destination / "run_metadata.json"
    metadata = read_json(metadata_path); metadata["status"] = status
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def package_run(
    run_dir: str | Path,
    exports_dir: str | Path,
    *,
    dataset: str,
    experiment: str,
    configuration: str,
    seed: int,
) -> Path:
    source = Path(run_dir)
    metadata = read_json(source / "run_metadata.json") if (source / "run_metadata.json").exists() else {}
    partial_suffix = "_partial" if metadata.get("status") != "complete" else ""
    export_root = Path(exports_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    zip_path = export_root / f"{dataset}_{experiment}_{configuration}_seed_{int(seed)}{partial_suffix}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        relative_root = run_relative_path(experiment, dataset, configuration, seed)
        for name in PORTABLE_ARTIFACTS:
            path = source / name
            if path.is_file():
                archive.write(path, (relative_root / name).as_posix())
        figures = source / "figures"
        if figures.is_dir():
            for path in figures.rglob("*"):
                if path.is_file():
                    archive.write(path, (relative_root / "figures" / path.relative_to(figures)).as_posix())
    return zip_path


def _checkpoint_identity(run_dir: Path) -> dict[str, Any]:
    config = read_json(run_dir / "config.json")
    metadata = read_json(run_dir / "run_metadata.json")
    run_state_path = run_dir / "_pipeline" / "run_state.json"
    run_state = read_json(run_state_path) if run_state_path.is_file() else {}
    identity = {
        "dataset": config.get("dataset_name", metadata.get("dataset")),
        "experiment": config.get("experiment_name", metadata.get("experiment")),
        "configuration": config.get("configuration", metadata.get("configuration")),
        "seed": config.get("training_seed", metadata.get("training_seed")),
        "commit_sha": metadata.get("commit_sha"),
        "run_signature": run_state.get("run_signature"),
    }
    missing = [key for key in ("dataset", "experiment", "configuration", "seed", "commit_sha") if identity[key] in (None, "")]
    if missing:
        raise RuntimeError(f"Cannot checkpoint {run_dir}; missing identity fields: {missing}")
    identity["seed"] = int(identity["seed"])
    return identity


def package_checkpoint(
    run_dir: str | Path,
    exports_dir: str | Path,
    *,
    dataset: str,
    experiment: str,
    configuration: str,
    seed: int,
) -> Path:
    """Package the complete run directory, including ``_pipeline``, for resume."""
    source = Path(run_dir)
    identity = _checkpoint_identity(source)
    expected = {
        "dataset": dataset,
        "experiment": experiment,
        "configuration": configuration,
        "seed": int(seed),
    }
    for key, value in expected.items():
        if identity[key] != value:
            raise RuntimeError(f"Checkpoint identity mismatch for {key}: {identity[key]!r} != {value!r}")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **identity,
    }
    export_root = Path(exports_dir)
    export_root.mkdir(parents=True, exist_ok=True)
    zip_path = export_root / f"{dataset}_{experiment}_{configuration}_seed_{int(seed)}_checkpoint.zip"
    relative_root = run_relative_path(experiment, dataset, configuration, seed)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(CHECKPOINT_MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, (relative_root / path.relative_to(source)).as_posix())
    return zip_path


def _checkpoint_archives(checkpoint_input: Path) -> list[Path]:
    if checkpoint_input.is_file():
        return [checkpoint_input] if checkpoint_input.suffix.lower() == ".zip" else []
    if checkpoint_input.is_dir():
        return sorted(checkpoint_input.rglob("*.zip"))
    return []


def _validate_checkpoint_identity(
    actual: dict[str, Any],
    *,
    dataset: str,
    experiment: str,
    configuration: str,
    seed: int,
    commit_sha: str,
    run_signature: dict[str, Any] | None,
) -> None:
    expected = {
        "dataset": dataset,
        "experiment": experiment,
        "configuration": configuration,
        "seed": int(seed),
        "commit_sha": commit_sha,
    }
    for key, value in expected.items():
        if actual.get(key) != value:
            raise RuntimeError(f"Checkpoint identity mismatch for {key}: {actual.get(key)!r} != {value!r}")
    if run_signature is not None and actual.get("run_signature") != run_signature:
        raise RuntimeError("Checkpoint run signature does not match the requested run signature")


def restore_checkpoint(
    checkpoint_input: str | Path,
    output_root: str | Path,
    *,
    dataset: str,
    experiment: str,
    configuration: str,
    seed: int,
    commit_sha: str,
    run_signature: dict[str, Any] | None = None,
) -> Path:
    """Validate and restore a checkpoint into its exact run path under ``output_root``."""
    source = Path(checkpoint_input)
    archives = _checkpoint_archives(source)
    if not archives:
        raise FileNotFoundError(f"No checkpoint ZIP found under {source}")
    relative = run_relative_path(experiment, dataset, configuration, seed)
    matching: list[tuple[Path, dict[str, Any]]] = []
    rejected: list[str] = []
    for archive_path in archives:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                if CHECKPOINT_MANIFEST not in archive.namelist():
                    continue
                manifest = json.loads(archive.read(CHECKPOINT_MANIFEST).decode("utf-8"))
            _validate_checkpoint_identity(
                manifest,
                dataset=dataset,
                experiment=experiment,
                configuration=configuration,
                seed=seed,
                commit_sha=commit_sha,
                run_signature=run_signature,
            )
            matching.append((archive_path, manifest))
        except RuntimeError as exc:
            rejected.append(f"{archive_path}: {exc}")
            continue
        except Exception as exc:
            raise RuntimeError(f"Invalid checkpoint archive {archive_path}: {exc}") from exc
    if not matching:
        details = f" Rejected: {'; '.join(rejected)}" if rejected else ""
        raise RuntimeError(f"No checkpoint matching {relative} and commit {commit_sha} found under {source}.{details}")
    if len(matching) > 1:
        raise RuntimeError(f"Multiple matching checkpoints found: {[str(path) for path, _ in matching]}")

    archive_path, manifest = matching[0]
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="checkpoint_restore_", dir=output.parent) as temporary:
        staging = Path(temporary)
        _safe_extract(archive_path, staging)
        staged_run = staging / relative
        if not staged_run.is_dir():
            raise RuntimeError(f"Checkpoint {archive_path} does not contain expected run path {relative}")
        staged_identity = _checkpoint_identity(staged_run)
        if staged_identity != {key: manifest.get(key) for key in staged_identity}:
            raise RuntimeError("Checkpoint manifest does not match the packaged run identity")
        _validate_checkpoint_identity(
            staged_identity,
            dataset=dataset,
            experiment=experiment,
            configuration=configuration,
            seed=seed,
            commit_sha=commit_sha,
            run_signature=run_signature,
        )
        destination = output / relative
        if destination.exists():
            existing_identity = _checkpoint_identity(destination)
            _validate_checkpoint_identity(
                existing_identity,
                dataset=dataset,
                experiment=experiment,
                configuration=configuration,
                seed=seed,
                commit_sha=commit_sha,
                run_signature=run_signature,
            )
        _copy_tree_read_only_source(staged_run, destination)
    return destination


def next_chunk_index(run_dir: str | Path, chunk_size: int) -> int:
    """Return the first not-fully-resolved test chunk for a restored run."""
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    run = Path(run_dir)
    predictions_path = run / "predictions.jsonl"
    resolved_ids: set[str] = set()
    if predictions_path.is_file():
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid predictions JSONL at line {line_number}: {predictions_path}") from exc
                if row.get("resolution_status", "resolved") == "resolved" and row.get("record_id") is not None:
                    resolved_ids.add(str(row["record_id"]))
    return len(resolved_ids) // int(chunk_size)


def write_run_summary(
    path: str | Path,
    *,
    run_dir: str | Path,
    commit_sha: str | None,
    seed: int,
    configuration: str,
) -> Path:
    run = Path(run_dir)
    metrics = read_json(run / "metrics.json") if (run / "metrics.json").exists() else {}
    metadata = read_json(run / "run_metadata.json") if (run / "run_metadata.json").exists() else {}
    selected = {
        key: metrics.get(key)
        for key in ("samples", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "llm_calls", "llm_call_ratio")
    }
    payload = {
        "output_path": str(run),
        "commit_sha": commit_sha,
        "seed": int(seed),
        "configuration": configuration,
        "status": metadata.get("status"),
        "metrics": selected,
    }
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return resolved


__all__ = [
    "CHECKPOINT_MANIFEST",
    "DEFAULT_REQUIRED",
    "PORTABLE_ARTIFACTS",
    "copy_run_artifacts",
    "find_run",
    "finalize_analysis_run",
    "initialize_analysis_run",
    "locate_or_materialize_run",
    "materialize_input_artifacts",
    "next_chunk_index",
    "package_checkpoint",
    "package_run",
    "read_json",
    "run_relative_path",
    "restore_checkpoint",
    "validate_run",
    "write_run_summary",
]
