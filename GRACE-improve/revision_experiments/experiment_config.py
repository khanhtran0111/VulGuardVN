"""Typed, shared configuration for APSIPA revision experiments.

The four random seeds deliberately have separate fields.  In particular,
``split_seed`` is never derived from ``training_seed`` so every model seed is
evaluated on the same held-out records.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable


DATASETS = ("devign", "bigvul", "reveal")
EXPERIMENTS = (
    "E01_multiseed",
    "E02_leave_one_view_out",
    "E03_branch_analysis",
    "E04_calibration_distribution",
    "E05_routing_policy",
    "E06_runtime_cost",
)
TRAINING_SEEDS = (1, 7, 21, 42, 100)
ALL_VIEWS = ("token", "ast", "semantic", "graph_numeric")
ABLATION_VIEWS = {
    "full": ALL_VIEWS,
    "no_token": tuple(view for view in ALL_VIEWS if view != "token"),
    "no_ast": tuple(view for view in ALL_VIEWS if view != "ast"),
    "no_semantic": tuple(view for view in ALL_VIEWS if view != "semantic"),
    "no_graph_numeric": tuple(view for view in ALL_VIEWS if view != "graph_numeric"),
}
ROUTING_POLICIES = {
    "direct_high": {"call_llm_for_inspect": True, "call_llm_for_high": False},
    "verify_high": {"call_llm_for_inspect": True, "call_llm_for_high": True},
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def parse_views(value: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        views = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        views = tuple(str(item).strip() for item in value if str(item).strip())
    unknown = set(views) - set(ALL_VIEWS)
    if unknown:
        raise ValueError(f"Unknown view(s): {sorted(unknown)}; expected a subset of {ALL_VIEWS}")
    if not views:
        raise ValueError("enabled_views cannot be empty")
    return tuple(view for view in ALL_VIEWS if view in views)


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_name: str = "devign"
    experiment_name: str = "E01_multiseed"
    split_seed: int = 42
    training_seed: int = 42
    demo_seed: int = 31415
    bootstrap_seed: int = 27182
    model_name: str = "hybrid_multiview_prefilter"
    calibration_method: str = "auto"
    tau_low: float | None = None
    tau_high: float | None = None
    target_recall: float = 0.995
    call_llm_for_inspect: bool = True
    call_llm_for_high: bool = False
    force_inspect_all: bool = False
    enabled_views: tuple[str, ...] = ALL_VIEWS
    output_directory: str = "GRACE-improve/revision_results"
    graph_backend: str = "auto"
    retrieval_backend: str = "auto"
    configuration: str = "proposed"

    def __post_init__(self) -> None:
        object.__setattr__(self, "dataset_name", self.dataset_name.strip().lower())
        object.__setattr__(self, "enabled_views", parse_views(self.enabled_views))
        if self.dataset_name not in DATASETS:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        if self.experiment_name not in EXPERIMENTS:
            raise ValueError(f"Unsupported experiment: {self.experiment_name}")
        if self.tau_low is not None and not 0.0 <= self.tau_low <= 1.0:
            raise ValueError("tau_low must be in [0, 1]")
        if self.tau_high is not None and not 0.0 <= self.tau_high <= 1.0:
            raise ValueError("tau_high must be in [0, 1]")
        if self.tau_low is not None and self.tau_high is not None and self.tau_low >= self.tau_high:
            raise ValueError("tau_low must be smaller than tau_high")
        if not 0.0 < self.target_recall <= 1.0:
            raise ValueError("target_recall must be in (0, 1]")
        if Path(self.output_directory).name == "artifacts":
            raise ValueError("Revision output_directory must not point at a legacy artifacts directory")

    @property
    def run_directory(self) -> Path:
        return (
            Path(self.output_directory)
            / self.experiment_name
            / self.dataset_name
            / self.configuration
            / f"seed_{self.training_seed}"
        )

    @property
    def is_direct_high(self) -> bool:
        return not self.call_llm_for_high

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["enabled_views"] = list(self.enabled_views)
        payload["run_directory"] = str(self.run_directory)
        payload["delta_high"] = 0 if self.call_llm_for_high else 1
        return payload

    def to_environment(self) -> dict[str, str]:
        internal = self.run_directory / "_pipeline"
        env = {
            "GRACE_DATASET": self.dataset_name,
            "GRACE_SPLIT_SEED": str(self.split_seed),
            "GRACE_PREFILTER_RANDOM_SEED": str(self.training_seed),
            "GRACE_DEMO_SEED": str(self.demo_seed),
            "GRACE_BOOTSTRAP_SEED": str(self.bootstrap_seed),
            "GRACE_PREFILTER_MODEL_NAME": self.model_name,
            "GRACE_CALIBRATION_METHOD": self.calibration_method,
            "GRACE_TARGET_RECALL": str(self.target_recall),
            "GRACE_CALL_LLM_FOR_INSPECT": str(self.call_llm_for_inspect).lower(),
            "GRACE_CALL_LLM_FOR_HIGH": str(self.call_llm_for_high).lower(),
            "GRACE_FORCE_INSPECT_ALL": str(self.force_inspect_all).lower(),
            "GRACE_ENABLED_VIEWS": ",".join(self.enabled_views),
            "GRACE_GRAPH_BACKEND": self.graph_backend,
            "GRACE_RETRIEVAL_BACKEND": self.retrieval_backend,
            "GRACE_EXPERIMENT_NAME": self.experiment_name,
            "GRACE_CONFIGURATION": self.configuration,
            "GRACE_RUN_OUTPUT_DIR": str(self.run_directory),
            "GRACE_ARTIFACTS_DIR": str(internal),
            "GRACE_FEATURES_DIR": str(internal / "features"),
            "GRACE_PROCESSED_DIR": str(internal / "processed"),
            "GRACE_MODELS_DIR": str(internal / "models"),
            "GRACE_RETRIEVAL_DIR": str(internal / "retrieval"),
            "GRACE_PREDICTIONS_DIR": str(internal / "predictions"),
            "GRACE_METRICS_DIR": str(internal / "metrics"),
            "GRACE_CACHE_DIR": str(internal / "cache"),
            "GRACE_SPLITS_DIR": str(internal / "splits"),
            "GRACE_GRAPH_DIR": str(internal / "graphs"),
            "GRACE_CALIBRATION_OUTPUT_PATH": str(self.run_directory / "calibration.json"),
            "GRACE_PREDICTIONS_PATH": str(self.run_directory / "predictions.jsonl"),
            "GRACE_RUN_STATE_PATH": str(internal / "run_state.json"),
            "GRACE_EVALUATION_OUTPUT_PATH": str(self.run_directory / "metrics.json"),
            "GRACE_BRANCH_METRICS_OUTPUT_PATH": str(self.run_directory / "branch_metrics.json"),
            "GRACE_FIGURES_DIR": str(self.run_directory / "figures"),
            "GRACE_DEMO_BANK_PATH": str(internal / "retrieval" / self.dataset_name / "demo_bank.joblib"),
        }
        if self.tau_low is not None:
            env["GRACE_TAU_LOW"] = str(self.tau_low)
        if self.tau_high is not None:
            env["GRACE_TAU_HIGH"] = str(self.tau_high)
        return env

    def with_updates(self, **updates: Any) -> "ExperimentConfig":
        return replace(self, **updates)


def from_mapping(payload: dict[str, Any]) -> ExperimentConfig:
    valid_names = {field.name for field in fields(ExperimentConfig)}
    payload = {key: value for key, value in payload.items() if key not in {"run_directory", "delta_high"}}
    unknown = set(payload) - valid_names
    if unknown:
        raise ValueError(f"Unknown experiment configuration keys: {sorted(unknown)}")
    values = dict(payload)
    if "enabled_views" in values:
        values["enabled_views"] = parse_views(values["enabled_views"])
    for name in ("call_llm_for_inspect", "call_llm_for_high", "force_inspect_all"):
        if name in values:
            values[name] = _as_bool(values[name])
    for name in ("split_seed", "training_seed", "demo_seed", "bootstrap_seed"):
        if name in values:
            values[name] = int(values[name])
    return ExperimentConfig(**values)


def load_config(path: str | Path | None = None, **overrides: Any) -> ExperimentConfig:
    payload: dict[str, Any] = {}
    if path:
        payload.update(json.loads(Path(path).read_text(encoding="utf-8")))
    payload.update({key: value for key, value in overrides.items() if value is not None})
    return from_mapping(payload)


def from_environment(default: ExperimentConfig | None = None) -> ExperimentConfig:
    base = (default or ExperimentConfig()).to_dict()
    base.pop("run_directory", None)
    mapping = {
        "GRACE_DATASET": "dataset_name",
        "GRACE_EXPERIMENT_NAME": "experiment_name",
        "GRACE_SPLIT_SEED": "split_seed",
        "GRACE_PREFILTER_RANDOM_SEED": "training_seed",
        "GRACE_DEMO_SEED": "demo_seed",
        "GRACE_BOOTSTRAP_SEED": "bootstrap_seed",
        "GRACE_PREFILTER_MODEL_NAME": "model_name",
        "GRACE_CALIBRATION_METHOD": "calibration_method",
        "GRACE_TAU_LOW": "tau_low",
        "GRACE_TAU_HIGH": "tau_high",
        "GRACE_TARGET_RECALL": "target_recall",
        "GRACE_CALL_LLM_FOR_INSPECT": "call_llm_for_inspect",
        "GRACE_CALL_LLM_FOR_HIGH": "call_llm_for_high",
        "GRACE_FORCE_INSPECT_ALL": "force_inspect_all",
        "GRACE_ENABLED_VIEWS": "enabled_views",
        "GRACE_RUN_OUTPUT_DIR": "output_directory",
        "GRACE_GRAPH_BACKEND": "graph_backend",
        "GRACE_RETRIEVAL_BACKEND": "retrieval_backend",
        "GRACE_CONFIGURATION": "configuration",
    }
    for env_name, field_name in mapping.items():
        if os.getenv(env_name) not in (None, ""):
            base[field_name] = os.environ[env_name]
    # GRACE_RUN_OUTPUT_DIR is a concrete run path, not the revision root.  Do
    # not reinterpret it when a default config already supplies the root.
    if os.getenv("GRACE_RUN_OUTPUT_DIR"):
        base["output_directory"] = (default or ExperimentConfig()).output_directory
    return from_mapping(base)


def configurations_for(experiment_name: str) -> tuple[str, ...]:
    if experiment_name == "E01_multiseed":
        return ("reproduced_baseline", "proposed")
    if experiment_name == "E02_leave_one_view_out":
        return tuple(ABLATION_VIEWS)
    if experiment_name == "E05_routing_policy":
        return ("reproduced_baseline", *ROUTING_POLICIES)
    return ("proposed",)


def validate_split_integrity(rows: Iterable[dict[str, Any]]) -> None:
    """Reject duplicate records and code-hash leakage across split names."""
    seen_ids: set[str] = set()
    hash_to_split: dict[str, str] = {}
    for row in rows:
        record_id = str(row["record_id"])
        split = str(row["split"])
        code_hash = str(row["code_hash"])
        if record_id in seen_ids:
            raise RuntimeError(f"Duplicate record_id in split manifest: {record_id}")
        seen_ids.add(record_id)
        previous = hash_to_split.setdefault(code_hash, split)
        if previous != split:
            raise RuntimeError(f"code_hash leakage for {code_hash}: {previous} and {split}")


def validate_fixed_test_records(manifests_by_training_seed: dict[int, Iterable[dict[str, Any]]]) -> None:
    """Ensure every training seed has exactly the same test record_id set."""
    reference_seed: int | None = None
    reference: set[str] | None = None
    for seed, rows in sorted(manifests_by_training_seed.items()):
        current = {str(row["record_id"]) for row in rows if str(row.get("split")) == "test"}
        if reference is None:
            reference_seed, reference = seed, current
        elif current != reference:
            raise RuntimeError(
                f"Test split changed between training_seed={reference_seed} and training_seed={seed}: "
                f"symmetric_difference={len(current ^ reference)}"
            )


def config_for_configuration(config: ExperimentConfig, configuration: str) -> ExperimentConfig:
    updates: dict[str, Any] = {"configuration": configuration}
    if configuration == "reproduced_baseline":
        # Non-selective local GRACE uses an explicit routing override.  This
        # also covers calibrated probabilities that are exactly 0 or 1.
        updates.update({"call_llm_for_inspect": True, "call_llm_for_high": True, "force_inspect_all": True})
    if config.experiment_name == "E02_leave_one_view_out":
        updates["enabled_views"] = ABLATION_VIEWS[configuration]
    if config.experiment_name == "E05_routing_policy" and configuration in ROUTING_POLICIES:
        updates.update(ROUTING_POLICIES[configuration])
    return config.with_updates(**updates)


__all__ = [
    "ABLATION_VIEWS",
    "ALL_VIEWS",
    "DATASETS",
    "EXPERIMENTS",
    "ExperimentConfig",
    "ROUTING_POLICIES",
    "TRAINING_SEEDS",
    "config_for_configuration",
    "configurations_for",
    "from_environment",
    "from_mapping",
    "load_config",
    "parse_views",
    "validate_fixed_test_records",
    "validate_split_integrity",
]
