from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

HAS_SPLIT_DEPS = importlib.util.find_spec("pandas") is not None and importlib.util.find_spec("sklearn") is not None
if HAS_SPLIT_DEPS:
    import pandas as pd


REVISION_DIR = Path(__file__).resolve().parents[1]
GRACE_ROOT = REVISION_DIR.parent
REPOSITORY_ROOT = GRACE_ROOT.parent
BASELINE2 = GRACE_ROOT / "baseline" / "baseline2"
sys.path.insert(0, str(REVISION_DIR))
sys.path.insert(0, str(BASELINE2))

from experiment_config import (  # noqa: E402
    ABLATION_VIEWS,
    RUN_SEEDS,
    ExperimentConfig,
    compute_test_split_fingerprint,
    config_for_configuration,
    config_for_run_seed,
    validate_paired_test_records,
    validate_split_integrity,
)
from metrics import compute_branch_metrics, paired_metric_deltas, validate_threshold_provenance  # noqa: E402


def load_split_module():
    spec = importlib.util.spec_from_file_location("revision_create_splits", BASELINE2 / "02_create_splits.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_calibration_module():
    stub = types.ModuleType("hybrid_prefilter")
    stub.DEFAULT_PREFILTER_MODEL_NAME = "fixture"
    stub.predict_feature_store = lambda *args, **kwargs: None
    previous = sys.modules.get("hybrid_prefilter")
    sys.modules["hybrid_prefilter"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "revision_calibrate_budget_controller", BASELINE2 / "05_calibrate_budget_controller.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("hybrid_prefilter", None)
        else:
            sys.modules["hybrid_prefilter"] = previous


def prediction(record_id: str, label: int, predicted: int, band: str, probability: float, llm: bool = False) -> dict:
    return {
        "record_id": record_id,
        "ground_truth": label,
        "prediction": predicted,
        "risk_band": band,
        "calibrated_probability": probability,
        "llm_called": llm,
    }


class RevisionSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = load_split_module() if HAS_SPLIT_DEPS else None
        cls.calibration = load_calibration_module()
        rows = []
        for index in range(180):
            rows.append({
                "record_id": f"r{index}",
                "code_hash": f"hash-{index // 2}",
                "label": (index // 2) % 2,
            })
        cls.frame = pd.DataFrame(rows) if HAS_SPLIT_DEPS else rows

    @unittest.skipUnless(HAS_SPLIT_DEPS, "pandas/scikit-learn are not installed in this lightweight environment")
    def test_split_implementation_is_deterministic(self):
        # training_seed is intentionally absent from _assign_splits.
        first = self.splits._assign_splits(self.frame, split_seed=123)
        second = self.splits._assign_splits(self.frame, split_seed=123)
        self.assertEqual(
            set(first.loc[first.split == "test", "record_id"]),
            set(second.loc[second.split == "test", "record_id"]),
        )

    @unittest.skipUnless(HAS_SPLIT_DEPS, "pandas/scikit-learn are not installed in this lightweight environment")
    def test_split_implementation_groups_code_hashes(self):
        assigned = self.splits._assign_splits(self.frame, split_seed=42)
        self.assertEqual(assigned.record_id.nunique(), len(assigned))
        by_hash = assigned.groupby("code_hash")["split"].nunique()
        self.assertTrue((by_hash == 1).all())

    def test_paired_arms_require_the_same_test_records_within_seed(self):
        rows = [
            {"record_id": "train-a", "code_hash": "h1", "split": "train"},
            {"record_id": "test-a", "code_hash": "h2", "split": "test"},
            {"record_id": "test-b", "code_hash": "h3", "split": "test"},
        ]
        fingerprint = validate_paired_test_records(rows, list(reversed(rows)))
        self.assertEqual(fingerprint, compute_test_split_fingerprint(rows))
        changed = rows[:-1] + [{"record_id": "test-c", "code_hash": "h4", "split": "test"}]
        with self.assertRaises(RuntimeError):
            validate_paired_test_records(rows, changed)

    def test_no_duplicate_record_id_and_no_code_hash_leakage(self):
        clean = [
            {"record_id": "a", "code_hash": "same", "split": "train"},
            {"record_id": "b", "code_hash": "same", "split": "train"},
            {"record_id": "c", "code_hash": "other", "split": "test"},
        ]
        validate_split_integrity(clean)
        with self.assertRaises(RuntimeError):
            validate_split_integrity(clean + [{"record_id": "a", "code_hash": "third", "split": "val"}])
        with self.assertRaises(RuntimeError):
            validate_split_integrity(clean + [{"record_id": "d", "code_hash": "same", "split": "test"}])

    def test_e01_run_seed_controls_split_and_training_for_both_arms(self):
        base = ExperimentConfig(experiment_name="E01_multiseed")
        for seed in RUN_SEEDS:
            seeded = config_for_run_seed(base, seed)
            baseline = config_for_configuration(seeded, "reproduced_baseline")
            selective = config_for_configuration(seeded, "proposed")
            for config in (baseline, selective):
                self.assertEqual(config.training_seed, seed)
                self.assertEqual(config.split_seed, seed)
                self.assertEqual(config.to_environment()["GRACE_SPLIT_SEED"], str(seed))
                self.assertEqual(config.to_environment()["GRACE_PREFILTER_RANDOM_SEED"], str(seed))
            self.assertEqual(baseline.split_seed, selective.split_seed)

    def test_different_run_seeds_may_have_different_grouped_splits(self):
        manifests = (
            [
                {"record_id": "a", "code_hash": "h1", "split": "train"},
                {"record_id": "b", "code_hash": "h2", "split": "val"},
                {"record_id": "c", "code_hash": "h3", "split": "test"},
            ],
            [
                {"record_id": "a", "code_hash": "h1", "split": "test"},
                {"record_id": "b", "code_hash": "h2", "split": "train"},
                {"record_id": "c", "code_hash": "h3", "split": "val"},
            ],
        )
        fingerprints = []
        for rows in manifests:
            validate_split_integrity(rows)
            validate_paired_test_records(rows, list(reversed(rows)))
            fingerprints.append(compute_test_split_fingerprint(rows))
        self.assertEqual(len(fingerprints), 2)

    def test_threshold_fallback_repairs_inclusive_high_threshold(self):
        probabilities = pd.Series([0.1, 0.4, 0.8]).to_numpy() if HAS_SPLIT_DEPS else __import__("numpy").array([0.1, 0.4, 0.8])
        labels = __import__("numpy").array([0, 1, 1])
        with patch.object(self.calibration, "ROUTING_RECALL_FLOOR", 2.0), \
             patch.object(self.calibration, "choose_low_threshold", return_value=0.4), \
             patch.object(self.calibration, "choose_best_f1_threshold", return_value=(0.4, 0.5)):
            tau_low, tau_high, strategy, _ = self.calibration._choose_routing_thresholds(probabilities, labels)
        self.assertEqual(strategy, "fallback_f1")
        self.assertTrue(0.0 <= tau_low < tau_high <= 1.0)

    def test_threshold_overrides_reject_equal_or_reversed_values(self):
        with self.assertRaises(ValueError):
            self.calibration._validate_threshold_overrides(0.5, 0.5)
        with self.assertRaises(ValueError):
            self.calibration._validate_threshold_overrides(0.7, 0.3)

    def test_routing_counts_and_required_branch_fields(self):
        rows = [
            prediction("a", 1, 0, "skip", 0.1),
            prediction("b", 0, 1, "high", 0.9),
            prediction("c", 1, 1, "inspect", 0.5, True),
        ]
        result = compute_branch_metrics(rows)
        self.assertEqual(sum(branch["sample_count"] for branch in result["branches"].values()), 3)
        self.assertEqual(result["skip_false_negatives"], 1)
        self.assertEqual(result["high_false_positives"], 1)
        with self.assertRaises(RuntimeError):
            compute_branch_metrics(rows + [prediction("d", 0, 0, "unknown", 0.2)])

    def test_thresholds_must_come_from_validation(self):
        validate_threshold_provenance({"threshold_selection_split": "validation"})
        with self.assertRaises(RuntimeError):
            validate_threshold_provenance({"threshold_selection_split": "test"})

    def test_baseline_and_proposed_use_identical_test_records(self):
        baseline = [prediction("a", 0, 0, "inspect", 0.2, True), prediction("b", 1, 0, "inspect", 0.4, True)]
        proposed = [prediction("b", 1, 1, "high", 0.8), prediction("a", 0, 0, "skip", 0.1)]
        comparison = paired_metric_deltas(baseline, proposed)
        self.assertTrue(comparison["record_ids_aligned"])
        self.assertEqual(comparison["aligned_samples"], 2)
        with self.assertRaises(RuntimeError):
            paired_metric_deltas(baseline, proposed[:1])

    def test_view_ablation_removes_expected_model_input(self):
        for name, views in ABLATION_VIEWS.items():
            config = ExperimentConfig(experiment_name="E02_leave_one_view_out", configuration=name, enabled_views=views)
            env = config.to_environment()
            enabled = set(env["GRACE_ENABLED_VIEWS"].split(","))
            self.assertEqual(enabled, set(views))
        self.assertNotIn("token", ABLATION_VIEWS["no_token"])
        self.assertNotIn("ast", ABLATION_VIEWS["no_ast"])
        self.assertNotIn("semantic", ABLATION_VIEWS["no_semantic"])
        self.assertNotIn("graph_numeric", ABLATION_VIEWS["no_graph_numeric"])

    def test_experiment_outputs_do_not_overlap(self):
        a = ExperimentConfig(experiment_name="E01_multiseed", dataset_name="devign", configuration="proposed", training_seed=1)
        b = ExperimentConfig(experiment_name="E01_multiseed", dataset_name="devign", configuration="proposed", training_seed=7)
        c = ExperimentConfig(experiment_name="E02_leave_one_view_out", dataset_name="devign", configuration="full", training_seed=1)
        self.assertEqual(len({a.run_directory, b.run_directory, c.run_directory}), 3)

    def test_rerun_handoff_layout_is_dataset_seed_and_arm(self):
        base = ExperimentConfig(
            experiment_name="E01_multiseed", dataset_name="devign", training_seed=7,
            split_seed=7, output_directory="outputs/rerun_2408",
        )
        baseline = config_for_configuration(base, "reproduced_baseline")
        selective = config_for_configuration(base, "proposed")
        self.assertEqual(baseline.run_directory, Path("outputs/rerun_2408/devign/7/baseline"))
        self.assertEqual(selective.run_directory, Path("outputs/rerun_2408/devign/7/selective"))

    def test_readme_notebook_links_and_runseed_artifact(self):
        readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("GRACE-improve/grace-improve.ipynb", readme)
        for relative in (
            "GRACE-improve/grace-improve-devign.ipynb",
            "GRACE-improve/grace-improve-bigvul.ipynb",
            "GRACE-improve/grace-improve-reveal.ipynb",
            "outputs/runseed_metrics.csv",
        ):
            self.assertIn(relative, readme)
            self.assertTrue((REPOSITORY_ROOT / relative).is_file())

    def test_runseed_metrics_historical_csv_shape(self):
        with (REPOSITORY_ROOT / "outputs" / "runseed_metrics.csv").open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        keys = [(row["dataset"], int(row["repetition"]), int(row["seed"])) for row in rows]
        self.assertEqual(len(rows), 75)
        self.assertEqual(len({dataset for dataset, _, _ in keys}), 3)
        self.assertEqual({seed for _, _, seed in keys}, {1, 7, 21, 42, 100})
        self.assertEqual(len(keys), len(set(keys)))
        for dataset in {dataset for dataset, _, _ in keys}:
            dataset_rows = [(repetition, seed) for current, repetition, seed in keys if current == dataset]
            self.assertEqual({repetition for repetition, _ in dataset_rows}, {1, 2, 3, 4, 5})
            for repetition in range(1, 6):
                self.assertEqual(
                    {seed for current, seed in dataset_rows if current == repetition},
                    {1, 7, 21, 42, 100},
                )

    def test_full_pipeline_notebook_has_seed_wiring_and_strict_guards(self):
        notebook = json.loads((REPOSITORY_ROOT / "full_pipeline.ipynb").read_text(encoding="utf-8"))
        source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("os.environ['GRACE_SPLIT_SEED'] = str(RUN_SEED)", source)
        self.assertIn("os.environ['GRACE_PREFILTER_RANDOM_SEED'] = str(RUN_SEED)", source)
        self.assertGreaterEqual(source.count("if tau_high <= tau_low:"), 4)
        self.assertGreaterEqual(source.count("tau_high = min(1.0, tau_low + 1e-6)"), 2)

    def test_original_notebooks_are_byte_identical(self):
        pairs = (
            (GRACE_ROOT / "grace-improve-devign.ipynb", GRACE_ROOT / "original" / "improve_devign_original.ipynb"),
            (GRACE_ROOT / "grace-improve-bigvul.ipynb", GRACE_ROOT / "original" / "improve_bigvul_original.ipynb"),
            (GRACE_ROOT / "grace-improve-reveal.ipynb", GRACE_ROOT / "original" / "improve_reveal_original.ipynb"),
        )
        for source, preserved in pairs:
            self.assertEqual(source.read_bytes(), preserved.read_bytes())


if __name__ == "__main__":
    unittest.main()
