from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

HAS_SPLIT_DEPS = importlib.util.find_spec("pandas") is not None and importlib.util.find_spec("sklearn") is not None
if HAS_SPLIT_DEPS:
    import pandas as pd


REVISION_DIR = Path(__file__).resolve().parents[1]
GRACE_ROOT = REVISION_DIR.parent
BASELINE2 = GRACE_ROOT / "baseline" / "baseline2"
sys.path.insert(0, str(REVISION_DIR))
sys.path.insert(0, str(BASELINE2))

from experiment_config import ABLATION_VIEWS, ExperimentConfig, validate_fixed_test_records, validate_split_integrity  # noqa: E402
from metrics import compute_branch_metrics, paired_metric_deltas, validate_threshold_provenance  # noqa: E402


def load_split_module():
    spec = importlib.util.spec_from_file_location("revision_create_splits", BASELINE2 / "02_create_splits.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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

    def test_fixed_test_records_across_training_seed_manifests(self):
        rows = [
            {"record_id": "train-a", "code_hash": "h1", "split": "train"},
            {"record_id": "test-a", "code_hash": "h2", "split": "test"},
            {"record_id": "test-b", "code_hash": "h3", "split": "test"},
        ]
        validate_fixed_test_records({1: rows, 7: list(reversed(rows)), 100: list(rows)})
        changed = rows[:-1] + [{"record_id": "test-c", "code_hash": "h4", "split": "test"}]
        with self.assertRaises(RuntimeError):
            validate_fixed_test_records({1: rows, 7: changed})

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

    def test_split_seed_is_fixed_while_training_seed_changes(self):
        first = ExperimentConfig(training_seed=1, split_seed=42)
        second = ExperimentConfig(training_seed=100, split_seed=42)
        first_env, second_env = first.to_environment(), second.to_environment()
        self.assertEqual(first_env["GRACE_SPLIT_SEED"], second_env["GRACE_SPLIT_SEED"])
        self.assertNotEqual(first_env["GRACE_PREFILTER_RANDOM_SEED"], second_env["GRACE_PREFILTER_RANDOM_SEED"])

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
