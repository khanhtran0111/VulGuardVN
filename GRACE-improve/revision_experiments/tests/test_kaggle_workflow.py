from __future__ import annotations

import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


REVISION_DIR = Path(__file__).resolve().parents[1]
GRACE_ROOT = REVISION_DIR.parent
NOTEBOOK_DIR = GRACE_ROOT / "kaggle_notebooks"
sys.path.insert(0, str(REVISION_DIR))

from analyze_branch_results import analyze as analyze_branches  # noqa: E402
from analyze_calibration_results import analyze as analyze_calibration  # noqa: E402
from analyze_routing_results import analyze as analyze_routing  # noqa: E402
from analyze_runtime_results import analyze as analyze_runtime  # noqa: E402
from experiment_config import ExperimentConfig, config_for_configuration  # noqa: E402
from kaggle_artifacts import (  # noqa: E402
    locate_or_materialize_run,
    next_chunk_index,
    package_checkpoint,
    package_run,
    restore_checkpoint,
    validate_run,
)
from kaggle_preflight import (  # noqa: E402
    prepare_devign_input,
    prepare_qwen_model,
    prepare_semantic_model,
)
import run_revision_experiments as revision_runner  # noqa: E402


EXPECTED_NOTEBOOKS = (
    "01_devign_E01_multiseed_kaggle.ipynb",
    "02_devign_E02_leave_one_view_out_kaggle.ipynb",
    "03_devign_E03_branch_analysis_kaggle.ipynb",
    "04_devign_E04_calibration_distribution_kaggle.ipynb",
    "05_devign_E05_routing_policy_kaggle.ipynb",
    "06_devign_E06_runtime_cost_kaggle.ipynb",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_run(root: Path, configuration: str) -> Path:
    run = root / "E01_multiseed" / "devign" / configuration / "seed_42"
    run.mkdir(parents=True, exist_ok=True)
    predictions = [
        {"record_id": "a", "ground_truth": 0, "prediction": 0, "risk_band": "skip", "calibrated_probability": .1, "llm_called": False, "decision_source": "prefilter"},
        {"record_id": "b", "ground_truth": 1, "prediction": 0, "risk_band": "skip", "calibrated_probability": .2, "llm_called": False, "decision_source": "prefilter"},
        {"record_id": "c", "ground_truth": 1, "prediction": 1, "risk_band": "inspect", "calibrated_probability": .6, "llm_called": True, "decision_source": "llm"},
        {"record_id": "d", "ground_truth": 0, "prediction": 1, "risk_band": "high", "calibrated_probability": .9, "llm_called": configuration == "reproduced_baseline", "decision_source": "llm" if configuration == "reproduced_baseline" else "prefilter"},
    ]
    (run / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8")
    write_json(run / "config.json", {"dataset_name": "devign", "experiment_name": "E01_multiseed", "configuration": configuration, "training_seed": 42, "call_llm_for_inspect": True, "call_llm_for_high": configuration == "reproduced_baseline", "delta_high": 0 if configuration == "reproduced_baseline" else 1, "force_inspect_all": configuration == "reproduced_baseline"})
    write_json(run / "run_metadata.json", {"status": "complete", "commit_sha": "fixture", "dataset": "devign", "experiment": "E01_multiseed", "configuration": configuration, "training_seed": 42})
    write_json(run / "metrics.json", {"samples": 4, "accuracy": .5, "precision": .5, "recall": .5, "f1": .5, "roc_auc": .75, "pr_auc": .7, "llm_calls": 4 if configuration == "reproduced_baseline" else 1, "llm_call_ratio": 1.0 if configuration == "reproduced_baseline" else .25})
    write_json(run / "calibration.json", {
        "calibration_method": "platt", "calibrator": {"method": "platt", "parameters": {"coef": 1.0, "intercept": 0.0}},
        "tau_low": .25, "tau_high": .8, "threshold_selection_split": "validation", "threshold_selection_strategy": "fixture",
        "calibration_metrics": {"ece": .1, "brier": .2, "nll": .3},
        "probability_records": [
            {"record_id": row["record_id"], "ground_truth": row["ground_truth"], "raw_fusion_score": row["calibrated_probability"], "calibrated_probability": row["calibrated_probability"], "calibration_method": "platt"}
            for row in predictions
        ],
    })
    write_json(run / "branch_metrics.json", {"total_samples": 4})
    write_json(run / "runtime.json", {
        "dataset_preprocessing_time_ms": 100.0, "token_ast_feature_extraction_time_ms": 40.0,
        "unixcoder_embedding_time_ms": 80.0, "numeric_graph_feature_extraction_time_ms": 20.0,
        "prefilter_inference_time_ms": 10.0, "calibration_routing_time_ms": 2.0,
        "graph_extraction_time_ms": 30.0, "retrieval_time_ms": 25.0, "llm_inference_time_ms": 200.0,
        "total_runtime_ms": 507.0 if configuration == "reproduced_baseline" else 307.0,
        "llm_call_count": 4 if configuration == "reproduced_baseline" else 1,
        "llm_call_ratio": 1.0 if configuration == "reproduced_baseline" else .25,
        "peak_gpu_memory_mb": None,
    })
    write_json(run / "_pipeline" / "run_state.json", {
        "complete": True,
        "resolved_samples": 4,
        "target_samples": 4,
        "run_signature": {"dataset": "devign", "configuration": configuration, "tau_low": .25, "tau_high": .8},
    })
    write_json(run / "_pipeline" / "stage_state.json", {"completed": ["inference", "evaluate"]})
    return run


class KaggleWorkflowTests(unittest.TestCase):
    def test_exact_notebook_set_and_single_central_config(self):
        self.assertEqual(tuple(sorted(path.name for path in NOTEBOOK_DIR.glob("*.ipynb"))), EXPECTED_NOTEBOOKS)
        for name in EXPECTED_NOTEBOOKS:
            notebook = json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))
            sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
            config_cells = [source for source in sources if 'REPOSITORY_URL = "https://github.com/khanhtran0111/VulGuardVN.git"' in source]
            self.assertEqual(len(config_cells), 1, name)
            self.assertIn('RUN_MODE = "smoke"', config_cells[0])
            self.assertIn('DATASET = "devign"', config_cells[0])
            self.assertIn("## A.", sources[0])
            self.assertEqual(sum("run_revision_experiments.py" in source for source in sources), 1 if name.startswith(("01_", "02_", "05_")) else 0)
            joined = "\n".join(sources)
            if name.startswith(("01_", "02_", "05_")):
                self.assertIn("DEVIGN_INPUT", config_cells[0])
                self.assertIn("SEMANTIC_MODEL_INPUT", config_cells[0])
                self.assertIn("AUTO_DOWNLOAD_SEMANTIC_MODEL", config_cells[0])
                self.assertIn("QWEN_MODEL_INPUT", config_cells[0])
                self.assertIn("prepare_devign_input", joined)
                self.assertIn("prepare_semantic_model", joined)
                self.assertIn("prepare_qwen_model", joined)
            else:
                self.assertNotIn("prepare_semantic_model", joined)
                self.assertNotIn("prepare_qwen_model", joined)

    def test_force_inspect_all_is_explicit(self):
        base = ExperimentConfig(experiment_name="E01_multiseed")
        baseline = config_for_configuration(base, "reproduced_baseline")
        self.assertTrue(baseline.force_inspect_all)
        self.assertTrue(baseline.call_llm_for_inspect)
        self.assertEqual(baseline.to_environment()["GRACE_FORCE_INSPECT_ALL"], "true")
        self.assertIsNone(baseline.tau_low)
        self.assertIsNone(baseline.tau_high)

    def test_analysis_only_fixture_and_portable_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "results"
            proposed = make_run(root, "proposed"); baseline = make_run(root, "reproduced_baseline")
            e03 = root / "E03_branch_analysis" / "devign" / "proposed" / "seed_42"
            branch = analyze_branches(proposed / "predictions.jsonl", e03, "devign", 42)
            self.assertEqual(branch["total_samples"], 4)
            self.assertEqual(branch["skip_false_negatives"], 1)
            e04 = root / "E04_calibration_distribution" / "devign" / "proposed" / "seed_42"
            calibration = analyze_calibration(proposed / "calibration.json", e04, "devign", 42)
            self.assertEqual(calibration["threshold_selection_split"], "validation")
            self.assertTrue((e04 / "probability_histogram.png").is_file())
            e05 = root / "E05_routing_policy" / "devign" / "direct_high" / "seed_42"
            routing = analyze_routing(proposed, e05, "devign", 42, baseline)
            self.assertEqual(routing["operating_points"], 26)
            e06 = root / "E06_runtime_cost" / "devign" / "proposed" / "seed_42"
            runtime = analyze_runtime(proposed, baseline, e06, "devign", 42)
            self.assertIsNotNone(runtime["speedup_vs_reproduced_baseline"])
            zip_path = package_run(proposed, Path(temporary) / "exports", dataset="devign", experiment="E01_multiseed", configuration="proposed", seed=42)
            self.assertTrue(zipfile.is_zipfile(zip_path))
            with zipfile.ZipFile(zip_path) as archive:
                self.assertFalse(any("_pipeline" in name for name in archive.namelist()))

    def test_materialize_uploaded_zip_without_writing_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); source_results = base / "source"
            proposed = make_run(source_results, "proposed")
            archive = package_run(proposed, base / "input", dataset="devign", experiment="E01_multiseed", configuration="proposed", seed=42)
            before = archive.read_bytes(); destination_results = base / "working" / "revision_results"
            located = locate_or_materialize_run(results_root=destination_results, experiment="E01_multiseed", dataset="devign", configuration="proposed", seed=42, input_path=archive, staging_root=base / "working" / "staging")
            validate_run(located, required=("config.json", "run_metadata.json", "metrics.json", "predictions.jsonl", "calibration.json", "runtime.json"), require_complete=True)
            self.assertEqual(before, archive.read_bytes())

    def test_checkpoint_packages_pipeline_and_restores_exact_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            source = make_run(base / "source", "proposed")
            checkpoint = package_checkpoint(
                source,
                base / "input",
                dataset="devign",
                experiment="E01_multiseed",
                configuration="proposed",
                seed=42,
            )
            with zipfile.ZipFile(checkpoint) as archive:
                names = archive.namelist()
                self.assertTrue(any(name.endswith("/_pipeline/run_state.json") for name in names))
                self.assertIn("checkpoint_manifest.json", names)

            restored = restore_checkpoint(
                checkpoint,
                base / "working" / "revision_results",
                dataset="devign",
                experiment="E01_multiseed",
                configuration="proposed",
                seed=42,
                commit_sha="fixture",
            )
            self.assertTrue((restored / "_pipeline" / "run_state.json").is_file())
            self.assertEqual(next_chunk_index(restored, 2), 2)
            with self.assertRaisesRegex(RuntimeError, "commit_sha"):
                restore_checkpoint(
                    checkpoint,
                    base / "other_results",
                    dataset="devign",
                    experiment="E01_multiseed",
                    configuration="proposed",
                    seed=42,
                    commit_sha="different-commit",
                )

    def test_partial_inference_is_not_completed_and_skips_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = ExperimentConfig(
                dataset_name="devign",
                experiment_name="E01_multiseed",
                configuration="proposed",
                training_seed=42,
                output_directory=str(Path(temporary) / "results"),
            )
            invoked: list[str] = []
            chunk_environment: dict[str, str] = {}

            def fake_stage(command, **kwargs):
                invoked.append(Path(command[1]).name)
                chunk_environment.update(kwargs["env"])
                run = config.run_directory
                write_json(run / "_pipeline" / "run_state.json", {
                    "complete": False,
                    "resolved_samples": 2,
                    "target_samples": 5,
                    "run_signature": {"dataset": "devign"},
                })
                (run / "predictions.jsonl").write_text(
                    "".join(json.dumps({"record_id": value, "resolution_status": "resolved"}) + "\n" for value in ("a", "b")),
                    encoding="utf-8",
                )
                return SimpleNamespace(returncode=0)

            def fake_metadata(_config, status, **extra):
                return {
                    "dataset": "devign", "experiment": "E01_multiseed", "configuration": "proposed",
                    "training_seed": 42, "commit_sha": "fixture", "status": status, **extra,
                }

            with patch.object(revision_runner.subprocess, "run", side_effect=fake_stage), \
                 patch.object(revision_runner, "build_metadata", side_effect=fake_metadata), \
                 patch.object(revision_runner, "_collect_runtime", return_value={}):
                status = revision_runner.run_one(
                    config,
                    dry_run=False,
                    resume=True,
                    smoke=False,
                    selected_stages={"inference", "evaluate"},
                    test_chunk_size=2,
                    test_chunk_index=0,
                )

            self.assertEqual(status, "partial")
            self.assertEqual(invoked, ["07_run_grace_hybrid.py"])
            self.assertEqual(chunk_environment["GRACE_TEST_CHUNK_SIZE"], "2")
            self.assertEqual(chunk_environment["GRACE_TEST_CHUNK_INDEX"], "0")
            stage_state = json.loads((config.run_directory / "_pipeline" / "stage_state.json").read_text())
            self.assertNotIn("inference", stage_state["completed"])
            self.assertEqual(stage_state["stage_status"]["inference"], "partial")
            self.assertEqual(stage_state["next_test_chunk_index"], 1)
            metadata = json.loads((config.run_directory / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "partial")

    def test_kaggle_model_preflight_copies_devign_and_accepts_mounted_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            devign_input = root / "input" / "devign"
            write_json(devign_input / "function.json", [{"func": "int f(){return 0;}", "target": 0}])
            destination = prepare_devign_input(devign_input, root / "repo")
            self.assertEqual(destination, root / "repo" / "GRACE-improve" / "data" / "function.json")
            self.assertTrue(destination.is_file())

            semantic = root / "input" / "semantic" / "snapshot"
            write_json(semantic / "config.json", {})
            (semantic / "tokenizer.json").write_text("{}", encoding="utf-8")
            (semantic / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(
                prepare_semantic_model(semantic.parent, root / "repo", auto_download=False),
                semantic,
            )

            qwen = root / "input" / "qwen"
            write_json(qwen / "config.json", {})
            (qwen / "model.safetensors").write_bytes(b"fixture")
            self.assertEqual(
                prepare_qwen_model(qwen, root / "repo", auto_download=False, required=True),
                qwen,
            )
            self.assertIsNone(
                prepare_qwen_model(root / "missing", root / "repo", auto_download=False, required=False)
            )

    def test_kaggle_preflight_fails_clearly_when_devign_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(FileNotFoundError, "Devign dataset was not found"):
                prepare_devign_input(Path(temporary) / "missing", Path(temporary) / "repo")


if __name__ == "__main__":
    unittest.main()
