# VulGuardVN

## Overview

VulGuardVN is an experimental workspace for function-level vulnerability detection in C/C++ under a GRACE-style pipeline. The current implementation lives in `GRACE-improve/baseline` and focuses on a practical local cascade rather than a pure paper reproduction.

Current pipeline:

`prefilter -> calibration + risk routing -> CodeT5 retrieval -> graph extraction -> local VulnLLM-R-7B -> evaluation`

The repository already contains one completed end-to-end run on `devign`, so this README documents the current baseline from the actual artifacts stored under `GRACE-improve/baseline/artifacts`.

## Scope and Datasets

The current baseline supports three benchmarks:

- `devign`
- `bigvul`
- `reveal`

At the moment, only `devign` has a completed end-to-end evaluation artifact.

Current processed statistics for `devign`:

- Total samples: `27,318`
- Vulnerable: `12,460`
- Non-vulnerable: `14,858`
- Positive ratio: `45.61%`

Current split sizes:

- Train: `21,854`
- Validation: `2,733`
- Test: `2,731`

The split builder uses `code_hash` grouping to reduce trivial leakage across train, validation, and test.

## Relevant Repository Structure

- `GRACE-improve/`: active research and implementation area
- `GRACE-improve/baseline/`: scripts for data prep, split creation, retrieval, graph extraction, inference, and evaluation
- `GRACE-improve/baseline/artifacts/`: processed data, splits, models, retrieval bank, graph cache, predictions, and metrics
- `GRACE-main/`: reference code kept for comparison with the original direction

## Current Baseline Pipeline

### 1. Dataset preparation and split creation

- `01_prepare_datasets.py` indexes dataset metadata into `artifacts/processed/<dataset>/index.csv`
- `02_create_splits.py` builds `train/val/test` using `StratifiedGroupKFold`
- `reveal` can preserve official splits when those are available from the raw source

This stage is mainly about keeping the benchmark protocol cleaner and reducing obvious leakage.

### 2. Prefilter and calibration

The prefilter is the first-stage risk scorer used to avoid sending every sample through the expensive GRACE-style path.

Current `devign` artifact state:

- Prefilter saved as an `ensemble_prefilter`
- Probability calibration uses `Platt scaling`
- Routing thresholds:
  - `tau_low = 0.141512`
  - `tau_high = 0.60686`

After calibration, each sample is routed into one of three bands:

- `skip`: low risk, predicted directly as `Non-vulnerable`
- `uncertain`: sent to retrieval + graph + local LLM
- `high`: high risk, currently predicted directly as `Vulnerable` in the stored run

### 3. Demonstration retrieval

`05_build_demo_bank.py` builds a balanced demonstration bank from the training split.

Current `devign` retrieval artifact:

- Total demonstrations: `8,000`
- Negative demonstrations: `4,000`
- Positive demonstrations: `4,000`
- Semantic backend: `CodeT5`

The retrieval stack currently works in two stages:

- semantic shortlist with `Salesforce/codet5-base`
- reranking with lexical `Jaccard` similarity and syntactic similarity over `AST sequence`

Compared with a very shallow retrieval baseline, this is much closer to the mechanism-aware retrieval idea behind GRACE.

### 4. Graph context

`graphs.py` prefers `Joern` and falls back to a heuristic graph backend when the Joern health check fails.

The graph prompt interface can carry:

- `node_info`
- `edge_info`
- `ast_sequence`

Current artifact state:

- Graph backend requested: `auto`
- Graph backend resolved: `heuristic`
- Backend notice: `Joern health probe failed: Joern export returned no nodes.`

So the current run is graph-aware at the interface level, but not yet a full Joern-backed GRACE reproduction.

### 5. Local detection stage

`06_run_grace_prefilter.py` executes the full cascade and calls the local model `Virtue-AI-HUB/VulnLLM-R-7B` when required.

The completed run in this repository used:

- `experiment_mode = local_uncertain_only`
- only `uncertain` samples were sent to the LLM
- `skip` and `high` samples were decided directly by the prefilter
- retrieval backend: `codet5`
- graph backend actually used during LLM calls: `heuristic`

### 6. Evaluation artifacts

Primary files for the current `devign` run:

- Predictions: `GRACE-improve/baseline/artifacts/predictions/devign/grace_prefilter_predictions.jsonl`
- Evaluation summary: `GRACE-improve/baseline/artifacts/metrics/devign/evaluation_summary.json`
- Retrieval summary: `GRACE-improve/baseline/artifacts/retrieval/devign/summary.json`

This README uses those artifacts as the source of truth.

## Current Devign Results

The current stored `devign` run reports:

| Metric | Value |
| --- | --- |
| Accuracy | `57.38%` |
| Precision | `51.81%` |
| Recall | `94.38%` |
| F1 | `0.6689` |
| ROC-AUC | `0.7067` |
| PR-AUC | `0.6587` |
| TP | `1176` |
| TN | `391` |
| FP | `1094` |
| FN | `70` |
| Bootstrap F1 mean | `0.6691` |
| Bootstrap F1 interval | `[0.6518, 0.6865]` |

Routing breakdown over `2,731` test samples:

- `skip`: `171`
- `uncertain`: `552`
- `high`: `2,008`
- LLM calls: `552 / 2,731 = 20.21%`
- Direct prefilter decisions: `2,179 / 2,731 = 79.79%`

Operational cost indicators:

- Average end-to-end runtime: `3383.053 ms/sample`
- Average retrieval latency on LLM-routed samples: `221.091 ms`
- Average graph latency on LLM-routed samples: `14.807 ms`
- Average local LLM latency on LLM-routed samples: `16501.231 ms`

## Comparison with Published GRACE Numbers

The local research note in `GRACE-improve/deep-research-report.md` summarizes published GRACE F1 values as:

- `Devign / FFmpeg+Qemu`: `0.651`
- `ReVeal`: `0.431`
- `Big-Vul`: `0.355`

Against that published `Devign` GRACE F1 reference, the current local baseline reaches:

- Current local `Devign` F1: `0.6689`
- Published GRACE `Devign` F1: `0.6510`
- Absolute delta: `+0.0179`
- Relative delta: `+2.76%`

A compact comparison:

| Dataset | Published GRACE F1 | Current Local Status | Current Local F1 | Delta vs Published |
| --- | --- | --- | --- | --- |
| Devign | `0.651` | evaluated | `0.6689` | `+0.0179` |
| ReVeal | `0.431` | not yet evaluated end-to-end | `N/A` | `N/A` |
| Big-Vul | `0.355` | not yet evaluated end-to-end | `N/A` | `N/A` |

### Comparison caveats

This is a useful directional comparison, but not a strict apples-to-apples reproduction.

- The current run uses `Virtue-AI-HUB/VulnLLM-R-7B`, while the GRACE overview is commonly described around a stronger hosted LLM setup.
- The current run fell back to `heuristic` graph extraction rather than full Joern-backed AST/CFG/PDG export.
- The local processed `devign` artifact has `27,318` samples, while the literature summary in `deep-research-report.md` quotes `22,361` samples for the `FFmpeg+Qemu (Devign)` benchmark variant.
- The current routing policy only sends the `uncertain` band to the LLM and auto-accepts the `high` band as vulnerable, which is an engineering choice not guaranteed to match the published GRACE inference protocol.

So the current result should be read as: the local cascade is competitive with, and slightly above, the published Devign F1 reference, but the protocol is not identical enough to claim a direct reproduction win.

## What Improved in This Baseline

### 1. LLM usage is heavily reduced

Only `20.21%` of the test samples go through the local LLM. That is the biggest practical gain in the current setup: the pipeline no longer pays full prompt cost for every function.

### 2. Retrieval is much closer to GRACE than a shallow baseline

The current retrieval stage includes:

- semantic retrieval with `CodeT5`
- lexical reranking with `Jaccard`
- syntactic reranking with `AST sequence similarity`
- a balanced demonstration bank across positive and negative examples

That makes the demonstrations more structurally relevant and gives the prompt better context than a token-overlap-only baseline.

### 3. The prompt interface is graph-aware

The pipeline already feeds `node_info`, `edge_info`, and `ast_sequence` into the prompting layer. Even though Joern is not yet stable in the current run, the baseline is architected around structure-aware prompting rather than plain code text.

### 4. Benchmark hygiene is stronger

The current implementation standardizes:

- processed dataset indexing
- split artifacts
- model artifacts
- prediction schema
- run-state tracking
- retrieval and graph caches

That makes the experiments easier to rerun, compare, and audit.

### 5. Resume and failure recovery are built in

Prediction files, run-state files, cache schemas, and compatibility checks make the pipeline much more practical to debug than a one-shot script chain.

## Current Limitations

### 1. Precision is still too low

The current run achieves very high recall (`94.38%`) but only `51.81%` precision, with `1094` false positives. That is acceptable for a recall-oriented screening system, but not yet strong enough as a final decision layer.

### 2. The `high` band is too aggressive

In the stored run, the `high` band is auto-accepted as `Vulnerable` without LLM review. That band has only about `54.48%` accuracy, which makes it the main false-positive source. If precision is the next target, this is the first place to fix.

### 3. Joern is not actually active in the final run

Although the repository already includes Joern tooling and graph scripts, the effective backend still fell back to `heuristic`. That means the current result is not yet a full GRACE-style AST/CFG/PDG graph run.

### 4. End-to-end evaluation exists only for Devign

`bigvul` and `reveal` already have preparation and split scripts, but the repository does not yet contain completed end-to-end metrics for those datasets.

### 5. LLM cost is still high on hard samples

Once a sample is routed into the `uncertain` band, the local LLM still costs about `16.5s` on average. The pipeline reduces the number of calls, but each call remains expensive.

## Quick Start

Run from the repository root:

```bash
python GRACE-improve/baseline/01_prepare_datasets.py
python GRACE-improve/baseline/02_create_splits.py
```

Train the prefilter from the notebook:

- open `GRACE-improve/baseline/03_train_prefilter.ipynb`
- set `DATASET_NAME`
- run all cells

Then calibrate and execute the cascade:

```bash
python GRACE-improve/baseline/04_calibrate_prefilter.py
python GRACE-improve/baseline/download_codet5_retrieval_model.py
python GRACE-improve/baseline/download_joern.py
python GRACE-improve/baseline/05_build_demo_bank.py
python GRACE-improve/baseline/download_vulnllm_r_7b.py
python GRACE-improve/baseline/06_run_grace_prefilter.py
```

A configuration close to the stored `devign` run:

```powershell
$env:GRACE_DATASET="devign"
$env:GRACE_RETRIEVAL_BACKEND="codet5"
$env:GRACE_GRAPH_BACKEND="auto"
$env:GRACE_CALL_LLM_FOR_UNCERTAIN="true"
$env:GRACE_CALL_LLM_FOR_HIGH_RISK="false"
python GRACE-improve/baseline/05_build_demo_bank.py
python GRACE-improve/baseline/06_run_grace_prefilter.py
```

To recompute or inspect evaluation metrics, use `GRACE-improve/baseline/07_evaluate_predictions.ipynb` or import the helpers in `GRACE-improve/baseline/evaluate_predictions.py`.

## Short Takeaway

The current `GRACE-improve` baseline already achieves three meaningful things:

- it restores the pipeline to a `prefilter + retrieval + graph + local LLM` architecture
- it cuts LLM usage sharply
- it reaches a Devign F1 that is slightly above the published GRACE reference reported in the local research note

The main next steps are clear:

- stop auto-accepting the `high` band so early
- restore a real Joern-backed graph pipeline
- complete end-to-end evaluation on `reveal` and `bigvul`
