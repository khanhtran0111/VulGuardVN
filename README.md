# VulGuardVN

VulGuardVN is a research project on function-level vulnerability detection for C/C++ source code. It rebuilds a GRACE-style graph-aware and retrieval-augmented LLM workflow, then extends that workflow with a learned multi-view prefilter. The goal is to reduce unnecessary detailed LLM inspection while preserving a security-oriented emphasis on vulnerability recall.

The project is based on the GRACE paper, [GRACE: Empowering LLM-based software vulnerability detection with graph structure and in-context learning](https://doi.org/10.1016/j.jss.2024.112031), published in the Journal of Systems and Software, Volume 212, June 2024, Article 112031. The original paper is also available on [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0164121224000748), and the official implementation is hosted in the [GRACE GitHub repository](https://github.com/P-E-Vul/GRACE).

## Research Objective

VulGuardVN studies whether a multi-view prefilter can improve the efficiency of GRACE-style binary vulnerability detection on Devign/FFmpeg+Qemu, Big-Vul, and ReVeal. Instead of sending every function directly to LLM inspection, the pipeline estimates vulnerability risk from lexical, syntactic, semantic, and graph-derived views, then uses calibrated thresholds to decide which samples need detailed analysis.

The central question is whether this routing strategy can reduce LLM usage while retaining enough suspicious cases for graph-aware and retrieval-augmented inspection.

## Public Artifacts

The main experiment notebook is [full_pipeline.ipynb](full_pipeline.ipynb) at the repository root. Earlier draft text referred to `GRACE-improve/baseline/baseline2/FINAL.ipynb`; that notebook has been moved and renamed in the public repository.

| Artifact | Path | Role |
| --- | --- | --- |
| Main experiment notebook | [full_pipeline.ipynb](full_pipeline.ipynb) | Canonical public notebook for the Kaggle-oriented end-to-end experiment. |
| Stage scripts | [GRACE-improve/baseline/baseline2/](GRACE-improve/baseline/baseline2/) | Script implementation of the same pipeline stages, from `00_verify_assets.py` through `08_evaluate_predictions.py`. |
| Pipeline figure | [figures/pipeline_overview.png](figures/pipeline_overview.png) | Static overview of the VulGuardVN pipeline. |
| GRACE reference material | [GRACE-main/](GRACE-main/) | Upstream GRACE assets retained for provenance and comparison. |

## Pipeline Overview

![Pipeline overview](figures/pipeline_overview.png)

The pipeline starts from normalized C/C++ function records. It builds numeric and graph-aware features, semantic embeddings, AST-like sequences, and token sequences. A hybrid prefilter estimates vulnerability probability, a calibration layer derives low/high routing thresholds, and the GRACE-style inspection module resolves uncertain cases using retrieved demonstrations, graph context, suspicious-slice localization, and a local LLM.

## Datasets

The project targets the three datasets associated with the GRACE benchmark setting.

| Dataset | Role in the experiment | Public source used by the notebook |
| --- | --- | --- |
| Devign/FFmpeg+Qemu | Function-level C/C++ vulnerability benchmark | [GRACE Devign dataset](https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/view?usp=sharing), with a [CodeXGLUE mirror](https://raw.githubusercontent.com/madlag/CodeXGLUE/main/Code-Code/Defect-detection/dataset/function.json) fallback |
| Big-Vul | Large-scale C/C++ vulnerability dataset collected from CVE-linked code changes | [GRACE Big-Vul dataset](https://drive.google.com/file/d/1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X/view?usp=sharing), with [Big-Vul on Hugging Face](https://huggingface.co/datasets/bstee615/bigvul) as a fallback |
| ReVeal | Real-world deep-learning vulnerability detection benchmark | [ReVeal on Hugging Face](https://huggingface.co/datasets/claudios/ReVeal) |

For direct script execution, raw data is expected under `GRACE-improve/data/` using the filenames handled by [datasets.py](GRACE-improve/baseline/baseline2/datasets.py).

| Dataset | Expected local location |
| --- | --- |
| Devign | `GRACE-improve/data/function.json` |
| Big-Vul | `GRACE-improve/data/MSR_data_cleaned.csv` or `GRACE-improve/data/bigvul_raw/{train,validation,test}-00000-of-00001.parquet` |
| ReVeal | `GRACE-improve/data/reveal/{train,val,test}.jsonl` or another supported ReVeal candidate directory |

ReVeal official splits are preserved when present. Devign and Big-Vul use reproducible stratified group splits.

## Method

VulGuardVN follows a compact GRACE-style hybrid design:

1. Normalize source-code records from the target datasets into a shared function-level schema.
2. Build reproducible train, validation, and test splits.
3. Extract lexical, syntactic, semantic, and graph-aware features from each function.
4. Train a dataset-specific hybrid prefilter to estimate vulnerability probability.
5. Calibrate validation probabilities and derive routing thresholds.
6. Apply retrieval-augmented and graph-aware LLM inspection only to samples that require detailed analysis.
7. Evaluate binary vulnerability detection and report the LLM call ratio as the main efficiency indicator.

## Environment and Models

The notebook is designed for running on GPU. The default semantic encoder is [microsoft/unixcoder-base-nine](https://huggingface.co/microsoft/unixcoder-base-nine), and the default local LLM is [unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit).

Graph extraction uses automatic backend selection. When Joern is available, the pipeline can use it; otherwise it falls back to the repository's heuristic graph extractor.

## Evaluation

Evaluation reports accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and LLM call ratio. The LLM call ratio measures the fraction of test samples that required detailed LLM inspection after calibrated routing.
