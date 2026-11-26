# VulGuardVN Devign revision experiments on Kaggle

These notebooks are thin orchestrators for the shared implementation in
`GRACE-improve/baseline/baseline2` and `GRACE-improve/revision_experiments`.
They do not contain a copied pipeline. Smoke outputs are development checks
and must never be reported as paper results.

## Notebook roles

| Notebook | Role | LLM in full mode | E01 artifact reuse | Runs per execution |
|---|---|---:|---:|---:|
| `01_devign_E01_multiseed_kaggle.ipynb` | Train/infer one seed and one baseline/proposed configuration | Yes | No | 1 |
| `02_devign_E02_leave_one_view_out_kaggle.ipynb` | Train/infer one view ablation | Depending on routing | Optional `full` for deltas | 1 |
| `03_devign_E03_branch_analysis_kaggle.ipynb` | Analysis-only branch errors | No | Proposed predictions | 1 analysis |
| `04_devign_E04_calibration_distribution_kaggle.ipynb` | Analysis-only calibration plots | No | Proposed calibration | 1 analysis |
| `05_devign_E05_routing_policy_kaggle.ipynb` | Run one routing policy | Yes, except smoke | Proposed may be reused as direct-high; baseline for matched recall | 1 |
| `06_devign_E06_runtime_cost_kaggle.ipynb` | Analysis-only runtime/cost comparison | No | Proposed and reproduced baseline | 1 analysis |

E01, E02, and E05 are model notebooks. E03, E04, and E06 only read existing
artifacts and do not retrain or invoke the LLM.

## Kaggle setup

1. Create a Kaggle notebook and enable **Internet** in Notebook options so the
   repository and any explicitly allowed model assets can be downloaded.
2. Select a GPU accelerator. A T4/P100 is adequate for preparation and small
   checks; full local-LLM inference may require a larger-memory GPU or 4-bit
   loading.
3. Import one notebook from this directory.
4. Edit only its centralized **User configuration** cell.
5. Run All. E01 defaults to `RUN_MODE = "full"`; use `"dry-run"` or `"smoke"`
   only for development checks.

The dependency cell checks imports before installing anything. It does not
upgrade existing TensorFlow, PyTorch, or CUDA packages.

## Automatic download

E01, E02, and E05 can start without any manually attached Kaggle Dataset. Keep:

```python
DEVIGN_SOURCE_PATH = None
RETRIEVAL_MODEL_SOURCE_DIR = None
LOCAL_LLM_SOURCE_DIR = None

AUTO_DOWNLOAD_DATASET_IF_MISSING = True
AUTO_DOWNLOAD_MISSING_MODELS = True
```

Enable **Kaggle Internet** and a **GPU** for full experiments. The asset cell:

1. downloads and validates Devign, then materializes it at
   `GRACE-improve/data/function.json`;
2. downloads `microsoft/unixcoder-base-nine`;
3. downloads `unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit` for full runs;
4. caches everything under `/kaggle/working/vulguard-assets` and sets the
   runtime environment variables consumed by baseline2.

Smoke mode skips Qwen because smoke inference disables LLM calls. Cached valid
assets are reused and are not downloaded again during the session.

## Mounted/offline input

To avoid downloads, attach your own Kaggle Dataset and configure existing paths:

```python
DEVIGN_SOURCE_PATH = "/kaggle/input/my-devign/function.json"
RETRIEVAL_MODEL_SOURCE_DIR = "/kaggle/input/my-models/unixcoder"
LOCAL_LLM_SOURCE_DIR = "/kaggle/input/my-models/qwen"

AUTO_DOWNLOAD_DATASET_IF_MISSING = False
AUTO_DOWNLOAD_MISSING_MODELS = False
```

Mounted paths take priority and remain read-only. Depending on
`COPY_MODELS_INSTEAD_OF_LINK`, model snapshots are copied into the working cache
or used via the configured path. Invalid HTML/JSON downloads, incomplete model
weights, missing tokenizer files, and missing shard indexes are rejected before
the experiment starts.

E03, E04, and E06 never download Devign, UniXcoder, or Qwen. Their
`Download and prepare required artifacts` cell searches an optional
`E01_RESULTS_SOURCE`, `/kaggle/input`, and `/kaggle/working`; it can download an
E01 result ZIP from `E01_RESULTS_DOWNLOAD_URL` when configured.

## Selecting one seed and configuration

Every execution runs exactly one `SEED` and one `CONFIGURATION`. E01 requires
separate sessions for seeds `1`, `7`, `21`, `42`, and `100`, first for
`proposed` and then for `reproduced_baseline`. E02 configurations are `full`,
`no_token`, `no_ast`, `no_semantic`, and `no_graph_numeric`. E05 configurations
are `direct_high`, `verify_high`, and `reproduced_baseline`.

Do not remove `SPLIT_SEED = 42`, `DEMO_SEED = 31415`, or
`BOOTSTRAP_SEED = 27182` from E01 when comparing training seeds.

## Moving E01 artifacts between sessions

At the end of a model run, download the ZIP from `/kaggle/working/exports/`.
To make it reusable:

1. Optionally create a Kaggle Dataset and upload one or more exported ZIP files;
   attached inputs are discovered automatically.
2. Or set `E01_RESULTS_SOURCE` to a local/mounted ZIP or directory.
3. Or set `E01_RESULTS_DOWNLOAD_URL` and leave
   `AUTO_DOWNLOAD_RESULTS_IF_MISSING = True`.

Analysis ZIPs are extracted under `/kaggle/working/revision_inputs`. The helper
validates exactly the files and fields required by E03, E04, or E06.

## Resume and the 12-hour limit

Model notebooks default to `RESUME = True`. The runner saves stage state after
every stage and inference flushes each prediction record. It tracks elapsed
time and stops before a new stage when the configured session buffer is
reached. `TEST_CHUNK_SIZE` limits each session to one inference chunk;
`TEST_CHUNK_INDEX = None` selects the next unresolved chunk automatically.

An incomplete run produces two distinct exports:

- The compact `_partial.zip` contains reportable artifacts but excludes
  `_pipeline`.
- The `_checkpoint.zip` contains the complete run directory, including
  `_pipeline`, models, feature stores, predictions, and stage state.

The checkpoint does not include `/kaggle/working/vulguard-assets`; Devign and
Hugging Face snapshots are independently resolved from cache, mounted overrides,
or automatic downloads in each new session.

Upload `_checkpoint.zip` as a Kaggle Dataset. In the next model session, set
`RESTORE_CHECKPOINT = True`, set `CHECKPOINT_INPUT` to that Dataset mount, and
keep the same commit, dataset, seed, experiment, and configuration. The helper
validates those fields and the packaged inference run signature before copying
anything into `OUTPUT_ROOT`; the runner then resumes the automatically selected
chunk. Evaluation is deferred until `_pipeline/run_state.json` reports
`complete=true`.

Never describe a `partial` run as a full result.

## Getting outputs

Portable ZIPs and `run_summary.json` are written to:

```text
/kaggle/working/exports/
```

The compact ZIP excludes raw datasets, downloaded model caches, feature stores,
and other large pipeline caches. Use the separate checkpoint ZIP—not the compact
ZIP—for multi-session resume.

## Recommended order and expected sessions

| Order | Work | Expected sessions |
|---:|---|---:|
| 1 | E01 proposed, five seeds | 5 |
| 1 | E01 reproduced baseline, five seeds | 5 |
| 2 | E02 five ablations | 5 |
| 3 | E03 branch analysis | 1 |
| 4 | E04 calibration analysis | 1 |
| 5 | E05 verify-high; direct-high can reuse E01 | 1–3 |
| 6 | E06 runtime analysis | 1 |

Expected total before inference chunking: approximately 19–21 Kaggle sessions.
Model runs may require additional sessions according to `TEST_CHUNK_SIZE`, GPU,
LLM cache state, graph backend, and whether E01 direct-high artifacts are reused.
