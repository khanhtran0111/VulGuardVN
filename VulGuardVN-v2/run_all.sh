#!/usr/bin/env bash
# VulGuard v2 - full local pipeline (plan §5). Every step is idempotent and resumable: re-running skips work
# that is already in artifacts/ (feature chunks, trained seeds, Joern batches, LLM cache keys).
#
#   ./run_all.sh local     P0 check, P1 data, P2 Joern, P3 features + 25 models + ablation retrains   (no API)
#   ./run_all.sh dev       P4 step 4: prompt loop on the 300-function dev set (~1-2 USD), prints dev_report
#   ./run_all.sh freeze    P4 step 5: freeze prompt hashes in configs/llm.yaml (D6) - only after the dev loop
#   ./run_all.sh llm       P4: A-pool -> test (+ zero-shot), A7/A8, RQ5 (gpt-oss-120b), uncertainty check
#   ./run_all.sh offline   P5 verify, P6 decision layer, P7 experiments + robustness                (no API)
#   ./run_all.sh b7        optional baseline B7: UniXcoder fine-tune on the local GPU
#   ./run_all.sh smoke     everything above on a small subset with a mock LLM (separate artifacts_smoke/)
#   ./run_all.sh status    LLM tokens / cost so far
#
# DATASETS=devign ./run_all.sh local      restricts the datasets (default: all four). Datasets can be run one by one;
#                                         P7 / summaries always merge every dataset that is already finished.
# Note: the prompt dev loop (dev/freeze) uses the PrimeVul dev set and must be done once before any `llm` run.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
DATASETS=${DATASETS:-all}
MOCK=${MOCK:-}
LOG_DIR=${VG2_ARTIFACTS:-v2/artifacts}/logs
mkdir -p "$LOG_DIR"

has_primevul() { [[ "$DATASETS" == "all" || ",$DATASETS," == *",primevul,"* ]]; }

run() {
  local name=$1; shift
  echo "=== [$(date +%H:%M:%S)] $name: $*"
  "$PY" -m "$@" 2>&1 | tee -a "$LOG_DIR/$name.log"
}

phase_local() {
  run p0 v2.scripts.run_p0_check
  run p1 v2.scripts.run_p1_data --datasets "$DATASETS" ${SMOKE:+--smoke}
  run p2 v2.scripts.run_p2_cpg --datasets "$DATASETS"
  run p3 v2.scripts.run_p3_prefilter --datasets "$DATASETS" --stage features
  run p3 v2.scripts.run_p3_prefilter --datasets "$DATASETS" --stage finetune ${QUICK:+--quick}
  run p3 v2.scripts.run_p3_prefilter --datasets "$DATASETS" --stage train ${QUICK:+--quick}
  run p3 v2.scripts.run_p3_prefilter --datasets "$DATASETS" --stage score ${QUICK:+--quick}
  if [[ -z "${SKIP_ABLATION_TRAINING:-}" ]]; then
    run p3 v2.scripts.run_p3_prefilter --datasets "$DATASETS" --stage ablations ${QUICK:+--quick}
  fi
}

phase_llm() {
  local extra=(${MOCK:+--mock} ${UNFROZEN:+--allow-unfrozen})
  run p4 v2.scripts.run_p4_llm run --datasets "$DATASETS" --subset a_pool "${extra[@]}"
  run p4 v2.scripts.run_p4_llm run --datasets "$DATASETS" --subset test "${extra[@]}"
  # A7/A8, RQ5 and the uncertainty check use the PrimeVul API subset (§7.4): only when PrimeVul is in DATASETS
  # (or API_DATASETS is given explicitly), so a single-dataset run never spends money on another dataset.
  local api_ds=${API_DATASETS:-}
  if [[ -z "$api_ds" ]] && has_primevul; then api_ds=primevul; fi
  if [[ -n "$api_ds" ]]; then
    run p4 v2.scripts.run_p4_llm ablations --datasets "$api_ds" "${extra[@]}"
    run p4 v2.scripts.run_p4_llm rq5 --datasets "$api_ds" "${extra[@]}"
    run p4 v2.scripts.run_p4_llm uncertainty --datasets "$api_ds" "${extra[@]}"
  fi
  run p4 v2.scripts.run_p4_llm status
}

phase_offline() {
  local extra=(${MOCK:+--mock})
  run p5 v2.scripts.run_p5_verify --datasets "$DATASETS" "${extra[@]}"
  run p6 v2.scripts.run_p6_decision --datasets "$DATASETS" "${extra[@]}"
  # P7 always covers every dataset that has finished P6, so results/ never loses a dataset run earlier
  run p7 v2.scripts.run_p7_experiments --datasets all "${extra[@]}" ${BOOTSTRAP:+--bootstrap $BOOTSTRAP}
  local robust_ds=${ROBUST_DATASETS:-}
  if [[ -z "$robust_ds" ]] && has_primevul; then robust_ds=primevul; fi
  if [[ -n "$robust_ds" ]]; then
    run p7 v2.scripts.run_p7_robustness --datasets "$robust_ds"
  fi
}

case "${1:-help}" in
  local)   phase_local ;;
  dev)     # prompt loop on the PrimeVul dev set; builds only PrimeVul records + Joern for the 300 dev functions
           run p1 v2.scripts.run_p1_data --datasets primevul
           run p2 v2.scripts.run_p2_cpg --datasets primevul --subset dev
           run p4 v2.scripts.run_p4_llm dev ${MOCK:+--mock} ;;
  freeze)  run p4 v2.scripts.run_p4_llm freeze ;;
  llm)     phase_llm ;;
  offline) phase_offline ;;
  b7)      run b7 v2.scripts.run_p3_prefilter --datasets "$DATASETS" --stage finetune ;;
  status)  run p4 v2.scripts.run_p4_llm status ;;
  smoke)
    export VG2_ARTIFACTS=v2/artifacts_smoke MOCK=1 UNFROZEN=1 SMOKE=1 QUICK=1 BOOTSTRAP=200 SKIP_ABLATION_TRAINING=${SKIP_ABLATION_TRAINING:-}
    export DATASETS=${DATASETS/all/primevul,devign} ROBUST_DATASETS=primevul API_DATASETS=primevul
    LOG_DIR=v2/artifacts_smoke/logs; mkdir -p "$LOG_DIR"
    phase_local
    run p4 v2.scripts.run_p4_llm dev --mock
    phase_llm
    phase_offline ;;
  *) sed -n '2,19p' "$0" ;;
esac
