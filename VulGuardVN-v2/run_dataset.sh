#!/usr/bin/env bash
# VulGuard v2 - run the pipeline for ONE dataset, with the seeds (repetitions) you choose.
#
#   ./run_dataset.sh <dataset> <phase> [--reps 1,2,...] [--seed N] [--mock] [--smoke]
#
#   dataset   primevul | bigvul | devign | reveal          (or use ./run_<dataset>.sh <phase> ...)
#   phase     check      what is done / missing, next command                              (free)
#             prep       P1 data + P2 Joern + P3 features (UniXcoder) + fine-tuned UniXcoder (K-fold, long,
#                        resumable per epoch; also writes baseline B7)                       (free)
#             train      P3 prefilter for --reps, then scores + calibrators                (free)
#             ablations  P3 retrains A1 (leave-one-view-out) and A3 (no pair loss)         (free)
#             estimate   P4 preflight + cost estimate only, no API call                    (free)
#             llm        P4: A-pool -> test (+ zero-shot B3); PrimeVul also A7/A8, RQ5, uncertainty   (API)
#             offline    P5 verify + P6 decision layer for this dataset, P7 over all finished datasets (free)
#             b7         only the fine-tuning step of prep (view `finetune` + baseline B7)         (free)
#             all        prep -> train -> llm -> offline
#             dev        prompt loop on the PrimeVul dev set (~0.3 USD; builds PrimeVul records + dev-set Joern only)
#             freeze     freeze the prompt hashes (D6) - once, after `dev`, before the first `llm`;
#                        `freeze --if-ok` freezes only if dev_report says ok_to_freeze (else stops a chain)
#
#   --reps    prefilter repetitions to train (default 1,2,3,4,5). Rep r = ensemble seeds {10r,...,10r+4};
#             rep r also seeds the router (LightGBM, 80% A subsample) and the random-routing baseline.
#             Reps are independent and resumable: e.g. `train --reps 1` today, `train --reps 2,3` tomorrow,
#             then `offline` again - P6/P7 always use every rep that is trained and scored.
#   --mock    offline mock LLM (separate cache), for testing.   --smoke  use the small smoke artifacts.
#
# Before the first `llm` phase of any dataset: `dev` then `freeze` (any wrapper, e.g. ./run_devign.sh dev)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

usage() { sed -n '2,29p' "$0"; exit "${1:-0}"; }
[[ $# -lt 2 ]] && usage 1
DATASET=$1; PHASE=$2; shift 2
case "$DATASET" in primevul|bigvul|devign|reveal) ;; *) echo "unknown dataset: $DATASET"; usage 1 ;; esac

REPS=${REPS:-1,2,3,4,5}
IFOK_FLAG=()
SMOKE_FLAG=()
MOCK_FLAG=()
UNFROZEN_FLAG=()
QUICK_FLAG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps) REPS=$2; shift 2 ;;
    --if-ok) IFOK_FLAG=(--if-ok); shift ;;
    --mock) MOCK_FLAG=(--mock); export VG2_LLM_MOCK=1; shift ;;
    --smoke) export VG2_ARTIFACTS=v2/artifacts_smoke; MOCK_FLAG=(--mock); export VG2_LLM_MOCK=1
             UNFROZEN_FLAG=(--allow-unfrozen); QUICK_FLAG=(--quick); SMOKE_FLAG=(--smoke); shift ;;
    -h|--help) usage 0 ;;
    *) echo "unknown option: $1"; usage 1 ;;
  esac
done
[[ "$REPS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "--reps must look like 1 or 1,2,3 (got '$REPS')"; exit 1; }

LOG_DIR=${VG2_ARTIFACTS:-v2/artifacts}/logs
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/${DATASET}_${PHASE}.log"

run() {
  echo "=== [$(date '+%F %T')] $DATASET/$PHASE: $*" | tee -a "$LOG"
  "$PY" -m "$@" 2>&1 | tee -a "$LOG"
}

phase_prep() {
  run v2.scripts.run_p0_check
  run v2.scripts.run_p1_data --datasets "$DATASET" "${SMOKE_FLAG[@]}"
  run v2.scripts.run_p2_cpg --datasets "$DATASET"
  run v2.scripts.run_p3_prefilter --datasets "$DATASET" --stage features
  run v2.scripts.run_p3_prefilter --datasets "$DATASET" --stage finetune "${QUICK_FLAG[@]}"
}

phase_train() {
  run v2.scripts.run_p3_prefilter --datasets "$DATASET" --stage train --reps "$REPS" "${QUICK_FLAG[@]}"
  run v2.scripts.run_p3_prefilter --datasets "$DATASET" --stage score "${QUICK_FLAG[@]}"
}

phase_llm() {
  run v2.scripts.run_p4_llm run --datasets "$DATASET" --subset a_pool "${MOCK_FLAG[@]}" "${UNFROZEN_FLAG[@]}"
  run v2.scripts.run_p4_llm run --datasets "$DATASET" --subset test "${MOCK_FLAG[@]}" "${UNFROZEN_FLAG[@]}"
  if [[ "$DATASET" == "primevul" ]]; then          # API subset experiments of §7.4 / RQ5 / P4 step 7
    run v2.scripts.run_p4_llm ablations --datasets primevul "${MOCK_FLAG[@]}" "${UNFROZEN_FLAG[@]}"
    run v2.scripts.run_p4_llm rq5 --datasets primevul "${MOCK_FLAG[@]}" "${UNFROZEN_FLAG[@]}"
    run v2.scripts.run_p4_llm uncertainty --datasets primevul "${MOCK_FLAG[@]}" "${UNFROZEN_FLAG[@]}"
  fi
  run v2.scripts.run_p4_llm status
}

phase_offline() {
  run v2.scripts.run_p5_verify --datasets "$DATASET" "${MOCK_FLAG[@]}"
  run v2.scripts.run_p6_decision --datasets "$DATASET" "${MOCK_FLAG[@]}"
  run v2.scripts.run_p7_experiments --datasets all "${MOCK_FLAG[@]}" ${QUICK_FLAG:+--bootstrap 200}
  if [[ "$DATASET" == "primevul" ]]; then
    run v2.scripts.run_p7_robustness --datasets primevul
  fi
}

case "$PHASE" in
  check)     "$PY" -m v2.scripts.run_status --datasets "$DATASET" "${MOCK_FLAG[@]}" ;;
  prep)      phase_prep ;;
  train)     phase_train ;;
  ablations) run v2.scripts.run_p3_prefilter --datasets "$DATASET" --stage ablations "${QUICK_FLAG[@]}" ;;
  estimate)
    for subset in a_pool test; do
      "$PY" -m v2.scripts.run_p4_llm run --datasets "$DATASET" --subset "$subset" --dry-run "${MOCK_FLAG[@]}" "${UNFROZEN_FLAG[@]}" || true
    done ;;
  llm)       phase_llm ;;
  offline)   phase_offline ;;
  b7)        run v2.scripts.run_p3_prefilter --datasets "$DATASET" --stage finetune "${QUICK_FLAG[@]}" ;;
  all)       phase_prep; phase_train; phase_llm; phase_offline ;;
  dev)       run v2.scripts.run_p1_data --datasets primevul "${SMOKE_FLAG[@]}"
             run v2.scripts.run_p2_cpg --datasets primevul --subset dev
             run v2.scripts.run_p4_llm dev "${MOCK_FLAG[@]}" ;;
  freeze)    run v2.scripts.run_p4_llm freeze "${IFOK_FLAG[@]}" "${MOCK_FLAG[@]}" ;;
  *) echo "unknown phase: $PHASE"; usage 1 ;;
esac
echo "=== [$(date '+%F %T')] $DATASET/$PHASE finished (log: $LOG)"
