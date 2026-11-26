#!/usr/bin/env bash
# VulGuard v2 - bigvul only. Same phases/options as run_dataset.sh, e.g.:
#   ./run_bigvul.sh check
#   ./run_bigvul.sh prep
#   ./run_bigvul.sh train --reps 1,2
#   ./run_bigvul.sh llm
#   ./run_bigvul.sh offline
exec "$(dirname "$0")/run_dataset.sh" bigvul "$@"
