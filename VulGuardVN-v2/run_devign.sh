#!/usr/bin/env bash
# VulGuard v2 - devign only. Same phases/options as run_dataset.sh, e.g.:
#   ./run_devign.sh check
#   ./run_devign.sh prep
#   ./run_devign.sh train --reps 1,2
#   ./run_devign.sh llm
#   ./run_devign.sh offline
exec "$(dirname "$0")/run_dataset.sh" devign "$@"
