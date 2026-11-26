#!/usr/bin/env bash
# VulGuard v2 - reveal only. Same phases/options as run_dataset.sh, e.g.:
#   ./run_reveal.sh check
#   ./run_reveal.sh prep
#   ./run_reveal.sh train --reps 1,2
#   ./run_reveal.sh llm
#   ./run_reveal.sh offline
exec "$(dirname "$0")/run_dataset.sh" reveal "$@"
