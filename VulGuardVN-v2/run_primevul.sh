#!/usr/bin/env bash
# VulGuard v2 - primevul only. Same phases/options as run_dataset.sh, e.g.:
#   ./run_primevul.sh check
#   ./run_primevul.sh prep
#   ./run_primevul.sh train --reps 1,2
#   ./run_primevul.sh llm
#   ./run_primevul.sh offline
exec "$(dirname "$0")/run_dataset.sh" primevul "$@"
