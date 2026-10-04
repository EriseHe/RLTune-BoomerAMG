#!/bin/zsh
set -euo pipefail
TASK_REPO_ROOT="$(cd -- "$(dirname -- "$0")/../../.." && pwd)"
cd "$TASK_REPO_ROOT"
exec caffeinate -i /Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python -u -m experiments.paper_final.run_06_frozen_methods run
