#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}_reproduction}"

cd "$REPO_ROOT"
"$PYTHON_BIN" -u experiments/joint/solve_control/run_joint_experiment.py \
  --config "$SCRIPT_DIR/experiment_config.json" \
  --output-dir "$OUTPUT_DIR"
