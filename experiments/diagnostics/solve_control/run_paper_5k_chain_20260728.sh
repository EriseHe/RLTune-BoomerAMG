#!/bin/zsh
set -euo pipefail

REPO_ROOT="/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG"
PYTHON_BIN="/opt/anaconda3/envs/rl/bin/python"
RUNNER="$REPO_ROOT/experiments/joint/solve_control/run_joint_experiment.py"
CHAIN_DIR="$REPO_ROOT/results/diagnostics/solve_control/paper_5k_chain_20260728"
STATUS_FILE="$CHAIN_DIR/status.json"

FIRST_CONFIG="$REPO_ROOT/experiments/joint/solve_control/configs/n60_setup_space_original_vs_recommended_5k_rerun.json"
FIRST_OUTPUT="$REPO_ROOT/results/diagnostics/solve_control/n60_setup_space_original_vs_recommended_5k_rerun_20260728"
FIRST_LOG="$CHAIN_DIR/01_setup_space_original_vs_recommended_5k.log"

SECOND_CONFIG="$REPO_ROOT/experiments/joint/solve_control/configs/n60_diffusion_advection_linucb_v4_vs_v5_lstdq_v3_5k.json"
SECOND_OUTPUT="$REPO_ROOT/results/diagnostics/solve_control/n60_diffusion_advection_linucb_v4_vs_v5_lstdq_v3_5k_20260728"
SECOND_LOG="$CHAIN_DIR/02_diffusion_advection_v4_vs_v5_lstdq_v3_5k.log"

first_state="pending"
second_state="pending"
active_stage="initializing"

mkdir -p "$CHAIN_DIR"
cd "$REPO_ROOT"

write_status() {
  local chain_state="$1"
  local timestamp
  timestamp="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf \
    '{"chain_state":"%s","active_stage":"%s","first_experiment":"%s","second_experiment":"%s","pid":%d,"updated_at":"%s"}\n' \
    "$chain_state" \
    "$active_stage" \
    "$first_state" \
    "$second_state" \
    "$$" \
    "$timestamp" \
    > "$STATUS_FILE.tmp"
  mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

handle_exit() {
  local exit_code="$?"
  if (( exit_code != 0 )); then
    if [[ "$first_state" == "running" ]]; then
      first_state="failed"
    elif [[ "$second_state" == "running" ]]; then
      second_state="failed"
    fi
    active_stage="$active_stage (exit $exit_code)"
    write_status "failed"
  fi
}
trap handle_exit EXIT

run_experiment() {
  local label="$1"
  local config_path="$2"
  local output_path="$3"
  local log_path="$4"

  if [[ -s "$output_path/result.json" ]]; then
    printf '[%s] Existing completed result found; skipping.\n' "$label"
    return 0
  fi
  if [[ -e "$output_path" ]]; then
    printf '[%s] Refusing to overwrite incomplete output: %s\n' \
      "$label" \
      "$output_path" \
      >&2
    return 73
  fi

  printf '[%s] Validating locked configuration.\n' "$label"
  "$PYTHON_BIN" -u "$RUNNER" \
    --config "$config_path" \
    --validate-only \
    2>&1 | tee -a "$log_path"

  printf '[%s] Starting full experiment.\n' "$label"
  "$PYTHON_BIN" -u "$RUNNER" \
    --config "$config_path" \
    --no-plots \
    2>&1 | tee -a "$log_path"

  if [[ ! -s "$output_path/result.json" ]]; then
    printf '[%s] Runner exited without a result.json file.\n' "$label" >&2
    return 74
  fi
  printf '[%s] Completed successfully: %s\n' "$label" "$output_path"
}

write_status "running"

first_state="running"
active_stage="setup_space_original_vs_recommended_5k"
write_status "running"
run_experiment \
  "setup-space 5K" \
  "$FIRST_CONFIG" \
  "$FIRST_OUTPUT" \
  "$FIRST_LOG"
first_state="complete"
active_stage="transition_to_diffusion_advection"
write_status "running"

second_state="running"
active_stage="diffusion_advection_v4_vs_v5_lstdq_v3_5k"
write_status "running"
run_experiment \
  "diffusion-advection V4/V5 + Recursive LSTDQ v3 5K" \
  "$SECOND_CONFIG" \
  "$SECOND_OUTPUT" \
  "$SECOND_LOG"
second_state="complete"
active_stage="complete"
write_status "complete"

trap - EXIT
printf 'Both 5K experiments completed successfully.\n'
