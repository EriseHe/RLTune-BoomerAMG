#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PROFILE="${1:-signal}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

case "$PROFILE" in
  smoke)
    : "${MATRIX_GRID_N:=16}"
    : "${SETUP_PARAM_RESOLUTION:=3}"
    : "${TOTAL_TIMESTEPS:=64}"
    : "${N_STEPS:=32}"
    : "${BATCH_SIZE:=32}"
    : "${EVAL_CASES:=6}"
    : "${N_EPOCHS:=4}"
    : "${LEARNING_RATE:=3e-4}"
    : "${TUNE7_CANDIDATE_POOL_SIZE:=256}"
    : "${TUNE7_CANDIDATE_POOL_SIZE_BURNIN:=512}"
    : "${PROGRESS_EVERY_EPISODES:=2}"
    ;;
  signal)
    : "${MATRIX_GRID_N:=30}"
    : "${SETUP_PARAM_RESOLUTION:=8}"
    : "${TOTAL_TIMESTEPS:=8192}"
    : "${N_STEPS:=128}"
    : "${BATCH_SIZE:=128}"
    : "${EVAL_CASES:=48}"
    : "${N_EPOCHS:=8}"
    : "${LEARNING_RATE:=3e-4}"
    : "${TUNE7_CANDIDATE_POOL_SIZE:=1024}"
    : "${TUNE7_CANDIDATE_POOL_SIZE_BURNIN:=4096}"
    : "${PROGRESS_EVERY_EPISODES:=100}"
    ;;
  *)
    printf 'Unknown profile: %s (expected smoke or signal)\n' "$PROFILE" >&2
    exit 2
    ;;
esac

: "${OUTPUT_DIR:=$REPO_ROOT/results/joint/online_joint_v1/run_logs/${PROFILE}_$(date +%Y%m%d_%H%M%S)}"
: "${ONLINE_SEED:=20260715}"
: "${SETUP_TUNE_DIM:=7}"
: "${TUNE7_VARIANT:=categorical}"
: "${SETUP_ACTION_SPACE:=safe_one_at_a_time}"
: "${SETUP_INITIAL_GUESS_ROUNDS:=4}"
: "${ACTION_MODE:=discrete_w}"
: "${MODEL_TYPE:=mlp}"
: "${DISCRETE_W_VALUES:=1.2,1.4,1.5,1.6}"
: "${FIXED_W_VALUES:=1.4,1.6}"
: "${OBS_MODE:=cycle_action_setup}"
: "${SOLVE_MAX_CYCLES:=50}"
: "${TRUNC_PENALTY:=1.0}"
: "${W_CENTER:=1.5}"
: "${W_SCALE:=0.02}"
: "${TORCH_NUM_THREADS:=2}"
: "${MAX_SETUP_ATTEMPTS:=8}"
: "${CONVERGENCE_GUARD_START_FRACTION:=0.6}"
: "${CONVERGENCE_GUARD_MULTIPLIER:=2.0}"
: "${DEPLOYMENT_MAX_CYCLE_FRACTION:=0.7}"
: "${RL_VERBOSE:=0}"
: "${GAMMA:=1.0}"
: "${POLICY_STEP_COST_SEC:=0.0002}"
: "${POTENTIAL_PROGRESS_SCALE:=1.0}"
: "${SIGNAL_WINDOW:=25}"

export ACTION_MODE BATCH_SIZE DISCRETE_W_VALUES EVAL_CASES FIXED_W_VALUES
export MATRIX_GRID_N MAX_SETUP_ATTEMPTS
export CONVERGENCE_GUARD_MULTIPLIER CONVERGENCE_GUARD_START_FRACTION
export DEPLOYMENT_MAX_CYCLE_FRACTION
export N_EPOCHS N_STEPS OBS_MODE ONLINE_SEED OUTPUT_DIR PROGRESS_EVERY_EPISODES
export MODEL_TYPE
export GAMMA LEARNING_RATE POLICY_STEP_COST_SEC POTENTIAL_PROGRESS_SCALE
export RL_VERBOSE SETUP_PARAM_RESOLUTION SETUP_TUNE_DIM SOLVE_MAX_CYCLES TORCH_NUM_THREADS
export SETUP_ACTION_SPACE SETUP_INITIAL_GUESS_ROUNDS
export SIGNAL_WINDOW
export TOTAL_TIMESTEPS TRUNC_PENALTY TUNE7_CANDIDATE_POOL_SIZE
export TUNE7_CANDIDATE_POOL_SIZE_BURNIN TUNE7_VARIANT W_CENTER W_SCALE
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export PYTHONUNBUFFERED=1

mkdir -p "$OUTPUT_DIR"
"$PYTHON_BIN" "$REPO_ROOT/experiments/joint/solve_control/run_online_bandit_rl.py" 2>&1 \
  | tee "$OUTPUT_DIR/console.log"
