#!/usr/bin/env bash
set -euo pipefail

# Train the current absolute-action PPO model:
# - no teacher / no rule warmstart
# - full-range family w in [1, 2]
# - actor initialized at the solver default
# - policy predicts an absolute physical weight each cycle
# - LSTM policy
#
# This script only trains and writes a new model directory.
# It also handles the saved mature-bandit state:
# - if BANDIT_STATE exists, reuse it
# - otherwise rebuild warmup state and save it there

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp44_common.sh"

cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-exp44_absolute_lstm_train2000_instances_seed39396939}"
EXP44_RUN_ROOT="${EXP44_RUN_ROOT:-$EXP44_RUNS_ROOT/$RUN_TAG}"
TRAIN_DIR="${EXP44_TRAIN_DIR:-$EXP44_RUN_ROOT/training}"
mkdir -p "$TRAIN_DIR" "$(dirname "$BANDIT_STATE")"

echo "Training Exp44-style model into:"
echo "  $TRAIN_DIR"

# Build the bandit-state env prefix once so the actual training command stays readable.
if [[ -f "$BANDIT_STATE" ]]; then
  echo "Using saved bandit state:"
  echo "  $BANDIT_STATE"
  BANDIT_ENV_PREFIX=(
    USE_SAVED_BANDIT=1
    BANDIT_STATE_PATH="$BANDIT_STATE"
  )
else
  echo "Saved bandit state missing; rebuilding and saving to:"
  echo "  $BANDIT_STATE"
  BANDIT_ENV_PREFIX=(
    SAVE_BANDIT_STATE_PATH="$BANDIT_STATE"
  )
fi

env "${BANDIT_ENV_PREFIX[@]}" \
MATRIX_GRID_N="$MATRIX_GRID_N" \
SETUP_PARAM_RESOLUTION="$SETUP_PARAM_RESOLUTION" \
EXPECTED_SETUP_ACTION_COUNT="$EXPECTED_SETUP_ACTION_COUNT" \
MODEL_BASENAME="$TRAIN_DIR/model" \
RESULT_PATH="$TRAIN_DIR/result.json" \
RL_TRAIN_SEEDS="$RL_TRAIN_SEEDS" \
RL_TRAIN_CASES="$RL_TRAIN_CASES" \
TRAIN_TRACE_SHUFFLE=1 \
EVAL_SEEDS="$EVAL_SEEDS" \
EVAL_CASES="$EVAL_CASES" \
ACTION_MODE="$ACTION_MODE" \
OBS_MODE="$OBS_MODE" \
W_ONLY=1 \
W_CENTER="$W_CENTER" \
W_SCALE="$W_SCALE" \
W_GLOBAL_MIN="$W_GLOBAL_MIN" \
W_GLOBAL_MAX="$W_GLOBAL_MAX" \
INITIAL_OBSERVATION_WEIGHT="$INITIAL_OBSERVATION_WEIGHT" \
INITIAL_POLICY_WEIGHT="$INITIAL_POLICY_WEIGHT" \
ALGO="$ALGO" \
MODEL_TYPE="$MODEL_TYPE" \
LEARNING_RATE="$LEARNING_RATE" \
ENT_COEF="$ENT_COEF" \
N_STEPS="$N_STEPS" \
BATCH_SIZE="$BATCH_SIZE" \
BC_WARMSTART=0 \
TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
TRAIN_EPISODES="$TRAIN_EPISODES" \
CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL" \
SELECT_MODEL=best \
CHECKPOINT_OBJECTIVE="$CHECKPOINT_OBJECTIVE" \
"$PYTHON_BIN" "$REPO_ROOT/experiments/joint/solve_control/run_mature_bandit_rl_pipeline.py" \
  2>&1 | tee "$TRAIN_DIR/run.log"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp44}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/report_exp44_results.py" training \
  --result "$TRAIN_DIR/result.json" \
  --output-dir "$TRAIN_DIR"

echo
echo "Done."
echo "Run root:     $EXP44_RUN_ROOT"
echo "Model file:   $TRAIN_DIR/model_best.zip"
echo "Result file:  $TRAIN_DIR/result.json"
echo "Table:        $TRAIN_DIR/validation_table.csv"
echo "Figures:      $TRAIN_DIR/figures"
