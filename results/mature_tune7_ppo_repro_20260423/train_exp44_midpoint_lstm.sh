#!/usr/bin/env bash
set -euo pipefail

# Train the "Exp44" model:
# - no teacher / no rule warmstart
# - full-range family w in [1, 2]
# - midpoint start at 1.5
# - residual control with a small step size
# - LSTM policy
#
# This script only trains and writes a new model directory.
# It also handles the saved mature-bandit state:
# - if BANDIT_STATE exists, reuse it
# - otherwise rebuild warmup state and save it there

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp44_common.sh"

cd "$REPO_ROOT"

# New run location.
RUN_TAG="${RUN_TAG:-exp44_midpoint_lstm_seedmeanstd_vsbandit_20260515T1}"
RUN_DIR="$REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/$RUN_TAG"
mkdir -p "$RUN_DIR"

echo "Training Exp44-style model into:"
echo "  $RUN_DIR"

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
MODEL_BASENAME="$RUN_DIR/model" \
RESULT_PATH="$RUN_DIR/result.json" \
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
ALGO="$ALGO" \
MODEL_TYPE="$MODEL_TYPE" \
LEARNING_RATE="$LEARNING_RATE" \
ENT_COEF="$ENT_COEF" \
N_STEPS="$N_STEPS" \
BATCH_SIZE="$BATCH_SIZE" \
BC_WARMSTART=0 \
TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
CHECKPOINT_INTERVAL="$CHECKPOINT_INTERVAL" \
SELECT_MODEL=best \
CHECKPOINT_OBJECTIVE="$CHECKPOINT_OBJECTIVE" \
"$PYTHON_BIN" SolvePhase/hypre/src/test/run_mature_bandit_rl_pipeline.py

echo
echo "Done."
echo "Run dir:      $RUN_DIR"
echo "Model file:   $RUN_DIR/model_ckpt_${CHECKPOINT_INTERVAL}.zip"
echo "Result file:  $RUN_DIR/result.json"
