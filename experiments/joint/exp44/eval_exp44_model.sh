#!/usr/bin/env bash
set -euo pipefail

# Evaluate one Exp44-compatible model on held-out forward continuation.
#
# Required:
# - MODEL_PATH: absolute or repo-relative path to the model zip
#
# Typical use:
# - compare an old preserved model vs a newly trained model
# - optionally include default so the result is a complete 4-method table

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp44_common.sh"

cd "$REPO_ROOT"

MODEL_PATH="${MODEL_PATH:?MODEL_PATH must point to a model .zip}"
RUN_ID="${RUN_ID:-exp44_model_eval_20260515T1}"
EXP44_RUN_ROOT="${EXP44_RUN_ROOT:-$EXP44_RUNS_ROOT/$RUN_ID}"
EXP44_EVAL_DIR="${EXP44_EVAL_DIR:-$EXP44_RUN_ROOT/evaluation}"

echo "Evaluating model:"
echo "  $MODEL_PATH"
echo "Run id:"
echo "  $RUN_ID"

MODEL_PATH="$MODEL_PATH" \
MATRIX_GRID_N="$MATRIX_GRID_N" \
SETUP_PARAM_RESOLUTION="$SETUP_PARAM_RESOLUTION" \
EXPECTED_SETUP_ACTION_COUNT="$EXPECTED_SETUP_ACTION_COUNT" \
ACTION_MODE="$ACTION_MODE" \
ALGO="$ALGO" \
MODEL_TYPE="$MODEL_TYPE" \
OBS_MODE="$OBS_MODE" \
W_CENTER="$W_CENTER" \
W_SCALE="$W_SCALE" \
W_GLOBAL_MIN="$W_GLOBAL_MIN" \
W_GLOBAL_MAX="$W_GLOBAL_MAX" \
INITIAL_OBSERVATION_WEIGHT="$INITIAL_OBSERVATION_WEIGHT" \
FORWARD_METHODS="$FORWARD_METHODS" \
FORWARD_SEEDS="$FORWARD_SEEDS" \
TRACE_T="$TRACE_T" \
EVAL_A_NAME="$EVAL_A_NAME" \
EVAL_A_START="$EVAL_A_START" \
EVAL_A_END="$EVAL_A_END" \
EVAL_B_NAME="$EVAL_B_NAME" \
EVAL_B_START="$EVAL_B_START" \
EVAL_B_END="$EVAL_B_END" \
RUN_ID="$RUN_ID" \
EXP44_RESULTS_ROOT="$EXP44_RESULTS_ROOT" \
EXP44_EVAL_DIR="$EXP44_EVAL_DIR" \
PRIMARY_WINDOW="$PRIMARY_WINDOW" \
bash "$SCRIPT_DIR/evaluate_saved_model_live_forward.sh"

echo
echo "Summary:"
echo "  $EXP44_EVAL_DIR/forward_continuation_summary.json"
echo "Table:"
echo "  $EXP44_EVAL_DIR/main_table.csv"
echo "Figures:"
echo "  $EXP44_EVAL_DIR/figures"
