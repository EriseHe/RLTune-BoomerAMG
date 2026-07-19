#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper:
# 1) train a new Exp44-style model
# 2) evaluate the newly trained model
#
# The actual logic now lives in:
# - train_exp44_absolute_lstm.sh
# - eval_exp44_model.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp44_common.sh"

cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-exp44_absolute_default_lstm_canonical_20260718}"
EXP44_RUN_ROOT="${EXP44_RUN_ROOT:-$EXP44_RUNS_ROOT/$RUN_TAG}"
export EXP44_RUN_ROOT

echo "[1/2] Train new Exp44-style model into:"
echo "      $EXP44_RUN_ROOT/training"
bash "$SCRIPT_DIR/train_exp44_absolute_lstm.sh"

NEW_MODEL="$EXP44_RUN_ROOT/training/model_best.zip"

echo
echo "[2/2] Test newly trained model:"
echo "      $NEW_MODEL"
MODEL_PATH="$NEW_MODEL" \
EXP44_EVAL_DIR="$EXP44_RUN_ROOT/evaluation" \
RUN_ID="$RUN_TAG" \
bash "$SCRIPT_DIR/eval_exp44_model.sh"

echo
echo "Done."
echo "Run root:           $EXP44_RUN_ROOT"
echo "Training result:    $EXP44_RUN_ROOT/training/result.json"
echo "Evaluation result:  $EXP44_RUN_ROOT/evaluation/forward_continuation_summary.json"
echo "Evaluation table:   $EXP44_RUN_ROOT/evaluation/main_table.csv"
echo "Figures:            $EXP44_RUN_ROOT/evaluation/figures"
