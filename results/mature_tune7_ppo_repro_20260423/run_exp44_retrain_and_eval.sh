#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper:
# 1) train a new Exp44-style model
# 2) evaluate the newly trained model
#
# The actual logic now lives in:
# - train_exp44_midpoint_lstm.sh
# - eval_exp44_model.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp44_common.sh"

cd "$REPO_ROOT"

RUN_TAG="${RUN_TAG:-exp44_absolute_default_lstm_canonical_20260718}"
RUN_DIR="$REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/$RUN_TAG"

echo "[1/3] Train new Exp44-style model into:"
echo "      $RUN_DIR"
bash "$SCRIPT_DIR/train_exp44_midpoint_lstm.sh"

NEW_MODEL="$RUN_DIR/model_best.zip"

echo
echo "[2/2] Test newly trained model:"
echo "      $NEW_MODEL"
MODEL_PATH="$NEW_MODEL" \
RUN_ID="${RUN_TAG}_new_model_eval" \
bash "$SCRIPT_DIR/eval_exp44_model.sh"

echo
echo "Done."
echo "New train dir:      $RUN_DIR"
echo "New eval summary:   $REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/${RUN_TAG}_new_model_eval/forward_continuation_summary.json"
