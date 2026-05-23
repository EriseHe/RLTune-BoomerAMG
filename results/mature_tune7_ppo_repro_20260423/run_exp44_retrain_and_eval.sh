#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper:
# 1) train a new Exp44-style model
# 2) evaluate the preserved old model
# 3) evaluate the newly trained model
#
# The actual logic now lives in:
# - train_exp44_midpoint_lstm.sh
# - eval_exp44_model.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp44_common.sh"

cd "$REPO_ROOT"

OLD_MODEL="${OLD_MODEL:-$REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/exp44_midpoint_lstm_seedmeanstd_vsbandit_20260512T1/model_ckpt_2500.zip}"
RUN_TAG="${RUN_TAG:-exp44_midpoint_lstm_seedmeanstd_vsbandit_20260515T1}"
RUN_DIR="$REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/$RUN_TAG"

echo "[1/3] Train new Exp44-style model into:"
echo "      $RUN_DIR"
bash results/mature_tune7_ppo_repro_20260423/train_exp44_midpoint_lstm.sh

NEW_MODEL="$RUN_DIR/model_ckpt_${CHECKPOINT_INTERVAL:-2500}.zip"

#if we dont have old model we can skip the cmd below
# echo
# echo "[2/3] Re-test old preserved model:"
# echo "      $OLD_MODEL"
# MODEL_PATH="$OLD_MODEL" \
# RUN_ID="${RUN_TAG}_old_model_eval" \
# bash results/mature_tune7_ppo_repro_20260423/eval_exp44_model.sh

echo
echo "[3/3] Test newly trained model:"
echo "      $NEW_MODEL"
MODEL_PATH="$NEW_MODEL" \
RUN_ID="${RUN_TAG}_new_model_eval" \
bash results/mature_tune7_ppo_repro_20260423/eval_exp44_model.sh

echo
echo "Done."
echo "New train dir:      $RUN_DIR"
# echo "Old eval summary:   $REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/${RUN_TAG}_old_model_eval/forward_continuation_summary.json"
echo "New eval summary:   $REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/${RUN_TAG}_new_model_eval/forward_continuation_summary.json"
