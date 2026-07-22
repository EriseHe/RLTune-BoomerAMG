#!/usr/bin/env bash
set -euo pipefail

# Numbered wrapper flow:
# 1) archive the input MODEL_PATH into the run directory
# 2) for each held-out seed, call eval_forward_continuation_dual.py
# 3) aggregate all per-seed JSON files into one summary JSON

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/exp44_common.sh"
REPO="$REPO_ROOT"
OUT_DIR="$EXP44_RESULTS_ROOT"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi
MODEL_PATH="${MODEL_PATH:-}"
FORWARD_METHODS="${FORWARD_METHODS:-default,bandit,fixed,ppo}"
FORWARD_SEEDS="${FORWARD_SEEDS:-39393939,39394939,39400939}"
TRACE_T="${TRACE_T:-2500}"
TRAIN_SEGMENT_START="${TRAIN_SEGMENT_START:-1000}"
TRAIN_SEGMENT_END="${TRAIN_SEGMENT_END:-1500}"
EVAL_A_NAME="${EVAL_A_NAME:-eval_1000}"
EVAL_A_START="${EVAL_A_START:-1500}"
EVAL_A_END="${EVAL_A_END:-2500}"
EVAL_B_NAME="${EVAL_B_NAME:-eval_1000_dup}"
EVAL_B_START="${EVAL_B_START:-1500}"
EVAL_B_END="${EVAL_B_END:-2500}"
RUN_ID="${RUN_ID:-eval_saved_model_$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="${EXP44_EVAL_DIR:-$EXP44_RUNS_ROOT/$RUN_ID/evaluation}"
RUN_LOG="$RUN_DIR/run.log"
RUN_META="$RUN_DIR/meta.txt"
SNAPSHOT_DIR="$RUN_DIR/outputs"
ARCHIVE_DIR="$RUN_DIR/model_snapshot"
PER_SEED_DIR="$RUN_DIR/per_seed"

if [[ -z "$MODEL_PATH" ]]; then
  echo "MODEL_PATH is required" >&2
  exit 2
fi

cd "$REPO"
mkdir -p "$RUN_DIR" "$SNAPSHOT_DIR" "$ARCHIVE_DIR" "$PER_SEED_DIR"

MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_SHA="$("$PYTHON_BIN" - <<'PY' "$MODEL_PATH"
from __future__ import annotations
import hashlib, sys
path = sys.argv[1]
h = hashlib.sha256()
with open(path, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
)"
ARCHIVED_MODEL_PATH="$ARCHIVE_DIR/${MODEL_BASENAME%.zip}_$MODEL_SHA.zip"
cp "$MODEL_PATH" "$ARCHIVED_MODEL_PATH"

GIT_REV="$(git rev-parse HEAD 2>/dev/null || echo unknown)"

{
  echo "run_id=$RUN_ID"
  echo "time_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "repo=$REPO"
  echo "script=$SCRIPT_DIR/evaluate_saved_model_live_forward.sh"
  echo "python_bin=$PYTHON_BIN"
  echo "git_rev=$GIT_REV"
  echo "source_model_path=$MODEL_PATH"
  echo "archived_model_path=$ARCHIVED_MODEL_PATH"
  echo "model_sha256=$MODEL_SHA"
  echo "forward_methods=$FORWARD_METHODS"
  echo "forward_seeds=$FORWARD_SEEDS"
  echo "trace_t=$TRACE_T"
  echo "train_segment=$TRAIN_SEGMENT_START:$TRAIN_SEGMENT_END"
  echo "eval_a=$EVAL_A_NAME:$EVAL_A_START:$EVAL_A_END"
  echo "eval_b=$EVAL_B_NAME:$EVAL_B_START:$EVAL_B_END"
} >"$RUN_META"

IFS=',' read -r -a FORWARD_SEED_ARRAY <<<"$FORWARD_SEEDS"

echo "[1/2] same-trace forward continuation evals"
for seed in "${FORWARD_SEED_ARRAY[@]}"; do
  seed="$(echo "$seed" | xargs)"
  [[ -n "$seed" ]] || continue
  echo "  seed=$seed"
  {
    echo "===== forward_continuation_seed_$seed ====="
    env \
      FORWARD_SEED="$seed" \
      MODEL_PATH="$ARCHIVED_MODEL_PATH" \
      FORWARD_METHODS="$FORWARD_METHODS" \
      TRACE_T="$TRACE_T" \
      TRAIN_SEGMENT_START="$TRAIN_SEGMENT_START" \
      TRAIN_SEGMENT_END="$TRAIN_SEGMENT_END" \
      EVAL_A_NAME="$EVAL_A_NAME" \
      EVAL_A_START="$EVAL_A_START" \
      EVAL_A_END="$EVAL_A_END" \
      EVAL_B_NAME="$EVAL_B_NAME" \
      EVAL_B_START="$EVAL_B_START" \
      EVAL_B_END="$EVAL_B_END" \
      EXP44_RESULTS_ROOT="$EXP44_RESULTS_ROOT" \
      RESULT_PATH="$PER_SEED_DIR/forward_seed${seed}.json" \
      "$PYTHON_BIN" "$SCRIPT_DIR/eval_forward_continuation_dual.py"
  } | tee -a "$RUN_LOG"
done

echo "[2/2] aggregate forward summary"
{
  echo "===== aggregate_forward_from_run ====="
  env \
    RUN_DIR="$RUN_DIR" \
    PER_SEED_DIR="$PER_SEED_DIR" \
    FORWARD_SEEDS="$FORWARD_SEEDS" \
    EVAL_A_NAME="$EVAL_A_NAME" \
    EVAL_B_NAME="$EVAL_B_NAME" \
    PRIMARY_WINDOW="${PRIMARY_WINDOW:-$EVAL_A_NAME}" \
    SUMMARY_PATH="$RUN_DIR/forward_continuation_summary.json" \
    "$PYTHON_BIN" "$SCRIPT_DIR/aggregate_forward_continuation_from_run.py"
} | tee -a "$RUN_LOG"

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-exp44}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/report_exp44_results.py" evaluation \
  --result "$RUN_DIR/forward_continuation_summary.json" \
  --output-dir "$RUN_DIR" \
  --window "${PRIMARY_WINDOW:-$EVAL_A_NAME}" | tee -a "$RUN_LOG"

cp "$RUN_DIR/forward_continuation_summary.json" "$SNAPSHOT_DIR/"

echo
echo "Done. Key outputs:"
echo "  $RUN_DIR/forward_continuation_summary.json"
echo "  $RUN_DIR/main_table.csv"
echo "  $RUN_DIR/figures"
echo "  $ARCHIVED_MODEL_PATH"
echo "  $RUN_LOG"
echo "  $RUN_META"
echo "  $SNAPSHOT_DIR"
