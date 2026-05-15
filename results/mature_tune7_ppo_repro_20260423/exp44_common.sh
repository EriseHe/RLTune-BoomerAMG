#!/usr/bin/env bash

# Shared defaults for the retained Exp44 workflow.
# Scripts source this file and can still override any variable from the shell.

REPO_ROOT="${REPO_ROOT:-/Users/jonathanwang/Desktop/RL_Hypre/RLTune-BoomerAMG}"
PYTHON_BIN="${PYTHON_BIN:-/Users/jonathanwang/anaconda3/envs/rl/bin/python}"

# Stable repo-local mature setup-bandit snapshot.
BANDIT_STATE="${BANDIT_STATE:-$REPO_ROOT/results/mature_tune7_ppo_repro_20260423/run_logs/mature40_tune7_bandit_state_case2.pkl}"

# Exp44 train/validation seeds.
RL_TRAIN_SEEDS="${RL_TRAIN_SEEDS:-39396939,39402939,39408939}"
RL_TRAIN_CASES="${RL_TRAIN_CASES:-500}"
EVAL_SEEDS="${EVAL_SEEDS:-39414939,39420939,39426939}"
EVAL_CASES="${EVAL_CASES:-500}"

# Canonical held-out forward seeds.
FORWARD_SEEDS="${FORWARD_SEEDS:-39393939,39394939,39400939,39406939,39412939}"
FORWARD_METHODS="${FORWARD_METHODS:-default,bandit,fixed,ppo}"
TRACE_T="${TRACE_T:-2500}"
TRAIN_SEGMENT_START="${TRAIN_SEGMENT_START:-1000}"
TRAIN_SEGMENT_END="${TRAIN_SEGMENT_END:-1500}"
EVAL_A_NAME="${EVAL_A_NAME:-eval_1000}"
EVAL_A_START="${EVAL_A_START:-1500}"
EVAL_A_END="${EVAL_A_END:-2500}"
EVAL_B_NAME="${EVAL_B_NAME:-eval_1000_dup}"
EVAL_B_START="${EVAL_B_START:-1500}"
EVAL_B_END="${EVAL_B_END:-2500}"

# Exp44 model design.
ACTION_MODE="${ACTION_MODE:-continuous_residual}"
ALGO="${ALGO:-ppo}"
MODEL_TYPE="${MODEL_TYPE:-lstm}"
OBS_MODE="${OBS_MODE:-cycle_action_setup}"
W_CENTER="${W_CENTER:-1.5}"
W_SCALE="${W_SCALE:-0.02}"
W_GLOBAL_MIN="${W_GLOBAL_MIN:-1.0}"
W_GLOBAL_MAX="${W_GLOBAL_MAX:-2.0}"

# PPO defaults.
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
ENT_COEF="${ENT_COEF:-0.0}"
N_STEPS="${N_STEPS:-256}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-2500}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-2500}"
CHECKPOINT_OBJECTIVE="${CHECKPOINT_OBJECTIVE:-seed_mean_minus_std_vs_bandit}"
