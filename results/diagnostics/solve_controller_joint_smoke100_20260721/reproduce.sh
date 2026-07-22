#!/usr/bin/env bash
set -euo pipefail
OUTPUT_DIR="${OUTPUT_DIR:-results/diagnostics/solve_controller_joint_smoke100_20260721_reproduction}"
/opt/anaconda3/envs/rl/bin/python -u experiments/joint/solve_control/run_joint_online_sarsa_4k.py --output-dir "$OUTPUT_DIR" --study-mode solve_controller_screen --seed 42900039 --bandit-seed 42960039 --controller-seed 42966039 --method-order-seed 42972039 --train-cases 100 --warmup-cases 0 --online-cases 100 --train-seed-groups 42900039,42906039 --train-shuffle-seeds 42948039 --train-cases-per-seed 50 --train-group-take 100 --matrix-grid-n 60 --setup-param-resolution 20 --max-cycles 50 --shared-action-profile 1to3_step0p05 --recursive-lstdq-v2-beta 4.0 --recalibrated-lsvi-beta 1.0 --progress-every 20 --smoke
