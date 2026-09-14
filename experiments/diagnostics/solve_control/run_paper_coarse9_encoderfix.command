#!/bin/zsh
# One-off paper rerun; invoke this file from an interactive terminal session.
set -u
set -o pipefail

cd '/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG' || exit 1
rltune_output='/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/results/joint/paper_n60_canonical8d_vs_physics_linear_coarse9_staged1000_5k_tol1e6_encoderfix_terminal_20260912'
rltune_log="${rltune_output}.log"

if [[ ! -t 0 || ! -t 1 ]]; then
  print -u2 'Run this script in an interactive terminal, not a detached/background job.'
  exit 1
fi
if [[ -e "$rltune_output" || -e "$rltune_log" ]]; then
  print -u2 "Refusing to overwrite existing run or log: $rltune_output"
  exit 1
fi

{
  print 'Paper rerun: 60^3 scalar diffusion / 5,000 problems / five branches.'
  print 'Launcher for this rerun: Codex PTY terminal; not macOS Terminal or launchd.'
  print 'Default; Canonical 8D and Physics 7D, each with default solve or LSTDQ v3.'
  print 'Smoothers 18/18/9; tolerance 1e-6; cap 50; corrected space_aware_v2 encoder.'
  print 'RL branches: setup-only cases 1-1000, joint learning cases 1001-5000.'
  print 'Keep this Terminal open and use AC power for comparable timing.'
  print 'Progress is printed every 100 cases; current plotter runs after completion.'
  /bin/date '+Started: %Y-%m-%d %H:%M:%S %z'
  /usr/bin/tty
  /usr/bin/pmset -g batt
  /usr/bin/shasum -a 256 \
    '/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/joint/solve_control/configs/paper_n60_canonical8d_vs_physics_linear_coarse9_staged1000_5k_tol1e6_encoderfix.json' \
    hypre/interfaces/libamg_runtime.dylib \
    hypre/install/lib/libHYPRE.3.0.0.dylib \
    solve/controllers/common/state_encoder.py \
    setup/space.py

  /usr/bin/caffeinate -i /usr/bin/env \
    MPLCONFIGDIR=/private/tmp/mplconfig_rtune PYTHONUNBUFFERED=1 \
    /opt/anaconda3/envs/rl/bin/python -u \
    experiments/joint/solve_control/run_joint_experiment.py \
    --config '/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG/experiments/joint/solve_control/configs/paper_n60_canonical8d_vs_physics_linear_coarse9_staged1000_5k_tol1e6_encoderfix.json'
} 2>&1 | /usr/bin/tee "$rltune_log"
rltune_status=$?
print "Paper experiment finished with exit code: $rltune_status"
exit "$rltune_status"
