#!/bin/bash
set -u
set -o pipefail
cd /Users/erisehe/Documents/GitHub/RLTune-BoomerAMG || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
staging=results/paper_final/03_activation/.advection_80_s2.launch
if [[ -e results/paper_final/03_activation/advection_80_s2 ]]; then
    printf 'This run already exists. Check its progress and logs; it will not be overwritten.\n'
    exit 1
fi
mkdir -p "$staging/logs"
printf '03: one fresh seed, activation after 250/500/750, 5000 inputs, 80^3 diffusion-advection.\nKeep the Mac on power with its lid open.\n'
/usr/bin/caffeinate -i /opt/anaconda3/envs/rl/bin/python -u \
    -m experiments.paper_final.run_03_single \
    --config experiments/paper_final/03_activation/early_starts.json 2>&1 \
    | /usr/bin/tee -a "$staging/logs/launcher.log"
run_status=${PIPESTATUS[0]}
printf '\nExit status %s. Results: results/paper_final/03_activation/advection_80_s2/\n' "$run_status"
exit "$run_status"
