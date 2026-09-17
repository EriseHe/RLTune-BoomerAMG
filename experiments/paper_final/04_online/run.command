#!/bin/bash
set -u
set -o pipefail
cd "$(dirname "$0")/../../.." || exit 1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
logs=results/paper_final/04_online/logs
mkdir -p "$logs"
printf '%s\n' "$$" > "$logs/diffusion_s1.pid"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/diffusion_s1.started.txt"
printf '04: diffusion seed 1, grids 40^3 -> 60^3 -> 80^3; three methods per grid.\n'
/usr/bin/caffeinate -i /opt/anaconda3/envs/rl/bin/python -u \
    -m experiments.paper_final.run_04_online --run --family diffusion --seed 1 2>&1 \
    | /usr/bin/tee -a "$logs/diffusion_s1.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$logs/diffusion_s1.exit_code"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/diffusion_s1.finished.txt"
printf '\nExit status %s. Results: results/paper_final/04_online/\n' "$run_status"
exit "$run_status"
