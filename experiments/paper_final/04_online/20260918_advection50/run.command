#!/bin/bash
set -u
set -o pipefail
cd "$(dirname "$0")/../../../.." || exit 1
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'Commit the reviewed source/config changes before launching this suite.\n' >&2
    exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
result_root=results/paper_final/04_online/20260918_advection50
logs="$result_root/logs"
mkdir -p "$logs"
printf '%s\n' "$$" > "$logs/suite.pid"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/suite.started.txt"
git rev-parse HEAD > "$logs/suite.commit.txt"
printf '04: seed 56700120; diffusion-advection only, cap 50, grids 40/60/80; three methods per group.\n'
/usr/bin/caffeinate -is /opt/anaconda3/envs/rl/bin/python -u \
    -m experiments.paper_final.run_04_online --run \
    --suite experiments/paper_final/04_online/20260918_advection50/suite.json \
    --output-root "$result_root" 2>&1 | /usr/bin/tee -a "$logs/suite.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$logs/suite.exit_code"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/suite.finished.txt"
printf '\nExit status %s. Results: %s\n' "$run_status" "$result_root"
exit "$run_status"
