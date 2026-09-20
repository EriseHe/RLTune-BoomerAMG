#!/bin/bash
set -u
set -o pipefail
cd "$(dirname "$0")/../../../.." || exit 1
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'Commit the reviewed source/config changes before launching this suite.\n' >&2
    exit 1
fi
launch_commit=$(git rev-parse HEAD)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
result_root=results/paper_final/04_online/20260920_formal
logs="$result_root/logs"
mkdir -p "$logs"
printf '%s\n' "$$" > "$logs/suite.pid"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/suite.requested.txt"
printf '%s\n' "$launch_commit" > "$logs/suite.commit.txt"
if [[ "$(/usr/bin/pmset -g batt)" != *"AC Power"* ]]; then
    printf 'Waiting for AC power before this long timing comparison.\n' | /usr/bin/tee "$logs/waiting_for_ac.txt"
    while [[ "$(/usr/bin/pmset -g batt)" != *"AC Power"* ]]; do
        /bin/sleep 30
    done
fi
if [[ "$(git rev-parse HEAD)" != "$launch_commit" || -n "$(git status --porcelain --untracked-files=normal)" ]]; then
    printf 'Source changed while waiting for power; refusing to launch a different version.\n' >&2
    exit 1
fi
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/suite.started.txt"
/usr/bin/pmset -g batt > "$logs/power_at_start.txt"
printf '04: seed 56700120; diffusion and advection 40/60/80; cap 50; 5000 paired problems per group and method; original failure rollback.\n'
/usr/bin/caffeinate -is /opt/anaconda3/envs/rl/bin/python -u \
    -m experiments.paper_final.run_04_online --run \
    --suite experiments/paper_final/04_online/20260920_formal/suite.json \
    --output-root "$result_root" 2>&1 | /usr/bin/tee -a "$logs/suite.log"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$logs/suite.exit_code"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/suite.finished.txt"
printf '\nExit status %s. Results: %s\n' "$run_status" "$result_root"
exit "$run_status"
