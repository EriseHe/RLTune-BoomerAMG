#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
if [[ -n "$(git status --porcelain)" ]]; then
    printf 'Commit the source and protocol before launching.\n' >&2
    exit 1
fi
if [[ -e results/paper_final/05_policy/controlled || -e results/paper_final/09_algorithms/timing ]]; then
    printf 'Existing study outputs found; refusing to overwrite them.\n' >&2
    exit 1
fi
launch_commit=$(git rev-parse HEAD)
native_hash=$(shasum -a 256 hypre/interfaces/libamg_runtime.dylib | cut -d ' ' -f 1)
logs=results/paper_final/05_policy/controlled_launch
mkdir -p "$logs"
if [[ -f "$logs/launcher.pid" ]]; then
    previous_pid=$(cat "$logs/launcher.pid")
    if [[ "$previous_pid" =~ ^[0-9]+$ ]] && kill -0 "$previous_pid" 2>/dev/null && \
       [[ "$(ps -p "$previous_pid" -o command=)" == *"05_policy/run.command"* ]]; then
        printf 'This study already has a live launcher.\n' >&2
        exit 1
    fi
fi
printf '%s\n' "$$" > "$logs/launcher.pid"
printf '%s\n' "$launch_commit" > "$logs/commit.txt"
printf '%s\n' "$native_hash" > "$logs/native.sha256"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/requested.txt"
while [[ "$(/usr/bin/pmset -g batt)" != *"AC Power"* ]]; do
    printf '{"status":"waiting_for_ac"}\n' > "$logs/status.json"
    /bin/sleep 30
done
if [[ "$(git rev-parse HEAD)" != "$launch_commit" || -n "$(git status --porcelain)" || \
      "$(shasum -a 256 hypre/interfaces/libamg_runtime.dylib | cut -d ' ' -f 1)" != "$native_hash" ]]; then
    printf '{"status":"failed","reason":"source or native binary changed while waiting"}\n' > "$logs/status.json"
    exit 1
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
/usr/bin/pmset -g batt > "$logs/power_at_start.txt"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/started.txt"
printf '{"status":"running"}\n' > "$logs/status.json"
set +e
/usr/bin/caffeinate -is /opt/anaconda3/envs/rl/bin/python -u \
    -m experiments.paper_final.run_05_controlled --run > "$logs/study.log" 2>&1
run_status=$?
set -e
printf '%s\n' "$run_status" > "$logs/exit_code"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$logs/finished.txt"
if [[ "$run_status" == 0 ]]; then
    printf '{"status":"complete"}\n' > "$logs/status.json"
else
    printf '{"status":"failed"}\n' > "$logs/status.json"
fi
exit "$run_status"
