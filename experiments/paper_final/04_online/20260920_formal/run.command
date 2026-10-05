#!/usr/bin/env bash
# Run the frozen six-group design with the active Python environment.
set -euo pipefail
cd "$(dirname "$0")/../../../.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MPLBACKEND=Agg
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/paper_final/04_online/reproduction_formal}"
exec "$PYTHON_BIN" -u -m experiments.paper_final.run_04_online --run \
    --suite experiments/paper_final/04_online/20260920_formal/suite.json \
    --output-root "$OUTPUT_ROOT"
