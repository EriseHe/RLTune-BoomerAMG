#!/usr/bin/env bash
set -euo pipefail

# Build libamg_setup_solver.dylib from repository-relative sources.
# This wrapper exists so a fresh checkout can build the setup-phase shared
# library without editing any absolute path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Common macOS Homebrew location for OpenMPI. This is only a fallback.
if ! command -v mpicc >/dev/null 2>&1 && [[ -x /opt/homebrew/bin/mpicc ]]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi

if ! command -v mpicc >/dev/null 2>&1; then
  echo "error: mpicc not found in PATH" >&2
  echo "Install MPI or export PATH to a directory containing mpicc." >&2
  exit 1
fi

cd "$SCRIPT_DIR"
make clean
make all MPICC="$(command -v mpicc)"
