#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This build helper is macOS-only." >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
HYPRE_SRC="${REPO_ROOT}/SolvePhase/hypre/src"
BUILD_ROOT="${BUILD_ROOT:-${REPO_ROOT}/build}"
HYPRE_BUILD="${BUILD_ROOT}/hypre-macos"
HYPRE_INSTALL="${BUILD_ROOT}/hypre-macos-install"
STAGE_DIR="${BUILD_ROOT}/setup-solver-macos"
SOLVER_DIR="${SCRIPT_DIR}"

require_cmd cmake
require_cmd mpicc
require_cmd install_name_tool

if [[ ! -d "${HYPRE_SRC}" ]]; then
  echo "Missing vendored HYPRE source tree at ${HYPRE_SRC}" >&2
  exit 1
fi

MPICC="${MPICC:-$(command -v mpicc)}"
DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-$(sw_vers -productVersion | awk -F. '{print $1 "." $2}')}"
JOBS="${JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

mkdir -p "${HYPRE_BUILD}" "${HYPRE_INSTALL}" "${STAGE_DIR}"

echo "==> Building vendored HYPRE for macOS ${DEPLOYMENT_TARGET}"
cmake -S "${HYPRE_SRC}" -B "${HYPRE_BUILD}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DHYPRE_ENABLE_MPI=ON \
  -DHYPRE_BUILD_TESTS=OFF \
  -DHYPRE_BUILD_EXAMPLES=OFF \
  -DCMAKE_C_COMPILER="${MPICC}" \
  -DCMAKE_INSTALL_PREFIX="${HYPRE_INSTALL}" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${DEPLOYMENT_TARGET}"

cmake --build "${HYPRE_BUILD}" --parallel "${JOBS}" --target install

HYPRE_DYLIB_SRC="${HYPRE_INSTALL}/lib/libHYPRE.3.0.0.dylib"
HYPRE_DYLIB_STAGE="${STAGE_DIR}/libHYPRE.3.0.0.dylib"
SOLVER_DYLIB_STAGE="${STAGE_DIR}/libamg_setup_solver.dylib"

if [[ ! -f "${HYPRE_DYLIB_SRC}" ]]; then
  echo "Expected HYPRE dylib not found at ${HYPRE_DYLIB_SRC}" >&2
  exit 1
fi

cp "${HYPRE_DYLIB_SRC}" "${HYPRE_DYLIB_STAGE}"
chmod u+w "${HYPRE_DYLIB_STAGE}"
install_name_tool -id "@loader_path/$(basename "${HYPRE_DYLIB_STAGE}")" "${HYPRE_DYLIB_STAGE}"

COMMON_CFLAGS=(
  -O3
  -fPIC
  -mmacosx-version-min="${DEPLOYMENT_TARGET}"
  -I"${HYPRE_INSTALL}/include"
)

echo "==> Building setup-phase solver shim"
"${MPICC}" "${COMMON_CFLAGS[@]}" \
  -c "${SOLVER_DIR}/amg_setup_solver.c" \
  -o "${STAGE_DIR}/amg_setup_solver.o"

"${MPICC}" "${COMMON_CFLAGS[@]}" \
  -Dmain=amg_cycle_unused_main \
  -Dhypre_printf=amg_setup_quiet_printf \
  -c "${REPO_ROOT}/SolvePhase/hypre/src/test/amg_cycle.c" \
  -o "${STAGE_DIR}/amg_cycle.o"

"${MPICC}" -dynamiclib \
  -o "${SOLVER_DYLIB_STAGE}" \
  "${STAGE_DIR}/amg_setup_solver.o" \
  "${STAGE_DIR}/amg_cycle.o" \
  "${HYPRE_DYLIB_STAGE}" \
  -mmacosx-version-min="${DEPLOYMENT_TARGET}" \
  -Wl,-install_name,@loader_path/libamg_setup_solver.dylib \
  -Wl,-rpath,@loader_path

install_name_tool -id "@loader_path/libamg_setup_solver.dylib" "${SOLVER_DYLIB_STAGE}"

cp "${HYPRE_DYLIB_STAGE}" "${SOLVER_DIR}/libHYPRE.3.0.0.dylib"
cp "${SOLVER_DYLIB_STAGE}" "${SOLVER_DIR}/libamg_setup_solver.dylib"

echo "==> Built macOS runtime libraries"
echo "    ${SOLVER_DIR}/libHYPRE.3.0.0.dylib"
echo "    ${SOLVER_DIR}/libamg_setup_solver.dylib"
echo
echo "Next step:"
echo "  python -c \"import sys; sys.path.insert(0, 'SetupPhase/learning-setup'); import solver; print('solver import ok')\""
