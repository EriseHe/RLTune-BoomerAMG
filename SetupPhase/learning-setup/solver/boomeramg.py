"""
High-level BoomerAMG solver for bandit experiments.

Thin Python wrapper around libamg_setup_solver.dylib (C calling HYPRE).
Matrix is built in C using the same Laplacian builders as the solve-phase
RL environment, ensuring identical problem instances.

    from solver import solve, SolveResult, TUNABLE_PARAMS

    result = solve(params={"strong_threshold": 0.25})
    print(result.work_units)
"""

from __future__ import annotations

import ctypes
import os
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# ---- load compiled C library ------------------------------------------------

def _default_lib_path() -> Path:
    # Allow overriding the solver shim library path for local setups.
    env = os.environ.get("AMG_SETUP_SOLVER_LIB", "").strip()
    if env:
        return Path(env)

    sysname = platform.system()
    if sysname == "Windows":
        ext = ".dll"
    elif sysname == "Darwin":
        ext = ".dylib"
    else:
        ext = ".so"
    return Path(__file__).parent / f"libamg_setup_solver{ext}"


_LIB_PATH = _default_lib_path()
if not _LIB_PATH.exists():
    raise RuntimeError(
        f"Compiled library not found at {_LIB_PATH}.\n"
        "Build it from `amg_setup_solver.c` and `SolvePhase/hypre/src/test/amg_cycle.c` "
        "and link it against HYPRE.\n"
        "On Windows, build a `libamg_setup_solver.dll` and keep it next to this file."
    )

# On Windows, ensure dependencies (e.g. HYPRE.dll) in this directory can be resolved.
if platform.system() == "Windows":
    os.add_dll_directory(str(Path(__file__).parent))

_lib = ctypes.CDLL(str(_LIB_PATH))

_VP = ctypes.c_void_p
_I = ctypes.c_int
_D = ctypes.c_double
_ULL = ctypes.c_ulonglong

_lib.amg_setup_create.restype = _VP
_lib.amg_setup_create.argtypes = [
    _I, _I, _I,       # nx, ny, nz
    _I,               # stencil_type (7 or 27)
    _I, _ULL,         # rhs_type, rhs_seed
    _D, _D,           # k, c         (7pt coeffs)
    _D, _D, _D, _D,   # a0, a1, a2, a3  (27pt coeffs)
]

_lib.amg_setup_solve.restype = _I
_lib.amg_setup_solve.argtypes = [
    _VP,                              # env
    _D, _I, _I, _D, _I, _I, _I, _I,  # strong_threshold..max_levels (incl. max_row_sum)
    _D, _I, _I, _I, _D, _I, _I,      # trunc_factor..max_coarse_size
    _D, _I,                           # tol, max_iter
    ctypes.POINTER(_I),               # out_iters
    ctypes.POINTER(_D),               # out_complexity
    ctypes.POINTER(_D),               # out_residual
    ctypes.POINTER(_D),               # out_runtime_sec (hypre_MPI_Wtime around Setup+Solve)
]

_lib.amg_setup_destroy.restype = None
_lib.amg_setup_destroy.argtypes = [_VP]

_lib.amg_setup_get_n.restype = _I
_lib.amg_setup_get_n.argtypes = [_VP]

_lib.amg_setup_get_nnz.restype = _I
_lib.amg_setup_get_nnz.argtypes = [_VP]

# ---- tunable params ---------------------------------------------------------

TUNABLE_PARAMS = {
    "strong_threshold": float,
    "coarsen_type":     int,
    "interp_type":      int,
    "max_row_sum":      float,
    "relax_type":       int,
    "num_sweeps":       int,
    "cycle_type":       int,
    "max_levels":       int,
    "trunc_factor":     float,
    "P_max_elmts":      int,
    "agg_num_levels":   int,
    "agg_interp_type":  int,
    "relax_wt":         float,
    "relax_order":      int,
    "max_coarse_size":  int,
}

_PARAM_ORDER = [
    "strong_threshold", "coarsen_type", "interp_type", "max_row_sum",
    "relax_type", "num_sweeps", "cycle_type", "max_levels",
    "trunc_factor", "P_max_elmts", "agg_num_levels", "agg_interp_type",
    "relax_wt", "relax_order", "max_coarse_size",
]

# ---- result -----------------------------------------------------------------

@dataclass
class SolveResult:
    iterations: int
    complexity: float
    runtime_sec: float
    residual_norm: float
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def work_units(self) -> float:
        return float(self.iterations) * self.complexity

# ---- public API -------------------------------------------------------------

def solve(
    params: Optional[Dict[str, Any]] = None,
    *,
    nx: int = 10, ny: int = 10, nz: int = 10,
    stencil: int = 7,
    rhs_type: int = 0,
    rhs_seed: int = 42,
    k: float = 1.0, c: float = 0.0,
    a0: float = 1.0, a1: float = 1.0, a2: float = 1.0, a3: float = 0.0,
    tol: float = 1e-8,
    max_iter: int = 10_000,
) -> SolveResult:
    """
    Build a Laplacian and solve with BoomerAMG using the given setup params.

    Parameters
    ----------
    params    : dict of setup-phase knobs (keys from TUNABLE_PARAMS).
    nx,ny,nz  : grid dimensions  (same as solve-phase environment).
    stencil   : 7 or 27 point stencil.
    rhs_type  : 0 = ones, 1 = random.
    k,c       : 7pt Laplacian coefficients.
    a0..a3    : 27pt Laplacian coefficients.
    tol       : relative convergence tolerance.
    max_iter  : V-cycle cap.

    Returns
    -------
    SolveResult  (.iterations, .complexity, .work_units, .residual_norm,
                  .params)
    """
    params = params or {}
    unknown = set(params) - set(TUNABLE_PARAMS)
    if unknown:
        raise ValueError(f"Unknown param(s): {unknown}")

    # Build matrix + vectors in C
    env = _lib.amg_setup_create(
        nx, ny, nz, stencil, rhs_type, rhs_seed,
        k, c, a0, a1, a2, a3,
    )
    if not env:
        raise RuntimeError("amg_setup_create failed")

    # Build C args: -1 sentinel => HYPRE default
    c_args = []
    for name in _PARAM_ORDER:
        if name in params:
            c_args.append(params[name])
        else:
            c_args.append(-1.0 if TUNABLE_PARAMS[name] is float else -1)

    out_iters = _I()
    out_comp  = _D()
    out_rt    = _D()
    out_res   = _D()

    try:
        rc = _lib.amg_setup_solve(env, *c_args, tol, max_iter,
                                  ctypes.byref(out_iters),
                                  ctypes.byref(out_comp),
                                  ctypes.byref(out_res),
                                  ctypes.byref(out_rt))
        if rc != 0:
            raise RuntimeError(f"amg_setup_solve returned {rc}")

        return SolveResult(
            iterations=out_iters.value,
            complexity=out_comp.value,
            runtime_sec=out_rt.value,
            residual_norm=out_res.value,
            params=dict(params),
        )
    finally:
        _lib.amg_setup_destroy(env)
