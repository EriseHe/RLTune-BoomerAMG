"""
High-level BoomerAMG solver for bandit experiments.

Thin Python wrapper around libamg_setup_solver.dylib (C calling HYPRE).
Matrix is built in C using the same Laplacian builders as the solve-phase
RL environment, ensuring identical problem instances.

    from hypre.bindings import solve, SolveResult, TUNABLE_PARAMS

    result = solve(params={"strong_threshold": 0.25})
    print(result.work_units)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# ---- load compiled C library ------------------------------------------------

_SOLVER_DIR = Path(__file__).parent
_REPO_ROOT = _SOLVER_DIR.parents[1]
_LIB_PATH = _REPO_ROOT / "hypre" / "interfaces" / "libamg_setup_solver.dylib"
_BUILD_SCRIPT = _SOLVER_DIR / "build_libamg_setup_solver.sh"
if not _LIB_PATH.exists():
    raise RuntimeError(
        f"Compiled library not found at {_LIB_PATH}.\n"
        f"Build it from this directory with:\n"
        f"  bash {_BUILD_SCRIPT}\n"
        "or run:\n"
        "  make"
    )
try:
    _lib = ctypes.CDLL(str(_LIB_PATH))
except OSError as exc:
    raise RuntimeError(
        f"Failed to load {_LIB_PATH}.\n"
        "Rebuild the setup solver with the portable wrapper:\n"
        f"  bash {_BUILD_SCRIPT}\n"
        "Then inspect the linked HYPRE path with:\n"
        f"  otool -L {_LIB_PATH}\n"
        f"  otool -l {_LIB_PATH} | rg 'LC_RPATH|path'\n"
        f"Original loader error: {exc}"
    ) from exc

_VP = ctypes.c_void_p
_I = ctypes.c_int
_D = ctypes.c_double
_ULL = ctypes.c_ulonglong

_lib.amg_setup_create.restype = _VP
_lib.amg_setup_create.argtypes = [
    _I, _I, _I,       # nx, ny, nz
    _I,               # stencil_type (0=difconv, 7 or 27=laplacian)
    _I, _ULL,         # rhs_type, rhs_seed
    _D, _D,           # k, c         (7pt coeffs)
    _D, _D, _D, _D,   # a0, a1, a2, a3  (27pt coeffs)
]

_lib.amg_setup_solve.restype = _I
_lib.amg_setup_solve.argtypes = [
    _VP,                              # env
    _D, _I, _I, _D, _I, _I, _I, _I,  # strong_threshold..max_levels (incl. max_row_sum)
    _D, _I, _I, _I, _D, _I, _D, _I, _I,      # trunc_factor..max_coarse_size
    _D, _I,                           # tol, max_iter
    ctypes.POINTER(_I),               # out_iters
    ctypes.POINTER(_D),               # out_complexity
    ctypes.POINTER(_D),               # out_residual
    ctypes.POINTER(_D),               # out_runtime_sec (hypre_MPI_Wtime around Setup+Solve)
    ctypes.POINTER(_D),               # out_setup_runtime_sec
    ctypes.POINTER(_D),               # out_solve_runtime_sec
]

_lib.amg_setup_destroy.restype = None
_lib.amg_setup_destroy.argtypes = [_VP]

_lib.amg_setup_get_n.restype = _I
_lib.amg_setup_get_n.argtypes = [_VP]

_lib.amg_setup_get_nnz.restype = _I
_lib.amg_setup_get_nnz.argtypes = [_VP]

_lib.amg_setup_prepare_rl.restype = _I
_lib.amg_setup_prepare_rl.argtypes = [
    _VP,
    _D, _I, _I, _D, _I, _I, _I, _I,
    _D, _I, _I, _I, _D, _I, _D, _I, _I,
    ctypes.POINTER(_D),
    ctypes.POINTER(_D),
]

_lib.amg_setup_step_rl.restype = _I
_lib.amg_setup_step_rl.argtypes = [
    _VP,
    _D, _I, _I, _I, _I, _I, _I, _I, _I, _I, _D, _D, _D, _I, _D, _I,
    ctypes.POINTER(_D),
    ctypes.POINTER(_D),
]

_lib.amg_setup_get_r0.restype = _D
_lib.amg_setup_get_r0.argtypes = [_VP]

_lib.amg_setup_get_r.restype = _D
_lib.amg_setup_get_r.argtypes = [_VP]

_lib.amg_setup_get_relax_weight.restype = _I
_lib.amg_setup_get_relax_weight.argtypes = [_VP, _I, ctypes.POINTER(_D)]

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
    "agg_tr":           float,
    "agg_Pmx":          int,
    "relax_wt":         float,
    "relax_order":      int,
    "max_coarse_size":  int,
}

_PARAM_ORDER = [
    "strong_threshold", "coarsen_type", "interp_type", "max_row_sum",
    "relax_type", "num_sweeps", "cycle_type", "max_levels",
    "trunc_factor", "P_max_elmts", "agg_num_levels", "agg_interp_type",
    "agg_tr", "agg_Pmx",
    "relax_wt", "relax_order", "max_coarse_size",
]

# ---- result -----------------------------------------------------------------

@dataclass
class SolveResult:
    iterations: int
    complexity: float
    runtime_sec: float
    setup_runtime_sec: float
    solve_runtime_sec: float
    residual_norm: float
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def work_units(self) -> float:
        return float(self.iterations) * self.complexity


@dataclass
class PrepareResult:
    setup_runtime_sec: float
    initial_residual_norm: float
    initial_relax_weight: float


class PreparedAMGEnv:
    def __init__(self, env_ptr: int):
        self._env = env_ptr

    def prepare_rl(self, params: Optional[Dict[str, Any]] = None) -> PrepareResult:
        params = params or {}
        unknown = set(params) - set(TUNABLE_PARAMS)
        if unknown:
            raise ValueError(f"Unknown param(s): {unknown}")

        c_args = []
        for name in _PARAM_ORDER:
            if name in params:
                c_args.append(params[name])
            else:
                c_args.append(-1.0 if TUNABLE_PARAMS[name] is float else -1)

        out_setup = _D()
        out_r0 = _D()
        rc = _lib.amg_setup_prepare_rl(self._env, *c_args, ctypes.byref(out_setup), ctypes.byref(out_r0))
        if rc != 0:
            raise RuntimeError(f"amg_setup_prepare_rl returned {rc}")
        out_relax_weight = _D()
        rc = _lib.amg_setup_get_relax_weight(
            self._env,
            0,
            ctypes.byref(out_relax_weight),
        )
        if rc != 0:
            raise RuntimeError(f"amg_setup_get_relax_weight returned {rc}")
        return PrepareResult(
            setup_runtime_sec=float(out_setup.value),
            initial_residual_norm=float(out_r0.value),
            initial_relax_weight=float(out_relax_weight.value),
        )

    def step_rl(
        self,
        *,
        relax_weight: float,
        sweeps_down: int,
        sweeps_up: int,
        coarse_sweeps: int | None = None,
        cycle_type: int | None = None,
        relax_type: int | None = None,
        pre_relax_type: int | None = None,
        post_relax_type: int | None = None,
        coarse_relax_type: int | None = None,
        relax_order: int | None = None,
        outer_weight: float | None = None,
        add_relax_weight: float | None = None,
        level_relax_weight: float | None = None,
        level_relax_level: int | None = None,
        level_outer_weight: float | None = None,
        level_outer_level: int | None = None,
    ) -> tuple[float, float]:
        out_r = _D()
        out_rt = _D()
        rc = _lib.amg_setup_step_rl(
            self._env,
            float(relax_weight),
            int(sweeps_down),
            int(sweeps_up),
            (-1 if coarse_sweeps is None else int(coarse_sweeps)),
            (-1 if cycle_type is None else int(cycle_type)),
            (-1 if relax_type is None else int(relax_type)),
            (-1 if pre_relax_type is None else int(pre_relax_type)),
            (-1 if post_relax_type is None else int(post_relax_type)),
            (-1 if coarse_relax_type is None else int(coarse_relax_type)),
            (-1 if relax_order is None else int(relax_order)),
            (-1.0 if outer_weight is None else float(outer_weight)),
            (-1.0 if add_relax_weight is None else float(add_relax_weight)),
            (-1.0 if level_relax_weight is None else float(level_relax_weight)),
            (-1 if level_relax_level is None else int(level_relax_level)),
            (-1.0 if level_outer_weight is None else float(level_outer_weight)),
            (-1 if level_outer_level is None else int(level_outer_level)),
            ctypes.byref(out_r),
            ctypes.byref(out_rt),
        )
        if rc != 0:
            raise RuntimeError(f"amg_setup_step_rl returned {rc}")
        return float(out_r.value), float(out_rt.value)

    @property
    def r0(self) -> float:
        return float(_lib.amg_setup_get_r0(self._env))

    @property
    def r(self) -> float:
        return float(_lib.amg_setup_get_r(self._env))

    def close(self) -> None:
        if self._env:
            _lib.amg_setup_destroy(self._env)
            self._env = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

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
    tol: Optional[float] = None,
    max_iter: Optional[int] = None,
) -> SolveResult:
    """
    Build a matrix (Laplacian or DifConv) and solve with BoomerAMG using the
    given setup params.

    Parameters
    ----------
    params    : dict of setup-phase knobs (keys from TUNABLE_PARAMS).
    nx,ny,nz  : grid dimensions  (same as solve-phase environment).
    stencil   : 7 or 27 point Laplacian stencil, or 0 for DifConv.
    rhs_type  : 0 = ones, 1 = random.
    k,c       : 7pt Laplacian coefficients.
    a0..a3    : 27pt Laplacian coefficients, or DifConv coefficients for
                stencil=0 (mapping matches solve-phase: cx=k, cy=c, cz=a0,
                ax=a1, ay=a2, az=a3).
    tol       : relative convergence tolerance. `None` => HYPRE default.
    max_iter  : V-cycle cap. `None` => HYPRE default.

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
    out_setup_rt = _D()
    out_solve_rt = _D()
    out_res   = _D()

    try:
        c_tol = (-1.0 if tol is None else float(tol))
        c_max_iter = (-1 if max_iter is None else int(max_iter))
        rc = _lib.amg_setup_solve(env, *c_args, c_tol, c_max_iter,
                                  ctypes.byref(out_iters),
                                  ctypes.byref(out_comp),
                                  ctypes.byref(out_res),
                                  ctypes.byref(out_rt),
                                  ctypes.byref(out_setup_rt),
                                  ctypes.byref(out_solve_rt))
        if rc != 0:
            raise RuntimeError(f"amg_setup_solve returned {rc}")

        return SolveResult(
            iterations=out_iters.value,
            complexity=out_comp.value,
            runtime_sec=out_rt.value,
            setup_runtime_sec=out_setup_rt.value,
            solve_runtime_sec=out_solve_rt.value,
            residual_norm=out_res.value,
            params=dict(params),
        )
    finally:
        _lib.amg_setup_destroy(env)


def create_env(
    *,
    nx: int = 10, ny: int = 10, nz: int = 10,
    stencil: int = 7,
    rhs_type: int = 0,
    rhs_seed: int = 42,
    k: float = 1.0, c: float = 0.0,
    a0: float = 1.0, a1: float = 1.0, a2: float = 1.0, a3: float = 0.0,
) -> PreparedAMGEnv:
    env = _lib.amg_setup_create(
        nx, ny, nz, stencil, rhs_type, rhs_seed,
        k, c, a0, a1, a2, a3,
    )
    if not env:
        raise RuntimeError("amg_setup_create failed")
    return PreparedAMGEnv(env)
