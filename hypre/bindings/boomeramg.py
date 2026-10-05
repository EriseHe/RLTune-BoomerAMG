"""
High-level BoomerAMG runtime shared by setup and solve experiments.

Thin Python wrapper around libamg_runtime (C calling HYPRE).
Matrix is built in C using the same Laplacian builders as the solve-phase
RL environment, ensuring identical problem instances.

    from hypre.bindings import solve, SolveResult, TUNABLE_PARAMS

    result = solve(params={"strong_threshold": 0.25})
    print(result.work_units)
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
import sys
from typing import Any, Dict, Optional

# ---- load compiled C library ------------------------------------------------

_SOLVER_DIR = Path(__file__).parent
_REPO_ROOT = _SOLVER_DIR.parents[1]
_LIB_SUFFIX = ".dylib" if sys.platform == "darwin" else ".so"
_LIB_PATH = _REPO_ROOT / "hypre" / "interfaces" / f"libamg_runtime{_LIB_SUFFIX}"
AMG_RUNTIME_LIBRARY = _LIB_PATH
_BUILD_SCRIPT = _SOLVER_DIR / "build_libamg_runtime.sh"
_VP = ctypes.c_void_p
_I = ctypes.c_int
_D = ctypes.c_double
_ULL = ctypes.c_ulonglong

_lib = None


def _load_native_library() -> None:
    """Load the native runtime on the first solver use, not on import."""
    global _lib
    if _lib is not None:
        return
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"Compiled library not found at {_LIB_PATH}.\n"
            f"Build it from this directory with:\n"
            f"  bash {_BUILD_SCRIPT}\n"
            "or run:\n"
            f"  make -C {_REPO_ROOT / 'hypre'}"
        )
    try:
        library = ctypes.CDLL(str(_LIB_PATH))
    except OSError as exc:
        inspect_command = "otool -L" if sys.platform == "darwin" else "ldd"
        raise RuntimeError(
            f"Failed to load {_LIB_PATH}.\n"
            "Rebuild the AMG runtime with the portable wrapper:\n"
            f"  bash {_BUILD_SCRIPT}\n"
            "Then inspect the linked HYPRE path with:\n"
            f"  {inspect_command} {_LIB_PATH}\n"
            f"Original loader error: {exc}"
        ) from exc

    library.amg_runtime_create.restype = _VP
    library.amg_runtime_create.argtypes = [
        _I, _I, _I,       # nx, ny, nz
        _I,               # stencil_type (0=difconv, 7 or 27=laplacian)
        _I, _ULL,         # rhs_type, rhs_seed
        _D, _D,           # k, c         (7pt coeffs)
        _D, _D, _D, _D,   # a0, a1, a2, a3  (27pt coeffs)
    ]

    library.amg_runtime_solve.restype = _I
    library.amg_runtime_solve.argtypes = [
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
        ctypes.POINTER(_I),               # out_status
    ]

    library.amg_runtime_destroy.restype = None
    library.amg_runtime_destroy.argtypes = [_VP]

    library.amg_runtime_get_n.restype = _I
    library.amg_runtime_get_n.argtypes = [_VP]

    library.amg_runtime_get_nnz.restype = _I
    library.amg_runtime_get_nnz.argtypes = [_VP]

    library.amg_runtime_prepare.restype = _I
    library.amg_runtime_prepare.argtypes = [
        _VP,
        _D, _I, _I, _D, _I, _I, _I, _I,
        _D, _I, _I, _I, _D, _I, _D, _I, _I,
        ctypes.POINTER(_D),
        ctypes.POINTER(_D),
    ]

    library.amg_runtime_step.restype = _I
    library.amg_runtime_step.argtypes = [
        _VP,
        _D, _I, _I, _I, _I, _I, _I, _I, _I, _I, _D, _D, _D, _I, _D, _I,
        _D, _I,
        ctypes.POINTER(_D),
        ctypes.POINTER(_D),
        ctypes.POINTER(_I),
    ]

    library.amg_runtime_get_r0.restype = _D
    library.amg_runtime_get_r0.argtypes = [_VP]

    library.amg_runtime_get_r.restype = _D
    library.amg_runtime_get_r.argtypes = [_VP]

    library.amg_runtime_get_cycle.restype = _I
    library.amg_runtime_get_cycle.argtypes = [_VP]

    library.amg_runtime_get_setup_time.restype = _D
    library.amg_runtime_get_setup_time.argtypes = [_VP]

    library.amg_runtime_get_cycle_type.restype = _I
    library.amg_runtime_get_cycle_type.argtypes = [_VP]

    library.amg_runtime_get_relax_type.restype = _I
    library.amg_runtime_get_relax_type.argtypes = [_VP]
    library.amg_runtime_get_cycle_relax_type.restype = _I
    library.amg_runtime_get_cycle_relax_type.argtypes = [_VP, _I]

    library.amg_runtime_get_relax_weight.restype = _I
    library.amg_runtime_get_relax_weight.argtypes = [_VP, _I, ctypes.POINTER(_D)]

    _lib = library


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


def _encode_params(params: Optional[Dict[str, Any]]) -> tuple[Dict[str, Any], list[Any]]:
    normalized = dict(params or {})
    unknown = set(normalized) - set(TUNABLE_PARAMS)
    if unknown:
        raise ValueError(f"Unknown param(s): {unknown}")
    c_args = [
        normalized[name]
        if name in normalized
        else (-1.0 if TUNABLE_PARAMS[name] is float else -1)
        for name in _PARAM_ORDER
    ]
    return normalized, c_args

# ---- result -----------------------------------------------------------------


class NativeCode(IntEnum):
    OK = 0
    INVALID_ARGUMENT = -1
    CREATE_ERROR = -2
    SETUP_ERROR = -3
    SOLVE_ERROR = -4
    NONFINITE_RESULT = -5


class SolveStatus(IntEnum):
    CONTINUE = 0
    CONVERGED = 1
    MAX_CYCLES = 2


class AMGNativeError(RuntimeError):
    """Native runtime failure with the work completed before the error."""

    def __init__(
        self,
        *,
        operation: str,
        code: int,
        setup_runtime_sec: float = 0.0,
        solve_runtime_sec: float = 0.0,
    ) -> None:
        try:
            self.code = NativeCode(int(code))
        except ValueError:
            self.code = int(code)
        self.operation = str(operation)
        self.setup_runtime_sec = max(0.0, float(setup_runtime_sec))
        self.solve_runtime_sec = max(0.0, float(solve_runtime_sec))
        super().__init__(f"{self.operation} failed with native code {int(code)}")

    @property
    def runtime_sec(self) -> float:
        return float(self.setup_runtime_sec + self.solve_runtime_sec)


@dataclass
class SolveResult:
    iterations: int
    complexity: float
    runtime_sec: float
    setup_runtime_sec: float
    solve_runtime_sec: float
    residual_norm: float
    status: SolveStatus
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def work_units(self) -> float:
        return float(self.iterations) * self.complexity


@dataclass
class PrepareResult:
    # Prepared setup/configuration plus the initial residual computation.
    setup_runtime_sec: float
    # Prepared and full-solve paths now expose the same residual convention:
    # ||r|| / ||r0||, so the initial relative residual is one.
    initial_residual_norm: float
    initial_relax_weight: float
    # Keep the raw norm available for diagnostics without leaking it into
    # convergence checks or controller features.
    absolute_initial_residual_norm: float = float("nan")


@dataclass(frozen=True)
class StepResult:
    # Relative residual norm, matching HYPRE_BoomerAMGSolve.
    residual_norm: float
    runtime_sec: float
    status: SolveStatus
    # Raw ||b - Ax|| retained only for diagnostics.
    absolute_residual_norm: float = float("nan")


def _relative_residual_norm(
    absolute_residual_norm: float,
    absolute_initial_residual_norm: float,
) -> float:
    residual = float(absolute_residual_norm)
    initial = float(absolute_initial_residual_norm)
    if initial > 0.0:
        return residual / initial
    if initial == 0.0 and residual == 0.0:
        return 0.0
    return float("inf")


class PreparedAMGEnv:
    def __init__(self, env_ptr: int):
        self._env = env_ptr
        self._absolute_initial_residual_norm = float("nan")
        self.last_step = StepResult(
            float("nan"),
            0.0,
            SolveStatus.CONTINUE,
            float("nan"),
        )

    def prepare_rl(self, params: Optional[Dict[str, Any]] = None) -> PrepareResult:
        """Prepare AMG and its initial residual, charging both to setup time."""
        _, c_args = _encode_params(params)

        out_setup = _D()
        out_r0 = _D()
        rc = _lib.amg_runtime_prepare(
            self._env,
            *c_args,
            ctypes.byref(out_setup),
            ctypes.byref(out_r0),
        )
        if rc != 0:
            raise AMGNativeError(
                operation="setup",
                code=rc,
                setup_runtime_sec=float(out_setup.value),
            )
        out_relax_weight = _D()
        rc = _lib.amg_runtime_get_relax_weight(
            self._env,
            0,
            ctypes.byref(out_relax_weight),
        )
        if rc != 0:
            raise AMGNativeError(
                operation="get_relax_weight",
                code=rc,
                setup_runtime_sec=float(out_setup.value),
            )
        absolute_initial_residual = float(out_r0.value)
        self._absolute_initial_residual_norm = absolute_initial_residual
        return PrepareResult(
            setup_runtime_sec=float(out_setup.value),
            initial_residual_norm=_relative_residual_norm(
                absolute_initial_residual,
                absolute_initial_residual,
            ),
            initial_relax_weight=float(out_relax_weight.value),
            absolute_initial_residual_norm=absolute_initial_residual,
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
        tol: float = 0.0,
        max_cycles: int = 0,
    ) -> tuple[float, float]:
        """Run one AMG cycle and return ``(relative_residual, runtime_sec)``.

        ``tol`` has the same relative-residual meaning as the tolerance passed
        to :func:`solve`, so native default and controlled solve paths stop at
        the same accuracy target. With positive tolerance, success requires
        a residual strictly below ``tol`` before ``max_cycles`` is reached.
        At the limit the status is ``MAX_CYCLES``, even if that final cycle
        reaches the target, matching the linked BoomerAMG full-solve status.

        ``runtime_sec`` covers native parameter updates, the AMG cycle,
        external residual monitoring and the stopping check. The initial
        residual belongs to :meth:`prepare_rl` and is charged there once.
        On native failure, the exception retains the completed step time.
        """

        out_r = _D()
        out_rt = _D()
        out_status = _I()
        rc = _lib.amg_runtime_step(
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
            float(tol),
            int(max_cycles),
            ctypes.byref(out_r),
            ctypes.byref(out_rt),
            ctypes.byref(out_status),
        )
        if rc != 0:
            raise AMGNativeError(
                operation="solve_step",
                code=rc,
                solve_runtime_sec=float(out_rt.value),
            )
        absolute_residual = float(out_r.value)
        relative_residual = _relative_residual_norm(
            absolute_residual,
            self._absolute_initial_residual_norm,
        )
        self.last_step = StepResult(
            residual_norm=relative_residual,
            runtime_sec=float(out_rt.value),
            status=SolveStatus(int(out_status.value)),
            absolute_residual_norm=absolute_residual,
        )
        return relative_residual, float(out_rt.value)

    def solve(
        self,
        params: Optional[Dict[str, Any]] = None,
        *,
        tol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> SolveResult:
        normalized, c_args = _encode_params(params)
        out_iters = _I()
        out_comp = _D()
        out_res = _D()
        out_rt = _D()
        out_setup_rt = _D()
        out_solve_rt = _D()
        out_status = _I()
        rc = _lib.amg_runtime_solve(
            self._env,
            *c_args,
            (-1.0 if tol is None else float(tol)),
            (-1 if max_iter is None else int(max_iter)),
            ctypes.byref(out_iters),
            ctypes.byref(out_comp),
            ctypes.byref(out_res),
            ctypes.byref(out_rt),
            ctypes.byref(out_setup_rt),
            ctypes.byref(out_solve_rt),
            ctypes.byref(out_status),
        )
        if rc != 0:
            raise AMGNativeError(
                operation="setup" if rc == NativeCode.SETUP_ERROR else "solve",
                code=rc,
                setup_runtime_sec=float(out_setup_rt.value),
                solve_runtime_sec=float(out_solve_rt.value),
            )
        return SolveResult(
            iterations=int(out_iters.value),
            complexity=float(out_comp.value),
            runtime_sec=float(out_rt.value),
            setup_runtime_sec=float(out_setup_rt.value),
            solve_runtime_sec=float(out_solve_rt.value),
            residual_norm=float(out_res.value),
            status=SolveStatus(int(out_status.value)),
            params=normalized,
        )

    @property
    def r0(self) -> float:
        absolute = self.absolute_r0
        return _relative_residual_norm(absolute, absolute)

    @property
    def absolute_r0(self) -> float:
        return float(_lib.amg_runtime_get_r0(self._env))

    @property
    def r(self) -> float:
        return _relative_residual_norm(
            self.absolute_r,
            self._absolute_initial_residual_norm,
        )

    @property
    def absolute_r(self) -> float:
        return float(_lib.amg_runtime_get_r(self._env))

    @property
    def cycle(self) -> int:
        return int(_lib.amg_runtime_get_cycle(self._env))

    @property
    def setup_runtime_sec(self) -> float:
        return float(_lib.amg_runtime_get_setup_time(self._env))

    @property
    def cycle_type(self) -> int:
        return int(_lib.amg_runtime_get_cycle_type(self._env))

    @property
    def relax_type(self) -> int:
        return int(_lib.amg_runtime_get_relax_type(self._env))

    @property
    def cycle_relax_types(self) -> tuple[int, int, int]:
        """Effective down/up/coarse smoothers, including native defaults."""
        return tuple(
            int(_lib.amg_runtime_get_cycle_relax_type(self._env, stage))
            for stage in (1, 2, 3)
        )

    @property
    def n(self) -> int:
        return int(_lib.amg_runtime_get_n(self._env))

    @property
    def nnz(self) -> int:
        return int(_lib.amg_runtime_get_nnz(self._env))

    def close(self) -> None:
        if self._env:
            _lib.amg_runtime_destroy(self._env)
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
    with create_env(
        nx=nx,
        ny=ny,
        nz=nz,
        stencil=stencil,
        rhs_type=rhs_type,
        rhs_seed=rhs_seed,
        k=k,
        c=c,
        a0=a0,
        a1=a1,
        a2=a2,
        a3=a3,
    ) as env:
        return env.solve(
            params=params,
            tol=tol,
            max_iter=max_iter,
        )


def create_env(
    *,
    nx: int = 10, ny: int = 10, nz: int = 10,
    stencil: int = 7,
    rhs_type: int = 0,
    rhs_seed: int = 42,
    k: float = 1.0, c: float = 0.0,
    a0: float = 1.0, a1: float = 1.0, a2: float = 1.0, a3: float = 0.0,
) -> PreparedAMGEnv:
    _load_native_library()
    env = _lib.amg_runtime_create(
        nx, ny, nz, stencil, rhs_type, rhs_seed,
        k, c, a0, a1, a2, a3,
    )
    if not env:
        raise AMGNativeError(operation="create", code=NativeCode.CREATE_ERROR)
    return PreparedAMGEnv(env)
