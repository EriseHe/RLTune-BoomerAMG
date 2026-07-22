"""
BoomerAMG Python binding shared by setup and solve workflows.

Thin Python binding to the single libamg_runtime library (C calling HYPRE).
Matrix construction, full solves, and per-cycle solves share one native env.
Supports multi-parameter tuning. Loss is computed by the caller.

    from hypre.bindings import solve, SolveResult, TUNABLE_PARAMS

    result = solve(params={"strong_threshold": 0.25})
    print(result.work_units)
"""

from .boomeramg import (
    AMG_RUNTIME_LIBRARY,
    AMGNativeError,
    NativeCode,
    PrepareResult,
    PreparedAMGEnv,
    SolveStatus,
    SolveResult,
    StepResult,
    TUNABLE_PARAMS,
    create_env,
    solve,
)
from .config import augment_setup_params
from .recovery import (
    AttemptOutcome,
    AttemptStatus,
    RecoveryOutcome,
    execute_attempt,
    run_with_default_fallback,
)

__all__ = [
    "AMG_RUNTIME_LIBRARY",
    "create_env",
    "solve",
    "AMGNativeError",
    "NativeCode",
    "SolveStatus",
    "SolveResult",
    "PrepareResult",
    "StepResult",
    "PreparedAMGEnv",
    "TUNABLE_PARAMS",
    "augment_setup_params",
    "AttemptOutcome",
    "AttemptStatus",
    "RecoveryOutcome",
    "execute_attempt",
    "run_with_default_fallback",
]
