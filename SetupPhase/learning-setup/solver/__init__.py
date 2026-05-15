"""
BoomerAMG solver interface for bandit experiments.

Thin Python binding to the platform-specific `libamg_setup_solver` shared
library (C calling HYPRE). Matrix construction happens in C using the same
Laplacian builders as the solve-phase RL environment. Supports multi-parameter
tuning. Loss is computed by the caller.

    from solver import solve, SolveResult, TUNABLE_PARAMS

    result = solve(params={"strong_threshold": 0.25})
    print(result.work_units)
"""

from .boomeramg import solve, SolveResult, TUNABLE_PARAMS

__all__ = ["solve", "SolveResult", "TUNABLE_PARAMS"]
