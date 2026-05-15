"""
BoomerAMG solver interface for bandit experiments.

Thin Python binding to libamg_setup_solver.dylib (C calling HYPRE).
Matrix built in C using the same Laplacian builders as the solve-phase RL env.
Supports multi-parameter tuning. Loss is computed by the caller.

    from solver import solve, SolveResult, TUNABLE_PARAMS

    result = solve(params={"strong_threshold": 0.25})
    print(result.work_units)
"""

from .boomeramg import create_env, solve, SolveResult, PrepareResult, PreparedAMGEnv, TUNABLE_PARAMS

__all__ = ["create_env", "solve", "SolveResult", "PrepareResult", "PreparedAMGEnv", "TUNABLE_PARAMS"]
