"""Solve outcome classification shared by controllers and evaluations."""

import numpy as np


def classify_rl_failure(
    *,
    residual_norm: float,
    iterations: int,
    solve_tol: float,
    solve_max_cycles: int,
) -> str:
    """Match BoomerAMG's positive-tolerance convergence status.

    Reaching the iteration limit is nonconvergence even when that cycle
    reaches the residual target. Before the limit, convergence is strict.
    """

    if not np.isfinite(float(residual_norm)):
        return "non_finite_residual_norm"
    below_tolerance = float(residual_norm) < float(solve_tol)
    if int(iterations) >= int(solve_max_cycles):
        prefix = "" if below_tolerance else "residual_above_solve_tol;"
        return prefix + "max_cycles_reached_without_convergence"
    if below_tolerance:
        return ""
    return "residual_above_solve_tol"
