"""Solve outcome classification shared by controllers and evaluations."""

import numpy as np


def classify_rl_failure(
    *,
    residual_norm: float,
    iterations: int,
    solve_tol: float,
    solve_max_cycles: int,
) -> str:
    if not np.isfinite(float(residual_norm)):
        return "non_finite_residual_norm"
    if float(residual_norm) <= float(solve_tol):
        return ""
    if int(iterations) >= int(solve_max_cycles):
        return "residual_above_solve_tol;max_cycles_reached_without_convergence"
    return "residual_above_solve_tol"
