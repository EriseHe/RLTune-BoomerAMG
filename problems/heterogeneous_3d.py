"""Exploratory 3D HYPRE families and contexts from assembly inputs only.

The material family parameterizes the three regions of GenerateVarDifConv.
The two RS families call GenerateRSVarDifConv unchanged. Grid, spatial layout,
boundary conditions and flow type are fixed within each experimental stream.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence
import numpy as np

STRESS_PROBLEMS = {
    "material_jump_3d": (100, "material_coefficients", 1e-2, 1e3),
    "rs_convection_3d_type2": (102, "rs_diffusion", 1e-4, 1e-1),
    "rs_convection_3d_type3": (103, "rs_diffusion", 1e-4, 1e-1),
}
STRESS_CONTEXT_FIELDS = {
    "material_coefficients": ("bias", "log_corner", "log_background", "log_center"),
    "rs_diffusion": ("bias", "log_diffusion_strength"),
}


def context_from_matrix_kwargs(matrix: Mapping[str, Any]) -> np.ndarray:
    stencil = int(matrix["stencil"])
    if stencil == 100:
        values = np.asarray([matrix["k"], matrix["c"], matrix["a0"]], dtype=float)
        lower, upper = -2.0, 3.0
    elif stencil in (102, 103):
        values = np.asarray([matrix["k"]], dtype=float)
        lower, upper = -4.0, -1.0
    else:
        raise ValueError("Expected a 3D heterogeneous/variable-flow stencil")
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("Assembly coefficients must be positive and finite")
    normalized = 2.0 * (np.log10(values) - lower) / (upper - lower) - 1.0
    return np.r_[1.0, normalized]


def validate_context(values: Sequence[float], mode: str) -> np.ndarray:
    context = np.asarray(values, dtype=float).reshape(-1)
    if context.size != len(STRESS_CONTEXT_FIELDS[mode]):
        raise ValueError(f"Wrong context dimension for {mode}")
    if not np.all(np.isfinite(context)) or context[0] != 1.0:
        raise ValueError("Context must be finite and start with an intercept")
    return context


def generate_instances(*, problem: str, count: int, seed: int,
                       grid_choices: Sequence[tuple[int, int, int]],
                       c_min: float, c_max: float):
    stencil, _mode, expected_min, expected_max = STRESS_PROBLEMS[problem]
    if (c_min, c_max) != (expected_min, expected_max):
        raise ValueError("Use the declared coefficient range for this exploratory family")
    if len(set(map(tuple, grid_choices))) != 1:
        raise ValueError("These contexts require a fixed grid within each stream")
    nx, ny, nz = map(int, grid_choices[0])
    if min(nx, ny, nz) < 2:
        raise ValueError("All three dimensions must contain interior grid points")
    result = []
    for child in np.random.SeedSequence(int(seed)).spawn(int(count)):
        rng = np.random.default_rng(child)
        values = 10.0 ** rng.uniform(np.log10(c_min), np.log10(c_max),
                                     size=3 if stencil == 100 else 1)
        matrix = dict(nx=nx, ny=ny, nz=nz, stencil=stencil, rhs_type=1,
                      rhs_seed=int(rng.integers(1, 2**31 - 1)), k=float(values[0]),
                      c=float(values[1]) if stencil == 100 else 0.0,
                      a0=float(values[2]) if stencil == 100 else 0.0,
                      a1=0.0, a2=0.0, a3=0.0)
        result.append((matrix, context_from_matrix_kwargs(matrix)))
    return result
