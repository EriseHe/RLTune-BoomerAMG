"""Generate the two accepted PDE streams without constructing native matrices."""

from __future__ import annotations
from typing import Any, Dict, Sequence, Tuple
import numpy as np
from problems.registry import (
    SCALAR_ANISOTROPIC_DIFFUSION,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
    normalize_problem_kind,
)
from problems.streams import (
    generate_scalar_anisotropic_diffusion_instances,
    generate_scalar_anisotropic_diffusion_advection_instances,
)


def generate_problem_instances(
    *,
    T: int,
    seed: int,
    grid_choices: Sequence[Tuple[int, int, int]],
    c_min: float,
    c_max: float,
    difconv_a: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    problem: str | None = None,
    advection_min: float | None = None,
    advection_max: float | None = None,
) -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    problem_kind = normalize_problem_kind(problem)
    if problem_kind == SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION:
        return generate_scalar_anisotropic_diffusion_advection_instances(
            count=int(T),
            seed=int(seed),
            grid_choices=grid_choices,
            c_min=float(c_min),
            c_max=float(c_max),
            advection_min=float(c_min if advection_min is None else advection_min),
            advection_max=float(c_max if advection_max is None else advection_max),
        )
    if problem_kind == SCALAR_ANISOTROPIC_DIFFUSION:
        if any((float(value) != 0.0 for value in difconv_a)):
            raise ValueError("scalar_anisotropic_diffusion requires zero advection")
        return generate_scalar_anisotropic_diffusion_instances(
            count=int(T),
            seed=int(seed),
            grid_choices=grid_choices,
            c_min=float(c_min),
            c_max=float(c_max),
        )
    raise ValueError(f"Unsupported official PDE family: {problem!r}")
