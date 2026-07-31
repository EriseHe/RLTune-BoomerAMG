from __future__ import annotations

from typing import Any, Callable, Dict, Sequence, Tuple

import numpy as np

from .amg import stencil_0_difconv_rl
from .scalar_anisotropic_diffusion import (
    stencil_0_scalar_anisotropic_diffusion_rl,
)
from .scalar_anisotropic_diffusion_advection import (
    stencil_0_scalar_anisotropic_diffusion_advection_rl,
)


ProblemSampler = Callable[
    ...,
    Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]],
]


def _generate_stencil_0_instances(
    *,
    count: int,
    seed: int,
    grid_choices: Sequence[Tuple[int, int, int]],
    c_min: float,
    c_max: float,
    sampler: ProblemSampler,
    sampler_kwargs: Dict[str, Any],
) -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    grids = [tuple(int(value) for value in grid) for grid in grid_choices]
    if not grids:
        raise ValueError("grid_choices must be non-empty")
    if int(count) < 0:
        raise ValueError("count must be non-negative")

    child_seeds = np.random.SeedSequence(int(seed)).spawn(int(count))
    instances = []
    for index, child_seed in enumerate(child_seeds):
        grid = grids[index % len(grids)]
        rng = np.random.default_rng(child_seed)
        matrix_kwargs, context, _metadata = sampler(
            rng=rng,
            t=index,
            trial=0,
            nx=int(grid[0]),
            ny=int(grid[1]),
            nz=int(grid[2]),
            n_min=int(max(grid)),
            n_max=int(max(grid)),
            c_min=float(c_min),
            c_max=float(c_max),
            **sampler_kwargs,
        )
        instances.append(
            (matrix_kwargs, np.asarray(context, dtype=float))
        )
    return instances


def generate_difconv_instances(
    *,
    count: int,
    seed: int,
    grid_choices: Sequence[Tuple[int, int, int]],
    c_min: float,
    c_max: float,
    advection: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    """Generate the deterministic DifConv stream shared by all phases."""
    return _generate_stencil_0_instances(
        count=count,
        seed=seed,
        grid_choices=grid_choices,
        c_min=c_min,
        c_max=c_max,
        sampler=stencil_0_difconv_rl,
        sampler_kwargs={
            "ax": float(advection[0]),
            "ay": float(advection[1]),
            "az": float(advection[2]),
        },
    )


def generate_scalar_anisotropic_diffusion_instances(
    *,
    count: int,
    seed: int,
    grid_choices: Sequence[Tuple[int, int, int]],
    c_min: float,
    c_max: float,
) -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    """Generate scalar diffusion with canonical zero-advection contexts."""

    return _generate_stencil_0_instances(
        count=count,
        seed=seed,
        grid_choices=grid_choices,
        c_min=c_min,
        c_max=c_max,
        sampler=stencil_0_scalar_anisotropic_diffusion_rl,
        sampler_kwargs={},
    )


def generate_scalar_anisotropic_diffusion_advection_instances(
    *,
    count: int,
    seed: int,
    grid_choices: Sequence[Tuple[int, int, int]],
    c_min: float,
    c_max: float,
    advection_min: float,
    advection_max: float,
) -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    """Generate a deterministic stream with per-case diffusion and advection."""
    return _generate_stencil_0_instances(
        count=count,
        seed=seed,
        grid_choices=grid_choices,
        c_min=c_min,
        c_max=c_max,
        sampler=stencil_0_scalar_anisotropic_diffusion_advection_rl,
        sampler_kwargs={
            "a_min": float(advection_min),
            "a_max": float(advection_max),
        },
    )
