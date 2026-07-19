from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .amg import stencil_0_difconv_rl


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
        matrix_kwargs, context, _metadata = stencil_0_difconv_rl(
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
            ax=float(advection[0]),
            ay=float(advection[1]),
            az=float(advection[2]),
        )
        instances.append((matrix_kwargs, np.asarray(context, dtype=float)))
    return instances
