"""Reusable matrix, right-hand-side, grid, and evaluation case specifications."""

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CaseSpec:
    nx: int
    ny: int
    nz: int
    stencil: int
    rhs_type: int
    rhs_seed: int
    a0: float
    a1: float
    a2: float
    a3: float


def generate_matrix_coeffs(
    rng: np.random.Generator,
    stencil: int,
    randomize_A: bool,
    a0_base: float,
    a1_base: float,
    a2_base: float,
    a3_base: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Mirrors BoomerAMGRelaxEnv.reset for coefficient generation.

    Returns (a0, a1, a2, a3, k, c) where k/c are only used for 7-pt cases.
    """
    if randomize_A:
        if stencil == 27:
            s1 = float(rng.uniform(0.5, 2.0))
            s2 = float(rng.uniform(0.5, 2.0))
            s3 = float(rng.uniform(0.5, 2.0))
            a1 = a1_base * s1
            a2 = a2_base * s2
            a3 = a3_base * s3
            c_val = float(rng.uniform(0.0, 5.0))
            a0 = -(6.0 * a1 + 12.0 * a2 + 8.0 * a3) + c_val
            return a0, a1, a2, a3, 1.0, 0.0
        s = float(rng.uniform(0.5, 2.0))
        a1 = a1_base * s
        a0 = -(6.0 * a1)
        return a0, a1, a2_base, a3_base, 1.0, 0.0

    if stencil == 27:
        s1, s2, s3 = 1.4, 2.0, 2.5
        a1 = a1_base * s1
        a2 = a2_base * s2
        a3 = a3_base * s3
        a0 = -(6.0 * a1 + 12.0 * a2 + 8.0 * a3)
        return a0, a1, a2, a3, 1.0, 0.0

    a1 = a1_base
    a0 = -(6.0 * a1)
    return a0, a1, a2_base, a3_base, 1.0, 0.0


def generate_rhs_seed(
    rng: np.random.Generator, randomize_b: bool, fixed_rhs_seed: int
) -> int:
    """Mirrors rhs_seed selection in BoomerAMGRelaxEnv.reset."""
    if randomize_b:
        return int(rng.integers(0, 2**63 - 1))
    return int(fixed_rhs_seed)


def sample_grid_choice(
    rng: np.random.Generator, grid_choices: Sequence[Tuple[int, int, int]]
) -> Tuple[int, int, int]:
    """Mirrors grid choice in BoomerAMGRelaxEnv.reset."""
    idx = int(rng.integers(0, len(grid_choices)))
    return tuple(int(x) for x in grid_choices[idx])


def make_eval_cases(
    seed_start: int,
    seed_count: int,
    grid_sizes: Sequence[Tuple[int, int, int]],
    randomize_A: bool,
    randomize_b: bool,
    fixed_rhs_seed: int,
    fixed_rhs_type: int,
    fixed_a_mode: bool = False,
    grid_mode: str = "zip",
) -> List[dict]:
    """
    Mirrors eval case construction from eval_and_plot/eval_default_vs_rl.
    """
    eval_seeds = list(range(int(seed_start), int(seed_start) + int(seed_count)))
    grid_sizes = list(grid_sizes)
    if fixed_a_mode:
        randomize_A = False
        randomize_b = False
        fixed_rhs_type = 1
        eval_seeds = [int(seed_start)]
        if not grid_sizes:
            grid_sizes = [(60, 60, 60)]

    if grid_mode == "full":
        cases = [
            {
                "seed": s,
                "grid": g,
                "randomize_A": randomize_A,
                "randomize_b": randomize_b,
                "fixed_rhs_seed": fixed_rhs_seed,
                "fixed_rhs_type": fixed_rhs_type,
            }
            for s in eval_seeds
            for g in grid_sizes
        ]
    else:
        cases = [
            {
                "seed": s,
                "grid": grid_sizes[i % len(grid_sizes)],
                "randomize_A": randomize_A,
                "randomize_b": randomize_b,
                "fixed_rhs_seed": fixed_rhs_seed,
                "fixed_rhs_type": fixed_rhs_type,
            }
            for i, s in enumerate(eval_seeds)
        ]
    return cases
