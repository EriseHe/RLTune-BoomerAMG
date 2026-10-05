"""Shared scientific transformations for recorded trajectories and fixed-grid comparisons."""

from __future__ import annotations

import numpy as np


def reduction(reference, candidate):
    reference, candidate = (
        np.asarray(reference, dtype=float),
        np.asarray(candidate, dtype=float),
    )
    if np.any(reference <= 0):
        raise ValueError("Reduction denominator must be positive")
    return 100 * (1 - candidate / reference)


def case_order(cycle_cube):
    """One shared order; ties use original input index, never an RL outcome."""
    difficulty = np.mean(cycle_cube, axis=tuple(range(cycle_cube.ndim - 1)))
    return np.lexsort((np.arange(len(difficulty)), difficulty))


def padded_trace(values, width):
    """NaN means unexecuted. Never extend the last action past termination."""
    if len(values) > width:
        raise ValueError("Plot width would truncate observed cycles")
    out = np.full(width, np.nan)
    out[: len(values)] = values
    return out


def select_oracles(cells, seed, source, case_ids, weights):
    """Select from mean costs, with every test case equally weighted."""
    valid = [
        name
        for name in weights
        if all(cells[seed, source, case, name]["success"] for case in case_ids)
    ]
    if not valid:
        raise ValueError(
            "No all-success global fixed weight; cannot draw the current figure protocol"
        )
    best = min(
        valid,
        key=lambda name: (
            sum(cells[seed, source, case, name]["native"] for case in case_ids),
            name,
        ),
    )
    individual = {}
    for case in case_ids:
        choices = [
            name for name in weights if cells[seed, source, case, name]["success"]
        ]
        if not choices:
            raise ValueError(
                "Uncovered per-case oracle; do not silently omit this case"
            )
        individual[case] = min(
            choices, key=lambda name: (cells[seed, source, case, name]["native"], name)
        )
    return best, individual
