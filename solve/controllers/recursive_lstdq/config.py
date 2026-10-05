from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RecursiveLstdqSpec:
    """Recursive LSTDQ with episode-cluster sandwich uncertainty."""

    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    inverse_denominator_floor: float = 1.0e-10
    lcb_lower_bound_sec: float | None = 0.0


__all__ = ["RecursiveLstdqSpec"]
