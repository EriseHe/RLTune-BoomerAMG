"""Shared solve-controller configuration and linear-algebra primitives."""

from .config import EpsilonScheduleSpec, SharedActionSpec, SolveStateSpec
from .linear_lcb import (
    _RollingFloatWindow,
    _SharedActionLcbController,
    _blas_dger,
    _greedy_cost_index,
    _json_dataclass,
    _mad_scale,
    _rank_one_accumulate,
    _rank_one_inverse_accumulate,
    _rank_one_inverse_update,
    _sandwich_quadratic,
)
from .types import ControllerBundle

__all__ = [
    "ControllerBundle",
    "EpsilonScheduleSpec",
    "SharedActionSpec",
    "SolveStateSpec",
]
