"""Shared solve-controller encoding, configuration, and linear primitives."""

from .action_space import build_action_basis, joint_action_features
from .config import EpsilonScheduleSpec, SharedActionSpec, SolveStateSpec
from .factory import (
    OnlineControllerFactoryRequest,
    build_shared_action_controller_bundle,
)
from .linear_lcb import (
    _SharedActionLcbController,
    _blas_dger,
    _greedy_cost_index,
    _json_dataclass,
    _rank_one_accumulate,
    _rank_one_inverse_accumulate,
    _sandwich_quadratic,
)
from .state_encoder import (
    CANONICAL_PROBLEM_CONTEXT,
    LEGACY_DIFFUSION_ONLY_CONTEXT,
    PROBLEM_CONTEXT_MODES,
    SolveStateEncoder,
)
from .types import ControllerBundle, FallbackAttempt, OnlineSolveCase

__all__ = [
    "ControllerBundle",
    "CANONICAL_PROBLEM_CONTEXT",
    "EpsilonScheduleSpec",
    "FallbackAttempt",
    "LEGACY_DIFFUSION_ONLY_CONTEXT",
    "OnlineControllerFactoryRequest",
    "OnlineSolveCase",
    "PROBLEM_CONTEXT_MODES",
    "SharedActionSpec",
    "SolveStateEncoder",
    "SolveStateSpec",
    "build_action_basis",
    "build_shared_action_controller_bundle",
    "joint_action_features",
]
