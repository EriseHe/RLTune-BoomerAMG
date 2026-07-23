"""Compatibility re-export of the public :mod:`solve.registry` API."""

from solve.registry import (
    AlgorithmSpec,
    COMPOSABLE_SOLVE_KINDS,
    ONLINE_SOLVE_KINDS,
    SOLVE_KIND_REGISTRY,
    OnlineControllerBuildSpec,
    OnlineSolveKind,
    SolveKindRegistration,
    build_online_solve_controller,
    make_online_controller_spec,
    solve_kind_registration,
)

__all__ = [
    "AlgorithmSpec",
    "COMPOSABLE_SOLVE_KINDS",
    "ONLINE_SOLVE_KINDS",
    "SOLVE_KIND_REGISTRY",
    "OnlineControllerBuildSpec",
    "OnlineSolveKind",
    "SolveKindRegistration",
    "build_online_solve_controller",
    "make_online_controller_spec",
    "solve_kind_registration",
]
