"""Compatibility namespace for the canonical lowercase :mod:`solve` package."""

from solve import (
    COMPOSABLE_SOLVE_KINDS,
    ONLINE_SOLVE_KINDS,
    OnlineControllerBuildSpec,
    build_online_solve_controller,
    make_online_controller_spec,
    solve_kind_registration,
)

__all__ = [
    "COMPOSABLE_SOLVE_KINDS",
    "ONLINE_SOLVE_KINDS",
    "OnlineControllerBuildSpec",
    "build_online_solve_controller",
    "make_online_controller_spec",
    "solve_kind_registration",
]
