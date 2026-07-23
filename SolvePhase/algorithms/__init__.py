"""Backward-compatible namespace for historical solve algorithm imports.

Canonical implementations live in ``solve.controllers`` and the public
construction registry lives in ``solve.registry``.
"""

from solve.registry import (
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
