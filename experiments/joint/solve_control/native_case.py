"""Execute and account for a controller episode with default native recovery."""

from __future__ import annotations

import argparse
from typing import Any, Dict, Mapping

from hypre.bindings import augment_setup_params
from setup.space import DEFAULT_SETUP_PARAMS
from solve.controllers.common import ControllerBundle, OnlineSolveCase

from .feedback import as_feedback
from .joint_online_common import report_online_outcome
from .native_evaluation import solve_no_rl_case


def run_case(
    *,
    bundle: ControllerBundle,
    setup_row: Mapping[str, Any],
    args: argparse.Namespace,
    learn: bool,
    explore: bool,
) -> Dict[str, Any]:
    mkw = dict(setup_row["mkw"])
    fallback_result: Dict[str, Any] = {}

    def fallback_attempt():
        result = solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=mkw,
            solver_tol=float(args.tol),
            solver_max_iter=int(args.max_cycles),
            augment_params=augment_setup_params,
        )
        fallback_result.update(result)
        return result

    native = bundle.run_case(
        OnlineSolveCase(
            mkw=mkw,
            params=dict(setup_row["params"]),
            solve_tol=float(args.tol),
            solve_max_cycles=int(args.max_cycles),
            learn=bool(learn),
            explore=bool(explore),
            record_action_metadata=True,
            fallback_attempt=fallback_attempt,
        )
    )
    outcome = report_online_outcome(
        as_feedback(native, include_controller=True),
        bandit_timing={},
    )
    outcome["completed_residual_norm"] = (
        fallback_result.get("residual_norm", float("nan"))
        if outcome.get("fallback_used", False)
        else outcome["residual_norm"]
    )
    return outcome


__all__ = ["run_case"]
