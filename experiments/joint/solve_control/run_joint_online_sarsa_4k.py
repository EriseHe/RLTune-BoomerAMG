"""Backward-compatible entry point for the historical 4K runner name.

The experiment is no longer SARSA-specific.  New code should import
``joint_4k_runner``; this module only preserves established imports and the
old command-line path.
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, Sequence

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import experiments.joint.solve_control.joint_4k_runner as _runner


# CLI and canonical execution.
build_parser = _runner.build_parser
run = _runner.run
main = _runner.main

# Composable experiment compatibility.
ComposableMethodSpec = _runner.ComposableMethodSpec
_parse_composable_method = _runner._parse_composable_method
_validate_composable_protocol = _runner._validate_composable_protocol

# Historical study constants.
BATCHED_LSVI_METHOD = _runner.BATCHED_LSVI_METHOD
BEHAVIOR_MODES = _runner.BEHAVIOR_MODES
DEFAULT_SETUP_METHOD = _runner.DEFAULT_SETUP_METHOD
LSVI_METHOD = _runner.LSVI_METHOD
LSVI_METHODS = _runner.LSVI_METHODS
RECALIBRATED_LSVI_METHOD = _runner.RECALIBRATED_LSVI_METHOD
RECURSIVE_LCB_METHODS = _runner.RECURSIVE_LCB_METHODS
RECURSIVE_LCB_PPO_METHODS = _runner.RECURSIVE_LCB_PPO_METHODS
RECURSIVE_LSTDQ_METHOD = _runner.RECURSIVE_LSTDQ_METHOD
RECURSIVE_LSTDQ_METHODS = _runner.RECURSIVE_LSTDQ_METHODS
RECURSIVE_LSTDQ_V2_METHOD = _runner.RECURSIVE_LSTDQ_V2_METHOD
RECURSIVE_MC_METHOD = _runner.RECURSIVE_MC_METHOD
REFERENCE_METHODS = _runner.REFERENCE_METHODS
SHARED_ACTION_PROFILES = _runner.SHARED_ACTION_PROFILES
SOLVE_CONTROLLER_SCREEN_METHODS = _runner.SOLVE_CONTROLLER_SCREEN_METHODS
SOLVE_CONTROLLER_SEED_OFFSETS = _runner.SOLVE_CONTROLLER_SEED_OFFSETS
STRUCTURED_MODEL_BASED_METHOD = _runner.STRUCTURED_MODEL_BASED_METHOD
SETUP_BANDIT_KINDS = _runner.SETUP_BANDIT_KINDS
SarsaCandidate = _runner.SarsaCandidate
candidate_grid = _runner.candidate_grid

# Controller-construction compatibility.  Active callers use
# joint_controller_build directly; these aliases keep archived diagnostics and
# downstream imports working.
_make_controller = _runner._make_controller
_make_encoder = _runner._make_encoder
_make_lsvi_controller = _runner._make_lsvi_controller
_make_online_controller_from_args = _runner._make_online_controller_from_args
_make_recalibrated_lsvi_controller = _runner._make_recalibrated_lsvi_controller
_make_recursive_blstdq_controller = _runner._make_recursive_blstdq_controller
_make_recursive_lstdq_controller = _runner._make_recursive_lstdq_controller
_make_recursive_lstdq_v2_controller = _runner._make_recursive_lstdq_v2_controller
_make_recursive_lstdq_v3_controller = _runner._make_recursive_lstdq_v3_controller
_make_recursive_mc_controller = _runner._make_recursive_mc_controller
_make_setup_obs_encoder = _runner._make_setup_obs_encoder
_make_shared_action_config = _runner._make_shared_action_config
_make_shared_action_spec = _runner._make_shared_action_spec
_make_solve_state_spec = _runner._make_solve_state_spec
_make_structured_model_based_controller = (
    _runner._make_structured_model_based_controller
)
_validate_lsvi_protocol = _runner._validate_lsvi_protocol
_validate_shared_lcb_protocol = _runner._validate_shared_lcb_protocol

# Reporting compatibility.
_action_summary = _runner._action_summary
_comparison_windows = _runner._comparison_windows
_empty_stream_summary = _runner._empty_stream_summary
_window_result = _runner._window_result
_write_solve_screen_report = _runner._write_solve_screen_report
_write_summary_csv = _runner._write_summary_csv

# Keep this binding patchable for older tests and diagnostics.
_report_online_outcome = _runner._report_online_outcome


def _run_default_setup_method(
    *,
    spec: ComposableMethodSpec,
    solver_fn: Any,
    args: argparse.Namespace,
    mkw: Dict[str, Any],
    controller_methods: Sequence[str],
) -> Dict[str, Any]:
    """Delegate while honoring a patched facade reporter."""

    return _runner._run_default_setup_method(
        spec=spec,
        solver_fn=solver_fn,
        args=args,
        mkw=mkw,
        controller_methods=controller_methods,
        report_online_outcome=_report_online_outcome,
    )


if __name__ == "__main__":
    main()
