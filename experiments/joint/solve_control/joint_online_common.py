"""Shared utilities for active joint-online AMG experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from typing import Any, Dict, Sequence

import numpy as np

from hypre.bindings.config import configure_smoother_profile
from experiments.joint.solve_control.online_td_experiment_common import _summarize
from problems.registry import (
    SCALAR_ANISOTROPIC_DIFFUSION,
    SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
    normalize_problem_kind,
)
from experiments.joint.solve_control.setup_aware_compare_common import (
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
    generate_difconv_instances,
)


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def _problem_stream_spec(
    args: argparse.Namespace,
) -> tuple[str, tuple[int, int, int], tuple[float, float, float]]:
    """Resolve the problem knobs while preserving the legacy stream defaults."""

    raw_problem = str(
        getattr(args, "problem", "scalar_anisotropic_diffusion")
        or "scalar_anisotropic_diffusion"
    )
    problem = normalize_problem_kind(raw_problem)

    raw_grid = getattr(args, "grid_shape", None)
    if raw_grid is None or (isinstance(raw_grid, str) and not raw_grid.strip()):
        grid_n = int(args.grid_n)
        grid = (grid_n, grid_n, grid_n)
    elif isinstance(raw_grid, str):
        grid = _parse_values(raw_grid, int)
    else:
        grid = tuple(int(value) for value in raw_grid)
    if len(grid) != 3 or any(int(value) <= 0 for value in grid):
        raise ValueError("grid_shape must contain exactly three positive integers")
    resolved_grid = tuple(int(value) for value in grid)

    raw_advection = getattr(args, "advection", (0.0, 0.0, 0.0))
    if raw_advection is None or (
        isinstance(raw_advection, str) and not raw_advection.strip()
    ):
        advection = (0.0, 0.0, 0.0)
    elif isinstance(raw_advection, str):
        advection = _parse_values(raw_advection, float)
    else:
        advection = tuple(float(value) for value in raw_advection)
    if len(advection) != 3 or not all(np.isfinite(advection)):
        raise ValueError("advection must contain exactly three finite floats")
    resolved_advection = tuple(float(value) for value in advection)
    if problem in {
        SCALAR_ANISOTROPIC_DIFFUSION,
        SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION,
    }:
        if any(value != 0.0 for value in resolved_advection):
            requested_kind = (
                "sampled advection"
                if problem == SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION
                else "zero advection"
            )
            raise ValueError(
                f"{problem} requires {requested_kind} through its own problem "
                "definition; do not also set a fixed advection vector"
            )
        resolved_advection = (0.0, 0.0, 0.0)

    return problem, resolved_grid, resolved_advection


def _sampled_advection_range(
    args: argparse.Namespace,
    *,
    problem: str,
) -> tuple[float, float] | None:
    if problem != SCALAR_ANISOTROPIC_DIFFUSION_ADVECTION:
        return None
    raw_min = getattr(args, "advection_min", None)
    raw_max = getattr(args, "advection_max", None)
    advection_min = float(args.c_min if raw_min is None else raw_min)
    advection_max = float(args.c_max if raw_max is None else raw_max)
    if (
        not np.isfinite(advection_min)
        or not np.isfinite(advection_max)
        or advection_max < advection_min
    ):
        raise ValueError(
            "advection_min/advection_max must be finite and satisfy "
            "advection_min <= advection_max"
        )
    return advection_min, advection_max


def _instance_stream_hash(
    stream: Sequence[tuple[Dict[str, Any], np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for mkw, context in stream:
        payload = {
            "mkw": dict(mkw),
            "context": np.asarray(context, dtype=float).tolist(),
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        )
        digest.update(b"\n")
    return digest.hexdigest()


def configure_paired_environment(args: argparse.Namespace) -> None:
    configure_smoother_profile(
        str(getattr(args, "smoother_profile", "legacy_l1_jacobi"))
    )
    setup_param_resolution = int(args.setup_param_resolution)
    _problem, grid, _advection = _problem_stream_spec(args)
    values = {
        # Legacy consumers accept one grid dimension only.  Keep them safe for
        # rectangular grids while the generated matrix kwargs retain all three.
        "MATRIX_GRID_N": max(grid),
        "SETUP_PARAM_RESOLUTION": setup_param_resolution,
        "SETUP_ACTION_SPACE": str(args.setup_action_space),
        "C_MIN": float(args.c_min),
        "C_MAX": float(args.c_max),
        "SOLVE_TOL": float(args.tol),
        "SOLVE_MAX_CYCLES": int(args.max_cycles),
        "ACTION_MODE": "discrete_w",
    }
    for name, value in values.items():
        os.environ[name] = str(value)
    if (
        str(args.setup_action_space) == "full_cartesian"
        and setup_param_resolution == EXP44_SETUP_PARAM_RESOLUTION
        and not bool(getattr(args, "setup_configuration_spaces", {}))
    ):
        os.environ["EXPECTED_SETUP_ACTION_COUNT"] = str(
            EXP44_TUNE7_CATEGORICAL_ACTION_COUNT
        )
    else:
        os.environ.pop("EXPECTED_SETUP_ACTION_COUNT", None)


def policy_last_arm(policy: Any) -> int:
    model = getattr(policy, "model", policy)
    last_arm = getattr(model, "_last_arm", None)
    if last_arm is not None:
        return int(last_arm)
    history = getattr(model, "history", None)
    if history:
        return int(getattr(history[-1], "arm_index", -1))
    return -1


def report_online_outcome(
    outcome: Dict[str, Any],
    *,
    bandit_timing: Dict[str, float],
) -> Dict[str, Any]:
    reported = dict(outcome)
    controller_runtime = float(reported.get("infer_runtime", 0.0))
    native_runtime = float(
        reported.get("native_runtime", float(reported["runtime"]) - controller_runtime)
    )
    native_solve_runtime = float(
        reported.get(
            "native_solve_runtime",
            float(reported.get("solve_runtime", 0.0)) - controller_runtime,
        )
    )
    bandit_overhead = float(bandit_timing.get("overhead_sec", 0.0))
    reported.update(
        {
            "runtime": native_runtime,
            "solve_runtime": native_solve_runtime,
            "infer_runtime": controller_runtime,
            "bandit_select_runtime": float(
                bandit_timing.get("select_sec", 0.0)
            ),
            "bandit_loss_eval_runtime": float(
                bandit_timing.get("loss_eval_sec", 0.0)
            ),
            "bandit_update_runtime": float(
                bandit_timing.get("update_sec", 0.0)
            ),
            "bandit_overhead_runtime": bandit_overhead,
            "end_to_end_runtime": float(
                native_runtime + controller_runtime + bandit_overhead
            ),
        }
    )
    if reported.get("failure_feedback_mode") == "budgeted_penalty":
        reported["penalized_cost"] = reported["end_to_end_runtime"] + float(reported["failure_penalty_sec"])
    return reported


def method_stream_summary(
    records: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    outcomes = [record["outcome"] for record in records]
    summary = _summarize(outcomes)
    fields = {
        "setup_runtime": "setup_runtime",
        "native_solve_runtime": "solve_runtime",
        "native_total_runtime": "runtime",
        "controller_runtime": "infer_runtime",
        "setup_bandit_overhead": "bandit_overhead_runtime",
        "setup_bandit_select": "bandit_select_runtime",
        "setup_bandit_loss_eval": "bandit_loss_eval_runtime",
        "setup_bandit_update": "bandit_update_runtime",
        "end_to_end_runtime": "end_to_end_runtime",
    }
    for label, field in (
        ("controller_lifecycle_runtime", "lifecycle_runtime"),
        ("method_wall_runtime", "method_wall_runtime"),
        ("failure_penalty", "failure_penalty_sec"),
        ("penalized_cost", "penalized_cost"),
    ):
        if any(field in outcome for outcome in outcomes):
            fields[label] = field
    summary["totals_sec"] = {
        label: float(sum(float(outcome.get(field, 0.0)) for outcome in outcomes))
        for label, field in fields.items()
    }
    summary["means_sec"] = {
        label: float(
            np.mean([float(outcome.get(field, 0.0)) for outcome in outcomes])
        )
        for label, field in fields.items()
    }
    summary["unique_setup_count"] = int(
        len({json.dumps(record["params"], sort_keys=True) for record in records})
    )
    summary["primary_failure_count"] = int(
        sum(
            str(
                outcome.get(
                    "first_primary_status",
                    outcome.get("primary_status", "success"),
                )
            )
            != "success"
            for outcome in outcomes
        )
    )
    summary["learned_setup_attempt_count"] = int(
        sum(int(outcome.get("primary_attempt_count", 1)) for outcome in outcomes)
    )
    summary["learned_setup_reselection_count"] = int(
        sum(int(outcome.get("learned_reselection_count", 0)) for outcome in outcomes)
    )
    summary["bandit_observation_count"] = int(
        sum(int(outcome.get("bandit_observation_count", 0)) for outcome in outcomes)
    )
    summary["setup_fallback_count"] = int(
        sum(bool(outcome.get("fallback_used", False)) for outcome in outcomes)
    )
    summary["recovered_failure_count"] = int(
        sum(bool(outcome.get("recovered", False)) for outcome in outcomes)
    )
    summary["unrecovered_failure_count"] = int(
        sum(bool(outcome.get("unrecovered_failure", False)) for outcome in outcomes)
    )
    summary["bandit_update_count"] = int(
        sum(bool(outcome.get("bandit_update_committed", False)) for outcome in outcomes)
    )
    summary["controller_update_count"] = int(
        sum(bool(outcome.get("controller_update_committed", False)) for outcome in outcomes)
    )
    return summary


def validate_recovery_stream(
    records: Sequence[Dict[str, Any]],
    *,
    expect_bandit_transaction: bool,
) -> Dict[str, Any]:
    """Validate bounded setup reselection and recovery timing identities."""

    fallback_count = 0
    recovered_count = 0
    unrecovered_count = 0
    bandit_update_count = 0
    controller_update_count = 0
    learned_attempt_count = 0
    learned_reselection_count = 0
    bandit_observation_count = 0
    for index, record in enumerate(records):
        outcome = record["outcome"]
        if outcome.get("failure_feedback_mode") not in {None, "rollback_unrecovered", "budgeted_penalty"}:
            raise AssertionError(f"record {index}: unknown failure-feedback protocol")
        fallback_used = bool(outcome.get("fallback_used", False))
        fallback_status = str(outcome.get("fallback_status", "not_run"))
        primary_status = str(outcome.get("primary_status", "success"))
        if fallback_used and primary_status == "success":
            raise AssertionError(f"record {index}: fallback followed primary success")
        if fallback_used != (fallback_status != "not_run"):
            raise AssertionError(f"record {index}: inconsistent fallback status")
        primary_attempts = int(outcome.get("primary_attempt_count", 1))
        if not (1 <= primary_attempts <= 3):
            raise AssertionError(
                f"record {index}: learned setup attempt count is outside [1, 3]"
            )
        attempt_count = primary_attempts + int(fallback_used)
        if attempt_count > 4:
            raise AssertionError(f"record {index}: more than four native attempts")

        native_total = float(outcome.get("runtime", 0.0))
        native_components = float(outcome.get("setup_runtime", 0.0)) + float(
            outcome.get("solve_runtime", 0.0)
        )
        if not np.isclose(native_total, native_components, rtol=1e-8, atol=1e-10):
            raise AssertionError(f"record {index}: native runtime components do not add up")
        expected_end_to_end = (
            native_total
            + float(outcome.get("infer_runtime", 0.0))
            + float(outcome.get("bandit_overhead_runtime", 0.0))
        )
        if not np.isclose(
            float(outcome.get("end_to_end_runtime", expected_end_to_end)),
            expected_end_to_end,
            rtol=1e-8,
            atol=1e-10,
        ):
            raise AssertionError(f"record {index}: end-to-end components do not add up")

        fallback_count += int(fallback_used)
        recovered_count += int(bool(outcome.get("recovered", False)))
        unrecovered_count += int(bool(outcome.get("unrecovered_failure", False)))
        bandit_update_count += int(bool(outcome.get("bandit_update_committed", False)))
        controller_update_count += int(
            bool(outcome.get("controller_update_committed", False))
        )
        learned_attempt_count += primary_attempts
        learned_reselection_count += int(
            outcome.get("learned_reselection_count", primary_attempts - 1)
        )
        bandit_observation_count += int(
            outcome.get("bandit_observation_count", 0)
        )
        if expect_bandit_transaction:
            expected_update = (
                not outcome.get("unrecovered_failure", False)
                or outcome.get("failure_feedback_mode") == "budgeted_penalty"
            )
            if bool(outcome.get("bandit_update_committed", False)) != expected_update:
                raise AssertionError(f"record {index}: bandit update contradicts failure-feedback protocol")
        if outcome.get("failure_feedback_mode") == "budgeted_penalty":
            penalty = float(outcome.get("failure_penalty_sec", float("nan")))
            if not np.isfinite(penalty) or penalty < 0.0 or (
                not outcome.get("unrecovered_failure", False) and penalty != 0.0
            ):
                raise AssertionError(f"record {index}: invalid objective penalty")
            if not np.isclose(
                outcome.get("penalized_cost", float("nan")),
                expected_end_to_end + penalty, rtol=1e-8, atol=1e-10,
            ):
                raise AssertionError(f"record {index}: objective cost must equal measured cost plus penalty")
    return {
        "processed_instances": int(len(records)),
        "primary_selections": int(learned_attempt_count),
        "learned_reselections": int(learned_reselection_count),
        "native_attempts": int(learned_attempt_count + fallback_count),
        "fallback_uses": int(fallback_count),
        "recovered_failures": int(recovered_count),
        "unrecovered_failures": int(unrecovered_count),
        "bandit_updates": int(bandit_update_count),
        "bandit_observations": int(bandit_observation_count),
        "controller_updates": int(controller_update_count),
        "valid": True,
    }


def build_paired_instance_stream(
    args: argparse.Namespace,
) -> tuple[list[tuple[Dict[str, Any], np.ndarray]], Dict[str, Any]]:
    problem, grid, advection = _problem_stream_spec(args)
    sampled_advection_range = _sampled_advection_range(
        args,
        problem=problem,
    )
    problem_manifest = {
        "problem": problem,
        "grid": [int(value) for value in grid],
        "advection": (
            [float(value) for value in advection]
            if sampled_advection_range is None
            else {
                "mode": "independent_uniform_per_component",
                "range": [
                    float(sampled_advection_range[0]),
                    float(sampled_advection_range[1]),
                ],
            }
        ),
    }

    def generate_segment(*, count: int, seed: int):
        stream_problem = (
            None
            if (
                problem == SCALAR_ANISOTROPIC_DIFFUSION
                and getattr(args, "joint_experiment_spec", None) is None
            )
            else problem
        )
        return generate_difconv_instances(
            T=int(count),
            seed=int(seed),
            grid_choices=[grid],
            c_min=float(args.c_min),
            c_max=float(args.c_max),
            difconv_a=advection,
            problem=stream_problem,
            advection_min=(
                None
                if sampled_advection_range is None
                else sampled_advection_range[0]
            ),
            advection_max=(
                None
                if sampled_advection_range is None
                else sampled_advection_range[1]
            ),
        )

    raw_groups = str(args.train_seed_groups).strip()
    if not raw_groups:
        instance_offset = int(args.instance_offset)
        if instance_offset < 0:
            raise ValueError("instance_offset must be non-negative")
        full_stream = generate_segment(
            count=int(args.train_cases) + instance_offset,
            seed=int(args.seed),
        )
        selected_stream = full_stream[instance_offset:]
        return selected_stream, {
            **problem_manifest,
            "mode": "single_seed_continuation",
            "seed": int(args.seed),
            "window": [
                instance_offset,
                instance_offset + int(args.train_cases),
            ],
            "sha256": _instance_stream_hash(selected_stream),
        }

    seed_groups = [
        _parse_values(group, int)
        for group in raw_groups.split(";")
        if group.strip()
    ]
    shuffle_seeds = _parse_values(args.train_shuffle_seeds, int)
    if len(seed_groups) != len(shuffle_seeds):
        raise ValueError(
            "train_seed_groups and train_shuffle_seeds must have equal lengths"
        )

    stream: list[tuple[Dict[str, Any], np.ndarray]] = []
    segment_metadata = []
    for segment_index, (seeds, shuffle_seed) in enumerate(
        zip(seed_groups, shuffle_seeds)
    ):
        candidates: list[tuple[Dict[str, Any], np.ndarray]] = []
        for seed in seeds:
            generated = generate_segment(
                count=int(args.instance_offset + args.train_cases_per_seed),
                seed=int(seed),
            )
            candidates.extend(generated[int(args.instance_offset) :])
        order = np.random.default_rng(int(shuffle_seed)).permutation(len(candidates))
        selected = [
            candidates[int(index)]
            for index in order[: int(args.train_group_take)]
        ]
        stream.extend(selected)
        segment_metadata.append(
            {
                "segment": int(segment_index),
                "seeds": [int(seed) for seed in seeds],
                "cases_per_seed": int(args.train_cases_per_seed),
                "window": [
                    int(args.instance_offset),
                    int(args.instance_offset + args.train_cases_per_seed),
                ],
                "candidate_cases": int(len(candidates)),
                "shuffle_seed": int(shuffle_seed),
                "selected_cases": int(len(selected)),
            }
        )
    if len(stream) != int(args.train_cases):
        raise ValueError(
            f"Grouped training stream produced {len(stream)} cases, "
            f"expected {args.train_cases}"
        )
    return stream, {
        **problem_manifest,
        "mode": "grouped_shuffled_segments",
        "segments": segment_metadata,
        "sha256": _instance_stream_hash(stream),
    }


# Private aliases retained only while active runners migrate their imports.
_git_revision = git_revision
_configure_paired_environment = configure_paired_environment
_policy_last_arm = policy_last_arm
_report_online_outcome = report_online_outcome
_method_stream_summary = method_stream_summary
_build_paired_instance_stream = build_paired_instance_stream
_validate_recovery_stream = validate_recovery_stream
