"""Mode-neutral execution for persistent joint-online comparisons.

Method planning and learner/controller construction intentionally stay with
the caller.  This module owns only the common per-instance execution,
checkpoint, audit, and reporting lifecycle once those objects have been
resolved.
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence
import numpy as np
from hypre.bindings import execute_attempt
from hypre.bindings.recovery import InvalidObservationError, validate_failure_penalty
from experiments.paper_final.online.artifacts import _write_json_line
from experiments.paper_final.online.method_spec import ComposableMethodSpec
from experiments.paper_final.online.case_loop import (
    _method_stream_summary,
    _report_online_outcome,
    _validate_recovery_stream,
)
from experiments.paper_final.online.reporting import (
    _action_summary,
    _comparison_windows,
    _window_result,
    _write_solve_screen_report,
    _write_summary_csv,
)
from experiments.paper_final.online.io import _json_ready, _write_json
from experiments.paper_final.online.feedback import as_feedback
from setup.space import DEFAULT_SETUP_PARAMS
from hypre.bindings import augment_setup_params
from experiments.paper_final.online.setup_branches import run_bandit_step_test_final
from experiments.paper_final.online.native_evaluation import (
    solve_default_baseline_case,
    solve_no_rl_case,
)
from solve.controllers.common import ControllerBundle, OnlineSolveCase

SetupInstance = tuple[Dict[str, Any], np.ndarray]
MethodSolver = Callable[[Dict[str, Any]], Dict[str, Any]]
MethodSolverFactory = Callable[..., MethodSolver]
DefaultSetupRunner = Callable[..., Dict[str, Any]]
OutcomeReporter = Callable[..., Dict[str, Any]]
SetupContextResolver = Callable[..., np.ndarray]


@dataclass(frozen=True)
class SolveExecutionConfig:
    """Solve controls shared by every branch in one comparison."""

    tolerance: float
    max_cycles: int
    failure_penalty_sec: float | None = None

    def __post_init__(self) -> None:
        validate_failure_penalty(self.failure_penalty_sec)


@dataclass(frozen=True)
class WarmupArtifacts:
    """Already-resolved warmup data embedded in the final result."""

    summary: Mapping[str, Any]
    trajectory: Any
    state: Any
    record_count: int


@dataclass(frozen=True)
class OnlineComparisonPlan:
    """All resolved state required by the mode-neutral online loop."""

    output_dir: Path
    trajectories_dir: Path
    checkpoints_dir: Path
    methods: tuple[str, ...]
    family_by_method: Mapping[str, str]
    bandit_methods: tuple[str, ...]
    online_instances: Sequence[SetupInstance]
    composable_specs: Mapping[str, ComposableMethodSpec]
    branches: Mapping[str, Any]
    controller_bundles: Mapping[str, ControllerBundle]
    protocol: Mapping[str, Any]
    warmup: WarmupArtifacts
    solve: SolveExecutionConfig
    warmup_cases: int
    online_cases: int
    method_order_seed: int
    progress_every: int
    aot_enabled: bool
    aot_max_selections_per_case: int
    include_solve_screen_report: bool


@dataclass(frozen=True)
class OnlineComparisonHooks:
    """Resolved solve, context, and outcome callbacks for the online loop."""

    method_solver: MethodSolverFactory
    run_default_setup_method: DefaultSetupRunner
    report_online_outcome: OutcomeReporter
    setup_context: SetupContextResolver | None = None


def _method_solve_tolerance(
    spec: ComposableMethodSpec | None, solve: SolveExecutionConfig
) -> float:
    if spec is None:
        return float(solve.tolerance)
    return spec.resolve_solve_tolerance(float(solve.tolerance))


def make_method_solver(
    method: str,
    *,
    solve: SolveExecutionConfig,
    mkw: Mapping[str, Any],
    case_progress: float,
    case_index: int = 0,
    problem_context: Sequence[float] | None = None,
    controller_bundles: Mapping[str, ControllerBundle],
    composable_specs: Mapping[str, ComposableMethodSpec] | None = None,
) -> MethodSolver:
    """Build the already-resolved solve callable for one method and case."""
    spec = None if composable_specs is None else composable_specs.get(method)
    solve_kind = None if spec is None else spec.solve_kind
    solve_tolerance = _method_solve_tolerance(spec, solve)
    controller_is_delayed = bool(
        spec is not None and int(spec.solve_activation_case) > int(case_index)
    )
    if (
        spec is None
        and method == "bandit_default"
        or solve_kind == "default"
        or controller_is_delayed
    ):

        def solve_default(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_no_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solver_tol=solve_tolerance,
                solver_max_iter=int(solve.max_cycles),
                augment_params=augment_setup_params,
            )
            return as_feedback(native, include_controller=False)

        return solve_default
    if method in controller_bundles:

        def solve_online_controller(params: Dict[str, Any]) -> Dict[str, Any]:
            native = controller_bundles[method].run_case(
                OnlineSolveCase(
                    mkw=dict(mkw),
                    params=dict(params),
                    solve_tol=solve_tolerance,
                    solve_max_cycles=int(solve.max_cycles),
                    learn=True,
                    explore=True,
                    problem_context=problem_context,
                    record_action_metadata=True,
                    failure_penalty_sec=solve.failure_penalty_sec,
                    fallback_attempt=lambda: solve_no_rl_case(
                        params=dict(DEFAULT_SETUP_PARAMS),
                        mkw=dict(mkw),
                        solver_tol=solve_tolerance,
                        solver_max_iter=int(solve.max_cycles),
                        augment_params=augment_setup_params,
                    ),
                )
            )
            return as_feedback(native, include_controller=True)

        return solve_online_controller
    raise ValueError(f"Unsupported method: {method}")


def run_default_setup_method(
    *,
    spec: ComposableMethodSpec,
    solver_fn: MethodSolver,
    solve: SolveExecutionConfig,
    mkw: Mapping[str, Any],
    controller_methods: Sequence[str],
    report_online_outcome: OutcomeReporter = _report_online_outcome,
) -> Dict[str, Any]:
    """Execute one default-setup branch under the active recovery protocol."""
    solve_tolerance = _method_solve_tolerance(spec, solve)
    if spec.solve_kind == "default":
        native = solve_default_baseline_case(
            mkw=dict(mkw),
            solver_tol=solve_tolerance,
            solver_max_iter=int(solve.max_cycles),
            augment_params=augment_setup_params,
        )
        feedback = as_feedback(native, include_controller=False)
    else:
        raise ValueError("The official default setup branch uses default solve")
    reported = report_online_outcome(feedback, bandit_timing={})
    reported["bandit_update_committed"] = False
    return reported


def _default_fallback_solver(
    *, plan: OnlineComparisonPlan, method: str, mkw: Mapping[str, Any]
) -> MethodSolver:
    solve_tolerance = _method_solve_tolerance(
        plan.composable_specs.get(method), plan.solve
    )

    def fallback_solver(_params: Dict[str, Any]) -> Dict[str, Any]:
        fallback_native = solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(mkw),
            solver_tol=solve_tolerance,
            solver_max_iter=int(plan.solve.max_cycles),
            augment_params=augment_setup_params,
        )
        return as_feedback(fallback_native, include_controller=False)

    return fallback_solver


def _run_one_method(
    *,
    plan: OnlineComparisonPlan,
    hooks: OnlineComparisonHooks,
    method: str,
    online_index: int,
    execution_rank: int,
    mkw: Mapping[str, Any],
    context: np.ndarray,
    progress_denom: int,
    previous_update: Dict[str, float],
    bandit_online_steps: Dict[str, int],
) -> Dict[str, Any]:
    spec = plan.composable_specs.get(method)
    problem_context = np.asarray(context, dtype=float)
    solver_fn = hooks.method_solver(
        method,
        mkw=dict(mkw),
        case_progress=float(online_index) / float(progress_denom),
        case_index=int(plan.warmup_cases + online_index),
        problem_context=problem_context,
    )
    if spec is not None and spec.setup_kind == "default":

        def default_attempt():
            return hooks.run_default_setup_method(
                spec=spec,
                solver_fn=solver_fn,
                mkw=dict(mkw),
                controller_methods=tuple(plan.controller_bundles),
            )

        outcome = (
            execute_attempt(default_attempt, require_valid_observation=True).result
            if plan.solve.failure_penalty_sec is not None
            else default_attempt()
        )
        return {
            "stream_index": int(plan.warmup_cases + online_index),
            "online_index": int(online_index),
            "execution_rank": int(execution_rank),
            "mkw": dict(mkw),
            "context": np.asarray(context, dtype=float).tolist(),
            "params": dict(DEFAULT_SETUP_PARAMS),
            "arm_index": -1,
            "fallback_used": int(bool(outcome.get("fallback_used", False))),
            "bandit_timing": {},
            "outcome": outcome,
        }
    learner_context = problem_context
    if hooks.setup_context is not None:
        learner_context = np.asarray(
            hooks.setup_context(method=method, mkw=dict(mkw), context=learner_context),
            dtype=float,
        )
    branch = plan.branches[method]
    (params, native, timing, fallback_used, update_sec) = run_bandit_step_test_final(
        policy=branch.policy,
        parameter_space=branch.parameter_space,
        problem_context=learner_context,
        solver_fn=solver_fn,
        fallback_solver_fn=_default_fallback_solver(plan=plan, method=method, mkw=mkw),
        prev_update_est=float(previous_update[method]),
        failure_penalty_sec=plan.solve.failure_penalty_sec,
    )
    previous_update[method] = float(update_sec)
    if plan.aot_enabled:
        branch.policy.model.finish_candidate_schedule_case(
            max_selections=int(plan.aot_max_selections_per_case)
        )
    bandit_online_steps[method] += 1
    return {
        "stream_index": int(plan.warmup_cases + online_index),
        "online_index": int(online_index),
        "execution_rank": int(execution_rank),
        "mkw": dict(mkw),
        "context": learner_context.tolist(),
        "params": dict(params),
        "arm_index": int(native.get("selected_arm_index", -1)),
        "fallback_used": int(fallback_used),
        "bandit_timing": dict(timing),
        "outcome": hooks.report_online_outcome(native, bandit_timing=timing),
    }


def _checkpoint_controllers(
    plan: OnlineComparisonPlan, *, completed_instances: int
) -> None:
    audit_episodes = {
        *range(1000, int(plan.online_cases) + 1, 1000),
        int(plan.online_cases),
    }
    if completed_instances not in audit_episodes:
        return
    for method, bundle in plan.controller_bundles.items():
        bundle.save(
            plan.checkpoints_dir / f"{method}_episode_{completed_instances}.npz"
        )


def _write_progress(
    plan: OnlineComparisonPlan,
    *,
    completed_instances: int,
    records: Mapping[str, list[Dict[str, Any]]],
    bandit_online_steps: Mapping[str, int],
    handles: Mapping[str, Any],
    order_handle: Any,
) -> None:
    if completed_instances % max(
        1, int(plan.progress_every)
    ) != 0 and completed_instances != len(plan.online_instances):
        return
    for handle in handles.values():
        handle.flush()
    order_handle.flush()
    progress = {
        "completed_online_instances": int(completed_instances),
        "methods": {
            method: _method_stream_summary(rows) for (method, rows) in records.items()
        },
        "controllers": {
            method: bundle.summary()
            for (method, bundle) in plan.controller_bundles.items()
        },
        "bandit_online_steps": dict(bandit_online_steps),
    }
    _write_json(plan.output_dir / "progress.json", progress)
    print(
        json.dumps(
            {
                "stage": "module04_online_progress",
                "done": int(completed_instances),
                "total": int(len(plan.online_instances)),
            }
        ),
        flush=True,
    )


def _execute_online_instances(
    plan: OnlineComparisonPlan, hooks: OnlineComparisonHooks
) -> tuple[Dict[str, list[Dict[str, Any]]], Dict[str, int]]:
    previous_update = {method: 0.0 for method in plan.bandit_methods}
    bandit_online_steps = {method: 0 for method in plan.bandit_methods}
    records: Dict[str, list[Dict[str, Any]]] = {method: [] for method in plan.methods}
    handles = {
        method: (plan.trajectories_dir / f"{method}.jsonl").open("w", encoding="utf-8")
        for method in plan.methods
    }
    order_handle = (plan.trajectories_dir / "method_order.jsonl").open(
        "w", encoding="utf-8"
    )
    order_rng = np.random.default_rng(int(plan.method_order_seed))
    progress_denom = max(1, len(plan.online_instances) - 1)
    try:
        for online_index, (mkw, context) in enumerate(plan.online_instances):
            order = [
                plan.methods[int(index)]
                for index in order_rng.permutation(len(plan.methods))
            ]
            _write_json_line(
                order_handle, {"online_index": int(online_index), "method_order": order}
            )
            case_rows = {}
            for execution_rank, method in enumerate(order):
                method_started = time.perf_counter()
                try:
                    row = _run_one_method(
                        plan=plan,
                        hooks=hooks,
                        method=method,
                        online_index=online_index,
                        execution_rank=execution_rank,
                        mkw=mkw,
                        context=context,
                        progress_denom=progress_denom,
                        previous_update=previous_update,
                        bandit_online_steps=bandit_online_steps,
                    )
                except InvalidObservationError as exc:
                    _write_json(
                        plan.output_dir / "invalid_observation.json",
                        {
                            "reason": str(exc),
                            "method": method,
                            "problem": online_index + 1,
                            "matrix": dict(mkw),
                            "requires_fresh_run": True,
                        },
                    )
                    raise
                row["outcome"]["method_wall_runtime"] = (
                    time.perf_counter() - method_started
                )
                new_feedback = (
                    plan.protocol.get("failure_feedback", {}).get("mode")
                    == "budgeted_penalty"
                )
                if new_feedback:
                    coefficient = float(plan.solve.failure_penalty_sec)
                    row["outcome"].update(
                        failure_feedback_mode="budgeted_penalty",
                        failure_penalty_sec=coefficient
                        if row["outcome"].get("unrecovered_failure", False)
                        else 0.0,
                    )
                case_rows[method] = row
            for method in plan.methods:
                outcome = case_rows[method]["outcome"]
                if outcome.get("failure_feedback_mode") == "budgeted_penalty":
                    outcome["penalized_cost"] = (
                        outcome["end_to_end_runtime"] + outcome["failure_penalty_sec"]
                    )
                records[method].append(case_rows[method])
                _write_json_line(handles[method], case_rows[method])
            done = online_index + 1
            _checkpoint_controllers(plan, completed_instances=done)
            _write_progress(
                plan,
                completed_instances=done,
                records=records,
                bandit_online_steps=bandit_online_steps,
                handles=handles,
                order_handle=order_handle,
            )
    finally:
        for handle in handles.values():
            handle.close()
        order_handle.close()
    return (records, bandit_online_steps)


def _validate_and_save_final_state(
    plan: OnlineComparisonPlan,
    *,
    records: Mapping[str, list[Dict[str, Any]]],
    bandit_online_steps: Mapping[str, int],
) -> tuple[Dict[str, Any], Path]:
    recovery_audit = {
        method: _validate_recovery_stream(
            rows, expect_bandit_transaction=method in plan.bandit_methods
        )
        for (method, rows) in records.items()
    }
    if bandit_online_steps and set(bandit_online_steps.values()) != {
        len(plan.online_instances)
    }:
        raise RuntimeError(
            f"Every LinUCB branch must process every online instance: {dict(bandit_online_steps)}"
        )
    final_bandit_dir = plan.output_dir / "final_bandit_states"
    final_bandit_dir.mkdir(parents=True, exist_ok=True)
    for method, branch in plan.branches.items():
        branch.policy.model.save_mutable_state(
            final_bandit_dir / f"{method}.npz",
            metadata={"method": method, "online_steps": bandit_online_steps[method]},
        )
    for method, bundle in plan.controller_bundles.items():
        bundle.save(plan.checkpoints_dir / f"{method}_final.npz")
    return (recovery_audit, final_bandit_dir)


def _build_result(
    plan: OnlineComparisonPlan,
    *,
    records: Mapping[str, list[Dict[str, Any]]],
    bandit_online_steps: Mapping[str, int],
    recovery_audit: Mapping[str, Any],
    final_bandit_dir: Path,
) -> Dict[str, Any]:
    windows = _comparison_windows(int(plan.online_cases))
    window_results = {
        name: _window_result(
            {method: rows[start:stop] for (method, rows) in records.items()},
            seed=int(plan.method_order_seed + start + stop),
        )
        for (name, (start, stop)) in windows.items()
    }
    window_actions = {
        name: {
            method: _action_summary(records[method][start:stop])
            for method in plan.controller_bundles
        }
        for (name, (start, stop)) in windows.items()
    }
    summary_name = f"summary_{int(plan.online_cases)}.csv"
    result = {
        "protocol": plan.protocol,
        "warmup": {
            "summary": plan.warmup.summary,
            "trajectory": plan.warmup.trajectory,
            "state": plan.warmup.state,
            "records": int(plan.warmup.record_count),
        },
        "windows": window_results,
        "controllers": {
            method: bundle.summary()
            for (method, bundle) in plan.controller_bundles.items()
        },
        "bandit_online_steps": dict(bandit_online_steps),
        "recovery_audit": dict(recovery_audit),
        "actions": {
            method: _action_summary(records[method])
            for method in plan.controller_bundles
        },
        "window_actions": window_actions,
        "artifacts": {
            "trajectories": str(plan.trajectories_dir),
            "summary_csv": str(plan.output_dir / summary_name),
            "checkpoints": str(plan.checkpoints_dir),
            "final_bandit_states": str(final_bandit_dir),
            **(
                {"screen_report": str(plan.output_dir / "screen_report.md")}
                if plan.include_solve_screen_report
                else {}
            ),
        },
    }
    _write_summary_csv(
        plan.output_dir / summary_name, dict(records), dict(plan.family_by_method)
    )
    if plan.include_solve_screen_report:
        _write_solve_screen_report(
            plan.output_dir / "screen_report.md",
            window_results=window_results,
            window_actions=window_actions,
        )
    _write_json(plan.output_dir / "result.json", result)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "module04_online_complete",
                    "output": plan.output_dir / "result.json",
                }
            )
        ),
        flush=True,
    )
    return result


def run_online_comparison(
    plan: OnlineComparisonPlan, *, hooks: OnlineComparisonHooks
) -> Dict[str, Any]:
    """Run and persist one already-resolved persistent online comparison."""
    (records, bandit_online_steps) = _execute_online_instances(plan, hooks)
    (recovery_audit, final_bandit_dir) = _validate_and_save_final_state(
        plan, records=records, bandit_online_steps=bandit_online_steps
    )
    return _build_result(
        plan,
        records=records,
        bandit_online_steps=bandit_online_steps,
        recovery_audit=recovery_audit,
        final_bandit_dir=final_bandit_dir,
    )


__all__ = [
    "OnlineComparisonHooks",
    "OnlineComparisonPlan",
    "SolveExecutionConfig",
    "WarmupArtifacts",
    "make_method_solver",
    "run_default_setup_method",
    "run_online_comparison",
]
