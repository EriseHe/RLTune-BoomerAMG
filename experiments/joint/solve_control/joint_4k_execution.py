"""Mode-neutral execution for persistent joint-online comparisons.

Method planning and learner/controller construction intentionally stay with
the caller.  This module owns only the common per-instance execution,
checkpoint, audit, and reporting lifecycle once those objects have been
resolved.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from hypre.bindings import run_with_default_fallback
from joint_artifacts import _write_json_line
from joint_method_spec import ComposableMethodSpec
from joint_online_common import (
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
    _validate_recovery_stream,
)
from joint_reporting import (
    _action_summary,
    _comparison_windows,
    _window_result,
    _write_solve_screen_report,
    _write_summary_csv,
)
from joint_rl_activation import ReliabilityActivationGate
from online_td_experiment_common import _json_ready, _write_json
from problems.amg import normalize_diffusion_advection_context
from run_online_methods_2k import _as_feedback
from setup.space import DEFAULT_SETUP_PARAMS
from setup_aware_compare_common import (
    augment_setup_params,
    classify_rl_failure,
    run_bandit_step_test_final,
    solve_default_baseline_case,
    solve_fixed_w_case,
    solve_no_rl_case,
    solve_setup_aware_rl_case,
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
    ppo_runner: Any
    protocol: Mapping[str, Any]
    warmup: WarmupArtifacts
    solve: SolveExecutionConfig
    warmup_cases: int
    online_cases: int
    method_order_seed: int
    progress_every: int
    aot_enabled: bool
    aot_max_selections_per_case: int
    default_setup_method: str
    include_solve_screen_report: bool
    setup_replay_rows: Sequence[Mapping[str, Any]] | None = None
    shared_online_prefix: bool = False


@dataclass(frozen=True)
class OnlineComparisonHooks:
    """Caller-owned seams retained for compatibility and focused tests.

    In particular, the legacy runner passes its wrapper functions here.  A
    monkeypatch applied to the runner's ``_report_online_outcome`` therefore
    still affects both default-setup execution and the shared online loop.
    """

    method_solver: MethodSolverFactory
    run_default_setup_method: DefaultSetupRunner
    report_online_outcome: OutcomeReporter
    setup_context: SetupContextResolver | None = None


class _FrozenSetupReplayPolicy:
    """One non-learning setup selection supplied by a recorded trajectory."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self._params = dict(params)

    def select(
        self,
        *,
        context: np.ndarray,
        parameter_space: Mapping[str, Any],
    ) -> Dict[str, Any]:
        del context, parameter_space
        return dict(self._params)


def _method_solve_tolerance(
    spec: ComposableMethodSpec | None,
    solve: SolveExecutionConfig,
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
    controller_enabled: bool | None = None,
    controller_bundles: Mapping[str, ControllerBundle],
    ppo_runner: Any,
    composable_specs: Mapping[str, ComposableMethodSpec] | None = None,
) -> MethodSolver:
    """Build the already-resolved solve callable for one method and case."""

    spec = None if composable_specs is None else composable_specs.get(method)
    solve_kind = None if spec is None else spec.solve_kind
    solve_tolerance = _method_solve_tolerance(spec, solve)
    if (
        spec is not None
        and spec.solve_activation is not None
        and controller_enabled is None
    ):
        raise ValueError("Dynamic activation requires the gate's pre-problem decision")
    controller_is_delayed = bool(
        controller_enabled is False
        or (spec is not None and int(spec.solve_activation_case) > int(case_index))
    )
    if (
        (spec is None and method == "bandit_default")
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
            return _as_feedback(native, include_controller=False)

        return solve_default
    if (spec is None and method == "bandit_fixed_w1.6") or solve_kind == "fixed":
        fixed_weight = 1.6 if spec is None else float(spec.fixed_weight)

        def solve_fixed(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_fixed_w_case(
                params=dict(params),
                mkw=dict(mkw),
                w=float(fixed_weight),
                sweeps_down=1,
                sweeps_up=1,
                solve_tol=solve_tolerance,
                solve_max_cycles=int(solve.max_cycles),
            )
            return _as_feedback(native, include_controller=False)

        return solve_fixed
    if (spec is None and method == "bandit_ppo") or solve_kind == "ppo":

        def solve_ppo(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_setup_aware_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solve_policy=ppo_runner,
                augment_params=augment_setup_params,
                classify_rl_failure=lambda *, residual_norm, iterations: (
                    classify_rl_failure(
                        residual_norm=float(residual_norm),
                        iterations=int(iterations),
                        solve_tol=solve_tolerance,
                        solve_max_cycles=int(solve.max_cycles),
                    )
                ),
                solve_max_cycles=int(solve.max_cycles),
                case_progress=float(case_progress),
            )
            return _as_feedback(native, include_controller=True)

        return solve_ppo
    if method in controller_bundles:

        def solve_online_controller(
            params: Dict[str, Any],
        ) -> Dict[str, Any]:
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
                    fallback_attempt=lambda: solve_no_rl_case(
                        params=dict(DEFAULT_SETUP_PARAMS),
                        mkw=dict(mkw),
                        solver_tol=solve_tolerance,
                        solver_max_iter=int(solve.max_cycles),
                        augment_params=augment_setup_params,
                    ),
                )
            )
            return _as_feedback(native, include_controller=True)

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
        feedback = _as_feedback(native, include_controller=False)
    elif spec.name in controller_methods:
        # Online controllers own their transaction and fallback so an
        # unrecovered failure can roll back controller state atomically.
        feedback = solver_fn(dict(DEFAULT_SETUP_PARAMS))
    else:
        recovery = run_with_default_fallback(
            lambda: solver_fn(dict(DEFAULT_SETUP_PARAMS)),
            lambda: solve_no_rl_case(
                params=dict(DEFAULT_SETUP_PARAMS),
                mkw=dict(mkw),
                solver_tol=solve_tolerance,
                solver_max_iter=int(solve.max_cycles),
                augment_params=augment_setup_params,
            ),
            primary_is_default=False,
        )
        native = recovery.to_result()
        native.update(
            {
                "recovery_protocol_applied": True,
                "bandit_update_committed": False,
                "controller_update_committed": False,
            }
        )
        feedback = _as_feedback(
            native,
            include_controller=spec.solve_kind == "ppo",
        )
    reported = report_online_outcome(feedback, bandit_timing={})
    reported["bandit_update_committed"] = False
    return reported


def _default_setup_row(
    *,
    plan: OnlineComparisonPlan,
    hooks: OnlineComparisonHooks,
    online_index: int,
    execution_rank: int,
    mkw: Mapping[str, Any],
    context: np.ndarray,
) -> Dict[str, Any]:
    native = solve_default_baseline_case(
        mkw=dict(mkw),
        solver_tol=float(plan.solve.tolerance),
        solver_max_iter=int(plan.solve.max_cycles),
        augment_params=augment_setup_params,
    )
    outcome = hooks.report_online_outcome(
        _as_feedback(native, include_controller=False),
        bandit_timing={},
    )
    return {
        "stream_index": int(plan.warmup_cases + online_index),
        "online_index": int(online_index),
        "execution_rank": int(execution_rank),
        "mkw": dict(mkw),
        "context": np.asarray(context, dtype=float).tolist(),
        "params": dict(DEFAULT_SETUP_PARAMS),
        "arm_index": -1,
        "fallback_used": 0,
        "bandit_timing": {},
        "outcome": outcome,
    }


def _default_fallback_solver(
    *,
    plan: OnlineComparisonPlan,
    method: str,
    mkw: Mapping[str, Any],
) -> MethodSolver:
    solve_tolerance = _method_solve_tolerance(
        plan.composable_specs.get(method),
        plan.solve,
    )

    def fallback_solver(_params: Dict[str, Any]) -> Dict[str, Any]:
        fallback_native = solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(mkw),
            solver_tol=solve_tolerance,
            solver_max_iter=int(plan.solve.max_cycles),
            augment_params=augment_setup_params,
        )
        return _as_feedback(fallback_native, include_controller=False)

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
    activation_gate: ReliabilityActivationGate | None = None,
) -> Dict[str, Any]:
    if (
        method == plan.default_setup_method
        and method not in plan.composable_specs
    ):
        return _default_setup_row(
            plan=plan,
            hooks=hooks,
            online_index=online_index,
            execution_rank=execution_rank,
            mkw=mkw,
            context=context,
        )

    spec = plan.composable_specs.get(method)
    problem_context = np.asarray(context, dtype=float)
    if (
        spec is not None
        and spec.setup_kind in {"linucb_v5", "linucb_v5_rbf"}
    ):
        problem_context = normalize_diffusion_advection_context(
            problem_context
        )
    activation_kwargs = (
        {} if activation_gate is None
        else {"controller_enabled": activation_gate.active}
    )
    solver_fn = hooks.method_solver(
        method,
        mkw=dict(mkw),
        case_progress=float(online_index) / float(progress_denom),
        case_index=int(plan.warmup_cases + online_index),
        problem_context=problem_context,
        **activation_kwargs,
    )
    if spec is not None and spec.setup_kind == "default":
        outcome = hooks.run_default_setup_method(
            spec=spec,
            solver_fn=solver_fn,
            mkw=dict(mkw),
            controller_methods=tuple(plan.controller_bundles),
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
            hooks.setup_context(
                method=method,
                mkw=dict(mkw),
                context=learner_context,
            ),
            dtype=float,
        )
    if (
        spec is not None
        and spec.setup_kind in {"linucb_v5", "linucb_v5_rbf"}
        and not np.array_equal(
            learner_context,
            problem_context,
        )
    ):
        raise ValueError(
            "LinUCB v5 setup and solve must receive the same canonical "
            "problem_context"
        )
    if plan.setup_replay_rows is not None:
        replay_row = plan.setup_replay_rows[int(online_index)]
        replay_params = dict(replay_row["params"])
        params, native, timing, fallback_used, _update_sec = (
            run_bandit_step_test_final(
                policy=_FrozenSetupReplayPolicy(replay_params),
                parameter_space={},
                problem_context=learner_context,
                solver_fn=solver_fn,
                fallback_solver_fn=_default_fallback_solver(
                    plan=plan,
                    method=method,
                    mkw=mkw,
                ),
                prev_update_est=0.0,
            )
        )
        return {
            "stream_index": int(plan.warmup_cases + online_index),
            "online_index": int(online_index),
            "execution_rank": int(execution_rank),
            "mkw": dict(mkw),
            "context": learner_context.tolist(),
            "params": dict(params),
            "arm_index": int(replay_row.get("arm_index", -1)),
            "fallback_used": int(fallback_used),
            "bandit_timing": dict(timing),
            "setup_replay_source_online_index": int(
                replay_row.get("online_index", online_index)
            ),
            "outcome": hooks.report_online_outcome(
                native,
                bandit_timing=timing,
            ),
        }
    branch = plan.branches[method]
    params, native, timing, fallback_used, update_sec = (
        run_bandit_step_test_final(
            policy=branch.policy,
            parameter_space=branch.parameter_space,
            problem_context=learner_context,
            solver_fn=solver_fn,
            fallback_solver_fn=_default_fallback_solver(
                plan=plan,
                method=method,
                mkw=mkw,
            ),
            prev_update_est=float(previous_update[method]),
        )
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
        "arm_index": _policy_last_arm(branch.policy),
        "fallback_used": int(fallback_used),
        "bandit_timing": dict(timing),
        "outcome": hooks.report_online_outcome(
            native,
            bandit_timing=timing,
        ),
    }


def _checkpoint_controllers(
    plan: OnlineComparisonPlan,
    *,
    completed_instances: int,
) -> None:
    audit_episodes = {
        *range(1000, int(plan.online_cases) + 1, 1000),
        int(plan.online_cases),
    }
    if completed_instances not in audit_episodes:
        return
    for method, bundle in plan.controller_bundles.items():
        bundle.save(
            plan.checkpoints_dir
            / f"{method}_episode_{completed_instances}.npz"
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
    if (
        completed_instances % max(1, int(plan.progress_every)) != 0
        and completed_instances != len(plan.online_instances)
    ):
        return
    for handle in handles.values():
        handle.flush()
    order_handle.flush()
    progress = {
        "completed_online_instances": int(completed_instances),
        "methods": {
            method: _method_stream_summary(rows)
            for method, rows in records.items()
        },
        "controllers": {
            method: bundle.summary()
            for method, bundle in plan.controller_bundles.items()
        },
        "bandit_online_steps": dict(bandit_online_steps),
    }
    _write_json(plan.output_dir / "progress.json", progress)
    print(
        json.dumps(
            {
                "stage": "joint_online_4k_compare",
                "done": int(completed_instances),
                "total": int(len(plan.online_instances)),
            }
        ),
        flush=True,
    )


def _fork_shared_online_prefix(
    plan: OnlineComparisonPlan, *, source: str, target: str,
    completed_instances: int, previous_update: Mapping[str, float],
    bandit_online_steps: Mapping[str, int],
    artifact_suffix: str = "",
) -> None:
    """Restore the existing full checkpoint into an independent AOT branch."""
    started_at = time.perf_counter()
    source_model = plan.branches[source].policy.model
    target_model = plan.branches[target].policy.model
    if source_model is target_model:
        raise RuntimeError("Shared-prefix branches must have independent models")
    state_path = plan.checkpoints_dir / f"shared_prefix{artifact_suffix}_setup.npz"
    metadata = {"source": source, "completed_instances": completed_instances}
    source_model.save_mutable_state(state_path, metadata=metadata)
    target_model.load_mutable_state(state_path)
    # Structured candidates also anchor on history[-1].arm_index. The existing
    # checkpoint resets history, so restore it as part of the decision state.
    target_model.history = copy.deepcopy(source_model.history)
    target_model.candidate_stats_history = copy.deepcopy(source_model.candidate_stats_history)
    target_model._local_neighbor_cache = copy.deepcopy(source_model._local_neighbor_cache)
    restored_path = plan.checkpoints_dir / f"shared_prefix{artifact_suffix}_restored_setup.npz"
    target_model.save_mutable_state(restored_path, metadata=metadata)
    with np.load(state_path, allow_pickle=False) as expected, np.load(
        restored_path, allow_pickle=False,
    ) as actual:
        equal = expected.files == actual.files and all(
            np.array_equal(expected[key], actual[key]) for key in expected.files
        )
    independent_arrays = all(
        not np.shares_memory(getattr(source_model, key), getattr(target_model, key))
        for key in ("A_inv", "b", "failure_A_inv", "failure_b")
    )
    source_schedule = source_model._candidate_schedule
    target_schedule = target_model._candidate_schedule
    independent_schedule = source_schedule is None or (
        source_schedule is not target_schedule
        and source_schedule.cursor == target_schedule.cursor
    )
    valid = (
        equal and independent_arrays and independent_schedule
        and source_model.rng is not target_model.rng
        and source_model._cand is not target_model._cand
        and previous_update[source] == previous_update[target]
        and bandit_online_steps[source] == bandit_online_steps[target] == completed_instances
    )
    audit = {
        "valid": valid,
        "source": source, "target": target,
        "completed_instances": completed_instances,
        "equal_checkpoint_arrays": equal,
        "independent_model_arrays": independent_arrays,
        "independent_schedule_cursor": independent_schedule,
        "candidate_schedule_cursor": None if source_schedule is None else source_schedule.cursor,
        "previous_update_sec": dict(previous_update),
        "bandit_online_steps": dict(bandit_online_steps),
        "setup_checkpoint_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "controller_summaries_at_fork": {
            method: bundle.summary() for method, bundle in plan.controller_bundles.items()
        },
        "fork_preparation_runtime_sec": time.perf_counter() - started_at,
        "fork_preparation_in_online_cost": False,
    }
    _write_json(plan.output_dir / f"shared_prefix{artifact_suffix}.json", audit)
    if not valid:
        raise RuntimeError("Shared-prefix fork state audit failed")
    print(json.dumps({"stage": "shared_prefix_fork", **audit}), flush=True)


def _execute_online_instances(
    plan: OnlineComparisonPlan,
    hooks: OnlineComparisonHooks,
) -> tuple[Dict[str, list[Dict[str, Any]]], Dict[str, int]]:
    previous_update = {method: 0.0 for method in plan.bandit_methods}
    bandit_online_steps = {
        method: 0 for method in plan.bandit_methods
    }
    records: Dict[str, list[Dict[str, Any]]] = {
        method: [] for method in plan.methods
    }
    handles = {
        method: (
            plan.trajectories_dir / f"{method}.jsonl"
        ).open("w", encoding="utf-8")
        for method in plan.methods
    }
    order_handle = (
        plan.trajectories_dir / "method_order.jsonl"
    ).open("w", encoding="utf-8")
    order_rng = np.random.default_rng(int(plan.method_order_seed))
    progress_denom = max(1, len(plan.online_instances) - 1)
    activation_gates = {
        method: ReliabilityActivationGate(spec.solve_activation)
        for method, spec in plan.composable_specs.items()
        if spec.solve_activation is not None
    }
    shared_prefix = plan.shared_online_prefix
    nested_prefix = shared_prefix and not activation_gates
    waiting = set()
    if nested_prefix:
        # The latest-starting RL branch supplies the reference prefix when no
        # setup-only baseline was requested. Every fork precedes its RL start.
        references = [method for method, spec in plan.composable_specs.items() if spec.solve_kind == "default"]
        source = references[0] if references else max(
            plan.methods, key=lambda method: plan.composable_specs[method].solve_activation_case
        )
        waiting = set(plan.methods) - {source}
        shared_prefix = False
    elif shared_prefix:
        source = next(method for method, spec in plan.composable_specs.items() if spec.solve_activation_case)
        target = next(iter(activation_gates))
        fixed_boundary = plan.composable_specs[source].solve_activation_case
    try:
        for online_index, (mkw, context) in enumerate(plan.online_instances):
            order = [
                plan.methods[int(index)]
                for index in order_rng.permutation(len(plan.methods))
            ]
            if shared_prefix:
                order = [source]
            elif nested_prefix:
                order = [method for method in order if method not in waiting]
            _write_json_line(
                order_handle,
                {
                    "online_index": int(online_index),
                    "method_order": order,
                    **({"shared_prefix_source": source} if shared_prefix or waiting else {}),
                },
            )
            case_rows = {}
            for execution_rank, method in enumerate(order):
                method_started = time.perf_counter()
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
                    activation_gate=activation_gates.get(method),
                )
                # Independent reporting stopwatch, not a replacement learning
                # target. Includes matrix construction and binding/wrapper work;
                # excludes outer trajectory I/O, checkpointing and plotting.
                row["outcome"]["method_wall_runtime"] = time.perf_counter() - method_started
                case_rows[method] = row
            if shared_prefix:
                # One measured solve/update contributes once to each method's
                # logical cost. Only the dynamic copy receives gate overhead.
                case_rows[target] = copy.deepcopy(case_rows[source])
                for row in case_rows.values():
                    row["shared_prefix_source"] = source
                previous_update[target] = previous_update[source]
                bandit_online_steps[target] = bandit_online_steps[source]
            elif waiting:
                for target in waiting:
                    case_rows[target] = copy.deepcopy(case_rows[source])
                    case_rows[target]["shared_prefix_source"] = source
                    previous_update[target] = previous_update[source]
                    bandit_online_steps[target] = bandit_online_steps[source]
                case_rows[source]["shared_prefix_source"] = source
            for method in plan.methods:
                if method in activation_gates:
                    gate = activation_gates[method]
                    outcome = case_rows[method]["outcome"]
                    started_at = time.perf_counter()
                    enabled_on_this_problem = gate.active
                    observed_on_this_problem = gate.can_observe
                    if observed_on_this_problem:
                        # primary_status can describe a later reselection; use
                        # the first attempt, including recovered construction failures.
                        if "first_primary_status" not in outcome:
                            raise RuntimeError(
                                "Dynamic activation requires first_primary_status"
                            )
                        failed = outcome["first_primary_status"] != "success"
                        gate.observe(first_attempt_failed=failed)
                    case_rows[method]["rl_activation"] = {
                        **gate.summary(),
                        "controller_enabled": enabled_on_this_problem,
                        "observed_this_problem": observed_on_this_problem,
                    }
                    elapsed = time.perf_counter() - started_at
                    outcome["activation_runtime"] = elapsed
                    # The existing controller-overhead component also accounts
                    # for the gate, preserving total = native + controller + bandit.
                    outcome["infer_runtime"] = (
                        float(outcome.get("infer_runtime", 0.0)) + elapsed
                    )
                    outcome["end_to_end_runtime"] = (
                        float(outcome["end_to_end_runtime"]) + elapsed
                    )
                    if not enabled_on_this_problem and gate.active:
                        print(json.dumps({
                            "stage": "rl_activation",
                            "method": method,
                            "completed_problem": gate.crossing_case,
                            "first_rl_problem": gate.crossing_case + 1,
                            "log_evalue": gate.log_evalue,
                            "log_threshold": gate.log_threshold,
                        }), flush=True)
                records[method].append(case_rows[method])
                _write_json_line(handles[method], case_rows[method])

            done = online_index + 1
            if shared_prefix and (done == fixed_boundary or activation_gates[target].active):
                _fork_shared_online_prefix(
                    plan, source=source, target=target, completed_instances=done,
                    previous_update=previous_update, bandit_online_steps=bandit_online_steps,
                )
                shared_prefix = False
            for target in plan.methods:
                if target in waiting and done == plan.composable_specs[target].solve_activation_case:
                    _fork_shared_online_prefix(
                        plan, source=source, target=target, completed_instances=done,
                        previous_update=previous_update, bandit_online_steps=bandit_online_steps,
                        artifact_suffix=f"_{target}",
                    )
                    waiting.remove(target)
            _checkpoint_controllers(
                plan,
                completed_instances=done,
            )
            _write_progress(
                plan,
                completed_instances=done,
                records=records,
                bandit_online_steps=bandit_online_steps,
                handles=handles,
                order_handle=order_handle,
            )
            if activation_gates and (
                done % max(1, plan.progress_every) == 0
                or done == len(plan.online_instances)
            ):
                _write_json(
                    plan.output_dir / "rl_activation.json",
                    {method: gate.summary() for method, gate in activation_gates.items()},
                )
    finally:
        for handle in handles.values():
            handle.close()
        order_handle.close()
    return records, bandit_online_steps


def _validate_and_save_final_state(
    plan: OnlineComparisonPlan,
    *,
    records: Mapping[str, list[Dict[str, Any]]],
    bandit_online_steps: Mapping[str, int],
) -> tuple[Dict[str, Any], Path]:
    recovery_audit = {
        method: _validate_recovery_stream(
            rows,
            expect_bandit_transaction=method in plan.bandit_methods,
        )
        for method, rows in records.items()
    }
    if (
        bandit_online_steps
        and set(bandit_online_steps.values())
        != {len(plan.online_instances)}
    ):
        raise RuntimeError(
            "Every LinUCB branch must process every online instance: "
            f"{dict(bandit_online_steps)}"
        )

    final_bandit_dir = plan.output_dir / "final_bandit_states"
    final_bandit_dir.mkdir(parents=True, exist_ok=True)
    for method, branch in plan.branches.items():
        branch.policy.model.save_mutable_state(
            final_bandit_dir / f"{method}.npz",
            metadata={
                "method": method,
                "online_steps": bandit_online_steps[method],
            },
        )
    for method, bundle in plan.controller_bundles.items():
        bundle.save(plan.checkpoints_dir / f"{method}_final.npz")
    return recovery_audit, final_bandit_dir


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
            {
                method: rows[start:stop]
                for method, rows in records.items()
            },
            seed=int(plan.method_order_seed + start + stop),
        )
        for name, (start, stop) in windows.items()
    }
    window_actions = {
        name: {
            method: _action_summary(records[method][start:stop])
            for method in plan.controller_bundles
        }
        for name, (start, stop) in windows.items()
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
            for method, bundle in plan.controller_bundles.items()
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
                {
                    "screen_report": str(
                        plan.output_dir / "screen_report.md"
                    )
                }
                if plan.include_solve_screen_report
                else {}
            ),
        },
    }
    activation_summary = {
        method: {
            **rows[-1]["rl_activation"],
            "rl_problem_count": sum(
                bool(row["rl_activation"]["controller_enabled"]) for row in rows
            ),
            "total_activation_runtime": sum(
                float(row["outcome"].get("activation_runtime", 0.0)) for row in rows
            ),
        }
        for method, rows in records.items()
        if rows and "rl_activation" in rows[-1]
    }
    if activation_summary:
        result["rl_activation"] = activation_summary
    if plan.ppo_runner is not None:
        result["ppo"] = {
            "forced_initial_action_count": int(
                plan.ppo_runner.forced_initial_action_count
            ),
        }
    _write_summary_csv(
        plan.output_dir / summary_name,
        dict(records),
        dict(plan.family_by_method),
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
                    "stage": "joint_online_4k_done",
                    "output": plan.output_dir / "result.json",
                }
            )
        ),
        flush=True,
    )
    return result


def run_online_comparison(
    plan: OnlineComparisonPlan,
    *,
    hooks: OnlineComparisonHooks,
) -> Dict[str, Any]:
    """Run and persist one already-resolved persistent online comparison."""

    records, bandit_online_steps = _execute_online_instances(plan, hooks)
    recovery_audit, final_bandit_dir = _validate_and_save_final_state(
        plan,
        records=records,
        bandit_online_steps=bandit_online_steps,
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
