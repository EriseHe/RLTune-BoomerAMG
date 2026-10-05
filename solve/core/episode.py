"""Shared transactional episode execution for online solve controllers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from hypre.bindings import (
    AMGNativeError,
    AttemptOutcome,
    AttemptStatus,
    RecoveryOutcome,
    SolveStatus,
    augment_setup_params,
    create_env,
    execute_attempt,
)
from hypre.bindings.recovery import (
    InvalidObservationError,
    validate_failure_penalty,
    validate_runtime_cost,
)
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.core.outcomes import classify_rl_failure


@dataclass
class _EpisodeTrace:
    """Diagnostic values collected outside the measured controller operations."""

    action_counts: dict[str, int] = field(default_factory=dict)
    cycle_actions: list[float] = field(default_factory=list)
    cycle_action_values: list[float] = field(default_factory=list)
    cycle_residuals: list[float] = field(default_factory=list)
    cycle_residual_ratios: list[float] = field(default_factory=list)
    cycle_times: list[float] = field(default_factory=list)
    cycle_explored: list[bool] = field(default_factory=list)
    cycle_greedy_indices: list[int] = field(default_factory=list)
    cycle_epsilons: list[float] = field(default_factory=list)
    cycle_selected_uncertainties: list[float] = field(default_factory=list)
    cycle_selected_q_values: list[float] = field(default_factory=list)
    cycle_uncertainty_td_scales: list[float] = field(default_factory=list)
    cycle_selection_scores: list[float] = field(default_factory=list)
    cycle_forced_default_actions: list[bool] = field(default_factory=list)
    td_errors: list[float] = field(default_factory=list)
    postfit_td_errors: list[float] = field(default_factory=list)

    def record_update(self, controller: Any, td_error: float) -> None:
        self.td_errors.append(float(td_error))
        if hasattr(controller, "last_postfit_td_error"):
            self.postfit_td_errors.append(float(controller.last_postfit_td_error))

    def record_cycle(
        self,
        *,
        action_index: int,
        action_value: float,
        weight: float,
        residual: float,
        native_cycle_time: float,
        residual_ratio: float | None = None,
        selection: Mapping[str, Any] | None = None,
    ) -> None:
        self.action_counts[str(action_index)] = int(
            self.action_counts.get(str(action_index), 0) + 1
        )
        self.cycle_actions.append(float(weight))
        self.cycle_action_values.append(float(action_value))
        self.cycle_residuals.append(float(residual))
        if residual_ratio is not None:
            self.cycle_residual_ratios.append(float(residual_ratio))
        self.cycle_times.append(float(native_cycle_time))
        if selection is not None:
            self.cycle_explored.append(bool(selection["explored"]))
            self.cycle_greedy_indices.append(int(selection["greedy_index"]))
            self.cycle_epsilons.append(float(selection["epsilon"]))
            uncertainty = selection.get("uncertainty")
            predicted_q = selection.get("q_values")
            scores = selection.get("selection_scores")
            self.cycle_selected_uncertainties.append(
                float(uncertainty[action_index]) if uncertainty is not None else 0.0
            )
            self.cycle_uncertainty_td_scales.append(
                float(selection.get("uncertainty_td_scale", 0.0))
            )
            self.cycle_selected_q_values.append(
                float(predicted_q[action_index])
                if predicted_q is not None else float("nan")
            )
            self.cycle_selection_scores.append(
                float(scores[action_index]) if scores is not None else float("nan")
            )
            self.cycle_forced_default_actions.append(
                bool(selection.get("forced_default_first_action", False))
            )

    def as_result(self) -> dict[str, Any]:
        # Return the original trace lists, preserving the result's references.
        return {
            "action_counts": self.action_counts,
            "cycle_actions": self.cycle_actions,
            "cycle_action_values": self.cycle_action_values,
            "cycle_residuals": self.cycle_residuals,
            "cycle_residual_ratios": self.cycle_residual_ratios,
            "cycle_times": self.cycle_times,
            "cycle_explored": self.cycle_explored,
            "cycle_greedy_indices": self.cycle_greedy_indices,
            "cycle_epsilons": self.cycle_epsilons,
            "cycle_selected_uncertainties": self.cycle_selected_uncertainties,
            "cycle_selected_q_values": self.cycle_selected_q_values,
            "cycle_uncertainty_td_scales": self.cycle_uncertainty_td_scales,
            "cycle_selection_scores": self.cycle_selection_scores,
            "cycle_forced_default_actions": self.cycle_forced_default_actions,
            "mean_abs_td_error": (
                float(np.mean(np.abs(self.td_errors))) if self.td_errors else 0.0
            ),
            "td_errors": self.td_errors,
            "postfit_td_errors": self.postfit_td_errors,
        }


def _run_recovery(
    primary: AttemptOutcome,
    fallback_attempt: Callable[[], Mapping[str, Any] | AttemptOutcome] | None,
    *,
    require_valid_observation: bool,
) -> RecoveryOutcome:
    """Execute the caller-selected fallback exactly once, when present."""
    return RecoveryOutcome(
        primary=primary,
        fallback=(
            None if fallback_attempt is None else execute_attempt(
                fallback_attempt,
                require_valid_observation=require_valid_observation,
            )
        ),
    )


def _terminal_transition_cost(
    native_cycle_cost: float,
    recovery: RecoveryOutcome,
    *,
    retain_failed_episodes: bool,
    failure_penalty_sec: float | None,
) -> float:
    """Charge recovery to the learning target, leaving native targets intact."""
    cost = float(native_cycle_cost)
    if recovery.fallback is not None and (
        retain_failed_episodes or not recovery.unrecovered_failure
    ):
        cost += float(recovery.fallback.end_to_end_runtime_sec)
    if retain_failed_episodes and recovery.unrecovered_failure:
        cost += float(failure_penalty_sec)
    return cost


def _merge_recovery_result(
    outcome: dict[str, Any],
    recovery: RecoveryOutcome | None,
    *,
    learn: bool,
    retain_failed_episodes: bool,
    failure_penalty_sec: float | None,
    deferred_transitions: list[tuple[np.ndarray, int, float]] | None,
) -> dict[str, Any]:
    """Serialize one completed transaction without changing result keys."""
    if recovery is not None:
        original_primary = recovery.primary
        recovery = RecoveryOutcome(
            primary=AttemptOutcome(
                status=original_primary.status,
                setup_runtime_sec=float(outcome["setup_runtime"]),
                solve_runtime_sec=float(outcome["native_solve_runtime"]),
                controller_runtime_sec=float(outcome["infer_runtime"]),
                failure_reason=original_primary.failure_reason,
                residual_norm=float(outcome["residual_norm"]),
                cycles=int(outcome["iterations"]),
                result=dict(outcome),
            ),
            fallback=recovery.fallback,
        )
        recovered_outcome = recovery.to_result()
        recovered_outcome.update({
            key: value for key, value in outcome.items()
            if key not in recovered_outcome
        })
        recovered_outcome["recovery_protocol_applied"] = True
        recovered_outcome["controller_update_committed"] = bool(
            learn and (retain_failed_episodes or not recovery.unrecovered_failure)
        )
        outcome = recovered_outcome
    else:
        failed = bool(outcome["failure_reason"])
        outcome.update({
            "recovery_protocol_applied": False,
            "primary_status": "success" if not failed else "nonconvergence",
            "fallback_used": False,
            "recovered": False,
            "unrecovered_failure": failed,
        })
    if deferred_transitions is not None:
        outcome["_monte_carlo_transitions"] = deferred_transitions
    outcome["failure_feedback_mode"] = (
        "budgeted_penalty" if retain_failed_episodes else "rollback_unrecovered"
    )
    outcome["failure_penalty_sec"] = (
        float(failure_penalty_sec or 0.0)
        if outcome.get("unrecovered_failure", False) else 0.0
    )
    return outcome


def run_td_episode(
    *,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    controller: Any,
    encoder: SolveStateEncoder,
    problem_context: Sequence[float] | None = None,
    solve_tol: float,
    solve_max_cycles: int,
    learn: bool,
    explore: bool,
    epsilon: float | None = None,
    defer_monte_carlo_update: bool = False,
    record_action_metadata: bool = False,
    initial_environment_weight_override: float | None = None,
    fallback_attempt: Callable[[], Mapping[str, Any] | AttemptOutcome] | None = None,
    failure_penalty_sec: float | None = None,
) -> Dict[str, Any]:
    """Run one shared solve episode, including its learning transaction.

    Setup selection and fallback policy are owned by the caller. Native
    costs, controller timings, and recovery costs retain separate accounting.
    """
    failure_penalty_sec = validate_failure_penalty(failure_penalty_sec)
    retain_failed_episodes = failure_penalty_sec is not None
    function_started = time.perf_counter()
    decision_runtime = 0.0
    feature_runtime = 0.0
    update_runtime = 0.0
    lifecycle_runtime = 0.0
    solve_runtime = 0.0
    trace = _EpisodeTrace()
    episode_transitions: list[tuple[np.ndarray, int, float]] = []
    residual = float("inf")
    iterations = 0
    initial_environment_weight = float(controller.initial_environment_weight)
    last_weight = initial_environment_weight
    selection: Dict[str, Any] = {}
    pending_action: tuple[int, float, Dict[str, Any]] | None = None
    recovery: RecoveryOutcome | None = None
    episode_started = False
    learning_snapshot = None
    prep = None
    fast_anchor_tail = bool(
        controller.config.adaptive_cycles is not None
        and int(controller.config.adaptive_cycles) > 0
        and controller.config.alpha == 0.0
        and controller.config.monte_carlo_alpha > 0.0
    )
    try:
        if learn and hasattr(controller, "snapshot_learning_state"):
            lifecycle_started = time.perf_counter()
            try:
                learning_snapshot = controller.snapshot_learning_state()
            finally:
                lifecycle_runtime += time.perf_counter() - lifecycle_started
        with create_env(**dict(mkw)) as env:
            prep = env.prepare_rl(params=augment_setup_params(dict(params)))
            if retain_failed_episodes:
                validate_runtime_cost(prep.setup_runtime_sec)
            initial_residual = float(prep.initial_residual_norm)
            initial_environment_weight = (
                float(prep.initial_relax_weight)
                if initial_environment_weight_override is None
                else float(initial_environment_weight_override)
            )
            residual = initial_residual
            previous_residual = initial_residual
            last_weight = initial_environment_weight
            last_cycle_time = 0.0
            lifecycle_started = time.perf_counter()
            try:
                controller.start_episode(
                    initial_environment_weight=initial_environment_weight,
                )
            finally:
                lifecycle_runtime += time.perf_counter() - lifecycle_started
            episode_started = True
            feature_started = time.perf_counter()
            features = encoder.encode(
                mkw=mkw,
                setup_params=params,
                problem_context=problem_context,
                initial_residual=initial_residual,
                residual=residual,
                previous_residual=previous_residual,
                cycle=0,
                last_weight=last_weight,
                last_cycle_time=last_cycle_time,
            )
            feature_runtime += float(time.perf_counter() - feature_started)
            if retain_failed_episodes and not np.all(np.isfinite(features)):
                raise InvalidObservationError("Nonfinite initial controller features")

            for cycle in range(int(solve_max_cycles)):
                adaptive_decision = bool(
                    not fast_anchor_tail
                    or cycle < int(controller.config.adaptive_cycles)
                )
                if adaptive_decision:
                    if pending_action is None:
                        decision_started = time.perf_counter()
                        action_index, action_value, selection = controller.select_action(
                            features,
                            explore=explore,
                            epsilon=epsilon,
                            cycle=cycle,
                            include_metadata=record_action_metadata,
                        )
                        decision_runtime += float(
                            time.perf_counter() - decision_started
                        )
                    else:
                        action_index, action_value, selection = pending_action
                        pending_action = None
                    weight = controller.environment_weight(
                        int(action_index),
                        float(last_weight),
                    )
                else:
                    action_index = int(controller.anchor_index)
                    action_value = float(controller.weights[action_index])
                    weight = controller.environment_weight(
                        int(action_index),
                        float(last_weight),
                    )
                try:
                    residual_new, cycle_time = env.step_rl(
                        relax_weight=float(weight),
                        sweeps_down=1,
                        sweeps_up=1,
                        tol=float(solve_tol),
                        max_cycles=int(solve_max_cycles),
                    )
                    if retain_failed_episodes:
                        validate_runtime_cost(cycle_time)
                except InvalidObservationError:
                    raise
                except Exception as step_exc:
                    if retain_failed_episodes and not isinstance(step_exc, AMGNativeError):
                        raise
                    failed_step = AttemptOutcome.from_exception(
                        step_exc,
                        elapsed_sec=0.0,
                    )
                    failed_step_runtime = float(failed_step.solve_runtime_sec)
                    if retain_failed_episodes:
                        validate_runtime_cost(failed_step_runtime)
                    solve_runtime += failed_step_runtime
                    iterations = cycle + 1
                    primary_payload = {
                        "runtime": float(prep.setup_runtime_sec + solve_runtime),
                        "setup_runtime": float(prep.setup_runtime_sec),
                        "native_solve_runtime": float(solve_runtime),
                        "solve_runtime": float(solve_runtime),
                        "infer_runtime": float(
                            feature_runtime + decision_runtime + update_runtime + lifecycle_runtime
                        ),
                        "failed": True,
                        "attempt_status": failed_step.status.value,
                        "failure_reason": failed_step.failure_reason,
                        "failure_stage": "solve",
                        "residual_norm": float(residual),
                        "iterations": int(iterations),
                    }
                    recovery = _run_recovery(
                        AttemptOutcome.from_mapping(primary_payload),
                        fallback_attempt,
                        require_valid_observation=retain_failed_episodes,
                    )
                    learning_allowed = bool(retain_failed_episodes or not recovery.unrecovered_failure)
                    transition_cost = _terminal_transition_cost(
                        failed_step_runtime,
                        recovery,
                        retain_failed_episodes=retain_failed_episodes,
                        failure_penalty_sec=failure_penalty_sec,
                    )
                    if learning_allowed:
                        episode_transitions.append(
                            (features.copy(), int(action_index), float(transition_cost))
                        )
                        if learn:
                            update_started = time.perf_counter()
                            td_error = controller.update(
                                features=features,
                                action_index=action_index,
                                cost=transition_cost,
                                next_features=features,
                                terminal=True,
                                next_cycle=iterations,
                                next_action_index=None,
                                native_cycle_cost=None,
                                residual_ratio=None,
                            )
                            update_runtime += float(
                                time.perf_counter() - update_started
                            )
                            trace.record_update(controller, td_error)
                    trace.record_cycle(
                        action_index=action_index,
                        action_value=action_value,
                        weight=weight,
                        residual=residual,
                        native_cycle_time=failed_step_runtime,
                    )
                    break
                cycle_time = float(cycle_time)
                solve_runtime += cycle_time
                iterations = cycle + 1
                native_status = env.last_step.status
                nonfinite_result = not math.isfinite(float(residual_new))
                terminated = native_status is SolveStatus.CONVERGED and not nonfinite_result
                truncated = bool(
                    nonfinite_result or native_status is SolveStatus.MAX_CYCLES
                    or ((not terminated) and iterations >= int(solve_max_cycles))
                )
                need_next_features = bool(
                    not (terminated or truncated)
                    and (
                        not fast_anchor_tail
                        or iterations < int(controller.config.adaptive_cycles)
                    )
                )
                next_features = None
                if need_next_features:
                    feature_started = time.perf_counter()
                    next_features = encoder.encode(
                        mkw=mkw,
                        setup_params=params,
                        problem_context=problem_context,
                        initial_residual=initial_residual,
                        residual=float(residual_new),
                        previous_residual=float(residual),
                        cycle=iterations,
                        last_weight=float(weight),
                        last_cycle_time=cycle_time,
                    )
                    feature_runtime += float(time.perf_counter() - feature_started)
                    if retain_failed_episodes and not np.all(np.isfinite(next_features)):
                        raise InvalidObservationError("Nonfinite successor controller features")

                transition_cost = float(cycle_time)
                residual_ratio = (
                    float(residual_new) / float(residual)
                    if np.isfinite(float(residual_new))
                    and np.isfinite(float(residual))
                    and float(residual) > 0.0
                    else None
                )
                if truncated:
                    primary = AttemptOutcome.from_mapping(
                        {
                            "runtime": float(prep.setup_runtime_sec + solve_runtime),
                            "setup_runtime": float(prep.setup_runtime_sec),
                            "solve_runtime": float(solve_runtime),
                            "infer_runtime": float(
                                feature_runtime + decision_runtime + update_runtime + lifecycle_runtime
                            ),
                            "failed": True,
                            "failure_reason": (
                                "nonfinite_residual" if nonfinite_result
                                else "max_cycles_reached_without_convergence"
                            ),
                            "failure_stage": "solve",
                            "residual_norm": float(residual_new),
                            "iterations": int(iterations),
                        }
                    )
                    recovery = _run_recovery(
                        primary,
                        fallback_attempt,
                        require_valid_observation=retain_failed_episodes,
                    )
                    transition_cost = _terminal_transition_cost(
                        cycle_time,
                        recovery,
                        retain_failed_episodes=retain_failed_episodes,
                        failure_penalty_sec=failure_penalty_sec,
                    )
                next_action_index = None
                if (
                    learn
                    and controller.requires_next_action
                    and not (terminated or truncated)
                ):
                    assert next_features is not None
                    decision_started = time.perf_counter()
                    pending_action = controller.select_action(
                        next_features,
                        explore=explore,
                        epsilon=epsilon,
                        cycle=iterations,
                        include_metadata=record_action_metadata,
                    )
                    decision_runtime += float(
                        time.perf_counter() - decision_started
                    )
                    next_action_index = int(pending_action[0])
                learning_allowed = bool(
                    retain_failed_episodes or recovery is None or not recovery.unrecovered_failure
                )
                if learning_allowed:
                    if adaptive_decision or not episode_transitions:
                        episode_transitions.append((features.copy(), int(action_index), float(transition_cost)))
                    else:
                        tail_features, tail_action, accumulated_cost = episode_transitions[-1]
                        episode_transitions[-1] = (
                            tail_features,
                            tail_action,
                            float(accumulated_cost + transition_cost),
                        )
                if learn and learning_allowed and fast_anchor_tail:
                    controller.steps += 1
                elif learn and learning_allowed:
                    assert next_features is not None or terminated or truncated
                    update_started = time.perf_counter()
                    td_error = controller.update(
                        features=features,
                        action_index=action_index,
                        cost=transition_cost,
                        next_features=(features if next_features is None else next_features),
                        terminal=bool(terminated or truncated),
                        next_cycle=iterations,
                        next_action_index=next_action_index,
                        native_cycle_cost=cycle_time,
                        residual_ratio=residual_ratio,
                    )
                    update_runtime += float(time.perf_counter() - update_started)
                    trace.record_update(controller, td_error)

                trace.record_cycle(
                    action_index=action_index,
                    action_value=action_value,
                    weight=weight,
                    residual=residual_new,
                    native_cycle_time=cycle_time,
                    residual_ratio=residual_ratio,
                    selection=selection if record_action_metadata else None,
                )
                if next_features is not None:
                    features = next_features
                previous_residual = float(residual)
                residual = float(residual_new)
                last_weight = float(weight)
                last_cycle_time = cycle_time
                if terminated or truncated:
                    break
            if (
                learn
                and (retain_failed_episodes or recovery is None or not recovery.unrecovered_failure)
                and controller.config.monte_carlo_alpha > 0.0
                and not defer_monte_carlo_update
            ):
                update_started = time.perf_counter()
                trace.td_errors.extend(controller.monte_carlo_update(episode_transitions))
                update_runtime += float(time.perf_counter() - update_started)
            if recovery is not None and recovery.unrecovered_failure and not retain_failed_episodes:
                if learning_snapshot is not None:
                    lifecycle_started = time.perf_counter()
                    try:
                        controller.restore_learning_state(learning_snapshot)
                    finally:
                        lifecycle_runtime += time.perf_counter() - lifecycle_started
            else:
                update_started = time.perf_counter()
                try:
                    controller.finish_episode(learned=learn)
                finally:
                    update_runtime += float(time.perf_counter() - update_started)

        failure_reason = classify_rl_failure(
            residual_norm=float(residual),
            iterations=int(iterations),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
        )
        outcome = {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "native_solve_runtime": float(solve_runtime),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": float(feature_runtime + decision_runtime + update_runtime + lifecycle_runtime),
            "feature_runtime": float(feature_runtime),
            "decision_runtime": float(decision_runtime),
            "update_runtime": float(update_runtime),
            "lifecycle_runtime": float(lifecycle_runtime),
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "residual_norm": float(residual),
            "iterations": int(iterations),
            "initial_environment_weight": initial_environment_weight,
            "final_w": float(last_weight),
            **trace.as_result(),
            "epsilon": float(controller.epsilon),
            "selection": selection,
            "controller_update_committed": bool(
                learn and (retain_failed_episodes or recovery is None or not recovery.unrecovered_failure)
            ),
        }
        return _merge_recovery_result(
            outcome,
            recovery,
            learn=learn,
            retain_failed_episodes=retain_failed_episodes,
            failure_penalty_sec=failure_penalty_sec,
            deferred_transitions=(
                episode_transitions if defer_monte_carlo_update else None
            ),
        )
    except Exception as exc:
        if isinstance(exc, InvalidObservationError) or (retain_failed_episodes and not isinstance(exc, AMGNativeError)):
            # Broken measurements and estimator/programming errors are not
            # numerical solver failures. Roll back any partial updates and
            # propagate so the experiment cannot keep training on them.
            if learning_snapshot is not None:
                controller.restore_learning_state(learning_snapshot)
            raise
        controller_runtime = float(feature_runtime + decision_runtime + update_runtime + lifecycle_runtime)
        native_failure = AttemptOutcome.from_exception(
            exc,
            elapsed_sec=float(time.perf_counter() - function_started),
        )
        if episode_started and native_failure.status is AttemptStatus.SETUP_FAILURE:
            native_failure = AttemptOutcome(
                status=AttemptStatus.SOLVE_FAILURE,
                setup_runtime_sec=native_failure.setup_runtime_sec,
                solve_runtime_sec=native_failure.solve_runtime_sec,
                controller_runtime_sec=native_failure.controller_runtime_sec,
                failure_reason=native_failure.failure_reason,
                residual_norm=native_failure.residual_norm,
                cycles=native_failure.cycles,
                result=dict(native_failure.result),
            )
        setup_runtime = float(
            (0.0 if prep is None else prep.setup_runtime_sec)
            + native_failure.setup_runtime_sec
        )
        solve_runtime += float(native_failure.solve_runtime_sec)
        primary = AttemptOutcome.from_mapping({
            "runtime": float(setup_runtime + solve_runtime),
            "setup_runtime": setup_runtime,
            "native_solve_runtime": float(solve_runtime),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": controller_runtime,
            "failed": True,
            "attempt_status": native_failure.status.value,
            "failure_reason": native_failure.failure_reason,
            "failure_stage": native_failure.failure_stage,
            "residual_norm": float(residual),
            "iterations": int(iterations),
        })
        setup_construction_failure = bool(
            native_failure.status is AttemptStatus.SETUP_FAILURE
            and not episode_started
        )
        recovery = _run_recovery(
            primary,
            None if setup_construction_failure else fallback_attempt,
            require_valid_observation=retain_failed_episodes,
        )
        if learning_snapshot is not None:
            lifecycle_started = time.perf_counter()
            try:
                controller.restore_learning_state(learning_snapshot)
            finally:
                lifecycle_runtime += time.perf_counter() - lifecycle_started
        elif episode_started:
            update_started = time.perf_counter()
            try:
                controller.finish_episode(learned=False)
            finally:
                update_runtime += time.perf_counter() - update_started
        # Rollback/finalization is completed after the failed attempt and must
        # be charged once, without contaminating the native cycle targets.
        recovery = RecoveryOutcome(
            primary=replace(
                recovery.primary,
                controller_runtime_sec=float(
                    feature_runtime + decision_runtime + update_runtime + lifecycle_runtime
                ),
            ),
            fallback=recovery.fallback,
        )
        outcome = recovery.to_result()
        outcome.update({
            "feature_runtime": float(feature_runtime),
            "decision_runtime": float(decision_runtime),
            "update_runtime": float(update_runtime),
            "lifecycle_runtime": float(lifecycle_runtime),
            "residual_norm": float(residual),
            "iterations": int(iterations),
            "initial_environment_weight": initial_environment_weight,
            "final_w": float("nan"),
            **trace.as_result(),
            "epsilon": float(controller.epsilon),
            "selection": {},
            "recovery_protocol_applied": not setup_construction_failure,
            "controller_update_committed": False,
        })
        return outcome


__all__ = ["run_td_episode"]
