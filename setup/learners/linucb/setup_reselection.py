"""Same-context setup retries with immediate per-attempt runtime feedback."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from hypre.bindings import (
    AttemptOutcome,
    AttemptStatus,
    RecoveryOutcome,
    execute_attempt,
)
from hypre.bindings.recovery import validate_failure_penalty


@dataclass(frozen=True)
class SetupReselectionResult:
    params: Dict[str, Any]
    outcome: Dict[str, Any]
    timing: Dict[str, float]
    fallback_used: int
    update_runtime_sec: float


def _selected_params_and_info(selected: Any) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(selected, tuple):
        params, info = selected
        return dict(params), dict(info)
    return dict(selected), {}


def _policy_model(policy: Any) -> Any:
    return getattr(policy, "model", getattr(policy, "m", policy))


def _pending_arm(model: Any, info: Mapping[str, Any]) -> int:
    arm = getattr(model, "_last_arm", None)
    if arm is not None:
        return int(arm)
    return int(info.get("arm_index", -1))


def _run_selection(
    policy: Any,
    *,
    context: np.ndarray,
    parameter_space: Mapping[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], float]:
    started = time.perf_counter_ns()
    selected = policy.select(context=context, parameter_space=parameter_space)
    elapsed = (time.perf_counter_ns() - started) / 1.0e9
    params, info = _selected_params_and_info(selected)
    return params, info, float(elapsed)


def _run_reselection(
    model: Any,
    *,
    context: np.ndarray,
    excluded_arms: Sequence[int],
) -> tuple[Dict[str, Any], Dict[str, Any], float]:
    started = time.perf_counter_ns()
    selected = model.select_recovery(
        np.asarray(context, dtype=float),
        excluded_arms=tuple(int(arm) for arm in excluded_arms),
    )
    elapsed = (time.perf_counter_ns() - started) / 1.0e9
    params, info = _selected_params_and_info(selected)
    return params, info, float(elapsed)


def _typed_solver_result(
    solver_fn: Callable[[Dict[str, Any]], Mapping[str, Any] | AttemptOutcome],
    params: Dict[str, Any],
    *,
    require_valid_observation: bool = False,
) -> tuple[AttemptOutcome, RecoveryOutcome | None]:
    attempted = execute_attempt(
        lambda: solver_fn(dict(params)), require_valid_observation=require_valid_observation,
    )
    if bool(attempted.result.get("recovery_protocol_applied", False)):
        recovery = RecoveryOutcome.from_mapping(attempted.result)
        return recovery.primary, recovery
    return attempted, None


def run_same_context_setup_reselection(
    *,
    policy: Any,
    parameter_space: Mapping[str, Any],
    context: np.ndarray,
    solver_fn: Callable[[Dict[str, Any]], Mapping[str, Any] | AttemptOutcome],
    fallback_solver_fn: Callable[[Dict[str, Any]], Mapping[str, Any] | AttemptOutcome]
    | None,
    default_params: Mapping[str, Any],
    prev_update_est: float,
    primary_is_default: bool = False,
    max_learned_attempts: int = 3,
    failure_penalty_sec: float | None = None,
) -> SetupReselectionResult:
    """Update each setup immediately with its own T on success, 2T on failure.

    Module 1 retries construction failures with another learned setup; solve
    failures go to default. Module 06 uses the same feedback with a shared
    three-attempt budget for both kinds of failure. The optional final-failure
    penalty configures the solve controller only, not the setup target.
    """
    if int(max_learned_attempts) < 1:
        raise ValueError("max_learned_attempts must be at least one")
    validate_failure_penalty(failure_penalty_sec)
    model = _policy_model(policy)
    can_reselect = not primary_is_default and callable(getattr(model, "select_recovery", None))
    can_update = not primary_is_default and callable(getattr(model, "update", None))
    attempts, rows, arms = [], [], []
    select_total = loss_total = update_total = 0.0
    fallback = embedded = None
    params, info, select_sec = _run_selection(
        policy, context=context, parameter_space=parameter_space,
    )
    try:
        while True:
            arm = _pending_arm(model, info)
            primary, embedded = _typed_solver_result(
                solver_fn, params, require_valid_observation=True,
            )
            attempts.append(primary)
            arms.append(arm)
            select_total += select_sec
            started = time.perf_counter_ns()
            cost = primary.end_to_end_runtime_sec + select_sec + float(prev_update_est)
            target = cost * (1.0 if primary.succeeded else 2.0)
            loss_total += (time.perf_counter_ns() - started) / 1.0e9
            if can_update:
                started = time.perf_counter_ns()
                if can_reselect:
                    model.update(target)
                else:
                    policy.update(loss=target, context=context, params=params, outcome=primary.result)
                update_total += (time.perf_counter_ns() - started) / 1.0e9
            rows.append(dict(arm_index=arm, params=dict(params), selection_info=dict(info),
                             selection_runtime=select_sec, attempt_cost=cost,
                             learning_cost=target if can_update else None))
            if (embedded is not None or primary.succeeded or not can_reselect
                or primary.status is not AttemptStatus.SETUP_FAILURE
                or len(attempts) >= int(max_learned_attempts)):
                break
            params, info, select_sec = _run_reselection(model, context=context, excluded_arms=arms)
        if embedded is not None:
            fallback = embedded.fallback
        elif not attempts[-1].succeeded and not primary_is_default and fallback_solver_fn is not None:
            fallback = execute_attempt(lambda: fallback_solver_fn(dict(default_params)),
                                       require_valid_observation=True)
    except Exception:
        cancel = getattr(model, "cancel_pending", None)
        if callable(cancel):
            cancel()
        raise

    recovery = RecoveryOutcome(primary_attempts=tuple(attempts), fallback=fallback)
    outcome = recovery.to_result()
    for row, credit in zip(outcome["primary_attempts"], rows):
        row.update(credit)
    outcome.update(
        selected_arm_index=arms[-1], failure_feedback_mode="attempt_2T",
        failure_penalty_sec=0.0, bandit_learning_cost=rows[-1]["learning_cost"],
        bandit_update_committed=can_update,
        bandit_observation_count=len(attempts) if can_update else 0,
        controller_update_committed=bool(not outcome["failed"] and
            attempts[-1].result.get("controller_update_committed", False)),
    )
    timing = dict(select_sec=select_total, loss_eval_sec=loss_total, update_sec=update_total,
                  overhead_sec=select_total + loss_total + update_total)
    return SetupReselectionResult(dict(params), outcome, timing, int(recovery.fallback_used), update_total)
