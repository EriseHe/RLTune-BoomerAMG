"""Shared same-context setup reselection and recovery transaction."""

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


@dataclass(frozen=True)
class SetupAttemptRecord:
    params: Dict[str, Any]
    selection_info: Dict[str, Any]
    selection_runtime_sec: float
    outcome: AttemptOutcome
    arm_index: int


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


def _supports_reselection_transaction(model: Any) -> bool:
    return all(
        callable(getattr(model, name, None))
        for name in (
            "begin_recovery_transaction",
            "observe_pending_failure",
            "select_recovery",
            "commit_deferred_observation",
            "commit_recovery_transaction",
            "rollback_recovery_transaction",
        )
    )


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
) -> tuple[AttemptOutcome, RecoveryOutcome | None]:
    attempted = execute_attempt(lambda: solver_fn(dict(params)))
    if bool(attempted.result.get("recovery_protocol_applied", False)):
        recovery = RecoveryOutcome.from_mapping(attempted.result)
        return recovery.primary, recovery
    return attempted, None


def _legacy_single_attempt(
    *,
    policy: Any,
    context: np.ndarray,
    parameter_space: Mapping[str, Any],
    solver_fn: Callable[[Dict[str, Any]], Mapping[str, Any] | AttemptOutcome],
    fallback_solver_fn: Callable[[Dict[str, Any]], Mapping[str, Any] | AttemptOutcome]
    | None,
    default_params: Mapping[str, Any],
    prev_update_est: float,
    primary_is_default: bool,
) -> SetupReselectionResult:
    params, _info, select_sec = _run_selection(
        policy,
        context=context,
        parameter_space=parameter_space,
    )
    primary, embedded = _typed_solver_result(solver_fn, params)
    if embedded is not None:
        recovery = embedded
    else:
        fallback = None
        if (
            not primary.succeeded
            and not primary_is_default
            and fallback_solver_fn is not None
        ):
            fallback = execute_attempt(
                lambda: fallback_solver_fn(dict(default_params))
            )
        recovery = RecoveryOutcome(primary=primary, fallback=fallback)

    loss_started = time.perf_counter_ns()
    loss = float(
        recovery.end_to_end_runtime_sec + select_sec + float(prev_update_est)
    )
    loss_eval_sec = (time.perf_counter_ns() - loss_started) / 1.0e9
    update_committed = bool(
        not primary_is_default
        and not recovery.unrecovered_failure
        and callable(getattr(policy, "update", None))
    )
    update_sec = 0.0
    if update_committed:
        started = time.perf_counter_ns()
        policy.update(
            loss=loss + float(loss_eval_sec),
            context=context,
            params=params,
            outcome=recovery.to_result(),
        )
        update_sec = (time.perf_counter_ns() - started) / 1.0e9
    elif recovery.unrecovered_failure:
        cancel = getattr(_policy_model(policy), "cancel_pending", None)
        if callable(cancel):
            cancel()

    outcome = recovery.to_result()
    outcome.update(
        {
            "bandit_update_committed": update_committed,
            "bandit_observation_count": int(update_committed),
            "controller_update_committed": bool(
                outcome.get("controller_update_committed", False)
            ),
        }
    )
    timing = {
        "select_sec": float(select_sec),
        "loss_eval_sec": float(loss_eval_sec),
        "update_sec": float(update_sec),
        "overhead_sec": float(select_sec + loss_eval_sec + update_sec),
    }
    return SetupReselectionResult(
        params=params,
        outcome=outcome,
        timing=timing,
        fallback_used=int(recovery.fallback_used),
        update_runtime_sec=float(update_sec),
    )


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
) -> SetupReselectionResult:
    """Run up to three learned setup attempts and one default fallback.

    Only setup construction failures trigger another learned selection. Solve
    failures go directly to the default fallback. All learned observations are
    committed together with their realized suffix costs.
    """

    if int(max_learned_attempts) < 1:
        raise ValueError("max_learned_attempts must be at least one")
    model = _policy_model(policy)
    if primary_is_default or not _supports_reselection_transaction(model):
        return _legacy_single_attempt(
            policy=policy,
            context=np.asarray(context, dtype=float),
            parameter_space=parameter_space,
            solver_fn=solver_fn,
            fallback_solver_fn=fallback_solver_fn,
            default_params=default_params,
            prev_update_est=float(prev_update_est),
            primary_is_default=bool(primary_is_default),
        )

    attempts: list[SetupAttemptRecord] = []
    deferred: list[Any] = []
    excluded_arms: list[int] = []
    embedded_fallback: AttemptOutcome | None = None
    transaction_started = False
    observation_update_total_sec = 0.0
    params, info, select_sec = _run_selection(
        policy,
        context=np.asarray(context, dtype=float),
        parameter_space=parameter_space,
    )

    while True:
        arm = _pending_arm(model, info)
        primary, embedded = _typed_solver_result(solver_fn, params)
        attempts.append(
            SetupAttemptRecord(
                params=dict(params),
                selection_info=dict(info),
                selection_runtime_sec=float(select_sec),
                outcome=primary,
                arm_index=int(arm),
            )
        )

        if primary.succeeded and len(attempts) == 1:
            loss_started = time.perf_counter_ns()
            loss = float(
                primary.end_to_end_runtime_sec
                + select_sec
                + float(prev_update_est)
            )
            loss_eval_sec = (time.perf_counter_ns() - loss_started) / 1.0e9
            update_started = time.perf_counter_ns()
            model.update(loss + float(loss_eval_sec), failure_label=0.0)
            update_sec = (time.perf_counter_ns() - update_started) / 1.0e9
            recovery = RecoveryOutcome(primary=primary)
            outcome = recovery.to_result()
            outcome.update(
                {
                    "bandit_update_committed": True,
                    "bandit_observation_count": 1,
                    "controller_update_committed": bool(
                        primary.result.get("controller_update_committed", False)
                    ),
                }
            )
            timing = {
                "select_sec": float(select_sec),
                "loss_eval_sec": float(loss_eval_sec),
                "update_sec": float(update_sec),
                "overhead_sec": float(select_sec + loss_eval_sec + update_sec),
            }
            return SetupReselectionResult(
                params=dict(params),
                outcome=outcome,
                timing=timing,
                fallback_used=0,
                update_runtime_sec=float(update_sec),
            )

        if not transaction_started:
            model.begin_recovery_transaction()
            transaction_started = True
        update_started = time.perf_counter_ns()
        deferred.append(
            model.observe_pending_failure(
                failure_label=0.0 if primary.succeeded else 1.0
            )
        )
        observation_update_sec = (time.perf_counter_ns() - update_started) / 1.0e9
        observation_update_total_sec += float(observation_update_sec)

        if embedded is not None:
            embedded_fallback = embedded.fallback
            break
        if primary.succeeded:
            break
        if (
            primary.status is AttemptStatus.SETUP_FAILURE
            and len(attempts) < int(max_learned_attempts)
        ):
            excluded_arms.append(int(arm))
            params, info, select_sec = _run_reselection(
                model,
                context=np.asarray(context, dtype=float),
                excluded_arms=excluded_arms,
            )
            continue
        break

    fallback = embedded_fallback
    if (
        not attempts[-1].outcome.succeeded
        and embedded_fallback is None
        and fallback_solver_fn is not None
    ):
        fallback = execute_attempt(
            lambda: fallback_solver_fn(dict(default_params))
        )
    recovery = RecoveryOutcome(
        primary_attempts=tuple(record.outcome for record in attempts),
        fallback=fallback,
    )

    loss_started = time.perf_counter_ns()
    suffix = 0.0 if fallback is None else float(fallback.end_to_end_runtime_sec)
    suffix_costs = [0.0] * len(attempts)
    for index in range(len(attempts) - 1, -1, -1):
        record = attempts[index]
        suffix += float(
            record.outcome.end_to_end_runtime_sec
            + record.selection_runtime_sec
            + float(prev_update_est)
        )
        suffix_costs[index] = float(suffix)
    loss_eval_sec = (time.perf_counter_ns() - loss_started) / 1.0e9

    update_sec = float(observation_update_total_sec)
    update_committed = not recovery.unrecovered_failure
    if update_committed:
        update_started = time.perf_counter_ns()
        for observation, suffix_cost in zip(deferred, suffix_costs):
            model.commit_deferred_observation(
                observation,
                loss=float(suffix_cost + loss_eval_sec),
            )
        model.commit_recovery_transaction()
        update_sec += (time.perf_counter_ns() - update_started) / 1.0e9
    else:
        model.rollback_recovery_transaction()

    outcome = recovery.to_result()
    attempt_rows = outcome.get("primary_attempts", [])
    for row, record, suffix_cost in zip(attempt_rows, attempts, suffix_costs):
        row.update(
            {
                "arm_index": int(record.arm_index),
                "params": dict(record.params),
                "selection_info": dict(record.selection_info),
                "selection_runtime": float(record.selection_runtime_sec),
                "suffix_cost": float(suffix_cost + loss_eval_sec),
            }
        )
    outcome.update(
        {
            "primary_attempts": attempt_rows,
            "bandit_update_committed": bool(update_committed),
            "bandit_observation_count": (
                int(len(attempts)) if update_committed else 0
            ),
            "controller_update_committed": bool(
                attempts[-1].outcome.result.get(
                    "controller_update_committed", False
                )
                and update_committed
            ),
        }
    )
    select_total = float(sum(record.selection_runtime_sec for record in attempts))
    timing = {
        "select_sec": select_total,
        "loss_eval_sec": float(loss_eval_sec),
        "update_sec": float(update_sec),
        "overhead_sec": float(select_total + loss_eval_sec + update_sec),
    }
    return SetupReselectionResult(
        params=dict(attempts[-1].params),
        outcome=outcome,
        timing=timing,
        fallback_used=int(recovery.fallback_used),
        update_runtime_sec=float(update_sec),
    )
