"""Shared attempt budget and delayed LSTDQ completion-cost feedback for Module 06."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any

import numpy as np

from hypre.bindings import AttemptOutcome, RecoveryOutcome, execute_attempt
from setup.learners.common import AOTCandidateSchedule
from setup.learners.linucb.setup_reselection import SetupReselectionResult
from setup.space import DEFAULT_SETUP_PARAMS
from .protocol import EXTRA_CANDIDATE_SEED_OFFSET


class CandidateRows:
    """Preserve Module 04's first three candidate rows on every problem."""

    def __init__(self, model, *, cases: int, attempts: int, directory: Path):
        self.model = model
        self.base = model._candidate_schedule
        self.extra = None
        if attempts > 3:
            source = self.base
            self.extra = AOTCandidateSchedule(
                directory=directory, catalog=source.catalog,
                rounds=cases * 2, pool_size=source.pool_size,
                seed=source.seed + EXTRA_CANDIDATE_SEED_OFFSET,
                chunk_rounds=source.chunk_rounds,
                sampling_method=source.sampling_method,
                factorized_cache=source.factorized_cache,
                **{key: getattr(source, key) for key in (
                    "structured_anchor_size", "structured_global_size",
                    "structured_local_size", "structured_sobol_size",
                )},
            )

    def before_attempt(self, case_index: int, attempt_index: int):
        if attempt_index < 3:
            schedule, row = self.base, case_index * 3 + attempt_index
        else:
            if self.extra is None or attempt_index >= 5:
                raise ValueError("Attempt exceeds the prepared candidate budget")
            schedule, row = self.extra, case_index * 2 + attempt_index - 3
        self.model._candidate_schedule = schedule
        self.model.set_candidate_schedule_cursor(row)

    def finish_case(self, case_index: int):
        self.model._candidate_schedule = self.base
        self.model.set_candidate_schedule_cursor((case_index + 1) * 3)


@dataclass
class RecordedEpisode:
    attempt_index: int
    initial_weight: float | None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    committed: bool = False


class EpisodeRecorder:
    """Record executed updates so delayed targets replace, rather than duplicate, data."""

    def __init__(self, controller):
        self.controller = controller
        self.snapshot = controller.snapshot_learning_state()
        self.episodes: list[RecordedEpisode] = []
        self.attempt_index = 0
        self.current = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def start_episode(self, *, initial_environment_weight=None):
        self.current = RecordedEpisode(self.attempt_index, initial_environment_weight)
        self.episodes.append(self.current)
        return self.controller.start_episode(
            initial_environment_weight=initial_environment_weight,
        )

    def update(self, **kwargs):
        result = self.controller.update(**kwargs)
        self.current.transitions.append({
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in kwargs.items()
        })
        return result

    def finish_episode(self, *, learned):
        result = self.controller.finish_episode(learned=learned)
        self.current.committed = bool(learned)
        return result

    def restore_learning_state(self, state):
        # run_td_episode may reject an exceptional attempt independently.
        if self.current is not None:
            self.current.committed = False
        return self.controller.restore_learning_state(state)

    def finalize(self, attempts, fallback, *, include_recovery):
        success = attempts[-1].succeeded or (fallback is not None and fallback.succeeded)
        additions = [0.0] * len(attempts)
        if not success:
            self.controller.restore_learning_state(self.snapshot)
            return additions, 0
        episodes = [e for e in self.episodes if e.committed]
        observed = {e.attempt_index for e in episodes}
        suffix = 0.0 if fallback is None else fallback.native_runtime_sec
        for i in range(len(attempts) - 1, -1, -1):
            if not attempts[i].succeeded and i in observed:
                additions[i] = float(suffix) if include_recovery else 0.0
            suffix += attempts[i].native_runtime_sec
        if not any(additions[e.attempt_index] for e in episodes):
            return additions, 0

        # Preserve actual action-selection state and consumed RNG draws. The
        # learning snapshot deliberately excludes the RNG, as in Module 04.
        flags = (self.controller._default_first_action_pending,
                 self.controller.forced_initial_action_count)
        self.controller.restore_learning_state(self.snapshot)
        replayed = 0
        for episode in episodes:
            self.controller.start_episode(initial_environment_weight=episode.initial_weight)
            for transition in episode.transitions:
                target = dict(transition)
                if target["terminal"]:
                    target["cost"] += additions[episode.attempt_index]
                self.controller.update(**target)
                replayed += 1
            self.controller.finish_episode(learned=True)
        (self.controller._default_first_action_pending,
         self.controller.forced_initial_action_count) = flags
        return additions, replayed


def run_fixed_case(solver, recorder, snapshot_runtime, include_recovery, *, fallback_solver=None,
                   max_attempts=3):
    """A bounded number of fresh fixed-setup RL solves, then default fallback."""
    if not isinstance(max_attempts, int) or max_attempts < 1:
        raise ValueError("Fixed-setup attempt budget must be a positive integer")
    attempts, raw_attempts = [], []
    fallback = None
    try:
        for index in range(max_attempts):
            recorder.attempt_index, recorder.current = index, None
            attempt = execute_attempt(solver, require_valid_observation=True)
            attempts.append(attempt)
            raw_attempts.append(dict(attempt.result))
            if attempt.succeeded:
                break
        if not attempts[-1].succeeded and fallback_solver is not None:
            fallback = execute_attempt(fallback_solver, require_valid_observation=True)
        recovery = RecoveryOutcome(primary_attempts=tuple(attempts), fallback=fallback)
        started = time.perf_counter()
        additions, replayed = recorder.finalize(attempts, fallback, include_recovery=include_recovery)
        finalization = time.perf_counter() - started
    except BaseException:
        recorder.controller.restore_learning_state(recorder.snapshot)
        raise
    extra = snapshot_runtime + finalization
    out = recovery.to_result()
    for field in ("runtime", "solve_runtime", "infer_runtime"):
        out[field] += extra
    out.update(
        bandit_update_committed=False, bandit_observation_count=0,
        controller_update_committed=bool(not out["failed"] and any(e.committed for e in recorder.episodes)),
        failure_feedback_mode="rollback_unrecovered", failure_penalty_sec=0.0,
        case_learning_runtime=extra, target_replay_runtime=finalization if replayed else 0.0,
        replayed_transitions=replayed, fixed_setup_retries=len(attempts) - 1,
        learned_reselection_count=0,
    )
    for index, row in enumerate(out["primary_attempts"]):
        row.update(
            rl_episode_recorded=any(e.committed and e.attempt_index == index for e in recorder.episodes),
            rl_recovery_cost=additions[index] if not out["failed"] else None,
            cycle_actions=raw_attempts[index].get("cycle_actions", []),
            cycle_times=raw_attempts[index].get("cycle_times", []),
            cycle_residuals=raw_attempts[index].get("cycle_residuals", []),
        )
    return out


def run_unified_case(*, policy, parameter_space, context, solver, fallback_solver=None,
                     variant, previous_update, before_attempt, recorder=None,
                     snapshot_runtime=0.0, update_attempt=None):
    """Run the shared learned budget, followed by the configured default fallback."""
    if variant.attempts < 1:
        raise ValueError("Attempt budget must be positive")
    if variant.default_fallback and fallback_solver is None:
        raise ValueError("Default fallback protocol requires a fallback solver")
    model = policy.model
    attempts, parameters, arms, selections, costs, targets, raw_attempts = [], [], [], [], [], [], []
    update_runtime = loss_runtime = 0.0
    fallback = None
    try:
        for index in range(variant.attempts):
            start = time.perf_counter()
            before_attempt(index)
            if index == 0:
                params, _ = policy.select(context=context, parameter_space=parameter_space)
            else:
                params, _ = model.select_recovery(context, excluded_arms=arms)
            selections.append(time.perf_counter() - start)
            arm = int(model._last_arm)
            if arm in arms:
                raise AssertionError("Same failed setup selected twice on one problem")
            arms.append(arm)
            parameters.append(dict(params))
            if recorder is not None:
                recorder.attempt_index = index
                recorder.current = None
            primary = execute_attempt(lambda: solver(params), require_valid_observation=True)
            attempts.append(primary)
            raw_attempts.append(dict(primary.result))
            start = time.perf_counter()
            cost = primary.end_to_end_runtime_sec + selections[-1] + previous_update
            target = cost * (1.0 if primary.succeeded else 2.0)
            costs.append(cost)
            targets.append(target)
            loss_runtime += time.perf_counter() - start
            start = time.perf_counter()
            if update_attempt is None:
                model.update(target)
            else:
                update_attempt(target, primary.succeeded)
            update_runtime += time.perf_counter() - start
            if primary.succeeded:
                break

        if not attempts[-1].succeeded and variant.default_fallback:
            fallback = execute_attempt(fallback_solver, require_valid_observation=True)
        recovery = RecoveryOutcome(primary_attempts=tuple(attempts), fallback=fallback)
        start = time.perf_counter()
        additions, replayed = ([0.0] * len(attempts), 0)
        if recorder is not None:
            additions, replayed = recorder.finalize(
                attempts, fallback, include_recovery=variant.rl_recovery_cost,
            )
        finalization_runtime = time.perf_counter() - start if recorder is not None else 0.0
        controller_extra = float(snapshot_runtime + finalization_runtime)

        committed = not recovery.unrecovered_failure
    except Exception:
        model.cancel_pending()
        if recorder is not None:
            recorder.controller.restore_learning_state(recorder.snapshot)
        raise

    outcome = recovery.to_result()
    outcome["infer_runtime"] += controller_extra
    outcome["runtime"] += controller_extra
    outcome["solve_runtime"] += controller_extra
    outcome.update(
        failure_feedback_mode="attempt_2T", failure_penalty_sec=0.0,
        bandit_update_committed=True,
        bandit_observation_count=len(attempts),
        bandit_learning_cost=targets[-1],
        controller_update_committed=bool(committed and recorder is not None
                                        and any(e.committed for e in recorder.episodes)),
        case_learning_runtime=controller_extra,
        target_replay_runtime=finalization_runtime if replayed else 0.0,
        replayed_transitions=replayed,
        selected_arm_index=arms[-1],
    )
    for i, row in enumerate(outcome["primary_attempts"]):
        row.update(arm_index=arms[i], params=parameters[i],
                   selection_runtime=selections[i],
                   attempt_cost=costs[i], learning_cost=targets[i],
                   rl_episode_recorded=bool(recorder is not None and any(
                       e.committed and e.attempt_index == i for e in recorder.episodes)),
                   rl_recovery_cost=additions[i] if committed else None,
                   cycle_actions=raw_attempts[i].get("cycle_actions", []),
                   cycle_times=raw_attempts[i].get("cycle_times", []),
                   cycle_residuals=raw_attempts[i].get("cycle_residuals", []))
    selection_runtime = sum(selections)
    timing = dict(select_sec=selection_runtime, loss_eval_sec=loss_runtime,
                  update_sec=update_runtime,
                  overhead_sec=selection_runtime + loss_runtime + update_runtime)
    return SetupReselectionResult(parameters[-1], outcome, timing,
                                 int(recovery.fallback_used), update_runtime)
