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


def run_unified_case(*, policy, parameter_space, context, solver, fallback_solver,
                     variant, previous_update, before_attempt, recorder=None,
                     snapshot_runtime=0.0):
    """Three/five learned attempts followed by at most one default attempt."""
    model = policy.model
    attempts, parameters, arms, selections, deferred, raw_attempts = [], [], [], [], [], []
    transaction = False
    update_runtime = 0.0
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
            if primary.succeeded and index == 0:
                break
            if not transaction:
                model.begin_recovery_transaction()
                transaction = True
            start = time.perf_counter()
            deferred.append(model.observe_pending_failure(failure_label=float(not primary.succeeded)))
            update_runtime += time.perf_counter() - start
            if primary.succeeded:
                break

        if not attempts[-1].succeeded:
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

        start = time.perf_counter()
        suffix = (0.0 if fallback is None else fallback.end_to_end_runtime_sec) + controller_extra
        costs = [0.0] * len(attempts)
        for i in range(len(attempts) - 1, -1, -1):
            suffix += attempts[i].end_to_end_runtime_sec + selections[i] + previous_update
            costs[i] = float(suffix)
        loss_runtime = time.perf_counter() - start
        committed = not recovery.unrecovered_failure
        start = time.perf_counter()
        if committed:
            if transaction:
                for observation, cost in zip(deferred, costs):
                    model.commit_deferred_observation(observation, loss=cost + loss_runtime)
                model.commit_recovery_transaction()
            else:
                model.update(costs[0] + loss_runtime, failure_label=0.0)
        else:
            model.rollback_recovery_transaction()
        update_runtime += time.perf_counter() - start
    except Exception:
        if transaction:
            model.rollback_recovery_transaction()
        else:
            model.cancel_pending()
        if recorder is not None:
            recorder.controller.restore_learning_state(recorder.snapshot)
        raise

    outcome = recovery.to_result()
    outcome["infer_runtime"] += controller_extra
    outcome["runtime"] += controller_extra
    outcome["solve_runtime"] += controller_extra
    outcome.update(
        failure_feedback_mode="rollback_unrecovered", failure_penalty_sec=0.0,
        bandit_update_committed=committed,
        bandit_observation_count=len(attempts) if committed else 0,
        bandit_learning_cost=costs[-1] + loss_runtime if committed else None,
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
                   suffix_learning_cost=costs[i] + loss_runtime if committed else None,
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
