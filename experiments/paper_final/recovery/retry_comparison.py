"""Conservative and failure-aware retries with delayed measured-cost targets.

The production learner remains a single runtime model. The optional failure
target vector exists only in this experiment's failure-aware subclass.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from hypre.bindings import RecoveryOutcome, execute_attempt
from setup.learners.linucb.learner import SharedLinUCB
from setup.learners.linucb.setup_reselection import SetupReselectionResult


RETRY_RULES = {
    "conservative_suffix": "Conservative retries + full remaining-attempt cost",
    "conservative_adjacent": "Conservative retries + adjacent-attempt cost",
    "failure_model_suffix": "Failure-model retries + full remaining-attempt cost",
    "failure_model_adjacent": "Failure-model retries + adjacent-attempt cost",
}


@dataclass(frozen=True)
class DeferredCost:
    phi: np.ndarray
    arm: int
    history_index: int
    provisional_cost: float


class DeferredCostLinUCB(SharedLinUCB):
    def defer_cost(self, succeeded):
        """Count the observation while preserving runtime means until case completion."""
        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("Deferred feedback requires a selected setup")
        phi = self._last_phi.copy()
        observation = DeferredCost(phi, int(self._last_arm), len(self.history)-1,
                                   float(phi @ (self.A_inv @ self.b)))
        self._rank_one_inverse_update(self.A_inv, phi)
        self._after_runtime_precision_update(phi)
        self.b += observation.provisional_cost * phi
        self.t += 1
        self._last_phi = self._last_arm = None
        return observation

    def resolve_cost(self, observation, loss):
        """Replace the provisional target without counting the observation twice."""
        loss = float(loss)
        if not np.isfinite(loss) or loss < 0:
            raise ValueError("Deferred runtime cost must be finite and nonnegative")
        if np.isfinite(self.history[observation.history_index].loss):
            raise RuntimeError("Deferred runtime cost was already resolved")
        self.b += (loss-observation.provisional_cost) * observation.phi
        self._cand.observe(observation.arm, loss)
        self.history[observation.history_index] = replace(
            self.history[observation.history_index], loss=loss)


class ConservativeRetryLinUCB(DeferredCostLinUCB):
    def _recovery_best_locations(
        self, x, *, candidates, candidate_features, runtime_score,
        runtime_mean, runtime_uncert,
    ):
        score = runtime_mean + self._effective_alpha() * runtime_uncert
        return np.flatnonzero(np.isclose(score, np.min(score), atol=1e-12, rtol=0))


class FailureAwareRetryLinUCB(DeferredCostLinUCB):
    """One binary target vector, sharing the runtime model's precision matrix."""

    failure_beta = 2.0  # Setup failure-score coefficient, not RL beta.

    def _recovery_best_locations(
        self, x, *, candidates, candidate_features, runtime_score,
        runtime_mean, runtime_uncert,
    ):
        prediction = self._linear_prediction_subset(
            x, theta=self.A_inv @ self.failure_b, arms=candidates,
            action_features=candidate_features,
        )
        upper = prediction + self.failure_beta * runtime_uncert
        best = int(np.lexsort((runtime_score, upper))[0])
        return np.flatnonzero(
            np.isclose(upper, upper[best], atol=1e-12, rtol=0)
            & np.isclose(runtime_score, runtime_score[best], atol=1e-12, rtol=0)
        )

    def defer_cost(self, succeeded):
        observation = super().defer_cost(succeeded)
        self.failure_b += float(not succeeded) * observation.phi
        return observation

    def _mutable_state_extra_arrays(self):
        return {"retry_failure_b": self.failure_b}

    def _load_mutable_state_extra_arrays(self, payload, *, checkpoint_version):
        vector = np.asarray(payload["retry_failure_b"], dtype=float)
        if vector.shape != self.failure_b.shape or not np.all(np.isfinite(vector)):
            raise ValueError("Invalid retry failure-model checkpoint")
        self.failure_b[:] = vector


def use_retry_rule(model, rule):
    """Adopt a freshly constructed canonical learner without rebuilding inputs."""
    if rule not in RETRY_RULES:
        raise ValueError("Unknown retry rule")
    if model.t or model.history or model._last_phi is not None or np.any(model.b):
        raise ValueError("Each retry rule must start with a fresh learner")
    if not np.array_equal(model.A_inv, np.eye(model.d_phi) / model.l2_reg):
        raise ValueError("Expected untrained setup precision")
    cls = ConservativeRetryLinUCB if rule.startswith("conservative_") else FailureAwareRetryLinUCB
    adapted = object.__new__(cls)
    adapted.__dict__ = model.__dict__.copy()
    if cls is FailureAwareRetryLinUCB:
        adapted.failure_b = np.zeros(model.d_phi, dtype=float)
    return adapted


def run_deferred_case(*, policy, parameter_space, context, solver,
                      fallback_solver, variant, previous_update, before_attempt,
                      feedback_mode, recorder=None, snapshot_runtime=0.0):
    """Resolve adjacent or full remaining costs once, after the case finishes.

    Successful attempts receive their own cost. Executed default fallback time
    belongs to the preceding cost sequence but adds no setup observation.
    All valid setup observations survive final case failure. Optional solve RL
    keeps its existing attempt costs, with no recovery augmentation or replay;
    only RL learning is rolled back when the entire case fails.
    """
    model = policy.model
    if not isinstance(model, DeferredCostLinUCB):
        raise TypeError("Delayed feedback requires a deferred-cost setup learner")
    if feedback_mode not in ("adjacent_delayed", "suffix_delayed"):
        raise ValueError("Unknown delayed-cost rule")
    if variant.rl_recovery_cost or not variant.default_fallback or variant.attempts < 1:
        raise ValueError("Expected learned attempts plus default, without RL recovery cost")
    if recorder is None and snapshot_runtime:
        raise ValueError("A snapshot cost requires an RL recorder")
    attempts, params, arms, selections, costs, observations = [], [], [], [], [], []
    raw_attempts = []
    loss_runtime = update_runtime = 0.0
    fallback = None
    try:
        for index in range(variant.attempts):
            tick = time.perf_counter()
            before_attempt(index)
            if index == 0:
                choice, _ = policy.select(context=context, parameter_space=parameter_space)
            else:
                choice, _ = model.select_recovery(context, excluded_arms=arms)
            selections.append(time.perf_counter()-tick)
            arm = int(model._last_arm)
            if arm in arms:
                raise AssertionError("Same failed setup selected twice on one problem")
            arms.append(arm)
            params.append(dict(choice))
            if recorder is not None:
                recorder.attempt_index, recorder.current = index, None
            attempt = execute_attempt(lambda: solver(choice), require_valid_observation=True)
            attempts.append(attempt)
            raw_attempts.append(dict(attempt.result))
            tick = time.perf_counter()
            costs.append(attempt.end_to_end_runtime_sec + selections[-1] + previous_update)
            loss_runtime += time.perf_counter()-tick
            tick = time.perf_counter()
            observations.append(model.defer_cost(attempt.succeeded))
            update_runtime += time.perf_counter()-tick
            if attempt.succeeded:
                break
        if not attempts[-1].succeeded:
            fallback = execute_attempt(fallback_solver, require_valid_observation=True)
        recovery = RecoveryOutcome(primary_attempts=tuple(attempts), fallback=fallback)
        tick = time.perf_counter()
        sequence = costs + ([] if fallback is None else [fallback.end_to_end_runtime_sec])
        targets = [cost if attempt.succeeded else sum(sequence[i:i+2]
            if feedback_mode == "adjacent_delayed" else sequence[i:])
            for i, (attempt, cost) in enumerate(zip(attempts, costs))]
        loss_runtime += time.perf_counter()-tick
        tick = time.perf_counter()
        for observation, target in zip(observations, targets):
            model.resolve_cost(observation, target)
        update_runtime += time.perf_counter()-tick
        finalization_runtime = 0.0
        if recorder is not None:
            tick = time.perf_counter()
            additions, replayed = recorder.finalize(attempts, fallback, include_recovery=False)
            finalization_runtime = time.perf_counter()-tick
            if replayed or any(additions):
                raise AssertionError("Recovery expense entered the no-recovery RL target")
    except Exception:
        model.cancel_pending()
        if recorder is not None:
            recorder.controller.restore_learning_state(recorder.snapshot)
        raise

    outcome = recovery.to_result()
    controller_extra = float(snapshot_runtime + finalization_runtime)
    for field in ("runtime", "solve_runtime", "infer_runtime"):
        outcome[field] += controller_extra
    outcome.update(failure_feedback_mode=feedback_mode, failure_penalty_sec=0.0,
        bandit_update_committed=True, bandit_observation_count=len(attempts),
        bandit_learning_cost=targets[-1],
        controller_update_committed=bool(not outcome["failed"] and recorder is not None
            and any(e.committed for e in recorder.episodes)),
        case_learning_runtime=controller_extra, target_replay_runtime=0.0, replayed_transitions=0,
        selected_arm_index=arms[-1])
    for i, row in enumerate(outcome["primary_attempts"]):
        row.update(arm_index=arms[i], params=params[i], selection_runtime=selections[i],
            attempt_cost=costs[i], learning_cost=targets[i],
            rl_episode_recorded=bool(recorder is not None and any(
                e.committed and e.attempt_index == i for e in recorder.episodes)),
            rl_recovery_cost=None if outcome["failed"] and recorder is not None else 0.0,
            cycle_actions=raw_attempts[i].get("cycle_actions", []),
            cycle_times=raw_attempts[i].get("cycle_times", []),
            cycle_residuals=raw_attempts[i].get("cycle_residuals", []))
    select_runtime = sum(selections)
    timing = dict(select_sec=select_runtime, loss_eval_sec=loss_runtime, update_sec=update_runtime,
                  overhead_sec=select_runtime+loss_runtime+update_runtime)
    return SetupReselectionResult(params[-1], outcome, timing, int(recovery.fallback_used), update_runtime)
