"""Attempt budgets, cost credit, candidate pairing and transactional recovery."""

import copy
import unittest

import numpy as np

from hypre.bindings import AttemptOutcome
from experiments.paper_final.recovery.execution import (
    CandidateRows, EpisodeRecorder, run_unified_case,
)
from experiments.paper_final.recovery.protocol import VARIANTS, select_stress_case
from experiments.paper_final.analyze_06_recovery import audit
from experiments.paper_final.online.case_loop import report_online_outcome
from experiments.paper_final.online.setup_branches import GenericBanditPolicy
from setup.tests import test_linucb
from solve.tests import test_recursive_lstdq


def outcome(status, *, setup=0.01, solve=0.02):
    return dict(attempt_status=status, failed=status != "success", failure_origin="solver",
                setup_runtime=setup, solve_runtime=solve, native_solve_runtime=solve,
                runtime=setup + solve, infer_runtime=0.0,
                residual_norm=1e-8 if status == "success" else 1.0,
                iterations=2 if status == "success" else 50)


class RecoveryStudyTests(unittest.TestCase):
    def test_stress_case_is_maximum_among_all_diffusion_advection_runs(self):
        config, selection = select_stress_case()
        self.assertEqual(config.name, "advection_40_s5.json")
        self.assertEqual(selection["selected"]["joint_unrecovered"], 62)
        self.assertEqual(len(selection["ranking"]), 18)
        self.assertEqual(selection["selected"]["base_seed"], 339923787)

    def _run(self, variant, statuses, *, fallback_success=True):
        model = test_linucb.SharedLinUCBTests._model()
        state = (model.A_inv.copy(), model.b.copy(), model.failure_b.copy())
        seen = []
        def solver(params):
            # Failure feedback must already be available before reselection.
            self.assertEqual(model.failure_observation_count, len(seen))
            seen.append(params["weight"])
            return outcome(statuses[min(len(seen) - 1, len(statuses) - 1)])
        fallback_calls = []
        def fallback():
            fallback_calls.append(True)
            return outcome("success" if fallback_success else "nonconvergence", setup=0.03, solve=0.04)
        result = run_unified_case(
            policy=GenericBanditPolicy(model), parameter_space={}, context=np.array([1.0, 0.5]),
            solver=solver, fallback_solver=fallback, variant=variant, previous_update=0.0,
            before_attempt=lambda _: None,
        )
        reported = report_online_outcome(result.outcome, bandit_timing=result.timing)
        reported["method_wall_runtime"] = 0.0
        audit([dict(problem_index=1, outcome=reported)], variant)
        self.assertEqual(len(seen), len(set(seen)))
        return model, state, result, len(fallback_calls)

    def test_nonconvergence_uses_three_or_five_attempts_then_one_fallback(self):
        for variant in VARIANTS[1:]:
            with self.subTest(variant=variant.name):
                model, _, result, fallbacks = self._run(variant, ["nonconvergence"])
                self.assertEqual(result.outcome["primary_attempt_count"], variant.attempts)
                self.assertEqual(model.failure_observation_count, variant.attempts)
                self.assertEqual(fallbacks, 1)
                self.assertAlmostEqual(result.outcome["native_runtime"], variant.attempts * 0.03 + 0.07)

    def test_mixed_construction_and_solve_failures_share_one_budget(self):
        _, _, result, fallbacks = self._run(VARIANTS[1], ["setup_failure", "nonconvergence", "success"])
        self.assertEqual(result.outcome["primary_attempt_count"], 3)
        self.assertTrue(result.outcome["learned_recovered"])
        self.assertEqual(fallbacks, 0)

    def test_success_stops_immediately(self):
        model, _, result, fallbacks = self._run(VARIANTS[3], ["success"])
        self.assertEqual(result.outcome["primary_attempt_count"], 1)
        self.assertEqual(model.t, 1)
        self.assertEqual(fallbacks, 0)

    def test_unrecovered_problem_rolls_back_all_bandit_observations(self):
        model, state, result, fallbacks = self._run(VARIANTS[3], ["nonconvergence"], fallback_success=False)
        for actual, before in zip((model.A_inv, model.b, model.failure_b), state):
            np.testing.assert_array_equal(actual, before)
        self.assertEqual(model.failure_observation_count, 0)
        self.assertFalse(result.outcome["bandit_update_committed"])
        self.assertEqual(fallbacks, 1)

    def test_candidate_extra_rows_do_not_shift_original_three_row_blocks(self):
        class Schedule:
            def set_cursor(self, value):
                self.cursor = value
        class Model:
            def set_candidate_schedule_cursor(self, value):
                self._candidate_schedule.set_cursor(value)
        rows = CandidateRows.__new__(CandidateRows)
        rows.model, rows.base, rows.extra = Model(), Schedule(), Schedule()
        for case in (0, 7, 4999):
            for attempt in range(5):
                rows.before_attempt(case, attempt)
                schedule = rows.base if attempt < 3 else rows.extra
                self.assertIs(rows.model._candidate_schedule, schedule)
                self.assertEqual(schedule.cursor, 3 * case + attempt if attempt < 3 else 2 * case + attempt - 3)
            rows.finish_case(case)
            self.assertIs(rows.model._candidate_schedule, rows.base)
            self.assertEqual(rows.base.cursor, 3 * (case + 1))


class DelayedTargetTests(unittest.TestCase):
    def _record(self):
        controller = test_recursive_lstdq.RecursiveLstdqEpisodeTests._controller()
        recorder = EpisodeRecorder(controller)
        for episode in range(2):
            recorder.attempt_index = episode
            recorder.start_episode(initial_environment_weight=1.0)
            for k in range(3):
                x = np.array([1.0, 0.2 * (k + 1), 0.4 * (episode + 1)])
                y = np.array([1.0, 0.2 * (k + 2), 0.4 * (episode + 1)])
                recorder.update(features=x, action_index=k % 2, cost=0.002,
                                next_features=y, terminal=k == 2,
                                next_action_index=None if k == 2 else (k + 1) % 2)
            recorder.finish_episode(learned=True)
        return controller, recorder

    def test_delayed_feedback_matches_direct_full_cost_lstdq_without_duplicate_samples(self):
        controller, recorder = self._record()
        attempts = [AttemptOutcome.from_mapping(outcome("nonconvergence")),
                    AttemptOutcome.from_mapping(outcome("success", setup=0.03, solve=0.04))]
        rng = copy.deepcopy(controller.rng.bit_generator.state)
        expected = test_recursive_lstdq.RecursiveLstdqEpisodeTests._controller()
        for episode in recorder.episodes:
            expected.start_episode(initial_environment_weight=episode.initial_weight)
            for transition in episode.transitions:
                data = dict(transition)
                if episode.attempt_index == 0 and data["terminal"]:
                    data["cost"] += 0.07
                expected.update(**data)
            expected.finish_episode(learned=True)
        additions, replayed = recorder.finalize(attempts, None, include_recovery=True)
        self.assertEqual(additions, [0.07, 0.0])
        self.assertEqual(replayed, 6)
        self.assertEqual(controller.steps, 6)
        self.assertEqual(controller.episodes, 2)
        self.assertEqual(controller.rng.bit_generator.state, rng)
        for key in ("a_matrix", "a_inverse", "b", "theta", "episode_moment_covariance"):
            np.testing.assert_array_equal(getattr(controller, key), getattr(expected, key))

    def test_no_recovery_target_leaves_attempt_cost_learning_intact(self):
        controller, recorder = self._record()
        before = controller.snapshot_learning_state()
        attempts = [AttemptOutcome.from_mapping(outcome("nonconvergence")),
                    AttemptOutcome.from_mapping(outcome("success"))]
        additions, replayed = recorder.finalize(attempts, None, include_recovery=False)
        self.assertEqual(additions, [0.0, 0.0])
        self.assertEqual(replayed, 0)
        for key in ("a_matrix", "b", "theta", "episode_moment_covariance"):
            np.testing.assert_array_equal(getattr(controller, key), before[key])

    def test_fallback_and_each_later_attempt_are_charged_exactly_once(self):
        controller, recorder = self._record()
        attempts = [AttemptOutcome.from_mapping(outcome("nonconvergence")),
                    AttemptOutcome.from_mapping(outcome("nonconvergence", setup=0.02, solve=0.03))]
        fallback = AttemptOutcome.from_mapping(outcome("success", setup=0.03, solve=0.04))
        additions, replayed = recorder.finalize(attempts, fallback, include_recovery=True)
        np.testing.assert_allclose(additions, [0.12, 0.07])
        self.assertEqual(controller.steps, 6)
        self.assertEqual(replayed, 6)

    def test_all_failed_attempts_restore_preproblem_rl_state(self):
        controller, recorder = self._record()
        attempts = [AttemptOutcome.from_mapping(outcome("nonconvergence"))] * 2
        recorder.finalize(attempts, attempts[0], include_recovery=True)
        self.assertEqual(controller.steps, 0)
        self.assertEqual(controller.episodes, 0)
        for key in ("a_matrix", "b", "theta", "episode_moment_covariance"):
            np.testing.assert_array_equal(getattr(controller, key), recorder.snapshot[key])


if __name__ == "__main__":
    unittest.main()
