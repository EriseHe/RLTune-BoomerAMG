"""Attempt budgets, cost credit, candidate pairing and transactional recovery."""

import copy
from dataclasses import asdict, replace
import unittest
from unittest.mock import patch

import numpy as np

from hypre.bindings import AttemptOutcome
from experiments.paper_final.recovery.execution import (
    CandidateRows, EpisodeRecorder, run_fixed_case, run_unified_case,
)
from experiments.paper_final.recovery.protocol import VARIANTS, recorded_variants, select_stress_case
from experiments.paper_final.analyze_06_recovery import audit, summarize
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
    def test_immediate_targets_match_direct_ridge_updates_without_future_cost(self):
        model = test_linucb.SharedLinUCBTests._model()
        context = np.array([1.0, 0.5])
        features = []
        targets = [0.2, 0.4, 0.3]

        def before_attempt(index):
            self.assertEqual(model.t, index)
            if index:
                phi = np.asarray(features)
                np.testing.assert_allclose(model.b, phi.T @ targets[:index])
                np.testing.assert_allclose(model.A_inv, np.linalg.inv(
                    np.eye(model.d_phi) + phi.T @ phi), atol=1e-12)

        def solver(_params):
            features.append(model._last_phi.copy())
            index = len(features) - 1
            return outcome("success" if index == 2 else "nonconvergence",
                           setup=0.0, solve=[0.1, 0.2, 0.3][index])

        with patch("time.perf_counter", return_value=0.0):
            result = run_unified_case(
                policy=GenericBanditPolicy(model), parameter_space={}, context=context,
                solver=solver, fallback_solver=lambda: self.fail("Unexpected fallback"),
                variant=VARIANTS[0], previous_update=0.0, before_attempt=before_attempt,
            )
        before_attempt(3)
        self.assertEqual([step.loss for step in model.history], targets)
        self.assertAlmostEqual(result.outcome["native_runtime"], 0.6)

    def test_execution_error_cancels_pending_selection_without_training(self):
        model = test_linucb.SharedLinUCBTests._model()
        before = model.A_inv.copy()
        def invalid(_params):
            raise OSError("Native execution unavailable")
        with self.assertRaisesRegex(OSError, "unavailable"):
            run_unified_case(policy=GenericBanditPolicy(model), parameter_space={},
                context=np.array([1.0, 0.5]), solver=invalid, fallback_solver=invalid,
                variant=VARIANTS[0], previous_update=0.0, before_attempt=lambda _: None)
        self.assertEqual(model.t, 0)
        self.assertIsNone(model._last_phi)
        self.assertEqual(model.history, [])
        np.testing.assert_array_equal(model.A_inv, before)

    def test_stress_case_is_maximum_among_all_diffusion_advection_runs(self):
        config, selection = select_stress_case()
        self.assertEqual(config.name, "advection_40_s5.json")
        self.assertEqual(selection["selected"]["joint_unrecovered"], 62)
        self.assertEqual(len(selection["ranking"]), 18)
        self.assertEqual(selection["selected"]["base_seed"], 339923787)

    def _run(self, variant, statuses, *, fallback_success=True):
        model = test_linucb.SharedLinUCBTests._model()
        state = (model.A_inv.copy(), model.b.copy())
        seen = []
        def solver(params):
            # Failure feedback must already be available before reselection.
            self.assertEqual(model.t, len(seen))
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

    def test_third_failed_attempt_stops_without_calling_successful_fallback(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant.name):
                model, _, result, fallbacks = self._run(
                    replace(variant, default_fallback=False), ["nonconvergence"] * 3 + ["success"],
                )
                self.assertEqual(result.outcome["primary_attempt_count"], 3)
                self.assertTrue(result.outcome["failed"])
                self.assertFalse(result.outcome["fallback_used"])
                self.assertEqual(model.t, 3)
                self.assertEqual(fallbacks, 0)
                self.assertAlmostEqual(result.outcome["native_runtime"], 0.09)

    def test_three_failed_learned_attempts_use_one_default_fallback(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant.name):
                model, _, result, fallbacks = self._run(
                    variant, ["nonconvergence"] * 3 + ["success"],
                )
                self.assertEqual(result.outcome["primary_attempt_count"], 3)
                self.assertEqual(fallbacks, 1)
                self.assertTrue(result.outcome["fallback_used"])
                self.assertFalse(result.outcome["failed"])
                self.assertEqual(model.t, 3)
                self.assertTrue(result.outcome["bandit_update_committed"])
                self.assertAlmostEqual(result.outcome["native_runtime"], 0.16)

    def test_mixed_construction_and_solve_failures_share_one_budget(self):
        _, _, result, fallbacks = self._run(VARIANTS[0], ["setup_failure", "nonconvergence", "success"])
        self.assertEqual(result.outcome["primary_attempt_count"], 3)
        self.assertTrue(result.outcome["learned_recovered"])
        self.assertEqual(fallbacks, 0)

    def test_success_stops_immediately(self):
        model, _, result, fallbacks = self._run(VARIANTS[0], ["success"])
        self.assertEqual(result.outcome["primary_attempt_count"], 1)
        self.assertEqual(model.t, 1)
        self.assertEqual(fallbacks, 0)

    def test_second_attempt_success_ends_the_case(self):
        model, _, result, fallbacks = self._run(VARIANTS[1], ["nonconvergence", "success"])
        self.assertEqual(result.outcome["primary_attempt_count"], 2)
        self.assertFalse(result.outcome["failed"])
        self.assertEqual(model.t, 2)
        self.assertEqual(fallbacks, 0)

    def test_failed_problem_retains_valid_bandit_observations(self):
        model, state, result, fallbacks = self._run(VARIANTS[1], ["nonconvergence"], fallback_success=False)
        for actual, before in zip((model.A_inv, model.b), state):
            self.assertFalse(np.array_equal(actual, before))
        self.assertEqual(model.t, 3)
        self.assertTrue(result.outcome["bandit_update_committed"])
        self.assertEqual(fallbacks, 1)
        self.assertTrue(result.outcome["failed"])
        self.assertAlmostEqual(result.outcome["native_runtime"], 0.16)

    def test_historical_fallback_is_explicit_and_audited_from_saved_protocol(self):
        historical = asdict(replace(VARIANTS[0], attempts=5))
        historical.pop("default_fallback")
        legacy = recorded_variants({"variants": [historical]})[0]
        self.assertTrue(legacy.default_fallback)
        _, _, result, fallbacks = self._run(legacy, ["nonconvergence"])
        self.assertEqual(result.outcome["primary_attempt_count"], 5)
        self.assertEqual(fallbacks, 1)
        self.assertFalse(result.outcome["failed"])
        strict = recorded_variants({"schema_version": 2, "variants": [asdict(replace(VARIANTS[0], default_fallback=False))]})[0]
        self.assertFalse(strict.default_fallback)
        current = recorded_variants({"schema_version": 3, "variants": [asdict(VARIANTS[0])]})[0]
        self.assertTrue(current.default_fallback)
        row = dict(problem_index=1, outcome=report_online_outcome(result.outcome, bandit_timing=result.timing))
        with self.assertRaisesRegex(ValueError, "Default fallback exceeds"):
            audit([row], replace(legacy, default_fallback=False))

    def test_comparison_summary_calls_final_outcomes_failures(self):
        _, _, result, _ = self._run(VARIANTS[0], ["nonconvergence"], fallback_success=False)
        reported = report_online_outcome(result.outcome, bandit_timing=result.timing)
        reported["method_wall_runtime"] = 0.0
        summary = summarize([dict(outcome=reported)])
        self.assertEqual(summary["failures"], 1)
        self.assertNotIn("unrecovered", summary)

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

    def test_fixed_setup_rolls_back_rl_when_all_three_attempts_and_fallback_fail(self):
        for include_recovery in (True, False):
            with self.subTest(include_recovery=include_recovery):
                controller = test_recursive_lstdq.RecursiveLstdqEpisodeTests._controller()
                recorder = EpisodeRecorder(controller)
                calls = []
                def solver():
                    calls.append(True)
                    recorder.start_episode(initial_environment_weight=1.0)
                    x = np.array([1.0, 0.2, 0.4])
                    recorder.update(features=x, action_index=0, cost=0.02,
                                    next_features=x, terminal=True)
                    recorder.finish_episode(learned=True)
                    return outcome("nonconvergence" if len(calls) <= 3 else "success")
                fallback_calls = []
                def fallback():
                    fallback_calls.append(True)
                    return outcome("nonconvergence", setup=0.03, solve=0.04)
                result = run_fixed_case(solver, recorder, 0.0, include_recovery, fallback_solver=fallback)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(fallback_calls), 1)
                self.assertTrue(result["failed"])
                self.assertTrue(result["fallback_used"])
                self.assertFalse(result["controller_update_committed"])
                self.assertEqual(controller.steps, 0)
                self.assertEqual(controller.episodes, 0)
                self.assertAlmostEqual(result["native_runtime"], 0.16)

    def test_fixed_setup_fallback_success_retains_rl_and_only_cost_variant_replays(self):
        for include_recovery in (True, False):
            with self.subTest(include_recovery=include_recovery):
                controller = test_recursive_lstdq.RecursiveLstdqEpisodeTests._controller()
                recorder = EpisodeRecorder(controller)
                calls, fallback_calls = [], []
                def solver():
                    calls.append(True)
                    recorder.start_episode(initial_environment_weight=1.0)
                    x = np.array([1.0, 0.2, 0.4])
                    recorder.update(features=x, action_index=0, cost=0.02,
                                    next_features=x, terminal=True)
                    recorder.finish_episode(learned=True)
                    return outcome("nonconvergence")
                def fallback():
                    fallback_calls.append(True)
                    return outcome("success", setup=0.03, solve=0.04)
                result = run_fixed_case(solver, recorder, 0.0, include_recovery, fallback_solver=fallback)
                self.assertEqual(len(calls), 3)
                self.assertEqual(len(fallback_calls), 1)
                self.assertFalse(result["failed"])
                self.assertTrue(result["controller_update_committed"])
                self.assertEqual(controller.steps, 3)
                self.assertEqual(controller.episodes, 3)
                self.assertEqual(result["replayed_transitions"], 3 if include_recovery else 0)
                expected = [0.13, 0.10, 0.07] if include_recovery else [0.0] * 3
                np.testing.assert_allclose([a["rl_recovery_cost"] for a in result["primary_attempts"]], expected)
                self.assertAlmostEqual(result["native_runtime"], 0.16)

    def test_fixed_setup_second_attempt_success_keeps_both_episodes(self):
        controller = test_recursive_lstdq.RecursiveLstdqEpisodeTests._controller()
        recorder = EpisodeRecorder(controller)
        calls = []
        def solver():
            calls.append(True)
            recorder.start_episode(initial_environment_weight=1.0)
            x = np.array([1.0, 0.2, 0.4])
            recorder.update(features=x, action_index=0, cost=0.02,
                            next_features=x, terminal=True)
            recorder.finish_episode(learned=True)
            return outcome("nonconvergence" if len(calls) == 1 else "success")
        result = run_fixed_case(solver, recorder, 0.0, True)
        self.assertEqual(len(calls), 2)
        self.assertFalse(result["failed"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(controller.steps, 2)
        self.assertEqual(controller.episodes, 2)
        self.assertEqual(result["replayed_transitions"], 2)
        self.assertAlmostEqual(result["primary_attempts"][0]["rl_recovery_cost"], 0.03)


if __name__ == "__main__":
    unittest.main()
