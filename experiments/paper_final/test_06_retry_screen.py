"""Numerical and protocol checks for the isolated delayed-cost retry comparison."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.paper_final import run_06_retry_screen as screen
from experiments.paper_final.common.artifacts import read
from experiments.paper_final.online.case_loop import report_online_outcome
from experiments.paper_final.online.setup_branches import GenericBanditPolicy
from experiments.paper_final.recovery.retry_comparison import (
    RETRY_RULES, FailureAwareRetryLinUCB, use_retry_rule, run_deferred_case,
)
from experiments.paper_final.test_06_recovery import outcome
from setup.tests import test_linucb


EXPECTED_FEEDBACK = {
    "conservative_suffix": "suffix_delayed",
    "conservative_adjacent": "adjacent_delayed",
    "failure_model_suffix": "suffix_delayed",
    "failure_model_adjacent": "adjacent_delayed",
}


def model_for(rule):
    return use_retry_rule(test_linucb.SharedLinUCBTests._model(), rule)


def learn(model, target, succeeded):
    observation = model.defer_cost(succeeded)
    model.resolve_cost(observation, target)


class RetryRuleTests(unittest.TestCase):
    def test_first_choices_and_cost_models_match_on_identical_observations(self):
        models = [model_for(rule) for rule in RETRY_RULES]
        for context, target, succeeded in [([1, .2], .4, False), ([1, .5], .2, True), ([1, .8], .6, False)]:
            choices = [m.predict(context) for m in models]
            self.assertTrue(all(choice == choices[0] for choice in choices))
            for model in models:
                learn(model, target, succeeded)
            for model in models[1:]:
                np.testing.assert_array_equal(model.A_inv, models[0].A_inv)
                np.testing.assert_array_equal(model.b, models[0].b)

    def test_conservative_retry_avoids_the_uncertain_optimistic_choice(self):
        data = dict(candidates=np.arange(2), candidate_features=None,
            runtime_mean=np.array([.4, .55]), runtime_uncert=np.array([.8, .05]),
            runtime_score=np.array([-.4, .5]))
        for rule in ("conservative_suffix", "conservative_adjacent"):
            model = model_for(rule)
            self.assertEqual(model._recovery_best_locations(np.ones(2), **data).tolist(), [1])
            self.assertFalse(hasattr(model, "failure_b"))
            self.assertEqual(model._mutable_state_extra_arrays(), {})

    def test_failure_retry_prioritizes_risk_and_uses_cost_only_for_ties(self):
        data = dict(candidates=np.arange(3), candidate_features=None,
            runtime_mean=np.array([.01, .5, .4]), runtime_uncert=np.array([.01, .1, .1]),
            runtime_score=np.array([0, .4, .3]))
        for rule in ("failure_model_suffix", "failure_model_adjacent"):
            model = model_for(rule)
            with patch.object(model, "_linear_prediction_subset", return_value=np.array([.8, .1, .1])):
                self.assertEqual(model._recovery_best_locations(np.ones(2), **data).tolist(), [2])

    def test_deferred_observation_preserves_runtime_mean_and_updates_precision_once(self):
        for rule in RETRY_RULES:
            with self.subTest(rule=rule):
                model = model_for(rule)
                model.predict([1, .2]); learn(model, .4, True)
                theta = model.A_inv @ model.b
                model.predict([1, .7]); phi = model._last_phi.copy()
                inverse = model.A_inv.copy()
                observation = model.defer_cost(False)
                np.testing.assert_allclose(model.A_inv @ model.b, theta, atol=1e-12)
                np.testing.assert_allclose(model.A_inv,
                    np.linalg.inv(np.linalg.inv(inverse)+np.outer(phi, phi)), atol=1e-12)
                if isinstance(model, FailureAwareRetryLinUCB):
                    np.testing.assert_allclose(model.failure_b, phi)
                precision = model.A_inv.copy()
                model.resolve_cost(observation, .6)
                np.testing.assert_array_equal(model.A_inv, precision)
                self.assertEqual(model.t, 2)
                with self.assertRaisesRegex(RuntimeError, "already resolved"):
                    model.resolve_cost(observation, .6)

    def test_measured_targets_batch_ridge_and_retention_for_all_outcomes(self):
        # Times are .1, .2, .3 for learned attempts and .4 for fallback.
        scenarios = (
            (["success"], None, 3, [.1], [.1]),
            (["nonconvergence", "success"], None, 3, [.3, .2], [.3, .2]),
            (["nonconvergence", "nonconvergence", "success"], None, 3, [.3, .5, .3], [.6, .5, .3]),
            (["nonconvergence"]*2, "success", 2, [.3, .6], [.7, .6]),
            (["nonconvergence"]*2, "nonconvergence", 2, [.3, .6], [.7, .6]),
            (["nonconvergence"]*3, "success", 3, [.3, .5, .7], [1., .9, .7]),
            (["nonconvergence"]*3, "nonconvergence", 3, [.3, .5, .7], [1., .9, .7]),
        )
        for rule in RETRY_RULES:
            mode = EXPECTED_FEEDBACK[rule]
            for statuses, fallback_status, budget, adjacent, suffix in scenarios:
                with self.subTest(rule=rule, statuses=statuses, fallback=fallback_status):
                    model = model_for(rule)
                    phis = []
                    variant = screen.retry_variant(budget)
                    expected = suffix if mode == "suffix_delayed" else adjacent
                    def before(index):
                        self.assertEqual(model.t, index)
                        np.testing.assert_allclose(model.A_inv @ model.b, 0, atol=1e-12)
                        self.assertTrue(all(np.isnan(step.loss) for step in model.history))
                        if phis and isinstance(model, FailureAwareRetryLinUCB):
                            np.testing.assert_allclose(model.failure_b, np.sum(phis, axis=0))
                    def solve(_params):
                        index = len(phis)
                        phis.append(model._last_phi.copy())
                        return outcome(statuses[index], setup=0, solve=.1*(index+1))
                    def default():
                        # Cost labels are still pending even when the default solve begins.
                        self.assertTrue(all(np.isnan(step.loss) for step in model.history))
                        return outcome(fallback_status or "success", setup=0, solve=.4)
                    fallback = Mock(side_effect=default)
                    with patch("time.perf_counter", return_value=0):
                        result = run_deferred_case(policy=GenericBanditPolicy(model),
                            parameter_space={}, context=[1, .5], solver=solve, fallback_solver=fallback,
                            variant=variant, previous_update=0, before_attempt=before, feedback_mode=mode)
                    np.testing.assert_allclose([a["learning_cost"] for a in result.outcome["primary_attempts"]], expected)
                    phi = np.asarray(phis)
                    np.testing.assert_allclose(model.b, phi.T @ expected, atol=1e-12)
                    np.testing.assert_allclose(model.A_inv,
                        np.linalg.inv(np.eye(model.d_phi)+phi.T @ phi), atol=1e-12)
                    if isinstance(model, FailureAwareRetryLinUCB):
                        np.testing.assert_allclose(model.failure_b,
                            phi.T @ [float(s != "success") for s in statuses])
                    self.assertEqual(model.t, len(statuses))
                    self.assertEqual(fallback.call_count, int(fallback_status is not None))
                    self.assertEqual(result.outcome["failed"], fallback_status == "nonconvergence")
                    self.assertEqual(len({a["arm_index"] for a in result.outcome["primary_attempts"]}), len(statuses))
                    self.assertAlmostEqual(result.outcome["runtime"],
                        sum(.1*(i+1) for i in range(len(statuses)))+(.4 if fallback_status else 0))
                    row = dict(problem_index=1, rl_active=False,
                        outcome=report_online_outcome(result.outcome, bandit_timing=result.timing))
                    screen.sweep.audit_case(row, screen.BETA, variant=variant, feedback_mode=mode)
                    changed = copy.deepcopy(row)
                    changed["outcome"]["primary_attempts"][0]["learning_cost"] += .1
                    with self.assertRaisesRegex(ValueError, "attempt-cost"):
                        screen.sweep.audit_case(changed, screen.BETA, variant=variant, feedback_mode=mode)

    def test_invalid_execution_never_updates_any_target(self):
        for rule in RETRY_RULES:
            model = model_for(rule)
            with self.assertRaisesRegex(OSError, "unavailable"):
                run_deferred_case(policy=GenericBanditPolicy(model), parameter_space={}, context=[1, .5],
                    solver=Mock(side_effect=OSError("unavailable")), fallback_solver=Mock(),
                    variant=screen.sweep.VARIANT, previous_update=0, before_attempt=lambda _: None,
                    feedback_mode=EXPECTED_FEEDBACK[rule])
            self.assertEqual(model.t, 0)
            self.assertFalse(np.any(model.b))
            self.assertIsNone(model._last_phi)
            if isinstance(model, FailureAwareRetryLinUCB):
                self.assertFalse(np.any(model.failure_b))

    def test_failure_vector_round_trips_in_experimental_checkpoint(self):
        for rule in ("failure_model_suffix", "failure_model_adjacent"):
            model = model_for(rule)
            model.predict([1, .5]); learn(model, .4, False)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/"bandit.npz"
                model.save_mutable_state(path)
                restored = model_for(rule); restored.load_mutable_state(path)
                np.testing.assert_array_equal(restored.failure_b, model.failure_b)
                np.testing.assert_array_equal(restored.A_inv, model.A_inv)
                np.testing.assert_array_equal(restored.b, model.b)
                self.assertEqual(restored.t, 1)

    def test_experimental_rules_cannot_adopt_a_trained_prefix(self):
        model = test_linucb.SharedLinUCBTests._model()
        model.predict([1, .5]); model.update(.2)
        with self.assertRaisesRegex(ValueError, "fresh"):
            use_retry_rule(model, "conservative_suffix")

    def test_variants_have_independent_learning_state(self):
        models = [model_for(rule) for rule in RETRY_RULES]
        for model in models[1:]:
            for key in ("A_inv", "b"):
                np.testing.assert_array_equal(getattr(models[0], key), getattr(model, key))
                self.assertFalse(np.shares_memory(getattr(models[0], key), getattr(model, key)))
        first, second = models[2:]
        self.assertFalse(np.shares_memory(first.failure_b, second.failure_b))
        first.predict([1, .5]); learn(first, .4, False)
        self.assertEqual(first.t, 1)
        self.assertEqual(second.t, 0)
        self.assertFalse(np.any(second.failure_b))


class RetryScreenTests(unittest.TestCase):
    def test_prepare_freezes_paired_seed_budget_and_only_requested_branches_without_solves(self):
        for seed, budget in ((5, 3), (6, 2), (6, 3), (7, 3), (8, 3), (9, 3)):
            with self.subTest(seed=seed, budget=budget), tempfile.TemporaryDirectory() as temporary, \
                 patch.object(screen, "source_hashes", return_value={}), \
                 patch.object(screen.sweep, "native_hashes", return_value={}), \
                 patch.object(screen.subprocess, "check_output", return_value="test"), \
                 patch("hypre.bindings.create_env", side_effect=AssertionError("Preparation solved")), \
                 patch("solve.core.episode.run_td_episode", side_effect=AssertionError("RL activated")):
                directory = Path(temporary)/"prepared"
                screen.prepare(directory, seed=seed, learned_attempts=budget)
                protocol = read(directory/"protocol.json")
                seed_source = read(screen.SEED_CONFIGS[seed])
                self.assertEqual(protocol["cases"], 1000)
                self.assertEqual(protocol["seed"], seed)
                self.assertEqual(protocol["learned_attempts"], budget)
                self.assertEqual(protocol["total_attempt_budget"], budget+1)
                self.assertEqual(protocol["feedback"], EXPECTED_FEEDBACK)
                self.assertEqual(set(protocol["branches"]), set(EXPECTED_FEEDBACK) | {"default"})
                self.assertEqual(protocol["rl_beta"], 5)
                self.assertEqual(protocol["rl_steps_expected"], 0)
                self.assertEqual(len((directory/"inputs.jsonl").read_text().splitlines()), 1000)
                self.assertEqual(protocol["full_stream_sha256"] == screen.FULL_STREAM_HASH, seed == 5)
                for name in screen.BRANCHES:
                    config = read(directory/"configs"/f"{name}.json")
                    self.assertEqual(config["seeds"], seed_source["seeds"])
                    for key in ("seed_groups", "shuffle_seeds"):
                        self.assertEqual(config["stream"][key], seed_source["stream"][key])
                    self.assertEqual(config["stream"]["expected_sha256"], protocol["full_stream_sha256"])
                    self.assertEqual(config["stream"]["cases"], 5000)
                    self.assertEqual(config["setup"]["candidate_schedule"]["max_selections_per_case"], 3)
                    self.assertFalse((directory/name).exists())

    def test_two_attempts_update_the_normal_model_and_fallback_adds_no_observation(self):
        from problems.heterogeneous_3d import context_from_matrix_kwargs
        mkw = dict(nx=60, ny=60, nz=60, stencil=102, rhs_type=1, rhs_seed=42, k=.01, c=0, a0=0, a1=0, a2=0, a3=0)
        for rule in RETRY_RULES:
            for final_status in ("success", "nonconvergence"):
                with self.subTest(rule=rule, final=final_status):
                    model = model_for(rule)
                    runtime = SimpleNamespace(args=SimpleNamespace(problem="rs_convection_3d_type2", tol=1e-6, max_cycles=50),
                        method=SimpleNamespace(setup_kind="linucb", setup_context="rs_diffusion"),
                        branch=SimpleNamespace(policy=GenericBanditPolicy(model), parameter_space={}),
                        stream=[(mkw, context_from_matrix_kwargs(mkw))]*1000, name=rule, rows=Mock())
                    phis = []
                    def solve(**_kwargs):
                        if len(phis) < 2:
                            phis.append(model._last_phi.copy())
                            return outcome("nonconvergence", setup=0, solve=.1*len(phis))
                        self.assertEqual(model.t, 2)
                        return outcome(final_status, setup=0, solve=.4)
                    with patch("time.perf_counter", return_value=0), \
                         patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case", side_effect=solve) as native:
                        row, _ = screen.execute_setup_case(runtime, 999, 0, variant=screen.retry_variant(2))
                    self.assertEqual(native.call_count, 3)
                    self.assertEqual(model.t, 2)
                    self.assertEqual(runtime.rows.before_attempt.call_args_list,
                        [unittest.mock.call(999, 0), unittest.mock.call(999, 1)])
                    expected = [.7, .6] if EXPECTED_FEEDBACK[rule] == "suffix_delayed" else [.3, .6]
                    np.testing.assert_allclose(model.b, np.asarray(phis).T @ expected, atol=1e-12)

    def test_rl_never_runs_including_at_input_1000(self):
        from problems.heterogeneous_3d import context_from_matrix_kwargs
        mkw = dict(nx=60, ny=60, nz=60, stencil=102, rhs_type=1, rhs_seed=42, k=.01, c=0, a0=0, a1=0, a2=0, a3=0)
        for rule in RETRY_RULES:
            model = model_for(rule)
            runtime = SimpleNamespace(args=SimpleNamespace(problem="rs_convection_3d_type2", tol=1e-6, max_cycles=50),
                method=SimpleNamespace(setup_kind="linucb", setup_context="rs_diffusion"),
                branch=SimpleNamespace(policy=GenericBanditPolicy(model), parameter_space={}),
                stream=[(mkw, context_from_matrix_kwargs(mkw))]*1001, name=rule, rows=Mock())
            with patch("solve.core.episode.run_td_episode", side_effect=AssertionError("RL activated")), \
                 patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case",
                    side_effect=[outcome("nonconvergence")]*3+[outcome("success")]) as native:
                row, _ = screen.execute_setup_case(runtime, 999, 0)
                self.assertEqual(native.call_count, 4)
                self.assertFalse(row["rl_active"])
                self.assertFalse(row["shared_prefix"])
                self.assertEqual(model.t, 3)
            with self.assertRaisesRegex(ValueError, "joint/RL"):
                screen.execute_setup_case(runtime, 1000, 0)

    def test_pairing_rejects_reused_state_wrong_feedback_and_missing_default(self):
        protocol = dict(cases=1000, native_sha256={}, full_stream_sha256=screen.FULL_STREAM_HASH,
            feedback=EXPECTED_FEEDBACK.copy(), branches=screen.BRANCHES,
            preparation_checks={n: dict(inputs_sha256="inputs") for n in screen.BRANCHES})
        state = dict(cases=1000, controller_steps=0, bandit_observations=0, full_stream_sha256=screen.FULL_STREAM_HASH,
            native_sha256={}, inputs_sha256="inputs", candidate_cursor=0, candidate_sha256="candidates",
            initial_bandit_sha256="bandit", initial_controller_sha256="controller")
        states = {n: copy.deepcopy(state) for n in screen.BRANCHES}
        for key in ("candidate_sha256", "initial_bandit_sha256", "initial_controller_sha256"):
            states["default"][key] = None
        screen.verify_pairing(states, protocol)
        wrong = copy.deepcopy(protocol)
        wrong["feedback"]["failure_model_suffix"] = "adjacent_delayed"
        with self.assertRaisesRegex(ValueError, "configured separately"):
            screen.verify_pairing(states, wrong)
        states["conservative_suffix"]["candidate_cursor"] = 3000
        with self.assertRaisesRegex(ValueError, "prefix"):
            screen.verify_pairing(states, protocol)
        del states["default"]
        with self.assertRaisesRegex(ValueError, "plus default"):
            screen.verify_pairing(states, protocol)


if __name__ == "__main__":
    unittest.main()
