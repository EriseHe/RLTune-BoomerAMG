from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import tempfile
import itertools
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from setup.learners import SharedLinUCB
from setup.learners.linucb import run_same_context_setup_reselection
from setup.learners.common import (
    AOTCandidateSchedule,
    CompactActionCatalog,
    FactorizedActionFeatureCache,
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    ParameterSpec,
)
from setup.space import build_actions_from_spec
from hypre.bindings.recovery import InvalidObservationError


class SharedLinUCBTests(unittest.TestCase):
    def test_failure_updates_twice_own_cost_with_matching_precision(self):
        model = self._model()
        context = np.asarray([1.0, 0.5])
        selected = {}

        class Policy:
            def __init__(self, learner):
                self.model = learner

            def select(self, *, context, **kwargs):
                params = self.model.predict(context)
                selected.update(
                    arm=self.model._last_arm, phi=self.model._last_phi.copy()
                )
                return params, {}

        failed = dict(
            attempt_status="nonconvergence",
            setup_runtime=0.01,
            solve_runtime=0.04,
            failed=True,
            residual_norm=1.0,
            iterations=50,
        )
        precision_before = np.linalg.inv(model.A_inv)
        with patch("time.perf_counter", side_effect=itertools.count(step=1.0)):
            result = run_same_context_setup_reselection(
                policy=Policy(model),
                parameter_space={},
                context=context,
                solver_fn=lambda _: dict(failed),
                fallback_solver_fn=lambda _: dict(failed),
                default_params={"weight": 1.5},
                prev_update_est=0.005,
                failure_penalty_sec=0.5,
            )
        phi = selected["phi"]
        loss = 2 * (0.05 + result.timing["select_sec"] + 0.005)
        np.testing.assert_allclose(
            model.A_inv, np.linalg.inv(precision_before + np.outer(phi, phi))
        )
        np.testing.assert_allclose(model.b, loss * phi)
        self.assertEqual(model.t, 1)
        self.assertTrue(result.outcome["unrecovered_failure"])
        self.assertTrue(result.outcome["bandit_update_committed"])
        self.assertAlmostEqual(result.outcome["native_runtime"], 0.1)
        self.assertAlmostEqual(result.outcome["bandit_learning_cost"], loss)
        self.assertEqual(result.outcome["selected_arm_index"], selected["arm"])
        # Verify the failure cost reaches the score used for ordinary choices.
        _score, means, _width = model._selection_score_subset(
            context,
            arms=np.array([selected["arm"]]),
            alpha=0.0,
        )
        self.assertAlmostEqual(means[0], float(phi @ model.A_inv @ (loss * phi)))
        self.assertGreater(means[0], float(phi @ model.A_inv @ ((loss / 2) * phi)))

    def test_invalid_fallback_does_not_erase_valid_primary_feedback(self):
        for mode in (0.0, "invalid"):
            with self.subTest(mode=mode):
                model = self._model()

                class Policy:
                    def __init__(self, learner):
                        self.model = learner

                    def select(self, *, context, **kwargs):
                        return self.model.predict(context), {}

                failed = dict(
                    attempt_status="nonconvergence",
                    setup_runtime=0.01,
                    solve_runtime=0.04,
                    failed=True,
                    residual_norm=1.0,
                )
                inverse = model.A_inv.copy()
                with patch("time.perf_counter", side_effect=itertools.count(step=1.0)):
                    kwargs = dict(
                        policy=Policy(model),
                        parameter_space={},
                        context=np.asarray([1.0, 0.5]),
                        solver_fn=lambda _: dict(failed),
                        fallback_solver_fn=lambda _: {
                            **failed,
                            "solve_runtime": float("nan")
                            if mode == "invalid"
                            else 0.04,
                        },
                        default_params={"weight": 1.5},
                        prev_update_est=0.0,
                        failure_penalty_sec=0.0,
                    )
                    if mode == "invalid":
                        with self.assertRaises(InvalidObservationError):
                            run_same_context_setup_reselection(**kwargs)
                        self.assertFalse(np.array_equal(model.A_inv, inverse))
                        self.assertEqual(model.t, 1)
                    else:
                        result = run_same_context_setup_reselection(**kwargs)
                        self.assertTrue(result.outcome["bandit_update_committed"])
                        self.assertEqual(result.outcome["failure_penalty_sec"], 0.0)

    def test_selected_arm_and_new_observation_survive_final_failure(self):
        model = self._model()
        context = np.asarray([1.0, 0.5])
        model.predict(context)
        previous_arm = model._last_arm
        model.update(0.1)
        current_arm = (previous_arm + 1) % model.K
        model._initial_guess_arm = current_arm
        model.initial_guess_rounds = model.t + 1

        class Policy:
            def __init__(self, learner):
                self.model = learner

            def select(self, *, context, **kwargs):
                return self.model.predict(context), {}

        failed = dict(
            attempt_status="nonconvergence",
            setup_runtime=0.01,
            solve_runtime=0.04,
            failed=True,
            residual_norm=1.0,
        )
        result = run_same_context_setup_reselection(
            policy=Policy(model),
            parameter_space={},
            context=context,
            solver_fn=lambda _: dict(failed),
            fallback_solver_fn=lambda _: dict(failed),
            default_params={"weight": 1.5},
            prev_update_est=0.0,
        )
        self.assertEqual(model.history[-2].arm_index, previous_arm)
        self.assertEqual(model.history[-1].arm_index, current_arm)
        self.assertEqual(result.outcome["selected_arm_index"], current_arm)
        self.assertEqual(
            result.outcome["primary_attempts"][-1]["arm_index"], current_arm
        )
        self.assertEqual(result.params, dict(model.actions[current_arm]))

    def test_structured_aot_balances_agg_state_and_builds_local_neighbors(self) -> None:
        parameter_spec = ParameterSpaceSpec(
            parameters=(
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.5, 2.0, 2.5),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
                ParameterSpec(
                    name="agg_num_levels",
                    kind="integer",
                    values=(0, 1, 2),
                    default=0,
                    center=0.0,
                    scale=1.0,
                ),
                ParameterSpec(
                    name="coarsen_type",
                    kind="categorical",
                    values=(0, 8, 10),
                    default=10,
                ),
                ParameterSpec(
                    name="interp_type",
                    kind="categorical",
                    values=(0, 6),
                    default=6,
                ),
            )
        )
        catalog = CompactActionCatalog(parameter_spec)
        encoder = GenericActionFeatureEncoder(parameter_spec)
        cache = FactorizedActionFeatureCache(catalog, encoder)
        with tempfile.TemporaryDirectory() as directory:
            schedule = AOTCandidateSchedule(
                directory=Path(directory),
                catalog=catalog,
                rounds=4,
                pool_size=16,
                seed=17,
                sampling_method="structured512",
                factorized_cache=cache,
                structured_anchor_size=4,
                structured_global_size=4,
                structured_local_size=4,
                structured_sobol_size=4,
            )
            first = schedule.next_arms()
            self.assertEqual(np.unique(first).size, 16)
            values, _active = catalog.decode_parameter_arrays(first)
            np.testing.assert_array_equal(
                np.sort(values["agg_num_levels"][:4] == 0),
                np.asarray([False, False, True, True]),
            )
            np.testing.assert_array_equal(
                np.sort(values["agg_num_levels"][4:8] == 0),
                np.asarray([False, False, True, True]),
            )

            schedule.set_cursor(0)
            model = SharedLinUCB(
                catalog,
                context_dim=2,
                parameter_spec=parameter_spec,
                context_interaction_indices=(0, 1),
                candidate_pool_size=16,
                candidate_strategy="structured",
                always_include_arms=[catalog.default_arm_index],
                elite_cache_size=4,
                candidate_schedule=schedule,
                action_feature_cache=cache,
                seed=19,
            )
            model.predict(np.asarray([1.0, 0.5]))
            stats = model.candidate_stats_history[-1]
            self.assertEqual(stats["strategy"], "structured_aot")
            self.assertEqual(stats["candidate_count"], 16)
            self.assertGreater(stats["local"], 0)
            anchors = model._structured_anchor_arms(4)
            first_neighbors = model._local_neighbor_arms(
                anchors, include_categorical=True, radius=2
            )
            cached_count = len(model._local_neighbor_cache)
            second_neighbors = model._local_neighbor_arms(
                anchors, include_categorical=True, radius=2
            )
            np.testing.assert_array_equal(first_neighbors, second_neighbors)
            self.assertEqual(len(model._local_neighbor_cache), cached_count)
            model.update(0.1)

    def test_aot_retry_scores_the_single_updated_cost_model(self) -> None:
        parameter_spec = ParameterSpaceSpec(
            parameters=(
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.5, 2.0),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
                ParameterSpec(
                    name="kind",
                    kind="categorical",
                    values=(0, 1, 2),
                    default=1,
                ),
            )
        )
        catalog = CompactActionCatalog(parameter_spec)
        encoder = GenericActionFeatureEncoder(parameter_spec)
        cache = FactorizedActionFeatureCache(catalog, encoder)
        with tempfile.TemporaryDirectory() as directory:
            schedule = AOTCandidateSchedule(
                directory=Path(directory),
                catalog=catalog,
                rounds=4,
                pool_size=4,
                seed=23,
            )
            model = SharedLinUCB(
                catalog,
                context_dim=2,
                parameter_spec=parameter_spec,
                context_interaction_indices=(0, 1),
                candidate_pool_size=4,
                always_include_arms=[catalog.default_arm_index],
                elite_cache_size=2,
                candidate_schedule=schedule,
                action_feature_cache=cache,
                seed=29,
            )
            context = np.asarray([1.0, 0.5])
            model.predict(context)
            failed_arm = int(model._last_arm)
            model.update(0.3)

            original = model._mean_uncertainty_subset
            contraction_calls = 0

            def counted(*args, **kwargs):
                nonlocal contraction_calls
                contraction_calls += 1
                return original(*args, **kwargs)

            model._mean_uncertainty_subset = counted
            model.select_recovery(context, excluded_arms=[failed_arm])
            self.assertEqual(contraction_calls, 1)
            model.update(0.2)
            self.assertIsNone(model._g_actions)
            self.assertEqual(schedule.cursor, 2)
            self.assertEqual(len(model._cand._sparse_arm_stats), 2)

    @staticmethod
    def _model(*, seed: int = 7) -> SharedLinUCB:
        parameter_spec = ParameterSpaceSpec(
            parameters=(
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.25, 1.5, 1.75, 2.0),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
            )
        )
        actions = build_actions_from_spec(parameter_spec)
        return SharedLinUCB(
            actions,
            context_dim=2,
            parameter_spec=parameter_spec,
            context_interaction_indices=(0, 1),
            seed=seed,
        )

    def test_materializes_the_shared_action_feature_table(self) -> None:
        parameter_spec = ParameterSpaceSpec(
            parameters=(
                ParameterSpec(
                    name="weight",
                    kind="continuous",
                    values=(1.0, 1.5, 2.0),
                    default=1.5,
                    center=1.5,
                    scale=0.5,
                ),
            )
        )
        actions = build_actions_from_spec(parameter_spec)
        model = SharedLinUCB(
            actions,
            context_dim=2,
            parameter_spec=parameter_spec,
            context_interaction_indices=(0, 1),
            candidate_pool_size=2,
            seed=7,
        )

        self.assertIsInstance(model.actions, list)
        self.assertIsInstance(model._g_actions, np.ndarray)
        self.assertEqual(model._g_actions.shape[0], len(actions))

    def test_retry_uses_updated_cost_score_and_excludes_tried_arm(self):
        model = self._model()
        context = np.array([1.0, 0.5])
        model.predict(context)
        failed_arm = model._last_arm
        model.update(0.2)
        candidates = np.setdiff1d(np.arange(model.K), [failed_arm])
        scores, _, _ = model._selection_score_subset(context, arms=candidates, alpha=model._effective_alpha())
        model.select_recovery(context, excluded_arms=[failed_arm])
        self.assertNotEqual(model._last_arm, failed_arm)
        selected = np.flatnonzero(candidates == model._last_arm)[0]
        self.assertAlmostEqual(scores[selected], min(scores))

    def test_setup_failures_reselect_at_most_three_learned_arms(self) -> None:
        model = self._model()

        class Policy:
            def __init__(self, learner):
                self.model = learner

            def select(self, *, context, **_kwargs):
                return self.model.predict(context), {}

        calls = []

        def fail_setup(params):
            calls.append(dict(params))
            return {
                "attempt_status": "setup_failure",
                "setup_runtime": 0.01,
                "solve_runtime": 0.0,
                "failed": True,
                "failure_stage": "setup",
            }

        result = run_same_context_setup_reselection(
            policy=Policy(model),
            parameter_space={},
            context=np.asarray([1.0, 0.5]),
            solver_fn=fail_setup,
            fallback_solver_fn=lambda _params: {
                "attempt_status": "success",
                "residual_norm": 1e-8,
                "setup_runtime": 0.02,
                "solve_runtime": 0.03,
                "failed": False,
            },
            default_params={"weight": 1.5},
            prev_update_est=0.0,
            max_learned_attempts=3,
        )

        self.assertEqual(len(calls), 3)
        self.assertEqual(len({row["weight"] for row in calls}), 3)
        self.assertEqual(result.outcome["primary_attempt_count"], 3)
        self.assertEqual(result.outcome["bandit_observation_count"], 3)
        self.assertTrue(result.outcome["fallback_used"])
        for row in result.outcome["primary_attempts"]:
            self.assertAlmostEqual(row["learning_cost"], 2 * row["attempt_cost"])
            self.assertLess(row["attempt_cost"], 0.05)

    def test_solve_failure_does_not_reselect_learned_setup(self) -> None:
        model = self._model()

        class Policy:
            def __init__(self, learner):
                self.model = learner

            def select(self, *, context, **_kwargs):
                return self.model.predict(context), {}

        calls = 0

        def fail_solve(_params):
            nonlocal calls
            calls += 1
            return {
                "attempt_status": "nonconvergence",
                "setup_runtime": 0.01,
                "solve_runtime": 0.04,
                "failed": True,
                "failure_stage": "solve",
                "iterations": 50,
                "residual_norm": 1.0,
            }

        result = run_same_context_setup_reselection(
            policy=Policy(model),
            parameter_space={},
            context=np.asarray([1.0, 0.5]),
            solver_fn=fail_solve,
            fallback_solver_fn=lambda _params: {
                "attempt_status": "success",
                "residual_norm": 1e-8,
                "setup_runtime": 0.02,
                "solve_runtime": 0.03,
                "failed": False,
            },
            default_params={"weight": 1.5},
            prev_update_est=0.0,
        )
        self.assertEqual(calls, 1)
        self.assertEqual(result.outcome["primary_attempt_count"], 1)
        self.assertTrue(result.outcome["fallback_used"])

    def test_final_failure_keeps_all_three_valid_setup_updates(self) -> None:
        model = self._model()

        class Policy:
            def __init__(self, learner):
                self.model = learner

            def select(self, *, context, **_kwargs):
                return self.model.predict(context), {}

        before_inverse = model.A_inv.copy()
        before_b = model.b.copy()
        before_rng = repr(model.rng.bit_generator.state)
        result = run_same_context_setup_reselection(
            policy=Policy(model),
            parameter_space={},
            context=np.asarray([1.0, 0.5]),
            solver_fn=lambda _params: {
                "attempt_status": "setup_failure",
                "setup_runtime": 0.01,
                "solve_runtime": 0.0,
                "failed": True,
                "failure_stage": "setup",
            },
            fallback_solver_fn=lambda _params: {
                "attempt_status": "nonconvergence",
                "setup_runtime": 0.02,
                "solve_runtime": 0.05,
                "failed": True,
                "failure_stage": "solve",
            },
            default_params={"weight": 1.5},
            prev_update_est=0.0,
        )
        self.assertFalse(np.array_equal(model.A_inv, before_inverse))
        self.assertFalse(np.array_equal(model.b, before_b))
        self.assertEqual(model.t, 3)
        self.assertEqual(len(model.history), 3)
        self.assertNotEqual(repr(model.rng.bit_generator.state), before_rng)
        self.assertTrue(result.outcome["unrecovered_failure"])
        self.assertTrue(result.outcome["bandit_update_committed"])


if __name__ == "__main__":
    unittest.main()
