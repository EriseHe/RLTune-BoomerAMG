from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import tempfile
import unittest
from pathlib import Path

import numpy as np

from setup.learners import SharedLinUCB_AMG_v4
from setup.learners.linucb import run_same_context_setup_reselection
from setup.learners.common import (
    AOTCandidateSchedule,
    CompactActionCatalog,
    FactorizedActionFeatureCache,
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    ParameterSpec,
)
from setup.utils.setup_amg import build_actions_from_spec


class SharedLinUCBV4Tests(unittest.TestCase):
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
            model = SharedLinUCB_AMG_v4(
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

    def test_aot_recovery_reuses_shared_inverse_uncertainty(self) -> None:
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
            model = SharedLinUCB_AMG_v4(
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
            model.begin_recovery_transaction()
            deferred = model.observe_pending_failure(failure_label=1.0)

            original = model._mean_uncertainty_subset
            contraction_calls = 0

            def counted(*args, **kwargs):
                nonlocal contraction_calls
                if kwargs.get("precomputed_uncertainty") is None:
                    contraction_calls += 1
                return original(*args, **kwargs)

            model._mean_uncertainty_subset = counted
            model.select_recovery(context, excluded_arms=[failed_arm])
            self.assertEqual(contraction_calls, 1)
            self.assertEqual(
                model.history[-1].pred_uncert,
                model.history[-1].pred_failure_uncert,
            )
            model.update(0.2)
            model.commit_deferred_observation(deferred, loss=0.3)
            model.commit_recovery_transaction()
            self.assertIsNone(model._g_actions)
            self.assertEqual(schedule.cursor, 2)
            self.assertEqual(len(model._cand._sparse_arm_stats), 2)

    @staticmethod
    def _model(*, seed: int = 7) -> SharedLinUCB_AMG_v4:
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
        return SharedLinUCB_AMG_v4(
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
        model = SharedLinUCB_AMG_v4(
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

    def test_failure_head_shares_information_across_actions(self) -> None:
        model = self._model()
        context = np.asarray([1.0, 0.5])
        model.predict(context)
        failed_arm = int(model._last_arm)
        model.begin_recovery_transaction()
        observation = model.observe_pending_failure(failure_label=1.0)
        model.commit_deferred_observation(observation, loss=0.1)
        model.commit_recovery_transaction()

        _upper, means, _uncertainty = model._failure_subset(
            context,
            arms=np.arange(model.K, dtype=int),
        )
        neighboring = max(0, failed_arm - 1) if failed_arm > 0 else 1
        self.assertGreater(means[failed_arm], 0.0)
        self.assertNotEqual(means[neighboring], 0.0)

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
        suffixes = [
            row["suffix_cost"] for row in result.outcome["primary_attempts"]
        ]
        self.assertGreater(suffixes[0], suffixes[1])
        self.assertGreater(suffixes[1], suffixes[2])

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

    def test_double_failure_rolls_back_learning_but_not_rng(self) -> None:
        model = self._model()

        class Policy:
            def __init__(self, learner):
                self.model = learner

            def select(self, *, context, **_kwargs):
                return self.model.predict(context), {}

        before_inverse = model.A_inv.copy()
        before_b = model.b.copy()
        before_failure_b = model.failure_b.copy()
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
        np.testing.assert_allclose(model.A_inv, before_inverse)
        np.testing.assert_allclose(model.b, before_b)
        np.testing.assert_allclose(model.failure_b, before_failure_b)
        self.assertEqual(model.t, 0)
        self.assertEqual(model.history, [])
        self.assertNotEqual(repr(model.rng.bit_generator.state), before_rng)
        self.assertTrue(result.outcome["unrecovered_failure"])
        self.assertFalse(result.outcome["bandit_update_committed"])


if __name__ == "__main__":
    unittest.main()
