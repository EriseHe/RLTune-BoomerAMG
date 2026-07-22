from __future__ import annotations

import _project_paths  # noqa: F401

import unittest

import numpy as np

from learners import SharedLinUCB_AMG_v4
from learners.linucb import run_same_context_setup_reselection
from learners.common import ParameterSpaceSpec, ParameterSpec
from utils.setup_amg import build_actions_from_spec


class SharedLinUCBV4Tests(unittest.TestCase):
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
