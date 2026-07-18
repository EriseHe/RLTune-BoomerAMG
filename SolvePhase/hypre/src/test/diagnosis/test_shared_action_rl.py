from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from online_td_lambda import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    build_action_basis,
)
from shared_action_rl import (
    BootstrapLcbSarsaController,
    BootstrapSarsaSpec,
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
    RecursiveMonteCarloLcbController,
    RecursiveMonteCarloLcbSpec,
    StagewiseLsviLcbController,
    StagewiseLsviLcbSpec,
)


WEIGHTS = tuple(float(value) for value in np.round(np.arange(1.0, 2.01, 0.1), 1))
CENTERS = (1.0, 1.25, 1.5, 1.75, 2.0)


def _config(**overrides: object) -> ExpectedSarsaLambdaConfig:
    values = {
        "weights": WEIGHTS,
        "anchor_weight": 1.0,
        "alpha": 0.001,
        "gamma": 1.0,
        "trace_lambda": 0.8,
        "epsilon_start": 0.3,
        "epsilon_final": 0.03,
        "epsilon_decay_steps": 20_000.0,
        "potential_scale_sec": 0.0,
        "failure_penalty_sec": 0.1,
        "initial_q_sec": 0.0,
        "action_rbf_sigma": 0.2,
        "action_basis_mode": "compact_rbf",
        "action_basis_centers": CENTERS,
        "td_algorithm": "true_online_sarsa",
        "force_default_first_action": False,
    }
    values.update(overrides)
    return ExpectedSarsaLambdaConfig(**values)


class SharedActionFeatureTests(unittest.TestCase):
    def test_compact_rbf_is_normalized_and_local(self) -> None:
        basis = build_action_basis(
            WEIGHTS,
            mode="compact_rbf",
            rbf_sigma=0.2,
            rbf_centers=CENTERS,
        )
        self.assertEqual(basis.shape, (11, 5))
        np.testing.assert_allclose(np.sum(basis, axis=1), 1.0)
        index_1p2 = WEIGHTS.index(1.2)
        near = float(basis[index_1p2] @ basis[WEIGHTS.index(1.3)])
        far = float(basis[index_1p2] @ basis[WEIGHTS.index(1.8)])
        self.assertGreater(near, far)

    def test_shared_sarsa_update_changes_neighbors_more_than_distant_actions(self) -> None:
        learner = ExpectedSarsaLambda(
            feature_dim=1,
            config=_config(alpha=0.1, trace_lambda=0.0),
            seed=11,
        )
        features = np.asarray([1.0])
        learner.start_episode(initial_environment_weight=1.0)
        learner.update(
            features=features,
            action_index=WEIGHTS.index(1.2),
            cost=1.0,
            next_features=features,
            terminal=True,
            next_action_index=None,
        )
        values = learner.q_values(features)
        self.assertGreater(values[WEIGHTS.index(1.3)], 0.0)
        self.assertGreater(
            values[WEIGHTS.index(1.3)],
            values[WEIGHTS.index(1.8)],
        )

    def test_compact_basis_checkpoint_round_trip(self) -> None:
        source = ExpectedSarsaLambda(
            feature_dim=2,
            config=_config(),
            seed=12,
        )
        source.theta[:] = np.arange(source.theta.size).reshape(source.theta.shape)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "shared.npz"
            source.save(path)
            restored = ExpectedSarsaLambda(
                feature_dim=2,
                config=_config(),
                seed=13,
            )
            restored.load(path)
        np.testing.assert_allclose(restored.theta, source.theta)
        np.testing.assert_allclose(restored.action_basis, source.action_basis)


class BootstrapSarsaTests(unittest.TestCase):
    def test_lcb_uses_member_disagreement_and_keeps_actual_action_update(self) -> None:
        controller = BootstrapLcbSarsaController(
            feature_dim=1,
            config=_config(epsilon_start=0.0, epsilon_final=0.0),
            spec=BootstrapSarsaSpec(
                members=3,
                episode_inclusion_probability=1.0,
                uncertainty_beta=1.0,
            ),
            seed=21,
        )
        features = np.asarray([1.0])
        controller.members[0].theta[:, 0] = 0.01
        controller.members[1].theta[:, 0] = 0.02
        controller.members[2].theta[:, 0] = 0.03
        controller.start_episode(initial_environment_weight=1.0)
        action, _weight, metadata = controller.select_action(
            features,
            explore=False,
        )
        self.assertGreater(max(metadata["uncertainty"]), 0.0)
        with self.assertRaises(ValueError):
            controller.update(
                features=features,
                action_index=action,
                cost=0.001,
                next_features=features,
                terminal=False,
                next_action_index=None,
            )

    def test_bootstrap_checkpoint_round_trip(self) -> None:
        source = BootstrapLcbSarsaController(
            feature_dim=2,
            config=_config(),
            spec=BootstrapSarsaSpec(members=3),
            seed=22,
        )
        source.members[1].theta[:] = 0.25
        source.steps = 17
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ensemble.npz"
            source.save(path)
            restored = BootstrapLcbSarsaController(
                feature_dim=2,
                config=_config(),
                spec=BootstrapSarsaSpec(members=3),
                seed=23,
            )
            restored.load(path)
        self.assertEqual(restored.steps, 17)
        np.testing.assert_allclose(restored.members[1].theta, 0.25)


class StagewiseLsviTests(unittest.TestCase):
    def _controller(self, *, horizon: int = 2) -> StagewiseLsviLcbController:
        config = _config(
            weights=(1.0,),
            anchor_weight=1.0,
            action_basis_centers=(1.0,),
            action_rbf_sigma=0.2,
            epsilon_start=0.0,
            epsilon_final=0.0,
        )
        return StagewiseLsviLcbController(
            feature_dim=1,
            config=config,
            spec=StagewiseLsviLcbSpec(
                horizon=horizon,
                ridge=1.0e-6,
                uncertainty_beta=0.0,
                residual_floor_sec=1.0e-6,
                q_max_sec=0.1,
            ),
            seed=31,
        )

    def test_backward_fit_propagates_terminal_cost(self) -> None:
        controller = self._controller(horizon=2)
        features = np.asarray([1.0])
        controller.start_episode(initial_environment_weight=1.0)
        controller.update(
            features=features,
            action_index=0,
            cost=0.001,
            next_features=features,
            terminal=False,
            next_cycle=1,
        )
        controller.update(
            features=features,
            action_index=0,
            cost=0.002,
            next_features=features,
            terminal=True,
            next_cycle=2,
        )
        controller.finish_episode(learned=True)
        self.assertAlmostEqual(
            float(controller.q_values(features, cycle=1)[0]),
            0.002,
            places=5,
        )
        self.assertAlmostEqual(
            float(controller.q_values(features, cycle=0)[0]),
            0.003,
            places=5,
        )

    def test_confidence_width_shrinks_with_repeated_observations(self) -> None:
        controller = self._controller(horizon=1)
        features = np.asarray([1.0])
        before = float(controller._values(features, cycle=0)[1][0])
        for _ in range(5):
            controller.start_episode(initial_environment_weight=1.0)
            controller.update(
                features=features,
                action_index=0,
                cost=0.001,
                next_features=features,
                terminal=True,
                next_cycle=1,
            )
            controller.finish_episode(learned=True)
        after = float(controller._values(features, cycle=0)[1][0])
        self.assertLess(after, before)

    def test_lsvi_checkpoint_round_trip(self) -> None:
        source = self._controller(horizon=1)
        features = np.asarray([1.0])
        source.start_episode(initial_environment_weight=1.0)
        source.update(
            features=features,
            action_index=0,
            cost=0.001,
            next_features=features,
            terminal=True,
            next_cycle=1,
        )
        source.finish_episode(learned=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lsvi.npz"
            source.save(path)
            restored = self._controller(horizon=1)
            restored.load(path)
        np.testing.assert_allclose(restored.theta, source.theta)
        np.testing.assert_allclose(restored.design_inverse, source.design_inverse)
        self.assertEqual(restored.episodes, 1)

    def test_batched_refit_keeps_policy_fixed_between_boundaries(self) -> None:
        config = _config(
            weights=(1.0,),
            anchor_weight=1.0,
            action_basis_centers=(1.0,),
            epsilon_start=0.0,
            epsilon_final=0.0,
        )
        controller = StagewiseLsviLcbController(
            feature_dim=1,
            config=config,
            spec=StagewiseLsviLcbSpec(
                horizon=1,
                ridge=1.0e-6,
                uncertainty_beta=0.0,
                residual_floor_sec=1.0e-6,
                q_max_sec=0.1,
                refit_interval_episodes=2,
            ),
            seed=32,
        )
        features = np.asarray([1.0])
        for episode in range(2):
            controller.start_episode(initial_environment_weight=1.0)
            controller.update(
                features=features,
                action_index=0,
                cost=0.002,
                next_features=features,
                terminal=True,
                next_cycle=1,
            )
            controller.finish_episode(learned=True)
            if episode == 0:
                self.assertAlmostEqual(float(controller.q_values(features)[0]), 0.0)
        self.assertAlmostEqual(float(controller.q_values(features)[0]), 0.002, places=5)
        self.assertEqual(controller.refit_count, 1)


class RecursiveMonteCarloTests(unittest.TestCase):
    def test_full_return_is_assigned_to_each_visited_state(self) -> None:
        config = _config(
            weights=(1.0,),
            anchor_weight=1.0,
            action_basis_centers=(1.0,),
            epsilon_start=0.0,
            epsilon_final=0.0,
        )
        controller = RecursiveMonteCarloLcbController(
            feature_dim=2,
            config=config,
            spec=RecursiveMonteCarloLcbSpec(
                ridge=1.0e-8,
                uncertainty_beta=0.0,
                residual_floor_sec=1.0e-6,
            ),
            seed=41,
        )
        first = np.asarray([1.0, 0.0])
        second = np.asarray([0.0, 1.0])
        controller.start_episode(initial_environment_weight=1.0)
        controller.update(
            features=first,
            action_index=0,
            cost=0.001,
            next_features=second,
            terminal=False,
        )
        controller.update(
            features=second,
            action_index=0,
            cost=0.002,
            next_features=second,
            terminal=True,
        )
        controller.finish_episode(learned=True)
        self.assertAlmostEqual(float(controller.q_values(first)[0]), 0.003, places=5)
        self.assertAlmostEqual(float(controller.q_values(second)[0]), 0.002, places=5)
        self.assertEqual(controller.summary()["stored_transition_count"], 0)


class RecursiveLstdqTests(unittest.TestCase):
    def _controller(self) -> RecursiveLstdqLcbController:
        config = _config(
            weights=(1.0,),
            anchor_weight=1.0,
            action_basis_centers=(1.0,),
            epsilon_start=0.0,
            epsilon_final=0.0,
            trace_lambda=0.8,
        )
        return RecursiveLstdqLcbController(
            feature_dim=2,
            config=config,
            spec=RecursiveLstdqLcbSpec(
                ridge=1.0e-3,
                uncertainty_beta=1.0,
                residual_floor_sec=1.0e-6,
            ),
            seed=51,
        )

    def test_recursive_inverse_matches_accumulated_lstd_system(self) -> None:
        controller = self._controller()
        first = np.asarray([1.0, 0.0])
        second = np.asarray([0.0, 1.0])
        controller.start_episode(initial_environment_weight=1.0)
        controller.update(
            features=first,
            action_index=0,
            cost=0.001,
            next_features=second,
            terminal=False,
            next_action_index=0,
        )
        controller.update(
            features=second,
            action_index=0,
            cost=0.002,
            next_features=second,
            terminal=True,
        )
        controller.finish_episode(learned=True)
        expected = np.linalg.solve(controller.a_matrix, controller.b)
        np.testing.assert_allclose(controller.theta, expected, atol=1.0e-9)
        self.assertEqual(controller.summary()["stored_transition_count"], 0)
        uncertainty = controller._values(first, cycle=0)[1]
        self.assertTrue(np.all(np.isfinite(uncertainty)))


if __name__ == "__main__":
    unittest.main()
