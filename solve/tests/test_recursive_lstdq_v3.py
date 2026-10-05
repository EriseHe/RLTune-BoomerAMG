from __future__ import annotations

from solve.tests import _project_paths  # noqa: F401

import tempfile
import unittest
from pathlib import Path

import numpy as np

from solve.controllers.recursive_lstdq import (
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV3LcbController,
    RecursiveLstdqV3LcbSpec,
)
from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig


def _config(
    *,
    weights: tuple[float, ...] = (1.0, 1.5),
    centers: tuple[float, ...] = (1.0, 1.5),
) -> ExpectedSarsaLambdaConfig:
    return ExpectedSarsaLambdaConfig(
        weights=weights,
        anchor_weight=weights[0],
        alpha=0.001,
        gamma=1.0,
        trace_lambda=0.8,
        epsilon_start=0.0,
        epsilon_final=0.0,
        epsilon_decay_steps=20_000.0,
        initial_q_sec=0.0,
        action_rbf_sigma=0.2,
        action_basis_mode="compact_rbf",
        action_basis_centers=centers,
        td_algorithm="true_online_sarsa",
        force_default_first_action=False,
    )


class RecursiveLstdqV3Tests(unittest.TestCase):
    @staticmethod
    def _controllers() -> tuple[
        RecursiveLstdqLcbController,
        RecursiveLstdqV3LcbController,
    ]:
        config = _config()
        base = RecursiveLstdqLcbController(
            feature_dim=3,
            config=config,
            spec=RecursiveLstdqLcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
            ),
            seed=71,
        )
        v3 = RecursiveLstdqV3LcbController(
            feature_dim=3,
            config=config,
            spec=RecursiveLstdqV3LcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
            ),
            seed=71,
        )
        return base, v3

    def test_mean_update_is_identical_to_numerical_base(self) -> None:
        base, v3 = self._controllers()
        rng = np.random.default_rng(7101)
        for _episode in range(4):
            base.start_episode(initial_environment_weight=1.0)
            v3.start_episode(initial_environment_weight=1.0)
            features = rng.normal(size=3)
            for cycle in range(6):
                terminal = cycle == 5
                next_features = rng.normal(size=3)
                action = int(rng.integers(0, 2))
                next_action = None if terminal else int(rng.integers(0, 2))
                kwargs = {
                    "features": features,
                    "action_index": action,
                    "cost": float(rng.uniform(1.0e-3, 5.0e-3)),
                    "next_features": next_features,
                    "terminal": terminal,
                    "next_action_index": next_action,
                }
                self.assertAlmostEqual(base.update(**kwargs), v3.update(**kwargs))
                for name in ("a_matrix", "a_inverse", "b", "theta", "trace"):
                    np.testing.assert_allclose(
                        getattr(base, name),
                        getattr(v3, name),
                        rtol=2.0e-12,
                        atol=2.0e-14,
                    )
                features = next_features
            base.finish_episode(learned=True)
            v3.finish_episode(learned=True)
            np.testing.assert_allclose(
                base.theta,
                v3.theta,
                rtol=2.0e-12,
                atol=2.0e-14,
            )

    def test_one_cluster_moment_is_committed_per_episode(self) -> None:
        _base, controller = self._controllers()
        initial_covariance = controller.episode_moment_covariance.copy()
        controller.start_episode(initial_environment_weight=1.0)
        a_start = controller.a_matrix.copy()
        b_start = controller.b.copy()
        rng = np.random.default_rng(7201)
        features = rng.normal(size=3)
        for cycle in range(5):
            terminal = cycle == 4
            next_features = rng.normal(size=3)
            controller.update(
                features=features,
                action_index=cycle % 2,
                cost=0.002 + cycle * 0.0001,
                next_features=next_features,
                terminal=terminal,
                next_action_index=None if terminal else (cycle + 1) % 2,
            )
            np.testing.assert_array_equal(
                controller.episode_moment_covariance,
                initial_covariance,
            )
            self.assertEqual(controller.episode_moment_count, 0)
            features = next_features

        theta = controller.a_inverse @ controller.b
        episode_a = controller.a_matrix - a_start
        episode_b = controller.b - b_start
        episode_score = episode_b - episode_a @ theta
        expected = initial_covariance + np.outer(
            episode_score,
            episode_score,
        )
        controller.finish_episode(learned=True)

        np.testing.assert_allclose(
            controller.episode_moment_covariance,
            expected,
            rtol=2.0e-12,
            atol=2.0e-14,
        )
        self.assertEqual(controller.episode_moment_count, 1)
        self.assertEqual(controller.episodes, 1)

    def test_factorized_sandwich_matches_full_joint_quadratic(self) -> None:
        weights = tuple(float(value) for value in np.linspace(1.0, 3.0, 41))
        centers = tuple(float(value) for value in np.linspace(1.0, 3.0, 9))
        controller = RecursiveLstdqV3LcbController(
            feature_dim=32,
            config=_config(weights=weights, centers=centers),
            spec=RecursiveLstdqV3LcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
            ),
            seed=73,
        )
        rng = np.random.default_rng(7301)
        features = rng.normal(size=controller.feature_dim)
        controller.theta[:] = rng.normal(
            scale=1.0e-2,
            size=controller.joint_dim,
        )
        inverse = rng.normal(
            scale=2.0e-2,
            size=(controller.joint_dim, controller.joint_dim),
        )
        controller.a_inverse[:] = inverse
        moment_factor = rng.normal(
            scale=1.0e-2,
            size=(controller.joint_dim, 24),
        )
        controller.episode_moment_covariance[:] = moment_factor @ moment_factor.T

        means, uncertainty, scores = controller._values(features, cycle=7)
        joint = controller.state_action_features(features)
        projected = joint @ controller.a_inverse
        reference_quadratic = np.sum(
            (projected @ controller.episode_moment_covariance) * projected,
            axis=1,
        )
        reference_uncertainty = np.sqrt(np.maximum(reference_quadratic, 0.0))
        reference_means = joint @ controller.theta
        reference_scores = np.maximum(
            reference_means
            - float(controller.spec.uncertainty_beta) * reference_uncertainty,
            float(controller.spec.lcb_lower_bound_sec),
        )

        np.testing.assert_allclose(
            means,
            reference_means,
            rtol=2.0e-12,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            uncertainty,
            reference_uncertainty,
            rtol=2.0e-12,
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            scores,
            reference_scores,
            rtol=2.0e-12,
            atol=2.0e-14,
        )

    def test_snapshot_rollback_and_checkpoint_restore_covariance(self) -> None:
        _base, controller = self._controllers()
        controller.start_episode(initial_environment_weight=1.0)
        controller.update(
            features=np.asarray([1.0, 0.0, 0.5]),
            action_index=0,
            cost=0.003,
            next_features=np.asarray([0.5, 1.0, 0.0]),
            terminal=True,
        )
        controller.finish_episode(learned=True)
        snapshot = controller.snapshot_learning_state()
        reference_values = controller._values(
            np.asarray([0.25, 0.5, 1.0]),
            cycle=0,
        )

        controller.start_episode(initial_environment_weight=1.0)
        controller.update(
            features=np.asarray([0.0, 1.0, 0.5]),
            action_index=1,
            cost=0.02,
            next_features=np.asarray([1.0, 0.0, 0.0]),
            terminal=True,
        )
        controller.finish_episode(learned=True)
        controller.restore_learning_state(snapshot)
        self.assertEqual(controller.episode_moment_count, 1)
        for actual, expected in zip(
            controller._values(np.asarray([0.25, 0.5, 1.0]), cycle=0),
            reference_values,
        ):
            np.testing.assert_allclose(actual, expected)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lstdq_v3.npz"
            controller.save(path)
            _unused, restored = self._controllers()
            restored.load(path)
        self.assertEqual(
            restored.episode_moment_count,
            controller.episode_moment_count,
        )
        np.testing.assert_allclose(
            restored.episode_moment_covariance,
            controller.episode_moment_covariance,
        )
        for actual, expected in zip(
            restored._values(np.asarray([0.25, 0.5, 1.0]), cycle=0),
            reference_values,
        ):
            np.testing.assert_allclose(actual, expected)
        original_action = controller.select_action(
            np.asarray([0.25, 0.5, 1.0]),
            explore=False,
            cycle=0,
        )
        restored_action = restored.select_action(
            np.asarray([0.25, 0.5, 1.0]),
            explore=False,
            cycle=0,
        )
        self.assertEqual(original_action[0], restored_action[0])
        self.assertEqual(original_action[1], restored_action[1])
        self.assertEqual(
            original_action[2]["greedy_index"],
            restored_action[2]["greedy_index"],
        )
        for key in ("q_values", "uncertainty", "selection_scores"):
            np.testing.assert_allclose(
                original_action[2][key],
                restored_action[2][key],
            )
        self.assertEqual(restored.summary()["stored_transition_count"], 0)

    def test_invalid_sandwich_state_fails_loudly(self) -> None:
        _base, controller = self._controllers()
        features = np.asarray([1.0, 0.5, 0.25])
        controller.episode_moment_covariance.fill(0.0)
        controller.episode_moment_covariance[0, 0] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "non-finite"):
            controller._values(features, cycle=0)

        controller.episode_moment_covariance[:] = -np.eye(
            controller.joint_dim,
            dtype=float,
        )
        with self.assertRaisesRegex(
            FloatingPointError,
            "materially negative",
        ):
            controller._values(features, cycle=0)

    def test_uncommitted_episode_does_not_change_cluster_covariance(self) -> None:
        _base, controller = self._controllers()
        covariance = controller.episode_moment_covariance.copy()
        controller.start_episode(initial_environment_weight=1.0)
        controller.update(
            features=np.asarray([1.0, 0.0, 0.5]),
            action_index=0,
            cost=0.003,
            next_features=np.asarray([0.5, 1.0, 0.0]),
            terminal=True,
        )
        controller.finish_episode(learned=False)
        np.testing.assert_array_equal(
            controller.episode_moment_covariance,
            covariance,
        )
        self.assertEqual(controller.episode_moment_count, 0)


if __name__ == "__main__":
    unittest.main()
