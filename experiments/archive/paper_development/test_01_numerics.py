"""Stage 1: LSTDQ recovery and complete-episode algebra; no PDE solves."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from solve.controllers.recursive_lstdq import (
    RecursiveLstdqLcbController, RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbController, RecursiveLstdqV2LcbSpec,
    RecursiveLstdqV3LcbController, RecursiveLstdqV3LcbSpec,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig


FAMILIES = (
    (RecursiveLstdqLcbController, RecursiveLstdqLcbSpec, 2),
    (RecursiveLstdqV2LcbController, RecursiveLstdqV2LcbSpec, 1),
    (RecursiveLstdqV3LcbController, RecursiveLstdqV3LcbSpec, 1),
)


def controller_for(family):
    controller, spec, _legacy_version = family
    return controller(
        feature_dim=2,
        config=ExpectedSarsaLambdaConfig(
            weights=(1.0,), anchor_weight=1.0, gamma=1.0,
            trace_lambda=0.8, epsilon_start=0.0, epsilon_final=0.0,
            action_basis_mode="compact_rbf", action_basis_centers=(1.0,),
            action_rbf_sigma=0.2,
        ),
        spec=spec(ridge=1.0, uncertainty_beta=2.0, residual_floor_sec=0.001),
        seed=9131601,
    )


def first_transition(controller, next_value=2.0):
    controller.start_episode(initial_environment_weight=1.0)
    controller.update(
        features=np.array([1.0, 0.0]), action_index=0, cost=1.0,
        next_features=np.array([next_value, 0.0]), terminal=False,
        next_action_index=0,
    )


def terminal_transition(controller, value=2.0):
    controller.update(
        features=np.array([value, 0.0]), action_index=0, cost=1.0,
        next_features=np.zeros(2), terminal=True,
    )
    controller.finish_episode(learned=True)


class NumericsTests(unittest.TestCase):
    def assert_batch_solution(self, controller):
        self.assertTrue(controller.inverse_is_valid)
        np.testing.assert_allclose(
            controller.theta, np.linalg.solve(controller.a_matrix, controller.b),
            rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_allclose(
            controller.a_matrix @ controller.a_inverse, np.eye(2),
            rtol=0, atol=1e-10,
        )
        np.testing.assert_allclose(
            controller.a_inverse @ controller.a_matrix, np.eye(2),
            rtol=0, atol=1e-10,
        )

    def test_singular_prefix_recovers_exact_nineteen_over_twenty_eight(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                first_transition(controller)
                self.assertFalse(controller.inverse_is_valid)
                np.testing.assert_array_equal(controller.a_matrix, np.diag([0., 1.]))
                terminal_transition(controller)
                self.assert_batch_solution(controller)
                np.testing.assert_allclose(controller.theta, [19 / 28, 0.])
                self.assertEqual(controller.inverse_rebuild_count, 2)

    def test_near_singular_invertible_prefix_uses_audited_inverse(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                first_transition(controller, next_value=2.0 - 1e-12)
                self.assertTrue(controller.inverse_is_valid)
                self.assertEqual(controller.inverse_rebuild_count, 1)
                self.assert_batch_solution(controller)
                terminal_transition(controller, value=2.0 - 1e-12)
                self.assert_batch_solution(controller)

    def test_truncated_pseudoinverse_is_not_used_in_sherman_morrison(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                value = 2.0 - np.spacing(2.0)
                # Simulate a failed factorization at a prefix where pinv drops
                # the tiny singular direction; the next update must rebuild.
                with patch("numpy.linalg.inv", side_effect=np.linalg.LinAlgError):
                    first_transition(controller, next_value=value)
                self.assertFalse(controller.inverse_is_valid)
                self.assertEqual(controller.a_inverse[0, 0], 0.0)
                terminal_transition(controller, value=value)
                self.assert_batch_solution(controller)

    def test_snapshot_and_rollback_restore_inverse_validity_and_trace(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                before = controller.snapshot_learning_state()
                first_transition(controller)
                singular = controller.snapshot_learning_state()
                terminal_transition(controller)
                controller.restore_learning_state(singular)
                self.assertFalse(controller.inverse_is_valid)
                terminal_transition(controller)
                self.assert_batch_solution(controller)
                controller.restore_learning_state(before)
                self.assertTrue(controller.inverse_is_valid)
                self.assertEqual(controller.inverse_rebuild_count, 0)
                np.testing.assert_array_equal(controller.a_matrix, np.eye(2))
                np.testing.assert_array_equal(controller.theta, np.zeros(2))
                np.testing.assert_array_equal(controller.trace, np.zeros(2))

    def test_checkpoint_preserves_invalid_prefix_flag(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                first_transition(controller)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "prefix.npz"
                    controller.save(path)
                    restored = controller_for(family)
                    restored.load(path)
                # File checkpoints start a new episode (traces are reset);
                # in-memory snapshots above support exact prefix restoration.
                self.assertFalse(restored.inverse_is_valid)
                np.testing.assert_array_equal(restored.a_inverse, controller.a_inverse)

    def test_legacy_checkpoint_with_bad_inverse_is_repaired(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                first_transition(controller)
                terminal_transition(controller)
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "legacy.npz"
                    controller.save(path)
                    with np.load(path, allow_pickle=False) as saved:
                        fields = {key: saved[key] for key in saved.files}
                    fields.pop("inverse_is_valid")
                    fields["checkpoint_version"] = np.asarray(family[2])
                    fields["a_inverse"] = np.zeros((2, 2))
                    fields["theta"] = np.zeros(2)
                    np.savez_compressed(path, **fields)
                    restored = controller_for(family)
                    restored.load(path)
                self.assert_batch_solution(restored)
                np.testing.assert_allclose(restored.theta, [19 / 28, 0.])

    def test_incomplete_singular_episode_cannot_commit_as_valid(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                first_transition(controller)
                with self.assertRaisesRegex(FloatingPointError, "episode boundary"):
                    controller.finish_episode(learned=True)
                self.assertEqual(controller.episodes, 0)

    def test_coherent_episodes_match_batch_and_trace_energy_identity(self):
        for family in FAMILIES:
            with self.subTest(family=family[0].__name__):
                controller = controller_for(family)
                rng = np.random.default_rng(9131602)
                for _episode in range(5):
                    features = rng.normal(size=(7, 2))
                    controller.start_episode()
                    before = controller.a_matrix.copy()
                    trace = np.zeros(2)
                    differences = []
                    for index, feature in enumerate(features):
                        previous = trace.copy()
                        trace = 0.8 * trace + feature
                        differences.append(trace - previous)
                        terminal = index == len(features) - 1
                        controller.update(
                            features=feature, action_index=0,
                            cost=float(rng.uniform(0.001, 0.01)),
                            next_features=np.zeros(2) if terminal else features[index + 1],
                            terminal=terminal, next_action_index=None if terminal else 0,
                        )
                    controller.finish_episode(learned=True)
                    episode_matrix = controller.a_matrix - before
                    delta = np.asarray(differences)
                    expected = 0.9 * delta.T @ delta + 0.1 * np.outer(trace, trace)
                    np.testing.assert_allclose(
                        (episode_matrix + episode_matrix.T) / 2, expected,
                        rtol=1e-12, atol=1e-12,
                    )
                    self.assertGreaterEqual(
                        np.linalg.eigvalsh(
                            (controller.a_matrix + controller.a_matrix.T) / 2
                        ).min(), 1.0 - 1e-12,
                    )
                    self.assert_batch_solution(controller)
                self.assertEqual(controller.inverse_rebuild_count, 0)


if __name__ == "__main__":
    unittest.main()
