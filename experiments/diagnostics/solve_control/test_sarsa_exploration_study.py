from __future__ import annotations

import _project_paths  # noqa: F401

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from online_td_lambda import ExpectedSarsaLambdaConfig
from run_sarsa_exploration_study import study_specs
from sarsa_exploration_policies import (
    ExplorationStudyController,
    ExplorationStudySpec,
)


def _controller(
    *,
    actions: tuple[float, ...],
    anchor: float,
    action_mode: str,
    behavior_mode: str,
    epsilon: float = 1.0,
    force_default_first_action: bool = True,
) -> ExplorationStudyController:
    config = ExpectedSarsaLambdaConfig(
        weights=actions,
        anchor_weight=anchor,
        alpha=0.005,
        gamma=1.0,
        trace_lambda=0.8,
        epsilon_start=epsilon,
        epsilon_final=epsilon,
        epsilon_decay_steps=20_000.0,
        initial_q_sec=0.0,
        exploration_mode="uniform",
        td_algorithm="true_online_sarsa",
    )
    return ExplorationStudyController(
        study_spec=ExplorationStudySpec(
            action_mode=action_mode,
            behavior_mode=behavior_mode,
            initial_weight=1.0,
            force_default_first_action=force_default_first_action,
        ),
        feature_dim=2,
        config=config,
        seed=19,
        initial_parameters=np.zeros(2, dtype=float),
    )


class ExplorationStudyControllerTests(unittest.TestCase):
    def test_factorial_contains_six_variants(self) -> None:
        self.assertEqual(
            {spec.name for spec in study_specs()},
            {
                "absolute_uniform",
                "absolute_local",
                "absolute_uncertainty_lcb",
                "residual_uniform",
                "residual_local",
                "residual_uncertainty_lcb",
            },
        )

    def test_residual_action_accumulates_and_clips_physical_weight(self) -> None:
        controller = _controller(
            actions=(-0.02, 0.0, 0.02),
            anchor=0.0,
            action_mode="residual",
            behavior_mode="uniform",
        )
        self.assertEqual(controller.initial_environment_weight, 1.0)
        self.assertAlmostEqual(controller.environment_weight(2, 1.00), 1.02)
        self.assertAlmostEqual(controller.environment_weight(2, 1.99), 2.00)
        self.assertAlmostEqual(controller.environment_weight(0, 1.01), 1.00)

    def test_only_first_online_decision_is_forced_to_default(self) -> None:
        controller = _controller(
            actions=(1.0, 1.1, 1.2),
            anchor=1.0,
            action_mode="absolute",
            behavior_mode="uniform",
            epsilon=0.0,
        )
        controller.theta[:, 0] = np.asarray([3.0, 2.0, 0.0])
        features = np.asarray([1.0, 0.0])
        first, _value, first_metadata = controller.select_action(
            features,
            explore=True,
        )
        second, _value, second_metadata = controller.select_action(
            features,
            explore=True,
        )
        self.assertEqual(first, 0)
        self.assertTrue(first_metadata["forced_default_first_action"])
        self.assertEqual(second, 2)
        self.assertFalse(second_metadata["forced_default_first_action"])
        self.assertEqual(controller.forced_initial_action_count, 1)

    def test_first_decision_uses_prepared_solver_default(self) -> None:
        controller = _controller(
            actions=(1.0, 1.1, 1.2),
            anchor=1.0,
            action_mode="absolute",
            behavior_mode="uniform",
            epsilon=0.0,
        )
        controller.start_episode(initial_environment_weight=1.2)
        action, action_value, metadata = controller.select_action(
            np.asarray([1.0, 0.0]),
            explore=True,
        )
        self.assertEqual(action, 2)
        self.assertEqual(action_value, 1.2)
        self.assertTrue(metadata["forced_default_first_action"])

    def test_residual_first_decision_uses_zero_delta_from_default(self) -> None:
        controller = _controller(
            actions=(-0.02, -0.01, 0.0, 0.01, 0.02),
            anchor=0.0,
            action_mode="residual",
            behavior_mode="local",
            epsilon=0.0,
        )
        controller.theta[:, 0] = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
        action, action_value, metadata = controller.select_action(
            np.asarray([1.0, 0.0]),
            explore=True,
        )
        self.assertEqual(action_value, 0.0)
        self.assertEqual(controller.environment_weight(action, 1.0), 1.0)
        self.assertTrue(metadata["forced_default_first_action"])

    def test_local_epsilon_selects_only_immediate_non_greedy_neighbors(self) -> None:
        controller = _controller(
            actions=(1.0, 1.1, 1.2, 1.3, 1.4),
            anchor=1.2,
            action_mode="absolute",
            behavior_mode="local",
            force_default_first_action=False,
        )
        controller.theta[:, 0] = np.asarray([3.0, 1.0, 0.0, 2.0, 4.0])
        features = np.asarray([1.0, 0.0])
        selected = {
            controller.select_action(features, explore=True)[0]
            for _ in range(100)
        }
        self.assertEqual(selected, {1, 3})

    def test_uncertainty_lcb_has_no_epsilon_and_covers_unseen_actions(self) -> None:
        controller = _controller(
            actions=(1.0, 1.1, 1.2),
            anchor=1.1,
            action_mode="absolute",
            behavior_mode="uncertainty_lcb",
            force_default_first_action=False,
        )
        features = np.asarray([1.0, 0.5])
        selections = []
        for _ in range(3):
            action, _value, metadata = controller.select_action(
                features,
                explore=True,
            )
            selections.append(action)
            self.assertEqual(metadata["epsilon"], 0.0)
        self.assertEqual(set(selections), {0, 1, 2})
        np.testing.assert_array_equal(controller.coverage_counts, np.ones(3))

    def test_uncertainty_is_not_used_in_frozen_greedy_selection(self) -> None:
        controller = _controller(
            actions=(1.0, 1.1, 1.2),
            anchor=1.1,
            action_mode="absolute",
            behavior_mode="uncertainty_lcb",
            force_default_first_action=False,
        )
        controller.theta[:, 0] = np.asarray([2.0, 1.0, 0.0])
        action, _value, metadata = controller.select_action(
            np.asarray([1.0, 0.0]),
            explore=False,
        )
        self.assertEqual(action, 2)
        self.assertEqual(metadata["epsilon"], 0.0)
        self.assertEqual(metadata["uncertainty"], [0.0, 0.0, 0.0])

    def test_uncertainty_checkpoint_restores_coverage_state(self) -> None:
        controller = _controller(
            actions=(1.0, 1.1, 1.2),
            anchor=1.1,
            action_mode="absolute",
            behavior_mode="uncertainty_lcb",
            force_default_first_action=False,
        )
        features = np.asarray([1.0, 0.5])
        action, _value, _metadata = controller.select_action(
            features,
            explore=True,
        )
        controller.update(
            features=features,
            action_index=action,
            cost=0.01,
            next_features=features,
            next_action_index=action,
            terminal=False,
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "controller.npz"
            controller.save(path)
            restored = _controller(
                actions=(1.0, 1.1, 1.2),
                anchor=1.1,
                action_mode="absolute",
                behavior_mode="uncertainty_lcb",
                force_default_first_action=False,
            )
            restored.load(path)
        np.testing.assert_allclose(
            restored.coverage_inverse,
            controller.coverage_inverse,
        )
        np.testing.assert_array_equal(
            restored.coverage_counts,
            controller.coverage_counts,
        )
        self.assertEqual(restored.td_error_count, controller.td_error_count)
        self.assertAlmostEqual(
            restored.td_error_sum_squares,
            controller.td_error_sum_squares,
        )


if __name__ == "__main__":
    unittest.main()
