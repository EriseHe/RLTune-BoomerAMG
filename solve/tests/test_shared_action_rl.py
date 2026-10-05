from __future__ import annotations

from solve.tests import _project_paths  # noqa: F401

import unittest
import numpy as np

from solve.controllers.common import build_action_basis, joint_action_features
from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
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

    def test_selected_joint_features_match_rows_from_full_feature_map(self) -> None:
        basis = build_action_basis(
            WEIGHTS,
            mode="compact_rbf",
            rbf_sigma=0.2,
            rbf_centers=CENTERS,
        )
        features = np.asarray([1.0, -0.5, 0.25])
        full = joint_action_features(basis, features)
        selected_indices = np.asarray([0, 4, 10])
        selected = joint_action_features(
            basis,
            features,
            action_indices=selected_indices,
        )
        scalar = joint_action_features(basis, features, action_indices=4)
        np.testing.assert_array_equal(selected, full[selected_indices])
        np.testing.assert_array_equal(scalar, full[[4]])


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
        joint = controller.state_action_features(first)
        projected = joint @ controller.a_inverse
        reference_quadratic = np.einsum(
            "ai,ij,aj->a",
            projected,
            controller.moment_covariance,
            projected,
            optimize=True,
        )
        np.testing.assert_allclose(
            uncertainty,
            np.sqrt(np.maximum(reference_quadratic, 0.0)),
            rtol=1.0e-12,
            atol=1.0e-15,
        )
        self.assertTrue(np.all(np.isfinite(uncertainty)))

    def test_fast_update_matches_full_normal_equation_recompute(self) -> None:
        config = _config(
            weights=(1.0, 1.5, 2.0),
            anchor_weight=1.5,
            action_basis_centers=(1.0, 1.5, 2.0),
            epsilon_start=0.0,
            epsilon_final=0.0,
            trace_lambda=0.8,
        )
        spec = RecursiveLstdqLcbSpec(
            ridge=1.0,
            uncertainty_beta=2.0,
            residual_floor_sec=1.0e-3,
        )
        fast = RecursiveLstdqLcbController(
            feature_dim=4,
            config=config,
            spec=spec,
            seed=53,
        )
        reference = RecursiveLstdqLcbController(
            feature_dim=4,
            config=config,
            spec=spec,
            seed=53,
        )
        rng = np.random.default_rng(5301)

        def reference_update(
            *,
            features: np.ndarray,
            action_index: int,
            cost: float,
            next_features: np.ndarray,
            terminal: bool,
            next_action_index: int | None,
        ) -> float:
            joint = reference.state_action_features(features)[int(action_index)]
            next_joint = (
                np.zeros(reference.joint_dim, dtype=float)
                if terminal
                else reference.state_action_features(next_features)[
                    int(next_action_index)
                ]
            )
            difference = joint - float(config.gamma) * next_joint
            td_error = float(
                cost
                + config.gamma * (next_joint @ reference.theta)
                - joint @ reference.theta
            )
            reference.trace = (
                float(config.gamma) * float(config.trace_lambda) * reference.trace
                + joint
            )
            projected_trace = reference.a_inverse @ reference.trace
            projected_difference = difference @ reference.a_inverse
            denominator = 1.0 + float(difference @ projected_trace)
            reference.a_matrix += np.outer(reference.trace, difference)
            reference.b += float(cost) * reference.trace
            reference.a_inverse -= (
                np.outer(projected_trace, projected_difference) / denominator
            )
            reference.theta[:] = reference.a_inverse @ reference.b
            reference.last_postfit_td_error = float(cost - difference @ reference.theta)
            moment = reference.trace * reference.last_postfit_td_error
            reference.moment_covariance += np.outer(moment, moment)
            reference.steps += 1
            reference.sample_count += 1
            return td_error

        for _episode in range(8):
            fast.start_episode(initial_environment_weight=1.5)
            reference.start_episode(initial_environment_weight=1.5)
            features = rng.normal(size=4)
            for cycle in range(12):
                fast_values = fast._values(features, cycle=cycle)
                reference_values = reference._values(features, cycle=cycle)
                for fast_value, reference_value in zip(
                    fast_values,
                    reference_values,
                ):
                    np.testing.assert_allclose(
                        fast_value,
                        reference_value,
                        rtol=2.0e-11,
                        atol=2.0e-13,
                    )
                fast_action = fast.select_action(features, explore=False, cycle=cycle)[
                    0
                ]
                reference_action = reference.select_action(
                    features,
                    explore=False,
                    cycle=cycle,
                )[0]
                self.assertEqual(fast_action, reference_action)

                terminal = cycle == 11
                next_features = rng.normal(size=4)
                next_action = None if terminal else int(rng.integers(0, 3))
                cost = float(rng.uniform(5.0e-4, 5.0e-3))
                fast_td_error = fast.update(
                    features=features,
                    action_index=fast_action,
                    cost=cost,
                    next_features=next_features,
                    terminal=terminal,
                    next_action_index=next_action,
                )
                reference_td_error = reference_update(
                    features=features,
                    action_index=reference_action,
                    cost=cost,
                    next_features=next_features,
                    terminal=terminal,
                    next_action_index=next_action,
                )
                self.assertAlmostEqual(fast_td_error, reference_td_error, places=13)
                for fast_value, reference_value in (
                    (fast.a_matrix, reference.a_matrix),
                    (fast.a_inverse, reference.a_inverse),
                    (fast.b, reference.b),
                    (fast.theta, reference.theta),
                    (fast.moment_covariance, reference.moment_covariance),
                ):
                    np.testing.assert_allclose(
                        fast_value,
                        reference_value,
                        rtol=2.0e-11,
                        atol=2.0e-13,
                    )
                features = next_features
            fast.finish_episode(learned=True)
            reference.finish_episode(learned=True)

        np.testing.assert_allclose(fast.theta, fast.a_inverse @ fast.b)
        self.assertEqual(fast.episodes, reference.episodes)
        self.assertEqual(fast.steps, reference.steps)

    def test_postfit_covariance_preserves_full_lstd_cost_target(self) -> None:
        controller = self._controller()
        features = np.asarray([1.0, 0.0])
        joint = controller.state_action_features(features)[0]
        initial_covariance = controller.moment_covariance.copy()
        controller.start_episode(initial_environment_weight=1.0)
        td_error = controller.update(
            features=features,
            action_index=0,
            cost=0.3,
            next_features=features,
            terminal=True,
        )
        np.testing.assert_allclose(controller.b, 0.3 * joint)
        expected_postfit = float(0.3 - joint @ controller.theta)
        np.testing.assert_allclose(
            controller.moment_covariance - initial_covariance,
            np.outer(expected_postfit * joint, expected_postfit * joint),
        )
        self.assertAlmostEqual(td_error, 0.3)
        self.assertAlmostEqual(controller.last_postfit_td_error, expected_postfit)

    def test_nonnegative_lcb_tie_prefers_lower_predicted_cost(self) -> None:
        config = _config(
            weights=(1.0, 2.0),
            anchor_weight=1.0,
            action_basis_centers=(1.0, 2.0),
            epsilon_start=0.0,
            epsilon_final=0.0,
        )
        controller = RecursiveLstdqLcbController(
            feature_dim=1,
            config=config,
            spec=RecursiveLstdqLcbSpec(
                ridge=1.0,
                uncertainty_beta=1.0,
                residual_floor_sec=1.0,
                lcb_lower_bound_sec=0.0,
            ),
            seed=52,
        )
        features = np.asarray([1.0])
        joint = controller.state_action_features(features)
        controller.theta[:] = np.linalg.solve(
            joint,
            np.asarray([0.2, 0.1]),
        )
        means, _uncertainty, scores = controller._values(features, cycle=0)
        np.testing.assert_allclose(means, [0.2, 0.1])
        np.testing.assert_allclose(scores, [0.0, 0.0])
        action_index, action, _metadata = controller.select_action(
            features,
            explore=False,
        )
        self.assertEqual(action_index, 1)
        self.assertEqual(action, 2.0)


if __name__ == "__main__":
    unittest.main()
