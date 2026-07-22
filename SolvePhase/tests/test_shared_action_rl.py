from __future__ import annotations

import _project_paths  # noqa: F401

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from SolvePhase.algorithms.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    build_action_basis,
    joint_action_features,
)
from SolvePhase.algorithms.lcb import (
    BootstrapLcbSarsaController,
    BootstrapSarsaSpec,
    HierarchicalLsviLcbController,
    HierarchicalLsviLcbSpec,
    RecursiveLstdqLcbController,
    RecursiveLstdqLcbSpec,
    RecursiveLstdqV2LcbController,
    RecursiveLstdqV2LcbSpec,
    RecursiveMonteCarloLcbController,
    RecursiveMonteCarloLcbSpec,
    StagewiseLsviLcbController,
    StagewiseLsviLcbSpec,
    StructuredModelBasedController,
    StructuredModelBasedSpec,
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
            reference.last_postfit_td_error = float(
                cost - difference @ reference.theta
            )
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
                fast_action = fast.select_action(features, explore=False, cycle=cycle)[0]
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


class RecursiveLstdqV2Tests(unittest.TestCase):
    @staticmethod
    def _controllers() -> tuple[
        RecursiveLstdqLcbController,
        RecursiveLstdqV2LcbController,
    ]:
        config = _config(
            weights=(1.0, 1.5),
            anchor_weight=1.0,
            action_basis_centers=(1.0, 1.5),
            epsilon_start=0.0,
            epsilon_final=0.0,
            trace_lambda=0.8,
        )
        v1 = RecursiveLstdqLcbController(
            feature_dim=3,
            config=config,
            spec=RecursiveLstdqLcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
            ),
            seed=61,
        )
        v2 = RecursiveLstdqV2LcbController(
            feature_dim=3,
            config=config,
            spec=RecursiveLstdqV2LcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
                coverage_ridge=1.0,
                residual_scale_window=8,
                residual_scale_min_samples=4,
            ),
            seed=61,
        )
        return v1, v2

    def test_mean_update_is_identical_to_v1_and_coverage_inverse_is_exact(self) -> None:
        v1, v2 = self._controllers()
        rng = np.random.default_rng(6101)
        for _episode in range(3):
            v1.start_episode(initial_environment_weight=1.0)
            v2.start_episode(initial_environment_weight=1.0)
            features = rng.normal(size=3)
            for cycle in range(5):
                terminal = cycle == 4
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
                self.assertAlmostEqual(v1.update(**kwargs), v2.update(**kwargs))
                for name in ("a_matrix", "a_inverse", "b", "theta", "trace"):
                    np.testing.assert_allclose(
                        getattr(v1, name), getattr(v2, name), rtol=1e-12, atol=1e-14
                    )
                np.testing.assert_allclose(
                    v2.coverage_inverse,
                    np.linalg.inv(v2.coverage_matrix),
                    rtol=1e-11,
                    atol=1e-13,
                )
                features = next_features
            v1.finish_episode(learned=True)
            v2.finish_episode(learned=True)

    def test_factorized_confidence_matches_full_joint_quadratic(self) -> None:
        weights = tuple(float(value) for value in np.linspace(1.0, 3.0, 41))
        centers = tuple(float(value) for value in np.linspace(1.0, 3.0, 9))
        config = _config(
            weights=weights,
            anchor_weight=1.6,
            action_basis_centers=centers,
            epsilon_start=0.0,
            epsilon_final=0.0,
        )
        controller = RecursiveLstdqV2LcbController(
            feature_dim=32,
            config=config,
            spec=RecursiveLstdqV2LcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
                coverage_ridge=1.0,
                residual_scale_window=64,
                residual_scale_min_samples=32,
            ),
            seed=62,
        )
        rng = np.random.default_rng(6201)
        features = rng.normal(size=controller.feature_dim)
        controller.theta[:] = rng.normal(
            scale=1.0e-2,
            size=controller.joint_dim,
        )
        factor = rng.normal(
            scale=1.0e-2,
            size=(controller.joint_dim, controller.joint_dim),
        )
        controller.coverage_inverse[:] = (
            factor @ factor.T + np.eye(controller.joint_dim)
        )
        controller.postfit_td_residuals.extend(
            rng.normal(scale=2.0e-3, size=64).tolist()
        )
        controller.sample_count = 64
        controller._scale_cache_sample_count = -1

        means, uncertainty, scores = controller._values(features, cycle=7)
        joint = controller.state_action_features(features)
        reference_means = joint @ controller.theta
        reference_quadratic = np.sum(
            (joint @ controller.coverage_inverse) * joint,
            axis=1,
        )
        reference_uncertainty = controller.residual_scale * np.sqrt(
            np.maximum(reference_quadratic, 0.0)
        )
        reference_scores = np.maximum(
            reference_means
            - float(controller.spec.uncertainty_beta) * reference_uncertainty,
            float(controller.spec.lcb_lower_bound_sec),
        )

        np.testing.assert_allclose(means, reference_means, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(
            uncertainty,
            reference_uncertainty,
            rtol=1e-13,
            atol=1e-14,
        )
        np.testing.assert_allclose(scores, reference_scores, rtol=1e-13, atol=1e-14)

    def test_optimized_v2_preserves_legacy_scores_actions_and_learning_trajectory(
        self,
    ) -> None:
        class LegacyRecursiveLstdqV2(RecursiveLstdqV2LcbController):
            """Pre-optimization equations retained as an exact test oracle."""

            def __init__(self, **kwargs: object) -> None:
                super().__init__(**kwargs)
                # Force the NumPy rank-one path used before the BLAS storage
                # optimization.  The production controller uses Fortran-order
                # matrices to enter the equivalent BLAS update.
                for name in (
                    "a_matrix",
                    "a_inverse",
                    "coverage_matrix",
                    "coverage_inverse",
                ):
                    setattr(
                        self,
                        name,
                        np.array(getattr(self, name), order="C", copy=True),
                    )

            def _values(
                self,
                features: np.ndarray,
                *,
                cycle: int | None,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                del cycle
                joint = self.state_action_features(features)
                means = joint @ self.theta
                quadratic = np.sum(
                    (joint @ self.coverage_inverse) * joint,
                    axis=1,
                )
                uncertainty = float(self.residual_scale) * np.sqrt(
                    np.maximum(quadratic, 0.0)
                )
                scores = means - float(self.spec.uncertainty_beta) * uncertainty
                if self.spec.lcb_lower_bound_sec is not None:
                    scores = np.maximum(
                        scores,
                        float(self.spec.lcb_lower_bound_sec),
                    )
                return means, uncertainty, scores

        weights = tuple(float(value) for value in np.linspace(1.0, 3.0, 41))
        config = _config(
            weights=weights,
            anchor_weight=1.6,
            action_basis_centers=tuple(
                float(value) for value in np.linspace(1.0, 3.0, 9)
            ),
            epsilon_start=0.0,
            epsilon_final=0.0,
            trace_lambda=0.8,
        )
        spec = RecursiveLstdqV2LcbSpec(
            ridge=1.0,
            uncertainty_beta=2.0,
            residual_floor_sec=1.0e-3,
            coverage_ridge=1.0,
            residual_scale_window=2048,
            residual_scale_min_samples=32,
        )
        optimized = RecursiveLstdqV2LcbController(
            feature_dim=8,
            config=config,
            spec=spec,
            seed=63,
        )
        legacy = LegacyRecursiveLstdqV2(
            feature_dim=8,
            config=config,
            spec=spec,
            seed=63,
        )
        rng = np.random.default_rng(6301)

        for _episode in range(25):
            optimized.start_episode(initial_environment_weight=1.6)
            legacy.start_episode(initial_environment_weight=1.6)
            features = rng.normal(size=8)
            action_optimized = optimized.select_action(
                features,
                explore=False,
                cycle=0,
            )[0]
            action_legacy = legacy.select_action(
                features,
                explore=False,
                cycle=0,
            )[0]
            self.assertEqual(action_optimized, action_legacy)

            for cycle in range(24):
                for optimized_value, legacy_value in zip(
                    optimized._values(features, cycle=cycle),
                    legacy._values(features, cycle=cycle),
                ):
                    np.testing.assert_allclose(
                        optimized_value,
                        legacy_value,
                        rtol=2.0e-11,
                        atol=2.0e-13,
                    )

                terminal = cycle == 23
                next_features = rng.normal(size=8)
                if terminal:
                    next_action_optimized = None
                    next_action_legacy = None
                else:
                    next_action_optimized = optimized.select_action(
                        next_features,
                        explore=False,
                        cycle=cycle + 1,
                    )[0]
                    next_action_legacy = legacy.select_action(
                        next_features,
                        explore=False,
                        cycle=cycle + 1,
                    )[0]
                    self.assertEqual(next_action_optimized, next_action_legacy)

                action_weight = weights[action_optimized]
                cost = float(
                    2.0e-3
                    + 3.0e-4 * (action_weight - 1.65) ** 2
                    + 1.0e-4 * (1.0 + np.tanh(features[0]))
                )
                optimized_td_error = optimized.update(
                    features=features,
                    action_index=action_optimized,
                    cost=cost,
                    next_features=next_features,
                    terminal=terminal,
                    next_action_index=next_action_optimized,
                )
                legacy_td_error = legacy.update(
                    features=features,
                    action_index=action_legacy,
                    cost=cost,
                    next_features=next_features,
                    terminal=terminal,
                    next_action_index=next_action_legacy,
                )
                self.assertAlmostEqual(optimized_td_error, legacy_td_error, places=13)
                self.assertAlmostEqual(
                    optimized.residual_scale,
                    legacy.residual_scale,
                    places=13,
                )
                for name in (
                    "a_matrix",
                    "a_inverse",
                    "b",
                    "theta",
                    "trace",
                    "coverage_matrix",
                    "coverage_inverse",
                ):
                    np.testing.assert_allclose(
                        getattr(optimized, name),
                        getattr(legacy, name),
                        rtol=2.0e-11,
                        atol=2.0e-13,
                    )

                features = next_features
                action_optimized = next_action_optimized
                action_legacy = next_action_legacy
            optimized.finish_episode(learned=True)
            legacy.finish_episode(learned=True)

        self.assertEqual(optimized.steps, 600)
        self.assertEqual(optimized.steps, legacy.steps)
        self.assertEqual(optimized.episodes, legacy.episodes)
        np.testing.assert_allclose(
            list(optimized.postfit_td_residuals),
            list(legacy.postfit_td_residuals),
            rtol=5.0e-12,
            atol=5.0e-15,
        )

    def test_rolling_mad_scale_checkpoint_and_rollback(self) -> None:
        _v1, controller = self._controllers()
        controller.postfit_td_residuals.extend([0.0, 1.0, 2.0, 100.0])
        controller.sample_count = 4
        controller._scale_cache_sample_count = -1
        self.assertAlmostEqual(controller.residual_scale, 1.4826, places=6)
        snapshot = controller.snapshot_learning_state()
        controller.coverage_matrix[:] = 7.0
        controller.postfit_td_residuals.append(200.0)
        controller.restore_learning_state(snapshot)
        np.testing.assert_allclose(
            controller.coverage_inverse,
            np.linalg.inv(controller.coverage_matrix),
        )
        self.assertEqual(list(controller.postfit_td_residuals), [0.0, 1.0, 2.0, 100.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lstdq_v2.npz"
            controller.save(path)
            _unused, restored = self._controllers()
            restored.load(path)
        np.testing.assert_allclose(restored.theta, controller.theta)
        np.testing.assert_allclose(
            restored.coverage_inverse, controller.coverage_inverse
        )
        self.assertEqual(
            list(restored.postfit_td_residuals),
            list(controller.postfit_td_residuals),
        )

    def test_rolling_mad_window_preserves_legacy_order_and_scale_after_wrap(
        self,
    ) -> None:
        _v1, controller = self._controllers()
        values = np.arange(12.0)
        controller.postfit_td_residuals.extend(values)
        controller.sample_count = int(values.size)
        controller._scale_cache_sample_count = -1
        retained = values[-int(controller.spec.residual_scale_window) :]
        center = float(np.median(retained))
        expected_scale = max(
            1.4826 * float(np.median(np.abs(retained - center))),
            float(controller.spec.residual_floor_sec),
        )

        np.testing.assert_array_equal(
            list(controller.postfit_td_residuals),
            retained,
        )
        self.assertEqual(controller.residual_scale, expected_scale)
        snapshot = controller.snapshot_learning_state()
        controller.postfit_td_residuals.append(99.0)
        controller.restore_learning_state(snapshot)
        np.testing.assert_array_equal(
            list(controller.postfit_td_residuals),
            retained,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wrapped_lstdq_v2.npz"
            controller.save(path)
            _unused, restored = self._controllers()
            restored.load(path)
        np.testing.assert_array_equal(
            list(restored.postfit_td_residuals),
            retained,
        )
        self.assertEqual(restored.residual_scale, expected_scale)


class StructuredModelBasedTests(unittest.TestCase):
    @staticmethod
    def _controller() -> StructuredModelBasedController:
        config = _config(
            weights=(1.0, 2.0),
            anchor_weight=1.0,
            action_basis_centers=(1.0, 2.0),
            epsilon_start=0.0,
            epsilon_final=0.0,
        )
        return StructuredModelBasedController(
            feature_dim=1,
            config=config,
            spec=StructuredModelBasedSpec(
                ridge=1.0e-6,
                minimum_samples=2,
                scale_window=8,
            ),
            seed=71,
        )

    def test_uses_native_cost_and_prefers_lower_time_per_progress(self) -> None:
        controller = self._controller()
        features = np.asarray([1.0])
        controller.start_episode(initial_environment_weight=1.0)
        controller.update(
            features=features,
            action_index=0,
            cost=100.0,
            native_cycle_cost=2.0,
            residual_ratio=math.exp(-1.0),
            next_features=features,
            terminal=True,
        )
        controller.update(
            features=features,
            action_index=1,
            cost=100.0,
            native_cycle_cost=1.0,
            residual_ratio=math.exp(-1.0),
            next_features=features,
            terminal=True,
        )
        action, weight, metadata = controller.select_action(
            features, explore=False
        )
        self.assertEqual(action, 1)
        self.assertEqual(weight, 2.0)
        self.assertLess(
            metadata["predicted_native_cycle_costs"][1],
            metadata["predicted_native_cycle_costs"][0],
        )
        self.assertLess(float(controller.cost_theta.max()), 3.0)

    def test_native_exception_without_next_residual_is_skipped(self) -> None:
        controller = self._controller()
        controller.update(
            features=np.asarray([1.0]),
            action_index=0,
            cost=1.0,
            native_cycle_cost=None,
            residual_ratio=None,
            next_features=np.asarray([1.0]),
            terminal=True,
        )
        self.assertEqual(controller.sample_count, 0)
        self.assertEqual(controller.skipped_transition_count, 1)


class HierarchicalLsviTests(unittest.TestCase):
    @staticmethod
    def _controller() -> HierarchicalLsviLcbController:
        config = _config(
            weights=(1.0,),
            anchor_weight=1.0,
            action_basis_centers=(1.0,),
            epsilon_start=0.0,
            epsilon_final=0.0,
        )
        return HierarchicalLsviLcbController(
            feature_dim=1,
            config=config,
            spec=HierarchicalLsviLcbSpec(
                horizon=3,
                ridge=1.0e-6,
                uncertainty_beta=0.0,
                residual_floor_sec=1.0e-6,
                refit_interval_episodes=1,
                refit_sweeps=3,
                residual_shrinkage_samples=4.0,
            ),
            seed=81,
        )

    def test_dynamic_cap_shared_prior_and_sparse_scale(self) -> None:
        controller = self._controller()
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
        self.assertAlmostEqual(controller.observed_return_max_sec, 0.003)
        self.assertGreater(float(np.linalg.norm(controller.shared_theta)), 0.0)
        np.testing.assert_allclose(controller.theta[2], controller.shared_theta)
        self.assertAlmostEqual(
            float(controller.residual_scales[2]),
            float(controller.global_residual_scale),
        )
        self.assertEqual(controller.refit_count, 1)

    def test_checkpoint_round_trip(self) -> None:
        source = self._controller()
        features = np.asarray([1.0])
        source.start_episode(initial_environment_weight=1.0)
        source.update(
            features=features,
            action_index=0,
            cost=0.002,
            next_features=features,
            terminal=True,
            next_cycle=1,
        )
        source.finish_episode(learned=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hierarchical_lsvi.npz"
            source.save(path)
            restored = self._controller()
            restored.load(path)
        np.testing.assert_allclose(restored.theta, source.theta)
        np.testing.assert_allclose(restored.shared_theta, source.shared_theta)
        self.assertEqual(restored.refit_count, source.refit_count)
        self.assertEqual(
            restored.observed_return_max_sec, source.observed_return_max_sec
        )


if __name__ == "__main__":
    unittest.main()
