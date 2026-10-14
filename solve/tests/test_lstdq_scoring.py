"""Check exact block projection against dense LSTDQ scoring and policy rules."""
from __future__ import annotations

from solve.tests import _project_paths  # noqa: F401

import unittest
import numpy as np

from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig
from solve.controllers.recursive_lstdq import RecursiveLstdqController, RecursiveLstdqSpec


class DenseLiftController(RecursiveLstdqController):
    def _values(self, features, *, cycle):
        state = self._factorized_scoring.prepare_state(features)
        projected = self._factorized_scoring.state_lift @ self.a_inverse
        covariance = projected @ self.episode_moment_covariance @ projected.T
        covariance = .5 * (covariance + covariance.T)
        means = self._factorized_scoring.means(self.theta, state)
        width = np.sqrt(np.maximum(
            self._factorized_scoring.action_quadratic(covariance), 0.))
        score = means - self.spec.uncertainty_beta * width
        if self.spec.lcb_lower_bound_sec is not None:
            score = np.maximum(score, self.spec.lcb_lower_bound_sec)
        return means, width, score


def config(*, legacy=False, epsilon=.3, force=False):
    return ExpectedSarsaLambdaConfig(
        weights=(1., 1.5, 2.) if legacy else tuple(np.linspace(1., 3., 41)),
        anchor_weight=1.5,
        action_basis_mode="legacy" if legacy else "compact_rbf",
        action_rbf_sigma=0. if legacy else .2,
        action_basis_centers=() if legacy else tuple(np.linspace(1., 3., 9)),
        gamma=1., trace_lambda=.8, epsilon_start=epsilon, epsilon_final=epsilon,
        force_default_first_action=force)


class LstdqBlockScoringTests(unittest.TestCase):
    def test_matches_full_joint_features_for_contexts_and_matrix_layouts(self):
        rng = np.random.default_rng(1427)
        for dimension, legacy in ((32, False), (35, False), (4, True)):
            for layout in ("C", "F", "strided"):
                with self.subTest(dimension=dimension, layout=layout, legacy=legacy):
                    model = RecursiveLstdqController(feature_dim=dimension,
                        config=config(legacy=legacy), spec=RecursiveLstdqSpec(), seed=3)
                    d = model.joint_dim
                    inverse = np.eye(d) + rng.normal(scale=.02, size=(d, d))
                    if layout == "strided":
                        storage = np.zeros((2*d, 2*d))
                        storage[::2, ::2] = inverse
                        model.a_inverse = storage[::2, ::2]
                    else:
                        model.a_inverse = np.array(inverse, order=layout)
                    factor = rng.normal(scale=.01, size=(d, 17))
                    model.episode_moment_covariance[:] = factor@factor.T + .001*np.eye(d)
                    model.theta[:] = rng.normal(scale=.01, size=d)
                    for x in (rng.normal(size=dimension), np.zeros(dimension)):
                        joint = model.state_action_features(x)
                        projected = joint@model.a_inverse
                        means = joint@model.theta
                        width = np.sqrt(np.maximum(np.sum(
                            (projected@model.episode_moment_covariance)*projected, axis=1), 0.))
                        scores = np.maximum(means-model.spec.uncertainty_beta*width, 0.)
                        actual = model._values(x, cycle=0)
                        for a, b in zip(actual, (means, width, scores)):
                            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-13)

    def test_action_sequence_rng_and_updates_match_dense_lift(self):
        rng = np.random.default_rng(7182)
        for epsilon in (0., .3, 1.):
            for force in (False, True):
                with self.subTest(epsilon=epsilon, force=force):
                    kwargs = dict(feature_dim=4, config=config(epsilon=epsilon, force=force),
                                  spec=RecursiveLstdqSpec(), seed=181)
                    fast = RecursiveLstdqController(**kwargs)
                    dense = DenseLiftController(**kwargs)
                    for _episode in range(3):
                        for model in (fast, dense):
                            model.start_episode(initial_environment_weight=1.)
                        states = rng.normal(size=(7, 4))
                        pending = [model.select_action(states[0], explore=True, cycle=0)
                                   for model in (fast, dense)]
                        for cycle in range(6):
                            self.assertEqual(pending[0][:2], pending[1][:2])
                            self.assertEqual(fast.rng.bit_generator.state, dense.rng.bit_generator.state)
                            for key in ("greedy_index", "explored", "forced_default_first_action"):
                                self.assertEqual(pending[0][2][key], pending[1][2][key])
                            terminal = cycle == 5
                            next_pending = None if terminal else [
                                model.select_action(states[cycle+1], explore=True, cycle=cycle+1)
                                for model in (fast, dense)]
                            cost = float(rng.uniform(.001, .01))
                            for i, model in enumerate((fast, dense)):
                                model.update(features=states[cycle], action_index=pending[i][0],
                                    cost=cost, next_features=states[cycle+1], terminal=terminal,
                                    next_action_index=None if terminal else next_pending[i][0])
                            for name in ("a_matrix", "a_inverse", "b", "theta", "trace"):
                                np.testing.assert_array_equal(getattr(fast, name), getattr(dense, name))
                            pending = next_pending
                        for model in (fast, dense):
                            model.finish_episode(learned=True)
                        np.testing.assert_array_equal(fast.episode_moment_covariance,
                                                      dense.episode_moment_covariance)

    def test_near_ties_and_clipped_scores_keep_anchor_and_secondary_rule(self):
        for gap, expected in ((.5e-12, 1), (1.5e-12, 0)):
            for clipped in (False, True):
                with self.subTest(gap=gap, clipped=clipped):
                    for cls in (RecursiveLstdqController, DenseLiftController):
                        model = cls(feature_dim=1, config=config(legacy=True, epsilon=0.),
                                    spec=RecursiveLstdqSpec(), seed=9)
                        model.theta[:] = (0., gap, 3e-12)
                        model.episode_moment_covariance[:] = np.eye(3) if clipped else 0.
                        action, _, _ = model.select_action(np.ones(1), explore=False, cycle=0)
                        self.assertEqual(action, expected)

    def test_shape_finiteness_and_negative_variance_checks_remain(self):
        model = RecursiveLstdqController(feature_dim=4, config=config(),
                                        spec=RecursiveLstdqSpec(), seed=3)
        with self.assertRaises(ValueError):
            model._values(np.ones(3), cycle=0)
        with self.assertRaises(FloatingPointError):
            model._values(np.array([1., np.nan, 0., 0.]), cycle=0)
        model.episode_moment_covariance[:] = -np.eye(model.joint_dim)
        with self.assertRaisesRegex(FloatingPointError, "negative action variance"):
            model._values(np.ones(4), cycle=0)


if __name__ == "__main__":
    unittest.main()
