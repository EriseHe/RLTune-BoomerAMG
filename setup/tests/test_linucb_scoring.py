"""Numerical and selection-contract checks for exact block scoring."""

from __future__ import annotations

from setup.tests import _project_paths  # noqa: F401

import copy
from types import MethodType
import unittest

import numpy as np

from experiments.paper_final.recovery.retry_comparison import use_retry_rule
from setup.learners.common import ParameterSpaceSpec, ParameterSpec
from setup.learners.linucb.learner import SharedLinUCB
from setup.space import build_actions_from_spec


def dense_lift_reference(model, x, *, arms, action_features=None):
    G = action_features if action_features is not None else (
        model._g_actions if arms is None else model._features_for_arms(arms))
    c = np.zeros(model.d_phi)
    c[:model.d_x] = x
    identity = np.eye(model.g_dim)
    lift = np.vstack([np.zeros((model.d_x, model.g_dim)), identity,
                      *(x[i]*identity for i in model.context_interaction_indices)])
    Ac = model.A_inv@c
    u = lift.T@Ac
    M = lift.T@model.A_inv@lift
    quad = c@Ac + 2*G@u + np.einsum("ij,jk,ik->i", G, M, G, optimize=False)
    mean = model._linear_prediction_subset(x, theta=model.A_inv@model.b,
                                          arms=arms, action_features=G)
    return mean, np.sqrt(np.maximum(0., quad))


class LinUCBScoringTests(unittest.TestCase):
    def test_matches_explicit_features_for_both_experiment_dimensions(self):
        rng = np.random.default_rng(21)
        for dx, indices in ((4, (1, 2, 3)), (7, (1, 2, 3, 4, 5, 6)),
                            (3, ()), (3, (2, 0, 2))):
            with self.subTest(context_dim=dx, interactions=indices):
                model = object.__new__(SharedLinUCB)
                model.d_x, model.g_dim = dx, 104
                model.context_interaction_indices = indices
                model.d_phi = dx + (len(indices)+1)*104
                model._g_actions = rng.normal(size=(9, 104))
                factor = rng.normal(size=(model.d_phi, 23))/np.sqrt(23)
                model.A_inv = factor@factor.T + np.eye(model.d_phi)*.1
                model.b = rng.normal(size=model.d_phi)
                x = rng.normal(size=dx)
                arms = np.array([5, 0, 5, 8])
                G = model._g_actions[arms]
                phi = np.array([model._phi_from_feature(x, g) for g in G])
                expected_mean = phi@(model.A_inv@model.b)
                expected_width = np.sqrt(np.sum((phi@model.A_inv)*phi, axis=1))
                for kwargs in (dict(arms=arms), dict(arms=None, action_features=G)):
                    mean, width = model._mean_uncertainty_subset(x, **kwargs)
                    np.testing.assert_allclose(mean, expected_mean, rtol=2e-12, atol=2e-12)
                    np.testing.assert_allclose(width, expected_width, rtol=2e-12, atol=2e-12)
                full_mean, full_width = model._mean_uncertainty_subset(x, arms=None)
                np.testing.assert_allclose(full_mean[arms], expected_mean, atol=2e-12)
                np.testing.assert_allclose(full_width[arms], expected_width, atol=2e-12)
                # Do not silently assume symmetry: preserve the legacy cross term.
                model.A_inv += rng.normal(scale=.001, size=model.A_inv.shape)
                actual = model._mean_uncertainty_subset(x, arms=arms)
                expected = dense_lift_reference(model, x, arms=arms)
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)

    @staticmethod
    def small_model():
        spec = ParameterSpaceSpec(parameters=(ParameterSpec(
            name="weight", kind="continuous", values=(-1., 0., 1.),
            default=0., center=0., scale=1.),))
        return SharedLinUCB(build_actions_from_spec(spec), context_dim=2,
            parameter_spec=spec, context_interaction_indices=(1,), seed=71)

    def test_roundoff_guard_and_factorized_full_space_error(self):
        model = self.small_model()
        model.A_inv *= -1
        mean, width = model._mean_uncertainty_subset(np.array([1., .3]), arms=None)
        np.testing.assert_array_equal(width, np.zeros(model.K))
        self.assertTrue(np.isfinite(mean).all())
        model._g_actions = None
        with self.assertRaisesRegex(RuntimeError, "full-space scoring"):
            model._mean_uncertainty_subset(np.array([1., .3]), arms=None)

    def test_near_ties_preserve_primary_and_failure_recovery_choices(self):
        context = np.array([1., .3])
        for coefficient, expected_ties in ((.2e-12, [0, 2]), (.7e-12, [0])):
            for recovery in (False, True):
                with self.subTest(coefficient=coefficient, recovery=recovery):
                    model = use_retry_rule(self.small_model(), "failure_model_adjacent")
                    # Opposite action values have identical width. Their score
                    # gap falls just inside or just outside the 1e-12 tie rule.
                    model.b[model.d_x] = coefficient
                    model.failure_b[model.d_x] = coefficient
                    model.failure_b[model.d_x+1] = -10.
                    reference = model.clone_for_independent_updates()
                    reference.failure_b = model.failure_b.copy()
                    reference._mean_uncertainty_subset = MethodType(dense_lift_reference, reference)
                    arms = np.arange(model.K)
                    score, mean, width = model._score_subset(context, arms=arms, alpha=model.alpha)
                    if recovery:
                        ties = model._recovery_best_locations(context, candidates=arms,
                            candidate_features=model._g_actions, runtime_score=score,
                            runtime_mean=mean, runtime_uncert=width)
                    else:
                        ties = np.flatnonzero(score <= np.min(score)+1e-12)
                    np.testing.assert_array_equal(ties, expected_ties)
                    rng_state = copy.deepcopy(model.rng.bit_generator.state)
                    for candidate in (model, reference):
                        candidate.rng.bit_generator.state = copy.deepcopy(rng_state)
                        if recovery:
                            candidate.select_recovery(context, excluded_arms=[])
                        else:
                            candidate.predict(context)
                    self.assertEqual(model._last_arm, reference._last_arm)
                    self.assertEqual(model.rng.bit_generator.state, reference.rng.bit_generator.state)


if __name__ == "__main__":
    unittest.main()
