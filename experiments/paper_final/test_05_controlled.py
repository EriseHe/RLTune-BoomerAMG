from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np

from experiments.paper_final.run_05_controlled import Study, match_hierarchies, choose_baselines, paired_comparisons
from experiments.paper_final.timing_selection import TimingSelection
from setup.learners.linucb.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from setup.learners.common import ParameterSpec, ParameterSpaceSpec
from setup.utils.setup_amg import build_actions_from_spec


class ControlledStudyTests(unittest.TestCase):
    @staticmethod
    def learner():
        spec = ParameterSpaceSpec(parameters=(ParameterSpec(name="weight", kind="continuous",
            values=(1., 1.25, 1.5, 1.75, 2.), default=1.5, center=1.5, scale=.5),))
        return SharedLinUCB_AMG_v4(build_actions_from_spec(spec), context_dim=2,
            parameter_spec=spec, context_interaction_indices=(0, 1), seed=7, alpha_decay=True)

    def test_seconds_milliseconds_replay_preserves_decisions_and_search_state(self):
        contexts = np.random.default_rng(14).normal(size=(80, 2))
        contexts[:, 0] = 1.
        for mode in ("raw", "stable", "near_tie"):
            with self.subTest(mode=mode):
                seconds, milliseconds = self.learner(), self.learner()
                seconds.experimental_score_selector = TimingSelection(mode, sigma=.03)
                milliseconds.alpha *= 1000
                milliseconds.experimental_score_selector = TimingSelection(
                    mode, sigma=30., numeric_tolerance=1e-9)
                for index, context in enumerate(contexts):
                    selected = seconds.predict(context)
                    self.assertEqual(selected, milliseconds.predict(context))
                    cost = .2 + .03 * selected["weight"] + .001 * index
                    seconds.update(cost)
                    milliseconds.update(1000 * cost)
                np.testing.assert_allclose(seconds.b * 1000, milliseconds.b, rtol=1e-12)
                np.testing.assert_array_equal(seconds.A_inv, milliseconds.A_inv)
                self.assertEqual(seconds.rng.bit_generator.state, milliseconds.rng.bit_generator.state)

    def test_raw_instrumentation_preserves_legacy_decisions_and_rng(self):
        original, audited = self.learner(), self.learner()
        audited.experimental_score_selector = TimingSelection("raw")
        for i in range(35):
            x = [1., np.cos(i)]
            self.assertEqual(original.predict(x), audited.predict(x))
            original.update(.1 + i * .002)
            audited.update(.1 + i * .002)
            self.assertEqual(original.rng.bit_generator.state, audited.rng.bit_generator.state)

    def test_propagated_pair_width_matches_explicit_design_matrix(self):
        model = self.learner()
        features = []
        for i in range(12):
            model.predict([1., np.sin(i)])
            features.append(model._last_phi.copy())
            model.update(.1 + i * .01)
        x = np.array([1., .3])
        difference = model._phi(x, 0) - model._phi(x, 4)
        projected = model.A_inv @ difference
        implicit = difference @ projected - model.l2_reg * (projected @ projected)
        explicit = np.linalg.norm(np.asarray(features) @ projected) ** 2
        self.assertAlmostEqual(implicit, explicit, places=12)

    def test_fixed_candidate_reference_survives_small_score_perturbation(self):
        model = self.learner()
        for i in range(10):
            model.predict([1., .2 * i])
            model.update(.2)
        rule = TimingSelection("near_tie", sigma=.1)
        model.experimental_score_selector = rule
        arms = np.arange(model.K)
        reference = model.history[-1].arm_index
        best = (reference + 1) % len(arms)
        scores = np.full(len(arms), 1.)
        scores[best], scores[reference] = 0., .00001
        self.assertEqual(rule(model, np.array([1., .7]), arms, scores), reference)
        self.assertEqual(rule(model, np.array([1., .7]), arms, scores + np.linspace(-1e-7, 1e-7, len(arms))), reference)

    @staticmethod
    def row(policy="fixed_1", fingerprint="abc", cost=.2, phase="selection"):
        return {"phase": phase, "source": "source", "policy": policy, "index": 0,
            "source_decision_runtime": .01, "outcome": {"hierarchy_fingerprint": fingerprint,
                "initial_cycle": 0, "primary_status": "success", "failed": False, "iterations": 3,
                "setup_runtime": .1, "solve_control_runtime": cost,
                "end_to_end_runtime": cost + .1, "algorithm_runtime": cost + .1}}

    def test_matching_checks_numerical_fingerprint_and_initial_state(self):
        rows = [self.row(), self.row("rl", fingerprint="different")]
        with self.assertRaises(AssertionError):
            match_hierarchies(rows)
        rows[1]["outcome"]["hierarchy_fingerprint"] = "abc"
        rows[1]["outcome"]["initial_cycle"] = 1
        with self.assertRaises(AssertionError):
            match_hierarchies(rows)

    def test_shared_setup_cancels_but_recovery_cost_does_not(self):
        rows = [self.row(cost=.2), self.row("rl", cost=.15)]
        rows[1]["outcome"]["setup_runtime"] = .13
        match_hierarchies(rows)
        self.assertAlmostEqual(rows[0]["outcome"]["logical_end_to_end_runtime"] -
                               rows[1]["outcome"]["logical_end_to_end_runtime"], .05)

    def test_baseline_selection_never_uses_heldout_outcomes(self):
        rows = [self.row("fixed_1", cost=.2), self.row("fixed_2", cost=.3),
                self.row("schedule_a", cost=.2), self.row("schedule_b", cost=.3),
                self.row("fixed_2", cost=.001, phase="evaluation")]
        self.assertEqual(choose_baselines(rows, ["source"]),
                         {"source": {"fixed": "fixed_1", "schedule": "schedule_a"}})

    def test_frozen_recommendation_does_not_advance_learning_or_rng(self):
        model = self.learner()
        model.predict([1., .2])
        model.update(.3)
        before = copy.deepcopy(model)
        for x in ([1., .7], [1., -.2]):
            model.recommend(x, candidate_arms=np.arange(model.K), alpha=model._effective_alpha())
        np.testing.assert_array_equal(model.A_inv, before.A_inv)
        np.testing.assert_array_equal(model.b, before.b)
        self.assertEqual(model.t, before.t)
        self.assertEqual(model.rng.bit_generator.state, before.rng.bit_generator.state)

    def test_controller_setup_failure_gets_one_default_fallback(self):
        study = object.__new__(Study)
        study.fallback = Mock(return_value={"failed": False, "attempt_status": "success",
            "runtime": .3, "setup_runtime": .2, "solve_runtime": .1,
            "residual_norm": 1e-8, "iterations": 8})
        bundle = SimpleNamespace(run_case=lambda case: {"failed": True, "attempt_status": "setup_failure",
            "runtime": .01, "native_runtime": .01, "setup_runtime": .01, "solve_runtime": 0.,
            "native_solve_runtime": 0., "infer_runtime": 0., "residual_norm": 1., "iterations": 0,
            "recovery_protocol_applied": False})
        outcome = study.solve({}, {}, bundle=bundle)
        study.fallback.assert_called_once()
        self.assertFalse(outcome["failed"])
        self.assertAlmostEqual(outcome["end_to_end_runtime"], .31)
        self.assertAlmostEqual(outcome["solve_control_runtime"], .3)

    def test_paired_report_does_not_drop_unmatched_or_failed_cases(self):
        rows = [self.row("fixed_1", cost=.2, phase="evaluation"),
                self.row("rl", cost=.3, phase="evaluation")]
        rows[1]["outcome"]["failed"] = True
        result = paired_comparisons(rows)
        self.assertEqual(result[0]["cases"], 1)
        self.assertAlmostEqual(result[0]["saved_sec"], -.1)
        rows[1]["index"] = 1
        with self.assertRaises(AssertionError):
            paired_comparisons(rows)


if __name__ == "__main__":
    unittest.main()
