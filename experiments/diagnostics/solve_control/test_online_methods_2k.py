from __future__ import annotations

import _project_paths  # noqa: F401

import json
import unittest
from types import SimpleNamespace

import numpy as np

from run_online_bandit_td_lambda import _build_paired_instance_stream
from run_online_methods_2k import (
    _as_feedback,
    _continuous_action_diagnostics,
    _paired_metric,
)
from setup_aware_compare_common import generate_difconv_instances


class OnlineMethods2KTests(unittest.TestCase):
    def test_grouped_stream_applies_offset_to_each_seed(self) -> None:
        args = SimpleNamespace(
            train_seed_groups="101,102;103,104",
            train_shuffle_seeds="201,202",
            train_cases_per_seed=2,
            train_group_take=4,
            train_cases=8,
            instance_offset=3,
            grid_n=4,
            c_min=1.0,
            c_max=10.0,
            seed=1,
        )
        stream, metadata = _build_paired_instance_stream(args)
        actual = {
            json.dumps(mkw, sort_keys=True)
            for mkw, _context in stream
        }
        expected = set()
        for seed in (101, 102, 103, 104):
            generated = generate_difconv_instances(
                T=5,
                seed=seed,
                grid_choices=[(4, 4, 4)],
                c_min=1.0,
                c_max=10.0,
                difconv_a=(0.0, 0.0, 0.0),
            )
            expected.update(json.dumps(mkw, sort_keys=True) for mkw, _context in generated[3:])
        self.assertEqual(actual, expected)
        self.assertEqual(metadata["segments"][0]["window"], [3, 5])
        self.assertEqual(metadata["segments"][1]["window"], [3, 5])

    def test_final_2k_stream_matches_locked_protocol_hash(self) -> None:
        args = SimpleNamespace(
            train_seed_groups="39394939,39400939;39406939,39412939",
            train_shuffle_seeds="39394016,39394022",
            train_cases_per_seed=500,
            train_group_take=1000,
            train_cases=2000,
            instance_offset=1500,
            grid_n=40,
            c_min=1.0,
            c_max=1000.0,
            seed=39394016,
        )
        stream, metadata = _build_paired_instance_stream(args)
        self.assertEqual(len(stream), 2000)
        self.assertEqual(
            metadata["sha256"],
            "33c8dbef37630eeea75607a69376e37dd0fbc777e0e14b704e4f9ab30f3dac41",
        )
        self.assertEqual(metadata["segments"][0]["window"], [1500, 2000])
        self.assertEqual(metadata["segments"][1]["window"], [1500, 2000])

    def test_feedback_keeps_native_and_controller_costs_separate(self) -> None:
        native = {
            "runtime": 0.08,
            "setup_runtime": 0.03,
            "solve_runtime": 0.05,
            "infer_runtime": 0.002,
        }
        feedback = _as_feedback(native, include_controller=True)
        self.assertAlmostEqual(feedback["native_runtime"], 0.08)
        self.assertAlmostEqual(feedback["native_solve_runtime"], 0.05)
        self.assertAlmostEqual(feedback["runtime"], 0.082)
        self.assertAlmostEqual(feedback["solve_runtime"], 0.052)

    def test_paired_metric_positive_means_candidate_is_faster(self) -> None:
        metric = _paired_metric(
            np.asarray([2.0, 2.0]),
            np.asarray([1.0, 1.0]),
            seed=7,
        )
        self.assertAlmostEqual(metric["candidate_improvement_pct"], 50.0)
        self.assertAlmostEqual(metric["candidate_savings_sec"], 2.0)
        self.assertEqual(metric["candidate_improvement_95pct"], [50.0, 50.0])

    def test_paired_metric_handles_zero_baseline_without_division(self) -> None:
        metric = _paired_metric(
            np.asarray([0.0, 0.0]),
            np.asarray([0.1, 0.2]),
            seed=8,
        )
        self.assertTrue(np.isnan(metric["candidate_improvement_pct"]))
        self.assertTrue(np.isnan(metric["candidate_improvement_95pct"][0]))
        self.assertAlmostEqual(metric["candidate_savings_sec"], -0.3)

    def test_continuous_action_diagnostics_preserves_cycle_quantiles(self) -> None:
        diagnostics = _continuous_action_diagnostics(
            [
                {"cycle_actions": [1.5, 1.52]},
                {"cycle_actions": [1.48, 1.50]},
            ]
        )
        self.assertEqual(diagnostics["by_cycle"]["0"]["count"], 2)
        self.assertAlmostEqual(diagnostics["by_cycle"]["0"]["mean"], 1.49)
        self.assertAlmostEqual(diagnostics["by_cycle"]["1"]["median"], 1.51)


if __name__ == "__main__":
    unittest.main()
