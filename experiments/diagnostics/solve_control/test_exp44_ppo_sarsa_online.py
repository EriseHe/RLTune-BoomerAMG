from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parents[1]
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from run_exp44_ppo_sarsa_online import (  # noqa: E402
    _clustered_metric,
    _record,
    _trace_summary,
    run,
)
from plot_exp44_ppo_sarsa_online import _paired_improvement_ci  # noqa: E402


def _row(value: float, *, params: dict | None = None) -> dict:
    native = float(value)
    return {
        "params": dict(params or {"strong_threshold": 0.25}),
        "outcome": {
            "runtime": native,
            "setup_runtime": 0.25 * native,
            "solve_runtime": 0.75 * native,
            "infer_runtime": 0.01 * native,
            "end_to_end_runtime": 1.01 * native,
            "iterations": 10,
            "failed": False,
        },
    }


class Exp44PpoSarsaOnlineTests(unittest.TestCase):
    def test_paired_plot_metric_uses_positive_for_faster_ppo(self) -> None:
        point, interval = _paired_improvement_ci(
            np.asarray([2.0, 4.0]),
            np.asarray([1.0, 2.0]),
            seed=7,
        )
        self.assertAlmostEqual(point, 50.0)
        self.assertAlmostEqual(interval[0], 50.0)
        self.assertAlmostEqual(interval[1], 50.0)

    def test_clustered_metric_preserves_positive_improvement(self) -> None:
        seed_records = []
        for scale in (1.0, 2.0):
            seed_records.append(
                {
                    "reference": [_row(scale), _row(2.0 * scale)],
                    "candidate": [_row(0.9 * scale), _row(1.8 * scale)],
                }
            )
        result = _clustered_metric(
            seed_records,
            reference="reference",
            candidate="candidate",
            accessor=lambda row: float(row["outcome"]["runtime"]),
            seed=7,
        )
        self.assertEqual(result["cases"], 4)
        self.assertEqual(result["forward_seed_clusters"], 2)
        self.assertAlmostEqual(result["candidate_improvement_pct"], 10.0)
        self.assertGreater(result["candidate_improvement_95pct"][0], 0.0)

    def test_trace_summary_separates_fallback_and_bandit_timing(self) -> None:
        records = [
            {
                "bandit_timing": {
                    "select_sec": 0.1,
                    "loss_eval_sec": 0.2,
                    "update_sec": 0.3,
                    "overhead_sec": 0.6,
                },
                "feedback_outcome": {
                    "runtime": 2.0,
                    "setup_runtime": 0.5,
                    "solve_runtime": 1.5,
                    "failed": False,
                    "fallback_used": True,
                },
            },
            {
                "bandit_timing": {
                    "select_sec": 0.4,
                    "loss_eval_sec": 0.5,
                    "update_sec": 0.6,
                    "overhead_sec": 1.5,
                },
                "feedback_outcome": {
                    "runtime": 3.0,
                    "setup_runtime": 1.0,
                    "solve_runtime": 2.0,
                    "failed": True,
                    "fallback_used": False,
                },
            },
        ]
        summary = _trace_summary(records)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["fallback_uses"], 1)
        self.assertEqual(summary["feedback_failures"], 1)
        self.assertAlmostEqual(summary["bandit_timing_totals_sec"]["overhead_sec"], 2.1)
        self.assertAlmostEqual(summary["feedback_totals_sec"]["solve_runtime"], 3.5)

    def test_record_keeps_native_and_controller_fields_separate(self) -> None:
        outcome = _row(1.0)["outcome"]
        record = _record(
            seed=1,
            instance_index=1500,
            phase="evaluation_1000",
            mkw={"k": 1.0},
            params={"strong_threshold": 0.25},
            outcome=outcome,
            execution_order=("bandit_ppo", "bandit_sarsa"),
        )
        self.assertEqual(record["instance_index"], 1500)
        self.assertEqual(record["outcome"]["runtime"], 1.0)
        self.assertEqual(record["outcome"]["infer_runtime"], 0.01)
        self.assertEqual(record["outcome"]["end_to_end_runtime"], 1.01)

    def test_formal_runner_rejects_non_1000_window(self) -> None:
        args = argparse.Namespace(
            eval_start=3,
            eval_end=5,
            trace_t=5,
            allow_noncanonical_smoke=False,
        )
        with self.assertRaisesRegex(ValueError, "1000-case"):
            run(args)


if __name__ == "__main__":
    unittest.main()
