from __future__ import annotations

import unittest

from experiments.diagnostics.solve_control.diagnose_max_cycle_headroom import (
    recorded_action_tail_schedule,
    summarize_replays,
)


class MaxCycleHeadroomTest(unittest.TestCase):
    def test_recorded_action_tail_schedule_compresses_and_extends(self) -> None:
        self.assertEqual(
            recorded_action_tail_schedule([1.0, 1.0, 2.0, 2.0, 1.5], 8),
            (
                (2, 1.0, 1, 1),
                (4, 2.0, 1, 1),
                (5, 1.5, 1, 1),
                (8, 1.5, 1, 1),
            ),
        )

    def test_summarize_replays_reports_rescue_rate_and_runtime(self) -> None:
        rows = [
            {
                "extended_converged": True,
                "extended_iterations": 58,
                "iteration_bucket": "51-60",
                "original_recovered": True,
                "estimated_native_runtime_delta_vs_original_protocol_sec": -0.2,
                "recorded_tail_geometric_residual_ratio": 0.8,
            },
            {
                "extended_converged": False,
                "extended_iterations": 100,
                "iteration_bucket": "still_failed_at_extended_cap",
                "original_recovered": False,
                "estimated_native_runtime_delta_vs_original_protocol_sec": 0.3,
                "recorded_tail_geometric_residual_ratio": 1.0,
            },
        ]

        summary = summarize_replays(rows)

        self.assertEqual(summary["cases_at_original_cap"], 2)
        self.assertEqual(summary["converged_by_extended_cap"], 1)
        self.assertEqual(summary["still_failed_at_extended_cap"], 1)
        self.assertAlmostEqual(summary["rescue_rate"], 0.5)
        self.assertAlmostEqual(
            summary["estimated_native_runtime_delta_total_sec"],
            0.1,
        )
        self.assertAlmostEqual(
            summary["recorded_tail_geometric_residual_ratio_median"],
            0.9,
        )


if __name__ == "__main__":
    unittest.main()
