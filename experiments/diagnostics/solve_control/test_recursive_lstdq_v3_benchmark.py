from __future__ import annotations

import _project_paths  # noqa: F401

import unittest

from bench_recursive_lstdq_v3 import benchmark


class RecursiveLstdqV3BenchmarkTests(unittest.TestCase):
    def test_benchmark_uses_production_dimensions_and_reports_gate(self) -> None:
        result = benchmark(
            decision_repetitions=2,
            training_episodes=2,
            cycles_per_episode=2,
            seed=91,
        )
        self.assertEqual(
            result["dimensions"],
            {
                "state_features": 32,
                "action_basis": 9,
                "actions": 41,
                "joint_features": 288,
            },
        )
        self.assertEqual(set(result["measurements"]), {"v2", "v3"})
        self.assertIn(
            "passes_locked_overhead_gate",
            result["v3_vs_v2"],
        )


if __name__ == "__main__":
    unittest.main()
