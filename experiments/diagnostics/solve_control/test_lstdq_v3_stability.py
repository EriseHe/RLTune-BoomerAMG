from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import unittest

from experiments.diagnostics.solve_control.run_lstdq_v3_stability import (
    V2_METHOD,
    V3_METHOD,
    _summarize_stability,
)


def _summary(runtime: float, failures: int = 0):
    return {
        "means_sec": {"end_to_end_runtime": float(runtime)},
        "primary_failure_count": int(failures),
        "unrecovered_failure_count": 0,
    }


class LstdqV3StabilityTests(unittest.TestCase):
    def test_legacy_case_helper_aliases_shared_execution(self) -> None:
        from experiments.diagnostics.solve_control import run_lstdq_v3_stability
        from experiments.joint.solve_control.native_case import run_case

        self.assertIs(run_lstdq_v3_stability._run_case, run_case)

    def test_stability_gate_uses_spread_and_failure_excess(self) -> None:
        repetitions = [
            {
                "held_out": {
                    V2_METHOD: _summary(1.00, 2),
                    V3_METHOD: _summary(v3_runtime, 3),
                },
                "finite_controller": {
                    V2_METHOD: True,
                    V3_METHOD: True,
                },
            }
            for v3_runtime in (0.98, 1.00, 1.02)
        ]
        result = _summarize_stability(repetitions)
        self.assertTrue(result["acceptance"]["passed"])
        self.assertLessEqual(
            result["methods"][V3_METHOD]["relative_max_min_spread"],
            0.05,
        )

        repetitions[-1]["held_out"][V3_METHOD] = _summary(1.20, 10)
        result = _summarize_stability(repetitions)
        self.assertFalse(result["acceptance"]["passed"])


if __name__ == "__main__":
    unittest.main()
