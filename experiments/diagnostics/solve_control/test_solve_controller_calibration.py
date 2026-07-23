from __future__ import annotations

import _project_paths  # noqa: F401

from types import SimpleNamespace
import unittest

from run_solve_controller_calibration import (
    MODEL_METHOD,
    V2_CALIBRATION_BASELINE,
    _apply_v3_calibration_gates,
    _build_controllers,
    _choose_candidate,
    _longest_excess_failure_chain,
)
from solve.controllers.common import ControllerBundle


def _summary(*, total: float, overhead: float, eligible: bool = True):
    return {
        "eligible": bool(eligible),
        "means_sec": {
            "end_to_end_runtime": float(total),
            "controller_runtime": float(overhead),
        },
    }


def _row(status: str):
    return {"outcome": {"primary_status": str(status)}}


class SolveControllerCalibrationTests(unittest.TestCase):
    def test_controller_grid_uses_typed_bundles(self) -> None:
        bundles, betas = _build_controllers(
            SimpleNamespace(
                tol=1.0e-6,
                max_cycles=50,
                c_max=1000.0,
                controller_seed=41966039,
            )
        )

        self.assertEqual(len(bundles), 10)
        self.assertTrue(
            all(
                isinstance(bundle, ControllerBundle)
                for bundle in bundles.values()
            )
        )
        self.assertEqual(
            bundles["lstdq_v3_beta_2"].controller.spec.uncertainty_beta,
            2.0,
        )
        self.assertEqual(
            bundles["lstdq_v2_beta_4"].controller.spec.uncertainty_beta,
            4.0,
        )
        self.assertEqual(
            bundles[
                "recalibrated_lsvi_beta_1"
            ].controller.spec.uncertainty_beta,
            1.0,
        )
        self.assertIsNone(betas[MODEL_METHOD])

    def test_v3_gate_requires_calibration_runtime_and_overhead(self) -> None:
        def candidate(
            *,
            total: float,
            overhead: float,
            fraction_4x: float,
            median_ratio: float,
        ):
            summary = _summary(total=total, overhead=overhead)
            summary["action_diagnostics"] = {
                "residual_to_parameter_width": {
                    "fraction_within_4x_parameter_width": fraction_4x,
                    "median_abs_residual_over_parameter_width": median_ratio,
                }
            }
            return summary

        summaries = {
            V2_CALIBRATION_BASELINE: candidate(
                total=1.0,
                overhead=0.020,
                fraction_4x=0.20,
                median_ratio=12.0,
            ),
            "lstdq_v3_beta_1": candidate(
                total=1.01,
                overhead=0.024,
                fraction_4x=0.75,
                median_ratio=3.5,
            ),
            "lstdq_v3_beta_2": candidate(
                total=1.03,
                overhead=0.024,
                fraction_4x=0.75,
                median_ratio=3.5,
            ),
            "lstdq_v3_beta_4": candidate(
                total=1.01,
                overhead=0.030,
                fraction_4x=0.60,
                median_ratio=8.0,
            ),
        }
        _apply_v3_calibration_gates(summaries)
        self.assertTrue(summaries["lstdq_v3_beta_1"]["eligible"])
        self.assertFalse(summaries["lstdq_v3_beta_2"]["eligible"])
        self.assertFalse(summaries["lstdq_v3_beta_4"]["eligible"])
        self.assertIn(
            "fraction_within_4x_below_0.70",
            summaries["lstdq_v3_beta_4"]["v3_calibration_gate"][
                "reasons"
            ],
        )

    def test_half_percent_tie_uses_overhead_then_smaller_beta(self) -> None:
        summaries = {
            "beta_1": _summary(total=1.004, overhead=0.010),
            "beta_2": _summary(total=1.000, overhead=0.020),
            "beta_4": _summary(total=1.020, overhead=0.001),
        }
        selected = _choose_candidate(
            summaries,
            summaries=summaries,
            betas={"beta_1": 1.0, "beta_2": 2.0, "beta_4": 4.0},
        )
        self.assertEqual(selected, "beta_1")

    def test_ineligible_candidate_cannot_win(self) -> None:
        summaries = {
            "invalid": _summary(total=0.1, overhead=0.0, eligible=False),
            "valid": _summary(total=1.0, overhead=0.1),
        }
        selected = _choose_candidate(
            summaries,
            summaries=summaries,
            betas={"invalid": 1.0, "valid": 2.0},
        )
        self.assertEqual(selected, "valid")

    def test_failure_chain_counts_only_controller_excess_failures(self) -> None:
        candidate = [
            _row("nonconvergence"),
            _row("nonconvergence"),
            _row("nonconvergence"),
            _row("success"),
        ]
        baseline = [
            _row("nonconvergence"),
            _row("success"),
            _row("success"),
            _row("success"),
        ]
        self.assertEqual(
            _longest_excess_failure_chain(candidate, baseline),
            2,
        )


if __name__ == "__main__":
    unittest.main()
