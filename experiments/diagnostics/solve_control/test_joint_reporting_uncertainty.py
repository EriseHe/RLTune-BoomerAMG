from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import unittest

from experiments.joint.solve_control.joint_reporting import _action_summary


class JointReportingUncertaintyTests(unittest.TestCase):
    def test_td_residual_ratio_is_not_labeled_predictive_coverage(self) -> None:
        action = _action_summary(
            [
                {
                    "outcome": {
                        "cycle_actions": [1.5],
                        "cycle_explored": [False],
                        "cycle_selected_uncertainties": [0.01],
                        "cycle_selected_q_values": [0.02],
                        "cycle_selection_scores": [0.0],
                        "postfit_td_errors": [0.02],
                    }
                }
            ]
        )
        diagnostic = action["residual_to_parameter_width"]
        self.assertEqual(
            diagnostic["fraction_within_2x_parameter_width"],
            1.0,
        )
        self.assertIn("not nominal", diagnostic["semantics"])
        self.assertIn("confidence_calibration", action)
        self.assertIn(
            "deprecated_semantics",
            action["confidence_calibration"],
        )


if __name__ == "__main__":
    unittest.main()
