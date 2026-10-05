from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import json
from pathlib import Path
import tempfile
import unittest

from experiments.diagnostics.solve_control.prepare_lstdq_v3_joint_smoke import prepare_config
from experiments.joint.solve_control.run_joint_experiment import parse_joint_experiment_config


class PrepareLstdqV3JointSmokeTests(unittest.TestCase):
    def test_prepared_config_requires_gates_and_resolves_typed_v3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calibration_path = root / "calibration.json"
            stability_path = root / "stability.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "selected": {
                            "lstdq_v3": {
                                "method": "lstdq_v3_beta_2",
                                "beta": 2.0,
                                "eligible": True,
                            }
                        },
                        "candidates": {
                            "lstdq_v3_beta_2": {
                                "v3_calibration_gate": {"passed": True}
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            stability_path.write_text(
                json.dumps(
                    {
                        "stability": {
                            "acceptance": {"passed": True}
                        }
                    }
                ),
                encoding="utf-8",
            )
            config = prepare_config(
                calibration_result_path=calibration_path,
                stability_result_path=stability_path,
                experiment_output_dir=root / "joint",
            )

        typed = parse_joint_experiment_config(config)
        self.assertEqual(typed.stream.cases, 500)
        self.assertTrue(typed.stream.smoke)
        self.assertEqual(
            tuple(method.solve_kind for method in typed.methods),
            ("default", "recursive_lstdq_v2", "recursive_lstdq_v3"),
        )
        self.assertEqual(
            typed.solve.recursive_lstdq_v3.uncertainty_beta,
            2.0,
        )


if __name__ == "__main__":
    unittest.main()
