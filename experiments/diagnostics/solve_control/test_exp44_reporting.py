from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP44_DIR = Path(__file__).resolve().parents[2] / "joint" / "exp44"
if str(EXP44_DIR) not in sys.path:
    sys.path.insert(0, str(EXP44_DIR))

from aggregate_forward_continuation_from_run import _normalized_method  # noqa: E402
from report_exp44_results import report_evaluation  # noqa: E402


def _method(
    native: float,
    *,
    setup: float,
    solve: float,
    controller: float = 0.0,
    bandit: float = 0.0,
) -> dict:
    cases = 2
    return {
        "cases": cases,
        "mean_runtime": native,
        "total_runtime": native * cases,
        "mean_setup_runtime": setup,
        "total_setup_runtime": setup * cases,
        "mean_solve_runtime": solve,
        "total_solve_runtime": solve * cases,
        "mean_infer_runtime": controller,
        "total_infer_runtime": controller * cases,
        "mean_runtime_with_controller": native + controller,
        "total_runtime_with_controller": (native + controller) * cases,
        "mean_bandit_overhead_runtime": bandit,
        "total_bandit_overhead_runtime": bandit * cases,
        "mean_end_to_end_runtime": native + controller + bandit,
        "total_end_to_end_runtime": (native + controller + bandit) * cases,
        "failed_count": 0,
    }


class Exp44ReportingTests(unittest.TestCase):
    def test_normalized_method_recovers_controller_inclusive_total(self) -> None:
        row = _normalized_method(
            {
                "cases": 4,
                "mean_runtime": 0.08,
                "mean_setup_runtime": 0.03,
                "mean_solve_runtime": 0.05,
                "mean_infer_runtime": 0.004,
                "failed_count": 0,
            },
            fallback_cases=4,
        )
        self.assertAlmostEqual(row["total_runtime"], 0.32)
        self.assertAlmostEqual(row["total_infer_runtime"], 0.016)
        self.assertAlmostEqual(row["mean_runtime_with_controller"], 0.084)
        self.assertAlmostEqual(row["mean_bandit_overhead_runtime"], 0.0)
        self.assertAlmostEqual(row["mean_end_to_end_runtime"], 0.084)

    def test_evaluation_report_writes_table_and_all_figures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "seed.json"
            source.write_text(
                json.dumps(
                    {
                        "windows": {
                            "eval_1000": {
                                "methods": {
                                    "ppo_best": {
                                        "per_case_mean_w": [1.4, 1.5],
                                        "mean_w_by_cycle": [1.3, 1.6],
                                    }
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            methods = {
                "default_setup_default_solve": _method(0.11, setup=0.07, solve=0.04),
                "bandit_only": _method(
                    0.09, setup=0.03, solve=0.06, bandit=0.005
                ),
                "fixed_w_1.60": _method(
                    0.08, setup=0.03, solve=0.05, bandit=0.005
                ),
                "ppo_best": _method(
                    0.078,
                    setup=0.03,
                    solve=0.048,
                    controller=0.004,
                    bandit=0.005,
                ),
            }
            result = root / "result.json"
            result.write_text(
                json.dumps(
                    {
                        "windows": {
                            "eval_1000": {
                                "combined": methods,
                                "rows": [
                                    {
                                        "seed": 7,
                                        "source_path": str(source),
                                        "methods": methods,
                                    }
                                ],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            report = report_evaluation(result, root / "report", window="eval_1000")
            self.assertTrue(Path(report["table"]).is_file())
            self.assertEqual(len(report["figures"]), 4)
            self.assertTrue(all(Path(path).is_file() for path in report["figures"]))
            header = Path(report["table"]).read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("native_solve_ms_per_instance", header)
            self.assertIn("controller_inclusive_ms_per_instance", header)
            self.assertIn("bandit_overhead_ms_per_instance", header)
            self.assertIn("end_to_end_ms_per_instance", header)


if __name__ == "__main__":
    unittest.main()
