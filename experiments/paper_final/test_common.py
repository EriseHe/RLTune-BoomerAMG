"""Shared accounting, durable resume, and provenance contracts for paper runs."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hypre.bindings import AttemptOutcome, RecoveryOutcome
from experiments.paper_final.common import artifacts, frozen_policy, policy_analysis


def timing_row(case, policy, cost, *, success=True, repeat=0):
    return {
        "seed": 1,
        "source": "bandit_lstdq",
        "case_id": case,
        "policy": policy,
        "kind": "fixed",
        "success": success,
        "recovery": False,
        "repeat": repeat,
        **{field: cost for field in policy_analysis.COST_FIELDS},
        "setup_sec": 0.0,
        "solve_sec": cost,
        "controller_sec": 0.0,
        "recovery_setup_sec": 0.0,
        "primary_cycles": 1,
        "fallback_cycles": 0,
    }


class AccountingTests(unittest.TestCase):
    def test_fixed_hindsight_uses_sum_of_case_costs(self):
        rows = [
            timing_row(0, "w1", 1),
            timing_row(1, "w1", 9),
            timing_row(0, "w2", 4),
            timing_row(1, "w2", 4),
        ]
        result = policy_analysis.fixed_hindsight(rows)
        self.assertEqual(result["best_fixed"]["policy"], "w2")
        self.assertEqual([row["policy"] for row in result["per_case"]], ["w1", "w2"])

    def test_all_repetitions_contribute_and_one_failure_disqualifies_fixed(self):
        rows = [
            timing_row(0, "w1", cost, repeat=repeat, success=repeat != 2)
            for repeat, cost in enumerate((1, 2, 9))
        ]
        averaged = policy_analysis.average_repetitions(rows)
        self.assertEqual(averaged[0]["native_continuation_sec"], 4.0)
        self.assertFalse(averaged[0]["success"])
        self.assertIsNone(policy_analysis.fixed_hindsight(averaged)["best_fixed"])

    def test_rl_setup_failure_retains_failed_cost_and_reference_recovery(self):
        primary = {
            "runtime": 0.23,
            "native_runtime": 0.2,
            "setup_runtime": 0.2,
            "solve_runtime": 0.03,
            "native_solve_runtime": 0.0,
            "infer_runtime": 0.03,
            "failed": True,
            "attempt_status": "setup_failure",
            "failure_origin": "solver",
            "residual_norm": float("nan"),
            "iterations": 0,
        }
        reported = frozen_policy.report_online_outcome(
            RecoveryOutcome(primary=AttemptOutcome.from_mapping(primary)).to_result(),
            bandit_timing={},
        )
        fallback = {
            "runtime": 0.3,
            "setup_runtime": 0.1,
            "solve_runtime": 0.2,
            "failed": False,
            "attempt_status": "success",
            "residual_norm": 1e-8,
            "iterations": 5,
        }
        job = {
            "seed": 1,
            "source": "bandit_lstdq",
            "case_id": 0,
            "input_id": "a",
            "hierarchy_id": "h",
            "mkw": {},
        }
        with (
            patch.object(frozen_policy, "_run_case", return_value=reported),
            patch.object(
                frozen_policy, "solve_no_rl_case", return_value=fallback
            ) as recovery,
        ):
            result = frozen_policy.run_policy(job, "rl", {"kind": "rl"}, None)
        recovery.assert_called_once()
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["native_continuation_sec"], 0.3)
        self.assertAlmostEqual(result["native_total_sec"], 0.5)
        self.assertAlmostEqual(result["inclusive_total_sec"], 0.53)

    def test_execution_error_aborts_instead_of_becoming_a_numerical_result(self):
        outcome = {
            "failed": True,
            "fallback_used": False,
            "failure_origin": "execution",
            "failure_reason": "exception:TypeError:incorrect call",
        }
        with patch.object(frozen_policy, "_run_case", return_value=outcome):
            with self.assertRaisesRegex(RuntimeError, "execution"):
                frozen_policy.run_policy({}, "rl", {"kind": "rl"}, None)


class ArtifactTests(unittest.TestCase):
    def test_resume_repairs_only_a_torn_final_record(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "worker.jsonl"
            path.write_bytes(b'{"x":1}\n{"x":')
            self.assertEqual(
                list(artifacts.read_records(path, repair_tail=True)), [{"x": 1}]
            )
            self.assertEqual(path.read_bytes(), b'{"x":1}\n')
            self.assertEqual(len(list(path.parent.glob("worker.torn.*"))), 1)
            path.write_bytes(b'{"x":\n{"x":2}\n')
            with self.assertRaises(ValueError):
                list(artifacts.read_records(path, repair_tail=True))

    def test_missing_historical_source_remains_a_changed_source(self):
        missing = "experiments/paper_final/run_05_policy.py"
        self.assertEqual(
            artifacts.source_differences({missing: "old-capture"}), [missing]
        )
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            artifacts.dump(output / "protocol.json", {})
            artifacts.dump(output / "inputs.json", {})
            artifacts.dump(
                output / "prepared.json",
                {
                    "protocol_sha256": artifacts.file_hash(output / "protocol.json"),
                    "inputs_sha256": artifacts.file_hash(output / "inputs.json"),
                    "jobs_sha256": {},
                    "checkpoint_sha256": {},
                },
            )
            artifacts.dump(output / "source_manifest.json", {missing: "old-capture"})
            with self.assertRaisesRegex(
                RuntimeError, "Experiment source changed after preparation"
            ):
                frozen_policy.verify_inputs(output)

    def test_source_snapshots_include_extracted_execution_and_rendering_helpers(self):
        required = {
            "experiments/joint/solve_control/native_case.py",
            "experiments/joint/solve_control/feedback.py",
            "experiments/joint/solve_control/comparison.py",
            "solve/core/episode.py",
            "solve/controllers/common/td_config.py",
            "experiments/paper_final/common/frozen_policy.py",
        }
        self.assertTrue(required.issubset(artifacts.support_source_paths()))
        self.assertIn(
            "experiments/paper_final/common/figure_style.py",
            artifacts.presentation_source_paths(),
        )
        self.assertIn(
            "experiments/paper_final/common/scientific_transforms.py",
            artifacts.presentation_source_paths(),
        )


if __name__ == "__main__":
    unittest.main()
