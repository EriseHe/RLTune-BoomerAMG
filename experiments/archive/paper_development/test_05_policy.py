"""Scientific accounting and restart checks; native paths have a separate preflight."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.archive.paper_development import run_05_policy as study
from experiments.paper_final.common import frozen_policy


def row(case, weight, cost, *, success=True, repeat=0):
    return {"seed": 1, "source": "bandit_default", "case_id": case, "policy": weight,
            "kind": "fixed", "success": success, "recovery": False, "repeat": repeat,
            **{key: cost for key in study.COST_FIELDS}, "setup_sec": 0., "solve_sec": cost,
            "controller_sec": 0., "recovery_setup_sec": 0., "primary_cycles": 1, "fallback_cycles": 0}


class HindsightTests(unittest.TestCase):
    def test_global_fixed_and_per_problem_minima_are_different(self):
        rows = [row(0, "w1", 1), row(1, "w1", 9), row(0, "w2", 6), row(1, "w2", 2)]
        result = study.fixed_hindsight(rows)
        self.assertEqual(result["best_fixed"]["policy"], "w2")
        self.assertEqual(result["best_fixed"]["native_continuation_sec"], 8)
        self.assertEqual(sum(r["native_continuation_sec"] for r in result["per_case"]), 3)

    def test_fast_failure_cannot_win_either_hindsight_comparison(self):
        rows = [row(0, "fast_failed", .001, success=False), row(1, "fast_failed", .001),
                row(0, "valid", 1), row(1, "valid", 1)]
        result = study.fixed_hindsight(rows)
        self.assertEqual(result["best_fixed"]["policy"], "valid")
        self.assertEqual(result["per_case"][0]["policy"], "valid")
        self.assertEqual(result["per_case"][1]["policy"], "fast_failed")

    def test_no_successful_global_fixed_is_reported_honestly(self):
        rows = [row(0, "w1", .1, success=False), row(1, "w1", 2),
                row(0, "w2", 1), row(1, "w2", .1, success=False)]
        result = study.fixed_hindsight(rows)
        self.assertIsNone(result["best_fixed"])
        self.assertEqual(result["uncovered_case_ids"], [])
        self.assertEqual(result["maximum_coverage_fixed"]["failures"], 1)

    def test_repetition_averages_do_not_overweight_repeated_cases(self):
        rows = [row(0, "w", 1), row(0, "w", 3, repeat=1), row(0, "w", 5, repeat=2), row(1, "w", 10)]
        averaged = study.average_repetitions(rows)
        self.assertEqual(study.totals(averaged)["native_continuation_sec"], 13)
        rows[1]["success"] = False
        self.assertIsNone(study.fixed_hindsight(study.average_repetitions(rows))["best_fixed"])

    def test_development_selection_cannot_read_test_results(self):
        rows = [row(0, "w1", 1), row(1, "w1", 9), row(0, "w2", 4), row(1, "w2", 4)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            study.dump(output / "protocol.json", {"selection_rule": "test"})
            with patch.object(study, "phase_rows", return_value=iter(rows)) as loader:
                selected = study.choose_development(output)
            loader.assert_called_once_with(output, "development")
            self.assertEqual(selected["selections"]["1/bandit_default"]["fixed"]["policy"], "w2")


class RestartTests(unittest.TestCase):
    def test_only_torn_tail_is_repairable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "worker.jsonl"
            path.write_bytes(b'{"x":1}\n{"x":')
            self.assertEqual(list(study.read_records(path, repair_tail=True)), [{"x": 1}])
            self.assertEqual(path.read_bytes(), b'{"x":1}\n')
            path.write_bytes(b'{"x":\n{"x":2}\n')
            with self.assertRaises(ValueError):
                list(study.read_records(path, repair_tail=True))

    def test_worker_resume_never_repeats_completed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            policies = {"fixed_1.00": {"kind": "fixed"}, "fixed_2.00": {"kind": "fixed"}}
            study.dump(output / "protocol.json", {"policies": policies, "seeds": {"order": 1}})
            study.dump(output / "jobs_development.json", [{"seed": 1, "source": "bandit_default", "case_id": i} for i in range(2)])
            calls = []
            def run(job, name, spec, bundle):
                calls.append((job["case_id"], name))
                return row(job["case_id"], name, 1)
            with patch.object(study, "verify_inputs"), patch.object(study, "load_bundle"), \
                    patch.object(study, "frozen_snapshot"), patch.object(study, "verify_frozen"), \
                    patch.object(study, "run_policy", side_effect=run):
                study.worker(output, "development", 0, 1)
                original = (output / "raw/development/worker_0.jsonl").read_bytes()
                study.worker(output, "development", 0, 1)
            self.assertEqual(len(calls), 4)
            self.assertEqual((output / "raw/development/worker_0.jsonl").read_bytes(), original)


class RecoveryTests(unittest.TestCase):
    def test_rl_setup_failure_gets_the_same_reference_recovery(self):
        from hypre.bindings import AttemptOutcome, RecoveryOutcome
        primary = {"runtime": .23, "native_runtime": .2, "setup_runtime": .2,
                   "solve_runtime": .03, "native_solve_runtime": 0., "infer_runtime": .03,
                   "failed": True, "attempt_status": "setup_failure", "failure_origin": "solver",
                   "residual_norm": float("nan"), "iterations": 0}
        reported = study.report_online_outcome(RecoveryOutcome(primary=AttemptOutcome.from_mapping(primary)).to_result(), bandit_timing={})
        fallback = {"runtime": .3, "setup_runtime": .1, "solve_runtime": .2, "failed": False,
                    "attempt_status": "success", "residual_norm": 1e-8, "iterations": 5}
        job = {"seed": 1, "source": "bandit_default", "case_id": 0, "input_id": "a", "hierarchy_id": "h", "mkw": {}}
        with patch.object(frozen_policy, "_run_case", return_value=reported), \
                patch.object(frozen_policy, "solve_no_rl_case", return_value=fallback) as recovery:
            result = study.run_policy(job, "rl", {"kind": "rl"}, None)
        recovery.assert_called_once()
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["native_continuation_sec"], .3)
        self.assertAlmostEqual(result["native_total_sec"], .5)
        self.assertAlmostEqual(result["inclusive_total_sec"], .53)

    def test_execution_bug_is_not_logged_as_an_ordinary_numerical_failure(self):
        bad = {"failed": True, "fallback_used": False, "failure_origin": "execution",
               "failure_reason": "exception:TypeError:incorrect API"}
        with patch.object(frozen_policy, "_run_case", return_value=bad):
            with self.assertRaisesRegex(RuntimeError, "execution"):
                study.run_policy({}, "rl", {"kind": "rl"}, None)


if __name__ == "__main__":
    unittest.main()
