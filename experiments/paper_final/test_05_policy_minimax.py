"""Guard the official Run 05 protocol, mathematical choice, and durable resume."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.paper_final import run_05_policy_minimax as run
from experiments.paper_final.common import policy_analysis as analysis
from experiments.paper_final.verify_period_two_minimax import evaluate


def evaluation_jobs():
    """Small official roster with shared fixed/oracle execution cells."""
    jobs = []
    for repeat in range(3):
        for seed in (1, 2):
            for case in (0, 1):
                methods = {
                    "reference": "fixed_1.00",
                    "fixed": "fixed_1.65",
                    "oracle": "fixed_1.65" if case == 0 else "fixed_1.00",
                    "periodic": run.POLICY,
                    "rl": "rl_frozen_lcb",
                }
                jobs.append(
                    {
                        "seed": seed,
                        "source": "bandit_lstdq",
                        "case_id": case,
                        "repeat": repeat,
                        "method_policies": methods,
                        "policy_order": sorted(set(methods.values())),
                        "mkw": {"rhs_seed": 100 + case},
                        "params": {"coarsen_type": 10},
                        "input_id": f"input-{case}",
                        "hierarchy_id": f"hierarchy-{seed}-{case}",
                    }
                )
    return jobs


class MinimaxTests(unittest.TestCase):
    def test_tracked_accepted_protocol_has_original_constants(self):
        protocol = run.artifacts.read(
            Path(__file__).with_name("reproduction") / "matched_policy/protocol.json"
        )
        self.assertEqual(protocol["run_number"], 5)
        self.assertEqual(protocol["training_seeds"], list(range(1, 7)))
        self.assertEqual(protocol["test_cases"], 100)
        self.assertEqual(protocol["repetitions_per_case"], 3)
        self.assertEqual(protocol["execution_count"], 8460)
        self.assertEqual(protocol["methods"], list(run.METHODS))
        self.assertEqual(
            protocol["policies"][run.POLICY],
            {"kind": "periodic", "pattern": [2.85, 1.10]},
        )

    def test_analytic_extrema_select_joint_grid_minimum(self):
        result = evaluate()
        self.assertTrue(result["passed"])
        self.assertEqual(result["unordered_pairs_checked"], 861)
        self.assertEqual(
            result["grid_minimizers_ordered"], [["2.85", "1.10"], ["1.10", "2.85"]]
        )
        self.assertFalse(result["selection_uses_pde_timings"])
        values = {row["name"]: float(row["eta"]) for row in result["comparisons"]}
        self.assertAlmostEqual(values["continuous_exact"], 0.04, places=14)
        self.assertLess(values["grid_minimizer"], values["componentwise_rounded"])

    def test_resume_keeps_records_and_run_number(self):
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            (output / "progress").mkdir()
            (output / "raw/test").mkdir(parents=True)
            jobs = evaluation_jobs()
            run.artifacts.dump(output / "jobs_test.json", jobs)
            run.artifacts.dump(
                output / "protocol.json",
                {
                    "policies": {
                        name: {} for job in jobs for name in job["policy_order"]
                    }
                },
            )
            calls = []

            def execute(job, policy, spec, bundle):
                calls.append((job["seed"], job["case_id"], job["repeat"], policy))
                return {key: job[key] for key in ("seed", "source", "case_id")} | {
                    "policy": policy,
                    "wall_sec": 0.1,
                }

            with (
                patch.object(run, "verify"),
                patch.object(run.policy, "load_bundle"),
                patch.object(run.policy, "frozen_snapshot"),
                patch.object(run.policy, "verify_frozen"),
                patch.object(run.policy, "run_policy", side_effect=execute),
            ):
                run.worker(output, "test", 0, 1)
                before = (output / "raw/test/worker_0.jsonl").read_bytes()
                run.worker(output, "test", 0, 1)
            self.assertEqual(len(calls), len(analysis.expected_cells(jobs)))
            self.assertEqual(before, (output / "raw/test/worker_0.jsonl").read_bytes())
            self.assertTrue(
                all(
                    row["run_number"] == 5
                    for row in run.artifacts.read_records(
                        output / "raw/test/worker_0.jsonl"
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
