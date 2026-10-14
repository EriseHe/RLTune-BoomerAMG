"""Check full-budget unconditional sequential dispatch without running any native solver."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from experiments.paper_final import run_06_beta_sequence as sequence
from experiments.paper_final import run_06_beta_sweep as sweep
from experiments.paper_final.common.artifacts import dump, file_hash, read, ready
from experiments.paper_final import test_06_beta_sweep as sweep_tests


class SequenceDispatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.template = Path(cls.temporary.name) / "prepared"
        forbidden = AssertionError("Preparation executed a solver or started a worker")
        with patch("solve.core.episode.run_td_episode", side_effect=forbidden), \
             patch("experiments.paper_final.online.native_evaluation.solve_no_rl_case", side_effect=forbidden), \
             patch("hypre.bindings.create_env", side_effect=forbidden), \
             patch.object(sweep, "build_named_setup_branches", side_effect=forbidden), \
             patch.object(sweep.subprocess, "Popen", side_effect=forbidden), \
             patch.object(sweep.subprocess, "check_output", return_value="fixture-revision"), \
             patch.object(sweep, "source_hashes", return_value={}), \
             patch.object(sweep, "native_hashes", return_value={}):
            sequence.prepare(cls.template)
        helper = sweep_tests.SweepRecoveryWiringTests("test_input_1000_is_setup_only_and_1001_activates_rl")
        row = helper.simulate(helper.runtime(), 0, [])[0]
        cls.prefix_records = [dict(row, problem_index=i+1) for i in range(1000)]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.run_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.run_dir.cleanup)
        self.output = Path(self.run_dir.name) / "sequence"
        shutil.copytree(self.template, self.output)
        self.source_guard = patch.object(sweep, "source_hashes", return_value={})
        self.native_guard = patch.object(sweep, "native_hashes", return_value={})
        self.source_guard.start()
        self.native_guard.start()
        self.addCleanup(self.source_guard.stop)
        self.addCleanup(self.native_guard.stop)

    def report(self, directory, values, times):
        protocol = read(directory / "protocol.json")
        totals = []
        for index, name in enumerate([*sweep.branch_betas(values), *sweep.BASELINES]):
            totals.append(dict(branch=name, beta=protocol["betas"].get(name), window="all_5000",
                cases=5000, total_sec=times[index], native_sec=times[index]/2, failures=index,
                fallbacks=index+1))
        # A successful-only or partial window with opposite ranking must never
        # influence the decision; only the complete all-input totals count.
        extra = [dict(row, window="joint_4000", cases=4000, total_sec=100-row["total_sec"])
                 for row in totals]
        dump(directory / "comparison.json", dict(complete=True, audits_passed=True,
             protocol=protocol, rows=totals+extra))
        dump(directory / "status.json", dict(status="complete", audits_passed=True))

    def prefix(self, directory):
        trajectory = directory / "prefix/trajectory.jsonl"
        trajectory.parent.mkdir()
        trajectory.write_text("".join(json.dumps(ready(row))+"\n" for row in self.prefix_records))
        dump(directory / "prefix/result.json", dict(complete=True, cases=1000,
             controller=dict(steps=0, episodes=0), trajectory_sha256=file_hash(trajectory), elapsed_sec=12.0))

    def test_preparation_pairs_two_diffusion_groups_and_three_fresh_prefixes(self):
        plan = sequence.validate_prepared(self.output)
        self.assertFalse(plan["prefix_reuse"])
        self.assertEqual(len({s["inputs_sha256"] for s in plan["stages"].values()}), 2)
        for name, values in sequence.STAGES.items():
            directory = self.output / name
            protocol = read(directory / "protocol.json")
            self.assertNotIn("reused_prefix", protocol)
            self.assertFalse((directory / "prefix").exists())
            self.assertEqual(set(protocol["preparation_checks"]), {*sweep.branch_betas(values), *sweep.BASELINES})
            for branch, beta in sweep.branch_betas(values).items():
                runtime = sweep.decode(read(directory / "configs" / f"{branch}.json"), directory / branch)
                self.assertEqual(runtime.bundle.controller.spec.uncertainty_beta, beta)
                self.assertEqual((runtime.bundle.controller.steps, runtime.bundle.controller.episodes), (0, 0))
                self.assertEqual(len(runtime.stream), 5000)

    def test_all_three_groups_run_in_order_regardless_of_time_ranking(self):
        called = []
        def simulated_run(directory, *, prefix_completed):
            if called:
                self.assertEqual(read(self.output / called[-1] / "status.json")["status"], "complete")
            called.append(directory.name)
            self.assertFalse((directory / "prefix").exists())
            self.assertNotIn("reused_prefix", read(directory / "protocol.json"))
            self.prefix(directory)
            prefix_completed(directory)
            self.report(directory, sequence.STAGES[directory.name], (9, 12, 10, 20, 15))
        with patch.object(sweep, "launch", side_effect=simulated_run):
            sequence.launch(self.output)
        self.assertEqual(called, list(sequence.STAGES))
        status = read(self.output / "status.json")
        self.assertEqual(status["completed_stages"], list(sequence.STAGES))
        self.assertEqual(status["status"], "complete")
        self.assertTrue((self.output / "sequence_comparison.csv").is_file())
        self.assertTrue((self.output / "REPORT.txt").is_file())
        for name in sequence.STAGES:
            self.assertEqual(read(self.output / name / "checkpoint_1000.json")["controller_steps"], 0)

    def test_incomplete_or_failed_first_stage_never_launches_second(self):
        first = next(iter(sequence.STAGES))
        def incomplete(directory, *, prefix_completed):
            self.report(directory, sequence.STAGES[first], (20, 20, 10, 30, 25))
            report = read(directory / "comparison.json")
            report["rows"][0]["cases"] = 4999
            dump(directory / "comparison.json", report)
        with patch.object(sweep, "launch", side_effect=incomplete) as launcher:
            with self.assertRaisesRegex(ValueError, "5000 inputs"):
                sequence.launch(self.output)
        self.assertEqual(launcher.call_count, 1)
        self.assertEqual(read(self.output / "status.json")["status"], "failed")
        self.assertFalse((self.output / "decision.json").exists())

    def test_missing_control_or_changed_configuration_prevents_dispatch(self):
        first = next(iter(sequence.STAGES))
        directory = self.output / first
        self.report(directory, sequence.STAGES[first], (20, 20, 10, 30, 25))
        report = read(directory / "comparison.json")
        report["rows"] = [r for r in report["rows"] if r["branch"] != "default"]
        dump(directory / "comparison.json", report)
        with self.assertRaisesRegex(ValueError, "both controls"):
            sequence.completed_totals(directory, sequence.STAGES[first])
        dump(directory / "status.json", dict(status="prepared"))
        config = directory / "configs/beta_5.json"
        raw = read(config)
        raw["solve"]["lstdq"]["beta"] = 9
        dump(config, raw)
        with patch.object(sweep, "launch") as launcher:
            with self.assertRaisesRegex(ValueError, "configuration changed"):
                sequence.launch(self.output)
            launcher.assert_not_called()


if __name__ == "__main__":
    unittest.main()
