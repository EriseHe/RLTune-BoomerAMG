"""Campaign ordering, no-solve preparation, frozen inputs and residual semantics."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.paper_final.common.artifacts import ROOT, read, dump
from experiments.paper_final import run_04_campaign as campaign
from experiments.paper_final.plot_04_factorial import cycle_profiles, plot_profile, plt


def fixture(root):
    root = Path(root)
    template = read(ROOT / "experiments/paper_final/04_online/20261009_zero_prefix_diffusion60/diffusion60_seed1.json")
    for experiment in ("small", "large"):
        directory = root / experiment
        directory.mkdir()
        for seed in (11, 27, 42):
            raw = copy.deepcopy(template)
            raw["name"] = f"{experiment}_{seed}"
            dump(directory / f"{seed}.json", raw)
        dump(directory / "suite.json", dict(name=experiment,
            seeds=[dict(label=str(s), config=f"{s}.json") for s in (11, 27, 42)]))
    path = root / "campaign.json"
    dump(path, dict(name="Campaign test", max_workers=2, seed_batches=[["11", "27"], ["42"]],
        experiments=[dict(label=e, suite=f"{e}/suite.json") for e in ("small", "large")]))
    return path


class CampaignTests(unittest.TestCase):
    def test_coverage_and_worker_limits_are_validated(self):
        with tempfile.TemporaryDirectory() as temp:
            path = fixture(temp)
            raw = read(path)
            raw["seed_batches"] = [["11", "27", "42"]]
            dump(path, raw)
            with self.assertRaisesRegex(ValueError, "max_workers"):
                campaign.load_plan(path)
            raw["seed_batches"] = [["11", "27"], ["27"]]
            dump(path, raw)
            with self.assertRaisesRegex(ValueError, "disjoint"):
                campaign.load_plan(path)
            raw["seed_batches"] = [["11", "27"]]
            dump(path, raw)
            with self.assertRaisesRegex(ValueError, "exactly the campaign seeds"):
                campaign.load_plan(path)

    def test_prepare_freezes_every_input_and_performs_no_solves(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(campaign.runner, "source_hashes", return_value={}), \
             patch.object(campaign.runner.sweep, "native_hashes", return_value={}), \
             patch.object(campaign.runner, "launch", side_effect=AssertionError("No execution during preparation")):
            path = fixture(temp)
            output = Path(temp) / "prepared"
            prepared = campaign.prepare(path, output, cases=2)
            self.assertEqual([(j["batch"], j["experiment"], j["seeds"]) for j in prepared["jobs"]],
                [(1, "small", ["11", "27"]), (1, "large", ["11", "27"]),
                 (2, "small", ["42"]), (2, "large", ["42"])])
            self.assertEqual(prepared["planned_method_evaluations"], 48)
            self.assertEqual(read(output / "status.json")["native_solves_executed"], 0)
            self.assertTrue(campaign.verify(output)["unstarted"])
            target = output / prepared["jobs"][0]["output"] / "seed_11/inputs.jsonl"
            rows = target.read_text().splitlines()
            changed = json.loads(rows[0]);changed["problem_index"] = 99
            target.write_text(json.dumps(changed) + "\n" + rows[1] + "\n")
            with self.assertRaisesRegex(ValueError, "input file changed"):
                campaign.verify(output)

    def test_launch_waits_for_each_job_before_next_batch_and_stops_on_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp).resolve()
            jobs = [dict(batch=b, experiment=e, seeds=[str(b)], output=f"batch{b}/{e}")
                    for b in (1, 2) for e in ("small", "large")]
            dump(output / "campaign.json", dict(jobs=jobs))
            calls = []
            def run_job(path):
                calls.append(str(path.relative_to(output)))
                self.assertEqual(read(output / "status.json")["completed_jobs"], len(calls)-1)
            with patch.object(campaign, "verify"), patch.object(campaign, "prevent_sleep"), \
                 patch.object(campaign, "stop_sleep_prevention"), patch.object(campaign, "analyze") as analyze, \
                 patch.object(campaign.runner, "launch", side_effect=run_job):
                campaign.launch(output)
                self.assertEqual(calls, [j["output"] for j in jobs])
                analyze.assert_called_once_with(output)
                self.assertEqual(read(output / "status.json")["status"], "complete")
            calls.clear()
            def fail_second(path):
                run_job(path)
                if len(calls) == 2:
                    raise RuntimeError("job failed")
            with patch.object(campaign, "verify"), patch.object(campaign, "prevent_sleep"), \
                 patch.object(campaign, "stop_sleep_prevention"), patch.object(campaign, "analyze") as analyze, \
                 patch.object(campaign.runner, "launch", side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError, "job failed"):
                    campaign.launch(output)
                self.assertEqual(calls, [j["output"] for j in jobs[:2]])
                analyze.assert_not_called()
                self.assertEqual(read(output / "status.json")["status"], "failed")

    def test_residual_mean_retains_failed_attempts_and_full_cycle_cap(self):
        weights = np.full((2, 2, 50), np.nan)
        weights[:, 0, :10] = 1.
        weights[:, 1, :] = 2.
        residuals = np.full((2, 2, 51), 1e-7)
        residuals[:, :, 0] = 1.
        residuals[:, 1, 1:] = .1
        count, _, mean = cycle_profiles(weights, residuals)
        self.assertTrue((count[:, -1] == 1).all())
        self.assertTrue((mean[:, -1] > 1e-6).all())
        fig = plot_profile(weights, residuals, "fixture", "Residual test", 1e-6, 0)
        self.assertGreaterEqual(fig.axes[-1].get_xlim()[1], 50)
        self.assertIn("First-attempt", fig.axes[-1].get_title(loc="left"))
        plt.close(fig)
        residuals[0, 1, -1] = np.nan
        self.assertTrue(np.isnan(cycle_profiles(weights, residuals)[2][0, -1]))


if __name__ == "__main__":
    unittest.main()
