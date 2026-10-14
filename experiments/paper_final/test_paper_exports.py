"""Paper exports must retain paired ratios, repetitions and honest missing data."""
import json
from pathlib import Path
import tempfile
import unittest

from experiments.runtime import configure_single_thread
configure_single_thread()
import numpy as np

from experiments.paper_final.online.factorial import METHODS as ONLINE, balanced_orders
from experiments.paper_final.plot_04_factorial import index_ticks
from experiments.paper_final.plot_05_matched_policy import read_study, paired_gains, generate, sha
from experiments.paper_final.matched_policy.configuration import METHODS, LABELS


def fixture(root, seeds=(11, 27)):
    root = Path(root)
    (root / "raw/test").mkdir(parents=True)
    (root / "analysis").mkdir()
    policies = {"fixed_1": {"kind": "fixed", "pattern": [1]},
                "fixed_2": {"kind": "fixed", "pattern": [2]},
                "fixed_3": {"kind": "fixed", "pattern": [3]},
                "periodic": {"kind": "periodic", "pattern": [2.85, 1.1]},
                "rl_frozen": {"kind": "rl"}}
    protocol = {"format": "matched-policy-fresh-v1", "name": "EXPORT VALIDATION FIXTURE",
                "methods": list(METHODS), "labels": LABELS, "training_seeds": list(seeds),
                "test_cases": 2, "repetitions_per_case": 2, "max_cycles": 5,
                "problem": {"kind": "scalar_anisotropic_diffusion", "grid": [12] * 3}, "policies": policies}
    mapping = dict(zip(METHODS, policies))
    means = {"fixed_1": (1., 9.), "fixed_2": (.8, 7.2), "fixed_3": (.6, 3.4),
             "periodic": (.4, 4.6), "rl_frozen": (.1, 4.5)}
    jobs, rows, per_seed = [], [], []
    for seed in seeds:
        for case, case_id in enumerate((101, 203)):
            for rep in range(2):
                job = dict(seed=seed, source="joint", case_id=case_id, repeat=rep,
                           input_id=f"input-{case_id}", hierarchy_id=f"hierarchy-{seed}-{case_id}",
                           method_policies=mapping, policy_order=list(policies))
                jobs.append(job)
                for name, spec in policies.items():
                    cost = means[name][case] * (.5 if rep == 0 else 1.5)
                    n = case + 2
                    actions = np.resize(spec.get("pattern", [1.2, 2.9]), n).tolist()
                    overhead = .01 if spec["kind"] == "rl" else 0.
                    o = dict(setup_runtime=.2, solve_runtime=cost, infer_runtime=overhead,
                             end_to_end_runtime=.2 + cost + overhead, primary_cycles=n,
                             cycle_actions=actions, failed=False)
                    rows.append(dict(**job, phase="test", policy=name, success=True, outcome=o,
                                     native_continuation_sec=cost, inclusive_continuation_sec=cost + overhead,
                                     native_total_sec=cost + .2, inclusive_total_sec=cost + .2 + overhead))
        for method, name in mapping.items():
            total = sum(means[name]);overhead = .02 if name == "rl_frozen" else 0.
            per_seed.append(dict(seed=seed, method=method, defined=True, failed_cases=0,
                                 native_continuation_sec=total, inclusive_continuation_sec=total + overhead,
                                 native_total_sec=total + .4, inclusive_total_sec=total + .4 + overhead))
    raw = root / "raw/test/worker_0.jsonl"
    raw.write_text("".join(json.dumps(r) + "\n" for r in rows))
    values = {"protocol.json": protocol, "jobs_test.json": jobs, "selections.json": {},
              "analysis/summary.json": {"protocol": protocol, "per_seed": per_seed,
                  "audit": {"exact_coverage": True, "evaluation_trials": len(rows)}},
              "raw/test/complete.json": {"trials": len(rows), "raw_sha256": {"raw/test/worker_0.jsonl": sha(raw)}}}
    for name, value in values.items():
        (root / name).write_text(json.dumps(value))
    return rows


class ExportTests(unittest.TestCase):
    def test_every_input_measures_all_four_independent_methods(self):
        orders = balanced_orders(31, 421)
        self.assertTrue(all(set(o) == set(ONLINE) for o in orders))
        self.assertEqual(sum(map(len, orders)), 4 * 31)
        self.assertEqual(orders, balanced_orders(31, 421))

    def test_short_run_ticks_stay_inside_recorded_input_range(self):
        for n in (4, 32, 1000, 5000):
            self.assertTrue(np.all(index_ticks(n) <= n))
            self.assertEqual(index_ticks(n, zero=False)[0], 1)

    def test_ratio_of_sums_after_repeat_averaging_and_single_seed(self):
        with tempfile.TemporaryDirectory() as name:
            fixture(name, seeds=(27,))
            _, seeds, cases, costs, success, weights, _ = read_study(Path(name))
            self.assertEqual(seeds, [27]);self.assertEqual(cases, [101, 203])
            self.assertAlmostEqual(paired_gains(costs, success)[0, 4], 54.)
            self.assertTrue(np.isnan(weights[0, 0, 0, 2:]).all())
            self.assertEqual(weights[0, 0, 0, :2].tolist(), [1., 1.])

    def test_failed_seed_comparison_remains_undefined(self):
        costs = np.ones((2, 5, 2, 4))
        successes = np.ones((2, 5, 2), dtype=bool)
        successes[0, 2, 1] = False
        gains = paired_gains(costs, successes)
        self.assertTrue(np.isnan(gains[0, 2]))
        self.assertEqual(gains[1, 2], 0.)

    def test_mutated_raw_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            fixture(name)
            p = Path(name) / "raw/test/worker_0.jsonl"
            p.write_text(p.read_text() + "\n")
            with self.assertRaises(AssertionError):
                read_study(Path(name))

    def test_summary_costs_must_match_independent_evaluation(self):
        with tempfile.TemporaryDirectory() as name:
            fixture(name)
            p = Path(name) / "analysis/summary.json"
            data = json.loads(p.read_text());data["per_seed"][0]["native_continuation_sec"] = 1.
            p.write_text(json.dumps(data))
            with self.assertRaises(AssertionError):
                read_study(Path(name))

    def test_export_packet_keeps_all_five_roles_and_every_seed(self):
        with tempfile.TemporaryDirectory() as name:
            fixture(name)
            info = generate(name)
            p = Path(info["output"])
            manifest = json.loads((p / "manifest.json").read_text())
            self.assertTrue(manifest["checks"]["passed"])
            self.assertEqual(manifest["methods"], list(METHODS))
            self.assertEqual(manifest["seeds"], [11, 27])
            for seed in manifest["seeds"]:
                self.assertTrue((p / f"matched_policy_weights_seed{seed}.pdf").is_file())
            for filename, expected in manifest["output_sha256"].items():
                self.assertEqual(sha(p / filename), expected)


if __name__ == "__main__":
    unittest.main()
