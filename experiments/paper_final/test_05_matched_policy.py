"""Fresh scans must select successful mean-cost comparators on matched cases."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.runtime import configure_single_thread

configure_single_thread()

from experiments.paper_final.common import frozen_policy
from experiments.paper_final.matched_policy.configuration import load_config, weight_grid
from experiments.paper_final.matched_policy.planning import evaluation_jobs, scan_jobs
from experiments.paper_final.matched_policy.selection import select_fixed


def scan_fixture():
    jobs, rows = [], []
    # Weight 1 is the successful global choice. Weight 2 wins only case 0;
    # its very fast case 1 failed. Weight 3 wins case 1. Repeated means, not
    # a best individual timing, determine both selections.
    observations = {
        (0, "fixed_1"): [(2, True), (2, True)],
        (0, "fixed_2"): [(1, True), (1, True)],
        (0, "fixed_3"): [(0.1, True), (7.9, True)],
        (1, "fixed_1"): [(2, True), (2, True)],
        (1, "fixed_2"): [(0.01, True), (0.01, False)],
        (1, "fixed_3"): [(1, True), (1, True)],
    }
    for case in (0, 1):
        for repeat in (0, 1):
            job = dict(seed=27, source="joint", case_id=case, repeat=repeat,
                       input_id=f"input-{case}", hierarchy_id=f"hierarchy-{case}",
                       policy_order=["fixed_1", "fixed_2", "fixed_3"])
            jobs.append(job)
            for name in job["policy_order"]:
                cost, success = observations[case, name][repeat]
                rows.append({**job, "policy": name, "native_continuation_sec": cost, "success": success})
    return jobs, rows


class FreshSelectionTests(unittest.TestCase):
    def test_success_constraints_and_mean_repetitions(self):
        jobs, rows = scan_fixture()
        choice = select_fixed(rows, jobs)["27"]
        self.assertEqual(choice["global"], "fixed_1")
        self.assertEqual(choice["per_instance"], {"0": "fixed_2", "1": "fixed_3"})

    def test_no_successful_candidate_stays_undefined(self):
        jobs, rows = scan_fixture()
        for row in rows:
            if row["case_id"] == 1:
                row["success"] = False
        choice = select_fixed(rows, jobs)["27"]
        self.assertIsNone(choice["global"])
        self.assertIsNone(choice["per_instance"]["1"])
        self.assertEqual(choice["uncovered_case_ids"], [1])

    def test_incomplete_duplicate_foreign_or_mismatched_evidence_rejected(self):
        jobs, rows = scan_fixture()
        variants = [rows[:-1], rows + [rows[0]], [{**rows[0], "seed": 99}] + rows[1:]]
        for field in ("input_id", "hierarchy_id"):
            variants.append([{**rows[0], field: "changed"}] + rows[1:])
        for variant in variants:
            with self.subTest(variant=variant[0]), self.assertRaises(ValueError):
                select_fixed(variant, jobs)

    def test_plan_fresh_scans_and_independent_retiming_with_arbitrary_labels(self):
        base = [dict(seed=seed, source="joint", case_id=case) for seed in (11, 27) for case in (0, 1)]
        cfg = dict(weights=[1, 2, 3], seeds={"order": 93}, scan_repetitions=2,
                   scan_extra_repetitions=3, evaluation_repetitions=4)
        scans = scan_jobs(base, cfg, [1])
        self.assertEqual(sum(len(j["policy_order"]) for j in scans), 42)
        selections = {str(s): {"global": "fixed_1", "per_instance": {"0": "fixed_1", "1": None}}
                      for s in (11, 27)}
        tests = evaluation_jobs(base, selections, cfg)
        self.assertEqual(len(tests), 16)
        self.assertEqual({j["repeat"] for j in tests}, {0, 1, 2, 3})
        for job in tests:
            self.assertEqual(set(job["method_policies"]), {"reference", "fixed", "oracle", "periodic", "rl"})
            self.assertEqual(set(job["policy_order"]), {"fixed_1", "periodic", "rl_frozen"})
        self.assertEqual(tests, evaluation_jobs(base, selections, cfg))


class FreshConfigTests(unittest.TestCase):
    def test_exact_default_weight_grid(self):
        values = weight_grid({"start": 1, "stop": 3, "step": .05})
        self.assertEqual(len(values), 41)
        self.assertEqual(values[37], 2.85)
        self.assertEqual(values[-1], 3)

    def test_relative_sources_labels_and_inherited_settings(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            run = root / "source"
            (run / "checkpoints").mkdir(parents=True)
            (run / "final_bandit_states").mkdir()
            training = dict(methods=[dict(id="joint", setup="linucb", solve="recursive_lstdq")],
                            problem={"kind": "scalar_anisotropic_diffusion", "grid": [12, 12, 12]},
                            solve={"tolerance": 1e-7, "max_cycles": 71, "smoother_profile": "hypre_default"})
            (run / "experiment_config.json").write_text(json.dumps(training))
            for path in ("checkpoints/joint_final.npz", "checkpoints/joint_final.npz.encoder.json",
                         "final_bandit_states/joint.npz"):
                (run / path).write_text("fixture")
            cfg = dict(training_runs=[dict(seed=27, run_dir="source", method="joint")],
                       seeds=dict(inputs=1, candidates=2, order=3, repeats=4), cases=7,
                       evaluation_repetitions=2, workers=4)
            path = root / "config.json"
            path.write_text(json.dumps(cfg))
            parsed = load_config(path)
            self.assertEqual(parsed["training_runs"][0]["seed"], 27)
            self.assertEqual(parsed["solver"], dict(tolerance=1e-7, max_cycles=71, profile="hypre_default"))
            self.assertEqual(parsed["cases"], 7)
            self.assertEqual(parsed["evaluation_repetitions"], 2)
            self.assertEqual(parsed["workers"], 4)
            self.assertIsNone(parsed["rl_beta"])
            self.assertEqual(Path(parsed["training_runs"][0]["config"]), (run / "experiment_config.json").resolve())
            for update in ({"seed": 4}, {"training_runs": cfg["training_runs"] * 2},
                           {"seeds": {"inputs": 1}}, {"weights": [1, 1]}, {"cases": 0}):
                path.write_text(json.dumps({**cfg, **update}))
                with self.subTest(update=update), self.assertRaises(ValueError):
                    load_config(path)


class SolverSettingsTests(unittest.TestCase):
    def test_custom_tolerance_is_used_by_audit(self):
        result = dict(setup_runtime=.1, solve_runtime=.2, infer_runtime=0,
                      end_to_end_runtime=.3, failed=False, residual_norm=5e-5,
                      completed_residual_norm=5e-5)
        frozen_policy.audit_outcome(result, tolerance=1e-4)
        with self.assertRaises(AssertionError):
            frozen_policy.audit_outcome(result)

    def test_schedule_and_fallback_receive_custom_solver_settings(self):
        def execute(primary, fallback):
            primary()
            fallback()
            raise RuntimeError("observed both calls")
        with patch.object(frozen_policy, "run_with_default_fallback", side_effect=execute), \
             patch.object(frozen_policy, "solve_schedule_case") as primary, \
             patch.object(frozen_policy, "solve_no_rl_case") as fallback:
            with self.assertRaisesRegex(RuntimeError, "observed"):
                frozen_policy.run_schedule({}, {}, [2.85, 1.1], tolerance=1e-4, max_cycles=7)
            self.assertEqual(primary.call_args.kwargs["solve_tol"], 1e-4)
            self.assertEqual(primary.call_args.kwargs["solve_max_cycles"], 7)
            self.assertEqual(primary.call_args.kwargs["schedule"][-1], (7, 2.85, 1, 1))
            self.assertEqual(fallback.call_args.kwargs["solver_tol"], 1e-4)
            self.assertEqual(fallback.call_args.kwargs["solver_max_iter"], 7)


if __name__ == "__main__":
    unittest.main()
