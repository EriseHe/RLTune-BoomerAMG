"""PAPER_FINAL contract and reporting tests; no native experiments."""
from __future__ import annotations

import _project_paths  # noqa: F401

import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

import joint_4k_execution as execution
import run_paper_final as suite
from analyze_paper_final import audit_run, aggregate_runs, WINDOWS
from composable_joint_4k import build_composable_solve_runtime, build_named_setup_branches, resolve_composable_study
from joint_4k_execution import make_method_solver, SolveExecutionConfig
from joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from joint_experiment_plotting import _compact_method_labels
from joint_method_spec import ComposableMethodSpec
from joint_online_common import method_stream_summary
from problems.amg import build_context_diffusion_advection_from_matrix_kwargs
from problems.registry import context_for_setup_method
from setup.space import DEFAULT_SETUP_PARAMS


class PaperFinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.configs = suite.load_suite()

    def runtime(self, family="diffusion_advection"):
        raw = next(raw for entry, _path, raw in self.configs if entry["family"] == family)
        return runtime_config_from_spec(parse_joint_experiment_config(raw))

    def test_nonzero_advection_has_identical_shared_context_and_no_mean(self):
        runtime = self.runtime()
        bundle = build_composable_solve_runtime(runtime, specs=runtime.methods).controller_bundles["bandit_lstdq"]
        full_spec = replace(runtime.methods[-1], solve_context="canonical")
        full = build_composable_solve_runtime(runtime, specs=(full_spec,)).controller_bundles["bandit_lstdq"]
        mkw = dict(nx=40, ny=40, nz=40, k=100., c=10., a0=1., a1=-99., a2=9., a3=999.)
        context = build_context_diffusion_advection_from_matrix_kwargs(
            mkw, grid_norm_div=40., c_norm_div=1000., a_norm_div=1000.)
        expected = np.r_[1., [2/3, 1/3, 0.],
                         np.sign([-99., 9., 999.]) * np.log1p([99., 9., 999.]) / np.log1p(1000.)]
        setup = context_for_setup_method(
            problem_kind=runtime.problem, setup_kind="linucb", setup_context="canonical_no_c_mean",
            matrix_kwargs=mkw, stream_context=context)
        np.testing.assert_allclose(setup, expected, atol=1e-15)
        inputs = dict(mkw=mkw, problem_context=context, initial_residual=1., residual=.02,
                      previous_residual=.1, cycle=4, last_weight=2.7, last_cycle_time=.004,
                      setup_params={**DEFAULT_SETUP_PARAMS, "coarsen_type": 3, "interp_type": 17})
        state = bundle.encoder.encode(**inputs)
        np.testing.assert_array_equal(state, np.delete(full.encoder.encode(**inputs), 10))
        np.testing.assert_array_equal(setup, np.r_[state[0], state[7:13]])
        self.assertEqual((bundle.encoder.feature_dim, bundle.controller.joint_dim), (34, 306))
        # Changing only the removed mean must leave every encoded coordinate fixed.
        changed = context.copy(); changed[4] = .987
        np.testing.assert_array_equal(bundle.encoder.encode(**{**inputs, "problem_context": changed}), state)
        self.assertNotIn("c_mean", bundle.protocol_metadata()["state_encoder"]["problem_context_fields"])
        for spec in runtime.methods:
            self.assertEqual(ComposableMethodSpec.from_runner_token(spec.to_runner_token()), spec)

    def test_new_bandits_share_rng_and_context_but_not_mutable_learning_state(self):
        runtime = self.runtime()
        study = resolve_composable_study(runtime)
        with tempfile.TemporaryDirectory() as tmp:
            shortened = replace(runtime, output_dir=Path(tmp), online_cases=2)
            artifacts = build_named_setup_branches(shortened, study=study, warmup_instances=[], stream_hash="test-only")
            a, b = [artifacts.branches[m].policy.model for m in ("bandit_default", "bandit_lstdq")]
            self.assertEqual((a.d_x, b.d_x), (7, 7))
            self.assertFalse(np.shares_memory(a.A_inv, b.A_inv))
            self.assertEqual(a.rng.bit_generator.state, b.rng.bit_generator.state)

    def test_formal_method_labels_and_actual_activation_boundary(self):
        for family in suite.CONTEXTS:
            runtime = self.runtime(family)
            specs = {s.name: s for s in runtime.methods}
            labels = _compact_method_labels({"methods": list(specs), "method_specs": [asdict(s) for s in runtime.methods]})
            self.assertEqual(labels, suite.METHOD_LABELS)
            bundle = Mock()
            bundle.run_case.return_value = {"controlled": True}
            with patch("joint_4k_execution.solve_no_rl_case", return_value={"controlled": False}), \
                    patch("joint_4k_execution._as_feedback", side_effect=lambda native, **kw: native):
                for index, expected in ((0, False), (999, False), (1000, True), (4999, True)):
                    solver = make_method_solver(
                        "bandit_lstdq", solve=SolveExecutionConfig(1e-6, 50), mkw={},
                        case_progress=index/4999, case_index=index,
                        controller_bundles={"bandit_lstdq": bundle}, ppo_runner=None,
                        composable_specs=specs)
                    self.assertEqual(solver(dict(DEFAULT_SETUP_PARAMS))["controlled"], expected)

    def test_default_cli_never_runs_and_low_disk_blocks_run(self):
        with patch.object(suite, "validate_suite", return_value={"valid": True}), \
                patch.object(suite, "run_suite") as run, patch("sys.stdout", new_callable=io.StringIO):
            suite.main([])
            run.assert_not_called()
        with patch.object(suite.shutil, "disk_usage", return_value=SimpleNamespace(free=2**29)):
            with self.assertRaisesRegex(RuntimeError, "Insufficient disk"):
                suite.require_disk_space(Path(tempfile.gettempdir()), 18)

    def test_prespecified_configs_detect_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "suite.json").write_text(json.dumps(self.manifest))
            for _entry, path, _raw in self.configs:
                (directory / path.name).write_bytes(path.read_bytes())
            changed = directory / self.configs[0][1].name
            changed.write_text(changed.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "config changed"):
                suite.load_suite(directory / "suite.json")

    def test_september18_suite_has_requested_seed_and_family_budgets(self):
        path = suite.SUITE.parent / "20260918/suite.json"
        manifest, configs = suite.load_suite(path)
        self.assertEqual(len(configs), 6)
        self.assertEqual(manifest["allow_unrecovered_families"], ["diffusion_advection"])
        for entry, _path, raw in configs:
            self.assertEqual(raw["seeds"], dict(
                base=56700120, bandit=56760120, controller=56766120, method_order=56772120
            ))
            self.assertEqual(raw["solve"]["max_cycles"], 50 if entry["family"] == "diffusion" else 100)
        self.assertTrue(suite.validate_suite(path)["valid"])

    def test_advection50_rerun_preserves_inputs_and_only_changes_cycle_budget(self):
        path = suite.SUITE.parent / "20260918_advection50/suite.json"
        manifest, configs = suite.load_suite(path)
        self.assertEqual(manifest["families"], ["diffusion_advection"])
        self.assertTrue(manifest["abort_on_clock_mismatch"])
        self.assertEqual([entry["grid"] for entry, _, _ in configs], [40, 60, 80])
        for entry, _, raw in configs:
            old_path = suite.SUITE.parent / "20260918" / entry["config"]
            expected = json.loads(old_path.read_text())
            expected["solve"]["max_cycles"] = 50
            for key in ("description", "output_dir"):
                expected[key] = raw[key]
            self.assertEqual(raw, expected)
            self.assertIn("20260918_advection50", raw["output_dir"])
        self.assertTrue(suite.validate_suite(path)["valid"])

    def test_clock_mismatch_stops_before_another_method_and_preserves_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            trajectories = output / "trajectories"
            trajectories.mkdir()
            plan = SimpleNamespace(
                output_dir=output, trajectories_dir=trajectories,
                bandit_methods=(), methods=("a", "b"), method_order_seed=19,
                online_instances=[({}, np.ones(1))], composable_specs={}, shared_online_prefix=False,
                protocol={"timing": {"abort_on_clock_mismatch": True}},
            )
            row = {"outcome": {"end_to_end_runtime": 300., "bandit_update_committed": True}}
            with patch.object(execution, "_run_one_method", return_value=row) as run, \
                    patch.object(execution.time, "perf_counter", side_effect=[0., 1.]):
                with self.assertRaisesRegex(RuntimeError, "Timing mismatch at problem 1"):
                    execution._execute_online_instances(plan, Mock())
            self.assertEqual(run.call_count, 1)
            anomaly = json.loads((output / "timing_anomaly.json").read_text())
            self.assertTrue(anomaly["requires_fresh_run"])
            self.assertEqual(anomaly["row"]["outcome"]["method_wall_runtime"], 1.)
            self.assertEqual(anomaly["row"]["outcome"]["end_to_end_runtime"], 300.)
            self.assertTrue(anomaly["row"]["outcome"]["bandit_update_committed"])

    def test_suite_dispatch_and_resume_are_sequential_and_source_checked(self):
        real_run = suite.subprocess.run
        commands = []
        def dispatch(command, **kwargs):
            if command[0] != suite.sys.executable:
                return real_run(command, **kwargs)
            commands.append(command)
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")
            self.assertNotIn("SETUP_CYCLE_TYPE", kwargs["env"])
            return suite.subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(suite, "source_state", return_value={"sha256": "test", "files": {}}) as source, \
                patch.object(suite, "require_disk_space"), \
                patch.object(suite.subprocess, "run", side_effect=dispatch), \
                patch("analyze_paper_final.audit_run", return_value={"unrecovered_failures": 0}), \
                patch("analyze_paper_final.write_reports"), patch("sys.stdout", new_callable=io.StringIO):
            output = Path(tmp)
            suite.run_suite(suite.SUITE, output)
            self.assertEqual(len(commands), 18)
            for command, (_entry, config_path, _raw) in zip(commands, self.configs):
                self.assertIn(str(config_path), command)
            self.assertEqual(len(list(output.glob("*.complete.json"))), 18)
            suite.run_suite(suite.SUITE, output)
            self.assertEqual(len(commands), 18)
            source.return_value = {"sha256": "changed", "files": {}}
            with self.assertRaisesRegex(RuntimeError, "changed since suite launch"):
                suite.run_suite(suite.SUITE, output)

    def test_family_and_seed_filter_dispatches_only_three_grids(self):
        real_run = suite.subprocess.run
        def dispatch_selected(command, **kwargs):
            if command[0] == suite.sys.executable:
                return suite.subprocess.CompletedProcess(command, 0)
            return real_run(command, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(suite, "source_state", return_value={"sha256": "test", "files": {}}), \
                patch.object(suite, "require_disk_space"), \
                patch.object(suite.subprocess, "run", side_effect=dispatch_selected) as dispatch, \
                patch("sys.stdout", new_callable=io.StringIO):
            audit = Mock(return_value={"unrecovered_failures": 0})
            report = Mock()
            suite.run_suite(suite.SUITE, Path(tmp), family="diffusion", seed=1,
                            run_audit=audit, report_writer=report)
            selected = [Path(call.args[0][4]).stem for call in dispatch.call_args_list
                        if call.args[0][0] == suite.sys.executable]
            self.assertEqual(selected, [f"diffusion_{n}_s1" for n in (40, 60, 80)])
            self.assertEqual(len(list((Path(tmp) / "logs").glob("*.log"))), 3)
            report.assert_called_with(suite.SUITE, Path(tmp), allow_partial=True)

    def test_prescribed_advection_failures_are_retained_and_allow_resume(self):
        configs = [c for c in self.configs if c[0]["family"] == "diffusion_advection"][:2]
        manifest = {"allow_unrecovered_families": ["diffusion_advection"],
                    "abort_on_clock_mismatch": True}
        real_run = suite.subprocess.run
        commands = []

        def dispatch(command, **kwargs):
            if command[0] != suite.sys.executable:
                return real_run(command, **kwargs)
            self.assertEqual(kwargs["env"]["RLTUNE_ABORT_ON_CLOCK_MISMATCH"], "1")
            commands.append(command)
            return suite.subprocess.CompletedProcess(command, 0)

        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(suite, "source_state", return_value={"sha256": "test", "files": {}}), \
                patch.object(suite, "require_disk_space"), \
                patch.object(suite.subprocess, "run", side_effect=dispatch), \
                patch("sys.stdout", new_callable=io.StringIO):
            output = Path(tmp)
            kwargs = dict(suite_loader=lambda _: (manifest, configs),
                          run_audit=Mock(return_value={"unrecovered_failures": 2}), report_writer=Mock())
            suite.run_suite(suite.SUITE, output, **kwargs)
            suite.run_suite(suite.SUITE, output, **kwargs)
            self.assertEqual(len(commands), 2)
            markers = list(output.glob("*.complete.json"))
            self.assertEqual(len(markers), 2)
            self.assertTrue(all(json.loads(p.read_text())["unrecovered_failures"] == 2 for p in markers))

    def write_fixture(self, path, *, timing_version=1):
        raw = self.configs[0][2]
        (path / "trajectories").mkdir()
        (path / "experiment_config.json").write_text(json.dumps(raw))
        stream = {"sha256": raw["stream"]["expected_sha256"]}
        (path / "stream_manifest.json").write_text(json.dumps(stream))
        (path / "progress.json").write_text(json.dumps({"completed_online_instances": 5000}))
        result = {"protocol": {"stream": stream}, "windows": {w: {"methods": {}} for w in WINDOWS}}
        if timing_version >= 2:
            result["protocol"]["timing"] = {"schema_version": timing_version}
        for method in suite.METHOD_LABELS:
            rows = []
            for i in range(5000):
                bandit = method != "default_setup_default_solve"
                controlled = method == "bandit_lstdq" and i >= 1000
                setup = .002 if bandit else .003
                solve = .003 if controlled else .006 if bandit else .007
                infer = .0002 if controlled else 0.
                overhead = .0001 if bandit else 0.
                outcome = dict(runtime=setup+solve, setup_runtime=setup, solve_runtime=solve,
                               infer_runtime=infer, bandit_overhead_runtime=overhead,
                               end_to_end_runtime=setup+solve+infer+overhead, iterations=2,
                               first_primary_status="nonconvergence" if i == 12 else "success",
                               primary_status="nonconvergence" if i == 12 else "success",
                               primary_residual_norm=.01 if i == 12 else 1e-8, primary_cycles=2,
                               fallback_used=i == 12, fallback_status="success" if i == 12 else "not_run",
                               fallback_residual_norm=1e-8 if i == 12 else float("nan"),
                               fallback_cycles=2 if i == 12 else 0,
                               completed_residual_norm=1e-8, completed_cycles=2,
                               completed_status="success", unrecovered_failure=False,
                               fallback_setup_runtime=.0005 if i == 12 else 0.,
                               fallback_solve_runtime=.0015 if i == 12 else 0., recovered=i == 12,
                               bandit_update_committed=bandit, controller_update_committed=controlled,
                               cycle_actions=[1., 2.5] if controlled else [])
                if timing_version >= 2:
                    outcome["method_wall_runtime"] = outcome["end_to_end_runtime"] + .001
                    if controlled:
                        outcome.update(feature_runtime=.00005, decision_runtime=.00005,
                                       update_runtime=.00008, lifecycle_runtime=.00002)
                rows.append({"online_index": i, "params": DEFAULT_SETUP_PARAMS,
                             "mkw": {"rhs_seed": i}, "outcome": outcome})
            (path / "trajectories" / f"{method}.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            for window, (start, stop) in WINDOWS.items():
                result["windows"][window]["methods"][method] = method_stream_summary(rows[start:stop])
        (path / "result.json").write_text(json.dumps(result))
        return raw

    def test_accounting_reports_include_prefix_and_recovery_without_double_counting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw = self.write_fixture(path)
            audited = audit_run(path, raw, require_completed_residuals=True)
            joint = audited["windows"]["all_5000"]["bandit_lstdq"]
            self.assertAlmostEqual(joint["totals_sec"]["end_to_end_runtime"], 29.3)
            self.assertAlmostEqual(joint["time_reduction_vs_default_pct"], 41.4)
            self.assertAlmostEqual(joint["fallback_native_sec"], .002)
            self.assertEqual(joint["primary_failure_count"], 1)
            self.assertAlmostEqual(audited["windows"]["last_1000"]["bandit_lstdq"]["time_reduction_vs_default_pct"], 47.)
            trajectory = path / "trajectories/bandit_lstdq.jsonl"
            rows = trajectory.read_text().splitlines()
            row = json.loads(rows[0]); row["outcome"]["end_to_end_runtime"] = float("nan")
            rows[0] = json.dumps(row)
            trajectory.write_text("\n".join(rows)+"\n")
            with self.assertRaises((ValueError, AssertionError)):
                audit_run(path, raw)

    def test_formal_audit_rejects_missing_or_inconsistent_final_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw = self.write_fixture(path)
            trajectory = path / "trajectories/default_setup_default_solve.jsonl"
            lines = trajectory.read_text().splitlines()
            original = json.loads(lines[0])
            for change in ("missing", "wrong_attempt", "unconverged"):
                row = copy.deepcopy(original)
                outcome = row["outcome"]
                if change == "missing":
                    del outcome["completed_residual_norm"]
                elif change == "wrong_attempt":
                    outcome["completed_residual_norm"] = 2e-8
                else:
                    outcome["completed_residual_norm"] = outcome["primary_residual_norm"] = 1e-6
                lines[0] = json.dumps(row)
                trajectory.write_text("\n".join(lines) + "\n")
                with self.subTest(change=change), self.assertRaises(ValueError):
                    audit_run(path, raw, require_completed_residuals=True)

    def test_new_timing_schema_audits_wall_and_controller_phase_totals(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw = self.write_fixture(path, timing_version=2)
            audited = audit_run(path, raw, require_completed_residuals=True)
            joint = audited["windows"]["all_5000"]["bandit_lstdq"]["totals_sec"]
            self.assertAlmostEqual(joint["controller_lifecycle_runtime"], .08)
            self.assertAlmostEqual(joint["method_wall_runtime"], joint["end_to_end_runtime"] + 5.)
            trajectory = path / "trajectories/bandit_lstdq.jsonl"
            lines = trajectory.read_text().splitlines()
            original = json.loads(lines[1000])
            for field, value in (("method_wall_runtime", 0.), ("lifecycle_runtime", .1)):
                row = copy.deepcopy(original)
                row["outcome"][field] = value
                lines[1000] = json.dumps(row)
                trajectory.write_text("\n".join(lines) + "\n")
                with self.subTest(field=field), self.assertRaises(ValueError):
                    audit_run(path, raw, require_completed_residuals=True)

    def test_failed_final_attempt_can_retain_a_null_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            raw = self.write_fixture(path)
            method = "default_setup_default_solve"
            trajectory = path / "trajectories" / f"{method}.jsonl"
            rows = [json.loads(line) for line in trajectory.read_text().splitlines()]
            rows[0]["outcome"].update(
                failed=True, primary_status="solve_failure", first_primary_status="solve_failure",
                completed_status="solve_failure", primary_residual_norm=None,
                completed_residual_norm=None, unrecovered_failure=True,
            )
            trajectory.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = json.loads((path / "result.json").read_text())
            for window, (start, stop) in WINDOWS.items():
                result["windows"][window]["methods"][method] = method_stream_summary(rows[start:stop])
            (path / "result.json").write_text(json.dumps(result))
            self.assertEqual(audit_run(path, raw, require_completed_residuals=True)["unrecovered_failures"], 1)

    def test_replicate_summary_uses_equal_seed_weights_and_sample_sd(self):
        runs = []
        for seed, reduction in enumerate((10., 20., 60.), 1):
            methods = {m: {"time_reduction_vs_default_pct": reduction,
                           "totals_sec": {"end_to_end_runtime": 100.-reduction}}
                       for m in suite.METHOD_LABELS}
            runs.append({"family": "diffusion", "grid": 60, "seed": seed,
                         "windows": {w: copy.deepcopy(methods) for w in WINDOWS}})
        row = aggregate_runs(runs)[0]
        self.assertEqual(row["seeds"], [1, 2, 3])
        self.assertAlmostEqual(row["mean_time_reduction_pct"], 30.)
        self.assertAlmostEqual(row["sample_sd_time_reduction_pct"], np.std([10., 20., 60.], ddof=1))


if __name__ == "__main__":
    unittest.main()
