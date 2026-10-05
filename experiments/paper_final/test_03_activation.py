"""Nested activation forks: real setup learners with bounded mock solve costs."""

import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import experiments.joint.solve_control.joint_4k_execution as execution
from experiments.joint.solve_control.composable_joint_4k import build_named_setup_branches, resolve_composable_study
from experiments.joint.solve_control.joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from experiments.joint.solve_control.joint_online_common import report_online_outcome
from setup.space import DEFAULT_SETUP_PARAMS

CONFIG = Path(__file__).with_name("03_activation") / "advection_80_s1.json"


class ActivationTests(unittest.TestCase):
    def test_nested_forks_share_exact_prefix_and_independent_suffixes(self):
        self._check_nested_forks(with_baseline=True)

    def test_three_rl_branches_share_latest_starts_prefix(self):
        self._check_nested_forks(with_baseline=False)

    def _check_nested_forks(self, *, with_baseline):
        raw = json.loads(CONFIG.read_text())
        if not with_baseline:
            template = raw["methods"][1]
            raw["methods"] = [dict(template, id=f"start_{tau}", solve_activation_case=tau)
                              for tau in (250, 500, 750)]
        runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
        boundaries = dict(zip([m.name for m in runtime.methods if m.solve_kind != "default"], (2, 4, 6, 8, 10)))
        source = "setup_only" if with_baseline else "start_750"
        targets = {name: tau for name, tau in boundaries.items() if name != source}
        specs = tuple(replace(m, solve_activation_case=boundaries.get(m.name, 0)) for m in runtime.methods)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            runtime = replace(runtime, output_dir=output, online_cases=12, methods=specs)
            study = resolve_composable_study(runtime)
            branches = build_named_setup_branches(runtime, study=study, warmup_instances=[], stream_hash="unit").branches
            trajectories = output / "trajectories"
            trajectories.mkdir()
            bundles = {m.name: Mock() for m in specs if m.solve_kind != "default"}
            for bundle in bundles.values():
                bundle.summary.return_value = {"steps": 0}
            plan = execution.OnlineComparisonPlan(
                output_dir=output, trajectories_dir=trajectories, checkpoints_dir=output / "checkpoints",
                methods=tuple(study.methods), family_by_method=study.family_by_method,
                bandit_methods=study.bandit_methods, composable_specs=study.specs_by_name,
                online_instances=[({"index": i}, np.array([1., .2, .5, .8, .1, .2, .3])) for i in range(12)],
                branches=branches, controller_bundles=bundles, ppo_runner=None, protocol={},
                warmup=execution.WarmupArtifacts({}, "", {}, 0), solve=execution.SolveExecutionConfig(1e-6, 50),
                warmup_cases=0, online_cases=12, method_order_seed=19, progress_every=12,
                aot_enabled=True, aot_max_selections_per_case=3, default_setup_method="unused",
                include_solve_screen_report=False, shared_online_prefix=True,
            )
            calls = []

            def native(method, *, case_index, **kwargs):
                active = method in boundaries and case_index >= boundaries[method]

                def solve(params):
                    calls.append((method, case_index, active))
                    return dict(runtime=.05, native_runtime=.05, setup_runtime=.03,
                                solve_runtime=.02, native_solve_runtime=.02, infer_runtime=.001 if active else 0.,
                                failed=False, iterations=2, primary_status="success", first_primary_status="success",
                                recovered=False, fallback_used=False, bandit_update_committed=True,
                                controller_update_committed=active)
                return solve

            def bandit(*, policy, problem_context, solver_fn, prev_update_est, **kwargs):
                model = policy.model
                self.assertEqual(prev_update_est, 0. if model.t == 0 else .004)
                params = model.predict(problem_context)
                outcome = solver_fn(params)
                model.update(.05 + prev_update_est + (.01 if outcome["controller_update_committed"] else 0.),
                             failure_label=0.)
                return params, outcome, {"overhead_sec": .002}, 0, .004

            hooks = execution.OnlineComparisonHooks(method_solver=native, run_default_setup_method=Mock(),
                                                    report_online_outcome=report_online_outcome)
            with patch.object(execution, "run_bandit_step_test_final", side_effect=bandit):
                rows, steps = execution._execute_online_instances(plan, hooks)
            self.assertEqual(len(calls), 12 + sum(12 - tau for tau in targets.values()))
            self.assertEqual(steps, {m.name: 12 for m in specs})
            for target, tau in targets.items():
                self.assertEqual([(i, active) for m, i, active in calls if m == target],
                                 [(i, True) for i in range(tau, 12)])
                for i in range(tau):
                    self.assertEqual(rows[target][i], rows[source][i])
                audit = json.loads((output / f"shared_prefix_{target}.json").read_text())
                self.assertTrue(audit["valid"])
                self.assertEqual(audit["completed_instances"], tau)
                self.assertEqual(audit["candidate_schedule_cursor"], 3 * tau)
                a, b = branches[source].policy.model, branches[target].policy.model
                self.assertFalse(np.shares_memory(a.A_inv, b.A_inv))
                self.assertIsNot(a.rng, b.rng)
                self.assertEqual(b._candidate_schedule.cursor, 36)
            self.assertEqual([(i, active) for m, i, active in calls if m == source],
                             [(i, i >= boundaries.get(source, 12)) for i in range(12)])

    def test_actual_solver_switches_on_boundary_plus_one(self):
        runtime = runtime_config_from_spec(parse_joint_experiment_config(json.loads(CONFIG.read_text())))
        specs = {s.name: s for s in runtime.methods}
        bundle = Mock()
        bundle.run_case.return_value = {"controlled": True}
        with patch.object(execution, "solve_no_rl_case", return_value={"controlled": False}), \
                patch.object(execution, "_as_feedback", side_effect=lambda value, **kwargs: value):
            for name, spec in specs.items():
                indices = (0, 4999) if name == "setup_only" else (spec.solve_activation_case - 1, spec.solve_activation_case)
                for index in indices:
                    solver = execution.make_method_solver(
                        name, solve=execution.SolveExecutionConfig(1e-6, 50), mkw={}, case_progress=0., case_index=index,
                        controller_bundles={name: bundle} if name != "setup_only" else {},
                        ppo_runner=None, composable_specs=specs)
                    self.assertEqual(solver(dict(DEFAULT_SETUP_PARAMS))["controlled"],
                                     name != "setup_only" and index >= spec.solve_activation_case)

    def test_mismatched_branch_seeds_or_solve_settings_are_rejected(self):
        raw = json.loads(CONFIG.read_text())
        for with_baseline in (True, False):
            for field, value in (("seed_offset", 1), ("solve_context", "canonical"), ("solve_tolerance", 1e-8)):
                invalid = copy.deepcopy(raw)
                if not with_baseline:
                    invalid["methods"] = invalid["methods"][1:4]
                invalid["methods"][2][field] = value
                with self.subTest(with_baseline=with_baseline, field=field), self.assertRaises(ValueError):
                    resolve_composable_study(runtime_config_from_spec(parse_joint_experiment_config(invalid)))


if __name__ == "__main__":
    unittest.main()
