"""Integration contracts for shared-prefix checkpoint preparation (no PDEs)."""
from experiments.archive.paper_development import run_05_checkpoint_training as study

from collections import Counter
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.archive.paper_development import calibrate_05_default_weight as calibration


NATIVE = {"runtime": .3, "native_runtime": .3, "setup_runtime": .1,
          "solve_runtime": .2, "native_solve_runtime": .2, "infer_runtime": 0.}


class Module05Contracts(unittest.TestCase):
    def setUp(self):
        # A fixture coefficient, never used to select the real run's constant.
        self.raw = study.make_config(Path("/tmp/module05_test_config"), 1.4, check=True)
        self.args = study.runtime(self.raw)
        self.specs = {m.name:m for m in self.args.methods}

    def solver_kwargs(self, index):
        return dict(solve=study.execution.SolveExecutionConfig(1e-6, 50), mkw={"nx":60},
                    case_progress=0., case_index=index, problem_context=np.asarray([1.,.1,.2,.3,.2,0.,0.,0.]),
                    controller_bundles={"bandit_lstdq":Mock(run_case=Mock(return_value=NATIVE))},
                    ppo_runner=None, composable_specs=self.specs)

    def test_boundary_keeps_all_policies_on_w1_then_activates_continuations(self):
        # Exercise the real inclusive 1000 boundary without solving 1000 PDEs.
        self.specs = {m:study.replace(s, solve_activation_case=0 if m==study.METHODS[0] else 1000)
                      for m,s in self.specs.items()}
        with patch.object(study.execution, "solve_no_rl_case", return_value=NATIVE) as w1, \
             patch.object(study.execution, "solve_fixed_w_case", return_value=NATIVE) as fixed, \
             patch.object(study, "solve_schedule_case", return_value=NATIVE) as periodic:
            for m in study.METHODS:
                kw = self.solver_kwargs(999)
                study.method_solver(m, **kw)({})
                kw["controller_bundles"]["bandit_lstdq"].run_case.assert_not_called()
            self.assertEqual(w1.call_count,len(study.METHODS))
            fixed.assert_not_called(); periodic.assert_not_called()
            for m in study.METHODS:
                kw = self.solver_kwargs(1000)
                study.method_solver(m, **kw)({})
                if m == "bandit_lstdq":
                    kw["controller_bundles"][m].run_case.assert_called_once()
            self.assertEqual(w1.call_count,len(study.METHODS)+1)
            self.assertEqual([c.kwargs["w"] for c in fixed.call_args_list],[1.4,1.6])
            periodic.assert_called_once()

    def test_periodic_phase_resets_and_dispatch_cost_is_charged(self):
        measured = dict(NATIVE, runtime=.31, solve_runtime=.21, infer_runtime=.01)
        with patch.object(study, "solve_schedule_case", return_value=measured) as native:
            fn = study.method_solver("bandit_periodic", **self.solver_kwargs(2))
            for _ in range(2):
                result = fn({})
                self.assertAlmostEqual(result["runtime"],.31)
            for call in native.call_args_list:
                self.assertEqual(call.kwargs["schedule"][:4],
                                 ((1,2.85,1,1),(2,1.10,1,1),(3,2.85,1,1),(4,1.10,1,1)))

    def test_periodic_primary_does_not_replace_default_recovery(self):
        failed = dict(NATIVE, failed=True, attempt_status="nonconvergence")
        plan = SimpleNamespace(composable_specs=self.specs,
                               solve=study.execution.SolveExecutionConfig(1e-6,50))
        with patch.object(study, "solve_schedule_case", return_value=failed), \
             patch.object(study.execution, "solve_no_rl_case", return_value=NATIVE) as fallback:
            out = study.method_solver("bandit_periodic", **self.solver_kwargs(2))({"coarsen_type":6})
            self.assertTrue(out["failed"])
            fallback.assert_not_called()
            recovery = study.execution._default_fallback_solver(plan=plan,method="bandit_periodic",mkw={"nx":60})
            recovery({"coarsen_type":6})
            self.assertEqual(fallback.call_args.kwargs["params"], study.base.DEFAULT_SETUP_PARAMS)

    def test_actual_shared_loop_executes_prefix_once_and_forks_all_four(self):
        calls, forks = [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = SimpleNamespace(methods=study.METHODS, bandit_methods=study.METHODS,
                composable_specs=self.specs, shared_online_prefix=True, method_order_seed=71,
                online_instances=[({"id":i},np.ones(4)) for i in range(8)],
                trajectories_dir=root, output_dir=root, protocol={}, controller_bundles={})
            def one(**kw):
                m,i = kw["method"],kw["online_index"]
                calls.append((m,i))
                kw["previous_update"][m] = (i+1)/1000
                kw["bandit_online_steps"][m] += 1
                return {"online_index":i,"outcome":{"runtime":1.},"params":{"choice":m}}
            def fork(_plan, **kw):
                self.assertEqual(kw["completed_instances"],2)
                self.assertEqual(set(kw["bandit_online_steps"].values()),{2})
                self.assertEqual(set(kw["previous_update"].values()),{.002})
                forks.append(kw["target"])
            with patch.object(study.execution,"_run_one_method",side_effect=one), \
                 patch.object(study.execution,"_fork_shared_online_prefix",side_effect=fork), \
                 patch.object(study.execution,"_checkpoint_controllers"), \
                 patch.object(study.execution,"_write_progress"):
                rows, steps = study.execution._execute_online_instances(plan, Mock())
            self.assertEqual(len(calls),32)
            self.assertEqual(Counter(i for _,i in calls),Counter({0:1,1:1,**{i:5 for i in range(2,8)}}))
            self.assertEqual(forks,list(study.METHODS[1:]))
            self.assertEqual(set(steps.values()),{8})
            for m in study.METHODS:
                self.assertEqual(rows[m][:2],rows[study.METHODS[0]][:2])

    def test_real_setup_state_fork_is_complete_and_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = study.replace(self.args, output_dir=root)
            resolved = study.resolve_composable_study(args)
            setup = study.build_named_setup_branches(args, study=resolved, warmup_instances=(),stream_hash="test")
            study.audit_initial_state(setup.branches)
            source = setup.branches["bandit_default"].policy.model
            source.b[0] = 12
            source.t = 2
            source.rng.random()
            source.set_candidate_schedule_cursor(6)
            arm = source.actions.default_arm_index
            source.history = [study.SharedLinUCBv4Step(t=2,arm_index=arm,loss=.7,pred_mean=.3,pred_uncert=.2)]
            source.candidate_stats_history = [{"random":128}]
            source._local_neighbor_cache[(arm,False,1)] = np.asarray([arm])
            source._cand.observe(arm,.7)
            (root/"checkpoints").mkdir()
            plan = SimpleNamespace(branches=setup.branches,output_dir=root,checkpoints_dir=root/"checkpoints",controller_bundles={})
            updates = {m:.002 for m in study.METHODS}
            steps = {m:2 for m in study.METHODS}
            for target in study.METHODS[1:]:
                study.execution._fork_shared_online_prefix(plan,source="bandit_default",target=target,
                    completed_instances=2,previous_update=updates,bandit_online_steps=steps,artifact_suffix="_"+target)
                model = setup.branches[target].policy.model
                self.assertEqual(study.base.digest(study.decision_state(model)),study.base.digest(study.decision_state(source)))
                self.assertEqual(model.rng.bit_generator.state,source.rng.bit_generator.state)
                model.b[0] += 1
                model.history.clear()
                self.assertEqual(source.b[0],12)
                self.assertEqual(len(source.history),1)
            saved = study.decision_state(source)
            source.history.clear()
            study.restore_decision_state(source,saved)
            self.assertEqual(study.base.digest(study.decision_state(source)),study.base.digest(saved))

    def test_mismatched_setup_or_activation_is_rejected(self):
        for field, value in (("seed_offset",1),("solve_activation_case",3)):
            raw = copy.deepcopy(self.raw)
            raw["methods"][1][field] = value
            with self.assertRaises(ValueError):
                study.runtime(raw)

    def test_controller_factory_only_builds_the_rl_learner(self):
        built = study.build_controllers(self.args, self.args.methods)
        self.assertEqual(set(built.controller_bundles), {"bandit_lstdq"})
        self.assertEqual(built.controller_bundles["bandit_lstdq"].summary()["sample_count"], 0)
        self.assertIsNone(built.ppo_runner)

    def test_historical_fixed_weight_cannot_silently_change(self):
        raw = copy.deepcopy(self.raw)
        next(m for m in raw["methods"] if m["id"] == "bandit_fixed_prior")["fixed_weight"] = 1.65
        with self.assertRaises(ValueError):
            study.runtime(raw)

    def test_full_training_budget_includes_both_fixed_branches(self):
        raw = study.make_config(Path("/tmp/module05_test_full_config"), 1.4)
        self.assertEqual(raw["checkpoint_training"]["physical_method_problem_executions"], 21000)
        self.assertEqual(raw["checkpoint_training"]["logical_method_problem_exposures"], 25000)
        self.assertEqual(raw["stream"]["cases"], 5000)

    def test_calibration_never_selects_a_fast_unrecovered_failure(self):
        rows = []
        for w in calibration.WEIGHTS:
            for i in range(calibration.CASES):
                cost = .01 if w==3 else (.5 if w==1.6 else 1.)
                rows.append(dict(weight=w,case_id=i,success=not(w==3 and i==0),
                    inclusive_continuation_sec=cost,native_continuation_sec=cost,wall_sec=cost,
                    outcome={"fallback_used":False,"end_to_end_runtime":cost}))
        self.assertEqual(calibration.summarize(rows)["selected_weight"],1.6)


if __name__ == "__main__":
    unittest.main()
