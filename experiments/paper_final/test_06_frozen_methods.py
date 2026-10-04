"""Contracts for frozen selection, complete costs, and paired evaluation."""
from experiments.paper_final import run_06_frozen_methods as study

from collections import Counter
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np


def native_result(failed=False):
    return dict(runtime=.33, native_runtime=.3, setup_runtime=.1,
                solve_runtime=.23, native_solve_runtime=.2, infer_runtime=.03,
                iterations=2, failed=failed, residual_norm=.1 if failed else 1e-7,
                attempt_status="nonconvergence" if failed else "success",
                cycle_actions=[2.85,1.1])


class Module06Contracts(unittest.TestCase):
    def test_plan_is_paired_complete_and_repeated_without_selecting_fastest(self):
        inputs=[dict(case_id=i) for i in range(study.CASES)]
        plan=study.make_plan(inputs)
        self.assertEqual(len(plan),2100)
        counts=Counter((r["method"],r["case_id"],r["repeat"]) for r in plan)
        self.assertEqual(set(counts.values()),{1})
        self.assertEqual(len(counts),7*100*3)
        for offset in range(0,len(plan),7):
            block=plan[offset:offset+7]
            self.assertEqual(len({r["case_id"] for r in block}),1)
            self.assertEqual({r["method"] for r in block},set(study.PAIRS))
        rows=[dict(method="m",case_id=0,repeat=i,**{k:v for k in study.METRICS})
              for i,v in enumerate((1.,2.,9.))]
        self.assertEqual(study.case_means(rows)["m",0]["inclusive_total_sec"],4.)
        with self.assertRaises(AssertionError): study.case_means(rows[:2])
        with self.assertRaises(AssertionError): study.case_means(rows[:2]+[rows[0]])

    def test_crosses_hold_the_selected_hierarchy_source_fixed(self):
        self.assertEqual(study.PAIRS["periodic_setup_rl"],dict(source="bandit_periodic",policy="rl"))
        self.assertEqual(study.PAIRS["rl_setup_periodic"],dict(source="bandit_lstdq",policy="periodic"))
        for method in study.PRIMARY:
            self.assertEqual(study.PAIRS[method]["source"],method)

    def test_fixed_dispatch_and_periodic_phase_reset(self):
        with patch.object(study,"solve_fixed_w_case",return_value=native_result()) as fixed, \
             patch.object(study,"solve_schedule_case",return_value=native_result()) as periodic:
            for p in ("fixed140","fixed160","periodic","periodic"):
                study.execute_policy({"mkw":{}},{"params":{}},p,None)
            self.assertEqual([c.kwargs["w"] for c in fixed.call_args_list],[1.4,1.6])
            for call in periodic.call_args_list:
                self.assertEqual(call.kwargs["schedule"][:4],[(1,2.85,1,1),(2,1.1,1,1),(3,2.85,1,1),(4,1.1,1,1)])

    def test_recovery_and_selection_are_charged_once(self):
        inp=dict(case_id=0,input_id="input",mkw={})
        choice=dict(params={"coarsen_type":6},arm_index=7,hierarchy_id="hierarchy")
        selectors={"bandit_periodic":Mock(select=Mock(return_value=(choice["params"],7,.01)))}
        with patch.object(study,"solve_schedule_case",return_value=native_result(failed=True)), \
             patch.object(study,"solve_no_rl_case",return_value=native_result()) as fallback:
            row=study.timed_trial(dict(method="bandit_periodic"),inp,
                {"bandit_periodic":[choice]},selectors,None)
        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.kwargs["params"],study.DEFAULT_SETUP_PARAMS)
        self.assertTrue(row["success"])
        self.assertTrue(row["outcome"]["fallback_used"])
        self.assertAlmostEqual(row["inclusive_total_sec"],.67)
        self.assertAlmostEqual(row["native_total_sec"],.6)
        self.assertAlmostEqual(row["recovery_sec"],.33)
        self.assertAlmostEqual(row["inclusive_continuation_sec"],.56)
        self.assertEqual(row["total_cycles"],4)
        for key in ("inclusive_total_sec","native_total_sec","recovery_sec","inclusive_continuation_sec"):
            corrupt=copy.deepcopy(row);corrupt[key]+=.1
            with self.assertRaises(AssertionError): study.audit_row(corrupt)
        corrupt=copy.deepcopy(row);corrupt["outcome"]["controller_update_committed"]=True
        with self.assertRaises(AssertionError): study.audit_row(corrupt)

    def test_frozen_rl_uses_no_learning_or_exploration(self):
        # The established frozen wrapper is responsible for actual RL execution.
        outcome=study.report_online_outcome(study.run_with_default_fallback(
            lambda:native_result(),lambda:native_result()).to_result(),bandit_timing={})
        inp=dict(mkw={},case_id=0,input_id="i")
        choice=dict(params={},hierarchy_id="h")
        bundle=Mock()
        with patch.object(study.base,"_run_case",return_value=outcome) as run:
            study.execute_policy(inp,choice,"rl",bundle)
        self.assertIs(run.call_args.kwargs["bundle"],bundle)
        self.assertFalse(run.call_args.kwargs["learn"])
        self.assertFalse(run.call_args.kwargs["explore"])

    def test_successful_rl_without_recovery_fields_preserves_measured_costs(self):
        outcome=study.report_online_outcome(native_result(),bandit_timing={})
        outcome.update(primary_status="success",fallback_used=False,unrecovered_failure=False)
        with patch.object(study.base,"_run_case",return_value=outcome):
            normalized=study.execute_policy(dict(mkw={},case_id=0,input_id="i"),
                                            dict(params={},hierarchy_id="h"),"rl",Mock())
        self.assertEqual(normalized["primary_cycles"],2)
        self.assertEqual(normalized["fallback_cycles"],0)
        self.assertEqual(normalized["fallback_setup_runtime"],0)
        for key in ("runtime","setup_runtime","solve_runtime","infer_runtime","end_to_end_runtime"):
            self.assertAlmostEqual(normalized[key],outcome[key])

    def test_real_frozen_selector_is_order_independent_and_retains_training_state(self):
        raw=study.base.read(study.PARENT/"experiment_config.json")
        study.training._configure_paired_environment(study.training.runtime(raw))
        source="bandit_periodic"
        originals=study.base.read(study.PARENT/"frozen_artifacts.json")["methods"][source]
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory);folder=output/"checkpoints"/source
            folder.mkdir(parents=True)
            shutil.copy2(originals["decision_state"],folder/"decision.json")
            with np.load(originals["statistics"],allow_pickle=False) as saved:
                arrays={k:saved[k].copy() for k in saved.files}
            arrays["candidate_schedule_cursor"]=np.asarray(0,dtype=np.int64)
            np.savez_compressed(folder/"evaluation.npz",**arrays)
            selector=study.FrozenSelector(output,source,raw)
            self.assertEqual(selector.model.t,5000)
            initial_history=len(selector.model.history)
            inputs=study.input_panel(2,study.SEEDS["preflight"],study.CASES)
            forward={p["case_id"]:selector.select(p)[:2] for p in inputs}
            reverse={p["case_id"]:selector.select(p)[:2] for p in reversed(inputs)}
            self.assertEqual(forward,reverse)
            self.assertEqual(len(selector.model.history),initial_history)
            self.assertTrue(selector.audit("tested")["statistics_unchanged"])


if __name__=="__main__": unittest.main()
