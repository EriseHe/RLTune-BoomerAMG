"""Guard the historical Run 05 protocol, mathematical choice and resume."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.paper_final import run_05_policy_minimax as run
from experiments.paper_final import run_05_policy_anchored as parent
from experiments.paper_final.test_05_policy_refresh import base_jobs, choices
from experiments.paper_final.verify_period_two_minimax import evaluate


def synthetic_jobs():
    return parent.replace_schedule(run.repeat.plan_jobs(base_jobs()[:2], choices()))


class MinimaxTests(unittest.TestCase):
    def test_exact_delta_from_all_actual_parent_jobs(self):
        old=run.first.read(run.PARENT/"jobs_test.json")
        before=copy.deepcopy(old);new=run.replace_schedule(old)
        self.assertEqual(old,before)
        self.assertEqual(len(new),1800)
        self.assertEqual(len(run.repeat.expected_cells(new)),8460)
        self.assertEqual(len(run.repeat.expected_cells(old))-len(run.repeat.expected_cells(new)),1800)
        for a,b in zip(old,new):
            expected=copy.deepcopy(a)
            expected["method_policies"].pop("periodic13")
            expected["method_policies"]["periodic"]=run.POLICY
            expected["policy_order"]=[run.POLICY if p==run.OLD_POLICY else p
                for p in a["policy_order"] if p!=run.REMOVED_POLICY]
            self.assertEqual(b,expected)
            self.assertEqual(set(b["method_policies"]),set(run.METHODS))
            self.assertNotIn(run.OLD_POLICY,b["policy_order"])
            self.assertNotIn(run.REMOVED_POLICY,b["policy_order"])

    def test_unexpected_parent_schedule_rejected(self):
        jobs=synthetic_jobs()
        jobs[0]["method_policies"]["periodic"]="periodic_3.00_1.00"
        with self.assertRaises(ValueError):run.replace_schedule(jobs)

    def test_analytic_extrema_select_joint_grid_minimum(self):
        result=evaluate()
        self.assertTrue(result["passed"])
        self.assertEqual(result["unordered_pairs_checked"],861)
        self.assertEqual(result["grid_minimizers_ordered"],[["2.85","1.10"],["1.10","2.85"]])
        self.assertFalse(result["selection_uses_pde_timings"])
        vals={r["name"]:float(r["eta"]) for r in result["comparisons"]}
        self.assertAlmostEqual(vals["continuous_exact"],.04,places=14)
        self.assertLess(vals["grid_minimizer"],vals["componentwise_rounded"])

    def test_resume_keeps_records_and_run_number(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);(output/"progress").mkdir();(output/"raw/test").mkdir(parents=True)
            jobs=run.replace_schedule(synthetic_jobs())
            run.first.dump(output/"jobs_test.json",jobs)
            run.first.dump(output/"protocol.json",{"policies":{n:{} for j in jobs for n in j["policy_order"]}})
            calls=[]
            def execute(job,policy,spec,bundle):
                calls.append((job["case_id"],job["repeat"],policy))
                return {k:job[k] for k in ("seed","source","case_id")}|{"policy":policy,"wall_sec":.1}
            with patch.object(run,"verify"),patch.object(run.first,"load_bundle"),patch.object(run.first,"frozen_snapshot"),patch.object(run.first,"verify_frozen"),patch.object(run.first,"run_policy",side_effect=execute):
                run.worker(output,"test",0,1)
                before=(output/"raw/test/worker_0.jsonl").read_bytes()
                run.worker(output,"test",0,1)
                self.assertEqual(len(calls),len(run.repeat.expected_cells(jobs)))
                self.assertEqual(before,(output/"raw/test/worker_0.jsonl").read_bytes())
                self.assertTrue(all(r["run_number"]==5 for r in run.first.read_records(output/"raw/test/worker_0.jsonl")))


if __name__=="__main__":unittest.main()
