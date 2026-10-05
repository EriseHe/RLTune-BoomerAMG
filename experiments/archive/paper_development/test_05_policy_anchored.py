"""Guard the Run 04 protocol delta and durable execution behavior."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.archive.paper_development import run_05_policy_anchored as run
from experiments.archive.paper_development.test_05_policy_refresh import base_jobs, choices
from experiments.archive.paper_development.verify_anchored_schedule import eta


class AnchoredTests(unittest.TestCase):
    def test_only_schedule_changes_in_all_planned_executions(self):
        old=run.repeat.plan_jobs(base_jobs(),choices(),order_seed=92705301)
        before=copy.deepcopy(old);new=run.replace_schedule(old)
        self.assertEqual(old,before);self.assertEqual(len(new),1800)
        self.assertEqual(len(run.repeat.expected_cells(new)),len(run.repeat.expected_cells(old)))
        for a,b in zip(old,new):
            restored=copy.deepcopy(b)
            restored["method_policies"]["periodic"]=run.OLD_POLICY
            restored["policy_order"]=[run.OLD_POLICY if p==run.POLICY else p for p in b["policy_order"]]
            self.assertEqual(a,restored)
            self.assertNotIn(run.OLD_POLICY,b["policy_order"])

    def test_unexpected_parent_schedule_rejected(self):
        old=run.repeat.plan_jobs(base_jobs()[:1],choices())
        old[0]["method_policies"]["periodic"]="periodic_3.00_1.00"
        with self.assertRaises(ValueError):run.replace_schedule(old)

    def test_grid_minimum_is_evaluated_not_just_rounded(self):
        self.assertEqual(min([i/20 for i in range(20,61)],key=eta),2.6)
        self.assertAlmostEqual(eta(2.6),0.04445682245697864,places=14)
        self.assertLess(eta(2.6),eta(2.65));self.assertLess(eta(2.6),eta(2.5))

    def test_resume_keeps_records_and_correct_run_number(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);(output/"progress").mkdir();(output/"raw/test").mkdir(parents=True)
            jobs=run.replace_schedule(run.repeat.plan_jobs(base_jobs()[:2],choices()))
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
                self.assertTrue(all(r["run_number"]==4 for r in run.first.read_records(output/"raw/test/worker_0.jsonl")))


if __name__=="__main__":unittest.main()
