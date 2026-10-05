"""Guard the follow-up's input matching, scan scope, and changed baseline choices."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.archive.paper_development import run_05_policy_refresh as run
from experiments.archive.paper_development import run_05_policy_repeat as repeat


def base_jobs():
    return [{"seed":s,"source":run.SOURCE,"case_id":c,"mkw":{"rhs_seed":c},"params":{"coarsen_type":s},
             "hierarchy_id":f"{s}/{c}"} for s in range(1,7) for c in range(100)]


def choices():
    return {f"{s}/{run.SOURCE}/{m}":v for s in range(1,7) for m,v in {
        "reference":"fixed_1.00","fixed":"fixed_1.60","oracle":{str(c):"fixed_1.55" for c in range(100)},
        "periodic":"periodic_2.50_1.00","rl":"rl_frozen_lcb"}.items()}


class RefreshTests(unittest.TestCase):
    def test_scan_only_seed4_and_prescribed_repetitions(self):
        base=base_jobs();before=copy.deepcopy(base);ids=list(range(0,100,10))
        jobs=run.scan_jobs(base,ids)
        self.assertEqual(base,before)
        self.assertEqual(len(repeat.expected_cells(jobs)),4920)
        self.assertEqual({j["seed"] for j in jobs},{4})
        for j in jobs:
            self.assertEqual(len(j["policy_order"]),41)
            if j["repeat"]:self.assertIn(j["case_id"],ids)
        self.assertEqual(jobs,run.scan_jobs(base,ids))

    def test_refresh_choices_do_not_change_untargeted_inputs_or_checkpoints(self):
        base=base_jobs();selected=choices();prior=copy.deepcopy(selected)
        selected[f"4/{run.SOURCE}/fixed"]="fixed_1.70"
        selected[f"4/{run.SOURCE}/oracle"]={str(c):"fixed_1.65" for c in range(100)}
        jobs=repeat.plan_jobs(base,selected,order_seed=92705301)
        self.assertEqual(len(jobs),1800)
        for j in jobs:
            original=next(b for b in base if b["seed"]==j["seed"] and b["case_id"]==j["case_id"])
            self.assertEqual(j["mkw"],original["mkw"]);self.assertEqual(j["params"],original["params"])
            self.assertEqual(set(j["method_policies"]),set(repeat.METHODS))
            self.assertNotIn("periodic_3.00_1.00",j["policy_order"])
            if j["seed"]!=4:self.assertEqual(j["method_policies"],repeat.method_map(prior,j["seed"],j["case_id"]))

    def test_worker_resume_does_not_execute_completed_trial_again(self):
        with tempfile.TemporaryDirectory() as name:
            output=Path(name);(output/"progress").mkdir();(output/"raw/test").mkdir(parents=True)
            jobs=repeat.plan_jobs(base_jobs()[:2],choices(),repeats=3)
            run.first.dump(output/"jobs_test.json",jobs)
            run.first.dump(output/"protocol.json",{"policies":{n:{} for j in jobs for n in j["policy_order"]}})
            calls=[]
            def execute(job,policy,spec,bundle):
                calls.append((job["case_id"],job["repeat"],policy))
                return {k:job[k] for k in ("seed","source","case_id")}|{"policy":policy,"wall_sec":.1}
            with patch.object(run,"verify"),patch.object(run.first,"load_bundle"),patch.object(run.first,"frozen_snapshot"),\
                 patch.object(run.first,"verify_frozen"),patch.object(run.first,"run_policy",side_effect=execute):
                run.worker(output,"test",0,1)
                before=(output/"raw/test/worker_0.jsonl").read_bytes()
                run.worker(output,"test",0,1)
                self.assertEqual(len(calls),len(repeat.expected_cells(jobs)))
                self.assertEqual(before,(output/"raw/test/worker_0.jsonl").read_bytes())


if __name__=="__main__":unittest.main()
