"""Protect matching, fixed selections, repetitions and durable Run 02 resume."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.paper_final import run_05_policy_repeat as run
from experiments.paper_final.analyze_05_policy_repeat import collect_cells


def choices():
    return {f"1/{run.SOURCE}/{m}":v for m,v in {
        "reference":"fixed_1.00","fixed":"fixed_1.60","oracle":{"0":"fixed_1.60","1":"fixed_1.55"},
        "periodic":"periodic_2.50_1.00","rl":"rl_frozen_lcb"}.items()}


def base_jobs():
    return [{"seed":1,"source":run.SOURCE,"case_id":c,"mkw":{"n":60,"rhs_seed":c},
             "params":{"coarsen_type":2},"input_id":str(c),"hierarchy_id":"h"+str(c)} for c in (0,1)]


def raw(cost,repeat,success=True):
    return {"seed":1,"source":run.SOURCE,"case_id":0,"policy":"fixed_1.60","kind":"fixed","repeat":repeat,
            "success":success,"outcome":{"setup_runtime":.1,"solve_runtime":cost,"primary_cycles":10},
            **{k:cost for k in run.first.COST_FIELDS}}


class RepeatTests(unittest.TestCase):
    def test_jobs_preserve_exact_inputs_and_lock_all_six_policy_roles(self):
        base=base_jobs();before=copy.deepcopy(base)
        jobs=run.plan_jobs(base,choices())
        self.assertEqual(base,before)
        self.assertEqual(len(jobs),6)
        for j in jobs:
            self.assertEqual(j["params"],before[j["case_id"]]["params"])
            self.assertEqual(j["mkw"],before[j["case_id"]]["mkw"])
            self.assertEqual(set(j["method_policies"]),set(run.METHODS))
            self.assertEqual(set(j["policy_order"]),set(j["method_policies"].values()))
        self.assertEqual(len(run.expected_cells(jobs)),33)
        self.assertFalse(any("periodic_3.00_1.00" in j["policy_order"] for j in jobs))
        self.assertEqual(run.MAIN_METHODS,("fixed","oracle","periodic13","rl"))

    def test_repeat_schedule_is_reproducible_without_reselecting_weights(self):
        a=run.plan_jobs(base_jobs(),choices());b=run.plan_jobs(base_jobs(),choices())
        self.assertEqual(a,b)
        for case in (0,1):
            jj=[j for j in a if j["case_id"]==case]
            self.assertEqual({j["repeat"] for j in jj},{0,1,2})
            self.assertEqual(len({j["method_policies"]["oracle"] for j in jj}),1)

    def test_average_all_repeats_and_keep_failed_outcome(self):
        result=collect_cells([raw(1,0),raw(3,1),raw(8,2,False)],[0,1,2])
        cell=result[1,0,"fixed_1.60"]
        self.assertEqual(cell["native_continuation_sec"],4)
        self.assertFalse(cell["success"])
        with self.assertRaises(ValueError):collect_cells([raw(1,0),raw(3,1)],[0,1,2])
        with self.assertRaises(ValueError):collect_cells([raw(1,0),raw(2,0),raw(3,2)],[0,1,2])

    def test_resume_skips_every_durable_trial(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);(out/"raw/test").mkdir(parents=True);(out/"progress").mkdir()
            jobs=run.plan_jobs(base_jobs(),choices())
            run.first.dump(out/"jobs_test.json",jobs)
            names={n for j in jobs for n in j["policy_order"]}
            run.first.dump(out/"protocol.json",{"policies":{n:{} for n in names}})
            calls=[]
            def fake(job,name,spec,bundle):
                calls.append((job["case_id"],job["repeat"],name))
                return {k:job[k] for k in ("seed","source","case_id")}|{"policy":name,"wall_sec":.1}
            with patch.object(run,"verify"),patch.object(run.first,"load_bundle"),patch.object(run.first,"frozen_snapshot"),\
                 patch.object(run.first,"verify_frozen"),patch.object(run.first,"run_policy",side_effect=fake):
                run.worker(out,0,1)
                original=(out/"raw/test/worker_0.jsonl").read_bytes()
                run.worker(out,0,1)
                self.assertEqual(len(calls),33)
                self.assertEqual(original,(out/"raw/test/worker_0.jsonl").read_bytes())


if __name__=="__main__":unittest.main()
