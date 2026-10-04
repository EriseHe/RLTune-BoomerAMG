"""Audit the checkpoint-refresh follow-up and compare policy roles with Run 02."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from experiments.paper_final import run_05_policy as first
from experiments.paper_final import run_05_policy_repeat as repeat
from experiments.paper_final.run_05_policy_refresh import verify
from experiments.paper_final.analyze_05_policy_repeat import collect_cells,statistics


def analyze(output):
    output=Path(output);verify(output)
    protocol=first.read(output/"protocol.json");selections=first.read(output/"selections.json")
    jobs=first.read(output/"jobs_test.json");marker=first.read(output/"raw/test/complete.json")
    rows=[];seen=set();traces={};incomplete={}
    for name,digest in marker["raw_sha256"].items():
        if first.file_hash(output/name)!=digest:raise ValueError("Raw data changed")
        for row in first.read_records(output/name):
            key=first.cell_key(row)
            if key in seen:raise ValueError("Duplicate trial")
            seen.add(key);first.audit_outcome(row["outcome"])
            if row["outcome"].get("controller_update_committed",False):raise ValueError("Frozen controller learned")
            rows.append(row)
            if row["repeat"]==0:
                k=(row["seed"],row["case_id"],row["policy"]);outcome=row["outcome"]
                traces[k]={f:np.asarray(outcome.get("cycle_"+f,[]),dtype=float) for f in ("actions","residuals","times")}
                if len({len(a) for a in traces[k].values()})!=1:raise ValueError("Misaligned cycle trace")
                incomplete[k]=bool(outcome.get("fallback_used",False) or not row["success"])
    if seen!=repeat.expected_cells(jobs) or len(seen)!=marker["trials"]:raise ValueError("Incomplete evaluation coverage")
    cells=collect_cells(rows,[0,1,2]);by_rep={(r["seed"],r["case_id"],r["policy"],r["repeat"]):first.compact_row(r) for r in rows}
    raw={(r["seed"],r["case_id"],r["policy"],r["repeat"]):r for r in rows}
    jobmap={(j["seed"],j["case_id"]):j for j in jobs if j["repeat"]==0}
    parent=Path(protocol["origin_run"]);old=first.read(parent/"analysis/summary.json")
    if old["seeds"]!=protocol["training_seeds"]:raise ValueError("Checkpoint labels misaligned")
    seeds=protocol["training_seeds"];cases=list(range(protocol["test_cases"]))
    fields={"native":"native_continuation_sec","inclusive":"inclusive_continuation_sec",
            "native_total":"native_total_sec","inclusive_total":"inclusive_total_sec","cycles":"primary_cycles"}
    costs={m:{f:[] for f in fields} for m in repeat.METHODS}
    per_seed=[];per_repeat=[];cross=[];diagnostics=[]
    for i,seed in enumerate(seeds):
        reference=first.totals([cells[seed,c,jobmap[seed,c]["method_policies"]["reference"]] for c in cases])
        for method in repeat.METHODS:
            keys=[(seed,c,jobmap[seed,c]["method_policies"][method]) for c in cases]
            current=[cells[k] for k in keys];total=first.totals(current)
            for short,long in fields.items():costs[method][short].append([r[long] for r in current])
            gain=first.reduction(reference["native_continuation_sec"],total["native_continuation_sec"])
            per_seed.append({"seed":seed,"method":method,"label":repeat.LABELS[method],**total,
                "native_reduction_vs_w1_pct":gain,
                "inclusive_reduction_vs_w1_pct":first.reduction(reference["inclusive_continuation_sec"],total["inclusive_continuation_sec"]),
                "mean_cycles":total["primary_cycles"]/len(cases)})
            prior=sum(old["costs"][method]["native"][i]);prior_reference=sum(old["costs"]["reference"]["native"][i])
            prior_gain=first.reduction(prior_reference,prior)
            cross.append({"seed":seed,"method":method,"run02_native_sec":prior,"run03_native_sec":total["native_continuation_sec"],
                "run02_reduction_pct":prior_gain,"run03_reduction_pct":gain,"change_pp":gain-prior_gain,
                "checkpoint_and_hierarchies_refreshed":seed==4,
                "comparison":"Same policy role; seed 4 has new Joint state and newly selected fixed baselines" if seed==4 else "Same frozen policy and hierarchy choices"})
            cvs=[];changed_actions=0;changed_cycles=0
            for k in keys:
                observations=[raw[(*k,rep)] for rep in range(3)]
                tt=[r["native_continuation_sec"] for r in observations]
                cvs.append(float(100*np.std(tt,ddof=1)/np.mean(tt)))
                actions=[r["outcome"].get("cycle_actions",[]) for r in observations]
                changed_actions+=any(a!=actions[0] for a in actions[1:]);changed_cycles+=len({len(a) for a in actions})>1
            diagnostics.append({"seed":seed,"method":method,"median_timing_cv_pct":float(np.median(cvs)),
                "p90_timing_cv_pct":float(np.percentile(cvs,90)),"changed_action_cases":changed_actions,"changed_cycle_cases":changed_cycles})
            for rep in range(3):
                total_rep=first.totals([by_rep[(*k,rep)] for k in keys])
                ref=first.totals([by_rep[seed,c,jobmap[seed,c]["method_policies"]["reference"],rep] for c in cases])
                per_repeat.append({"seed":seed,"method":method,"repeat":rep+1,**total_rep,
                    "native_reduction_vs_w1_pct":first.reduction(ref["native_continuation_sec"],total_rep["native_continuation_sec"])})
    aggregate=[];comparisons=[]
    for method in repeat.METHODS:
        rr=[r for r in per_seed if r["method"]==method]
        aggregate.append({"method":method,"label":repeat.LABELS[method],
            "native_reduction_vs_w1_pct":statistics([r["native_reduction_vs_w1_pct"] for r in rr]),
            "inclusive_reduction_vs_w1_pct":statistics([r["inclusive_reduction_vs_w1_pct"] for r in rr]),
            "failed_case_seed_pairs":sum(r["failures"] for r in rr),"recovered_case_seed_pairs":sum(r["recoveries"] for r in rr)})
        if method=="rl":continue
        for field in ("native","inclusive"):
            base=np.asarray(costs[method][field]);candidate=np.asarray(costs["rl"][field])
            gains=100*(1-candidate.sum(axis=1)/base.sum(axis=1))
            comparisons.append({"comparator":method,"cost":field,**statistics(gains),"winning_seeds":int(np.sum(gains>0)),
                "case_wins_by_seed":np.sum(candidate<base,axis=1).tolist()})
    payload={"run_number":3,"comparison_run":2,"created_at":first.now(),"protocol":protocol,"seeds":seeds,
        "methods":list(repeat.METHODS),"labels":repeat.LABELS,"main_methods":list(repeat.MAIN_METHODS),
        "case_order_0_based":selections["case_order_0_based"],"costs":costs,"comparison_costs":old["costs"],
        "per_seed":per_seed,"per_repeat":per_repeat,"aggregate":aggregate,"rl_comparisons":comparisons,
        "run02_comparison":cross,"repetition_diagnostics":diagnostics,
        "audit":{"trials":len(rows),"exact_coverage":True,"all_repetitions_present":True,
            "failures":sum(not r["success"] for r in rows),"recovered_trials":sum(bool(r["outcome"].get("fallback_used",False)) for r in rows),
            "raw_sha256":marker["raw_sha256"]},"scan_audit":first.read(output/"raw/scan/complete.json"),
        "interpretation":"Three fresh repetitions of each case; recovery retained; only seed 4 has new setup and RL checkpoints. Fixed choices are independently retimed. This is a diagnostic checkpoint refresh, not six new training seeds."}
    width=max(len(t["actions"]) for t in traces.values());arrays={"case_order_0_based":np.asarray(selections["case_order_0_based"])}
    for seed in seeds:
        for method in repeat.METHODS:
            keys=[(seed,c,jobmap[seed,c]["method_policies"][method]) for c in cases]
            for field in ("actions","residuals","times"):
                matrix=np.full((len(cases),width),np.nan)
                for i,k in enumerate(keys):matrix[i,:len(traces[k][field])]=traces[k][field]
                arrays[f"seed{seed}__{method}__{field}"]=matrix
            arrays[f"seed{seed}__{method}__primary_incomplete"]=np.asarray([incomplete[k] for k in keys])
    np.savez_compressed(output/"analysis/trace_arrays.npz",**arrays);first.dump(output/"analysis/summary.json",payload)
    for name,data in (("per_seed",per_seed),("per_repeat",per_repeat),("run02_comparison",cross),("repetition_diagnostics",diagnostics)):
        first.write_csv(output/"analysis"/(name+".csv"),data)
    first.emit(output,"Run 03 analysis complete",trials=len(rows),failures=payload["audit"]["failures"])
    return payload
