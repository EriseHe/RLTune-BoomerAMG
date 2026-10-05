"""Audit Run 02 and compare newly measured policies with their Run 01 timings."""
from __future__ import annotations

from experiments.paper_final.common.policy_analysis import (
    statistics, collect_cells,
)

from collections import defaultdict
from pathlib import Path
import numpy as np

from experiments.archive.paper_development import run_05_policy as first
from experiments.archive.paper_development.run_05_policy_repeat import METHODS, MAIN_METHODS, LABELS, SOURCE, expected_cells, verify


def analyze(output):
    output = Path(output)
    verify(output)
    protocol = first.read(output / "protocol.json")
    selections = first.read(output / "selections.json")
    jobs = first.read(output / "jobs_test.json")
    parent = Path(protocol["origin_run"])
    marker = first.read(output / "raw/test/complete.json")
    rows, seen, traces, primary_incomplete = [], set(), {}, {}
    for name, expected_hash in marker["raw_sha256"].items():
        if first.file_hash(output / name) != expected_hash:
            raise ValueError(f"Raw timing data changed: {name}")
        for r in first.read_records(output / name):
            key = first.cell_key(r)
            if key in seen:
                raise ValueError("Duplicated trial")
            seen.add(key)
            first.audit_outcome(r["outcome"])
            if r["outcome"].get("controller_update_committed", False):
                raise ValueError("Frozen RL unexpectedly learned")
            rows.append(r)
            if r["repeat"] == 0:
                k = (r["seed"], r["case_id"], r["policy"])
                o = r["outcome"]
                traces[k] = {field: np.asarray(o.get("cycle_"+field, []), dtype=float) for field in ("actions", "residuals", "times")}
                if len({len(a) for a in traces[k].values()}) != 1:
                    raise ValueError("Unaligned action/residual/time trajectory")
                primary_incomplete[k] = bool(o.get("fallback_used", False) or not r["success"])
    if seen != expected_cells(jobs) or len(seen) != marker["trials"]:
        raise ValueError("Run 02 evaluation coverage is incomplete")
    cells = collect_cells(rows, [0, 1, 2])
    by_repeat = {(r["seed"],r["case_id"],r["policy"],r["repeat"]):first.compact_row(r) for r in rows}
    jobmap = {(j["seed"],j["case_id"]):j for j in jobs if j["repeat"] == 0}
    needed = set(cells)
    previous, previous_traces = [], {}
    for phase in ("test", "repetitions"):
        m = first.read(parent / "raw" / phase / "complete.json")
        for name, saved_hash in m["raw_sha256"].items():
            if first.file_hash(parent / name) != saved_hash:
                raise ValueError("Run 01 raw reference changed")
            for r in first.read_records(parent / name):
                k = (r["seed"],r["case_id"],r["policy"])
                if r["source"] == SOURCE and k in needed:
                    previous.append(r)
                    if r["repeat"] == 0: previous_traces[k] = r["outcome"].get("cycle_actions", [])
    old = collect_cells(previous, None)
    if set(old) != needed:
        raise AssertionError("Missing Run 01 reference for a selected policy")
    seeds, cases = protocol["training_seeds"], list(range(protocol["test_cases"]))
    field_map = {"native": "native_continuation_sec", "inclusive": "inclusive_continuation_sec",
                 "native_total": "native_total_sec", "inclusive_total": "inclusive_total_sec", "cycles": "primary_cycles"}
    costs = {m:{f:[] for f in field_map} for m in METHODS}
    old_costs = {m:{f:[] for f in field_map} for m in METHODS}
    summaries, repeats, cross_run, repetition_diagnostics = [], [], [], []
    for seed in seeds:
        reference = first.totals([cells[seed,c,jobmap[seed,c]["method_policies"]["reference"]] for c in cases])
        old_reference = first.totals([old[seed,c,jobmap[seed,c]["method_policies"]["reference"]] for c in cases])
        for method in METHODS:
            keys = [(seed,c,jobmap[seed,c]["method_policies"][method]) for c in cases]
            current, before = [cells[k] for k in keys], [old[k] for k in keys]
            total, prior = first.totals(current), first.totals(before)
            for short, long in field_map.items():
                costs[method][short].append([r[long] for r in current])
                old_costs[method][short].append([r[long] for r in before])
            native = first.reduction(reference["native_continuation_sec"],total["native_continuation_sec"])
            inclusive = first.reduction(reference["inclusive_continuation_sec"],total["inclusive_continuation_sec"])
            summaries.append({"seed": seed, "method": method, "label": LABELS[method], **total,
                "native_reduction_vs_w1_pct": native, "inclusive_reduction_vs_w1_pct": inclusive,
                "mean_cycles": total["primary_cycles"]/len(cases)})
            prior_gain = first.reduction(old_reference["native_continuation_sec"],prior["native_continuation_sec"])
            cross_run.append({"seed": seed, "method": method, "run01_native_sec": prior["native_continuation_sec"],
                "run02_native_sec": total["native_continuation_sec"],
                "runtime_change_pct": 100*(total["native_continuation_sec"]/prior["native_continuation_sec"]-1),
                "run01_saving_vs_w1_pct": prior_gain, "run02_saving_vs_w1_pct": native,
                "saving_change_pp": native-prior_gain, "run01_mean_cycles": prior["primary_cycles"]/len(cases),
                "run02_mean_cycles": total["primary_cycles"]/len(cases),
                "changed_first_pass_actions": sum(not np.array_equal(traces[k]["actions"],previous_traces[k]) for k in keys)})
            cvs, changed_actions, changed_cycles = [], 0, 0
            rawmap = {(r["seed"],r["case_id"],r["policy"],r["repeat"]):r for r in rows if r["seed"] == seed}
            for k in keys:
                rr = [rawmap[(*k,rep)] for rep in range(3)]
                tt = [r["native_continuation_sec"] for r in rr]
                cvs.append(float(100*np.std(tt,ddof=1)/np.mean(tt)))
                actions = [r["outcome"].get("cycle_actions",[]) for r in rr]
                changed_actions += any(a != actions[0] for a in actions[1:])
                changed_cycles += len({len(a) for a in actions}) > 1
            repetition_diagnostics.append({"seed":seed,"method":method,"median_timing_cv_pct":float(np.median(cvs)),
                "p90_timing_cv_pct":float(np.percentile(cvs,90)),"changed_action_cases":changed_actions,"changed_cycle_cases":changed_cycles})
            for rep in range(3):
                rt = first.totals([by_repeat[(*k,rep)] for k in keys])
                ref = first.totals([by_repeat[seed,c,jobmap[seed,c]["method_policies"]["reference"],rep] for c in cases])
                repeats.append({"seed":seed,"method":method,"repeat":rep+1,**rt,
                    "native_reduction_vs_w1_pct":first.reduction(ref["native_continuation_sec"],rt["native_continuation_sec"])})
    aggregates = []
    for method in METHODS:
        rr = [r for r in summaries if r["method"] == method]
        aggregates.append({"method":method,"label":LABELS[method],
            "native_reduction_vs_w1_pct":statistics([r["native_reduction_vs_w1_pct"] for r in rr]),
            "inclusive_reduction_vs_w1_pct":statistics([r["inclusive_reduction_vs_w1_pct"] for r in rr]),
            "failed_case_seed_pairs":sum(r["failures"] for r in rr),"recovered_case_seed_pairs":sum(r["recoveries"] for r in rr)})
    comparisons = []
    for method in METHODS:
        if method == "rl": continue
        for field in ("native","inclusive"):
            base=np.asarray(costs[method][field]); candidate=np.asarray(costs["rl"][field])
            gains=100*(1-candidate.sum(axis=1)/base.sum(axis=1))
            comparisons.append({"comparator":method,"cost":field,**statistics(gains),
                "winning_seeds":int(np.sum(gains>0)),"case_wins_by_seed":np.sum(candidate<base,axis=1).tolist()})
    payload = {"run_number":2,"created_at":first.now(),"protocol":protocol,"seeds":seeds,"methods":list(METHODS),
        "labels":LABELS,"main_methods":list(MAIN_METHODS),"case_order_0_based":selections["case_order_0_based"],
        "costs":costs,"run01_costs":old_costs,"per_seed":summaries,"aggregate":aggregates,
        "rl_comparisons":comparisons,"per_repeat":repeats,"run01_comparison":cross_run,
        "repetition_diagnostics":repetition_diagnostics,
        "audit":{"trials":len(rows),"exact_coverage":True,"all_repetitions_present":True,
                 "failures":sum(not r["success"] for r in rows),"recovered_trials":sum(bool(r["outcome"].get("fallback_used",False)) for r in rows),
                 "raw_sha256":marker["raw_sha256"]},
        "interpretation":"Costs are averaged over all three repetitions within each case, then summed. Six seed points reuse the same 100 inputs. Fixed weights are frozen Run 01 grid selections; no Run 02 minima are claimed. All attempts and recovery costs remain charged."}
    arrays = {"case_order_0_based":np.asarray(selections["case_order_0_based"])}
    width=max(len(t["actions"]) for t in traces.values())
    for seed in seeds:
        for method in METHODS:
            keys=[(seed,c,jobmap[seed,c]["method_policies"][method]) for c in cases]
            for field in ("actions","residuals","times"):
                matrix=np.full((len(cases),width),np.nan)
                for i,k in enumerate(keys): matrix[i,:len(traces[k][field])]=traces[k][field]
                arrays[f"seed{seed}__{method}__{field}"]=matrix
            arrays[f"seed{seed}__{method}__primary_incomplete"]=np.asarray([primary_incomplete[k] for k in keys])
    np.savez_compressed(output / "analysis/trace_arrays.npz",**arrays)
    first.dump(output / "analysis/summary.json",payload)
    for name, records in [("per_seed",summaries),("per_repeat",repeats),("run01_comparison",cross_run),("repetition_diagnostics",repetition_diagnostics)]:
        first.write_csv(output / "analysis" / (name+".csv"),records)
    first.emit(output,"Run 02 analysis complete",trials=len(rows),failures=payload["audit"]["failures"])
    return payload
