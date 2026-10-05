"""Module 05 Run 04: replace (2.5,1) by the prescribed anchored pair (2.6,1)."""
from __future__ import annotations

from experiments.runtime import (
    configure_single_thread, prevent_sleep, stop_sleep_prevention, single_thread_environment,
)

if __name__ == "__main__":
    configure_single_thread()

from experiments.paper_final.common.workers import run_frozen_worker, run_frozen_phase
from experiments.archive.paper_development import run_05_policy as first
from experiments.archive.paper_development import run_05_policy_repeat as repeat
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

ROOT = first.ROOT
PARENT = ROOT/"results/paper_final/05_policy/20260927_run03_new_seed4_joint_6seeds_100cases"
DEFAULT = ROOT/"results/paper_final/05_policy/20260927_run04_anchored26_joint_6seeds_100cases"
OLD_POLICY = "periodic_2.50_1.00"
POLICY = "periodic_2.60_1.00"
METHODS = repeat.METHODS
MAIN_METHODS = ("fixed", "oracle", "periodic", "rl")
LABELS = {**repeat.LABELS, "periodic":"Anchored (2.6, 1)"}
THEORY_FILES = ["docs/theory/anchored_schedule_review_20260927.tex",
    "docs/theory/anchored_schedule_review_20260927.md",
    "docs/theory/anchored_schedule_verification_20260927.json"]


def replace_schedule(jobs):
    """Preserve Run 03 case and policy order, changing just the prescribed pair."""
    updated=copy.deepcopy(jobs)
    for j in updated:
        if j["method_policies"]["periodic"] != OLD_POLICY:
            raise ValueError("Unexpected parent schedule")
        j["method_policies"]["periodic"]=POLICY
        j["policy_order"]=[POLICY if p==OLD_POLICY else p for p in j["policy_order"]]
    return updated


def verify(output):
    first.verify_inputs(output)
    saved=first.read(output/"prepared.json")
    if first.file_hash(output/"selections.json")!=saved["selections_sha256"]:
        raise RuntimeError("Prespecified selections changed")


def prepare(output,parent=PARENT):
    if output.exists() and any(output.iterdir()):raise ValueError("Use a new Run 04 directory")
    first.verify_inputs(parent,check_source=False)
    saved=first.read(parent/"prepared.json")
    if first.file_hash(parent/"selections.json")!=saved["selections_sha256"]:
        raise RuntimeError("Parent selections changed")
    if first.read(parent/"complete.json")["status"]!="complete":raise ValueError("Incomplete parent")
    math=first.read(ROOT/THEORY_FILES[-1])
    if not math["math"]["passed"] or not math["implementation"]["passed"]:
        raise ValueError("Theory/profile audit must pass before preparation")
    manifest=first.read(parent/"source_manifest.json")
    differences=first.source_differences(manifest)
    numerical=[p for p in differences if not any(s in p for s in ("plot_","analyze_","test_","module05_figures/"))]
    if numerical:raise RuntimeError(f"Parent execution sources changed: {numerical}")
    output.mkdir(parents=True)
    for name in ("analysis","progress","raw/test","provenance/source","theory"):
        (output/name).mkdir(parents=True,exist_ok=True)
    shutil.copytree(parent/"checkpoints",output/"checkpoints")
    for name in ("inputs.json","jobs_base_test.json","jobs_preflight.json"):
        shutil.copy2(parent/name,output/name)
    jobs=replace_schedule(first.read(parent/"jobs_test.json"))
    chosen=first.read(parent/"selections.json")
    for key,value in chosen["chosen_policies"].items():
        if key.endswith("/periodic"):
            if value!=OLD_POLICY:raise ValueError("Parent selection mismatch")
            chosen["chosen_policies"][key]=POLICY
    chosen.update(run_number=4,origin_run=str(parent),selection_frozen_before_final_retiming=True,
        note="Run 03 fixed choices retained; (2.6,1) prescribed by the anchored 0.05-grid minimax rule before Run 04.")
    old=first.read(parent/"protocol.json");protocol=copy.deepcopy(old)
    for name in ("scan_execution_count","scan_extra_repeat_case_ids","changed_test_hierarchies_seed4","conventional_references"):
        protocol.pop(name,None)
    protocol["policies"].pop(OLD_POLICY)
    protocol["policies"][POLICY]={"kind":"periodic","pattern":[2.6,1.]}
    protocol.update(run_number=4,comparison_run=3,created_at=first.now(),origin_run=str(parent),
        scope="Matched repeat of Run 03 with a theory-prescribed (2.6,1) schedule",
        unchanged_checkpoint_seeds=list(range(1,7)),main_heatmap_methods=list(MAIN_METHODS),
        labels=LABELS,execution_count=len(repeat.expected_cells(jobs)),
        selection_rule="All Run 03 fixed-weight choices reused, including its refreshed seed-4 scan; no new scan or outcome selection",
        periodic_rule="(2.6,1) is the analytic anchored minimax choice on {1,1.05,...,3}; high first, reset each primary solve",
        test_schedule_rule="Prescribed (2.6,1) and retained (1,3); same full-cycle action and charged recovery",
        theory_scope="Normalized SPD smoothing surrogate and fixed exact two-grid bound; no multilevel runtime-optimality guarantee",
        ordering_rule="Exact Run 03 case/policy/repetition order, with only the periodic policy identifier replaced",
        interpretation="Same accepted checkpoints and previously observed test cases; a prescribed policy follow-up, not new independent seeds or a fresh holdout",
        sorting="Run 03 baseline-difficulty case order retained; no sorting on Run 04 outcomes")
    first.dump(output/"protocol.json",protocol);first.dump(output/"jobs_test.json",jobs)
    first.dump(output/"selections.json",chosen)
    for relative in THEORY_FILES:shutil.copy2(ROOT/relative,output/"theory"/Path(relative).name)
    extras=["run_05_policy_anchored.py","analyze_05_policy_anchored.py","plot_05_policy_anchored.py",
            "test_05_policy_anchored.py","verify_anchored_schedule.py"]
    paths=set(manifest)|first.support_source_paths()|{"experiments/archive/paper_development/"+p for p in extras}|set(THEORY_FILES)
    manifest={p:first.file_hash(ROOT/p) for p in paths}
    for relative in manifest:
        target=output/"provenance/source"/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/relative,target)
    first.dump(output/"source_manifest.json",manifest)
    first.dump(output/"parent_audit.json",{"parent":str(parent),"checkpoint_and_input_hashes_verified":True,
        "unchanged_execution_sources":True,"historical_reporting_source_differences":differences,
        "parent_raw_complete_sha256":first.file_hash(parent/"raw/test/complete.json"),
        "parent_selections_sha256":first.file_hash(parent/"selections.json"),
        "parent_summary_sha256":first.file_hash(parent/"analysis/summary.json")})
    first.dump(output/"prepared.json",{"at":first.now(),"protocol_sha256":first.file_hash(output/"protocol.json"),
        "inputs_sha256":first.file_hash(output/"inputs.json"),"selections_sha256":first.file_hash(output/"selections.json"),
        "jobs_sha256":{p:first.file_hash(output/f"jobs_{p}.json") for p in ("base_test","preflight","test")},
        "checkpoint_sha256":{str(p.relative_to(output)):first.file_hash(p) for p in (output/"checkpoints").rglob("*") if p.is_file()}})
    first.environment_check(output)
    first.emit(output,"Run 04 prepared",trials=protocol["execution_count"],pair=[2.6,1.])


def preflight(output):
    verify(output);protocol=first.read(output/"protocol.json");jobs=first.read(output/"jobs_preflight.json")
    checks=[]
    for seed in range(1,7):
        bundle=first.load_bundle(output,seed);before=first.frozen_snapshot(bundle)
        job=next(j for j in jobs if j["seed"]==seed and j["case_id"]==0)
        names=["fixed_1.00",POLICY,"periodic_1.00_3.00","rl_frozen_lcb",POLICY,"fixed_1.00"]
        results=[first.run_policy(job,n,protocol["policies"][n],bundle) for n in names]
        for x,y in ((0,5),(1,4)):
            np.testing.assert_allclose(results[x]["outcome"]["cycle_residuals"],results[y]["outcome"]["cycle_residuals"],rtol=1e-12,atol=1e-15)
        actions=results[1]["outcome"]["cycle_actions"]
        assert actions and actions==[[2.6,1.][i%2] for i in range(len(actions))]
        first.verify_frozen(bundle,before)
        checks.append({"seed":seed,"repeated_reference_and_schedule_agree":True,"phase_resets":True,"controller_frozen":True})
    first.dump(output/"preflight.json",{"at":first.now(),"passed":True,"checks":checks})


def run(output):
    if not (output/"prepared.json").exists():prepare(output)
    with (output/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        awake = prevent_sleep()
        def stop(signum,frame):raise KeyboardInterrupt(f"Signal {signum}")
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            verify(output);first.environment_check(output)
            if not (output/"preflight.json").exists():preflight(output)
            if not (output/"raw/test/complete.json").exists():run_phase(output,"test")
            from experiments.archive.paper_development.analyze_05_policy_anchored import analyze
            first.dump(output/"status.json",{"status":"running","phase":"analysis","run_number":4,"at":first.now()})
            analyze(output);verify(output)
            first.dump(output/"numerical_complete.json",{"status":"complete","run_number":4,"at":first.now()})
            from experiments.archive.paper_development.plot_05_policy_anchored import generate
            first.dump(output/"status.json",{"status":"running","phase":"figures","run_number":4,"at":first.now()})
            generate(output);verify(output)
            final={"status":"complete","phase":"complete","run_number":4,"at":first.now(),"elapsed_sec":time.monotonic()-start}
            first.dump(output/"complete.json",final);first.dump(output/"status.json",final)
        except BaseException as exc:
            first.dump(output/"status.json",{"status":"failed","phase":"stopped","run_number":4,"at":first.now(),"error":str(exc)})
            raise
        finally:
            stop_sleep_prevention(awake)


# The durable worker and phase supervisor below retain Run 03's execution,
# recovery, timing, and resume semantics; their run metadata is now 4.

def worker(output, phase, worker_id, workers):
    run_frozen_worker(output, phase, worker_id, workers, run_number=4, verify=verify, policy=first)


def run_phase(output, phase, workers=3):
    run_frozen_phase(output, phase, runner_module=__spec__.name, run_number=4, workers=workers)



if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run","worker"))
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--phase",choices=("test",),default="test")
    parser.add_argument("--worker",type=int)
    args=parser.parse_args()
    if args.command=="worker":worker(args.output,args.phase,args.worker,3)
    elif args.command=="prepare":prepare(args.output)
    else:run(args.output)
