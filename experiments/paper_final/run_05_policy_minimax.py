"""Historical Module 05 Run 05: six frozen checkpoints and free pair (2.85,1.10)."""
from __future__ import annotations

from experiments.runtime import (
    configure_single_thread, prevent_sleep, stop_sleep_prevention, single_thread_environment,
)

if __name__ == "__main__":
    configure_single_thread()

from experiments.paper_final import run_05_policy as first
from experiments.paper_final import run_05_policy_repeat as repeat
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
PARENT = ROOT/"results/paper_final/05_policy/20260927_run04_anchored26_joint_6seeds_100cases"
DEFAULT = ROOT/"results/paper_final/05_policy/20260929_run05_minimax285_joint_6seeds_100cases"
OLD_POLICY = "periodic_2.60_1.00"
POLICY = "periodic_2.85_1.10"
REMOVED_POLICY = "periodic_1.00_3.00"
METHODS = ("reference", "fixed", "oracle", "periodic", "rl")
MAIN_METHODS = ("fixed", "oracle", "periodic", "rl")
LABELS = {"reference":"Fixed w=1", "fixed":"Stream-wide fixed*",
          "oracle":"Per-instance fixed*", "periodic":"Periodic (2.85, 1.10)",
          "rl":"Frozen RL"}
THEORY_FILES = ["docs/theory/period_two_weighted_minimax_20260928.md",
    "docs/theory/anchored_schedule_verification_20260927.json",
    "docs/theory/period_two_minimax_verification_20260928.json"]


def replace_schedule(jobs):
    """Retain parent jobs/order; substitute the free pair and remove (1,3)."""
    updated=copy.deepcopy(jobs)
    for j in updated:
        if j["method_policies"]["periodic"]!=OLD_POLICY:
            raise ValueError("Unexpected parent schedule")
        if j["method_policies"].pop("periodic13")!=REMOVED_POLICY:
            raise ValueError("Unexpected endpoint schedule")
        j["method_policies"]["periodic"]=POLICY
        j["policy_order"]=[POLICY if n==OLD_POLICY else n
                           for n in j["policy_order"] if n!=REMOVED_POLICY]
        if set(j["method_policies"])!=set(METHODS):
            raise ValueError("Unexpected method roles")
        if set(j["policy_order"])!=set(j["method_policies"].values()):
            raise ValueError("Planned execution differs from method map")
    return updated

def verify(output):
    first.verify_inputs(output)
    saved=first.read(output/"prepared.json")
    if first.file_hash(output/"selections.json")!=saved["selections_sha256"]:
        raise RuntimeError("Prespecified selections changed")
    protocol=first.read(output/"protocol.json")
    if protocol["run_number"]!=5 or protocol["methods"]!=list(METHODS):
        raise RuntimeError("Incorrect Run 05 method roster")
    periodic={k:v for k,v in protocol["policies"].items() if v["kind"]=="periodic"}
    if periodic!={POLICY:{"kind":"periodic","pattern":[2.85,1.10]}}:
        raise RuntimeError("Incorrect prescribed periodic policy")
    for job in first.read(output/"jobs_test.json"):
        names=job["policy_order"]
        if (set(job["method_policies"])!=set(METHODS)
                or job["method_policies"]["periodic"]!=POLICY
                or set(names)!=set(job["method_policies"].values())
                or len(names)!=len(set(names))
                or OLD_POLICY in names or REMOVED_POLICY in names):
            raise RuntimeError("Incorrect evaluation job")


def prepare(output,parent=PARENT):
    if output.exists() and any(output.iterdir()):raise ValueError("Use a new Run 05 directory")
    first.verify_inputs(parent,check_source=False)
    saved=first.read(parent/"prepared.json")
    if first.file_hash(parent/"selections.json")!=saved["selections_sha256"]:
        raise RuntimeError("Parent selections changed")
    if first.read(parent/"complete.json")["status"]!="complete":raise ValueError("Incomplete parent")
    math=first.read(ROOT/THEORY_FILES[-1])
    implementation=first.read(ROOT/THEORY_FILES[1])
    if (not math["passed"] or math["grid_minimizers_ordered"]!=[["2.85","1.10"],["1.10","2.85"]]
            or not implementation["implementation"]["passed"]):
        raise ValueError("Free minimax and smoother-profile audits must pass")
    manifest=first.read(parent/"source_manifest.json")
    differences=[p for p,h in manifest.items() if first.file_hash(ROOT/p)!=h]
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
    chosen["chosen_policies"]={k:v for k,v in chosen["chosen_policies"].items()
        if "/bandit_lstdq/" in k and k.rsplit("/",1)[-1] in METHODS}
    for key,value in chosen["chosen_policies"].items():
        if key.endswith("/periodic"):
            if value!=OLD_POLICY:raise ValueError("Parent selection mismatch")
            chosen["chosen_policies"][key]=POLICY
    chosen.update(run_number=5,origin_run=str(parent),selection_frozen_before_final_retiming=True,
        note="Run 04 fixed choices retained; (2.85,1.10) prescribed by the free 0.05-grid weighted-minimax rule before Run 05.")
    old=first.read(parent/"protocol.json");protocol=copy.deepcopy(old)
    for name in ("scan_execution_count","scan_extra_repeat_case_ids","changed_test_hierarchies_seed4","conventional_references"):
        protocol.pop(name,None)
    protocol["policies"].pop(OLD_POLICY)
    protocol["policies"].pop(REMOVED_POLICY)
    protocol["policies"][POLICY]={"kind":"periodic","pattern":[2.85,1.10]}
    used={n for j in jobs for n in j["policy_order"]}
    protocol["policies"]={k:v for k,v in protocol["policies"].items() if k in used}
    protocol.update(run_number=5,comparison_run=4,methods=list(METHODS),created_at=first.now(),origin_run=str(parent),
        scope="Matched repeat of Run 04 with a theory-prescribed (2.85,1.10) schedule",
        unchanged_checkpoint_seeds=list(range(1,7)),main_heatmap_methods=list(MAIN_METHODS),
        labels=LABELS,execution_count=len(repeat.expected_cells(jobs)),
        selection_rule="All Run 04 fixed-weight choices reused, including its refreshed seed-4 scan; no new scan or outcome selection",
        periodic_rule="(2.85,1.10) is the joint free weighted-minimax choice on {1,1.05,...,3}; high first, reset each primary solve",
        test_schedule_rule="Only prescribed (2.85,1.10); (1,3) and (2.6,1) excluded; same full-cycle action and charged recovery",
        theory_scope="Normalized SPD smoothing surrogate and fixed exact two-grid bound; no multilevel runtime-optimality guarantee",
        ordering_rule="Exact Run 04 case/policy/repetition order, with the periodic pair replaced and endpoint policy removed",
        interpretation="Same accepted checkpoints and previously observed test cases; a prescribed policy follow-up, not new independent seeds or a fresh holdout",
        sorting="Run 04 baseline-difficulty case order retained; no sorting on Run 05 outcomes")
    first.dump(output/"protocol.json",protocol);first.dump(output/"jobs_test.json",jobs)
    first.dump(output/"selections.json",chosen)
    for relative in THEORY_FILES:shutil.copy2(ROOT/relative,output/"theory"/Path(relative).name)
    extras=["run_05_policy_minimax.py","analyze_05_policy_minimax.py",
            "test_05_policy_minimax.py","verify_period_two_minimax.py"]
    paths=set(manifest)|{"experiments/paper_final/"+p for p in extras}|set(THEORY_FILES)
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
    first.emit(output,"Run 05 prepared",trials=protocol["execution_count"],pair=[2.85,1.10])


def preflight(output):
    verify(output);protocol=first.read(output/"protocol.json");jobs=first.read(output/"jobs_preflight.json")
    checks=[]
    for seed in range(1,7):
        bundle=first.load_bundle(output,seed);before=first.frozen_snapshot(bundle)
        job=next(j for j in jobs if j["seed"]==seed and j["case_id"]==0)
        constant=first.read(output/"selections.json")["chosen_policies"][f"{seed}/bandit_lstdq/fixed"]
        names=["fixed_1.00",constant,POLICY,"rl_frozen_lcb",POLICY,"fixed_1.00"]
        results=[first.run_policy(job,n,protocol["policies"][n],bundle) for n in names]
        for x,y in ((0,5),(2,4)):
            np.testing.assert_allclose(results[x]["outcome"]["cycle_residuals"],results[y]["outcome"]["cycle_residuals"],rtol=1e-12,atol=1e-15)
        actions=results[2]["outcome"]["cycle_actions"]
        assert actions and actions==[[2.85,1.10][i%2] for i in range(len(actions))]
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
            from experiments.paper_final.analyze_05_policy_minimax import analyze
            first.dump(output/"status.json",{"status":"running","phase":"analysis","run_number":5,"at":first.now()})
            analyze(output);verify(output)
            first.dump(output/"numerical_complete.json",{"status":"complete","run_number":5,"at":first.now()})
            final={"status":"complete","phase":"complete","run_number":5,"at":first.now(),"elapsed_sec":time.monotonic()-start}
            first.dump(output/"complete.json",final);first.dump(output/"status.json",final)
        except BaseException as exc:
            first.dump(output/"status.json",{"status":"failed","phase":"stopped","run_number":5,"at":first.now(),"error":str(exc)})
            raise
        finally:
            stop_sleep_prevention(awake)


# The durable worker and phase supervisor below retain Run 04's execution,
# recovery, timing, and resume semantics; their run metadata is now 5.

def worker(output,phase,worker_id,workers):
    if workers not in range(1,4) or worker_id not in range(workers):raise ValueError("At most three workers")
    directory=output/"raw"/phase;directory.mkdir(parents=True,exist_ok=True)
    with (directory/f"worker_{worker_id}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify(output)
        protocol=first.read(output/"protocol.json");jobs=first.read(output/f"jobs_{phase}.json")[worker_id::workers]
        expected=repeat.expected_cells(jobs);path=directory/f"worker_{worker_id}.jsonl";done=set();wall=0.
        for row in first.read_records(path,repair_tail=True):
            key=first.cell_key(row)
            if key not in expected or key in done:raise ValueError("Duplicate/foreign trial")
            done.add(key);wall+=row["wall_sec"]
        state={"phase":phase,"worker":worker_id,"workers":workers,"pid":os.getpid(),"done":len(done),"total":len(expected),"status":"running"}
        bundles={};snapshots={}
        with path.open("a",buffering=1) as handle:
            for job in jobs:
                seed=job["seed"]
                if seed not in bundles:bundles[seed]=first.load_bundle(output,seed);snapshots[seed]=first.frozen_snapshot(bundles[seed])
                for name in job["policy_order"]:
                    key=first.cell_key({**job,"policy":name})
                    if key in done:continue
                    state.update(at=first.now(),seed=seed,case=job["case_id"]+1,policy=name,done=len(done),wall_sec=wall)
                    first.dump(output/"progress"/f"{phase}_{worker_id}.json",state)
                    row=first.run_policy(job,name,protocol["policies"][name],bundles[seed])
                    row.update(phase=phase,repeat=job["repeat"],worker=worker_id,at=first.now(),run_number=5)
                    handle.write(json.dumps(first.ready(row),separators=(",",":"),allow_nan=False)+"\n")
                    done.add(key);wall+=row["wall_sec"]
                handle.flush();os.fsync(handle.fileno());first.verify_frozen(bundles[seed],snapshots[seed])
        if done!=expected:raise AssertionError("Incomplete worker")
        state.update(status="complete",done=len(done),wall_sec=wall,at=first.now(),frozen_audit_passed=True)
        first.dump(output/"progress"/f"{phase}_{worker_id}.json",state);first.dump(directory/f"worker_{worker_id}.complete.json",state)


def run_phase(output,phase,workers=3):
    jobs=first.read(output/f"jobs_{phase}.json");expected=repeat.expected_cells(jobs)
    children=[];logs=[];start=time.monotonic()
    try:
        for i in range(workers):
            log=(output/"raw"/phase/f"worker_{i}.log").open("a",buffering=1);logs.append(log)
            children.append(subprocess.Popen([sys.executable,"-u","-m",__spec__.name,"worker","--output",str(output),"--phase",phase,"--worker",str(i)],cwd=ROOT,env=single_thread_environment(),stdout=log,stderr=subprocess.STDOUT))
        while True:
            codes=[c.poll() for c in children]
            if any(c not in (None,0) for c in codes):raise RuntimeError(f"{phase} worker stopped: {codes}")
            progress=[first.read(p) for p in (output/"progress").glob(f"{phase}_*.json")]
            done=sum(p["done"] for p in progress);elapsed=time.monotonic()-start
            first.dump(output/"status.json",{"status":"running","phase":phase,"run_number":5,"at":first.now(),"done":done,"total":len(expected),
                "elapsed_sec":elapsed,"estimated_phase_remaining_sec":elapsed*(len(expected)-done)/done if done else None,"worker_pids":[p.pid for p in children]})
            if all(c==0 for c in codes):break
            time.sleep(5)
        seen=set()
        for i in range(workers):
            marker=first.read(output/"raw"/phase/f"worker_{i}.complete.json")
            if marker["done"]!=marker["total"] or not marker["frozen_audit_passed"]:raise AssertionError("Worker audit failed")
        for p in (output/"raw"/phase).glob("worker_*.jsonl"):
            for row in first.read_records(p):
                key=first.cell_key(row)
                if key in seen:raise ValueError("Duplicate trial")
                first.audit_outcome(row["outcome"]);seen.add(key)
        if seen!=expected:raise AssertionError("Phase coverage mismatch")
        marker={"at":first.now(),"trials":len(seen),"elapsed_sec":time.monotonic()-start,
            "raw_sha256":{str(p.relative_to(output)):first.file_hash(p) for p in (output/"raw"/phase).glob("worker_*.jsonl")}}
        first.dump(output/"raw"/phase/"complete.json",marker);first.emit(output,f"Run 05 {phase} complete",trials=len(seen))
    finally:
        for child in children:
            if child.poll() is None:child.terminate()
        for log in logs:log.close()



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
