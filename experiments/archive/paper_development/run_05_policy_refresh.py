"""Module 05 Run 03: new seed-4 Joint checkpoint, original other five replicas."""
from __future__ import annotations

from experiments.runtime import configure_single_thread, single_thread_environment

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
import random
import shutil
import signal
import subprocess
import sys
import time
import numpy as np

ROOT=first.ROOT
PARENT=repeat.DEFAULT_OUTPUT
DEFAULT=ROOT/"results/paper_final/05_policy/20260927_run03_new_seed4_joint_6seeds_100cases"
NEW4=ROOT/"results/paper_final/04_online/20260927_diffusion60_repeat_seeds4to6/seed_4/diffusion_60_s4"
SOURCE=repeat.SOURCE


def verify(output):
    first.verify_inputs(output)
    prepared=first.read(output/"prepared.json")
    if "selections_sha256" in prepared and first.file_hash(output/"selections.json")!=prepared["selections_sha256"]:
        raise RuntimeError("Frozen Run 03 fixed-weight selections changed")


def seed4_jobs(output, inputs, raw, candidate_seed):
    """Use the original candidate rows and fresh setup statistics, without updates."""
    dest=output/"checkpoints/seed_4"
    copied=dest/"bandit_lstdq_setup_original.npz"
    with np.load(copied,allow_pickle=False) as state:
        normalized={k:state[k].copy() for k in state.files}
    normalized["candidate_schedule_cursor"]=np.asarray(0,dtype=np.int64)
    normalized_path=dest/"bandit_lstdq_setup_evaluation.npz"
    np.savez_compressed(normalized_path,**normalized)
    exclude=["candidate_schedule_cursor"]
    if first.checkpoint_payload_hash(copied,exclude=exclude)!=first.checkpoint_payload_hash(normalized_path,exclude=exclude):
        raise AssertionError("Setup state changed beyond the evaluation cursor")
    ordered=[(phase,row) for phase in ("development","test","preflight") for row in inputs[phase]]
    space=first.SetupConfigurationSpace.from_mapping("recommended",raw["setup"]["configuration_spaces"]["recommended"])
    branch,_=first.build_online_linucb_branch(seed=raw["seeds"]["bandit"],learner_kind="linucb",tune_dim=7,
        tune7_variant="categorical",action_space_mode="full_cartesian",solver_tol=1e-6,solver_max_iter=50,
        parameter_resolution=20,configuration_space=space,candidate_schedule_dir=output/"candidate_schedule",
        candidate_schedule_rounds=len(ordered),candidate_schedule_seed=candidate_seed,candidate_sampling="structured512",
        context_dim=4,context_interaction_indices=(1,2,3))
    model=branch.policy.model;model.load_mutable_state(normalized_path)
    ignored=["candidate_schedule_cursor","rng_state","metadata"]
    before=first.checkpoint_payload_hash(normalized_path,exclude=ignored)
    jobs={phase:[] for phase in inputs}
    for phase,row in ordered:
        context=first.context_for_setup_method(problem_kind="scalar_anisotropic_diffusion",setup_kind="linucb",
            matrix_kwargs=row["mkw"],stream_context=np.asarray(row["context"]),setup_context="diffusion3d",grid_norm_div=60)
        params,info=branch.policy.select(context=context,parameter_space=branch.parameter_space)
        job={**row,"seed":4,"source":SOURCE,"params":params,"setup_selection":info,
             "candidate_row":model._candidate_schedule.cursor-1}
        job["hierarchy_id"]=first.digest({"mkw":row["mkw"],"params":params,"profile":first.PROFILE})
        jobs[phase].append(job);branch.policy.cancel_pending()
    after=dest/"bandit_lstdq_setup_after_selection.npz";model.save_mutable_state(after)
    if before!=first.checkpoint_payload_hash(after,exclude=ignored):raise AssertionError("Setup learner updated")
    first.dump(output/"setup_freeze_audit.json",{"seed":4,"unchanged":True,"statistics_hash":before,"choices":len(ordered)})
    return jobs


def scan_jobs(base, repeated_case_ids, order_seed=92705300):
    names=[f"fixed_{1+i*.05:.2f}" for i in range(41)]
    jobs=[]
    for rep in (0,1,2):
        batch=[]
        for row in base:
            if row["seed"]!=4 or (rep and row["case_id"] not in repeated_case_ids):continue
            job=copy.deepcopy(row);job["repeat"]=rep;job["policy_order"]=list(names)
            random.Random(first.digest([order_seed,rep,row["case_id"]])).shuffle(job["policy_order"])
            batch.append(job)
        random.Random(order_seed+rep).shuffle(batch);jobs.extend(batch)
    return jobs


def prepare(output, new4=NEW4, parent=PARENT):
    if output.exists() and any(output.iterdir()):raise ValueError("New Run 03 directory required")
    repeat.verify(parent)
    if first.read(new4.parent/f"{new4.name}.complete.json")["unrecovered_failures"]:raise ValueError("New seed 4 audit failed")
    old=first.read(parent/"protocol.json");inputs=first.read(parent/"inputs.json")
    output.mkdir(parents=True)
    for name in ("analysis","progress","raw/scan","raw/test","checkpoints","provenance/source"):(output/name).mkdir(parents=True,exist_ok=True)
    for seed in range(1,7):
        dest=output/"checkpoints"/f"seed_{seed}";dest.mkdir()
        for name in ("experiment_config.json","bandit_lstdq_final.npz","bandit_lstdq_final.npz.encoder.json","bandit_lstdq_setup_original.npz"):
            if seed!=4:source=parent/"checkpoints"/f"seed_{seed}"/name
            elif name=="experiment_config.json":source=new4/name
            elif name=="bandit_lstdq_setup_original.npz":source=new4/"final_bandit_states/bandit_lstdq.npz"
            else:source=new4/"checkpoints"/name
            shutil.copy2(source,dest/name)
        if seed!=4:
            for name in ("bandit_lstdq_final.npz","bandit_lstdq_setup_original.npz"):
                assert first.file_hash(dest/name)==first.file_hash(parent/"checkpoints"/f"seed_{seed}"/name)
    new_jobs=seed4_jobs(output,inputs,first.read(new4/"experiment_config.json"),old["seeds"]["candidates"])
    base=[{k:v for k,v in j.items() if k not in ("repeat","policy_order","method_policies")}
          for j in first.read(parent/"jobs_test.json") if j["repeat"]==0 and j["seed"]!=4]+new_jobs["test"]
    preflight=[j for j in first.read(parent/"jobs_preflight.json") if j["seed"]!=4]+new_jobs["preflight"]
    parent1=Path(old["origin_run"])
    repeated=first.read(parent1/"protocol.json")["test_repeat_case_ids"]
    scan=scan_jobs(base,repeated)
    origin_jobs={(j["seed"],j["case_id"]):j for j in first.read(parent/"jobs_test.json") if j["repeat"]==0}
    for job in base:
        original=origin_jobs[job["seed"],job["case_id"]]
        if job["mkw"]!=original["mkw"]:raise AssertionError("Test inputs changed")
        if job["seed"]!=4 and job["params"]!=original["params"]:raise AssertionError("Untargeted hierarchy changed")
    protocol=copy.deepcopy(old)
    protocol.update(run_number=3,created_at=first.now(),origin_run=str(parent),comparison_run=2,
        new_seed4_directory=str(new4),new_seed4_components=["setup selector","RL controller"],
        unchanged_checkpoint_seeds=[1,2,3,5,6],order_seed=92705301,
        scope="Timing-sensitivity follow-up with new seed 4 Joint state and original other five checkpoints",
        scan_extra_repeat_case_ids=repeated,scan_execution_count=len(repeat.expected_cells(scan)),
        selection_rule="Seed 4 fixed weights selected by new exhaustive 41-grid test scan, including original prescribed repetitions; other seeds reuse Run 01 choices",
        oracle_rule="Successful best-observed grid selections, then independently retimed three times; no continuous optimum claim",
        periodic_rule="Retain the previously development-tuned (2.5,1) candidate and periodic (1,3); no new schedule tuning",
        test_schedule_rule="Exactly the previously tuned (2.5,1) and prescribed (1,3) schedules; no (3,1) or prefix-tail",
        development_cases=0,development_inputs_use="100 original development contexts only advance the shared candidate schedule; no development solves",
        execution_count=None,policies={k:v for k,v in first.policy_library().items()
            if v["kind"] in ("fixed","rl") or k in ("periodic_2.50_1.00","periodic_1.00_3.00")},
        changed_test_hierarchies_seed4=sum(j["params"]!=origin_jobs[4,j["case_id"]]["params"] for j in new_jobs["test"]),
        interpretation="Diagnostic repeat after inspecting seed variation. Preserve original six-seed evidence; no additional independent seed is created.")
    first.dump(output/"protocol.json",protocol);shutil.copy2(parent/"inputs.json",output/"inputs.json")
    for phase,jobs in (("base_test",base),("preflight",preflight),("scan",scan)):first.dump(output/f"jobs_{phase}.json",jobs)
    shutil.copy2(parent/"selections.json",output/"previous_selections.json")
    manifest=first.read(parent/"source_manifest.json")
    for name in ("run_05_policy_refresh.py","analyze_05_policy_refresh.py","plot_05_policy_refresh.py","test_05_policy_refresh.py"):
        relative="experiments/archive/paper_development/"+name;manifest[relative]=first.file_hash(ROOT/relative)
    manifest.update({name: first.file_hash(ROOT/name) for name in first.support_source_paths()})
    for relative in manifest:
        dest=output/"provenance/source"/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/relative,dest)
    first.dump(output/"source_manifest.json",manifest)
    first.dump(output/"prepared.json",{"at":first.now(),"protocol_sha256":first.file_hash(output/"protocol.json"),
        "inputs_sha256":first.file_hash(output/"inputs.json"),
        "jobs_sha256":{phase:first.file_hash(output/f"jobs_{phase}.json") for phase in ("base_test","preflight","scan")},
        "checkpoint_sha256":{str(p.relative_to(output)):first.file_hash(p) for p in (output/"checkpoints").rglob("*") if p.is_file()}})
    first.environment_check(output)
    first.emit(output,"Run 03 prepared",new_seed4_hierarchies=protocol["changed_test_hierarchies_seed4"],scan_trials=protocol["scan_execution_count"])


def preflight(output):
    verify(output);protocol=first.read(output/"protocol.json");jobs=first.read(output/"jobs_preflight.json")
    checks=[]
    for seed in range(1,7):
        bundle=first.load_bundle(output,seed);before=first.frozen_snapshot(bundle)
        job=next(j for j in jobs if j["seed"]==seed and j["case_id"]==0)
        names=["fixed_1.00","periodic_2.50_1.00","periodic_1.00_3.00","rl_frozen_lcb","fixed_1.00"]
        results=[first.run_policy(job,name,protocol["policies"][name],bundle) for name in names]
        np.testing.assert_allclose(results[0]["outcome"]["cycle_residuals"],results[-1]["outcome"]["cycle_residuals"],rtol=1e-12,atol=1e-15)
        first.verify_frozen(bundle,before);checks.append({"seed":seed,"reference_repeat_agrees":True,"frozen":True})
    first.dump(output/"preflight.json",{"at":first.now(),"passed":True,"checks":checks})


def worker(output, phase, worker_id, workers):
    run_frozen_worker(output, phase, worker_id, workers, run_number=3, verify=verify, policy=first)


def run_phase(output, phase, workers=3):
    run_frozen_phase(output, phase, runner_module=__spec__.name, run_number=3, workers=workers)


def choose_fixed(output):
    from experiments.archive.paper_development.module05_figures.data import select_oracles
    rows=[];marker=first.read(output/"raw/scan/complete.json")
    for name,digest in marker["raw_sha256"].items():
        if first.file_hash(output/name)!=digest:raise ValueError("Scan records changed")
        rows.extend(first.compact_row(r) for r in first.read_records(output/name))
    averaged=first.average_repetitions(rows)
    cells={(r["seed"],r["source"],r["case_id"],r["policy"]):{"native":r["native_continuation_sec"],"success":r["success"]} for r in averaged}
    weights=[f"fixed_{1+i*.05:.2f}" for i in range(41)]
    fixed,individual=select_oracles(cells,4,SOURCE,list(range(100)),weights)
    selections=first.read(output/"previous_selections.json")
    selections.update(run_number=3,origin_run=str(PARENT),seed4_scan_run=str(output),selection_frozen_before_run02=False,selection_frozen_before_final_retiming=True,
        seed4_selection={"global_fixed":fixed,"per_instance_fixed":individual,"scan_raw_sha256":marker["raw_sha256"]},
        note="New seed 4 fixed weights from the successful 41-grid scan; other five seeds retain Run 01 choices. All selected policies retimed independently.")
    selections["chosen_policies"][f"4/{SOURCE}/fixed"]=fixed
    selections["chosen_policies"][f"4/{SOURCE}/oracle"]={str(k):v for k,v in individual.items()}
    jobs=repeat.plan_jobs(first.read(output/"jobs_base_test.json"),selections["chosen_policies"],order_seed=92705301)
    first.dump(output/"selections.json",selections);first.dump(output/"jobs_test.json",jobs)
    prepared=first.read(output/"prepared.json");first.dump(output/"prepared_scan.json",prepared)
    prepared["jobs_sha256"]["test"]=first.file_hash(output/"jobs_test.json")
    prepared["selections_sha256"]=first.file_hash(output/"selections.json");first.dump(output/"prepared.json",prepared)
    first.dump(output/"final_evaluation_plan.json",{"at":first.now(),"trials":len(repeat.expected_cells(jobs)),"global_fixed_seed4":fixed})
    first.emit(output,"Seed 4 fixed choices frozen before fresh timing",global_fixed=fixed,trials=len(repeat.expected_cells(jobs)))


def run(output,new4):
    if not (output/"prepared.json").exists():prepare(output,new4)
    with (output/"supervisor.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.monotonic()
        try:
            verify(output);first.environment_check(output)
            if not (output/"preflight.json").exists():preflight(output)
            if not (output/"raw/scan/complete.json").exists():run_phase(output,"scan")
            if not (output/"selections.json").exists():choose_fixed(output)
            if not (output/"raw/test/complete.json").exists():run_phase(output,"test")
            from experiments.archive.paper_development.analyze_05_policy_refresh import analyze
            first.dump(output/"status.json",{"status":"running","phase":"analysis","run_number":3,"at":first.now()})
            analyze(output);verify(output)
            # Numerical completion is distinct from the all-artifacts completion marker.
            first.dump(output/"numerical_complete.json",{"status":"complete","at":first.now()})
            from experiments.archive.paper_development.plot_05_policy_refresh import generate
            first.dump(output/"status.json",{"status":"running","phase":"figures","run_number":3,"at":first.now()})
            generate(output);verify(output)
            final={"status":"complete","phase":"complete","run_number":3,"at":first.now(),"elapsed_sec":time.monotonic()-start}
            first.dump(output/"complete.json",final);first.dump(output/"status.json",final)
        except BaseException as exc:
            first.dump(output/"status.json",{"status":"failed","phase":"stopped","at":first.now(),"error":str(exc)})
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run","worker"))
    parser.add_argument("--output",type=Path,default=DEFAULT)
    parser.add_argument("--new-seed4",type=Path,default=NEW4)
    parser.add_argument("--phase",choices=("scan","test"));parser.add_argument("--worker",type=int)
    args=parser.parse_args()
    if args.command=="worker":worker(args.output,args.phase,args.worker,3)
    else:run(args.output,args.new_seed4)
