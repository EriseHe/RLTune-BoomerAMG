"""Fresh frozen complete-method comparison and the periodic/RL 2x2 cross."""

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

from experiments.archive.paper_development import run_05_policy as base
from experiments.archive.paper_development import run_05_checkpoint_training as training

import argparse
from collections import defaultdict
import copy
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import traceback

import numpy as np

from hypre.bindings import create_env, run_with_default_fallback
from hypre.bindings.config import configure_smoother_profile
from experiments.joint.solve_control.joint_online_common import report_online_outcome
from problems.registry import context_for_setup_method
from problems.streams import generate_scalar_anisotropic_diffusion_instances
from setup.space import DEFAULT_SETUP_PARAMS, SetupConfigurationSpace
from hypre.bindings import augment_setup_params
from experiments.joint.solve_control.setup_branches import build_online_linucb_branch
from experiments.joint.solve_control.native_evaluation import solve_fixed_w_case, solve_no_rl_case, solve_schedule_case

ROOT = base.ROOT
PARENT = training.OUTPUT / "training"
OUTPUT = ROOT / "results/paper_final/06_policy/20260928_frozen_five_methods"
CASES, REPEATS, PREFLIGHT_CASES = 100, 3, 2
SEEDS = dict(test=92806001, preflight=92806002, candidates=92806003,
             order=92806004, ties=92806005, bootstrap=92806006)
PRIMARY = training.METHODS
PAIRS = {m: {"source": m, "policy": p} for m, p in zip(
    PRIMARY, ("w1", "fixed140", "fixed160", "periodic", "rl"))}
PAIRS.update(periodic_setup_rl={"source": "bandit_periodic", "policy": "rl"},
             rl_setup_periodic={"source": "bandit_lstdq", "policy": "periodic"})
LABELS = dict(bandit_default="W1", bandit_fixed="Fixed 1.40", bandit_fixed_prior="Fixed 1.60",
              bandit_periodic="Periodic (2.85,1.10)", bandit_lstdq="Frozen LSTDQ",
              periodic_setup_rl="Periodic setup + frozen LSTDQ",
              rl_setup_periodic="RL setup + periodic")
METRICS = ("inclusive_total_sec", "native_total_sec", "setup_selection_sec", "setup_sec",
           "solve_sec", "controller_sec", "recovery_sec", "inclusive_continuation_sec",
           "native_continuation_sec", "wall_sec", "primary_cycles", "total_cycles")


def learning_history(model):
    return base.digest({"history": [asdict(s) for s in model.history],
                        "candidate_stats_history": model.candidate_stats_history})


class FrozenSelector:
    def __init__(self, output, source, raw):
        self.output, self.source = output, source
        checkpoint = output/"checkpoints"/source
        space = SetupConfigurationSpace.from_mapping("recommended", raw["setup"]["configuration_spaces"]["recommended"])
        self.branch, _ = build_online_linucb_branch(
            seed=raw["seeds"]["bandit"], learner_kind="linucb", tune_dim=7,
            tune7_variant="categorical", action_space_mode="full_cartesian",
            solver_tol=1e-6, solver_max_iter=50, parameter_resolution=20,
            configuration_space=space, candidate_schedule_dir=output/"candidates",
            candidate_schedule_rounds=3*(CASES+PREFLIGHT_CASES),
            candidate_schedule_seed=SEEDS["candidates"], candidate_sampling="structured512",
            context_dim=4, context_interaction_indices=(1,2,3))
        self.model = self.branch.policy.model
        self.model.load_mutable_state(checkpoint/"evaluation.npz")
        training.restore_decision_state(self.model, base.read(checkpoint/"decision.json"))
        self.rng_state = copy.deepcopy(self.model.rng.bit_generator.state)
        self.statistics = base.checkpoint_payload_hash(checkpoint/"evaluation.npz", exclude=("metadata",))
        self.history = learning_history(self.model)
        self.audit("loaded")

    def select(self, inp):
        """Evaluate a deterministic per-input candidate row without feedback."""
        tick = time.perf_counter()
        model = self.model
        old_cursor = model._candidate_schedule.cursor
        model.set_candidate_schedule_cursor(3*inp["candidate_index"])
        model.rng.bit_generator.state = np.random.default_rng(SEEDS["ties"]+inp["candidate_index"]).bit_generator.state
        try:
            context = context_for_setup_method(
                problem_kind="scalar_anisotropic_diffusion", setup_kind="linucb",
                matrix_kwargs=inp["mkw"], stream_context=np.asarray(inp["context"]),
                setup_context="diffusion3d", grid_norm_div=60)
            params, _ = self.branch.policy.select(context=context, parameter_space=self.branch.parameter_space)
            arm = int(model.history[-1].arm_index)
        finally:
            self.branch.policy.cancel_pending()
            model.set_candidate_schedule_cursor(old_cursor)
            model.rng.bit_generator.state = copy.deepcopy(self.rng_state)
        return dict(params), arm, time.perf_counter()-tick

    def audit(self, tag):
        path = self.output/"freeze_audits"/f"{self.source}_{tag}.npz"
        self.model.save_mutable_state(path)
        if base.checkpoint_payload_hash(path, exclude=("metadata",)) != self.statistics:
            raise AssertionError(f"Frozen setup statistics/RNG/cursor changed: {self.source}")
        if learning_history(self.model) != self.history:
            raise AssertionError(f"Frozen setup history changed: {self.source}")
        return {"source": self.source, "statistics_unchanged": True, "history_unchanged": True,
                "training_observations": self.model.t,
                "cache_note": "Only deterministic action-feature/local-neighbor caches may warm; no outcome enters selection."}


def load_frozen(output):
    raw = base.read(output/"training_config.json")
    args = training.runtime(raw)
    training._configure_paired_environment(args)
    selectors = {m: FrozenSelector(output,m,raw) for m in PRIMARY}
    bundle = training.build_controllers(args,args.methods).controller_bundles["bandit_lstdq"]
    bundle.load(output/"checkpoints/rl/bandit_lstdq_final.npz")
    # Initialize only episode scratch buffers, as in the established frozen runner.
    bundle.controller.start_episode(initial_environment_weight=1.)
    bundle.controller.finish_episode(learned=False)
    return selectors, bundle


def input_panel(count, seed, offset):
    return [{"case_id": i, "mkw": m, "context": c.tolist(), "input_id": base.digest(m),
             "candidate_index": offset+i} for i,(m,c) in enumerate(
                 generate_scalar_anisotropic_diffusion_instances(count=count,seed=seed,
                     grid_choices=[(60,60,60)],c_min=1,c_max=1000))]


def overlap_audit(panels):
    files = [PARENT/"inputs.json", training.OUTPUT/"functional_check/inputs.json",
             training.OUTPUT/"default_calibration/inputs.json",
             ROOT/"results/paper_final/05_policy/20260927_diffusion60_6seeds_100cases/inputs.json",
             ROOT/"results/paper_final/05_policy/20260927_run04_anchored26_joint_6seeds_100cases/inputs.json"]
    existing = set()
    def collect(obj):
        if isinstance(obj,dict):
            if "mkw" in obj:
                existing.add(base.digest(obj["mkw"]))
            for value in obj.values():
                collect(value)
        elif isinstance(obj,list):
            for value in obj: collect(value)
    for f in files: collect(base.read(f))
    ids = [r["input_id"] for panel in panels.values() for r in panel]
    if len(ids) != len(set(ids)) or existing.intersection(ids):
        raise AssertionError("Test/preflight inputs overlap each other or training/calibration/historical evaluation")
    return {"passed": True, "prior_unique_inputs_checked":len(existing),
            "new_test_inputs":CASES,"separate_preflight_inputs":PREFLIGHT_CASES,
            "source_files":{str(p.relative_to(ROOT)):base.file_hash(p) for p in files}}


def make_plan(inputs):
    rng = random.Random(SEEDS["order"])
    rows = []
    for repeat in range(REPEATS):
        cases = list(inputs)
        rng.shuffle(cases)
        for inp in cases:
            methods = list(PAIRS)
            rng.shuffle(methods)
            for position,method in enumerate(methods):
                rows.append({"trial_index":len(rows),"repeat":repeat,"case_id":inp["case_id"],
                             "method":method,"position":position})
    return rows


def prepare(output):
    if output.exists():
        raise FileExistsError("Preparation requires a new output directory")
    parent = base.read(PARENT/"frozen_artifacts.json")
    if not parent["eligible_for_module06"] or not base.read(PARENT/"complete.json")["audit_passed"]:
        raise ValueError("Module 05 checkpoints are not verified for evaluation")
    output.mkdir(parents=True)
    environment = base.environment_check(output)
    environment["parallelism"] = "one serial process, one MPI rank, one library thread"
    base.dump(output/"environment.json",environment)
    panels = {"test":input_panel(CASES,SEEDS["test"],0),
              "preflight":input_panel(PREFLIGHT_CASES,SEEDS["preflight"],CASES)}
    base.dump(output/"input_overlap_audit.json",overlap_audit(panels))
    base.dump(output/"inputs.json",panels)
    parent_hashes = {}
    def take(src,dst,expected=None):
        digest = base.file_hash(src)
        if expected is not None and digest != expected: raise AssertionError(f"Checkpoint hash changed: {src}")
        parent_hashes[str(Path(src).relative_to(ROOT))] = digest
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,dst)
    take(PARENT/"experiment_config.json",output/"training_config.json")
    take(PARENT/"frozen_artifacts.json",output/"module05_frozen_artifacts.json")
    take(PARENT/"complete.json",output/"module05_complete.json")
    for source,record in parent["methods"].items():
        folder = output/"checkpoints"/source
        take(record["statistics"],folder/"original.npz",record["statistics_sha256"])
        take(record["decision_state"],folder/"decision.json",record["decision_state_sha256"])
        with np.load(folder/"original.npz",allow_pickle=False) as saved:
            arrays = {k:saved[k].copy() for k in saved.files}
        assert int(arrays["candidate_schedule_cursor"]) == 15000
        arrays["candidate_schedule_cursor"] = np.asarray(0,dtype=np.int64)
        np.savez_compressed(folder/"evaluation.npz",**arrays)
        if base.checkpoint_payload_hash(folder/"original.npz",exclude=("candidate_schedule_cursor",)) != base.checkpoint_payload_hash(folder/"evaluation.npz",exclude=("candidate_schedule_cursor",)):
            raise AssertionError("Evaluation copy changed more than the candidate cursor")
    controller = parent["rl_controller"]
    take(controller["path"],output/"checkpoints/rl/bandit_lstdq_final.npz",controller["sha256"])
    take(Path(controller["path"]).with_suffix(".npz.encoder.json"),output/"checkpoints/rl/bandit_lstdq_final.npz.encoder.json")
    protocol = {"module":"06","created_at":base.now(),"scope":"One Module 05 training replicate; fresh frozen complete methods and a periodic/RL 2x2 cross",
        "cases":CASES,"repetitions":REPEATS,"primary_trials":CASES*REPEATS*len(PRIMARY),
        "crossed_additional_trials":CASES*REPEATS*2,"total_trials":CASES*REPEATS*len(PAIRS),
        "primary_methods":list(PRIMARY),"pairs":PAIRS,"labels":LABELS,"seeds":SEEDS,
        "profile":base.PROFILE,"tolerance":1e-6,"cap":50,"sweeps_down":1,"sweeps_up":1,
        "setup_policy":"Frozen original LinUCB selection rule, complete training decision history retained; no observations or updates",
        "candidates":"Fresh common structured512 AOT schedule; fixed-width three-row blocks, primary uses row 3*input_index; original copies retained, evaluation cursor reset from 15000 to 0",
        "selection_replays":"Select on every timed trial; same saved per-input candidate row and common per-input tie seed; cancel pending action and restore RNG/cursor; deterministic caches may warm",
        "solve_policy":"W1 and fixed wrappers unchanged; high-first (2.85,1.10) resets per primary solve; LSTDQ learn=False/explore=False with existing beta=2 LCB",
        "recovery":"Existing frozen-evaluation rule: one selected-hierarchy primary then unchanged Default-setup/W1 fallback; no learned reselection or test-feedback updates",
        "timing":"Serial native execution; randomized case order each repetition and seven-method order within each input; same selected setup per source/input across policies and repetitions",
        "primary_cost":"Setup selection + native setup/solve + solve controller + all recovery; no learning-update cost because models are frozen; matrix generation, preparation and audit I/O outside accounted cost, with method wall time also logged",
        "mechanism_cost":"Within each matched hierarchy source: solve + controller + all recovery; common initial setup/selection excluded",
        "aggregation":"Average three prescribed repetitions per input before equally weighting 100 inputs; never select fastest repetition",
        "uncertainty":"Paired bootstrap over 100 fresh input IDs, conditional on one trained checkpoint per method and measured repetitions; no claim about variation over retraining seeds",
        "preflight":"Two separate inputs, two passes of all seven pairings; excluded from reported test results",
        "parent_hashes":parent_hashes}
    base.dump(output/"protocol.json",protocol)
    selectors,bundle = load_frozen(output)
    before = base.frozen_snapshot(bundle)
    choices = {}
    for phase,inputs in panels.items():
        choices[phase] = {}
        for source,selector in selectors.items():
            choices[phase][source] = []
            for inp in inputs:
                params,arm,seconds = selector.select(inp)
                choices[phase][source].append({"case_id":inp["case_id"],"input_id":inp["input_id"],
                    "params":params,"arm_index":arm,"preparation_selection_sec":seconds,
                    "hierarchy_id":base.digest({"mkw":inp["mkw"],"params":params,"profile":base.PROFILE})})
    base.dump(output/"choices.json",choices)
    base.dump(output/"setup_preparation_audit.json",[s.audit("prepared") for s in selectors.values()])
    base.verify_frozen(bundle,before)
    base.dump(output/"trial_plan.json",make_plan(panels["test"]))
    sources = training.source_state()["files"]
    sources.update({name: base.file_hash(ROOT/name) for name in base.support_source_paths()})
    for path in [Path(__file__), Path(training.__file__), *Path(__file__).with_name("common").glob("*.py")]:
        sources[str(path.relative_to(ROOT))] = base.file_hash(path)
    base.dump(output/"source_manifest.json",sources)
    immutable = [output/n for n in ("protocol.json","inputs.json","choices.json","trial_plan.json",
                 "training_config.json","input_overlap_audit.json","source_manifest.json")]
    immutable += [p for folder in (output/"checkpoints",output/"candidates") for p in folder.rglob("*") if p.is_file()]
    base.dump(output/"prepared.json",{"at":base.now(),"files":{str(p.relative_to(output)):base.file_hash(p) for p in immutable}})
    base.emit(output,"Module 06 prepared",test_inputs=CASES,trials=protocol["total_trials"])


def verify_prepared(output):
    for name,digest in base.read(output/"prepared.json")["files"].items():
        if base.file_hash(output/name) != digest: raise AssertionError(f"Prepared file changed: {name}")
    training.unchanged(base.read(output/"source_manifest.json"))
    training.unchanged(base.read(output/"protocol.json")["parent_hashes"])


def execute_policy(inp, choice, policy, bundle):
    params,mkw = choice["params"],inp["mkw"]
    if policy == "rl":
        job = {**inp,**choice,"seed":1,"source":"frozen_module05"}
        outcome = base.run_policy(job,"rl_frozen_lcb",{"kind":"rl"},bundle)["outcome"]
        # A successful RL primary omits unused recovery fields. Normalize the
        # reporting schema without changing the measured native/controller cost.
        normalized = report_online_outcome(base.RecoveryOutcome.from_mapping(outcome).to_result(),bandit_timing={})
        for key in ("runtime","setup_runtime","solve_runtime","infer_runtime","end_to_end_runtime"):
            if not math.isclose(outcome[key],normalized[key],rel_tol=1e-10,abs_tol=1e-12):
                raise AssertionError(f"RL schema normalization changed {key}")
        return normalized
    def fallback():
        return solve_no_rl_case(params=dict(DEFAULT_SETUP_PARAMS),mkw=dict(mkw),
            solver_tol=1e-6,solver_max_iter=50,augment_params=augment_setup_params)
    def primary():
        if policy == "w1":
            return solve_no_rl_case(params=dict(params),mkw=dict(mkw),solver_tol=1e-6,
                                   solver_max_iter=50,augment_params=augment_setup_params)
        if policy in ("fixed140","fixed160"):
            return solve_fixed_w_case(params=dict(params),mkw=dict(mkw),w=1.4 if policy=="fixed140" else 1.6,
                                     sweeps_down=1,sweeps_up=1,solve_tol=1e-6,solve_max_cycles=50)
        if policy != "periodic": raise ValueError(policy)
        return solve_schedule_case(params=dict(params),mkw=dict(mkw),solve_tol=1e-6,solve_max_cycles=50,
            schedule=[(i+1,(2.85,1.1)[i%2],1,1) for i in range(50)])
    recovered = run_with_default_fallback(primary,fallback)
    outcome = report_online_outcome(recovered.to_result(),bandit_timing={})
    outcome["completed_residual_norm"] = recovered.fallback.residual_norm if recovered.fallback_used else recovered.primary.residual_norm
    return outcome


def timed_trial(trial, inp, choices, selectors, bundle):
    method = trial["method"]
    source,policy = PAIRS[method]["source"],PAIRS[method]["policy"]
    expected = choices[source][inp["case_id"]]
    start = time.perf_counter()
    params,arm,selection = selectors[source].select(inp)
    if params != expected["params"] or arm != expected["arm_index"]:
        raise AssertionError("Frozen setup choice depends on execution order or changed state")
    outcome = execute_policy(inp,expected,policy,bundle)
    wall = time.perf_counter()-start
    row = {**trial,"source":source,"policy":policy,"input_id":inp["input_id"],
           "hierarchy_id":expected["hierarchy_id"],"params":params,"arm_index":arm,
           "setup_selection_sec":selection,"outcome":outcome,"success":not outcome["unrecovered_failure"],
           "native_total_sec":outcome["runtime"],"inclusive_total_sec":outcome["end_to_end_runtime"]+selection,
           "native_continuation_sec":outcome["solve_runtime"]+outcome["fallback_setup_runtime"],
           "inclusive_continuation_sec":outcome["solve_runtime"]+outcome["fallback_setup_runtime"]+outcome["infer_runtime"],
           "setup_sec":outcome["setup_runtime"],"solve_sec":outcome["solve_runtime"],
           "controller_sec":outcome["infer_runtime"],
           "recovery_sec":sum(outcome.get(k,0) for k in ("fallback_setup_runtime","fallback_solve_runtime","fallback_controller_runtime")),
           "primary_cycles":outcome["primary_cycles"],"total_cycles":outcome["primary_cycles"]+outcome["fallback_cycles"],
           "wall_sec":wall,"at":base.now()}
    audit_row(row)
    return row


def audit_row(row):
    o = row["outcome"]
    base.reject_execution_error(o)
    base.audit_outcome(o)
    if not math.isclose(row["inclusive_total_sec"],sum(row[k] for k in ("setup_selection_sec","setup_sec","solve_sec","controller_sec")),rel_tol=1e-10,abs_tol=1e-12):
        raise AssertionError("Complete-method timing components do not add up")
    expected = {
        "native_total_sec": row["setup_sec"]+row["solve_sec"],
        "native_continuation_sec": row["solve_sec"]+o["fallback_setup_runtime"],
        "inclusive_continuation_sec": row["solve_sec"]+o["fallback_setup_runtime"]+row["controller_sec"],
        "recovery_sec": sum(o.get(k,0) for k in ("fallback_setup_runtime","fallback_solve_runtime","fallback_controller_runtime")),
        "primary_cycles": o["primary_cycles"],
        "total_cycles": o["primary_cycles"]+o["fallback_cycles"],
    }
    if any(not math.isclose(row[k],v,rel_tol=1e-10,abs_tol=1e-12) for k,v in expected.items()):
        raise AssertionError("Native, continuation or recovery accounting mismatch")
    if row["success"] != (not o["unrecovered_failure"]):
        raise AssertionError("Completion status mismatch")
    if any(not math.isfinite(row[k]) or row[k]<0 for k in METRICS):
        raise AssertionError("Invalid cost component")
    if o.get("controller_update_committed",False) or o.get("bandit_update_committed",False):
        raise AssertionError("Test observation updated a learner")
    if not 0<=o["primary_cycles"]<=50 or not 0<=o["fallback_cycles"]<=50:
        raise AssertionError("Cycle cap violated")
    actions = o.get("cycle_actions",[])
    policy = row["policy"]
    if policy in ("fixed140","fixed160") and any(w!=(1.4 if policy=="fixed140" else 1.6) for w in actions):
        raise AssertionError("Incorrect fixed policy")
    if policy=="periodic" and actions!=[(2.85,1.1)[i%2] for i in range(len(actions))]:
        raise AssertionError("Periodic phase/weight changed")
    if policy=="rl" and any(not np.any(np.isclose(w,np.linspace(1,3,41),rtol=0,atol=1e-12)) for w in actions):
        raise AssertionError("RL action outside locked weight grid")


def preflight(output,selectors,bundle):
    panels = base.read(output/"inputs.json")
    choices = base.read(output/"choices.json")["preflight"]
    before = base.frozen_snapshot(bundle)
    rows = []
    configure_smoother_profile(base.PROFILE)
    for inp in panels["preflight"]:
        for repeat in range(2):
            for method in PAIRS:
                rows.append(timed_trial({"method":method,"case_id":inp["case_id"],"repeat":repeat},inp,choices,selectors,bundle))
        for source in PRIMARY:
            with create_env(**inp["mkw"]) as env:
                env.prepare_rl(params=augment_setup_params(choices[source][inp["case_id"]]["params"]))
                if env.cycle_relax_types != (18,18,9): raise AssertionError("Smoother profile changed")
    for method,pair in PAIRS.items():
        if pair["policy"]=="rl": continue
        for inp in panels["preflight"]:
            rr=[r for r in rows if r["method"]==method and r["case_id"]==inp["case_id"]]
            if rr[0]["outcome"]["primary_cycles"] != rr[1]["outcome"]["primary_cycles"]:
                raise AssertionError("Repeated prescribed solve changed cycles")
            np.testing.assert_allclose(rr[0]["outcome"].get("cycle_residuals",[]),rr[1]["outcome"].get("cycle_residuals",[]),rtol=1e-12,atol=1e-15)
    base.verify_frozen(bundle,before)
    base.dump(output/"preflight.json",{"passed":True,"at":base.now(),"excluded_from_results":True,
        "trials":len(rows),"unrecovered_failures":sum(not r["success"] for r in rows),
        "setup_freeze_audits":[s.audit("preflight") for s in selectors.values()],"controller_unchanged":True,"rows":rows})


def case_means(rows):
    grouped = defaultdict(list)
    for row in rows: grouped[row["method"],row["case_id"]].append(row)
    result = {}
    for key,rr in grouped.items():
        if len(rr)!=REPEATS or {r["repeat"] for r in rr}!=set(range(REPEATS)):
            raise AssertionError("Missing or duplicate repetitions")
        result[key] = {k:float(np.mean([r[k] for r in rr])) for k in METRICS}
    return result


def analyze(output):
    verify_prepared(output)
    rows = list(base.read_records(output/"raw.jsonl"))
    plan = base.read(output/"trial_plan.json")
    if len(rows)!=len(plan): raise AssertionError("Incomplete evaluation")
    panels = base.read(output/"inputs.json")["test"]
    choices = base.read(output/"choices.json")["test"]
    for row,job in zip(rows,plan):
        if any(row[k]!=v for k,v in job.items()): raise AssertionError("Trial order/identity changed")
        expected = choices[PAIRS[row["method"]]["source"]][row["case_id"]]
        if any(row[k]!=expected[k] for k in ("input_id","hierarchy_id","params","arm_index")):
            raise AssertionError("Input/hierarchy matching failed")
        if row["input_id"] != panels[row["case_id"]]["input_id"]: raise AssertionError("Wrong input")
        audit_row(row)
    means = case_means(rows)
    overall = {m:{k:float(np.mean([means[m,i][k] for i in range(CASES)])) for k in METRICS} for m in PAIRS}
    rng = np.random.default_rng(SEEDS["bootstrap"])
    indices = rng.integers(0,CASES,size=(10000,CASES))
    def compare(reference,target,metric):
        a=np.array([means[reference,i][metric] for i in range(CASES)])
        b=np.array([means[target,i][metric] for i in range(CASES)])
        samples=100*(1-b[indices].mean(axis=1)/a[indices].mean(axis=1))
        return {"reference":reference,"target":target,"metric":metric,
                "reduction_pct":float(100*(1-b.sum()/a.sum())),
                "paired_input_bootstrap_95_pct":np.quantile(samples,[.025,.975]).tolist(),
                "input_wins":int((b<a).sum())}
    failure = {m:{"fallback_trials":sum(r["outcome"]["fallback_used"] for r in rows if r["method"]==m),
                  "unrecovered_trials":sum(not r["success"] for r in rows if r["method"]==m)} for m in PAIRS}
    primary_comparisons=[compare(m,"bandit_lstdq","inclusive_total_sec") for m in PRIMARY if m!="bandit_lstdq"]
    mechanism = [compare("bandit_periodic","periodic_setup_rl",metric) for metric in ("inclusive_continuation_sec","native_continuation_sec")]
    mechanism += [compare("rl_setup_periodic","bandit_lstdq",metric) for metric in ("inclusive_continuation_sec","native_continuation_sec")]
    paired_setups = {f"{a}/{b}":sum(choices[a][i]["params"]==choices[b][i]["params"] for i in range(CASES))
                    for a,b in (("bandit_periodic","bandit_lstdq"),("bandit_fixed_prior","bandit_lstdq"))}
    summary={"at":base.now(),"cases":CASES,"repetitions":REPEATS,"trials":len(rows),
             "training_replicates":1,"overall_mean_seconds":overall,"failures":failure,
             "all_test_solves_completed":not any(f["unrecovered_trials"] for f in failure.values()),
             "rl_complete_method_comparisons":primary_comparisons,"rl_within_hierarchy_comparisons":mechanism,
             "identical_setup_choices_between_sources":paired_setups,
             "uncertainty_scope":base.read(output/"protocol.json")["uncertainty"]}
    base.dump(output/"summary.json",summary)
    base.write_csv(output/"case_means.csv",[{"method":m,"case_id":i,**v} for (m,i),v in means.items()])
    lines=["# Module 06: fresh frozen-method evaluation","",
        "100 fresh diffusion 60³ inputs, three timing repetitions, one trained checkpoint per method. All learners frozen.",
        "Primary costs include setup selection, construction, solve, controller and recovery. Repetitions are averaged within each input.","",
        "| Complete frozen method | Mean total (ms) | Native setup + solve (ms) | Setup selection (ms) | Controller (ms) | Fallbacks / 300 | Unrecovered |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for m in PRIMARY:
        v=overall[m];f=failure[m]
        lines.append(f"| {LABELS[m]} | {1000*v['inclusive_total_sec']:.4f} | {1000*v['native_total_sec']:.4f} | {1000*v['setup_selection_sec']:.4f} | {1000*v['controller_sec']:.4f} | {f['fallback_trials']} | {f['unrecovered_trials']} |")
    lines += ["","## Periodic/RL crossed comparison","",
        "Mean solve + controller + recovery costs (common initial setup/selection excluded), in milliseconds.","",
        "| Frozen hierarchy source | Periodic | Frozen LSTDQ |","|---|---:|---:|"]
    for label,a,b in (("Periodic-adapted","bandit_periodic","periodic_setup_rl"),("RL-adapted","rl_setup_periodic","bandit_lstdq")):
        lines.append(f"| {label} | {1000*overall[a]['inclusive_continuation_sec']:.4f} | {1000*overall[b]['inclusive_continuation_sec']:.4f} |")
    lines += ["","## RL complete-method comparisons","",
        "Positive reductions favor RL. Intervals resample fresh input IDs, conditional on the trained checkpoints and measured timing repetitions.","",
        "| Reference | RL total-cost reduction (%) | Paired input-bootstrap 95% interval (%) |","|---|---:|---:|"]
    for c in primary_comparisons:
        lo,hi=c['paired_input_bootstrap_95_pct'];lines.append(f"| {LABELS[c['reference']]} | {c['reduction_pct']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["","These results concern one trained checkpoint set, not variation over independent retraining seeds. No coefficient, checkpoint, hierarchy rule or input subset was selected using these test outcomes.",
        "", "The original Module 04/05 checkpoints and implementation remain unchanged. See protocol.json, preflight.json, freeze_audit.json, raw.jsonl and summary.json for reproduction details.",""]
    if not summary["all_test_solves_completed"]:
        lines += ["Some test trials did not complete. Cost comparisons must be read alongside the failure counts; shorter unsuccessful solves are not performance wins.",""]
    (output/"REPORT.md").write_text("\n".join(lines))
    return summary


def run(output):
    verify_prepared(output)
    with (output/".run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (output/"complete.json").exists(): raise FileExistsError("Evaluation already completed")
        started=time.perf_counter()
        status={"status":"running","pid":os.getpid(),"started_at":base.now(),"done":0,"total":CASES*REPEATS*len(PAIRS)}
        base.dump(output/"status.json",status)
        try:
            base.environment_check(output/"run_environment")
            selectors,bundle=load_frozen(output)
            before=base.frozen_snapshot(bundle)
            preflight(output,selectors,bundle)
            inputs=base.read(output/"inputs.json")["test"]
            choices=base.read(output/"choices.json")["test"]
            plan=base.read(output/"trial_plan.json")
            raw=output/"raw.jsonl"
            rows=list(base.read_records(raw,repair_tail=True)) if raw.exists() else []
            for i,row in enumerate(rows):
                if any(row[k]!=v for k,v in plan[i].items()): raise AssertionError("Resume order mismatch")
                audit_row(row)
            with raw.open("a") as handle:
                for job in plan[len(rows):]:
                    inp=inputs[job["case_id"]]
                    row=timed_trial(job,inp,choices,selectors,bundle)
                    handle.write(json.dumps(base.ready(row),allow_nan=False)+"\n");handle.flush();rows.append(row)
                    if len(rows)%70==0 or len(rows)==len(plan):
                        base.verify_frozen(bundle,before)
                        status.update(done=len(rows),at=base.now(),elapsed_sec=time.perf_counter()-started,
                                      last_repeat=job["repeat"])
                        base.dump(output/"status.json",status)
                        print(json.dumps({"completed_trials":len(rows),"total_trials":len(plan)}),flush=True)
            freeze={"passed":True,"setup":[s.audit("final") for s in selectors.values()],"rl_unchanged":True,"at":base.now()}
            base.verify_frozen(bundle,before)
            base.dump(output/"freeze_audit.json",freeze)
            summary=analyze(output)
            base.dump(output/"complete.json",{"status":"complete","at":base.now(),"trials":len(rows),
                "passed":True,"elapsed_sec":time.perf_counter()-started,"raw_sha256":base.file_hash(raw),
                "summary_sha256":base.file_hash(output/"summary.json"),"learners_unchanged":True,
                "unrecovered_trials":sum(f["unrecovered_trials"] for f in summary["failures"].values())})
            status.update(status="completed",finished_at=base.now(),done=len(rows),elapsed_sec=time.perf_counter()-started)
            base.dump(output/"status.json",status)
        except BaseException:
            status.update(status="failed",at=base.now(),error=traceback.format_exc())
            base.dump(output/"status.json",status)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","run","analyze"))
    parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args()
    {"prepare":prepare,"run":run,"analyze":analyze}[args.action](args.output.resolve())


if __name__ == "__main__":
    main()
