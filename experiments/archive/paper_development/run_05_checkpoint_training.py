"""Prepare solve-specific frozen setup selectors from one shared W1 prefix.

Local Module 05 extensions are a prescribed periodic policy and a shared
training prefix for fixed policies. Shared solver/learner code stays unchanged.
Use --prepare, then --check; --run explicitly launches the full training run.
"""
from __future__ import annotations


from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

from experiments.archive.paper_development.common.decision_state import (
    decision_state, restore_decision_state,
)

from experiments.archive.paper_development import run_05_policy as base

import argparse
import copy
import fcntl
import itertools
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import time

import numpy as np

import experiments.joint.solve_control.joint_4k_execution as execution
from experiments.joint.solve_control.composable_joint_4k import (
    SETUP_BANDIT_KINDS, build_composable_solve_runtime, build_named_setup_branches,
    resolve_composable_study, validate_setup_branch_independence,
)
from experiments.joint.solve_control.joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from experiments.joint.solve_control.joint_method_spec import ComposableMethodSpec
from experiments.joint.solve_control.joint_online_common import (
    _build_paired_instance_stream, _configure_paired_environment, _git_revision,
    _report_online_outcome,
)
from experiments.joint.solve_control.joint_protocol import build_joint_protocol
from experiments.joint.solve_control.legacy_joint_studies import (
    BATCHED_LSVI_METHOD, BEHAVIOR_MODES, DEFAULT_SETUP_METHOD,
    RECURSIVE_LSTDQ_METHOD, RECURSIVE_MC_METHOD,
)
from problems.registry import context_for_setup_method
from experiments.joint.solve_control.run_online_methods_2k import _as_feedback
from experiments.joint.solve_control.run_paper_final import source_state
from setup.learners.linucb.SharedLinUCB_AMG_v4 import SharedLinUCBv4Step
from experiments.joint.solve_control.native_evaluation import solve_schedule_case

ROOT = base.ROOT
OUTPUT = ROOT / "results/paper_final/05_online_policies/20260928_shared_prefix"
CONFIG_DIR = ROOT / "experiments/archive/paper_development/05_online_policies"
REFERENCE = (ROOT / "results/paper_final/05_policy/"
             "20260927_run04_anchored26_joint_6seeds_100cases/checkpoints/seed_1/experiment_config.json")
METHODS = ("bandit_default", "bandit_fixed", "bandit_fixed_prior", "bandit_periodic", "bandit_lstdq")
PATTERN = (2.85, 1.10)
HISTORICAL_FIXED_WEIGHT = 1.60


@dataclass(frozen=True)
class PeriodicSpec(ComposableMethodSpec):
    @property
    def family(self):
        return "periodic"

    @property
    def label(self):
        return "LinUCB + periodic (2.85, 1.10)"


def runtime(raw):
    """Validate the local prefix contract before using the general assembler."""
    raw = copy.deepcopy(raw)
    contract = raw.pop("checkpoint_training")
    prefix = int(contract["shared_prefix_cases"])
    count = int(raw["stream"]["cases"])
    if type(contract.get("functional_check")) is not bool:
        raise ValueError("functional_check must be explicit and boolean")
    if (count, prefix) != ((8,2) if contract["functional_check"] else (5000,1000)):
        raise ValueError("The training and functional-check budgets are prescribed")
    if prefix <= 0 or prefix >= count or contract["continuation_cases"] != count-prefix:
        raise ValueError("Invalid shared-prefix/continuation partition")
    if [m["id"] for m in raw["methods"]] != list(METHODS):
        raise ValueError("Exactly the five prescribed branches are required")
    if not raw["setup"].pop("shared_online_prefix", False):
        raise ValueError("This checkpoint-training design requires a shared prefix")
    if raw["stream"]["warmup_cases"] != 0 or raw["stream"]["online_cases"] != count:
        raise ValueError("The common prefix is part of the 5000 training problems")
    reference = base.read(REFERENCE)
    if raw["solve"] != reference["solve"] or raw["setup"] != reference["setup"]:
        raise ValueError("Module 05 must retain the accepted setup/solve configuration")
    if raw["problem"] != reference["problem"] or raw["failure_feedback"] != reference["failure_feedback"]:
        raise ValueError("The diffusion problem/recovery protocol changed")
    setup_keys = ("setup", "setup_space", "setup_context", "candidate_sampling", "seed_offset", "setup_warmup_cases")
    w1 = raw["methods"][0]
    if w1 != next(m for m in reference["methods"] if m["id"] == "bandit_default"):
        raise ValueError("The W1 setup learner must retain the accepted Module 04 definition")
    for m in raw["methods"]:
        if any(m.get(k) != w1.get(k) for k in setup_keys):
            raise ValueError("Branches must have matching setup initialization and candidate rules")
        if m.get("solve_activation_case", 0) != (0 if m["id"] == METHODS[0] else prefix):
            raise ValueError("All non-W1 policies must start immediately after the common prefix")
        if m.get("solve_activation") is not None:
            raise ValueError("Dynamic activation is outside this protocol")
    by_id = {m["id"]: m for m in raw["methods"]}
    for name in ("bandit_fixed", "bandit_fixed_prior"):
        if by_id[name]["solve"] != "fixed":
            raise ValueError("Both fixed branches must use constant solves")
    if by_id["bandit_fixed_prior"]["fixed_weight"] != HISTORICAL_FIXED_WEIGHT:
        raise ValueError("The historical fixed reference is locked at 1.60")
    periodic = dict(by_id["bandit_periodic"])
    if periodic.pop("periodic_weights") != list(PATTERN) or periodic.pop("phase_reset") != "each_primary_attempt":
        raise ValueError("The periodic weights and phase reset are prescribed")
    periodic.update(solve="default", solve_activation_case=0)
    validated = ComposableMethodSpec.from_mapping(periodic)
    periodic_spec = PeriodicSpec(**(asdict(validated) | {"solve_kind": "periodic", "solve_activation_case": prefix}))
    # Generic configuration validation forbids delayed fixed policies because
    # it addresses deployment. Here the delay belongs to checkpoint training.
    raw["methods"] = [m for m in raw["methods"] if m["id"] != "bandit_periodic"]
    for method in raw["methods"]:
        if method["solve"] == "fixed":
            method["solve_activation_case"] = 0
    normal = runtime_config_from_spec(parse_joint_experiment_config(raw))
    specs = {m.name: m for m in normal.methods}
    for name in ("bandit_fixed", "bandit_fixed_prior"):
        specs[name] = replace(specs[name], solve_activation_case=prefix)
    specs["bandit_periodic"] = periodic_spec
    resolved = replace(normal, methods=tuple(specs[m] for m in METHODS))
    # Validate all general constraints without invoking the legacy RL-only
    # prefix roster check. The five-branch prefix is validated above and is
    # explicitly enabled on the common execution plan below.
    resolve_composable_study(resolved)
    return resolved


def method_solver(method, **kwargs):
    spec = kwargs["composable_specs"][method]
    if spec.solve_kind != "periodic" or kwargs.get("case_index", 0) < spec.solve_activation_case:
        return execution.make_method_solver(method, **kwargs)
    solve = kwargs["solve"]
    schedule = tuple((i+1, PATTERN[i % 2], 1, 1) for i in range(solve.max_cycles))

    def primary(params):
        native = solve_schedule_case(
            params=dict(params), mkw=dict(kwargs["mkw"]), schedule=schedule,
            solve_tol=spec.resolve_solve_tolerance(solve.tolerance), solve_max_cycles=solve.max_cycles,
        )
        # Native schedule dispatch is charged. The common execution loop owns
        # setup reselection, Default/W1 fallback, and failure rollback.
        return _as_feedback(native, include_controller=True)

    return primary


def build_controllers(args, specs):
    # The prescribed periodic branch has no controller to construct. Keep it
    # out of the generic factory, whose registry covers existing solve kinds.
    return build_composable_solve_runtime(args, specs=tuple(s for s in specs if s.solve_kind != "periodic"))


def make_config(output, weight, *, check=False):
    raw = copy.deepcopy(base.read(REFERENCE))
    count, prefix = (8, 2) if check else (5000, 1000)
    seed = 92805102 if check else 92815001
    raw.update(name="module05_" + ("functional_check" if check else "checkpoint_training"),
               description="Shared W1 prefix, then solve-specific setup adaptation; checkpoint preparation for Module 06.",
               output_dir=str(output))
    raw["stream"] = dict(cases=count, warmup_cases=0, online_cases=count, instance_offset=0,
                         seed_groups=([seed] if check else [seed+6000*i for i in range(8)]),
                         shuffle_seeds=[seed+48000], cases_per_seed=count if check else 625,
                         group_take=count, expected_sha256="", smoke=check)
    raw["seeds"] = dict(base=seed, bandit=seed+60000, controller=seed+66000, method_order=seed+72000)
    w1 = next(m for m in raw["methods"] if m["id"] == "bandit_default")
    rl = next(m for m in raw["methods"] if m["id"] == "bandit_lstdq")
    rl["solve_activation_case"] = prefix
    raw["methods"] = [w1,
        dict(w1, id="bandit_fixed", solve="fixed", fixed_weight=weight, solve_activation_case=prefix),
        dict(w1, id="bandit_fixed_prior", solve="fixed", fixed_weight=HISTORICAL_FIXED_WEIGHT, solve_activation_case=prefix),
        dict(w1, id="bandit_periodic", solve="periodic", periodic_weights=list(PATTERN),
             phase_reset="each_primary_attempt", solve_activation_case=prefix), rl]
    raw["setup"]["shared_online_prefix"] = True
    raw["reporting"].update(progress_every=1 if check else 100, generate_plots=False)
    raw["checkpoint_training"] = {
        "shared_prefix_cases": prefix, "continuation_cases": count-prefix,
        "purpose": "produce comparably trained frozen models for Module 06",
        "functional_check": check,
        "development_replicates": 1, "training_speed_is_acceptance_criterion": False,
        "physical_method_problem_executions": prefix+len(METHODS)*(count-prefix),
        "logical_method_problem_exposures": len(METHODS)*count,
        "historical_fixed_weight": HISTORICAL_FIXED_WEIGHT,
    }
    _, manifest = _build_paired_instance_stream(runtime(raw))
    raw["stream"]["expected_sha256"] = manifest["sha256"]
    return raw


def unchanged(hashes):
    changed = [p for p,h in hashes.items() if not (ROOT/p).is_file() or base.file_hash(ROOT/p) != h]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")


def prepare(output, calibration):
    if (output/"prepared.json").exists():
        raise FileExistsError("Already prepared; refusing to replace a locked run")
    selection = base.read(calibration/"selection.json")
    protocol = base.read(calibration/"protocol.json")
    if (base.file_hash(calibration/"protocol.json") != selection["protocol_sha256"] or
        base.file_hash(calibration/"raw.jsonl") != selection["raw_sha256"] or
        protocol["setup_params"] != base.DEFAULT_SETUP_PARAMS):
        raise ValueError("Default-only calibration provenance mismatch")
    historical_path = output/"historical_fixed_reference.json"
    historical = base.read(historical_path)
    if historical["selected_weight"] != HISTORICAL_FIXED_WEIGHT:
        raise ValueError("Historical scan reference does not match the added branch")
    unchanged(historical["source_hashes"])
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    all_ids = {r["input_id"] for r in base.read(calibration/"inputs.json")}
    for name in ("functional_check", "training"):
        raw = make_config(output/name, selection["selected_weight"], check=name=="functional_check")
        raw["checkpoint_training"]["calibration_selection"] = str(calibration/"selection.json")
        raw["checkpoint_training"]["calibration_selection_sha256"] = base.file_hash(calibration/"selection.json")
        raw["checkpoint_training"].update(
            historical_fixed_reference=str(historical_path),
            historical_fixed_reference_sha256=base.file_hash(historical_path))
        path = CONFIG_DIR/f"{name}.json"
        if path.exists():
            raise FileExistsError(f"Refusing to replace an existing prepared configuration: {path}")
        base.dump(path, raw)
        hashes[str(path.relative_to(ROOT))] = base.file_hash(path)
        stream, _ = _build_paired_instance_stream(runtime(raw))
        ids = [base.digest(m) for m,_ in stream]
        if len(ids) != len(set(ids)) or all_ids.intersection(ids):
            raise ValueError("Calibration/check/training inputs overlap")
        all_ids.update(ids)
    protected = base.read(ROOT/"tmp/module05_plan_module04_hashes.json")
    protected.update({p:h for p,h in source_state()["files"].items()
                      if p.startswith(("hypre/", "setup/", "solve/", "problems/", "experiments/joint/"))})
    unchanged(protected)
    base.dump(output/"protected_sources.json", protected)
    adapter_paths = [Path(__file__), Path(__file__).with_name("verify_05_checkpoint_training.py"),
                     Path(__file__).with_name("test_05_checkpoint_training.py"),
                     Path(__file__).with_name("calibrate_05_default_weight.py"), CONFIG_DIR/"run.command"]
    base.dump(output/"prepared.json", {
        "at": base.now(), "git_revision": _git_revision(), "full_training_launched": False,
        "config_hashes": hashes, "calibration_selection": str(calibration/"selection.json"),
        "fixed_weight": selection["selected_weight"], "disjoint_input_count": len(all_ids),
        "historical_fixed_weight": HISTORICAL_FIXED_WEIGHT,
        "historical_fixed_reference": str(historical_path),
        "historical_fixed_reference_sha256": base.file_hash(historical_path),
        "training_primary_executions": 1000+len(METHODS)*4000, "training_exposures_per_branch": 5000,
        "purpose": "freeze setup selectors and LSTDQ controller at the prescribed endpoint",
        "adapter_hashes": {str(p.relative_to(ROOT)):base.file_hash(p) for p in adapter_paths},
    })
    print(json.dumps({"prepared": str(output), "fixed_weights": [selection["selected_weight"], HISTORICAL_FIXED_WEIGHT], "training_started": False}), flush=True)


def audit_initial_state(branches):
    validate_setup_branch_independence(branches, named_setup_spaces=True, setup_candidate_mode="aot")
    models = [branches[m].policy.model for m in METHODS]
    for a,b in itertools.combinations(models, 2):
        for name in ("A_inv", "b", "failure_A_inv", "failure_b"):
            x,y = getattr(a,name), getattr(b,name)
            if np.shares_memory(x,y) or not np.array_equal(x,y):
                raise AssertionError("Setup priors are unequal or aliased")
        for name in ("rng", "history", "_candidate_schedule", "_cand"):
            if getattr(a,name) is getattr(b,name):
                raise AssertionError(f"Shared mutable object: {name}")
        if base.digest(a.rng.bit_generator.state) != base.digest(b.rng.bit_generator.state):
            raise AssertionError("Setup initialization RNGs differ")
    return {"passed": True, "equal_priors": True, "separate_mutable_state": True,
            "immutable_AOT_candidate_files_shared": True}


def export_final_decision_states(plan):
    functional = plan.protocol["checkpoint_training"]["functional_check"]
    manifest = {"purpose": "functional-check artifacts only" if functional else "frozen Module 06 input; no test-feedback updates",
                "eligible_for_module06": not functional, "methods": {}}
    for method, branch in plan.branches.items():
        model = branch.policy.model
        state = decision_state(model)
        stats_path = plan.output_dir/"final_bandit_states"/f"{method}.npz"
        path = stats_path.with_suffix(".decision.json")
        base.dump(path, state)
        original_hash = base.checkpoint_payload_hash(stats_path, exclude=("metadata",))
        model.load_mutable_state(stats_path)
        restore_decision_state(model, base.read(path))
        if base.digest(decision_state(model)) != base.digest(state):
            raise AssertionError("Decision-state roundtrip failed")
        roundtrip = plan.output_dir/"checkpoints"/f"{method}_roundtrip.npz"
        model.save_mutable_state(roundtrip)
        if base.checkpoint_payload_hash(roundtrip, exclude=("metadata",)) != original_hash:
            raise AssertionError("Statistics checkpoint roundtrip failed")
        manifest["methods"][method] = {
            "statistics": str(stats_path), "statistics_sha256": base.file_hash(stats_path),
            "decision_state": str(path), "decision_state_sha256": base.file_hash(path),
            "training_problems": plan.online_cases, "roundtrip_passed": True,
        }
    controller = plan.checkpoints_dir/"bandit_lstdq_final.npz"
    manifest["rl_controller"] = {"path": str(controller), "sha256": base.file_hash(controller)}
    manifest["evaluation_note"] = (
        "Module 06 must supply a fresh evaluation candidate schedule under a fixed common rule; "
        "retain original checkpoints, document any cursor reset, and verify unchanged regression statistics. "
        "Use frozen LSTDQ with learn=False/explore=False and the prescribed scoring rule."
    )
    base.dump(plan.output_dir/"frozen_artifacts.json", manifest)


def execute(output, *, check=False):
    prepared = base.read(output/"prepared.json")
    unchanged(prepared["config_hashes"])
    unchanged(prepared["adapter_hashes"])
    protected = base.read(output/"protected_sources.json")
    unchanged(protected)
    name = "functional_check" if check else "training"
    if not check and not base.read(output/"functional_check/audit.json")["passed"]:
        raise ValueError("The functional check must pass before full training")
    raw = base.read(CONFIG_DIR/f"{name}.json")
    calibration = raw["checkpoint_training"]
    if base.file_hash(calibration["calibration_selection"]) != calibration["calibration_selection_sha256"]:
        raise ValueError("The locked Default calibration selection changed")
    if base.file_hash(calibration["historical_fixed_reference"]) != calibration["historical_fixed_reference_sha256"]:
        raise ValueError("The locked historical fixed-weight reference changed")
    unchanged(base.read(calibration["historical_fixed_reference"])["source_hashes"])
    args = runtime(raw)
    run_dir = args.output_dir
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    with (output/".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.perf_counter()
        env = base.environment_check(run_dir)
        env["parallelism"] = "one process, one MPI rank, one library thread; serial native solves"
        base.dump(run_dir/"environment.json", env)
        base.dump(run_dir/"experiment_config.json", raw)
        _configure_paired_environment(args)
        stream, stream_manifest = _build_paired_instance_stream(args)
        if stream_manifest["sha256"] != raw["stream"]["expected_sha256"]:
            raise ValueError("Training input stream changed")
        base.dump(run_dir/"stream_manifest.json", stream_manifest)
        base.dump(run_dir/"inputs.json", [{"mkw": m, "context": c.tolist(), "input_id": base.digest(m)} for m,c in stream])
        study = resolve_composable_study(args)
        setup = build_named_setup_branches(args, study=study, warmup_instances=(), stream_hash=stream_manifest["sha256"])
        base.dump(run_dir/"initial_state_audit.json", audit_initial_state(setup.branches))
        controllers = build_controllers(args, study.specs)
        protocol = build_joint_protocol(
            args, git_revision=_git_revision(), stream_manifest=stream_manifest, methods=study.methods,
            family_by_method=study.family_by_method, composable_specs=study.specs, candidates=(),
            setup_configuration_spaces=study.setup_configuration_spaces, bandit_methods=study.bandit_methods,
            behavior_modes=BEHAVIOR_MODES, setup_bandit_kinds=SETUP_BANDIT_KINDS,
            recursive_mc_method=RECURSIVE_MC_METHOD, recursive_lstdq_method=RECURSIVE_LSTDQ_METHOD,
            batched_lsvi_method=BATCHED_LSVI_METHOD,
            solve_controller_protocols={m:b.protocol_metadata() for m,b in controllers.controller_bundles.items()},
        )
        protocol.update(purpose=raw["description"], shared_online_prefix=True,
                        checkpoint_training=raw["checkpoint_training"])
        protocol["setup_bandit"].update(
            common_snapshot_mutable_state_cloned=True,
            shared_prefix_source="bandit_default",
            shared_prefix_cases=raw["checkpoint_training"]["shared_prefix_cases"],
            independent_updates_start_problem=raw["checkpoint_training"]["shared_prefix_cases"]+1,
        )
        protocol["checkpoint_training"].update(
            periodic_weights=list(PATTERN), phase_reset="each primary attempt",
            fallback="unchanged Default setup / W1", training_costs="diagnostic, not a victory/payback criterion",
            reporting_scope="prefix measured once; logical per-method totals each include that same shared prefix",
            source_extension="Module 05 local adapter; generic Module 04 configuration remains unchanged",
        )
        base.dump(run_dir/"config.json", protocol)
        for sub in ("trajectories", "checkpoints"):
            (run_dir/sub).mkdir()
        solve = execution.SolveExecutionConfig(args.tol, args.max_cycles)
        plan = execution.OnlineComparisonPlan(
            output_dir=run_dir, trajectories_dir=run_dir/"trajectories", checkpoints_dir=run_dir/"checkpoints",
            methods=study.methods, family_by_method=study.family_by_method, bandit_methods=study.bandit_methods,
            online_instances=stream, composable_specs=study.specs_by_name, branches=setup.branches,
            controller_bundles=controllers.controller_bundles, ppo_runner=None, protocol=protocol,
            warmup=execution.WarmupArtifacts(setup.warmup_summary, setup.warmup_trajectory, setup.warmup_state, 0),
            solve=solve, warmup_cases=0, online_cases=args.online_cases, method_order_seed=args.method_order_seed,
            progress_every=args.progress_every, aot_enabled=setup.aot_enabled,
            aot_max_selections_per_case=args.aot_max_selections_per_case,
            default_setup_method=DEFAULT_SETUP_METHOD, include_solve_screen_report=False, shared_online_prefix=True,
        )
        hooks = execution.OnlineComparisonHooks(
            method_solver=lambda method, **kw: method_solver(
                method, solve=solve, controller_bundles=controllers.controller_bundles,
                ppo_runner=None, composable_specs=study.specs_by_name, **kw),
            run_default_setup_method=execution.run_default_setup_method,
            report_online_outcome=_report_online_outcome,
            setup_context=lambda method, mkw, context: context_for_setup_method(
                problem_kind=args.problem, setup_kind=study.specs_by_name[method].setup_kind,
                matrix_kwargs=mkw, stream_context=context, grid_norm_div=args.grid_n,
                setup_context=study.specs_by_name[method].setup_context),
        )
        sources = source_state()
        sources["files"].update({name: base.file_hash(ROOT/name) for name in base.support_source_paths()})
        for path in [Path(__file__), *Path(__file__).with_name("common").glob("*.py")]:
            sources["files"][str(path.relative_to(ROOT))] = base.file_hash(path)
        base.dump(run_dir/"source_manifest.json", sources)
        preparation_sec = time.perf_counter()-started
        base.emit(run_dir, "Checkpoint training starting", cases=args.online_cases,
                  shared_prefix=raw["checkpoint_training"]["shared_prefix_cases"], functional_check=check)
        tick = time.perf_counter()
        records, steps = execution._execute_online_instances(plan, hooks)
        recovery, final_dir = execution._validate_and_save_final_state(
            plan, records=records, bandit_online_steps=steps)
        base.dump(run_dir/"result.json", {
            "bandit_online_steps": steps, "recovery_audit": recovery,
            "controllers": {m:b.summary() for m,b in controllers.controller_bundles.items()},
            "final_bandit_states": str(final_dir),
            "purpose": "checkpoint preparation; training costs are diagnostics only",
        })
        training_sec = time.perf_counter()-tick
        export_final_decision_states(plan)
        unchanged(sources["files"])
        unchanged(protected)
        from experiments.archive.paper_development.verify_05_checkpoint_training import verify
        audit = verify(run_dir)
        base.dump(run_dir/"complete.json", {"at": base.now(), "audit_passed": audit["passed"],
                  "preparation_elapsed_sec": preparation_sec, "training_and_reporting_elapsed_sec": training_sec,
                  "protected_sources_unchanged": True, "functional_check": check,
                  "trajectory_hashes": {m:base.file_hash(run_dir/"trajectories"/f"{m}.jsonl") for m in METHODS}})
        base.emit(run_dir, "Checkpoint artifacts verified", physical_executions=audit["physical_executions"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    p.add_argument("--calibration", type=Path, default=OUTPUT/"default_calibration")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--check", action="store_true")
    group.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare(args.output.resolve(), args.calibration.resolve())
    else:
        execute(args.output.resolve(), check=args.check)


if __name__ == "__main__":
    main()
