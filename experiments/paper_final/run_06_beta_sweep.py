"""Prepare or explicitly launch the paired seed-5 LSTDQ beta study.

Preparation constructs configurations, streams and untrained controllers only.
Execution runs one common setup prefix, then three beta workers and both controls.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.runtime import (
    configure_single_thread, single_thread_environment, prevent_sleep,
    stop_sleep_prevention, THREAD_KEYS,
)

configure_single_thread()

import numpy as np

from experiments.paper_final.common.artifacts import (
    ROOT, checkpoint_payload_hash, dump, file_hash, now, read, ready, write_csv,
)
from experiments.paper_final.online.configuration import parse_joint_experiment_config, runtime_config_from_spec
from experiments.paper_final.online.case_loop import (
    build_paired_instance_stream, configure_paired_environment, report_online_outcome,
)
from experiments.paper_final.online.methods import (
    build_composable_solve_runtime, build_named_setup_branches, resolve_composable_study,
)
from experiments.paper_final.recovery.execution import CandidateRows, EpisodeRecorder, run_unified_case
from experiments.paper_final.recovery.protocol import RecoveryVariant
from experiments.paper_final.analyze_06_recovery import audit
from experiments.paper_final.run_06_recovery import source_hashes, request_performance_qos
from problems.heterogeneous_3d import context_from_matrix_kwargs
from problems.registry import context_for_setup_method

BETAS = {"beta_0p5": 0.5, "beta_1": 1.0, "beta_2": 2.0}
BASELINES = {"default": "Default setup + default solve", "linucb": "LinUCB setup + default solve"}
METHOD = "bandit_lstdq"
CASES, PREFIX = 5000, 1000
CONFIG = ROOT / "experiments/paper_final/reproduction/recovery/type2_beta_seed5.json"
DIFFUSION_CONFIG = ROOT / "experiments/paper_final/reproduction/recovery/diffusion_beta_seed5.json"
VARIANT = RecoveryVariant("no_cost", "Three learned + default; no RL recovery cost", 3, True, False, True)
TRACE_FIELDS = (
    "cycle_explored", "cycle_epsilons", "cycle_selected_uncertainties",
    "cycle_selected_q_values", "cycle_selection_scores", "cycle_forced_default_actions",
    "td_errors", "postfit_td_errors",
)


def decode(raw, directory):
    spec = parse_joint_experiment_config(raw, output_dir_override=directory)
    args = runtime_config_from_spec(spec)
    configure_paired_environment(args)
    stream, manifest = build_paired_instance_stream(args)
    if args.expected_stream_hash and manifest["sha256"] != args.expected_stream_hash:
        raise ValueError("Generated stream differs from the prepared input hash")
    method = spec.methods[0]
    bundle = build_composable_solve_runtime(args, specs=spec.methods).controller_bundles.get(method.name)
    return SimpleNamespace(args=args, spec=spec, method=method, bundle=bundle,
                           stream=stream, manifest=manifest)


def branch_betas(values):
    values = tuple(float(value) for value in values)
    if len(values) != 3 or len(set(values)) != 3 or any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Provide three distinct, finite, nonnegative beta values")
    branches = {"beta_" + format(value, ".12g").replace(".", "p"): value for value in values}
    if len(branches) != 3:
        raise ValueError("Beta values are too close to produce distinct branch names")
    return branches


def validate_configuration(raw, beta):
    if raw["schema_version"] != 2:
        raise ValueError("The configurable sweep requires schema_version 2")
    if "lstdq_v2" in raw["solve"] or "lstdq_v3" in raw["solve"]:
        raise ValueError("Legacy beta overrides are forbidden in this sweep")
    if raw["solve"]["lstdq"]["beta"] != beta:
        raise ValueError("Configured beta does not match the branch")
    problems = {
        "rs_convection_3d_type2": dict(kind="rs_convection_3d_type2", grid=[60]*3, c_min=1e-4, c_max=1e-1),
        "scalar_anisotropic_diffusion": dict(kind="scalar_anisotropic_diffusion", grid=[60]*3,
                                           c_min=1.0, c_max=1000.0, advection=[0.0]*3),
    }
    if raw["problem"] != problems.get(raw["problem"]["kind"]):
        raise ValueError("Problem definition differs from the original seed-5 study")
    context = "rs_diffusion" if raw["problem"]["kind"] == "rs_convection_3d_type2" else "diffusion3d"
    stream = raw["stream"]
    if (stream["cases"], stream["online_cases"], stream["warmup_cases"],
        stream["cases_per_seed"], stream["group_take"]) != (5000, 5000, 0, 625, 5000):
        raise ValueError("Expected the original full-budget stream construction")
    if len(raw["methods"]) != 1 or raw["methods"][0]["solve_activation_case"] != PREFIX:
        raise ValueError("Expected one joint method activated after input 1000")
    if raw["methods"][0]["seed_offset"] != 0:
        raise ValueError("All branches must use the same seeds")
    method = raw["methods"][0]
    if (method["id"], method["setup"], method["setup_context"], method["solve_context"],
        method["candidate_sampling"]) != (METHOD, "linucb", context, context, "structured512"):
        raise ValueError("Expected the original joint method and contexts")
    if raw["solve"]["lstdq"]["lcb_lower_bound_sec"] != 0:
        raise ValueError("Expected the original zero lower bound for selection scores")
    if raw["failure_feedback"]["mode"] != "rollback_unrecovered":
        raise ValueError("Expected rollback of RL learning when the whole case fails")


def normalized_configuration(raw):
    result = copy.deepcopy(raw)
    for key in ("name", "description", "output_dir"):
        result.pop(key, None)
    result["solve"]["lstdq"].pop("beta")
    return result


def baseline_configuration(base_config, name):
    raw = copy.deepcopy(base_config)
    if name == "linucb":
        method = raw["methods"][0]
        method.update(id=name, solve="default")
        method.pop("solve_activation_case")
        method.pop("solve_context")
    elif name == "default":
        raw["methods"] = [dict(id=name, setup="default", solve="default",
                               solve_tolerance=raw["solve"]["tolerance"])]
    else:
        raise ValueError("Unknown baseline")
    return raw


def controller_metadata(bundle):
    result = bundle.protocol_metadata()
    result.update(target="Native cycle cost until the current attempt terminates",
                  rl_recovery_cost=False,
                  uncertainty_semantics="Exploration statistic; no calibrated coverage claim")
    return result


def native_hashes():
    from hypre.bindings import AMG_RUNTIME_LIBRARY
    paths = [Path(AMG_RUNTIME_LIBRARY)] + sorted((ROOT / "hypre/install/lib").glob("libHYPRE.*"))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.is_file()}


def prepare(output, *, betas=tuple(BETAS.values()), config_path=CONFIG, prefix_from=None):
    """Write a reviewed plan without building any hierarchy or executing a solve."""
    branches = branch_betas(betas)
    reference = "beta_2" if "beta_2" in branches else next(iter(branches))
    base_config = read(config_path)
    output.mkdir(parents=True, exist_ok=False)
    checks, normalized = {}, []
    for name, beta in branches.items():
        raw = copy.deepcopy(base_config)
        raw["solve"]["lstdq"]["beta"] = beta
        validate_configuration(raw, beta)
        normalized.append(normalized_configuration(raw))
        runtime = decode(raw, output / name)
        if runtime.bundle.controller.spec.uncertainty_beta != beta:
            raise ValueError("Effective controller beta differs from the requested value")
        if runtime.bundle.controller.steps or runtime.bundle.controller.episodes:
            raise ValueError("Preparation trained a controller")
        if len(runtime.stream) != CASES or runtime.method.solve_activation_case != PREFIX:
            raise ValueError("Wrong stream length or activation boundary")
        if raw["problem"]["kind"] == "rs_convection_3d_type2" and any(
            not np.array_equal(context, context_from_matrix_kwargs(mkw)) for mkw, context in runtime.stream
        ):
            raise ValueError("Context differs from actual assembly inputs")
        raw["stream"]["expected_sha256"] = runtime.manifest["sha256"]
        raw["output_dir"] = str(output / name)
        dump(output / "configs" / f"{name}.json", raw)
        initial = output / "preparation" / f"{name}_initial_controller.npz"
        runtime.bundle.save(initial)
        dump(output / "preparation" / f"{name}_controller_protocol.json", controller_metadata(runtime.bundle))
        checks[name] = dict(effective_beta=beta, stream_sha256=runtime.manifest["sha256"],
            initial_controller_sha256=checkpoint_payload_hash(initial, exclude=("spec",)),
            config_sha256=file_hash(output / "configs" / f"{name}.json"),
            steps=runtime.bundle.controller.steps, episodes=runtime.bundle.controller.episodes)
        if name == reference:
            dump(output / "stream_manifest.json", runtime.manifest)
            with (output / "inputs.jsonl").open("w") as handle:
                for index, (mkw, context) in enumerate(runtime.stream):
                    handle.write(json.dumps(dict(problem_index=index+1, mkw=mkw, context=context.tolist())) + "\n")
    if any(row != normalized[0] for row in normalized[1:]):
        raise ValueError("Configurations differ beyond beta and artifact names")
    for key in ("stream_sha256", "initial_controller_sha256"):
        if len({row[key] for row in checks.values()}) != 1:
            raise ValueError(f"Preparation pairing failed: {key}")
    for name in BASELINES:
        raw = baseline_configuration(base_config, name)
        runtime = decode(raw, output / name)
        if runtime.bundle is not None or runtime.method.solve_kind != "default":
            raise ValueError("A baseline unexpectedly created an RL controller")
        if runtime.manifest["sha256"] != checks[reference]["stream_sha256"]:
            raise ValueError("Baseline inputs differ from the beta branches")
        raw["stream"]["expected_sha256"] = runtime.manifest["sha256"]
        raw["output_dir"] = str(output / name)
        dump(output / "configs" / f"{name}.json", raw)
        checks[name] = dict(effective_beta=None, stream_sha256=runtime.manifest["sha256"],
            initial_controller_sha256=None, config_sha256=file_hash(output / "configs" / f"{name}.json"),
            steps=0, episodes=0)
    hashes = source_hashes()
    dump(output / "source_manifest.json", hashes)
    for name in hashes:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    protocol = dict(schema_version=2, status="prepared", prepared_at=now(), cases=CASES,
        prefix_cases=PREFIX, joint_cases=CASES-PREFIX, family=base_config["problem"]["kind"], grid=base_config["problem"]["grid"],
        seed=5, betas=branches, baselines=BASELINES, reference_branch=reference, seeds=raw["seeds"], variant=asdict(VARIANT),
        base_configuration=str(config_path), base_configuration_sha256=file_hash(config_path),
        learned_attempt_budget=3, default_fallback_budget=1, total_attempt_budget=4,
        rl_recovery_cost=False, failure="All three learned attempts and the default solve fail",
        failure_learning="Keep each valid setup update; roll back RL observations only when the whole case fails",
        setup_feedback="attempt_2T",
        setup_target="Immediate T on success, 2T on solver failure; T = own native + controller runtime + own selection time + previous-update estimate; no fallback or later-attempt cost",
        accounting="Charge every executed attempt, fallback and learner operation",
        shared_prefix="Execute one common 1000-input LinUCB prefix; clone complete state, RNG and candidate cursor into each beta branch and LinUCB-only control; charge the prefix once per learned branch",
        concurrency="Three beta branches and two controls, each single-threaded, released by one readiness barrier",
        stream_note="Original seed-5 full-stream recipe: eight groups of 625, then the original shuffle. This changes ordering relative to the 2000-input pilot; a fresh shared prefix is required.",
        comparison_scope="Full 5000-input default, LinUCB-only, and three joint beta branches",
        baseline_protocols={"default": "One default setup/default solve per input, as in the original baseline; no learner or repeated default attempt",
                            "linucb": "Three learned setup attempts plus one default fallback; no RL at any input"},
        native_sha256=native_hashes(), preparation_checks=checks,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    dump(output / "protocol.json", protocol)
    if prefix_from is not None:
        reuse_prefix(output, Path(prefix_from).resolve())
    dump(output / "status.json", dict(status="prepared", native_solves_executed=0,
         experiment_started=False, ready_for_explicit_run=True))
    command = f'{sys.executable} -m experiments.paper_final.run_06_beta_sweep --run --output "{output}"'
    (output / "RUN_INSTRUCTIONS.txt").write_text(
        "PREPARED ONLY: no native solves or experiment workers have run.\n\n"
        f"From {ROOT}, run only when requested:\n{command}\n\n"
        "The manager uses the verified common setup prefix, then starts all three beta workers and both controls together.\n"
        "The input stream, setup/RL seeds, candidate schedules and initial states are paired.\n"
        "The only experimental controller difference is beta; RL recovery-target additions are disabled.\n"
        "A failed learned budget receives one fresh default setup/default solve.\n"
        "Default runs one ordinary default solve per input. LinUCB-only uses the same shared recovery budget as the beta branches.\n")
    return read(output / "protocol.json")


def reuse_prefix(output, source):
    """Import a verified completed prefix after adding the missing controls."""
    protocol, original = read(output / "protocol.json"), read(source / "protocol.json")
    result = read(source / "prefix" / "result.json")
    if not result["complete"] or result["cases"] != PREFIX:
        raise ValueError("Reusable prefix must contain exactly 1000 completed inputs")
    if result["controller"]["steps"] or result["controller"]["episodes"]:
        raise ValueError("Reusable prefix contains RL learning")
    if original["variant"] != protocol["variant"] or original["native_sha256"] != protocol["native_sha256"]:
        raise ValueError("Reusable prefix used different recovery rules or native binaries")
    old_source, new_source = read(source / "source_manifest.json"), read(output / "source_manifest.json")
    changed = {p for p in set(old_source) | set(new_source) if old_source.get(p) != new_source.get(p)}
    allowed = {"experiments/paper_final/run_06_beta_sweep.py", "experiments/paper_final/test_06_beta_sweep.py"}
    if changed - allowed:
        raise ValueError("Reusable prefix used different solver or learner code")
    old_config = read(source / "configs" / f"{original['reference_branch']}.json")
    new_config = read(output / "configs" / f"{protocol['reference_branch']}.json")
    if normalized_configuration(old_config) != normalized_configuration(new_config):
        raise ValueError("Reusable prefix configuration differs")
    trajectory = source / "prefix" / "trajectory.jsonl"
    if result["trajectory_sha256"] != file_hash(trajectory):
        raise ValueError("Reusable prefix trajectory changed")
    rows = [json.loads(line) for line in trajectory.read_text().splitlines()]
    inputs = [json.loads(line) for line in (output / "inputs.jsonl").read_text().splitlines()]
    if len(rows) != PREFIX:
        raise ValueError("Reusable prefix length differs")
    for index, row in enumerate(rows):
        if row["problem_index"] != index+1 or row["rl_active"] or not row["shared_prefix"]:
            raise ValueError("Reusable prefix activation or order differs")
        if (row["mkw"], row["context"]) != (inputs[index]["mkw"], inputs[index]["context"]):
            raise ValueError("Reusable prefix input pairing differs")
        audit_case(row, original["betas"][original["reference_branch"]])
    checks = protocol["preparation_checks"][protocol["reference_branch"]]
    if any(result["pairing"][key] != checks[key] for key in ("stream_sha256", "initial_controller_sha256")):
        raise ValueError("Reusable prefix initial state differs")
    checkpoint_hash = checkpoint_payload_hash(source / "prefix" / "final_bandit.npz", exclude=("metadata",))
    if checkpoint_hash != result["pairing"]["prefix_bandit_sha256"]:
        raise ValueError("Reusable prefix checkpoint changed")
    candidate_path = source / "prefix/aot_candidate_schedules/recommended/structured512/candidate_ids.npy"
    if file_hash(candidate_path) != result["pairing"]["candidate_sha256"]:
        raise ValueError("Reusable prefix candidate schedule changed")
    shutil.copytree(source / "prefix", output / "prefix")
    dump(output / "preparation/prefix_source_manifest.json", old_source)
    protocol["reused_prefix"] = dict(source=str(source), cases=PREFIX,
        trajectory_sha256=result["trajectory_sha256"], checkpoint_sha256=checkpoint_hash,
        source_manifest_sha256=file_hash(source / "source_manifest.json"), changed_driver_files=sorted(changed),
        solver_learner_and_recovery_code_unchanged=True, verified_at=now())
    dump(output / "protocol.json", protocol)


def execute_case(runtime, index, previous_update):
    """Use the same canonical recovery engine for the prefix and joint phases."""
    from hypre.bindings import augment_setup_params
    from setup.space import DEFAULT_SETUP_PARAMS
    from solve.core.episode import run_td_episode
    from experiments.paper_final.online.native_evaluation import solve_no_rl_case

    args, method = runtime.args, runtime.method
    mkw, context = runtime.stream[index]
    active = runtime.bundle is not None and index >= method.solve_activation_case
    started = time.perf_counter()
    recorder, snapshot_time = None, 0.0
    if active:
        tick = time.perf_counter()
        recorder = EpisodeRecorder(runtime.bundle.controller)
        snapshot_time = time.perf_counter() - tick
    traces = []
    def native(params):
        return solve_no_rl_case(params=dict(params), mkw=dict(mkw), solver_tol=args.tol,
            solver_max_iter=args.max_cycles, augment_params=augment_setup_params)
    def learned(params):
        if recorder is None:
            raw = native(params)
        else:
            raw = run_td_episode(mkw=dict(mkw), params=dict(params), controller=recorder,
                encoder=runtime.bundle.encoder, problem_context=context,
                solve_tol=args.tol, solve_max_cycles=args.max_cycles,
                learn=True, explore=True, record_action_metadata=True,
                fallback_attempt=None, failure_penalty_sec=0.0)
            raw["recovery_protocol_applied"] = False
        traces.append(raw)
        return raw
    learner_context = context_for_setup_method(problem_kind=args.problem,
        setup_kind=method.setup_kind, setup_context=method.setup_context,
        matrix_kwargs=mkw, stream_context=context)
    result = run_unified_case(policy=runtime.branch.policy,
        parameter_space=runtime.branch.parameter_space, context=learner_context,
        solver=learned, fallback_solver=lambda: native(DEFAULT_SETUP_PARAMS),
        variant=VARIANT, previous_update=previous_update,
        before_attempt=lambda attempt: runtime.rows.before_attempt(index, attempt),
        recorder=recorder, snapshot_runtime=snapshot_time)
    runtime.rows.finish_case(index)
    for attempt, raw in zip(result.outcome["primary_attempts"], traces):
        attempt.update({key: raw.get(key, []) for key in TRACE_FIELDS})
    outcome = report_online_outcome(result.outcome, bandit_timing=result.timing)
    outcome["method_wall_runtime"] = time.perf_counter() - started
    row = dict(problem_index=index+1, stream_index=index, method=runtime.name,
        mkw=mkw, context=context.tolist(), params=result.params, outcome=outcome,
        rl_active=active, shared_prefix=index < PREFIX)
    audit_case(row, runtime.bundle.controller.spec.uncertainty_beta if runtime.bundle is not None else None)
    return row, result.update_runtime_sec


def execute_default_case(runtime, index):
    from experiments.paper_final.online.native_evaluation import solve_default_baseline_case
    from setup.space import DEFAULT_SETUP_PARAMS
    mkw, context = runtime.stream[index]
    started = time.perf_counter()
    out = solve_default_baseline_case(mkw=dict(mkw), solver_tol=runtime.args.tol,
                                     solver_max_iter=runtime.args.max_cycles)
    out = report_online_outcome(out, bandit_timing={})
    out["method_wall_runtime"] = time.perf_counter() - started
    row = dict(problem_index=index+1, stream_index=index, method="default", mkw=mkw,
               context=context.tolist(), params=dict(DEFAULT_SETUP_PARAMS), outcome=out,
               rl_active=False, shared_prefix=False)
    audit_default_case(row)
    return row


def audit_default_case(row):
    out = row["outcome"]
    if row["rl_active"] or out["bandit_update_committed"] or out["controller_update_committed"]:
        raise ValueError("Default baseline unexpectedly used a learner")
    if out["primary_attempt_count"] != 1 or out["fallback_used"] or len(out["primary_attempts"]) != 1:
        raise ValueError("Default baseline must execute one ordinary default solve")
    attempt = out["primary_attempts"][0]
    if bool(out["failed"]) != (attempt["status"] != "success"):
        raise ValueError("Default baseline failure classification differs")
    for actual, expected in ((out["runtime"], attempt["setup_runtime"]+attempt["solve_runtime"]),
                             (out["end_to_end_runtime"], out["runtime"])):
        if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Default baseline runtime accounting does not close")
    if out["infer_runtime"] or out["bandit_overhead_runtime"]:
        raise ValueError("Default baseline contains learner runtime")


def audit_case(row, beta, *, variant=VARIANT, feedback_mode="attempt_2T"):
    audit([dict(row, problem_index=1)], variant, feedback_mode=feedback_mode)
    out = row["outcome"]
    if out.get("replayed_transitions", 0) or out.get("target_replay_runtime", 0.0):
        raise ValueError("No-recovery-cost branch performed a target replay")
    for attempt in out["primary_attempts"]:
        if attempt["rl_recovery_cost"] not in (None, 0.0):
            raise ValueError("Recovery expense leaked into an RL target")
        if not row["rl_active"] and (attempt["rl_episode_recorded"] or attempt["cycle_actions"]):
            raise ValueError("RL ran during the setup-only prefix")
        if row["rl_active"]:
            arrays = [attempt[key] for key in (
                "cycle_actions", "cycle_selected_q_values", "cycle_selected_uncertainties",
                "cycle_selection_scores", "cycle_explored", "cycle_forced_default_actions")]
            if len({len(a) for a in arrays}) != 1:
                raise ValueError("Incomplete per-cycle beta diagnostics")
            for q, width, score in zip(*arrays[1:4]):
                if any(v is None or not math.isfinite(v) for v in (q, width, score)) or width < 0:
                    raise ValueError("Invalid per-cycle beta diagnostics")
                if not math.isclose(max(q-beta*width, 0.0), score, rel_tol=1e-8, abs_tol=1e-10):
                    raise ValueError("Recorded score does not match the branch beta")


def verify_pairing(states, prefix, betas=BETAS):
    if set(states) != set(betas) | set(BASELINES):
        raise ValueError("Expected three beta workers and both controls")
    for name, state in states.items():
        if state["effective_beta"] != betas.get(name):
            raise ValueError("Worker beta differs from its branch")
        if state["controller_steps"] != 0:
            raise ValueError("RL started before the readiness barrier")
        for key in ("stream_sha256", "native_sha256"):
            if state[key] != prefix[key]:
                raise ValueError(f"Worker pairing differs: {name}, {key}")
        if name == "default":
            if any(state.get(key) is not None for key in ("candidate_sha256", "initial_bandit_sha256",
                "initial_controller_sha256", "prefix_bandit_sha256", "candidate_cursor")):
                raise ValueError("Default baseline unexpectedly created learner state")
            continue
        for key in ("stream_sha256", "candidate_sha256", "initial_bandit_sha256",
                    "initial_controller_sha256", "prefix_bandit_sha256", "native_sha256"):
            if key == "initial_controller_sha256" and name == "linucb":
                if state[key] is not None:
                    raise ValueError("LinUCB-only baseline unexpectedly created RL state")
                continue
            if state[key] != prefix[key]:
                raise ValueError(f"Worker pairing differs: {name}, {key}")
        if state["controller_steps"] != 0 or state["candidate_cursor"] != 3*PREFIX:
            raise ValueError("Wrong learner activation boundary")


def worker(output, name):
    from mpi4py import MPI
    import os
    protocol = read(output / "protocol.json")
    if name not in {"prefix", *protocol["betas"], *protocol["baselines"]}:
        raise ValueError("Unknown prepared branch")
    directory = output / name
    directory.mkdir(exist_ok=False)
    config_name = protocol["reference_branch"] if name == "prefix" else name
    runtime = decode(read(output / "configs" / f"{config_name}.json"), directory)
    runtime.name = name
    qos = request_performance_qos()
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("Expected one MPI rank per worker")
    model = None
    state = dict(effective_beta=None, stream_sha256=runtime.manifest["sha256"],
        candidate_sha256=None, initial_bandit_sha256=None, initial_controller_sha256=None,
        native_sha256=native_hashes(), controller_steps=0, candidate_cursor=None,
        prefix_bandit_sha256=None)
    if name != "default":
        study = resolve_composable_study(runtime.args)
        setup = build_named_setup_branches(runtime.args, study=study,
            warmup_instances=(), stream_hash=runtime.manifest["sha256"])
        runtime.branch = setup.branches[runtime.method.name]
        runtime.rows = CandidateRows(runtime.branch.policy.model, cases=CASES, attempts=3,
                                     directory=directory / "unused_extra_candidates")
        model = runtime.branch.policy.model
        state.update(candidate_sha256=file_hash(Path(runtime.rows.base._ids.filename)),
            initial_bandit_sha256=checkpoint_payload_hash(directory / "bandit_warmup_states" / f"{runtime.method.name}.npz", exclude=("metadata",)))
    if runtime.bundle is not None:
        runtime.bundle.save(directory / "initial_controller.npz")
        state.update(effective_beta=runtime.bundle.controller.spec.uncertainty_beta,
            initial_controller_sha256=checkpoint_payload_hash(directory / "initial_controller.npz", exclude=("spec",)))
    prepared = protocol["preparation_checks"][config_name]
    if state["initial_controller_sha256"] != prepared["initial_controller_sha256"] or state["stream_sha256"] != prepared["stream_sha256"]:
        raise ValueError("Worker differs from the prepared configuration")
    if state["native_sha256"] != protocol["native_sha256"]:
        raise ValueError("Native libraries changed after preparation")
    dump(directory / "environment.json", dict(python=sys.version, executable=sys.executable,
        qos=qos, mpi=MPI.Get_library_version(), mpi_world_size=MPI.COMM_WORLD.Get_size(),
        thread_environment={key: os.environ.get(key) for key in THREAD_KEYS},
        packages={key: importlib.metadata.version(key) for key in ("numpy", "scipy", "mpi4py")},
        native_sha256=state["native_sha256"]))
    dump(directory / "controller_protocol.json", controller_metadata(runtime.bundle) if runtime.bundle is not None
         else dict(rl_active=False, setup=runtime.method.setup_kind, solve="default"))
    previous, records = 0.0, []
    if name not in ("prefix", "default"):
        prefix = read(output / "prefix" / "result.json")
        model.load_mutable_state(output / "prefix" / "final_bandit.npz")
        previous = prefix["previous_update"]
        model.save_mutable_state(directory / "prefix_bandit.npz")
        state.update(prefix_bandit_sha256=checkpoint_payload_hash(directory / "prefix_bandit.npz", exclude=("metadata",)),
            controller_steps=runtime.bundle.controller.steps if runtime.bundle is not None else 0,
            candidate_cursor=model._candidate_schedule.cursor)
        records = [dict(json.loads(line), method=name) for line in (output / "prefix" / "trajectory.jsonl").read_text().splitlines()]
    if name != "prefix":
        dump(directory / "ready.json", state)
        while not (output / "joint_start.flag").exists():
            time.sleep(0.2)
    started = time.perf_counter()
    with (directory / "trajectory.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(ready(row), allow_nan=False) + "\n")
        for index in range(0 if name in ("prefix", "default") else PREFIX, PREFIX if name == "prefix" else CASES):
            if name == "default":
                row = execute_default_case(runtime, index)
            else:
                row, previous = execute_case(runtime, index, previous)
            records.append(row)
            handle.write(json.dumps(ready(row), allow_nan=False) + "\n")
            if (index+1) % 100 == 0:
                handle.flush()
                progress = dict(branch=name, processed=index+1, elapsed_sec=time.perf_counter()-started,
                                failures=sum(r["outcome"]["failed"] for r in records))
                dump(directory / "progress.json", progress)
                print(json.dumps(progress), flush=True)
            if (index+1) % 500 == 0:
                if model is not None:
                    model.save_mutable_state(directory / f"bandit_{index+1}.npz")
                if runtime.bundle is not None:
                    runtime.bundle.save(directory / f"controller_{index+1}.npz")
    if source_hashes() != read(output / "source_manifest.json"):
        raise ValueError("Source changed during execution")
    if model is not None:
        model.save_mutable_state(directory / "final_bandit.npz")
        final_hash = checkpoint_payload_hash(directory / "final_bandit.npz", exclude=("metadata",))
        if name == "prefix":
            state["prefix_bandit_sha256"] = final_hash
    if runtime.bundle is not None:
        runtime.bundle.save(directory / "final_controller.npz")
    dump(directory / "result.json", dict(complete=True, cases=len(records), previous_update=previous,
        controller=runtime.bundle.summary() if runtime.bundle is not None else dict(steps=0, episodes=0),
        bandit_observations=0 if model is None else model.t,
        trajectory_sha256=file_hash(directory / "trajectory.jsonl"),
        pairing=state, elapsed_sec=time.perf_counter()-started))


def metrics(records):
    outcomes = [r["outcome"] for r in records]
    attempts = [a for o in outcomes for a in o["primary_attempts"]]
    scores = [s for a in attempts for s in a.get("cycle_selection_scores", []) if s is not None]
    actions = [w for a in attempts for w in a.get("cycle_actions", [])]
    return dict(cases=len(records), failures=sum(o["failed"] for o in outcomes),
        first_failures=sum(o["first_primary_status"] != "success" for o in outcomes),
        fallbacks=sum(o["fallback_used"] for o in outcomes),
        native_sec=sum(o["runtime"] for o in outcomes), total_sec=sum(o["end_to_end_runtime"] for o in outcomes),
        native_fallback_sec=sum(o["fallback_setup_runtime"]+o["fallback_solve_runtime"] for o in outcomes),
        cycles=sum(a["cycles"] for a in attempts)+sum(o["fallback_cycles"] for o in outcomes),
        rl_actions=len(actions), zero_selected_score_fraction=sum(s <= 0 for s in scores)/len(scores) if scores else None,
        random_actions=sum(v for a in attempts for v in a.get("cycle_explored", [])))


def analyze(output):
    protocol = read(output / "protocol.json")
    betas = protocol["betas"]
    branches = {**{name: f"LinUCB + RL beta={beta:g}" for name, beta in betas.items()}, **protocol["baselines"]}
    prefix = read(output / "prefix" / "result.json")
    verify_pairing({name: read(output / name / "ready.json") for name in branches}, prefix["pairing"], betas)
    records, table = {}, []
    for name in branches:
        beta = betas.get(name)
        path = output / name / "trajectory.jsonl"
        result = read(output / name / "result.json")
        if not result["complete"] or result["trajectory_sha256"] != file_hash(path):
            raise ValueError("Incomplete or changed worker output")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != CASES:
            raise ValueError("Incomplete 5000-input run")
        for index, row in enumerate(rows):
            if row["problem_index"] != index+1 or row["rl_active"] != (name in betas and index >= PREFIX):
                raise ValueError("Wrong problem order or activation boundary")
            if name == "default":
                audit_default_case(row)
            else:
                audit_case(row, beta)
        expected_observations = 0 if name == "default" else sum(r["outcome"]["primary_attempt_count"] for r in rows)
        if result["bandit_observations"] != expected_observations:
            raise ValueError("Setup observations differ from executed learned attempts")
        retained = [a for r in rows if r["outcome"]["controller_update_committed"] for a in r["outcome"]["primary_attempts"] if a.get("rl_episode_recorded", False)]
        if result["controller"]["steps"] != sum(len(a["cycle_actions"]) for a in retained) or result["controller"]["episodes"] != len(retained):
            raise ValueError("RL counters differ from retained observations")
        records[name] = rows
        windows = {"all_5000": (0, CASES), "setup_1000": (0, PREFIX), "joint_4000": (PREFIX, CASES), "last_1000": (4000, CASES)}
        windows.update({f"inputs_{i+1}_{i+250}": (i, i+250) for i in range(PREFIX, CASES, 250)})
        for window, (lo, hi) in windows.items():
            table.append(dict(branch=name, beta=beta, window=window, **metrics(rows[lo:hi])))
    reference = records[protocol["reference_branch"]]
    for rows in records.values():
        for i, (a, b) in enumerate(zip(rows, reference)):
            if (a["mkw"], a["context"]) != (b["mkw"], b["context"]):
                raise ValueError("Input pairing differs")
            if rows is not records["default"] and i < PREFIX and (a["params"], a["outcome"]) != (b["params"], b["outcome"]):
                raise ValueError("Shared prefix differs")
    common = [i for i in range(CASES) if all(not rows[i]["outcome"]["failed"] for rows in records.values())]
    comparisons = {}
    for name in betas:
        paired = [i for i in range(PREFIX, CASES)
                  if not records[name][i]["outcome"]["failed"] and not records["linucb"][i]["outcome"]["failed"]]
        comparisons[name] = dict(common_success_joint_cases=len(paired),
            joint_total_sec=sum(records[name][i]["outcome"]["end_to_end_runtime"] for i in paired),
            linucb_total_sec=sum(records["linucb"][i]["outcome"]["end_to_end_runtime"] for i in paired))
    comparison = dict(complete=True, protocol=protocol, audits_passed=True, rows=table,
        paired_joint_vs_linucb=comparisons,
        common_success_cases=len(common), common_success_total_sec={n:sum(rows[i]["outcome"]["end_to_end_runtime"] for i in common) for n, rows in records.items()})
    dump(output / "comparison.json", comparison)
    write_csv(output / "comparison.csv", table)
    lines = [f"{protocol['family']} beta comparison: seed 5, 60^3, 5000 inputs per branch.",
        "Setup feedback: immediate T on success, 2T on failure; valid setup updates survive final case failure.",
        "Shared 1000-input setup prefix; 4000 joint inputs; three learned attempts plus default; no RL recovery target.",
        "Controls: full default and LinUCB-only runs on the identical input stream. Default executes one ordinary solve per input; LinUCB-only uses the same three-learned-plus-default recovery rule.",
        "All incurred native and learner costs are included. All four learned branches share the exact setup prefix.", ""]
    for row in table:
        if row["window"] in ("all_5000", "joint_4000", "last_1000"):
            lines.append(f"{branches[row['branch']]}, {row['window']}: total={row['total_sec']:.3f}s, failures={row['failures']}, fallbacks={row['fallbacks']}")
    (output / "REPORT.txt").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, rows in records.items():
        style = {"linestyle": "--", "color": "black"} if name == "linucb" else {"linestyle": ":", "color": "0.5"} if name == "default" else {}
        axes[0].plot(np.arange(1, CASES+1), np.cumsum([r["outcome"]["end_to_end_runtime"] for r in rows]), label=branches[name], **style)
        axes[1].plot(np.arange(1, CASES+1), np.cumsum([r["outcome"]["failed"] for r in rows]), **style)
    for ax in axes:
        ax.axvline(PREFIX, color="0.6", linestyle=":")
        ax.set_xlabel("Problem index")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative total time (s)")
    axes[1].set_ylabel("Cumulative failures")
    axes[0].legend()
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


def launch(output, *, prefix_completed=None):
    if read(output / "status.json")["status"] != "prepared":
        raise ValueError("Only a prepared, unstarted experiment may be launched")
    protocol = read(output / "protocol.json")
    betas = protocol["betas"]
    branches = [*betas, *protocol["baselines"]]
    if source_hashes() != read(output / "source_manifest.json") or native_hashes() != protocol["native_sha256"]:
        raise ValueError("Code or native libraries changed since preparation; prepare a fresh output")
    for name in branches:
        if file_hash(output / "configs" / f"{name}.json") != protocol["preparation_checks"][name]["config_sha256"]:
            raise ValueError("Prepared configuration changed")
    processes, handles = [], []
    guard = prevent_sleep()
    def spawn(name):
        handle = (output / f"{name}.log").open("w")
        handles.append(handle)
        process = subprocess.Popen([sys.executable, "-u", "-m", "experiments.paper_final.run_06_beta_sweep",
            "--worker", name, "--output", str(output)], cwd=ROOT, env=single_thread_environment(),
            stdout=handle, stderr=subprocess.STDOUT)
        processes.append(process)
        return process
    dump(output / "status.json", dict(status="running_prefix", started_at=now()))
    try:
        if "reused_prefix" in protocol:
            prefix_result = read(output / "prefix" / "result.json")
            if prefix_result["trajectory_sha256"] != file_hash(output / "prefix" / "trajectory.jsonl"):
                raise ValueError("Imported prefix trajectory changed")
            if prefix_result["pairing"]["prefix_bandit_sha256"] != checkpoint_payload_hash(output / "prefix" / "final_bandit.npz", exclude=("metadata",)):
                raise ValueError("Imported prefix checkpoint changed")
        else:
            prefix_process = spawn("prefix")
            dump(output / "workers.json", {"prefix": prefix_process.pid})
            if prefix_process.wait() != 0:
                raise RuntimeError("Setup prefix failed; inspect prefix.log")
        if prefix_completed is not None:
            prefix_completed(output)
        children = {name: spawn(name) for name in branches}
        dump(output / "workers.json", {name: p.pid for name, p in children.items()})
        deadline = time.monotonic()+1200
        while not all((output / name / "ready.json").exists() for name in branches):
            if any(p.poll() is not None for p in children.values()):
                raise RuntimeError("Joint worker failed while preparing")
            if time.monotonic() > deadline:
                raise TimeoutError("Worker preparation exceeded 20 minutes")
            time.sleep(0.5)
        states = {name: read(output / name / "ready.json") for name in branches}
        verify_pairing(states, read(output / "prefix" / "result.json")["pairing"], betas)
        dump(output / "pairing.json", dict(passed=True, workers=states))
        (output / "joint_start.flag").write_text("All three beta branches and both controls verified; released together.\n")
        dump(output / "status.json", dict(status="running_joint", joint_started_at=now()))
        while any(p.poll() is None for p in children.values()):
            if any(p.poll() not in (None, 0) for p in children.values()):
                raise RuntimeError("A joint worker failed; inspect its log")
            time.sleep(2)
        if any(p.returncode != 0 for p in children.values()):
            raise RuntimeError("A joint worker failed")
        analyze(output)
        dump(output / "status.json", dict(status="complete", completed_at=now(), audits_passed=True))
    except BaseException as error:
        dump(output / "status.json", dict(status="failed", error=str(error), failed_at=now()))
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for handle in handles:
            handle.close()
        stop_sleep_prevention(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=CONFIG, help="Shared configuration used during preparation")
    parser.add_argument("--betas", type=float, nargs=3, default=tuple(BETAS.values()), help="Three beta values used during preparation")
    parser.add_argument("--prefix-from", type=Path, help="Reuse a verified completed setup prefix during preparation")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output, betas=args.betas, config_path=args.config.resolve(), prefix_from=args.prefix_from)
        print(f"Prepared only; no native solves executed: {output}")
    elif args.run:
        launch(output)
    elif args.analyze:
        analyze(output)
    else:
        worker(output, args.worker)


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    main()
