"""Compare setup retry rules and default on a fresh prefix with a frozen budget."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
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
    ROOT, checkpoint_payload_hash, digest, dump, file_hash, now, read, ready, write_csv,
)
from experiments.paper_final import run_06_beta_sweep as sweep
from experiments.paper_final.recovery.retry_comparison import (
    RETRY_RULES, FailureAwareRetryLinUCB, use_retry_rule, run_deferred_case,
)
from experiments.paper_final.recovery.execution import CandidateRows
from experiments.paper_final.run_06_recovery import source_hashes, request_performance_qos
from experiments.paper_final.online.methods import build_named_setup_branches, resolve_composable_study
from experiments.paper_final.online.case_loop import report_online_outcome
from problems.registry import context_for_setup_method


BRANCHES = {**RETRY_RULES, "default": "Default setup + default solve"}
CASES, BETA = 1000, 5.0
FULL_STREAM_HASH = "f91500d5563a74995defa02a3138a852c57bcabc702de3d8a951ed40e18434a9"
SEED_CONFIGS = {
    1: ROOT / "experiments/paper_final/reproduction/online/seeds_1_to_3/advection_60.json",
    2: ROOT / "experiments/paper_final/reproduction/online/seeds_1_to_3/advection_60_s2.json",
    3: ROOT / "experiments/paper_final/reproduction/online/seeds_1_to_3/advection_60_s3.json",
    5: sweep.CONFIG,
    6: ROOT / "experiments/paper_final/reproduction/online/seeds_4_to_6/advection_60_s6.json",
    7: ROOT / "experiments/paper_final/reproduction/recovery/seeds_7_to_9/type2_seed7.json",
    8: ROOT / "experiments/paper_final/reproduction/recovery/seeds_7_to_9/type2_seed8.json",
    9: ROOT / "experiments/paper_final/reproduction/recovery/seeds_7_to_9/type2_seed9.json",
}


def retry_variant(attempts):
    if attempts not in (2, 3):
        raise ValueError("The screen supports two or three learned attempts total")
    return replace(sweep.VARIANT, attempts=attempts,
                   label=f"{attempts} learned + default; no RL recovery cost")


def feedback_for(name):
    if name not in RETRY_RULES:
        raise ValueError("Unknown retry branch")
    return "suffix_delayed" if name.endswith("_suffix") else "adjacent_delayed"


def input_rows(runtime, cases):
    return [dict(problem_index=i+1, mkw=mkw, context=context.tolist())
            for i, (mkw, context) in enumerate(runtime.stream[:cases])]


def prepare(output, *, cases=CASES, seed=5, learned_attempts=3, branches=None):
    if not 1 <= cases <= CASES:
        raise ValueError("This screen is restricted to the first 1000 setup-only inputs")
    retry_variant(learned_attempts)
    if seed not in SEED_CONFIGS:
        raise ValueError("Unsupported seed")
    selected = dict(BRANCHES) if branches is None else {n: BRANCHES[n] for n in branches}
    if "default" not in selected or len(selected) < 2:
        raise ValueError("Expected learned branches and the default baseline")
    learned = [n for n in selected if n != "default"]
    seed_source = read(SEED_CONFIGS[seed])
    output.mkdir(parents=True, exist_ok=False)
    raw = read(sweep.CONFIG)
    raw["seeds"] = copy.deepcopy(seed_source["seeds"])
    for key in ("seed_groups", "shuffle_seeds"):
        raw["stream"][key] = list(seed_source["stream"][key])
    raw["stream"]["expected_sha256"] = ""
    raw["description"] = (f"Fresh seed-{seed}, {cases}-input setup-only retry screen; "
        f"{learned_attempts} learned attempts total plus default; "
        f"Prepared branches: {', '.join(selected)}.")
    raw["solve"]["lstdq"]["beta"] = BETA
    sweep.validate_configuration(raw, BETA)
    checks, full_stream_hash = {}, None
    for name in selected:
        config = sweep.baseline_configuration(raw, "default") if name == "default" else copy.deepcopy(raw)
        config["name"] = f"type2_retry_{name}_seed{seed}_{cases}"
        config["output_dir"] = str(output / name)
        runtime = sweep.decode(config, output / name)
        if seed == 5 and runtime.manifest["sha256"] != FULL_STREAM_HASH:
            raise ValueError("The original seed-5 input stream changed")
        if full_stream_hash is None:
            full_stream_hash = runtime.manifest["sha256"]
        if runtime.manifest["sha256"] != full_stream_hash:
            raise ValueError("Full input streams are not paired")
        if name != "default" and (runtime.method.solve_activation_case != 1000
            or runtime.bundle.controller.spec.uncertainty_beta != BETA):
            raise ValueError("RL beta or setup-only boundary changed")
        config["stream"]["expected_sha256"] = full_stream_hash
        dump(output / "configs" / f"{name}.json", config)
        checks[name] = dict(config_sha256=file_hash(output / "configs" / f"{name}.json"),
                            inputs_sha256=digest(input_rows(runtime, cases)))
        if name == next(iter(selected)):
            with (output / "inputs.jsonl").open("w") as f:
                for row in input_rows(runtime, cases):
                    f.write(json.dumps(row) + "\n")
    if len({row["inputs_sha256"] for row in checks.values()}) != 1:
        raise ValueError("Prepared inputs are not paired")
    hashes = source_hashes()
    dump(output / "source_manifest.json", hashes)
    for name in hashes:
        destination = output / "source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    dump(output / "protocol.json", dict(
        cases=cases, branches=selected, family="rs_convection_3d_type2", grid=[60]*3,
        seed=seed, seeds=raw["seeds"], seed_groups=raw["stream"]["seed_groups"],
        shuffle_seeds=raw["stream"]["shuffle_seeds"],
        seed_configuration=str(SEED_CONFIGS[seed].relative_to(ROOT)),
        seed_configuration_sha256=file_hash(SEED_CONFIGS[seed]),
        rl_beta=BETA, rl_activation_case=1000, rl_steps_expected=0,
        rl_recovery_cost=False, learned_attempts=learned_attempts, default_fallback_attempts=1,
        total_attempt_budget=learned_attempts+1,
        feedback={n: feedback_for(n) for n in learned},
        delayed_feedback="All runtime targets resolve at case end. Success receives own T. Adjacent: a failed attempt receives own T plus only the next executed attempt. Full remaining: a failed attempt receives the sum from itself through the last executed attempt. Both include actual default fallback runtime when it lies in that cost sequence.",
        failure_score_update="Each learned branch counts each observation and updates precision immediately, preserving runtime means with a provisional label until case completion. Failure-model branches also update binary failure outcomes immediately. Measured cost targets replace provisional labels once, without counting observations twice.",
        historical_difference="Full remaining cost restores the archived suffix target. All four branches retain valid setup observations on fully failed cases and keep the existing delayed-cost update wiring, so this is a comparison of cost targets and retry scores, not a complete restoration of historical rollback behavior.",
        T="Own native/controller time + own selection time + previous update estimate",
        failure_feedback_retention="Keep setup observations even when the whole case fails",
        retry_scores={n: ("cost_mean + alpha(t) * uncertainty" if n.startswith("conservative_")
            else "failure_mean + 2 * uncertainty; cost LCB breaks ties") for n in learned},
        failure_model="Binary failure target, shared runtime covariance; immediate updates",
        first_attempt="Identical ordinary LinUCB cost LCB for all learned branches",
        default="One ordinary default setup/solve per input, without repeated attempts",
        prefix="Each branch learns independently from input 1; no trained prefix is reused",
        inputs="First cases inputs of a shuffled 5000-input stream using the selected seed bundle",
        candidate_pairing="Three scheduled candidate rows per input; unused rows are skipped, not executed",
        full_stream_sha256=full_stream_hash, preparation_checks=checks,
        accounting="All inputs, native attempts, fallback and measured learner work",
        native_sha256=sweep.native_hashes(), prepared_at=now(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    ))
    dump(output / "status.json", dict(status="prepared", cases=cases, native_solves_executed=0))


def execute_setup_case(runtime, index, previous_update, *, variant=sweep.VARIANT):
    from hypre.bindings import augment_setup_params
    from setup.space import DEFAULT_SETUP_PARAMS
    from experiments.paper_final.online.native_evaluation import solve_no_rl_case

    if index >= 1000:
        raise ValueError("Retry screen cannot execute the joint/RL phase")
    started = time.perf_counter()
    mkw, context = runtime.stream[index]
    model = runtime.branch.policy.model
    def native(params):
        return solve_no_rl_case(params=dict(params), mkw=dict(mkw), solver_tol=runtime.args.tol,
            solver_max_iter=runtime.args.max_cycles, augment_params=augment_setup_params)
    learner_context = context_for_setup_method(problem_kind=runtime.args.problem,
        setup_kind=runtime.method.setup_kind, setup_context=runtime.method.setup_context,
        matrix_kwargs=mkw, stream_context=context)
    kwargs = dict(policy=runtime.branch.policy,
        parameter_space=runtime.branch.parameter_space, context=learner_context,
        solver=native, fallback_solver=lambda: native(DEFAULT_SETUP_PARAMS),
        variant=variant, previous_update=previous_update,
        before_attempt=lambda attempt: runtime.rows.before_attempt(index, attempt))
    feedback = feedback_for(runtime.name)
    result = run_deferred_case(**kwargs, feedback_mode=feedback)
    runtime.rows.finish_case(index)
    outcome = report_online_outcome(result.outcome, bandit_timing=result.timing)
    outcome["method_wall_runtime"] = time.perf_counter() - started
    row = dict(problem_index=index+1, stream_index=index, method=runtime.name,
        mkw=mkw, context=context.tolist(), params=result.params, outcome=outcome,
        rl_active=False, shared_prefix=False, retry_rule=runtime.name)
    sweep.audit_case(row, BETA, variant=variant, feedback_mode=feedback)
    return row, result.update_runtime_sec


def audit_final_model(runtime, records):
    """Check every retained observation against a direct batch ridge calculation."""
    model = runtime.branch.policy.model
    features, targets, failures = [], [], []
    for row in records:
        context = context_for_setup_method(problem_kind=runtime.args.problem,
            setup_kind=runtime.method.setup_kind, setup_context=runtime.method.setup_context,
            matrix_kwargs=row["mkw"], stream_context=np.asarray(row["context"]))
        for attempt in row["outcome"]["primary_attempts"]:
            features.append(model._phi(np.asarray(context), attempt["arm_index"]))
            targets.append(attempt["learning_cost"])
            failures.append(float(attempt["status"] != "success"))
    phi = np.asarray(features)
    expected_inverse = np.linalg.inv(model.l2_reg*np.eye(model.d_phi) + phi.T @ phi)
    if model.t != len(features) or not np.allclose(model.A_inv, expected_inverse, rtol=1e-7, atol=1e-9):
        raise ValueError("Setup precision or observation count differs from executed attempts")
    if not np.allclose(model.b, phi.T @ targets, rtol=1e-10, atol=1e-8):
        raise ValueError("Runtime labels differ from retained observations")
    if isinstance(model, FailureAwareRetryLinUCB) and not np.allclose(
        model.failure_b, phi.T @ failures, rtol=1e-10, atol=1e-8
    ):
        raise ValueError("Failure model differs from immediate binary observations")
    return dict(passed=True, observations=len(features), failure_observations=int(sum(failures)),
                precision_max_abs_error=float(np.max(np.abs(model.A_inv-expected_inverse))))


def worker(output, name):
    from mpi4py import MPI
    protocol = read(output / "protocol.json")
    cases = protocol["cases"]
    variant = retry_variant(protocol["learned_attempts"])
    if name not in BRANCHES:
        raise ValueError("Unknown retry branch")
    directory = output / name
    directory.mkdir(exist_ok=False)
    if file_hash(output / "configs" / f"{name}.json") != protocol["preparation_checks"][name]["config_sha256"]:
        raise ValueError("Configuration changed")
    runtime = sweep.decode(read(output / "configs" / f"{name}.json"), directory)
    runtime.name = name
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("One MPI rank is required")
    qos = request_performance_qos()
    state = dict(pid=os.getpid(), cases=cases, inputs_sha256=digest(input_rows(runtime, cases)),
        full_stream_sha256=runtime.manifest["sha256"], native_sha256=sweep.native_hashes(),
        candidate_sha256=None, initial_bandit_sha256=None, initial_controller_sha256=None,
        bandit_observations=0, controller_steps=0, candidate_cursor=None)
    model = None
    if name != "default":
        setup = build_named_setup_branches(runtime.args, study=resolve_composable_study(runtime.args),
            warmup_instances=(), stream_hash=runtime.manifest["sha256"])
        runtime.branch = setup.branches[runtime.method.name]
        model = use_retry_rule(runtime.branch.policy.model, name)
        runtime.branch.policy.model = model
        runtime.rows = CandidateRows(model, cases=sweep.CASES, attempts=variant.attempts,
                                     directory=directory / "unused_extra_candidates")
        model.save_mutable_state(directory / "initial_bandit.npz")
        runtime.bundle.save(directory / "initial_controller.npz")
        state.update(candidate_sha256=file_hash(Path(runtime.rows.base._ids.filename)),
            initial_bandit_sha256=checkpoint_payload_hash(directory / "initial_bandit.npz",
                exclude=("metadata", "retry_failure_b")),
            initial_controller_sha256=checkpoint_payload_hash(directory / "initial_controller.npz"),
            candidate_cursor=model._candidate_schedule.cursor)
    if state["inputs_sha256"] != protocol["preparation_checks"][name]["inputs_sha256"]:
        raise ValueError("Worker inputs differ from preparation")
    if state["native_sha256"] != protocol["native_sha256"] or source_hashes() != read(output / "source_manifest.json"):
        raise ValueError("Source/native code changed")
    dump(directory / "environment.json", dict(python=sys.version, executable=sys.executable,
        qos=qos, mpi=MPI.Get_library_version(), thread_environment={k: os.environ.get(k) for k in THREAD_KEYS}))
    dump(directory / "ready.json", state)
    while not (output / "start.flag").exists():
        time.sleep(0.2)
    records, previous = [], 0.0
    started = time.perf_counter()
    with (directory / "trajectory.jsonl").open("w") as handle:
        for index in range(cases):
            if name == "default":
                row = sweep.execute_default_case(runtime, index)
            else:
                row, previous = execute_setup_case(runtime, index, previous,
                    variant=variant)
            records.append(row)
            handle.write(json.dumps(ready(row), allow_nan=False) + "\n")
            if (index+1) % 100 == 0 or index+1 == cases:
                handle.flush()
                progress = dict(branch=name, processed=index+1, elapsed_sec=time.perf_counter()-started,
                    failures=sum(r["outcome"]["failed"] for r in records),
                    fallbacks=sum(r["outcome"]["fallback_used"] for r in records))
                dump(directory / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    elapsed = time.perf_counter()-started
    model_audit = None
    if model is not None:
        model_audit = audit_final_model(runtime, records)
        model.save_mutable_state(directory / "final_bandit.npz")
        runtime.bundle.save(directory / "final_controller.npz")
        if (runtime.bundle.controller.steps or runtime.bundle.controller.episodes or
            checkpoint_payload_hash(directory / "final_controller.npz") != state["initial_controller_sha256"]):
            raise ValueError("RL learned during the setup-only screen")
        if model._candidate_schedule.cursor != cases*3:
            raise ValueError("Wrong candidate schedule boundary")
    if source_hashes() != read(output / "source_manifest.json") or sweep.native_hashes() != protocol["native_sha256"]:
        raise ValueError("Source/native code changed during the screen")
    dump(directory / "result.json", dict(complete=True, cases=cases, elapsed_sec=elapsed,
        trajectory_sha256=file_hash(directory / "trajectory.jsonl"), pairing=state,
        controller=dict(steps=0, episodes=0), model_audit=model_audit,
        bandit_observations=0 if model is None else model.t))


def verify_pairing(states, protocol):
    learned = [n for n in protocol["branches"] if n != "default"]
    if set(states) != set(protocol["branches"]) or not learned or "default" not in states:
        raise ValueError("Expected the prepared learned branches plus default")
    if protocol["feedback"] != {n: feedback_for(n) for n in learned}:
        raise ValueError("Adjacent and full remaining feedback branches are not configured separately")
    for name, state in states.items():
        if (state["cases"] != protocol["cases"] or state["controller_steps"] != 0
            or state["bandit_observations"] != 0 or state["full_stream_sha256"] != protocol["full_stream_sha256"]
            or state["native_sha256"] != protocol["native_sha256"]
            or state["inputs_sha256"] != protocol["preparation_checks"][name]["inputs_sha256"]):
            raise ValueError("Worker preparation differs from the frozen protocol")
        if name != "default" and state["candidate_cursor"] != 0:
            raise ValueError("A learned prefix was reused")
    for key in ("candidate_sha256", "initial_bandit_sha256", "initial_controller_sha256"):
        if states["default"][key] is not None or len({states[n][key] for n in learned}) != 1:
            raise ValueError(f"Initial pairing failed: {key}")


def metrics(records):
    from setup.space import DEFAULT_SETUP_PARAMS
    result = sweep.metrics(records)
    outs = [r["outcome"] for r in records]
    result.update(bandit_sec=sum(o["bandit_overhead_runtime"] for o in outs),
        second_attempts=sum(o["primary_attempt_count"] >= 2 for o in outs),
        second_successes=sum(o["primary_attempt_count"] >= 2 and o["primary_attempts"][1]["status"] == "success" for o in outs),
        third_attempts=sum(o["primary_attempt_count"] >= 3 for o in outs),
        third_successes=sum(o["primary_attempt_count"] >= 3 and o["primary_attempts"][2]["status"] == "success" for o in outs))
    result["learned_default_retries"] = sum(
        all(np.isclose(a["params"][k],v,rtol=1e-12,atol=1e-12) for k,v in DEFAULT_SETUP_PARAMS.items())
        for o in outs for a in o["primary_attempts"][1:])
    for index in range(3):
        result[f"attempt_{index+1}_native_sec"] = sum(
            o["primary_attempts"][index]["setup_runtime"] + o["primary_attempts"][index]["solve_runtime"]
            for o in outs if len(o["primary_attempts"]) > index)
    return result


def analyze(output):
    protocol = read(output / "protocol.json")
    cases = protocol["cases"]
    variant = retry_variant(protocol["learned_attempts"])
    verify_pairing({n: read(output/n/"ready.json") for n in protocol["branches"]}, protocol)
    expected_inputs = [json.loads(s) for s in (output / "inputs.jsonl").read_text().splitlines()]
    records, table = {}, []
    for name in protocol["branches"]:
        result = read(output / name / "result.json")
        path = output / name / "trajectory.jsonl"
        if not result["complete"] or result["trajectory_sha256"] != file_hash(path):
            raise ValueError("Incomplete or changed worker output")
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if len(rows) != cases or result["controller"] != dict(steps=0, episodes=0):
            raise ValueError("Wrong case count or RL activity")
        for i, row in enumerate(rows):
            if row["problem_index"] != i+1 or row["rl_active"]:
                raise ValueError("Wrong input order or RL activity")
            if {k: row[k] for k in ("problem_index", "mkw", "context")} != expected_inputs[i]:
                raise ValueError("Input pairing changed")
            if name == "default":
                sweep.audit_default_case(row)
            else:
                sweep.audit_case(row, BETA, variant=variant,
                    feedback_mode=feedback_for(name))
        expected_n = 0 if name == "default" else sum(r["outcome"]["primary_attempt_count"] for r in rows)
        if result["bandit_observations"] != expected_n:
            raise ValueError("Setup observations differ from executed attempts")
        if name != "default" and not result["model_audit"]["passed"]:
            raise ValueError("Setup ridge audit did not pass")
        records[name] = rows
        table.append(dict(branch=name, window=f"all_{cases}", **metrics(rows)))
        if cases > 100:
            table.append(dict(branch=name, window="first_100", **metrics(rows[:100])))
    dump(output / "comparison.json", dict(complete=True, audits_passed=True, protocol=protocol, rows=table))
    write_csv(output / "comparison.csv", table)
    lines = [f"Type-2 retry comparison: seed {protocol['seed']}, 60^3, {cases} inputs per branch.",
        "Fresh independent setup learners; identical inputs, initial state and candidate schedule.",
        "Fixed RL beta=5; RL remains inactive for this entire setup-only screen.",
        f"Setup feedback by branch: {protocol['feedback']}. Valid setup updates survive final failure.",
        f"{variant.attempts} learned attempts total plus one default fallback; default baseline has one ordinary solve.",
        "Totals include failures, retries, fallback and all measured learner work. All audits passed.", ""]
    lines[5:5] = [protocol["delayed_feedback"], protocol["failure_score_update"], protocol["historical_difference"]]
    for row in table:
        lines.append(f"{row['branch']} / {row['window']}: total={row['total_sec']:.6f}s, failures={row['failures']}, "
            f"fallbacks={row['fallbacks']}, second={row['second_successes']}/{row['second_attempts']} successes, "
            f"third={row['third_successes']}/{row['third_attempts']} successes, "
            f"learned retries selecting default={row['learned_default_retries']}, bandit={row['bandit_sec']:.6f}s")
    (output / "REPORT.txt").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    labels = {"conservative_suffix": "Conservative: full remaining",
              "conservative_adjacent": "Conservative: adjacent",
              "failure_model_suffix": "Failure model: full remaining",
              "failure_model_adjacent": "Failure model: adjacent", "default": "Default"}
    for name, rows in records.items():
        style = dict(color="0.4", linestyle="--") if name == "default" else {}
        for ax, field in zip(axes, ("end_to_end_runtime", "fallback_used", "failed")):
            ax.plot(np.arange(1, cases+1), np.cumsum([r["outcome"][field] for r in rows]),
                    label=labels[name], **style)
    for ax, label in zip(axes, ("Cumulative total time (s)", "Cumulative default fallbacks", "Cumulative failures")):
        ax.set(xlabel="Problem index", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=9)
    fig.suptitle(f"Type-2, seed {protocol['seed']}: {variant.attempts} learned attempts + default; setup-only retry comparison")
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


def launch(output):
    if read(output / "status.json")["status"] != "prepared":
        raise ValueError("Only an unstarted prepared screen can be launched")
    protocol = read(output / "protocol.json")
    if source_hashes() != read(output / "source_manifest.json") or sweep.native_hashes() != protocol["native_sha256"]:
        raise ValueError("Code changed after preparation")
    with (output / "manager.lock").open("x") as f:
        f.write(str(os.getpid()))
    dump(output / "manager.json", dict(pid=os.getpid(), started_at=now(), module=__name__))
    dump(output / "status.json", dict(status="starting", started_at=now()))
    processes, handles = {}, []
    guard = prevent_sleep()
    try:
        for name in protocol["branches"]:
            handle = (output / f"{name}.log").open("w")
            handles.append(handle)
            processes[name] = subprocess.Popen([sys.executable, "-u", "-m",
                "experiments.paper_final.run_06_retry_screen", "--worker", name, "--output", str(output)],
                cwd=ROOT, env=single_thread_environment(), stdout=handle, stderr=subprocess.STDOUT)
        dump(output / "workers.json", {n: p.pid for n, p in processes.items()})
        deadline = time.monotonic()+1200
        while not all((output/n/"ready.json").exists() for n in protocol["branches"]):
            if any(p.poll() is not None for p in processes.values()):
                raise RuntimeError("A worker failed during preparation; inspect branch logs")
            if time.monotonic() > deadline:
                raise TimeoutError("Worker preparation exceeded 20 minutes")
            time.sleep(0.5)
        states = {n: read(output/n/"ready.json") for n in protocol["branches"]}
        verify_pairing(states, protocol)
        dump(output / "pairing.json", dict(passed=True, workers=states))
        (output / "start.flag").write_text("All five fresh workers verified; released together.\n")
        dump(output / "status.json", dict(status="running", started_at=now(), cases=protocol["cases"]))
        while any(p.poll() is None for p in processes.values()):
            if any(p.poll() not in (None, 0) for p in processes.values()):
                raise RuntimeError("A worker failed; inspect branch logs")
            time.sleep(2)
        analyze(output)
        dump(output / "status.json", dict(status="complete", completed_at=now(), audits_passed=True))
    except BaseException as error:
        dump(output / "status.json", dict(status="failed", error=str(error), failed_at=now()))
        raise
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for handle in handles:
            handle.close()
        stop_sleep_prevention(guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=CASES, help="Only reduce for a separate preflight smoke run")
    parser.add_argument("--seed", type=int, choices=tuple(SEED_CONFIGS), default=5)
    parser.add_argument("--learned-attempts", type=int, choices=(2, 3), default=3)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--analyze", action="store_true")
    mode.add_argument("--worker", choices=tuple(BRANCHES), help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.prepare:
        prepare(output, cases=args.cases, seed=args.seed, learned_attempts=args.learned_attempts)
    elif args.run:
        launch(output)
    elif args.analyze:
        analyze(output)
    else:
        worker(output, args.worker)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128+signum))
    main()
