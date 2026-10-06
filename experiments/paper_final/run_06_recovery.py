"""Run four recovery variants concurrently on the selected Module 04 stress stream."""

from __future__ import annotations

import argparse
from dataclasses import replace
import ctypes
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

from experiments.runtime import (
    configure_single_thread, single_thread_environment,
    prevent_sleep, stop_sleep_prevention, THREAD_KEYS,
)
from experiments.paper_final.common.artifacts import (
    ROOT, dump, file_hash, read, ready, checkpoint_payload_hash,
)
from experiments.paper_final.recovery.protocol import (
    METHOD, VARIANTS, select_stress_case, study_protocol,
)


def source_files():
    return sorted(p for name in ("setup", "solve", "problems", "experiments", "hypre/bindings")
                  for p in (ROOT / name).rglob("*.py"))


def source_hashes():
    return {str(p.relative_to(ROOT)): file_hash(p) for p in source_files()}


def request_performance_qos():
    if sys.platform != "darwin":
        return {"requested": False, "reason": "non-macOS"}
    library = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
    function = library.pthread_set_qos_class_self_np
    function.argtypes = [ctypes.c_uint, ctypes.c_int]
    function.restype = ctypes.c_int
    status = int(function(0x19, 0))  # QOS_CLASS_USER_INITIATED, pthread/qos.h
    if status:
        raise OSError(status, "Cannot request user-initiated worker QoS")
    return {"requested": True, "class": "user_initiated", "hard_affinity": False}


def worker(output: Path, variant_name: str, smoke_cases: int):
    configure_single_thread()
    qos = request_performance_qos()
    import numpy as np
    from mpi4py import MPI
    from hypre.bindings import AMG_RUNTIME_LIBRARY, augment_setup_params
    from setup.space import DEFAULT_SETUP_PARAMS
    from solve.core.episode import run_td_episode
    from problems.registry import context_for_setup_method
    from experiments.paper_final.online.configuration import (
        parse_joint_experiment_config, runtime_config_from_spec,
    )
    from experiments.paper_final.online.methods import (
        resolve_composable_study, build_named_setup_branches, build_composable_solve_runtime,
    )
    from experiments.paper_final.online.execution import make_method_solver, SolveExecutionConfig
    from experiments.paper_final.online.setup_branches import run_bandit_step_test_final
    from experiments.paper_final.online.native_evaluation import solve_no_rl_case
    from experiments.paper_final.online.feedback import as_feedback
    from experiments.paper_final.online.case_loop import (
        configure_paired_environment, build_paired_instance_stream,
        report_online_outcome, method_stream_summary,
    )
    from experiments.paper_final.recovery.execution import (
        CandidateRows, EpisodeRecorder, run_unified_case,
    )

    variant = next(v for v in VARIANTS if v.name == variant_name)
    directory = output / variant.name
    directory.mkdir()
    config_path, selection = select_stress_case()
    raw = read(config_path)
    spec = parse_joint_experiment_config(raw, output_dir_override=directory)
    method = next(m for m in spec.methods if m.name == METHOD)
    if smoke_cases:
        method = replace(method, solve_activation_case=1)
    spec = replace(spec, methods=(method,))
    args = runtime_config_from_spec(spec)
    configure_paired_environment(args)
    stream, manifest = build_paired_instance_stream(args)
    if manifest["sha256"] != selection["stream_sha256"]:
        raise ValueError("Recorded stream hash changed")
    if smoke_cases:
        stream = stream[:smoke_cases]
    # Preparation retains the original 5000-case schedule even for a smoke run.
    study = resolve_composable_study(args)
    setup = build_named_setup_branches(args, study=study, warmup_instances=(),
                                       stream_hash=manifest["sha256"])
    branch = setup.branches[METHOD]
    model = branch.policy.model
    bundle = build_composable_solve_runtime(args, specs=(method,)).controller_bundles[METHOD]
    rows = CandidateRows(model, cases=args.train_cases, attempts=variant.attempts,
                         directory=directory / "extra_retry_candidates")
    initial_controller = directory / "initial_controller.npz"
    bundle.save(initial_controller)
    native_paths = [Path(AMG_RUNTIME_LIBRARY)] + sorted((ROOT / "hypre/install/lib").glob("libHYPRE.*"))
    environment = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "qos": qos,
        "thread_environment": {key: os.environ.get(key) for key in THREAD_KEYS},
        "mpi": MPI.Get_library_version(), "mpi_world_size": MPI.COMM_WORLD.Get_size(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "mpi4py")},
        "native_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in native_paths if p.is_file()},
    }
    if environment["mpi_world_size"] != 1:
        raise ValueError("Module 06 expects one MPI rank per worker")
    dump(directory / "environment.json", environment)
    dump(directory / "stream_manifest.json", manifest)
    dump(directory / "controller_protocol.json", bundle.protocol_metadata())
    dump(directory / "ready.json", {
        "pid": os.getpid(), "variant": variant.name,
        "stream_sha256": manifest["sha256"],
        "candidate_sha256": file_hash(Path(rows.base._ids.filename)),
        "initial_bandit_sha256": checkpoint_payload_hash(directory / "bandit_warmup_states" / f"{METHOD}.npz", exclude=("metadata",)),
        "initial_controller_sha256": checkpoint_payload_hash(initial_controller),
        "source_sha256": source_hashes(),
    })
    while not (output / "start.flag").exists():
        time.sleep(0.2)

    records = []
    previous_update = 0.0
    start_run = time.perf_counter()
    with (directory / "trajectory.jsonl").open("w") as trajectory:
        for index, (mkw, context) in enumerate(stream):
            start_case = time.perf_counter()
            problem_context = np.asarray(context, dtype=float)
            learner_context = context_for_setup_method(
                problem_kind=args.problem, setup_kind=method.setup_kind,
                setup_context=method.setup_context, matrix_kwargs=mkw,
                stream_context=problem_context,
            )
            def fallback():
                return solve_no_rl_case(params=dict(DEFAULT_SETUP_PARAMS), mkw=dict(mkw),
                                        solver_tol=args.tol, solver_max_iter=args.max_cycles,
                                        augment_params=augment_setup_params)

            if not variant.unified:
                solver = make_method_solver(
                    METHOD, solve=SolveExecutionConfig(args.tol, args.max_cycles),
                    mkw=mkw, case_progress=index / len(stream), case_index=index,
                    problem_context=problem_context, controller_bundles={METHOD: bundle},
                    composable_specs={METHOD: method},
                )
                params, native, timing, _, previous_update = run_bandit_step_test_final(
                    policy=branch.policy, parameter_space=branch.parameter_space,
                    problem_context=learner_context, solver_fn=solver,
                    fallback_solver_fn=lambda _: as_feedback(fallback(), include_controller=False),
                    prev_update_est=previous_update,
                )
                model.finish_candidate_schedule_case(max_selections=3)
            else:
                recorder = None
                snapshot_runtime = 0.0
                if index >= method.solve_activation_case:
                    started = time.perf_counter()
                    recorder = EpisodeRecorder(bundle.controller)
                    snapshot_runtime = time.perf_counter() - started

                def solver(params):
                    if recorder is None:
                        return solve_no_rl_case(
                            params=dict(params), mkw=dict(mkw), solver_tol=args.tol,
                            solver_max_iter=args.max_cycles, augment_params=augment_setup_params,
                        )
                    result = run_td_episode(
                        mkw=dict(mkw), params=dict(params), controller=recorder,
                        encoder=bundle.encoder, problem_context=problem_context,
                        solve_tol=args.tol, solve_max_cycles=args.max_cycles,
                        learn=True, explore=True, record_action_metadata=True,
                        fallback_attempt=None, failure_penalty_sec=0.0,
                    )
                    # The existing zero-penalty hook retains finite attempted
                    # transitions provisionally. The outer case owns rollback.
                    result["recovery_protocol_applied"] = False
                    return result

                result = run_unified_case(
                    policy=branch.policy, parameter_space=branch.parameter_space,
                    context=learner_context, solver=solver, fallback_solver=fallback,
                    variant=variant, previous_update=previous_update,
                    before_attempt=lambda attempt: rows.before_attempt(index, attempt),
                    recorder=recorder, snapshot_runtime=snapshot_runtime,
                )
                params, native, timing = result.params, result.outcome, result.timing
                previous_update = result.update_runtime_sec
                rows.finish_case(index)
            outcome = report_online_outcome(native, bandit_timing=timing)
            outcome["method_wall_runtime"] = time.perf_counter() - start_case
            record = dict(problem_index=index + 1, stream_index=index, method=variant.name,
                          params=params, mkw=mkw, context=problem_context.tolist(), outcome=outcome)
            trajectory.write(json.dumps(ready(record), allow_nan=False) + "\n")
            records.append(record)
            if (index + 1) % 100 == 0 or index + 1 == len(stream):
                trajectory.flush()
                progress = dict(processed=index + 1, total=len(stream),
                                unrecovered=sum(bool(r["outcome"]["unrecovered_failure"]) for r in records),
                                fallback=sum(bool(r["outcome"]["fallback_used"]) for r in records),
                                elapsed_sec=time.perf_counter() - start_run)
                dump(directory / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    if source_hashes() != read(output / "source_manifest.json"):
        raise RuntimeError("Source changed during the run")
    bundle.save(directory / "final_controller.npz")
    model.save_mutable_state(directory / "final_bandit.npz")
    windows = {"all": records, "last_1000": records[-1000:],
               "rl_active": records[method.solve_activation_case:]}
    dump(directory / "result.json", {
        "complete": True, "variant": variant.name,
        "stream_sha256": manifest["sha256"], "cases": len(records),
        "source_sha256": read(output / "source_manifest.json"),
        "elapsed_sec": time.perf_counter() - start_run,
        "windows": {name: method_stream_summary(data) for name, data in windows.items() if data},
        "trajectory_sha256": file_hash(directory / "trajectory.jsonl"),
        "controller": bundle.summary(),
    })


def launch(output: Path, smoke_cases: int):
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    output.mkdir(parents=True)
    config_path, _ = select_stress_case()
    shutil.copy2(config_path, output / "module04_config.json")
    protocol = study_protocol()
    protocol["smoke_cases"] = smoke_cases
    protocol["source_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dump(output / "protocol.json", protocol)
    manifest = source_hashes()
    dump(output / "source_manifest.json", manifest)
    for name in manifest:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    processes = []
    keep_awake = prevent_sleep()
    handles = []
    try:
        for variant in VARIANTS:
            handle = (output / f"{variant.name}.log").open("w")
            handles.append(handle)
            cmd = [sys.executable, "-u", "-m", "experiments.paper_final.run_06_recovery",
                   "--output", str(output), "--worker", variant.name]
            if smoke_cases:
                cmd += ["--smoke-cases", str(smoke_cases)]
            process = subprocess.Popen(cmd, cwd=ROOT, env=single_thread_environment(),
                                       stdout=handle, stderr=subprocess.STDOUT)
            processes.append(process)
        dump(output / "workers.json", {v.name: p.pid for v, p in zip(VARIANTS, processes)})
        deadline = time.monotonic() + 1200
        while not all((output / v.name / "ready.json").exists() for v in VARIANTS):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError("A worker exited during preparation; inspect its log")
            if time.monotonic() >= deadline:
                raise TimeoutError("Worker preparation exceeded 20 minutes")
            time.sleep(1)
        states = [read(output / v.name / "ready.json") for v in VARIANTS]
        for key in ("stream_sha256", "candidate_sha256", "initial_bandit_sha256", "initial_controller_sha256"):
            if len({state[key] for state in states}) != 1:
                raise ValueError(f"Worker pairing mismatch: {key}")
        if any(state["source_sha256"] != manifest for state in states):
            raise ValueError("Worker source mismatch")
        dump(output / "pairing.json", {key: states[0][key] for key in (
            "stream_sha256", "candidate_sha256", "initial_bandit_sha256", "initial_controller_sha256")})
        (output / "start.flag").write_text("All four workers prepared and paired.\n")
        print(f"Started four paired workers: {output}", flush=True)
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A worker failed; inspect its log before rerunning")
            time.sleep(2)
        if any(p.returncode for p in processes):
            raise RuntimeError("A worker failed")
        from experiments.paper_final.analyze_06_recovery import analyze
        analyze(output)
        print(f"All four variants completed: {output / 'comparison.md'}", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for handle in handles:
            handle.close()
        stop_sleep_prevention(keep_awake)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-cases", type=int, default=0,
                        help="Diagnostic prefix only; activates RL at problem 2")
    parser.add_argument("--worker", choices=[v.name for v in VARIANTS], help=argparse.SUPPRESS)
    cli = parser.parse_args()
    if cli.smoke_cases and not 2 <= cli.smoke_cases <= 5000:
        parser.error("smoke-cases must be zero or between 2 and 5000")
    configure_single_thread()
    if cli.worker:
        worker(cli.output.resolve(), cli.worker, cli.smoke_cases)
    else:
        launch(cli.output.resolve(), cli.smoke_cases)


if __name__ == "__main__":
    main()
