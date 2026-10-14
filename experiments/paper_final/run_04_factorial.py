"""Configurable Module 04 four-branch suite, one single-thread worker per seed.

Prepare freezes configs, paired inputs, source and native-library hashes.
Run refuses partial state and releases workers only after all are ready.
All four branches are measured independently from input one. Both setup learners
and both LSTDQ controllers start cold; no prefix, copied solves or state forks.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

from experiments.runtime import (configure_single_thread, single_thread_environment,
    prevent_sleep, stop_sleep_prevention, THREAD_KEYS)
configure_single_thread()
import numpy as np

from experiments.paper_final.common.artifacts import (
    ROOT, dump, read, digest, file_hash, ready, now, write_csv, read_records,
)
from experiments.paper_final.common.frozen_inputs import execution_source_paths
from experiments.paper_final.online.configuration import parse_joint_experiment_config, runtime_config_from_spec
from experiments.paper_final.online.case_loop import build_paired_instance_stream, configure_paired_environment
from experiments.paper_final.online.methods import (
    build_named_setup_branches, build_composable_solve_runtime, resolve_composable_study,
    validate_setup_branch_independence,
)
from experiments.paper_final.online import factorial
from experiments.paper_final.recovery.execution import CandidateRows
from experiments.paper_final.recovery.retry_comparison import use_retry_rule
from experiments.paper_final import run_06_beta_sweep as sweep
from experiments.paper_final import run_06_retry_screen as screen
from experiments.paper_final.run_06_retry_repeatability import observed_model
from experiments.paper_final.run_06_final_validation import audit_controller_counts
from setup.space import DEFAULT_SETUP_PARAMS


def source_hashes():
    return {name: file_hash(ROOT/name) for name in sorted(execution_source_paths())}


def decode(raw, directory):
    spec = parse_joint_experiment_config(raw, output_dir_override=directory)
    factorial.validate_methods(spec)
    args = runtime_config_from_spec(spec)
    configure_paired_environment(args)
    stream, manifest = build_paired_instance_stream(args)
    if not args.expected_stream_hash or args.expected_stream_hash != manifest["sha256"]:
        raise ValueError("A pinned, matching input stream hash is required")
    return SimpleNamespace(spec=spec, args=args, stream=stream, manifest=manifest)


def input_rows(runtime, cases):
    return [dict(problem_index=i+1, mkw=mkw, context=context.tolist())
            for i, (mkw, context) in enumerate(runtime.stream[:cases])]


def prepare(suite_path, output, *, cases=None):
    suite = read(suite_path)
    if set(suite) != {"name", "seeds"} or not suite["seeds"]:
        raise ValueError("Suite needs a name and nonempty seeds [{label, config}]")
    labels = [str(item["label"]) for item in suite["seeds"]]
    if len(set(labels)) != len(labels) or any(not label or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in label) for label in labels):
        raise ValueError("Seed labels must be unique safe directory names")
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(name=suite["name"], prepared_at=now(), methods=factorial.METHODS,
        seeds=labels, recovery="failure_model_adjacent", feedback_mode="adjacent_delayed",
        learned_attempts=3, fixed_rl_attempts=1, default_fallback_attempts=1,
        default_attempts=1, rl_recovery_cost=False, rl_whole_case_failure_rollback=True,
        setup_failure_observation_retention=True, failure_score_beta=2.0,
        shared_prefix=False, copied_measurements=False, exact_default=DEFAULT_SETUP_PARAMS,
        learning_start="Both setup learners and both LSTDQ controllers start cold at input 1",
        concurrency="One single-thread worker per seed; four methods serially in balanced permutation blocks",
        native_sha256=sweep.native_hashes(),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        stages={})
    for item, label in zip(suite["seeds"], labels):
        config_path = Path(item["config"])
        if not config_path.is_absolute():
            config_path = suite_path.parent / config_path
        raw = copy.deepcopy(read(config_path))
        stage = output/f"seed_{label}"
        stage.mkdir()
        count = int(raw["stream"]["online_cases"] if cases is None else cases)
        if count < 1 or count > int(raw["stream"]["online_cases"]):
            raise ValueError("Case count exceeds the declared stream")
        raw["output_dir"] = str(stage)
        runtime = decode(raw, stage)
        joint_method = next(m for m in runtime.spec.methods if m.name == "bandit_lstdq")
        if runtime.args.warmup_cases or runtime.args.aot_max_selections_per_case != 3:
            raise ValueError("Expected cold starts and three candidate rows per input")
        # Explicit grids protect final experiments from inherited shell settings.
        for space in runtime.spec.setup.configuration_spaces.values():
            if set(space.numeric_grids) != {"strong_threshold", "max_row_sum", "trunc_factor", "P_max_elmts", "agg_num_levels"}:
                raise ValueError("The new runner requires all five numerical grids explicitly")
        dump(stage/"experiment_config.json", raw)
        dump(stage/"stream_manifest.json", runtime.manifest)
        rows = input_rows(runtime, count)
        (stage/"inputs.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        dump(stage/"status.json", dict(status="prepared", processed=0, cases=count))
        protocol["stages"][label] = dict(cases=count, smoke=count != int(raw["stream"]["online_cases"]),
            joint_activation_case=joint_method.solve_activation_case,
            config_sha256=file_hash(stage/"experiment_config.json"),
            source_config=str(config_path.resolve()), source_config_sha256=file_hash(config_path),
            inputs_sha256=digest(rows), full_stream_sha256=runtime.manifest["sha256"])
    hashes = source_hashes()
    dump(output/"source_manifest.json", hashes)
    for name in hashes:
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, destination)
    dump(output/"protocol.json", protocol)
    dump(output/"status.json", dict(status="prepared", native_solves_executed=0))
    return protocol


def check_frozen(output):
    if source_hashes() != read(output/"source_manifest.json"):
        raise ValueError("Source changed since preparation")
    if sweep.native_hashes() != read(output/"protocol.json")["native_sha256"]:
        raise ValueError("Native libraries changed since preparation")


def initialize(output, label):
    protocol = read(output/"protocol.json")
    stage = output/f"seed_{label}"
    if read(stage/"status.json")["status"] != "prepared":
        raise ValueError("Refusing to resume partial learned state; use a fresh prepared output")
    checks = protocol["stages"][label]
    if file_hash(stage/"experiment_config.json") != checks["config_sha256"]:
        raise ValueError("Prepared configuration changed")
    root = decode(read(stage/"experiment_config.json"), stage)
    if digest(input_rows(root, checks["cases"])) != checks["inputs_sha256"]:
        raise ValueError("Input pairing changed")
    study = resolve_composable_study(root.args)
    setup = build_named_setup_branches(root.args, study=study, warmup_instances=(),
                                       stream_hash=root.manifest["sha256"])
    validate_setup_branch_independence(setup.branches, named_setup_spaces=True, setup_candidate_mode="aot")
    solve = build_composable_solve_runtime(root.args, specs=root.spec.methods)
    runtimes = {}
    for method in root.spec.methods:
        runtime = SimpleNamespace(**vars(root), name=method.name, method=method,
                                  bundle=solve.controller_bundles.get(method.name))
        if method.name in setup.branches:
            runtime.branch = setup.branches[method.name]
            model = observed_model(use_retry_rule(runtime.branch.policy.model, "failure_model_adjacent"))
            runtime.branch.policy.model = model
            runtime.rows = CandidateRows(model, cases=len(root.stream), attempts=3, directory=stage/"unused")
            destination = stage/"initial_bandit_states"/f"{method.name}.npz"
            destination.parent.mkdir(exist_ok=True)
            model.save_mutable_state(destination)
        if runtime.bundle is not None:
            destination = stage/"initial_controllers"/f"{method.name}.npz"
            destination.parent.mkdir(exist_ok=True)
            runtime.bundle.save(destination)
        runtimes[method.name] = runtime
    features = {name: factorial.audit_features(runtimes[name]) for name in factorial.METHODS[2:]}
    independence = factorial.audit_independence(runtimes)
    candidate_hashes = {name: file_hash(Path(runtimes[name].rows.base._ids.filename))
                        for name in factorial.METHODS[2:]}
    if len(set(candidate_hashes.values())) != 1:
        raise ValueError("Paired setup candidate schedules differ")
    dump(stage/"initial_audit.json", dict(features=features, independence=independence,
         candidates=candidate_hashes, inputs_sha256=checks["inputs_sha256"], passed=True))
    return runtimes


def save_checkpoints(stage, runtimes, suffix):
    for name, runtime in runtimes.items():
        if hasattr(runtime, "branch"):
            directory = stage/("final_bandit_states" if suffix == "final" else "bandit_checkpoints")
            directory.mkdir(exist_ok=True)
            filename = f"{name}.npz" if suffix == "final" else f"{name}_{suffix}.npz"
            runtime.branch.policy.model.save_mutable_state(directory/filename)
        if runtime.bundle is not None:
            directory = stage/"checkpoints"
            directory.mkdir(exist_ok=True)
            runtime.bundle.save(directory/f"{name}_{suffix}.npz")


def controller_summary(runtime):
    if runtime.bundle is None:
        return dict(steps=0, episodes=0)
    controller = runtime.bundle.controller
    return dict(steps=controller.steps, episodes=controller.episodes, **controller.summary())


def worker(output, label):
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("Exactly one MPI rank is required per seed")
    stage = output/f"seed_{label}"
    handles = {}
    try:
        check_frozen(output)
        runtimes = initialize(output, label)
        first = runtimes[factorial.METHODS[0]]
        protocol = read(output/"protocol.json")
        cases = protocol["stages"][label]["cases"]
        dump(stage/"environment.json", dict(pid=os.getpid(), executable=sys.executable,
            python=sys.version, qos=screen.request_performance_qos(), mpi=MPI.Get_library_version(),
            threads={key: os.environ.get(key) for key in THREAD_KEYS}))
        warmup = [sweep.execute_default_case(first, i) for i in range(min(3, cases))]
        dump(stage/"native_warmup.json", dict(measured=False, learner_updates=0, records=warmup))
        dump(stage/"status.json", dict(status="ready", pid=os.getpid(), processed=0, cases=cases))
        while not (output/"release.json").exists():
            time.sleep(0.2)
        rows = {name: [] for name in factorial.METHODS}
        previous = dict.fromkeys(factorial.METHODS, 0.0)
        orders = factorial.balanced_orders(cases, first.args.method_order_seed)
        factorial.audit_independence(runtimes)
        dump(stage/"activation_audit.json", dict(passed=True, problem_index=1,
            joint_controller_steps=0, fixed_controller_steps=0,
            setup_observations={name: runtimes[name].branch.policy.model.t for name in factorial.METHODS[2:]}))
        (stage/"trajectories").mkdir()
        handles = {name: (stage/"trajectories"/f"{name}.jsonl").open("w") for name in rows}
        started = time.perf_counter()
        with (stage/"execution.jsonl").open("w") as schedule:
            for index, order in enumerate(orders):
                for rank, name in enumerate(order):
                    begin = time.monotonic_ns()
                    row, previous[name] = factorial.execute(runtimes[name], index, previous[name])
                    end = time.monotonic_ns()
                    row.update(execution_rank=rank, execution=dict(pid=os.getpid(), position=rank))
                    rows[name].append(row)
                    handles[name].write(json.dumps(ready(row), allow_nan=False)+"\n")
                    schedule.write(json.dumps(dict(input_index=index, method=name, position=rank,
                                                   start_ns=begin, end_ns=end, pid=os.getpid()))+"\n")
                if (index+1) % first.args.progress_every == 0 or index+1 == cases:
                    for handle in handles.values():
                        handle.flush()
                    schedule.flush()
                    dump(stage/"status.json", dict(status="running", pid=os.getpid(), processed=index+1,
                        cases=cases, elapsed_sec=time.perf_counter()-started,
                        methods={name: sweep.metrics(records) for name, records in rows.items()}))
                    print(f"seed {label}: {index+1}/{cases}", flush=True)
                if (index+1) % 1000 == 0:
                    save_checkpoints(stage, runtimes, str(index+1))
        for handle in handles.values():
            handle.close()
        final_audits = {}
        for name, runtime in runtimes.items():
            measured_inputs = [dict(problem_index=row["problem_index"], mkw=row["mkw"],
                                    context=row["context"]) for row in rows[name]]
            if digest(measured_inputs) != protocol["stages"][label]["inputs_sha256"]:
                raise ValueError(f"Measured inputs differ from the frozen stream: {name}")
            controller = controller_summary(runtime)
            if runtime.bundle is not None:
                audit_controller_counts(rows[name], controller)
            model = screen.audit_final_model(runtime, rows[name]) if hasattr(runtime, "branch") else None
            final_audits[name] = dict(controller=controller, model=model,
                cases=len(rows[name]), trajectory_sha256=file_hash(stage/"trajectories"/f"{name}.jsonl"))
        save_checkpoints(stage, runtimes, "final")
        check_frozen(output)
        checkpoint_hashes = {str(path.relative_to(stage)): file_hash(path)
            for directory in (stage/"checkpoints", stage/"final_bandit_states")
            for path in directory.iterdir() if path.is_file()}
        dump(stage/"result.json", dict(complete=True, passed=True, cases=cases,
            elapsed_sec=time.perf_counter()-started, audits=final_audits,
            checkpoint_sha256=checkpoint_hashes,
            methods={name: sweep.metrics(records) for name, records in rows.items()}))
        dump(stage/"status.json", dict(status="complete", processed=cases, cases=cases,
                                      elapsed_sec=time.perf_counter()-started))
    except BaseException:
        dump(stage/"status.json", dict(status="failed", pid=os.getpid(), traceback=traceback.format_exc()))
        raise
    finally:
        for handle in handles.values():
            handle.close()


def analyze(output):
    protocol = read(output/"protocol.json")
    table, series = [], {}
    for label in protocol["seeds"]:
        stage = output/f"seed_{label}"
        result = read(stage/"result.json")
        if not result["complete"] or not result["passed"]:
            raise ValueError(f"Seed {label} incomplete or failed its audit")
        for name in factorial.METHODS:
            if file_hash(stage/"trajectories"/f"{name}.jsonl") != result["audits"][name]["trajectory_sha256"]:
                raise ValueError(f"Measured trajectory changed for seed {label}: {name}")
        records = {name: list(read_records(stage/"trajectories"/f"{name}.jsonl")) for name in factorial.METHODS}
        series[label] = {name: np.asarray([row["outcome"]["runtime"] for row in values])
                         for name, values in records.items()}
        cases = result["cases"]
        windows = {"full": (0, cases), "first_1000": (0, min(1000, cases))}
        activation = next(m.get("solve_activation_case", 0) for m in read(stage/"experiment_config.json")["methods"] if m["id"] == "bandit_lstdq")
        if activation < cases:
            windows["joint"] = (activation, cases)
        if cases >= 1000:
            windows["last_1000"] = (cases-1000, cases)
        for window, (start, stop) in windows.items():
            # Shared final-success subset is explicit and identical for all four methods.
            common = [i for i in range(start, stop) if all(not records[n][i]["outcome"]["failed"] for n in records)]
            default = sweep.metrics(records[factorial.METHODS[0]][start:stop])
            common_default = sweep.metrics([records[factorial.METHODS[0]][i] for i in common])
            for name in factorial.METHODS:
                metrics = sweep.metrics(records[name][start:stop])
                matched = sweep.metrics([records[name][i] for i in common])
                table.append(dict(seed=label, window=window, method=name, **metrics,
                    native_reduction_pct=100*(1-metrics["native_sec"]/default["native_sec"]),
                    inclusive_reduction_pct=100*(1-metrics["total_sec"]/default["total_sec"]),
                    common_success_cases=len(common),
                    common_success_native_reduction_pct=100*(1-matched["native_sec"]/common_default["native_sec"]) if common else None))
    write_csv(output/"summary.csv", table)
    aggregates = []
    for window in dict.fromkeys(row["window"] for row in table):
        for name in factorial.METHODS:
            selected = [r for r in table if r["window"] == window and r["method"] == name]
            aggregates.append(dict(window=window, method=name, seeds=len(selected),
                **{f"{field}_{stat}": float(np.mean(values) if stat == "mean" else np.std(values, ddof=1))
                   for field in ("native_reduction_pct", "inclusive_reduction_pct")
                   for values in ([r[field] for r in selected],)
                   for stat in (("mean", "std") if len(selected) > 1 else ("mean",))}))
    dump(output/"summary.json", dict(protocol=protocol, rows=table, aggregates=aggregates, passed=True))
    accounting = "All four methods were measured independently from their first paired input."
    text = [f"{protocol['name']}\n", accounting + " Native time includes every setup/solve attempt and fallback. Inclusive time also includes measured learner costs.\n",
        "| Seed | Method | Native seconds | Native reduction | Inclusive reduction | Final failures |",
        "|---|---|---:|---:|---:|---:|"]
    for row in table:
        if row["window"] == "full":
            text.append(f"| {row['seed']} | {factorial.LABELS[row['method']]} | {row['native_sec']:.3f} | {row['native_reduction_pct']:.2f}% | {row['inclusive_reduction_pct']:.2f}% | {row['failures']} |")
    text += ["", "Per-input trajectories and final setup/controller checkpoints are in each seed directory. Source and native hashes, feature checks, activation checks and batch reconstruction of both setup regressions passed."]
    (output/"report.md").write_text("\n".join(text)+"\n")
    factorial.audit_execution(output)
    plot_results(output, series)
    from experiments.paper_final.plot_04_factorial import generate
    figures = generate(output)
    with (output/"report.md").open("a") as handle:
        handle.write("\n[Paper figures](paper_figures/module04_paper_figures.pdf) · [Joint LSTDQ heat maps](paper_figures/joint_lstdq_heatmaps_all_seeds.png) · [Plot data and provenance](paper_figures/manifest.json)\n")
    return table


def plot_results(output, series):
    from experiments.paper_final.common.figure_style import apply_style, plt
    apply_style()
    colors = ("#686868", "#009e73", "#e69f00", "#0072b2")
    fig, axes = plt.subplots(len(series), 2, figsize=(10, 2.6*len(series)), squeeze=False)
    for (label, methods), row in zip(series.items(), axes):
        n = len(methods[factorial.METHODS[0]])
        x = np.arange(1, n+1)
        reference = np.cumsum(methods[factorial.METHODS[0]])
        for (name, values), color in zip(methods.items(), colors):
            cumulative = np.cumsum(values)
            row[0].plot(x, cumulative, color=color, label=factorial.LABELS[name])
            if name != factorial.METHODS[0]:
                row[1].plot(x, 100*(1-cumulative/reference), color=color)
        for ax in row:
            activation = next(m.get("solve_activation_case", 0) for m in read(output/f"seed_{label}"/"experiment_config.json")["methods"] if m["id"] == "bandit_lstdq")
            if 0 < activation < n:
                ax.axvline(activation + .5, color="#555555", ls=":", lw=.8)
            ax.grid(alpha=.2)
            ax.set_xlabel("Problem index")
        row[0].set(title=f"Seed {label}: cumulative native time", ylabel="Seconds")
        row[1].set(title=f"Seed {label}: cumulative reduction vs Default", ylabel="Reduction (%)")
        row[1].axhline(0, color="#686868", lw=.6)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0,0,1,.96))
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output/f"learning_curves.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def launch(output):
    check_frozen(output)
    if read(output/"status.json")["status"] != "prepared":
        raise ValueError("Suite is not fresh; partial experiment resume is unsupported")
    protocol = read(output/"protocol.json")
    children, logs = [], []
    sleep_guard = prevent_sleep()
    try:
        dump(output/"status.json", dict(status="initializing", pid=os.getpid()))
        for label in protocol["seeds"]:
            log = (output/f"seed_{label}"/"worker.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen([sys.executable, "-m", "experiments.paper_final.run_04_factorial",
                "worker", "--output", str(output), "--label", label], cwd=ROOT, stdout=log,
                stderr=subprocess.STDOUT, env=single_thread_environment()))
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("A worker exited during preparation; inspect seed worker.log/status.json")
            if all(read(output/f"seed_{label}"/"status.json")["status"] == "ready" for label in protocol["seeds"]):
                break
            time.sleep(0.5)
        dump(output/"release.json", dict(at=now(), worker_pids=[child.pid for child in children]))
        dump(output/"status.json", dict(status="running", pid=os.getpid(), worker_pids=[child.pid for child in children]))
        print(f"Released {len(children)} seed workers", flush=True)
        while any(child.poll() is None for child in children):
            if any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError("A worker failed; stopping the paired suite")
            time.sleep(1)
        if any(child.returncode for child in children):
            raise RuntimeError("A worker failed; inspect worker.log/status.json")
        check_frozen(output)
        analyze(output)
        dump(output/"status.json", dict(status="complete", at=now(), passed=True))
        print("All seed experiments and audits complete", flush=True)
    except BaseException:
        dump(output/"status.json", dict(status="failed", traceback=traceback.format_exc()))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        for log in logs:
            log.close()
        stop_sleep_prevention(sleep_guard)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "worker", "analyze"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--cases", type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "prepare":
        if args.suite is None:
            parser.error("prepare requires --suite")
        prepare(args.suite.resolve(), output, cases=args.cases)
    elif args.command == "worker":
        worker(output, args.label)
    elif args.command == "run":
        launch(output)
    else:
        analyze(output)


if __name__ == "__main__":
    main()
