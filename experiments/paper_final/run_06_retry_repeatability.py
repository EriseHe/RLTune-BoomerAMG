"""Serial, balanced, repeated setup-recovery comparisons on paired inputs.

One process owns independent learners and executes a whole method case
at a time. Preparation, warmup, logging, and deliberate scheduling waits never
enter a learner's measured cost. No concurrent native worker is launched.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from experiments.runtime import configure_single_thread, prevent_sleep, stop_sleep_prevention, THREAD_KEYS

configure_single_thread()

import numpy as np

from experiments.paper_final import run_06_retry_screen as screen
from experiments.paper_final.common.artifacts import (
    ROOT, checkpoint_payload_hash, digest, dump, emit, file_hash, now, read, ready, write_csv,
)
from experiments.paper_final.recovery.retry_comparison import (
    ConservativeRetryLinUCB, FailureAwareRetryLinUCB, use_retry_rule,
)
from setup.space import DEFAULT_SETUP_PARAMS

METHODS = ("failure_model_suffix", "failure_model_adjacent", "default")
ALL_METHODS = tuple(screen.BRANCHES)
ORDERS = tuple(itertools.permutations(METHODS))
MODULE04_ORDER = "module04_seeded_per_input_permutation"


def method_order(index, repetition):
    """Each method occupies each position once per input over three repetitions."""
    base = ORDERS[index % len(ORDERS)]
    rotation = repetition % len(METHODS)
    return base[rotation:] + base[:rotation]


def stage_methods(protocol):
    return tuple(protocol.get("methods_order", METHODS))


def execution_orders(protocol):
    methods = stage_methods(protocol)
    if protocol.get("order_rule") == MODULE04_ORDER:
        rng = np.random.default_rng(int(protocol["seeds"]["method_order"]))
        return [tuple(methods[int(i)] for i in rng.permutation(len(methods)))
                for _ in range(protocol["cases"])]
    if methods != METHODS:
        raise ValueError("Additional methods require Module 04 execution order")
    return [method_order(i, protocol["repetition"]) for i in range(protocol["cases"])]


class ObservedSelection:
    """Capture already computed scores; preserve the selector and its RNG calls."""

    def _selection_score_subset(self, x, **kwargs):
        score, mean, width = super()._selection_score_subset(x, **kwargs)
        arms = kwargs["arms"]
        self._capture = dict(arms=np.arange(self.K) if arms is None else arms,
                             score=score, mean=mean, width=width, alpha=kwargs["alpha"])
        return score, mean, width

    def _record_choice(self, retry):
        data = self._capture
        if data is None:
            record = dict(arm=int(self._last_arm), strategy="initial_guess")
        else:
            location = int(np.flatnonzero(data["arms"] == self._last_arm)[0])
            objective = data["retry_score"] if retry else data["score"]
            ranked = np.lexsort((data["score"], objective))
            def candidate(i):
                result = dict(arm=int(data["arms"][i]), mean=float(data["mean"][i]),
                    width=float(data["width"][i]), runtime_lcb=float(data["score"][i]))
                if retry and "failure_mean" in data:
                    result.update(failure_mean=float(data["failure_mean"][i]),
                                  failure_upper=float(data["failure_upper"][i]))
                elif retry:
                    result["runtime_ucb"] = float(data["retry_score"][i])
                return result
            strategy = ("failure_upper" if "failure_mean" in data else "runtime_ucb") if retry else "runtime_lcb"
            record = dict(strategy=strategy,
                observations=self.t, alpha=float(data["alpha"]), selected=candidate(location),
                best_two_gap=float(objective[ranked[1]]-objective[ranked[0]]) if len(ranked)>1 else None,
                candidate_count=len(ranked), candidate_arms=data["arms"].tolist(),
                top_candidates=[candidate(int(i)) for i in ranked[:8]])
        self.selection_diagnostics.append(record)
        self._capture = None

    def predict(self, context):
        self._capture = None
        result = super().predict(context)
        self._record_choice(False)
        return result

    def select_recovery(self, context, *, excluded_arms):
        self._capture = None
        result = super().select_recovery(context, excluded_arms=excluded_arms)
        self._record_choice(True)
        return result


class ObservedFailureAware(ObservedSelection, FailureAwareRetryLinUCB):
    def _recovery_best_locations(self, x, **kwargs):
        prediction = self._linear_prediction_subset(
            x, theta=self.A_inv @ self.failure_b, arms=kwargs["candidates"],
            action_features=kwargs["candidate_features"])
        upper = prediction + self.failure_beta * kwargs["runtime_uncert"]
        runtime_score = kwargs["runtime_score"]
        best = int(np.lexsort((runtime_score, upper))[0])
        self._capture.update(failure_mean=prediction, failure_upper=upper, retry_score=upper)
        return np.flatnonzero(np.isclose(upper, upper[best], atol=1e-12, rtol=0)
            & np.isclose(runtime_score, runtime_score[best], atol=1e-12, rtol=0))


class ObservedConservative(ObservedSelection, ConservativeRetryLinUCB):
    def _recovery_best_locations(self, x, **kwargs):
        score = kwargs["runtime_mean"] + self._effective_alpha() * kwargs["runtime_uncert"]
        self._capture["retry_score"] = score
        return np.flatnonzero(np.isclose(score, np.min(score), atol=1e-12, rtol=0))


def observed_model(model):
    if isinstance(model, FailureAwareRetryLinUCB):
        cls = ObservedFailureAware
    elif isinstance(model, ConservativeRetryLinUCB):
        cls = ObservedConservative
    else:
        raise TypeError("Unknown retry learner")
    observed = object.__new__(cls)
    observed.__dict__ = model.__dict__.copy()
    observed.selection_diagnostics, observed._capture = [], None
    return observed


def initialize(output, name):
    protocol = read(output/"protocol.json")
    directory = output/name
    directory.mkdir(exist_ok=False)
    config_path = output/"configs"/f"{name}.json"
    if file_hash(config_path) != protocol["preparation_checks"][name]["config_sha256"]:
        raise ValueError("Prepared configuration changed")
    runtime = screen.sweep.decode(read(config_path), directory)
    runtime.name = name
    model = None
    state = dict(pid=os.getpid(), cases=protocol["cases"],
        inputs_sha256=digest(screen.input_rows(runtime, protocol["cases"])),
        full_stream_sha256=runtime.manifest["sha256"], native_sha256=screen.sweep.native_hashes(),
        candidate_sha256=None, initial_bandit_sha256=None, initial_controller_sha256=None,
        bandit_observations=0, controller_steps=0, candidate_cursor=None)
    if name != "default":
        setup = screen.build_named_setup_branches(runtime.args,
            study=screen.resolve_composable_study(runtime.args), warmup_instances=(),
            stream_hash=runtime.manifest["sha256"])
        runtime.branch = setup.branches[runtime.method.name]
        model = observed_model(use_retry_rule(runtime.branch.policy.model, name))
        runtime.branch.policy.model = model
        runtime.rows = screen.CandidateRows(model, cases=screen.sweep.CASES, attempts=3,
                                           directory=directory/"unused_extra_candidates")
        model.save_mutable_state(directory/"initial_bandit.npz")
        runtime.bundle.save(directory/"initial_controller.npz")
        exact_default = [int(a) for a in model._cand.always_include_arms
                         if dict(model.actions[int(a)]) == DEFAULT_SETUP_PARAMS]
        if not exact_default:
            raise ValueError("The catalog reference is not exactly Default")
        state.update(candidate_sha256=file_hash(Path(runtime.rows.base._ids.filename)),
            initial_bandit_sha256=checkpoint_payload_hash(directory/"initial_bandit.npz", exclude=("metadata", "retry_failure_b")),
            initial_controller_sha256=checkpoint_payload_hash(directory/"initial_controller.npz"),
            candidate_cursor=model._candidate_schedule.cursor,
            exact_default_arms=exact_default, action_count=model.K)
    dump(directory/"ready.json", state)
    return SimpleNamespace(runtime=runtime, model=model, state=state, directory=directory,
        rows=[], previous=0.0, elapsed=0.0)


def audit_execution(output):
    protocol = read(output/"protocol.json")
    methods = stage_methods(protocol)
    orders = execution_orders(protocol)
    schedule = [json.loads(s) for s in (output/"execution.jsonl").read_text().splitlines()]
    if len(schedule) != protocol["cases"]*len(methods):
        raise ValueError("Missing method-case executions")
    previous_end = -1
    for index in range(protocol["cases"]):
        block = schedule[index*len(methods):(index+1)*len(methods)]
        if tuple(r["method"] for r in block) != orders[index]:
            raise ValueError("Execution order differs from the prepared schedule")
        for row in block:
            if row["input_index"] != index or row["start_ns"] < previous_end or row["end_ns"] < row["start_ns"]:
                raise ValueError("Overlapping or out-of-order method execution")
            previous_end = row["end_ns"]
    if len({r["pid"] for r in schedule}) != 1:
        raise ValueError("Expected exactly one numerical process")
    dump(output/"execution_audit.json", dict(passed=True, method_cases=len(schedule),
         overlap_count=0, numerical_processes=1, order_verified=True,
         balanced_order=protocol.get("order_rule") != MODULE04_ORDER,
         module04_method_order=protocol.get("order_rule") == MODULE04_ORDER,
         timing="Each complete case is timed internally; scheduler/logging work excluded"))


def run_stage(output):
    protocol = read(output/"protocol.json")
    methods = stage_methods(protocol)
    orders = execution_orders(protocol)
    if read(output/"status.json")["status"] != "prepared":
        raise ValueError("Only a fresh prepared stage can run")
    if screen.source_hashes() != read(output/"source_manifest.json") or screen.sweep.native_hashes() != protocol["native_sha256"]:
        raise ValueError("Source/native code changed after freezing")
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        raise ValueError("Expected one MPI rank")
    dump(output/"status.json", dict(status="initializing", pid=os.getpid(), started_at=now()))
    qos = screen.request_performance_qos()
    workers = {n: initialize(output, n) for n in methods}
    states = {n: w.state for n, w in workers.items()}
    screen.verify_pairing(states, protocol)
    learned = [w.model for w in workers.values() if w.model is not None]
    for a, b in itertools.combinations(learned, 2):
        arrays = ("A_inv", "b", "failure_b") if hasattr(a, "failure_b") and hasattr(b, "failure_b") else ("A_inv", "b")
        if a.rng is b.rng or any(np.shares_memory(getattr(a,k), getattr(b,k)) for k in arrays):
            raise ValueError("Methods share mutable learner state")
    dump(output/"pairing.json", dict(passed=True, workers=states, independent_state=True))
    dump(output/"environment.json", dict(python=sys.version, executable=sys.executable, pid=os.getpid(),
        qos=qos, mpi=MPI.Get_library_version(), threads={k:os.environ.get(k) for k in THREAD_KEYS},
        execution="Single process, complete serial method cases; prepared method order"))
    # Native warmup uses no learner and is never reused as measured work.
    warmup = [screen.sweep.execute_default_case(workers["default"].runtime, i) for i in range(3)]
    dump(output/"native_warmup.json", dict(measured=False, learner_updates=0, records=warmup))
    handles = {n:(output/n/"trajectory.jsonl").open("w") for n in methods}
    started = time.perf_counter()
    try:
        with (output/"execution.jsonl").open("w") as schedule:
            for index in range(protocol["cases"]):
                for position, name in enumerate(orders[index]):
                    worker = workers[name]
                    begin = time.monotonic_ns()
                    if name == "default":
                        row = screen.sweep.execute_default_case(worker.runtime, index)
                    else:
                        before = len(worker.model.selection_diagnostics)
                        row, worker.previous = screen.execute_setup_case(worker.runtime, index, worker.previous)
                    end = time.monotonic_ns()
                    worker.elapsed += (end-begin)*1e-9
                    if name != "default":
                        row["selection_diagnostics"] = worker.model.selection_diagnostics[before:]
                        if len(row["selection_diagnostics"]) != row["outcome"]["primary_attempt_count"]:
                            raise ValueError("Missing selection diagnostic")
                    row["execution"] = dict(repetition=protocol["repetition"], position=position, pid=os.getpid())
                    worker.rows.append(row)
                    handles[name].write(json.dumps(ready(row), allow_nan=False)+"\n")
                    schedule.write(json.dumps(dict(input_index=index, method=name, position=position,
                        pid=os.getpid(), start_ns=begin, end_ns=end))+"\n")
                    if worker.model is not None and index < 30:
                        worker.model.save_mutable_state(worker.directory/"early_states"/f"after_{index+1:04d}.npz")
                if (index+1)%100 == 0 or index+1 == protocol["cases"]:
                    for handle in handles.values():handle.flush()
                    schedule.flush()
                    progress = dict(status="running", pid=os.getpid(), processed_per_method=index+1,
                        cases=protocol["cases"], elapsed_sec=time.perf_counter()-started,
                        methods={n:screen.metrics(w.rows) for n,w in workers.items()})
                    dump(output/"status.json", progress)
                    emit(output, "Serial block complete", processed=index+1,
                         totals={n:round(v["total_sec"],3) for n,v in progress["methods"].items()})
    finally:
        for handle in handles.values():handle.close()
    for name, worker in workers.items():
        model_audit = None
        if worker.model is not None:
            model_audit = screen.audit_final_model(worker.runtime, worker.rows)
            worker.model.save_mutable_state(worker.directory/"final_bandit.npz")
            worker.runtime.bundle.save(worker.directory/"final_controller.npz")
            controller = worker.runtime.bundle.controller
            if controller.steps or controller.episodes or checkpoint_payload_hash(worker.directory/"final_controller.npz") != worker.state["initial_controller_sha256"]:
                raise ValueError("RL changed in a setup-only screen")
            if worker.model._candidate_schedule.cursor != protocol["cases"]*3:
                raise ValueError("Wrong candidate schedule boundary")
        dump(worker.directory/"result.json", dict(complete=True, cases=protocol["cases"],
            elapsed_sec=worker.elapsed, trajectory_sha256=file_hash(worker.directory/"trajectory.jsonl"),
            pairing=worker.state, controller=dict(steps=0,episodes=0), model_audit=model_audit,
            bandit_observations=0 if worker.model is None else worker.model.t))
    if screen.source_hashes() != read(output/"source_manifest.json") or screen.sweep.native_hashes() != protocol["native_sha256"]:
        raise ValueError("Frozen source/native code changed during the stage")
    screen.analyze(output)
    audit_execution(output)
    dump(output/"status.json", dict(status="complete", completed_at=now(), audits_passed=True))


def prepare(output, *, cases=1000, seeds=(5,6), repetitions=3, methods=METHODS,
            module04_order=False, concurrent_seeds=False):
    methods = tuple(methods)
    if repetitions not in (1, 3):
        raise ValueError("Choose one fresh run or three execution repetitions")
    if methods not in (METHODS, ALL_METHODS):
        raise ValueError("Expected the feedback pair or all four retry methods plus Default")
    if methods != METHODS and not module04_order:
        raise ValueError("All four methods require Module 04 ordering")
    if concurrent_seeds and repetitions != 1:
        raise ValueError("Concurrent seeds require one run per seed")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    module04_audit = None
    if module04_order:
        reference_path = ROOT/"experiments/paper_final/04_online/20260920_formal/advection_60.json"
        reference, current = read(reference_path), read(screen.sweep.CONFIG)
        checks = {"setup_space_and_candidate_schedule": current["setup"] == reference["setup"]}
        checks.update({key:current["solve"][key] == reference["solve"][key]
                       for key in ("tolerance", "max_cycles", "smoother_profile")})
        keys = ("cases", "warmup_cases", "online_cases", "instance_offset", "cases_per_seed", "group_take")
        checks["full_stream_layout"] = all(current["stream"][key] == reference["stream"][key] for key in keys)
        if not all(checks.values()):
            raise ValueError(f"Module 04 core settings differ: {checks}")
        module04_audit = dict(passed=True, checks=checks, reference_path=str(reference_path),
            reference_sha256=file_hash(reference_path), method_order=MODULE04_ORDER,
            online_learning="Fresh independent persistent setup state; complete cases serial within each seed",
            scope="Type-2 setup-only recovery screen; retain the four existing retry rules and delayed cost labels",
            inputs="First 1000 of each seed bundle's 5000-input shuffled stream; no trained state reused")
    output.mkdir(parents=True, exist_ok=False)
    if module04_audit is not None:
        dump(output/"module04_consistency.json", module04_audit)
    stages = []
    for seed in seeds:
        for repetition in range(repetitions):
            name = f"seed_{seed}" if repetitions == 1 else f"seed_{seed}_repeat_{repetition+1}"
            directory = output/name
            screen.prepare(directory, cases=cases, seed=seed, learned_attempts=3, branches=methods)
            protocol = read(directory/"protocol.json")
            protocol.update(repetition=repetition, execution="serial_complete_cases", methods_order=list(methods),
                order_rule=MODULE04_ORDER if module04_order else "6 permutations cycled by input; rotate by repetition; each position once per input",
                exact_reference_values=DEFAULT_SETUP_PARAMS, diagnostics="captured selection scores and first 30 post-case states")
            dump(directory/"protocol.json", protocol)
            stages.append(dict(name=name, seed=seed, repetition=repetition,
                protocol_sha256=file_hash(directory/"protocol.json"),
                inputs_sha256=file_hash(directory/"inputs.jsonl")))
    dump(output/"sequence_protocol.json", dict(stages=stages, cases=cases, repetitions=repetitions,
        methods=methods, seeds=seeds, prepared_at=now(), fresh_learners=True,
        concurrent_seeds=concurrent_seeds, serial_methods=True, module04_order=module04_order,
        decision="Report repetition-level differences; no automatic selection from a pooled narrow win"))
    dump(output/"status.json", dict(status="prepared", stages=len(stages), native_solves_executed=0))


def analyze(output):
    plan = read(output/"sequence_protocol.json")
    methods = tuple(plan["methods"])
    learned = [n for n in methods if n != "default"]
    table, paths, records = [], {}, {}
    for stage in plan["stages"]:
        directory = output/stage["name"]
        if read(directory/"status.json")["status"] != "complete":
            raise ValueError("A stage has not completed")
        if not read(directory/"execution_audit.json")["passed"]:
            raise ValueError("Serial execution audit failed")
        for name in methods:
            rows = [json.loads(s) for s in (directory/name/"trajectory.jsonl").read_text().splitlines()]
            records[(stage["seed"],stage["repetition"],name)] = rows
            for label, subset in (("all",rows),("first_250",rows[:250]),("last_250",rows[-250:])):
                table.append(dict(seed=stage["seed"], repetition=stage["repetition"]+1,
                    branch=name, window=label, **screen.metrics(subset)))
        paths[stage["name"]] = str(directory/"REPORT.txt")
    # Prove identical initial states, schedules, and inputs across repetitions.
    for seed in plan["seeds"]:
        stages = [s for s in plan["stages"] if s["seed"]==seed]
        if len({s["inputs_sha256"] for s in stages}) != 1:
            raise ValueError("Repeated input streams changed")
        for key in ("candidate_sha256","initial_bandit_sha256","initial_controller_sha256"):
            values = {read(output/s["name"]/n/"ready.json")[key] for s in stages for n in learned}
            if len(values) != 1:raise ValueError(f"Repeated starting state differs: {key}")
        # Default must execute the same numerical work across all repetitions.
        baseline = records[(seed,0,"default")]
        for repetition in range(1,plan["repetitions"]):
            other = records[(seed,repetition,"default")]
            for a,b in zip(baseline,other):
                for key in ("failed","completed_cycles","completed_residual_norm","completed_status"):
                    if a["outcome"][key] != b["outcome"][key]:
                        raise ValueError(f"Repeated Default numerical result differs: {seed}, {key}")
    write_csv(output/"comparison.csv", table)
    totals = [r for r in table if r["window"]=="all"]
    summary = []
    for seed in plan["seeds"]:
        for name in methods:
            group = [r for r in totals if r["seed"]==seed and r["branch"]==name]
            times = [r["total_sec"] for r in group]
            summary.append(dict(seed=seed, branch=name, mean_total_sec=float(np.mean(times)),
                min_total_sec=min(times), max_total_sec=max(times),
                sample_sd_sec=float(np.std(times,ddof=1)) if len(times)>1 else None,
                total_failures=sum(r["failures"] for r in group)))
    dump(output/"comparison.json", dict(complete=True,audits_passed=True,rows=table,summary=summary,stages=paths))
    lines = ["Serial setup-recovery comparison", f"{plan['repetitions']} fresh run(s) per seed; one complete method case at a time.",
        "All measured attempts, failures, fallback and learner overhead are included. RL was inactive.",
        "The intended catalog reference uses exact Default values. Raw records and original measurements are retained.", ""]
    if plan.get("module04_order"):
        lines[1:1] = ["Methods follow Module 04's seeded per-input random permutation, with separate scheduling RNG.",
                      "Seeds ran concurrently in separate single-threaded processes." if plan.get("concurrent_seeds") else "Seeds ran sequentially."]
    for row in totals:
        lines.append(f"seed {row['seed']}, repeat {row['repetition']}, {row['branch']}: {row['total_sec']:.6f}s; "
            f"failures={row['failures']}; final fallbacks={row['fallbacks']}; default learned retries={row['learned_default_retries']}; "
            f"second={row['second_attempts']}; third={row['third_attempts']}")
    lines += ["", "Whole-run summary:"]
    for row in summary:lines.append(json.dumps(row))
    (output/"REPORT.txt").write_text("\n".join(lines)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,len(plan["seeds"]),figsize=(6.4*len(plan["seeds"]),4.8),squeeze=False,constrained_layout=True)
    label_by_method = {
        "conservative_suffix": "Conservative\nfull suffix",
        "conservative_adjacent": "Conservative\nadjacent",
        "failure_model_suffix": "Failure-aware\nfull suffix",
        "failure_model_adjacent": "Failure-aware\nadjacent", "default": "Default",
    }
    labels = [label_by_method[n] for n in methods]
    for ax,seed in zip(axes[0],plan["seeds"]):
        for i,name in enumerate(methods):
            times = [r["total_sec"] for r in totals if r["seed"]==seed and r["branch"]==name]
            ax.scatter(np.arange(len(times))*.12-.12+i,times,s=50)
            ax.plot([i-.18,i+.18],[np.mean(times)]*2,color="black")
        ax.set(xticks=range(len(methods)),xticklabels=labels,ylabel="Total measured time (s)",
               title=f"Seed {seed}: {plan['repetitions']} fresh run(s)")
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y",alpha=.2)
    fig.savefig(output/"comparison.png",dpi=180);fig.savefig(output/"comparison.pdf");plt.close(fig)


def run_concurrent_stages(output, plan):
    workers, handles = {}, []
    try:
        for stage in plan["stages"]:
            directory = output/stage["name"]
            if file_hash(directory/"protocol.json") != stage["protocol_sha256"]:
                raise ValueError("Prepared protocol changed")
            log = (output/f"{stage['name']}.log").open("w")
            handles.append(log)
            worker = subprocess.Popen([sys.executable,"-u","-m",__package__+".run_06_retry_repeatability",
                "--stage","--output",str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            workers[stage["name"]] = worker
            emit(output,"Started independent seed with serial methods",stage=stage["name"],pid=worker.pid)
        dump(output/"active_workers.json", {name:dict(pid=p.pid,started_at=now()) for name,p in workers.items()})
        pending = dict(workers)
        dump(output/"status.json",dict(status="running",pid=os.getpid(),active_stages=list(pending),updated_at=now()))
        while pending:
            for name, worker in list(pending.items()):
                code = worker.poll()
                if code is None:
                    continue
                if code:
                    dump(output/name/"status.json",dict(status="failed",exit_code=code,failed_at=now()))
                    raise RuntimeError(f"Stage {name} exited {code}; inspect its log")
                if read(output/name/"status.json")["status"] != "complete":
                    raise ValueError(f"Stage {name} exited without completion audits")
                pending.pop(name)
                emit(output,"Seed completed and audited",stage=name)
                dump(output/"status.json",dict(status="running",pid=os.getpid(),active_stages=list(pending),updated_at=now()))
            if pending:
                time.sleep(0.5)
    finally:
        for worker in workers.values():
            if worker.poll() is None:
                worker.terminate()
        for worker in workers.values():
            if worker.poll() is None:
                worker.wait(timeout=30)
        for handle in handles:
            handle.close()


def run(output):
    if read(output/"status.json")["status"] != "prepared":
        raise ValueError("Only an unstarted sequence can run")
    with (output/"manager.lock").open("x") as handle:handle.write(str(os.getpid()))
    dump(output/"manager.json", dict(pid=os.getpid(),module=__name__,started_at=now()))
    plan = read(output/"sequence_protocol.json")
    guard = prevent_sleep()
    try:
        if plan.get("concurrent_seeds"):
            run_concurrent_stages(output, plan)
        for stage in (() if plan.get("concurrent_seeds") else plan["stages"]):
            directory = output/stage["name"]
            if file_hash(directory/"protocol.json") != stage["protocol_sha256"]:
                raise ValueError("Prepared protocol changed")
            dump(output/"status.json",dict(status="running",pid=os.getpid(),active_stage=stage["name"],updated_at=now()))
            emit(output,"Starting fresh serial repetition",stage=stage["name"])
            # A new process clears native/Python state between repetitions.
            with (output/f"{stage['name']}.log").open("w") as log:
                worker = subprocess.Popen([sys.executable,"-u","-m",__package__+".run_06_retry_repeatability",
                    "--stage","--output",str(directory)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                dump(output/"active_worker.json",dict(pid=worker.pid,stage=stage["name"],started_at=now()))
                code = worker.wait()
            if code != 0:raise RuntimeError(f"Stage {stage['name']} exited {code}; inspect its log")
            emit(output,"Serial repetition complete",stage=stage["name"])
        analyze(output)
        dump(output/"status.json",dict(status="complete",completed_at=now(),audits_passed=True))
        emit(output,"All serial repetitions completed and audited")
    except BaseException as error:
        dump(output/"status.json",dict(status="failed",failed_at=now(),error=str(error)))
        raise
    finally:
        if 'worker' in locals() and worker.poll() is None:
            worker.terminate();worker.wait(timeout=30)
        stop_sleep_prevention(guard)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--cases",type=int,default=1000)
    parser.add_argument("--seeds",type=int,nargs="+",default=[5,6],choices=tuple(screen.SEED_CONFIGS))
    parser.add_argument("--repetitions",type=int,choices=(1,3),default=3)
    parser.add_argument("--all-methods",action="store_true")
    parser.add_argument("--module04-order",action="store_true")
    parser.add_argument("--concurrent-seeds",action="store_true")
    modes=parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare","run","stage","analyze"):modes.add_argument("--"+mode,action="store_true")
    args=parser.parse_args();output=args.output.resolve()
    if args.prepare:prepare(output,cases=args.cases,seeds=tuple(args.seeds),repetitions=args.repetitions,
        methods=ALL_METHODS if args.all_methods else METHODS,
        module04_order=args.module04_order,concurrent_seeds=args.concurrent_seeds)
    elif args.run:run(output)
    elif args.stage:run_stage(output)
    else:analyze(output)


if __name__=="__main__":
    signal.signal(signal.SIGTERM,lambda signum,frame:sys.exit(128+signum))
    main()
