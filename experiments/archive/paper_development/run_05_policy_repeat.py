"""Module 05 Run 02: retime frozen policies on the exact Run 01 Joint jobs."""
from __future__ import annotations

from experiments.paper_final.common.policy_analysis import (
    expected_cells,
)

from experiments.runtime import (
    configure_single_thread, prevent_sleep, stop_sleep_prevention, single_thread_environment,
)

if __name__ == "__main__":
    configure_single_thread()

from experiments.archive.paper_development import run_05_policy as first

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

ROOT = first.ROOT
DEFAULT_OUTPUT = ROOT / "results/paper_final/05_policy/20260927_run02_diffusion60_joint_6seeds_100cases"
SOURCE = "bandit_lstdq"
METHODS = ("reference", "fixed", "oracle", "periodic", "periodic13", "rl")
MAIN_METHODS = ("fixed", "oracle", "periodic13", "rl")
LABELS = {"reference": "Default w=1", "fixed": "Global best fixed*", "oracle": "Per-instance best fixed*",
          "periodic": "Tuned periodic", "periodic13": "Periodic (1, 3)",
          "rl": "Frozen RL"}


def method_map(chosen, seed, case):
    result = {}
    for method in ("reference", "fixed", "oracle", "periodic", "rl"):
        value = chosen[f"{seed}/{SOURCE}/{method}"]
        result[method] = value.get(str(case), value.get(case)) if isinstance(value, dict) else value
        if not result[method]:
            raise ValueError(f"Missing preselected policy: {seed}/{case}/{method}")
    result.update(periodic13="periodic_1.00_3.00")
    return result


def plan_jobs(base_jobs, chosen, repeats=3, order_seed=92705201):
    jobs = []
    for repeat in range(repeats):
        batch = []
        for original in base_jobs:
            if original["source"] != SOURCE:
                continue
            row = copy.deepcopy(original)
            row.update(repeat=repeat, method_policies=method_map(chosen, row["seed"], row["case_id"]))
            names = sorted(set(row["method_policies"].values()))
            random.Random(first.digest([order_seed, repeat, row["seed"], row["case_id"]])).shuffle(names)
            row["policy_order"] = names
            batch.append(row)
        random.Random(order_seed + repeat).shuffle(batch)
        jobs.extend(batch)
    return jobs


def prepare(output, parent=first.DEFAULT_OUTPUT):
    from experiments.archive.paper_development.module05_figures.data import load_dataset
    output, parent = Path(output).resolve(), Path(parent).resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Run 02 directory already exists; use its saved protocol to resume")
    first.verify_inputs(parent)
    data = load_dataset(parent)
    chosen = data.data_export()["chosen_policies"]
    base_jobs = first.read(parent / "jobs_test.json")
    joint = [j for j in base_jobs if j["source"] == SOURCE]
    if len(joint) != 600 or {j["seed"] for j in joint} != set(range(1, 7)):
        raise ValueError("Expected all six checkpoints and 100 original Joint test inputs")
    jobs = plan_jobs(base_jobs, chosen)
    expected = expected_cells(jobs)
    output.mkdir(parents=True)
    for name in ("progress", "analysis", "raw/test", "checkpoints", "provenance"):
        (output / name).mkdir(parents=True, exist_ok=True)
    selection = {"origin_run": str(parent), "selection_frozen_before_run02": True,
                 "chosen_policies": chosen, "case_order_0_based": data.order.tolist(),
                 "note": "Fixed choices are Run 01 best-observed grid choices, retimed without reselection. No new oracle search."}
    first.dump(output / "selections.json", selection)
    protocol = copy.deepcopy(data.protocol)
    used = {name for j in jobs for name in j["policy_order"]}
    protocol.update(run_number=2, created_at=first.now(), origin_run=str(parent), sources=[SOURCE],
        development_cases=0, repetitions_per_case=3, test_repeat_case_ids=list(range(100)), additional_repeats=2,
        methods=list(METHODS), main_heatmap_methods=list(MAIN_METHODS), policies={k: v for k, v in protocol["policies"].items() if k in used},
        execution_count=len(expected), order_seed=92705201,
        scope="Fresh timing only: exact existing inputs, frozen Joint setups and RL checkpoints; six reported policies; three repetitions of every case",
        selection_rule="Reuse Run 01 selected constants and periodic schedule; never select again using Run 02 timings",
        oracle_rule="Retimed Run 01 global/per-instance best-observed 41-grid choices, not new minima of Run 02 timings",
        timing="At most three single-thread workers; three randomized passes, policy order randomized within each case; identical fixed choices share one execution",
        sorting="Identical baseline-difficulty problem order to Run 01; no sorting on Run 02 RL performance")
    first.dump(output / "protocol.json", protocol)
    shutil.copy2(parent / "inputs.json", output / "inputs.json")
    first.dump(output / "jobs_test.json", jobs)
    first.dump(output / "jobs_preflight.json", [j for j in first.read(parent / "jobs_preflight.json") if j["source"] == SOURCE])
    for seed in protocol["training_seeds"]:
        dest = output / "checkpoints" / f"seed_{seed}"
        dest.mkdir()
        for name in ("experiment_config.json", "bandit_lstdq_final.npz", "bandit_lstdq_final.npz.encoder.json", "bandit_lstdq_setup_original.npz"):
            shutil.copy2(parent / "checkpoints" / f"seed_{seed}" / name, dest / name)
    manifest = first.read(parent / "source_manifest.json")
    extras = ["run_05_policy_repeat.py", "analyze_05_policy_repeat.py", "plot_05_policy_repeat.py", "test_05_policy_repeat.py",
              "module05_figures/data.py", "module05_figures/style.py", "module05_figures/panels.py", "05_policy/RUN02.md"]
    for name in extras:
        relative = "experiments/archive/paper_development/" + name
        manifest[relative] = first.file_hash(ROOT / relative)
        dest = output / "provenance/source" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, dest)
    for name in first.support_source_paths():
        manifest[name] = first.file_hash(ROOT/name)
        target = output / "provenance/source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    first.dump(output / "source_manifest.json", manifest)
    first.dump(output / "prepared.json", {"at": first.now(), "protocol_sha256": first.file_hash(output / "protocol.json"),
        "inputs_sha256": first.file_hash(output / "inputs.json"),
        "jobs_sha256": {k: first.file_hash(output / f"jobs_{k}.json") for k in ("test", "preflight")},
        "checkpoint_sha256": {str(p.relative_to(output)): first.file_hash(p) for p in (output / "checkpoints").rglob("*") if p.is_file()},
        "selections_sha256": first.file_hash(output / "selections.json"),
        "origin_raw_hashes": data.provenance["source_hashes"]})
    (output / "README.md").write_text(
        "# Module 05 — Run 02\n\nSix existing frozen checkpoints; the same 100 test inputs and Joint setups as Run 01. "
        "Three fresh repetitions of every problem. No training or new parameter search.\n\n"
        "Six methods: weight 1, Run-01-selected global fixed, Run-01-selected per-instance fixed, "
        "tuned periodic (2.5,1), periodic (1,3), frozen RL. Main comparison heatmaps contain only "
        "the global fixed, per-instance fixed, periodic (1,3) and RL. The (1,3) ordering is chosen "
        "from Run 01 before new timings, where it beat (3,1) on every seed.\n\n"
        "`status.json` and `supervisor.log` show progress. `analysis/summary.json` and the CSV files "
        "contain all reductions, repetitions, coverage and Run 01 comparisons. Figures and captions are "
        "under `analysis/paper_figures/`; `index.html` is the gallery.\n")
    first.emit(output, "Run 02 prepared; original inputs, setups and selected weights locked", executions=len(expected), workers=3)


def verify(output):
    first.verify_inputs(output)
    if first.file_hash(output / "selections.json") != first.read(output / "prepared.json")["selections_sha256"]:
        raise RuntimeError("Run 01 policy selections changed")


def preflight(output):
    verify(output)
    first.environment_check(output)
    protocol = first.read(output / "protocol.json")
    jobs = first.read(output / "jobs_preflight.json")
    audit = {"started_at": first.now(), "checks": [], "passed": False}
    for seed in protocol["training_seeds"]:
        bundle = first.load_bundle(output, seed)
        before = first.frozen_snapshot(bundle)
        job = next(j for j in jobs if j["seed"] == seed and j["case_id"] == 0)
        first.configure_smoother_profile(first.PROFILE)
        with first.create_env(**job["mkw"]) as env:
            env.prepare_rl(params=job["params"])
            if env.cycle_relax_types != (18, 18, 9):
                raise AssertionError("Primary smoother changed")
        names = ["fixed_1.00", "periodic_2.50_1.00", "periodic_1.00_3.00", "rl_frozen_lcb", "fixed_1.00"]
        results = [first.run_policy(job, name, protocol["policies"][name], bundle) for name in names]
        np.testing.assert_allclose(results[0]["outcome"]["cycle_residuals"], results[-1]["outcome"]["cycle_residuals"], rtol=1e-12, atol=1e-15)
        first.verify_frozen(bundle, before)
        audit["checks"].append({"seed": seed, "repeated_reference_agrees": True, "controller_frozen": True,
            "policies": [{"policy": r["policy"], "success": r["success"]} for r in results]})
        first.emit(output, "Run 02 native preflight passed", seed=seed)
    audit.update(passed=True, finished_at=first.now())
    first.dump(output / "preflight.json", audit)


def worker(output, worker_id, workers):
    if workers not in range(1, 4) or worker_id not in range(workers):
        raise ValueError("At most three single-thread workers")
    directory = output / "raw/test"
    with (directory / f"worker_{worker_id}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(output)
        protocol = first.read(output / "protocol.json")
        jobs = first.read(output / "jobs_test.json")[worker_id::workers]
        expected = expected_cells(jobs)
        path = directory / f"worker_{worker_id}.jsonl"
        done, elapsed = set(), 0.
        for row in first.read_records(path, repair_tail=True):
            key = first.cell_key(row)
            if key not in expected or key in done:
                raise AssertionError("Duplicate or foreign completed trial")
            done.add(key); elapsed += row["wall_sec"]
        state = {"worker": worker_id, "workers": workers, "pid": os.getpid(), "status": "running",
                 "started_at": first.now(), "done": len(done), "total": len(expected), "wall_sec": elapsed}
        progress = output / "progress" / f"test_{worker_id}.json"
        bundles, snapshots = {}, {}
        with path.open("a", buffering=1) as handle:
            for job in jobs:
                seed = job["seed"]
                if seed not in bundles:
                    bundles[seed] = first.load_bundle(output, seed)
                    snapshots[seed] = first.frozen_snapshot(bundles[seed])
                for name in job["policy_order"]:
                    if first.cell_key({**job, "policy": name}) in done:
                        continue
                    state.update(at=first.now(), seed=seed, source=SOURCE, case=job["case_id"]+1,
                                 repeat=job["repeat"]+1, policy=name, done=len(done), wall_sec=elapsed)
                    first.dump(progress, state)
                    row = first.run_policy(job, name, protocol["policies"][name], bundles[seed])
                    row.update(phase="test", repeat=job["repeat"], worker=worker_id, at=first.now(), run_number=2)
                    handle.write(json.dumps(first.ready(row), separators=(",", ":"), allow_nan=False)+"\n")
                    done.add(first.cell_key(row)); elapsed += row["wall_sec"]
                handle.flush(); os.fsync(handle.fileno())
                first.verify_frozen(bundles[seed], snapshots[seed])
        if done != expected:
            raise AssertionError("Incomplete worker")
        state.update(at=first.now(), status="complete", done=len(done), wall_sec=elapsed, frozen_audit_passed=True)
        first.dump(progress, state)
        first.dump(directory / f"worker_{worker_id}.complete.json", state)


def supervisor(output, workers=3):
    if workers not in range(1, 4):
        raise ValueError("At most three workers")
    with (output / "supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(output); first.environment_check(output)
        if not first.read(output / "preflight.json")["passed"]:
            raise RuntimeError("Native preflight must pass")
        if (output / "complete.json").exists():
            return
        prior_workers = {first.read(p)["workers"] for p in (output / "progress").glob("test_*.json")}
        if prior_workers and prior_workers != {workers}:
            raise ValueError("Keep the same worker count on resume")
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        awake = prevent_sleep()
        children, logs = [], []
        start = time.monotonic()
        total = len(expected_cells(first.read(output / "jobs_test.json")))
        state = {"run_number": 2, "started_at": first.now(), "at": first.now(), "status": "running", "phase": "test",
                 "pid": os.getpid(), "awake_pid": (awake.pid if awake is not None else None), "workers": workers, "phase_total": total,
                 "phase_done": 0, "output": str(output)}
        first.dump(output / "status.json", state)
        first.emit(output, "Run 02 started; all six policies freshly timed", executions=total, workers=workers, sleep_protection=awake is not None)
        try:
            for i in range(workers):
                log = (output / "raw/test" / f"worker_{i}.log").open("a")
                logs.append(log)
                children.append(subprocess.Popen([sys.executable, "-u", "-m", "experiments.archive.paper_development.run_05_policy_repeat", "worker",
                    "--output", str(output), "--worker-id", str(i), "--workers", str(workers)], cwd=ROOT, env=single_thread_environment(), stdout=log, stderr=subprocess.STDOUT))
            state["worker_pids"] = [c.pid for c in children]
            last_print = 0.
            while True:
                codes = [c.poll() for c in children]
                if any(c not in (None, 0) for c in codes):
                    raise RuntimeError(f"Worker stopped: {codes}; see raw/test/worker_*.log")
                progress = [first.read(p) for p in (output / "progress").glob("test_*.json")]
                done = sum(r["done"] for r in progress)
                elapsed = time.monotonic()-start
                state.update(at=first.now(), phase_done=done, worker_progress=progress, elapsed_sec=elapsed,
                             estimated_remaining_sec=elapsed*(total-done)/done if done else None)
                first.dump(output / "status.json", state)
                if time.monotonic()-last_print >= 30:
                    first.emit(output, "Run 02 progress", complete=done, total=total,
                               remaining_minutes=round(state["estimated_remaining_sec"]/60, 1) if done else None)
                    last_print = time.monotonic()
                if all(code == 0 for code in codes):
                    break
                time.sleep(5)
            for log in logs: log.close()
            logs = []
            for i in range(workers):
                audit = first.read(output / "raw/test" / f"worker_{i}.complete.json")
                if not audit["frozen_audit_passed"] or audit["done"] != audit["total"]:
                    raise AssertionError("Worker audit failed")
            first.dump(output / "raw/test/complete.json", {"at": first.now(), "trials": total,
                "raw_sha256": {str(p.relative_to(output)): first.file_hash(p) for p in (output / "raw/test").glob("worker_*.jsonl")}})
            state.update(phase="analysis", at=first.now(), estimated_remaining_sec=None)
            first.dump(output / "status.json", state)
            from experiments.archive.paper_development.analyze_05_policy_repeat import analyze
            from experiments.archive.paper_development.plot_05_policy_repeat import generate
            analyze(output)
            state.update(phase="figures", at=first.now())
            first.dump(output / "status.json", state)
            first.emit(output, "Native evaluations complete; producing Run 02 figures")
            generate(output)
            verify(output)
            state.update(status="complete", phase="complete", at=first.now(), elapsed_sec=time.monotonic()-start)
            first.dump(output / "complete.json", state); first.dump(output / "status.json", state)
            first.emit(output, "Run 02 complete; comparisons and figures saved")
        except BaseException as exc:
            state.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", at=first.now(), error=str(exc))
            first.dump(output / "status.json", state)
            first.emit(output, "Run 02 stopped; completed trials preserved", error=str(exc))
            raise
        finally:
            first.stop_children(children)
            for log in logs: log.close()
            stop_sleep_prevention(awake)


def watch(output):
    previous = None
    while True:
        if (output / "status.json").exists():
            s = first.read(output / "status.json")
            marker = (s["status"], s["phase"], s.get("phase_done"))
            if marker != previous:
                print(f"\nMODULE 05 — RUN 02 | six frozen seeds | 100 problems | Joint hierarchies | three repetitions\n"
                      f"{first.now()}  {s['status'].upper()} · {s['phase']} · {s.get('phase_done', 0):,}/{s['phase_total']:,}", flush=True)
                for w in sorted(s.get("worker_progress", []), key=lambda r:r["worker"]):
                    print(f"  Core worker {w['worker']+1}: seed {w.get('seed','-')} · repetition {w.get('repeat','-')}/3 · "
                          f"problem {w.get('case','-')} · {w.get('policy','-')} · {w['done']:,}/{w['total']:,}", flush=True)
                eta = s.get("estimated_remaining_sec")
                if eta is not None: print(f"Estimated native time remaining: {eta/60:.1f} min; figures follow automatically.", flush=True)
                print("Closing this viewer does not stop the experiment.", flush=True)
                if s.get("error"): print(s["error"], flush=True)
                previous = marker
            if s["status"] in ("complete", "failed", "interrupted"): return
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "preflight", "run", "worker", "watch", "analyze", "plot"])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent", type=Path, default=first.DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--worker-id", type=int)
    args = parser.parse_args(); output = args.output.resolve()
    if args.command == "prepare": prepare(output, args.parent)
    elif args.command == "preflight": preflight(output)
    elif args.command == "run": supervisor(output, args.workers)
    elif args.command == "worker": worker(output, args.worker_id, args.workers)
    elif args.command == "watch": watch(output)
    elif args.command == "analyze":
        from experiments.archive.paper_development.analyze_05_policy_repeat import analyze
        analyze(output)
    else:
        from experiments.archive.paper_development.plot_05_policy_repeat import generate
        generate(output)


if __name__ == "__main__": main()
