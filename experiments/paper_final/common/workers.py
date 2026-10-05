"""Durable worker execution shared by frozen matched-hierarchy studies."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

from experiments.runtime import single_thread_environment
from . import artifacts, frozen_policy, policy_analysis as analysis
from .artifacts import ROOT


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    for child in children:
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run_frozen_worker(output, phase, worker_id, workers, *, run_number, verify, policy):
    if workers not in range(1, 4) or worker_id not in range(workers):
        raise ValueError("At most three workers")
    directory = output / "raw" / phase
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"worker_{worker_id}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify(output)
        protocol = artifacts.read(output / "protocol.json")
        jobs = artifacts.read(output / f"jobs_{phase}.json")[worker_id::workers]
        expected = analysis.expected_cells(jobs)
        path = directory / f"worker_{worker_id}.jsonl"
        done = set()
        wall = 0.0
        for row in artifacts.read_records(path, repair_tail=True):
            key = analysis.cell_key(row)
            if key not in expected or key in done:
                raise ValueError("Duplicate/foreign trial")
            done.add(key)
            wall += row["wall_sec"]
        state = {
            "phase": phase,
            "worker": worker_id,
            "workers": workers,
            "pid": os.getpid(),
            "done": len(done),
            "total": len(expected),
            "status": "running",
        }
        bundles = {}
        snapshots = {}
        with path.open("a", buffering=1) as handle:
            for job in jobs:
                seed = job["seed"]
                if seed not in bundles:
                    bundles[seed] = policy.load_bundle(output, seed)
                    snapshots[seed] = policy.frozen_snapshot(bundles[seed])
                for name in job["policy_order"]:
                    key = analysis.cell_key({**job, "policy": name})
                    if key in done:
                        continue
                    state.update(
                        at=artifacts.now(),
                        seed=seed,
                        case=job["case_id"] + 1,
                        policy=name,
                        done=len(done),
                        wall_sec=wall,
                    )
                    artifacts.dump(
                        output / "progress" / f"{phase}_{worker_id}.json", state
                    )
                    row = policy.run_policy(
                        job, name, protocol["policies"][name], bundles[seed]
                    )
                    row.update(
                        phase=phase,
                        repeat=job["repeat"],
                        worker=worker_id,
                        at=artifacts.now(),
                        run_number=run_number,
                    )
                    handle.write(
                        json.dumps(
                            artifacts.ready(row), separators=(",", ":"), allow_nan=False
                        )
                        + "\n"
                    )
                    done.add(key)
                    wall += row["wall_sec"]
                handle.flush()
                os.fsync(handle.fileno())
                policy.verify_frozen(bundles[seed], snapshots[seed])
        if done != expected:
            raise AssertionError("Incomplete worker")
        state.update(
            status="complete",
            done=len(done),
            wall_sec=wall,
            at=artifacts.now(),
            frozen_audit_passed=True,
        )
        artifacts.dump(output / "progress" / f"{phase}_{worker_id}.json", state)
        artifacts.dump(directory / f"worker_{worker_id}.complete.json", state)


def run_frozen_phase(output, phase, *, runner_module, run_number, workers=3):
    jobs = artifacts.read(output / f"jobs_{phase}.json")
    expected = analysis.expected_cells(jobs)
    children = []
    logs = []
    start = time.monotonic()
    try:
        for i in range(workers):
            log = (output / "raw" / phase / f"worker_{i}.log").open("a", buffering=1)
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        runner_module,
                        "worker",
                        "--output",
                        str(output),
                        "--phase",
                        phase,
                        "--worker",
                        str(i),
                    ],
                    cwd=ROOT,
                    env=single_thread_environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        while True:
            codes = [c.poll() for c in children]
            if any(c not in (None, 0) for c in codes):
                raise RuntimeError(f"{phase} worker stopped: {codes}")
            progress = [
                artifacts.read(p) for p in (output / "progress").glob(f"{phase}_*.json")
            ]
            done = sum(p["done"] for p in progress)
            elapsed = time.monotonic() - start
            artifacts.dump(
                output / "status.json",
                {
                    "status": "running",
                    "phase": phase,
                    "run_number": run_number,
                    "at": artifacts.now(),
                    "done": done,
                    "total": len(expected),
                    "elapsed_sec": elapsed,
                    "estimated_phase_remaining_sec": elapsed
                    * (len(expected) - done)
                    / done
                    if done
                    else None,
                    "worker_pids": [p.pid for p in children],
                },
            )
            if all(c == 0 for c in codes):
                break
            time.sleep(5)
        seen = set()
        for i in range(workers):
            marker = artifacts.read(
                output / "raw" / phase / f"worker_{i}.complete.json"
            )
            if marker["done"] != marker["total"] or not marker["frozen_audit_passed"]:
                raise AssertionError("Worker audit failed")
        for p in (output / "raw" / phase).glob("worker_*.jsonl"):
            for row in artifacts.read_records(p):
                key = analysis.cell_key(row)
                if key in seen:
                    raise ValueError("Duplicate trial")
                frozen_policy.audit_outcome(row["outcome"])
                seen.add(key)
        if seen != expected:
            raise AssertionError("Phase coverage mismatch")
        marker = {
            "at": artifacts.now(),
            "trials": len(seen),
            "elapsed_sec": time.monotonic() - start,
            "raw_sha256": {
                str(p.relative_to(output)): artifacts.file_hash(p)
                for p in (output / "raw" / phase).glob("worker_*.jsonl")
            },
        }
        artifacts.dump(output / "raw" / phase / "complete.json", marker)
        artifacts.emit(
            output, f"Run {run_number:02d} {phase} complete", trials=len(seen)
        )
    finally:
        stop_children(children)
        for log in logs:
            log.close()
