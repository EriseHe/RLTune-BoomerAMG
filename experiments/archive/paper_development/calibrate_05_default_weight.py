"""Select Module 05's constant on separate Default-setup development inputs."""
from __future__ import annotations

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

from experiments.archive.paper_development import run_05_policy as base

import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np

from hypre.bindings.config import configure_smoother_profile
from problems.streams import generate_scalar_anisotropic_diffusion_instances
from setup.space import DEFAULT_SETUP_PARAMS

ROOT = base.ROOT
DEFAULT_OUTPUT = ROOT / "results/paper_final/05_online_policies/20260928_shared_prefix/default_calibration"
WEIGHTS = tuple(round(1 + .05*i, 2) for i in range(41))
CASES = 100
INPUT_SEED = 92805101
ORDER_SEED = 92805103


def source_hashes():
    paths = [Path(__file__), ROOT/"experiments/archive/paper_development/run_05_policy.py",
             ROOT/"experiments/archive/paper_development/run_02_diagnostics.py",
             ROOT/"experiments/joint/solve_control/setup_aware_compare_common.py"]
    paths += list((ROOT/"hypre/bindings").rglob("*.py"))
    paths += list((ROOT/"problems").rglob("*.py"))
    return {str(p.relative_to(ROOT)): base.file_hash(p) for p in paths}


def prepare(output):
    if (output/"protocol.json").exists():
        return
    output.mkdir(parents=True, exist_ok=True)
    env = base.environment_check(output)
    env["parallelism"] = "one process, one MPI rank, one library thread; serial solves"
    base.dump(output/"environment.json", env)
    inputs = [{"case_id": i, "mkw": m, "context": c.tolist(), "input_id": base.digest(m)}
              for i, (m,c) in enumerate(generate_scalar_anisotropic_diffusion_instances(
                  count=CASES, seed=INPUT_SEED, grid_choices=[(60,60,60)], c_min=1, c_max=1000))]
    base.dump(output/"inputs.json", inputs)
    protocol = {
        "purpose": "Default-setup development calibration; freeze a single constant before Module 05 training",
        "weights": list(WEIGHTS), "cases": CASES, "repetitions": 1,
        "input_seed": INPUT_SEED, "order_seed": ORDER_SEED,
        "setup_params": DEFAULT_SETUP_PARAMS, "smoother_profile": base.PROFILE,
        "tol": 1e-6, "cap": 50, "sweeps_down": 1, "sweeps_up": 1,
        "selection_rule": "Among weights completing every input, minimize mean inclusive continuation cost; tie: smaller weight",
        "selection_cost": "primary native solve + policy dispatch + all recovery setup, solve and dispatch; common initial setup excluded",
        "failure_rule": "unchanged Default setup / W1 recovery; never reward a fast unrecovered failure",
        "ordering": "serial, independently permuted weights within each input",
        "hierarchy": "rebuild the same Default hierarchy configuration for each weight",
        "input_sha256": base.file_hash(output/"inputs.json"),
        "source_hashes": source_hashes(), "at": base.now(),
        "label": "Default-setup development-selected fixed weight",
    }
    base.dump(output/"protocol.json", protocol)


def summarize(rows):
    ranking = []
    for w in WEIGHTS:
        selected = [r for r in rows if r["weight"] == w]
        if len(selected) != CASES or len({r["case_id"] for r in selected}) != CASES:
            raise ValueError("Incomplete or duplicated Default calibration")
        ranking.append({"weight": w, "completed": sum(r["success"] for r in selected),
                        "mean_inclusive_continuation_sec": float(np.mean([r["inclusive_continuation_sec"] for r in selected])),
                        "mean_native_continuation_sec": float(np.mean([r["native_continuation_sec"] for r in selected])),
                        "fallbacks": sum(r["outcome"]["fallback_used"] for r in selected)})
    ranking.sort(key=lambda r: (-r["completed"], r["mean_inclusive_continuation_sec"], r["weight"]))
    if ranking[0]["completed"] != CASES:
        raise RuntimeError("No constant completed the full development panel; do not lock a weight")
    return {"selected_weight": ranking[0]["weight"], "ranking": ranking,
            "trials": len(rows), "distinct_development_inputs": CASES,
            "inclusive_method_seconds": sum(r["outcome"]["end_to_end_runtime"] for r in rows),
            "sum_method_wall_seconds": sum(r["wall_sec"] for r in rows),
            "primary_comparison_includes_initial_setup": False,
            "training_or_test_outcomes_used": False}


def run(output):
    prepare(output)
    if (output/"selection.json").exists():
        raise FileExistsError("Calibration already selected; refusing to repeat or retune")
    protocol = base.read(output/"protocol.json")
    if base.file_hash(output/"inputs.json") != protocol["input_sha256"]:
        raise ValueError("Development inputs changed")
    if source_hashes() != protocol["source_hashes"]:
        raise ValueError("Calibration implementation changed")
    configure_smoother_profile(base.PROFILE)
    rng = np.random.default_rng(ORDER_SEED)
    jobs = [(r, float(w), rank) for r in base.read(output/"inputs.json")
            for rank, w in enumerate(rng.permutation(WEIGHTS))]
    started = time.perf_counter()
    with (output/".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = list(base.read_records(output/"raw.jsonl", repair_tail=True)) if (output/"raw.jsonl").exists() else []
        done = {(r["case_id"], r["weight"]) for r in rows}
        if len(done) != len(rows):
            raise ValueError("Duplicated calibration records")
        with (output/"raw.jsonl").open("a") as handle:
            for inp, weight, rank in jobs:
                if (inp["case_id"], weight) in done:
                    continue
                tick = time.perf_counter()
                outcome = base.run_schedule(inp["mkw"], DEFAULT_SETUP_PARAMS, [weight])
                wall = time.perf_counter() - tick
                base.audit_outcome(outcome)
                native = outcome["solve_runtime"] + outcome["fallback_setup_runtime"]
                row = {"case_id": inp["case_id"], "input_id": inp["input_id"], "weight": weight,
                       "execution_rank": rank, "success": not outcome["unrecovered_failure"],
                       "native_continuation_sec": native,
                       "inclusive_continuation_sec": native + outcome["infer_runtime"],
                       "outcome": outcome, "wall_sec": wall, "at": base.now()}
                handle.write(json.dumps(base.ready(row), allow_nan=False)+"\n")
                rows.append(row)
                if rank == 40:
                    handle.flush()
                    progress = {"completed_inputs": inp["case_id"]+1, "total_inputs": CASES, "trials": len(rows)}
                    base.dump(output/"progress.json", progress)
                    print(json.dumps(progress), flush=True)
        result = summarize(rows)
        result.update(at=base.now(), protocol_sha256=base.file_hash(output/"protocol.json"),
                      raw_sha256=base.file_hash(output/"raw.jsonl"), session_elapsed_sec=time.perf_counter()-started,
                      label=protocol["label"])
        base.dump(output/"selection.json", result)
        print(json.dumps({"selected_weight": result["selected_weight"], "trials": len(rows)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    (run if args.run else prepare)(args.output.resolve())


if __name__ == "__main__":
    main()
