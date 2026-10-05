"""Paired serial timing of (2.85,1.10) and (2.90,1.10) on frozen Run 04 jobs."""
from __future__ import annotations

from experiments.runtime import configure_single_thread, prevent_sleep, stop_sleep_prevention

if __name__ == "__main__":
    configure_single_thread()

# Import first: the established runner sets all numerical library thread counts.
from experiments.archive.paper_development import run_05_policy as base

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np

PARENT = base.ROOT / "results/paper_final/05_policy/20260927_run04_anchored26_joint_6seeds_100cases"
OUTPUT = base.ROOT / "results/paper_final/05_policy/20260928_pair285_vs290_joint_6seeds_100cases"
NAMES = ("periodic_2.85_1.10", "periodic_2.90_1.10")
SPECS = {name: {"kind": "periodic", "pattern": [weight, 1.1]}
         for name, weight in zip(NAMES, (2.85, 2.9))}
REPEATS = 4
ORDER_SEED = 2026092805
FIELDS = ("native_continuation_sec", "inclusive_continuation_sec",
          "native_total_sec", "inclusive_total_sec", "wall_sec")


def prepare(output):
    if output.exists() and any(output.iterdir()):
        raise ValueError("This comparison requires a new empty output directory")
    base.verify_inputs(PARENT, check_source=False)
    assert base.read(PARENT / "complete.json")["status"] == "complete"
    old_manifest = base.read(PARENT / "source_manifest.json")
    changes = base.source_differences(old_manifest)
    numerical_changes = [p for p in changes if not any(
        s in p for s in ("plot_", "analyze_", "test_", "module05_figures/"))]
    if numerical_changes:
        raise RuntimeError(f"Run 04 execution sources changed: {numerical_changes}")
    jobs = base.read(PARENT / "jobs_base_test.json")
    assert len(jobs) == 600
    assert {(j["seed"], j["case_id"]) for j in jobs} == {
        (s, c) for s in range(1, 7) for c in range(100)}
    assert all(j["source"] == "bandit_lstdq" for j in jobs)
    for c in range(100):
        assert len({j["input_id"] for j in jobs if j["case_id"] == c}) == 1
    rng = random.Random(ORDER_SEED)
    first_at_even = {}
    for seed in range(1, 7):
        cases = list(range(100))
        rng.shuffle(cases)
        first_at_even.update({(seed, case): i % 2 for i, case in enumerate(cases)})
    pairs = []
    for repeat in range(REPEATS):
        order = list(jobs)
        rng.shuffle(order)
        for job in order:
            first = (first_at_even[job["seed"], job["case_id"]] + repeat) % 2
            pairs.append({**job, "repeat": repeat, "pair_index": len(pairs),
                          "policy_order": [NAMES[first], NAMES[1 - first]]})
    for seed in range(1, 7):
        for repeat in range(REPEATS):
            assert Counter(p["policy_order"][0] for p in pairs
                           if p["seed"] == seed and p["repeat"] == repeat) == dict.fromkeys(NAMES, 50)
    output.mkdir(parents=True)
    shutil.copy2(PARENT / "jobs_base_test.json", output / "jobs_base_test.json")
    shutil.copy2(PARENT / "jobs_preflight.json", output / "jobs_preflight.json")
    shutil.copy2(PARENT / "inputs.json", output / "inputs.json")
    base.dump(output / "jobs_pairs.json", pairs)
    base.dump(output / "protocol.json", {
        "created_at": base.now(), "parent": str(PARENT), "comparison": "2.85 versus 2.90; second coefficient 1.10",
        "scope": "Exploratory paired comparison on previously used test cases; not the full Run 05",
        "policies": SPECS, "repetitions": REPEATS, "training_seeds": list(range(1, 7)),
        "cases": 100, "pairs": len(pairs), "timed_solves": 2 * len(pairs),
        "order_seed": ORDER_SEED,
        "ordering": "Random case order each round; adjacent A/B pair; first policy reversed each round and exactly balanced within every seed/round",
        "parallelism": "One serial worker, one MPI rank, one thread per numerical library; no claimed core affinity",
        "numerics": "Same input, RHS, hierarchy parameters, fresh initial state and rebuild per solve as Run 04",
        "cycle_semantics": "High weight first, reset every solve; alternate complete V-cycles; one pre/post sweep each",
        "tolerance": 1e-6, "cycle_cap": 50,
        "recovery": "Unchanged run_05_policy.run_policy/run_02_diagnostics.run_schedule fallback and cost accounting",
        "cost": "Primary native continuation excludes common initial setup, includes any recovery setup/solve; inclusive adds schedule dispatch",
        "analysis": "Ratio of aggregate mean times, per-checkpoint and per-round results; paired bootstrap over 100 shared PDE cases conditional on six frozen checkpoints",
        "warmup": "ABBA on one separate preflight input per checkpoint; excluded from timed data",
        "source_changes_since_parent": changes,
    })
    relative = str(Path(__file__).resolve().relative_to(base.ROOT))
    manifest = {p: base.file_hash(base.ROOT / p) for p in set(old_manifest) | {relative} | base.support_source_paths()}
    base.dump(output / "source_manifest.json", manifest)
    shutil.copy2(__file__, output / Path(__file__).name)
    prepared = {"parent_prepared_sha256": base.file_hash(PARENT / "prepared.json"),
                "files": {p.name: base.file_hash(p) for p in output.iterdir() if p.is_file()}}
    base.dump(output / "prepared.json", prepared)


def verify(output):
    prepared = base.read(output / "prepared.json")
    assert base.file_hash(PARENT / "prepared.json") == prepared["parent_prepared_sha256"]
    base.verify_inputs(PARENT, check_source=False)
    for name, digest in prepared["files"].items():
        assert base.file_hash(output / name) == digest, name
    for name, digest in base.read(output / "source_manifest.json").items():
        if not (base.ROOT/name).is_file() or base.file_hash(base.ROOT/name) != digest:
            raise RuntimeError(f"Experiment source changed after preparation: {name}")


def audit(row):
    base.audit_outcome(row["outcome"])
    outcome = row["outcome"]
    actions = outcome.get("cycle_actions", [])
    pattern = SPECS[row["policy"]]["pattern"]
    assert actions == [pattern[i % 2] for i in range(len(actions))]
    assert not outcome.get("controller_update_committed", False)
    if row["success"]:
        assert np.isfinite(outcome["completed_residual_norm"])
        assert outcome["completed_residual_norm"] < 1e-6
    assert np.isclose(row["native_continuation_sec"], outcome["solve_runtime"] + outcome.get("fallback_setup_runtime", 0), rtol=1e-12)
    assert np.isclose(row["inclusive_continuation_sec"], row["native_continuation_sec"] + outcome.get("infer_runtime", 0), rtol=1e-12)


def preflight(output):
    base.configure_smoother_profile(base.PROFILE)
    jobs = base.read(output / "jobs_preflight.json")
    rows = []
    for seed in range(1, 7):
        job = next(j for j in jobs if j["seed"] == seed and j["case_id"] == 0)
        with base.create_env(**job["mkw"]) as env:
            env.prepare_rl(params=job["params"])
            assert env.cycle_relax_types == (18, 18, 9)
        repeated = []
        for name in (NAMES[0], NAMES[1], NAMES[1], NAMES[0]):
            row = base.run_policy(job, name, SPECS[name], None)
            audit(row)
            repeated.append(row)
        for a, b in ((0, 3), (1, 2)):
            assert repeated[a]["success"] == repeated[b]["success"]
            np.testing.assert_allclose(repeated[a]["outcome"]["cycle_residuals"],
                                       repeated[b]["outcome"]["cycle_residuals"], rtol=1e-12, atol=1e-15)
        rows.extend(repeated)
    base.dump(output / "preflight.json", {"passed": True, "at": base.now(), "excluded_from_analysis": True, "rows": rows})


def analyze(output):
    rows = list(base.read_records(output / "raw.jsonl"))
    jobs = base.read(output / "jobs_pairs.json")
    expected = {(p["seed"], p["case_id"], p["repeat"], n) for p in jobs for n in NAMES}
    seen = {}
    values = np.full((6, 100, REPEATS, 2, len(FIELDS)), np.nan)
    cycles = np.zeros((6, 100, REPEATS, 2), dtype=int)
    identity = {(p["seed"], p["case_id"], p["repeat"]): p for p in jobs}
    for row in rows:
        audit(row)
        seed, case, repeat, name = row["seed"], row["case_id"], row["repeat"], row["policy"]
        key = seed, case, repeat, name
        assert key in expected and key not in seen
        job = identity[seed, case, repeat]
        assert all(row[k] == job[k] for k in ("input_id", "hierarchy_id", "source", "pair_index"))
        assert job["policy_order"][row["position"]] == name
        seen[key] = row
        idx = seed - 1, case, repeat, NAMES.index(name)
        values[idx] = [row[f] for f in FIELDS]
        cycles[idx] = len(row["outcome"]["cycle_actions"])
    assert set(seen) == expected and np.all(np.isfinite(values))
    np.testing.assert_array_equal(cycles, np.broadcast_to(cycles[:, :, :1, :], cycles.shape))
    trace_pairs = 0
    for seed in range(1, 7):
        for case in range(100):
            for name in NAMES:
                initial = seen[seed, case, 0, name]["outcome"]["cycle_residuals"]
                for repeat in range(1, REPEATS):
                    np.testing.assert_allclose(initial, seen[seed, case, repeat, name]["outcome"]["cycle_residuals"], rtol=1e-12, atol=1e-15)
                    trace_pairs += 1

    def group_summary(array, counts):
        mean = array.reshape(-1, 2, len(FIELDS)).mean(axis=0)
        return {"mean_ms": {n: {f: float(mean[i, j] * 1000) for j, f in enumerate(FIELDS)} for i, n in enumerate(NAMES)},
                "reduction_290_vs_285_pct": {f: float(100 * (1 - mean[1, j] / mean[0, j])) for j, f in enumerate(FIELDS)},
                "mean_primary_cycles": {n: float(counts.reshape(-1, 2).mean(axis=0)[i]) for i, n in enumerate(NAMES)}}

    summary = {"at": base.now(), "timed_solves": len(rows), "matched_pairs": len(jobs),
               "overall": group_summary(values, cycles),
               "per_seed": {str(s + 1): group_summary(values[s], cycles[s]) for s in range(6)},
               "per_round": {str(r): group_summary(values[:, :, r], cycles[:, :, r]) for r in range(REPEATS)},
               "failures": {n: sum(not r["success"] for r in rows if r["policy"] == n) for n in NAMES},
               "recoveries": {n: sum(bool(r["outcome"].get("fallback_used")) for r in rows if r["policy"] == n) for n in NAMES},
               "audit": {"full_coverage": True, "balanced_order": True, "schedule_trace_matches": True,
                         "deterministic_repeated_cycles": True, "residual_trace_comparisons": trace_pairs,
                         "parent_and_execution_source_hashes_verified": True}}
    delta = cycles[:, :, 0, 1] - cycles[:, :, 0, 0]
    summary["cycle_comparison_600_cases"] = {"290_fewer": int((delta < 0).sum()),
        "equal": int((delta == 0).sum()), "285_fewer": int((delta > 0).sum())}
    # Resample shared PDE case IDs together across all six checkpoint hierarchies.
    rng = np.random.default_rng(2026092806)
    resampled = rng.integers(0, 100, size=(10000, 100))
    case_means = values.mean(axis=(0, 2))
    samples = case_means[resampled].mean(axis=1)
    reductions = 100 * (1 - samples[:, 1] / samples[:, 0])
    summary["paired_case_bootstrap_95_pct"] = {
        f: np.quantile(reductions[:, j], [.025, .975]).tolist() for j, f in enumerate(FIELDS)}
    summary["bootstrap_scope"] = "10,000 paired resamples of 100 shared input IDs; six checkpoints and four observed timing rounds fixed; not a hardware-noise or new-training-seed interval"
    summary["order_groups"] = {}
    for first in NAMES:
        grouped = [p for p in jobs if p["policy_order"][0] == first]
        means = np.asarray([[seen[p["seed"], p["case_id"], p["repeat"], n]["native_continuation_sec"] for n in NAMES] for p in grouped]).mean(axis=0)
        summary["order_groups"][first + "_first"] = {"pairs": len(grouped), "mean_native_ms": (means * 1000).tolist(),
                                                    "reduction_290_vs_285_pct": float(100 * (1 - means[1] / means[0]))}
    base.dump(output / "summary.json", summary)
    table = []
    for seed in range(6):
        for case in range(100):
            record = {"seed": seed + 1, "case_id": case}
            for i, n in enumerate(NAMES):
                record[n + "_primary_cycles"] = int(cycles[seed, case, 0, i])
                for j, f in enumerate(FIELDS):
                    record[n + "_" + f] = float(values[seed, case, :, i, j].mean())
            table.append(record)
    base.write_csv(output / "case_means.csv", table)
    overall = summary["overall"]
    lines = ["# Paired comparison: (2.85,1.10) versus (2.90,1.10)", "",
             "100 diffusion 60^3 inputs x six frozen Run 04 checkpoint-selected hierarchies x four repetitions per schedule = 4,800 timed solves.", "",
             "One serial worker; consecutive matched pairs; balanced and reversed A/B order; 24 separate warm-up solves excluded. Exact Run 04 solver, tolerance, cap, recovery and cost accounting. This is exploratory evaluation on previously used test cases, not the full Run 05.", "",
             "| Schedule | Mean native continuation (ms) | Including schedule dispatch (ms) | Mean primary cycles |", "|---|---:|---:|---:|"]
    for name in NAMES:
        m = overall["mean_ms"][name]
        lines.append(f"| {SPECS[name]['pattern']} | {m[FIELDS[0]]:.6f} | {m[FIELDS[1]]:.6f} | {overall['mean_primary_cycles'][name]:.6f} |")
    lines.extend(["", "Positive percentage reductions below mean (2.90,1.10) is faster, with (2.85,1.10) as denominator.", "",
                  f"Aggregate native reduction: {overall['reduction_290_vs_285_pct'][FIELDS[0]]:.4f}%.",
                  f"Aggregate controller-inclusive reduction: {overall['reduction_290_vs_285_pct'][FIELDS[1]]:.4f}%.",
                  f"Paired case-bootstrap 95% interval for native reduction: {summary['paired_case_bootstrap_95_pct'][FIELDS[0]]} percent.",
                  summary["bootstrap_scope"] + ".", "",
                  "| Frozen checkpoint | Native reduction (%) | Inclusive reduction (%) |", "|---|---:|---:|"])
    for seed, result in summary["per_seed"].items():
        r = result["reduction_290_vs_285_pct"]
        lines.append(f"| {seed} | {r[FIELDS[0]]:.4f} | {r[FIELDS[1]]:.4f} |")
    lines.extend(["", "Timing rounds (native percentage reduction): " + ", ".join(
        f"{r['reduction_290_vs_285_pct'][FIELDS[0]]:.4f}" for r in summary["per_round"].values()) + ".", "",
        f"Cycle comparison over 600 checkpoint/input pairs: {summary['cycle_comparison_600_cases']}.",
        f"Unrecovered failures: {summary['failures']}; recoveries: {summary['recoveries']}.", "",
        "All coverage, cost, prescribed action, repeated numerical trace, source, input and checkpoint hash checks passed.", "",
        "The weighted polynomial minimax objective still favors (2.85,1.10). Measured full multilevel runtime is a different objective, and a winner selected on this comparison is empirically selected on these cases.", "",
        "See protocol.json, environment.json, prepared.json, source_manifest.json, raw.jsonl, case_means.csv and summary.json for reproducibility.", ""])
    (output / "REPORT.md").write_text("\n".join(lines))
    return summary


def run(output):
    if not (output / "prepared.json").exists():
        prepare(output)
    with (output / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "complete.json").exists():
            raise ValueError("Comparison already complete")
        if (output / "raw.jsonl").exists():
            raise ValueError("Partial comparison exists; refusing to mix timing sessions")
        start = time.monotonic()
        awake = prevent_sleep()
        try:
            verify(output)
            environment = base.environment_check(output)
            environment["parallelism"] = base.read(output / "protocol.json")["parallelism"]
            base.dump(output / "environment.json", environment)
            preflight(output)
            base.emit(output, "Preflight passed; paired timing starts", solves=4800)
            jobs = base.read(output / "jobs_pairs.json")
            with (output / "raw.jsonl").open("x", buffering=1) as handle:
                for job in jobs:
                    # No raw-file writes or background analysis between the paired solves.
                    rows = []
                    for position, name in enumerate(job["policy_order"]):
                        row = base.run_policy(job, name, SPECS[name], None)
                        row.update(repeat=job["repeat"], pair_index=job["pair_index"], position=position, at=base.now())
                        rows.append(row)
                    for row in rows:
                        audit(row)
                        handle.write(json.dumps(base.ready(row), separators=(",", ":"), allow_nan=False) + "\n")
                    done = 2 * (job["pair_index"] + 1)
                    if done % 20 == 0:
                        handle.flush()
                        os.fsync(handle.fileno())
                        elapsed = time.monotonic() - start
                        base.dump(output / "status.json", {"status": "running", "phase": "paired timing", "at": base.now(),
                            "pid": os.getpid(), "done": done, "total": 2 * len(jobs), "round": job["repeat"], "elapsed_sec": elapsed,
                            "estimated_remaining_sec": elapsed * (2 * len(jobs) - done) / done})
                    if done % 1200 == 0:
                        base.emit(output, "Timing round complete", round=job["repeat"], done=done)
                handle.flush()
                os.fsync(handle.fileno())
            verify(output)
            result = analyze(output)
            final = {"status": "complete", "at": base.now(), "elapsed_sec": time.monotonic() - start,
                     "trials": result["timed_solves"], "raw_sha256": base.file_hash(output / "raw.jsonl"),
                     "summary_sha256": base.file_hash(output / "summary.json")}
            base.dump(output / "complete.json", final)
            base.dump(output / "status.json", final)
            base.emit(output, "Paired comparison complete", overall=result["overall"])
        except BaseException as exc:
            base.dump(output / "status.json", {"status": "failed", "at": base.now(), "error": repr(exc)})
            raise
        finally:
            stop_sleep_prevention(awake)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "analyze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        run(args.output)
    else:
        verify(args.output)
        analyze(args.output)
