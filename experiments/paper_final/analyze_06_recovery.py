"""Audit Module 06 workers against the protocol recorded with their results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments.paper_final.common.artifacts import dump, file_hash, read, write_csv
from experiments.paper_final.recovery.protocol import recorded_variants


def audit(records, variant, *, feedback_mode="attempt_2T"):
    if feedback_mode not in ("attempt_2T", "adjacent_delayed", "suffix_delayed"):
        raise ValueError("Unknown setup feedback mode")
    for index, record in enumerate(records):
        if record["problem_index"] != index + 1:
            raise ValueError("Incomplete or reordered problem stream")
        out = record["outcome"]
        attempts = out["primary_attempts"]
        if not 1 <= len(attempts) <= variant.attempts:
            raise ValueError("Learned attempt budget violated")
        if any(row["status"] == "success" for row in attempts[:-1]):
            raise ValueError("Retried after success")
        if not variant.unified and any(row["status"] != "setup_failure" for row in attempts[:-1]):
            raise ValueError("Original policy retried a solve failure")
        if variant.unified and out["fallback_used"] and len(attempts) != variant.attempts:
            raise ValueError("Fallback preceded exhaustion of the shared budget")
        if out["fallback_used"] and attempts[-1]["status"] == "success":
            raise ValueError("Fallback followed learned success")
        if out["fallback_used"] and not variant.default_fallback:
            raise ValueError("Default fallback exceeds the total attempt budget")
        if variant.default_fallback and attempts[-1]["status"] != "success" and not out["fallback_used"]:
            raise ValueError("Missing required default fallback after learned failure")
        committed = not out["failed"]
        if not out["bandit_update_committed"] or out["bandit_observation_count"] != len(attempts):
            raise ValueError("Every valid learned attempt must update the setup model")
        if out["failure_feedback_mode"] != feedback_mode:
            raise ValueError("Setup feedback differs from the requested protocol")
        for i, attempt in enumerate(attempts):
            expected_target = attempt["attempt_cost"]
            if attempt["status"] != "success":
                if feedback_mode == "attempt_2T":
                    expected_target *= 2
                elif feedback_mode == "suffix_delayed":
                    expected_target += sum(a["attempt_cost"] for a in attempts[i+1:])
                    expected_target += sum(out[f"fallback_{part}_runtime"] for part in ("setup", "solve", "controller"))
                elif i+1 < len(attempts):
                    expected_target += attempts[i+1]["attempt_cost"]
                else:
                    expected_target += sum(out[f"fallback_{part}_runtime"] for part in ("setup", "solve", "controller"))
            if attempt["attempt_cost"] < 0 or not math.isfinite(expected_target):
                raise ValueError("Invalid setup target")
            if not math.isclose(attempt["learning_cost"], expected_target, rel_tol=1e-12):
                raise ValueError("Setup target does not match the requested attempt-cost rule")
        if not committed and out["controller_update_committed"]:
            raise ValueError("Failed problem retained RL updates")
        if variant.unified and len({a["arm_index"] for a in attempts}) != len(attempts):
            raise ValueError("Repeated exact setup within a problem")
        native = sum(a["setup_runtime"] + a["solve_runtime"] for a in attempts)
        native += out["fallback_setup_runtime"] + out["fallback_solve_runtime"]
        controller = sum(a["controller_runtime"] for a in attempts)
        controller += out["fallback_controller_runtime"] + out.get("case_learning_runtime", 0.0)
        expected = native + controller + out["bandit_overhead_runtime"]
        for actual, wanted in ((out["runtime"], native), (out["infer_runtime"], controller),
                               (out["end_to_end_runtime"], expected)):
            if abs(actual - wanted) > 1e-8 * max(1, abs(wanted)):
                raise ValueError("Attempt/recovery timing accounting does not close")
        if variant.unified and committed:
            suffix = out["fallback_setup_runtime"] + out["fallback_solve_runtime"]
            for attempt in reversed(attempts):
                expected_cost = suffix if (variant.rl_recovery_cost
                    and attempt["status"] != "success"
                    and attempt["rl_episode_recorded"]) else 0.0
                if abs(attempt["rl_recovery_cost"] - expected_cost) > 1e-9:
                    raise ValueError("RL recovery target differs from executed subsequent native work")
                suffix += attempt["setup_runtime"] + attempt["solve_runtime"]


def summarize(records):
    outcomes = [r["outcome"] for r in records]
    return dict(
        cases=len(outcomes),
        failures=sum(bool(o["failed"]) for o in outcomes),
        first_attempt_failures=sum(o["first_primary_status"] != "success" for o in outcomes),
        learned_recoveries=sum(bool(o["learned_recovered"]) for o in outcomes),
        fallback=sum(bool(o["fallback_used"]) for o in outcomes),
        learned_attempts=sum(o["primary_attempt_count"] for o in outcomes),
        native_sec=sum(o["runtime"] for o in outcomes),
        total_sec=sum(o["end_to_end_runtime"] for o in outcomes),
        wall_sec=sum(o["method_wall_runtime"] for o in outcomes),
        replay_sec=sum(o.get("target_replay_runtime", 0.0) for o in outcomes),
    )


def analyze(output: Path):
    protocol = read(output / "protocol.json")
    variants = recorded_variants(protocol)
    records = {}
    results = {}
    for variant in variants:
        directory = output / variant.name
        result = read(directory / "result.json")
        if not result["complete"] or result["source_sha256"] != read(output / "source_manifest.json"):
            raise ValueError("Incomplete worker or mismatched source")
        if file_hash(directory / "trajectory.jsonl") != result["trajectory_sha256"]:
            raise ValueError("Trajectory hash changed")
        data = [json.loads(line) for line in (directory / "trajectory.jsonl").read_text().splitlines()]
        expected = protocol["smoke_cases"] or protocol["cases"]
        if len(data) != expected:
            raise ValueError("Incomplete stream")
        audit(data, variant)
        records[variant.name] = data
        results[variant.name] = result
    reference = records[variants[0].name]
    for data in records.values():
        if any((a["mkw"], a["context"]) != (b["mkw"], b["context"]) for a, b in zip(reference, data)):
            raise ValueError("Unpaired matrix/context inputs")
    common = [i for i in range(len(reference))
              if all(not data[i]["outcome"]["failed"] for data in records.values())]
    table = []
    for variant in variants:
        data = records[variant.name]
        table.append(dict(variant=variant.name, label=variant.label, **summarize(data),
                          common_success_cases=len(common),
                          common_success_native_sec=sum(data[i]["outcome"]["runtime"] for i in common),
                          common_success_total_sec=sum(data[i]["outcome"]["end_to_end_runtime"] for i in common)))
    windows = {"all": table, "last_1000": [], "rl_active": []}
    activation = 1 if protocol["smoke_cases"] else 1000
    for variant in variants:
        data = records[variant.name]
        windows["last_1000"].append(dict(variant=variant.name, **summarize(data[-1000:])))
        windows["rl_active"].append(dict(variant=variant.name, **summarize(data[activation:])))
    comparison = dict(complete=True, protocol=protocol, pairing_verified=True,
                      attempt_and_cost_audit_passed=True, windows=windows)
    dump(output / "comparison.json", comparison)
    write_csv(output / "comparison.csv", table)
    selected = protocol["selection"]["selected"]
    assembly = protocol.get("matrix_assembly", {"advection_scheme": "forward", "atype": 0})
    lines = ["# Module 06 recovery comparison", "",
             f"Exploratory stress test: {selected['name']}, global seed {selected['global_seed']}, base seed {selected['base_seed']}.",
             f"Advection discretization: {assembly['advection_scheme']} (atype={assembly['atype']}). Input coefficients and RHS seeds match the recorded Module 04 stream; the upwind and forward schemes assemble different matrices.",
             f"Selected from Module 04 because its joint learner had {selected['joint_unrecovered']} failures in 5000 problems.",
             "", f"{len(variants)} concurrent single-thread workers; identical recorded matrix/context inputs and initial learner states. Native times include every executed attempt. Total times add learner work, including delayed-target replay.",
             "", protocol.get("fallback", "Historical default fallback follows the learned attempt budget."),
             "", "| Variant | Failures | Fallbacks | Learned attempts | Native (s) | Total (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['label']} | {row['failures']} | {row['fallback']} | {row['learned_attempts']} | {row['native_sec']:.3f} | {row['total_sec']:.3f} |")
    lines += ["", f"All {len(variants)} variants succeeded on {len(common)} common problems. Comparison JSON/CSV also reports costs on that intersection; full-stream totals above retain failed work.",
              "", "The source, native binaries, configuration, problem stream, candidate schedule, and initial model pairing are recorded alongside per-attempt outcomes and final checkpoints.",
              "", "This deliberately selected single-seed stress test is exploratory. Timing-trained trajectories may differ from the earlier Module 04 run and across concurrent workers.", ""]
    if protocol["smoke_cases"]:
        lines.insert(2, "SMOKE CHECK ONLY: shortened prefix, early RL activation.\n")
    (output / "comparison.md").write_text("\n".join(lines))
    if not protocol["smoke_cases"]:
        plot_comparison(output, records, variants)
    return comparison


def plot_comparison(output, records, variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for variant in variants:
        out = [r["outcome"] for r in records[variant.name]]
        axes[0].plot(np.arange(1, len(out) + 1), np.cumsum([o["end_to_end_runtime"] for o in out]), label=variant.label)
        axes[1].plot(np.arange(1, len(out) + 1), np.cumsum([o["failed"] for o in out]), label=variant.label)
    for ax in axes:
        ax.axvline(1000, color="0.6", linestyle=":", linewidth=1)
        ax.set_xlabel("Problem index")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative total time (s)")
    axes[1].set_ylabel("Cumulative failures")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output.resolve())
