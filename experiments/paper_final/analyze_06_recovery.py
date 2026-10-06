"""Audit and compare the four completed Module 06 recovery workers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.paper_final.common.artifacts import dump, file_hash, read, write_csv
from experiments.paper_final.recovery.protocol import VARIANTS


def audit(records, variant):
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
        committed = not out["unrecovered_failure"]
        if bool(out["bandit_update_committed"]) != committed:
            raise ValueError("Bandit transaction does not match final success")
        if not committed and out["controller_update_committed"]:
            raise ValueError("Unrecovered problem retained RL updates")
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
        unrecovered=sum(bool(o["unrecovered_failure"]) for o in outcomes),
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
    records = {}
    results = {}
    for variant in VARIANTS:
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
    reference = records[VARIANTS[0].name]
    for data in records.values():
        if any((a["mkw"], a["context"]) != (b["mkw"], b["context"]) for a, b in zip(reference, data)):
            raise ValueError("Unpaired matrix/context inputs")
    common = [i for i in range(len(reference))
              if all(not data[i]["outcome"]["unrecovered_failure"] for data in records.values())]
    table = []
    for variant in VARIANTS:
        data = records[variant.name]
        table.append(dict(variant=variant.name, label=variant.label, **summarize(data),
                          common_success_cases=len(common),
                          common_success_native_sec=sum(data[i]["outcome"]["runtime"] for i in common),
                          common_success_total_sec=sum(data[i]["outcome"]["end_to_end_runtime"] for i in common)))
    windows = {"all": table, "last_1000": [], "rl_active": []}
    activation = 1 if protocol["smoke_cases"] else 1000
    for variant in VARIANTS:
        data = records[variant.name]
        windows["last_1000"].append(dict(variant=variant.name, **summarize(data[-1000:])))
        windows["rl_active"].append(dict(variant=variant.name, **summarize(data[activation:])))
    comparison = dict(complete=True, protocol=protocol, pairing_verified=True,
                      attempt_and_cost_audit_passed=True, windows=windows)
    dump(output / "comparison.json", comparison)
    write_csv(output / "comparison.csv", table)
    selected = protocol["selection"]["selected"]
    lines = ["# Module 06 recovery comparison", "",
             f"Exploratory stress test: {selected['name']}, global seed {selected['global_seed']}, base seed {selected['base_seed']}.",
             f"Selected from Module 04 because its joint learner had {selected['joint_unrecovered']} unrecovered failures in 5000 problems.",
             "", "Four concurrent single-thread workers; identical recorded matrix/context inputs and initial learner states. Native times include every attempt and fallback. Total times add learner work, including delayed-target replay.",
             "", "| Variant | Unrecovered | Fallbacks | Learned attempts | Native (s) | Total (s) |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in table:
        lines.append(f"| {row['label']} | {row['unrecovered']} | {row['fallback']} | {row['learned_attempts']} | {row['native_sec']:.3f} | {row['total_sec']:.3f} |")
    lines += ["", f"All four variants succeeded on {len(common)} common problems. Comparison JSON/CSV also reports costs on that intersection; full-stream totals above retain failed work.",
              "", "The source, native binaries, configuration, problem stream, candidate schedule, and initial model pairing are recorded alongside per-attempt outcomes and final checkpoints.",
              "", "This deliberately selected single-seed stress test is exploratory. Timing-trained trajectories may differ from the earlier Module 04 run and across concurrent workers.", ""]
    if protocol["smoke_cases"]:
        lines.insert(2, "SMOKE CHECK ONLY: shortened prefix, early RL activation.\n")
    (output / "comparison.md").write_text("\n".join(lines))
    if not protocol["smoke_cases"]:
        plot_comparison(output, records)
    return comparison


def plot_comparison(output, records):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for variant in VARIANTS:
        out = [r["outcome"] for r in records[variant.name]]
        axes[0].plot(np.arange(1, len(out) + 1), np.cumsum([o["end_to_end_runtime"] for o in out]), label=variant.label)
        axes[1].plot(np.arange(1, len(out) + 1), np.cumsum([o["unrecovered_failure"] for o in out]), label=variant.label)
    for ax in axes:
        ax.axvline(1000, color="0.6", linestyle=":", linewidth=1)
        ax.set_xlabel("Problem index")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Cumulative total time (s)")
    axes[1].set_ylabel("Cumulative unrecovered failures")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "comparison.png", dpi=180)
    fig.savefig(output / "comparison.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args().output.resolve())
