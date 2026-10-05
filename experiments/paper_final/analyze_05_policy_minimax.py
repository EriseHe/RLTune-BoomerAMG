"""Audit Run 05 and report the prescribed (2.85,1.10) comparison."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from experiments.paper_final.common import (
    artifacts,
    frozen_policy as policy,
    policy_analysis as analysis,
)
from experiments.paper_final import run_05_policy_minimax as repeat
from experiments.paper_final.run_05_policy_minimax import verify
from experiments.paper_final.common.policy_analysis import collect_cells, statistics


def analyze(output):
    output = Path(output)
    verify(output)
    protocol = artifacts.read(output / "protocol.json")
    selections = artifacts.read(output / "selections.json")
    jobs = artifacts.read(output / "jobs_test.json")
    marker = artifacts.read(output / "raw/test/complete.json")
    rows = []
    seen = set()
    traces = {}
    incomplete = {}
    for name, digest in marker["raw_sha256"].items():
        if artifacts.file_hash(output / name) != digest:
            raise ValueError("Raw data changed")
        for row in artifacts.read_records(output / name):
            key = analysis.cell_key(row)
            if key in seen:
                raise ValueError("Duplicate trial")
            seen.add(key)
            policy.audit_outcome(row["outcome"])
            if row["outcome"].get("controller_update_committed", False):
                raise ValueError("Frozen controller learned")
            if row["run_number"] != 5:
                raise ValueError("Foreign run metadata")
            spec = protocol["policies"][row["policy"]]
            if spec["kind"] in ("fixed", "periodic"):
                aa = row["outcome"].get("cycle_actions", [])
                pattern = spec["pattern"]
                if aa != [pattern[i % len(pattern)] for i in range(len(aa))]:
                    raise ValueError("Executed schedule differs from protocol")
            rows.append(row)
            if row["repeat"] == 0:
                k = (row["seed"], row["case_id"], row["policy"])
                outcome = row["outcome"]
                traces[k] = {
                    f: np.asarray(outcome.get("cycle_" + f, []), dtype=float)
                    for f in ("actions", "residuals", "times")
                }
                if len({len(a) for a in traces[k].values()}) != 1:
                    raise ValueError("Misaligned cycle trace")
                incomplete[k] = bool(
                    outcome.get("fallback_used", False) or not row["success"]
                )
    if seen != analysis.expected_cells(jobs) or len(seen) != marker["trials"]:
        raise ValueError("Incomplete evaluation coverage")
    cells = collect_cells(rows, [0, 1, 2])
    by_rep = {
        (r["seed"], r["case_id"], r["policy"], r["repeat"]): analysis.compact_row(r)
        for r in rows
    }
    raw = {(r["seed"], r["case_id"], r["policy"], r["repeat"]): r for r in rows}
    jobmap = {(j["seed"], j["case_id"]): j for j in jobs if j["repeat"] == 0}
    parent = Path(protocol["origin_run"])
    old = artifacts.read(parent / "analysis/summary.json")
    if old["seeds"] != protocol["training_seeds"]:
        raise ValueError("Checkpoint labels misaligned")
    seeds = protocol["training_seeds"]
    cases = list(range(protocol["test_cases"]))
    fields = {
        "native": "native_continuation_sec",
        "inclusive": "inclusive_continuation_sec",
        "native_total": "native_total_sec",
        "inclusive_total": "inclusive_total_sec",
        "cycles": "primary_cycles",
    }
    costs = {m: {f: [] for f in fields} for m in repeat.METHODS}
    per_seed = []
    per_repeat = []
    diagnostics = []
    for i, seed in enumerate(seeds):
        reference = analysis.totals(
            [
                cells[seed, c, jobmap[seed, c]["method_policies"]["reference"]]
                for c in cases
            ]
        )
        for method in repeat.METHODS:
            keys = [
                (seed, c, jobmap[seed, c]["method_policies"][method]) for c in cases
            ]
            current = [cells[k] for k in keys]
            total = analysis.totals(current)
            for short, long in fields.items():
                costs[method][short].append([r[long] for r in current])
            gain = analysis.reduction(
                reference["native_continuation_sec"], total["native_continuation_sec"]
            )
            per_seed.append(
                {
                    "seed": seed,
                    "method": method,
                    "label": repeat.LABELS[method],
                    **total,
                    "native_reduction_vs_w1_pct": gain,
                    "inclusive_reduction_vs_w1_pct": analysis.reduction(
                        reference["inclusive_continuation_sec"],
                        total["inclusive_continuation_sec"],
                    ),
                    "mean_cycles": total["primary_cycles"] / len(cases),
                }
            )
            cvs = []
            changed_actions = 0
            changed_cycles = 0
            for k in keys:
                observations = [raw[(*k, rep)] for rep in range(3)]
                tt = [r["native_continuation_sec"] for r in observations]
                cvs.append(float(100 * np.std(tt, ddof=1) / np.mean(tt)))
                actions = [r["outcome"].get("cycle_actions", []) for r in observations]
                changed_actions += any(a != actions[0] for a in actions[1:])
                changed_cycles += len({len(a) for a in actions}) > 1
            diagnostics.append(
                {
                    "seed": seed,
                    "method": method,
                    "median_timing_cv_pct": float(np.median(cvs)),
                    "p90_timing_cv_pct": float(np.percentile(cvs, 90)),
                    "changed_action_cases": changed_actions,
                    "changed_cycle_cases": changed_cycles,
                }
            )
            for rep in range(3):
                total_rep = analysis.totals([by_rep[(*k, rep)] for k in keys])
                ref = analysis.totals(
                    [
                        by_rep[
                            seed,
                            c,
                            jobmap[seed, c]["method_policies"]["reference"],
                            rep,
                        ]
                        for c in cases
                    ]
                )
                per_repeat.append(
                    {
                        "seed": seed,
                        "method": method,
                        "repeat": rep + 1,
                        **total_rep,
                        "native_reduction_vs_w1_pct": analysis.reduction(
                            ref["native_continuation_sec"],
                            total_rep["native_continuation_sec"],
                        ),
                    }
                )
    aggregate = []
    comparisons = []
    for method in repeat.METHODS:
        rr = [r for r in per_seed if r["method"] == method]
        aggregate.append(
            {
                "method": method,
                "label": repeat.LABELS[method],
                "native_reduction_vs_w1_pct": statistics(
                    [r["native_reduction_vs_w1_pct"] for r in rr]
                ),
                "inclusive_reduction_vs_w1_pct": statistics(
                    [r["inclusive_reduction_vs_w1_pct"] for r in rr]
                ),
                "failed_case_seed_pairs": sum(r["failures"] for r in rr),
                "recovered_case_seed_pairs": sum(r["recoveries"] for r in rr),
            }
        )
        if method == "rl":
            continue
        for field in ("native", "inclusive"):
            base = np.asarray(costs[method][field])
            candidate = np.asarray(costs["rl"][field])
            gains = 100 * (1 - candidate.sum(axis=1) / base.sum(axis=1))
            comparisons.append(
                {
                    "comparator": method,
                    "cost": field,
                    **statistics(gains),
                    "winning_seeds": int(np.sum(gains > 0)),
                    "case_wins_by_seed": np.sum(candidate < base, axis=1).tolist(),
                }
            )
    payload = {
        "run_number": 5,
        "comparison_run": 4,
        "created_at": artifacts.now(),
        "protocol": protocol,
        "seeds": seeds,
        "methods": list(repeat.METHODS),
        "labels": repeat.LABELS,
        "main_methods": list(repeat.MAIN_METHODS),
        "case_order_0_based": selections["case_order_0_based"],
        "costs": costs,
        "per_seed": per_seed,
        "per_repeat": per_repeat,
        "aggregate": aggregate,
        "rl_comparisons": comparisons,
        "repetition_diagnostics": diagnostics,
        "audit": {
            "trials": len(rows),
            "exact_coverage": True,
            "all_repetitions_present": True,
            "failures": sum(not r["success"] for r in rows),
            "recovered_trials": sum(
                bool(r["outcome"].get("fallback_used", False)) for r in rows
            ),
            "raw_sha256": marker["raw_sha256"],
        },
        "fixed_selection_provenance": {"source": str(parent), "no_new_scan": True},
        "interpretation": "Three fresh repetitions; all Run 04 checkpoints, hierarchies and fixed choices retained. (2.85,1.10) prescribed before timing. Same previously observed cases; not new independent training seeds or a fresh holdout.",
    }
    width = max(len(t["actions"]) for t in traces.values())
    arrays = {"case_order_0_based": np.asarray(selections["case_order_0_based"])}
    for seed in seeds:
        for method in repeat.METHODS:
            keys = [
                (seed, c, jobmap[seed, c]["method_policies"][method]) for c in cases
            ]
            for field in ("actions", "residuals", "times"):
                matrix = np.full((len(cases), width), np.nan)
                for i, k in enumerate(keys):
                    matrix[i, : len(traces[k][field])] = traces[k][field]
                arrays[f"seed{seed}__{method}__{field}"] = matrix
            arrays[f"seed{seed}__{method}__primary_incomplete"] = np.asarray(
                [incomplete[k] for k in keys]
            )
    np.savez_compressed(output / "analysis/trace_arrays.npz", **arrays)
    artifacts.dump(output / "analysis/summary.json", payload)
    for name, data in (
        ("per_seed", per_seed),
        ("per_repeat", per_repeat),
        ("repetition_diagnostics", diagnostics),
    ):
        artifacts.write_csv(output / "analysis" / (name + ".csv"), data)
    artifacts.emit(
        output,
        "Run 05 analysis complete",
        trials=len(rows),
        failures=payload["audit"]["failures"],
    )
    write_report(output, payload)
    return payload


def write_report(output, data):
    lines = [
        "# Module 05 Run 05 — grid-constrained weighted-minimax relaxation",
        "",
        "Diffusion 60³; the same six Run 04 checkpoints (refreshed seed 4), 100 inputs and three fresh timing repetitions. The prescribed (2.85,1.10) replaces (2.6,1), and (1,3) is omitted; fixed weights and all hierarchy choices are retained.",
        "",
        "Percentages are ratios of summed costs after averaging the three repetitions per case. Native continuation includes primary solves and all recovery setup/solve; common initial setup and controller dispatch are excluded. Inclusive continuation adds actual controller costs.",
        "",
        "| RL compared with | Native reduction, mean ± sample SD (pp) | Inclusive reduction, mean ± sample SD (pp) | Native winning seeds |",
        "|---|---:|---:|---:|",
    ]
    for method in data["methods"]:
        if method == "rl":
            continue
        rr = {r["cost"]: r for r in data["rl_comparisons"] if r["comparator"] == method}
        n = rr["native"]
        i = rr["inclusive"]
        lines.append(
            f"| {data['labels'][method]} | {n['mean']:.2f}% ± {n['sd']:.2f} | {i['mean']:.2f}% ± {i['sd']:.2f} | {n['winning_seeds']}/6 |"
        )
    lines.extend(
        [
            "",
            f"Audit: {data['audit']['trials']:,} distinct executions, exact planned coverage; {data['audit']['failures']} unrecovered failures and {data['audit']['recovered_trials']} recovered executions (charged).",
            "",
            "The jointly selected pair minimizes a normalized SPD polynomial-smoothing surrogate. It does not minimize the measured multilevel runtime by theorem. Full-cycle scheduling and high-first phase are explicit experimental transfers. The rigorous derivation and implementation audit are in `theory/`.",
            "",
            "All checkpoints and test cases were already inspected in prior work; seed 4 was retrained after earlier results. This is a prescribed policy follow-up, not a fresh holdout or six newly trained seeds. Fixed comparators use successful best-observed 41-grid choices followed by retiming. The illustrative best seed is selected after evaluation; all six remain in the tables and figures.",
            "",
            "Run 04 is retained separately as historical evidence. All five current policy roles are freshly timed together; current comparisons do not combine measurements from different runs.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
