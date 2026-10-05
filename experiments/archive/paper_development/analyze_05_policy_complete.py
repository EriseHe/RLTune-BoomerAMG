"""Read-only audit and paired analysis of the completed Module 05 experiment."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json

from experiments.archive.paper_development import run_05_policy as study
import numpy as np
from scipy.stats import t


OUTPUT = study.DEFAULT_OUTPUT
METHODS = ["Weight 1", "Development fixed", "Best fixed - test hindsight",
           "Per-problem best fixed - test hindsight", "Development periodic",
           "Development prefix-tail", "Frozen LSTDQ", "Native HYPRE smoother", "Native Chebyshev"]


def stats(values):
    a = np.asarray(values, dtype=float)
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.
    half = float(t.ppf(.975, len(a)-1) * sd / np.sqrt(len(a))) if len(a) > 1 else 0.
    return {"values": a.tolist(), "mean": float(np.mean(a)), "sd": sd,
            "min": float(np.min(a)), "max": float(np.max(a)),
            "seed_t_interval_95": [float(np.mean(a)-half), float(np.mean(a)+half)]}


def main():
    p = OUTPUT
    study.verify_inputs(p)
    summary = study.read(p / "analysis/summary.json")
    protocol = summary["protocol"]
    raw = []
    audits = {}
    trace_sets = defaultdict(list)
    for phase in ["development", "test", "repetitions"]:
        marker = study.read(p / "raw" / phase / "complete.json")
        for name, expected in marker["raw_sha256"].items():
            if study.file_hash(p / name) != expected:
                raise AssertionError(f"Raw data hash mismatch: {name}")
        seen, counts, kind_counts = set(), Counter(), defaultdict(Counter)
        for path in sorted((p / "raw" / phase).glob("worker_*.jsonl")):
            for row in study.read_records(path):
                key = study.cell_key(row)
                if key in seen:
                    raise AssertionError("Duplicate evaluation")
                seen.add(key)
                o = row["outcome"]
                study.audit_outcome(o)
                study.reject_execution_error(o)
                if o.get("controller_update_committed", False):
                    raise AssertionError("A frozen controller committed an update")
                counts["evaluations"] += 1
                counts["primary_failures"] += o.get("primary_status", "success") != "success"
                counts["recoveries_used"] += bool(o.get("fallback_used", False))
                counts["final_failures"] += not row["success"]
                for label, value in [("evaluations", 1), ("recoveries", bool(o.get("fallback_used", False))),
                                     ("failures", not row["success"])]:
                    kind_counts[row["kind"]][label] += value
                if phase != "development":
                    raw.append(study.compact_row(row))
                    if row["case_id"] in protocol["test_repeat_case_ids"]:
                        trace_sets[(row["seed"], row["source"], row["case_id"], row["policy"])].append({
                            "repeat": row["repeat"], "actions": o.get("cycle_actions"),
                            "residuals": o.get("cycle_residuals"),
                            "cycles": o.get("primary_cycles", o.get("iterations", 0)),
                            "native_sec": row["native_continuation_sec"]})
        if seen != study.expected_cells(p, phase) or len(seen) != marker["trials"]:
            raise AssertionError("Unexpected evaluation coverage")
        for q in (p / "raw" / phase).glob("worker_*.complete.json"):
            assert study.read(q)["frozen_audit_passed"]
        audits[phase] = {**counts, "by_kind": dict(kind_counts), "raw_hashes_match": True,
                         "coverage_exact": True, "frozen_audits_passed": True}

    averaged = study.average_repetitions(raw)
    cells = {(r["seed"], r["source"], r["case_id"], r["policy"]): r for r in averaged}
    totals = {(r["seed"], r["source"], r["method"]): r for r in summary["per_seed"]}
    groups = defaultdict(list)
    for row in averaged:
        groups[(row["seed"], row["source"])].append(row)
    method_cases = {}
    for (seed, source), rows in groups.items():
        for method in METHODS:
            total = totals[(seed, source, method)]
            selected = (study.fixed_hindsight(rows)["per_case"] if method == "Per-problem best fixed - test hindsight"
                        else [r for r in rows if r["policy"] == total["policy"]])
            selected = sorted(selected, key=lambda r: r["case_id"])
            assert len(selected) == 100
            method_cases[(seed, source, method)] = selected
            calculated = study.totals(selected)
            for key in (*study.COST_FIELDS, "failures", "recoveries", "primary_cycles"):
                if not np.isclose(calculated[key], total[key], rtol=1e-12, atol=1e-10):
                    raise AssertionError(f"Saved summary disagrees with raw data: {seed}/{source}/{method}/{key}")

    comparisons = []
    for source in study.SOURCES:
        for comparator in ["Weight 1", "Best fixed - test hindsight", "Per-problem best fixed - test hindsight",
                           "Development periodic", "Development prefix-tail"]:
            detail = []
            for seed in protocol["training_seeds"]:
                rl = totals[(seed, source, "Frozen LSTDQ")]
                baseline = totals[(seed, source, comparator)]
                rl_cases = method_cases[(seed, source, "Frozen LSTDQ")]
                base_cases = method_cases[(seed, source, comparator)]
                costs_rl = np.array([r["native_continuation_sec"] for r in rl_cases])
                costs_base = np.array([r["native_continuation_sec"] for r in base_cases])
                detail.append({"seed": seed,
                    "native_reduction_pct": study.reduction(baseline["native_continuation_sec"], rl["native_continuation_sec"]),
                    "inclusive_reduction_pct": study.reduction(baseline["inclusive_continuation_sec"], rl["inclusive_continuation_sec"]),
                    "case_wins": int(np.sum(costs_rl < costs_base)),
                    "case_wins_over_1pct": int(np.sum(costs_rl < .99*costs_base)),
                    "case_losses_over_1pct": int(np.sum(costs_rl > 1.01*costs_base)),
                    "cycle_reduction_pct": study.reduction(baseline["primary_cycles"], rl["primary_cycles"])})
            comparisons.append({"source": source, "comparator": comparator,
                "native": stats([r["native_reduction_pct"] for r in detail]),
                "inclusive": stats([r["inclusive_reduction_pct"] for r in detail]),
                "cycle_reduction": stats([r["cycle_reduction_pct"] for r in detail]),
                "winning_seeds_native": sum(r["native_reduction_pct"] > 0 for r in detail),
                "case_win_total": sum(r["case_wins"] for r in detail), "case_count": 600,
                "detail": detail})

    repeats = []
    for source in study.SOURCES:
        for method in ["Weight 1", "Best fixed - test hindsight", "Development periodic", "Development prefix-tail", "Frozen LSTDQ"]:
            cvs, changed_actions, changed_cycles = [], 0, 0
            replicate_reductions = []
            for seed in protocol["training_seeds"]:
                policy = totals[(seed, source, method)]["policy"]
                for case_id in protocol["test_repeat_case_ids"]:
                    values = sorted(trace_sets[(seed, source, case_id, policy)], key=lambda r: r["repeat"])
                    assert [v["repeat"] for v in values] == [0, 1, 2]
                    times = [v["native_sec"] for v in values]
                    cvs.append(100*np.std(times, ddof=1)/np.mean(times))
                    changed_actions += any(v["actions"] != values[0]["actions"] for v in values[1:])
                    changed_cycles += any(v["cycles"] != values[0]["cycles"] for v in values[1:])
            for rep in [0, 1, 2]:
                reference, current = 0., 0.
                for seed in protocol["training_seeds"]:
                    policy = totals[(seed, source, method)]["policy"]
                    for case_id in protocol["test_repeat_case_ids"]:
                        reference += next(v for v in trace_sets[(seed, source, case_id, "fixed_1.00")] if v["repeat"] == rep)["native_sec"]
                        current += next(v for v in trace_sets[(seed, source, case_id, policy)] if v["repeat"] == rep)["native_sec"]
                replicate_reductions.append(study.reduction(reference, current))
            repeats.append({"source": source, "method": method, "repeat_triplets": len(cvs),
                "median_cv_pct": float(np.median(cvs)), "p90_cv_pct": float(np.percentile(cvs, 90)),
                "changed_action_sequences": changed_actions, "changed_cycle_counts": changed_cycles,
                "subset_reduction_vs_weight1_by_repeat_pct": replicate_reductions})

    hierarchy = []
    jobs = study.read(p / "jobs_test.json")
    jobmap = {(j["seed"], j["source"], j["case_id"]): j for j in jobs}
    for seed in protocol["training_seeds"]:
        per_source = {}
        for source in study.SOURCES:
            per_source[source] = {"unique_setups": len({study.digest(j["params"]) for j in jobs if j["seed"] == seed and j["source"] == source})}
        hierarchy.append({"seed": seed, "identical_setup_cases_between_sources": sum(
            jobmap[(seed, study.SOURCES[0], case)]["params"] == jobmap[(seed, study.SOURCES[1], case)]["params"] for case in range(100)),
            "by_source": per_source})

    grids, cycle_means, overhead, oracle_weights = [], [], [], []
    for source in study.SOURCES:
        for i in range(41):
            policy = f"fixed_{1+i*.05:.2f}"
            values, recovered, failed = [], 0, 0
            for seed in protocol["training_seeds"]:
                rr = [cells[(seed, source, case, policy)] for case in range(100)]
                ref = totals[(seed, source, "Weight 1")]["native_continuation_sec"]
                values.append(study.reduction(ref, sum(r["native_continuation_sec"] for r in rr)))
                recovered += sum(r["recovery"] for r in rr)
                failed += sum(not r["success"] for r in rr)
            grids.append({"source": source, "weight": 1+i*.05, "reduction_pct": stats(values),
                          "recovered_cases": recovered, "final_failed_cases": failed})
        weights = Counter()
        for seed in protocol["training_seeds"]:
            weights.update(r["policy"] for r in method_cases[(seed, source, "Per-problem best fixed - test hindsight")])
        oracle_weights.append({"source": source, "counts": dict(sorted(weights.items()))})
        for method in METHODS:
            selected = [totals[(seed, source, method)] for seed in protocol["training_seeds"]]
            cycle_means.append({"source": source, "method": method,
                               "cycles_per_problem": stats([r["primary_cycles"]/100 for r in selected])})
            overhead.append({"source": source, "method": method,
                "share_of_inclusive_solve_pct": stats([100*r["controller_sec"]/r["inclusive_continuation_sec"] for r in selected]),
                "share_of_inclusive_setup_solve_pct": stats([100*r["controller_sec"]/r["inclusive_total_sec"] for r in selected])})

    result = {"created_at": study.now(), "audit": audits, "rl_comparisons": comparisons,
              "repeat_diagnostics": repeats, "hierarchy_diagnostics": hierarchy,
              "fixed_grid": grids, "per_case_oracle_weights": oracle_weights,
              "cycles": cycle_means, "overhead": overhead,
              "uncertainty_note": "Seed t intervals are descriptive approximate intervals over six paired trained replicas, conditional on the shared 100 test problems and selected finite policy library. Case win counts reuse the same 100 inputs; they are not 600 independent test matrices. No equivalence test or universal feedback-necessity claim."}
    study.dump(p / "analysis/complete_analysis.json", result)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    colors = {"Best fixed - test hindsight": "#6c7885", "Per-problem best fixed - test hindsight": "#9e77b5",
              "Development periodic": "#d19635", "Development prefix-tail": "#31977c", "Frozen LSTDQ": "#245fb5"}
    labels = {"Best fixed - test hindsight": "Best fixed (hindsight)", "Per-problem best fixed - test hindsight": "Per-problem fixed (hindsight)",
              "Development periodic": "Tuned periodic", "Development prefix-tail": "Tuned prefix–tail", "Frozen LSTDQ": "Frozen RL"}
    for ax, source, title in zip(axes, study.SOURCES, ["Setup-only hierarchies", "Joint hierarchies"]):
        for method, color in colors.items():
            values = [totals[(seed, source, method)]["native_reduction_vs_weight1_pct"] for seed in protocol["training_seeds"]]
            ax.plot(protocol["training_seeds"], values, marker="o", color=color, linewidth=2 if method == "Frozen LSTDQ" else 1.4,
                    markersize=5, label=labels[method])
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Training seed")
        ax.set_xticks(protocol["training_seeds"])
        ax.set_ylim(28, 46)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Native solve time reduction vs matched weight 1 (%)")
    fig.suptitle("Module 05 · 60³ diffusion · 100 shared fresh test problems", fontsize=14)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=[0, .13, 1, .95])
    figures = p / "analysis/figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "matched_policy_reductions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"audits": audits, "comparisons": [{"source": r["source"], "comparator": r["comparator"],
        "native": r["native"], "inclusive_mean": r["inclusive"]["mean"], "wins": r["winning_seeds_native"],
        "case_wins": r["case_win_total"]} for r in comparisons], "repeats": repeats,
        "hierarchy": hierarchy, "oracle_weights": oracle_weights}, indent=2))


if __name__ == "__main__":
    main()
