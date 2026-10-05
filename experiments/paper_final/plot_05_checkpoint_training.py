"""Two Module 05 cost figures using the unchanged Module 04 bar renderer.

Reads completed training logs only. No training, solves or source-data writes.
"""
from experiments.paper_final import run_05_checkpoint_training as training
from experiments.paper_final import run_05_policy as base

import argparse
import csv
import json
import math
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import experiments.joint.solve_control.joint_experiment_plotting as module04

ROOT = base.ROOT
DEFAULT = training.OUTPUT / "training"
METHODS = training.METHODS
LABELS = dict(zip(METHODS, ("W1", "Fixed 1.40", "Fixed 1.60", "Periodic\n(2.85, 1.10)", "LSTDQ")))
COMPONENTS = {"setup_runtime": "setup_runtime", "native_solve_runtime": "solve_runtime",
              "controller_runtime": "infer_runtime", "setup_bandit_overhead": "bandit_overhead_runtime"}
WINDOWS = {"all_5000": (0, 5000), "last_1000": (4000, 5000),
           "first_1000": (0, 1000), "active_4000": (1000, 5000), "early_active_3000": (1000, 4000)}
SAVED_WINDOWS = {"all_5000": "logical_full", "last_1000": "last_1000",
                 "first_1000": "shared_prefix", "active_4000": "continuation"}


def reviewed_data(source):
    completion = base.read(source / "complete.json")
    assert completion["audit_passed"] and not completion["functional_check"]
    sources = {source / "trajectories" / f"{method}.jsonl": sha
               for method, sha in completion["trajectory_hashes"].items()}
    for path, sha in sources.items():
        assert base.file_hash(path) == sha, path
    rows = {method: [json.loads(line) for line in (source / "trajectories" / f"{method}.jsonl").open()]
            for method in METHODS}
    inputs = base.read(source / "inputs.json")
    saved = base.read(source / "training_diagnostics.json")
    maximum_accounting_error = 0.
    for method in METHODS:
        assert len(rows[method]) == len(inputs) == 5000
        for i, row in enumerate(rows[method]):
            assert row["stream_index"] == row["online_index"] == i
            assert row["mkw"] == inputs[i]["mkw"]
            if i < 1000:
                assert row == rows[METHODS[0]][i], "Prefix rows must be shared, not remeasured"
            o = row["outcome"]
            assert not o["unrecovered_failure"]
            assert math.isclose(o["runtime"], o["setup_runtime"] + o["solve_runtime"], rel_tol=1e-10, abs_tol=1e-12)
            assert math.isclose(o["native_solve_runtime"], o["solve_runtime"], rel_tol=1e-10, abs_tol=1e-12)
            components = [o[key] for key in COMPONENTS.values()]
            assert all(math.isfinite(v) and v >= 0 for v in components)
            error = abs(math.fsum(components) - o["end_to_end_runtime"])
            maximum_accounting_error = max(maximum_accounting_error, error)
            assert error < 1e-10
    result = {"protocol": {"methods": list(METHODS), "families": {m: m for m in METHODS},
                           "method_labels": LABELS}, "windows": {}}
    for window, (start, stop) in WINDOWS.items():
        summaries = {}
        for method in METHODS:
            rr = rows[method][start:stop]
            totals = {key: math.fsum(r["outcome"][field] for r in rr) for key, field in COMPONENTS.items()}
            totals["native_total_runtime"] = math.fsum(r["outcome"]["runtime"] for r in rr)
            totals["end_to_end_runtime"] = math.fsum(r["outcome"]["end_to_end_runtime"] for r in rr)
            if window in SAVED_WINDOWS:
                reference = saved[SAVED_WINDOWS[window]][method]
                for key, field in COMPONENTS.items():
                    np.testing.assert_allclose(totals[key], reference[field], rtol=1e-12, atol=1e-9)
                np.testing.assert_allclose(totals["end_to_end_runtime"], reference["end_to_end_runtime"], rtol=1e-12, atol=1e-9)
            summaries[method] = {"cases": len(rr), "totals_sec": totals,
                "means_sec": {key: value/len(rr) for key, value in totals.items()},
                "fallback_cases": sum(bool(r["outcome"]["fallback_used"]) for r in rr),
                "unrecovered_cases": 0}
        result["windows"][window] = {"problem_start": start+1, "problem_stop": stop, "methods": summaries}
    prefix = result["windows"]["first_1000"]["methods"][METHODS[0]]["totals_sec"]["end_to_end_runtime"]
    physical_cost = prefix + math.fsum(result["windows"]["active_4000"]["methods"][m]["totals_sec"]["end_to_end_runtime"] for m in METHODS)
    audit = base.read(source / "audit.json")
    np.testing.assert_allclose(physical_cost, audit["actual_inclusive_training_seconds"], rtol=1e-12, atol=1e-9)
    result["accounting"] = {"logical_problems_per_branch": 5000, "physical_executions": 21000,
        "shared_prefix_seconds": prefix, "physical_accounted_seconds": physical_cost,
        "shared_prefix_note": "Each full-stream bar includes the same measured W1 prefix once. Summing all five full-stream bars would count the shared prefix five times; use physical_accounted_seconds for actual aggregate training work.",
        "cost_scope": "Setup and native solve include all primary/recovery work; controller includes inference and learning; bandit overhead includes selection, loss evaluation and updates.",
        "replication": "One adaptive training stream per branch; no error bars or independent-training significance claim.",
        "maximum_row_accounting_error_sec": maximum_accounting_error}
    source_paths = list(sources) + [source / name for name in ("inputs.json", "complete.json", "audit.json", "training_diagnostics.json", "experiment_config.json")]
    result["source_sha256"] = {str(p.relative_to(ROOT)): base.file_hash(p) for p in source_paths}
    return result


def settings(window):
    return (("Problem", "Diffusion 60 x 60 x 60"), ("Tolerance / cycle cap", "1e-6 / 50"),
            ("Setup learners", "5 independent LinUCB"), ("Shared W1 prefix", "Problems 1-1000"),
            ("Branch adaptation", "Problems 1001-5000"),
            ("Displayed window", "1-5000" if window == "all_5000" else "4001-5000"),
            ("Training replicates", "1 checkpoint set"),
            ("Prefix accounting", "Executed once; included\n  once in each full bar"),
            ("All recovery costs", "Included"))


def render(result, destination):
    destination.mkdir(parents=True, exist_ok=True)
    captured = []
    original_close = plt.close
    figures = []
    with PdfPages(destination / "module05_runtime_comparisons.pdf") as atlas:
        for window, title in (("all_5000", "Module 05 | All 5,000 training problems"),
                              ("last_1000", "Module 05 | Final 1,000 training problems (4001-5000)")):
            stem = window + "_runtime_breakdown"
            # Reuse the actual Module 04 renderer. Adapter hooks only supply
            # current settings, a stable branch order, and additional exports.
            with patch.object(module04, "_shared_plot_settings", lambda protocol: settings(window)), \
                 patch.object(module04, "_methods_by_native_runtime", lambda methods, summary: tuple(methods)), \
                 patch.object(module04.plt, "close", lambda fig: captured.append(fig)):
                module04._plot_all_runtime_breakdown(path=destination / f"{stem}.png", result=result,
                    window_key=window, figure_title=title)
            figure = captured.pop()
            for axis in figure.axes[:2]:
                axis.set_xticks(np.arange(len(METHODS)), [LABELS[m] for m in METHODS], rotation=0, ha="center")
                axis.set_xlabel("Branch (each with its own setup learner)")
                axis.set_ylim(0, 225)  # Same units and scale for both windows.
                axis.legend(frameon=False, loc="upper center", ncol=2)
            figure.axes[2].text(0, .015, "Training costs are diagnostics.\nModels freeze after problem 5000.",
                transform=figure.axes[2].transAxes, fontsize=9, va="bottom", color="#454545")
            # Check that the unchanged renderer drew the reviewed component
            # heights and shared branch order in each panel.
            expected_components = (tuple(COMPONENTS)[:2], tuple(COMPONENTS))
            for axis, components in zip(figure.axes[:2], expected_components):
                expected = [1000*result["windows"][window]["methods"][m]["means_sec"][key]
                            for key in components for m in METHODS]
                np.testing.assert_allclose([p.get_height() for p in axis.patches], expected, rtol=1e-12, atol=1e-10)
            for ext in ("png", "pdf", "svg"):
                figure.savefig(destination / f"{stem}.{ext}", dpi=220, bbox_inches="tight")
            figure.savefig(destination / f"{stem}_preview.png", dpi=125, bbox_inches="tight")
            atlas.savefig(figure, bbox_inches="tight")
            figures.append({"stem": stem, "window": window, "title": title})
            original_close(figure)
    return figures


def write_report(result, destination):
    lines = ["# Module 05 training costs and the Module 06 comparison", "",
        "These two figures use the existing Module 04 native/end-to-end stacked-bar renderer with current Module 05 measurements. The horizontal axis is the branch. Both windows use the same 0-225 ms/case vertical scale, branch order and component colors.", "",
        "![Full stream](all_5000_runtime_breakdown_preview.png)", "",
        "![Final thousand](last_1000_runtime_breakdown_preview.png)", "",
        "## Accounted training totals", "",
        "| Branch | Full 5,000 total (s) | Full 5,000 mean (ms) | Final 1,000 total (s) | Final 1,000 mean (ms) |",
        "|---|---:|---:|---:|---:|"]
    for m in METHODS:
        a, z = [result["windows"][w]["methods"][m] for w in ("all_5000", "last_1000")]
        lines.append(f"| {LABELS[m].replace(chr(10), ' ')} | {a['totals_sec']['end_to_end_runtime']:.6f} | {1000*a['means_sec']['end_to_end_runtime']:.6f} | {z['totals_sec']['end_to_end_runtime']:.6f} | {1000*z['means_sec']['end_to_end_runtime']:.6f} |")
    lines += ["", "Setup and native solve include recovery; controller includes inference and learning; setup-bandit overhead includes selection, loss evaluation and updates. The charts display per-problem means, like Module 04. The table also reports window sums.", "",
        result["accounting"]["shared_prefix_note"], "",
        "One training replicate is available. Individual problems in an adaptive stream are not independent retraining replicates, so these descriptive figures have no training-seed confidence intervals.", "",
        "## Why the ranking changes", "",
        "| Comparison | Periodic | LSTDQ | RL cost reduction relative to Periodic |",
        "|---|---:|---:|---:|"]
    comparisons = {}
    for window in ("all_5000", "early_active_3000", "last_1000"):
        values = result["windows"][window]["methods"]
        p, r = [values[m]["totals_sec"]["end_to_end_runtime"] for m in ("bandit_periodic", "bandit_lstdq")]
        comparisons[window] = {"periodic_seconds": p, "rl_seconds": r, "rl_saving_percent": 100*(1-r/p)}
        lines.append(f"| Training {result['windows'][window]['problem_start']}-{result['windows'][window]['problem_stop']} | {p:.6f} s | {r:.6f} s | {100*(1-r/p):+.4f}% |")
    module06 = ROOT / "results/paper_final/06_policy/20260928_frozen_five_methods"
    summary = base.read(module06 / "summary.json")
    p, r = [summary["overall_mean_seconds"][m] for m in ("bandit_periodic", "bandit_lstdq")]
    lines.append(f"| Module 06 frozen, fresh inputs | {1000*p['inclusive_total_sec']:.6f} ms/input | {1000*r['inclusive_total_sec']:.6f} ms/input | {100*(1-r['inclusive_total_sec']/p['inclusive_total_sec']):+.4f}% |")
    lines += ["", "Positive reductions favor RL. RL's early training advantage remains in the 5,000-problem cumulative total, while Periodic is already cheaper in the final thousand. The frozen result agrees with that late-training ordering. A cumulative learning-history total and the performance of final frozen models answer different questions.", "",
        "The final-thousand breakdown makes the distinction concrete: RL still has a slightly smaller native solve cost, but costs more in setup construction and controller work. The same pattern occurs in Module 06. These are cost decompositions, not causal proof of how training produced the final models.", "",
        "| RL minus Periodic, ms/problem | Setup | Native solve | Controller | Setup learner | Total |",
        "|---|---:|---:|---:|---:|---:|"]
    last = result["windows"]["last_1000"]["methods"]
    delta = {k: 1000*(last["bandit_lstdq"]["means_sec"][k]-last["bandit_periodic"]["means_sec"][k]) for k in COMPONENTS}
    lines.append("| Training final 1,000 | " + " | ".join(f"{delta[k]:+.6f}" for k in COMPONENTS) + f" | {math.fsum(delta.values()):+.6f} |")
    fields = ("setup_sec", "solve_sec", "controller_sec", "setup_selection_sec")
    frozen_delta = {k: 1000*(r[k]-p[k]) for k in fields}
    lines.append("| Module 06 frozen | " + " | ".join(f"{frozen_delta[k]:+.6f}" for k in fields) + f" | {math.fsum(frozen_delta.values()):+.6f} |")
    lines += ["", "Module 06 uses 100 fresh inputs, three timing repetitions and final frozen models, with setup selection but no updates or epsilon exploration. Each method uses its own trained setup selector. Training uses a changing setup learner and an evolving exploratory RL controller. The 100 test inputs and the 1,000 late-training inputs are not identical, and their sessions are distinct.", "",
        "If 'official run' refers to the older frozen Run 04 instead: that study compared matched Joint hierarchies and the older (2.6,1) schedule over six checkpoints. Its reported 3.22% RL advantage was native continuation only; including controller overhead already changed it to a 1.16% RL disadvantage. It is not the same policy, hierarchy population or cost scope as current Module 06.", "",
        "The previous controller-transfer diagnostic concerns RL on Periodic-selected hierarchies. That is a separate crossed diagnostic; it does not by itself explain the primary comparison of each method on its own setup selector.", "",
        "## Files", "", "`module05_runtime_comparisons.pdf` contains both figures. Individual PNG/PDF/SVG files are also supplied. `figure_data.json` and `window_costs.csv` retain all component values and source hashes. The Module 04 plotting source is unchanged.", ""]
    (destination / "REPORT.md").write_text("\n".join(lines))
    (destination / "CAPTIONS.md").write_text("\n".join([
        "# Figure captions", "",
        "1. Module 05 full-stream training cost over problems 1-5000. Each branch is charged the identical W1 prefix once, followed by its own 4000-problem continuation. Left: native setup and solve, including recovery. Right: native work plus controller and setup-learner overhead. Bars are means in milliseconds per problem; branches remain in a common order. The common prefix was physically executed only once.", "",
        "2. Module 05 late-training cost over problems 4001-5000, using the same component colors, branch order and vertical scale as Figure 1. Setup selectors and the RL controller were still adapting in this window. It is not a frozen test set. Costs are descriptive for one trained checkpoint set; no independent-training error bars are available.", ""]))
    result["ranking_comparison"] = {**comparisons,
        "module06_rl_saving_percent": 100*(1-r["inclusive_total_sec"]/p["inclusive_total_sec"]),
        "last1000_rl_minus_periodic_ms": delta, "module06_rl_minus_periodic_ms": frozen_delta}
    result["source_sha256"][str((module06 / "summary.json").relative_to(ROOT))] = base.file_hash(module06 / "summary.json")
    base.dump(destination / "figure_data.json", result)
    with (destination / "window_costs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("window", "method", "problems", "metric", "total_seconds", "mean_milliseconds"))
        writer.writeheader()
        for window, data in result["windows"].items():
            for method, row in data["methods"].items():
                for metric, total in row["totals_sec"].items():
                    writer.writerow(dict(window=window, method=method, problems=row["cases"], metric=metric,
                        total_seconds=total, mean_milliseconds=1000*row["means_sec"][metric]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT)
    options = parser.parse_args()
    destination = options.source / "analysis/paper_figures"
    original_renderer = Path(module04.__file__).resolve()
    renderer_hash = base.file_hash(original_renderer)
    result = reviewed_data(options.source)
    figures = render(result, destination)
    write_report(result, destination)
    assert base.file_hash(original_renderer) == renderer_hash
    for path, sha in result["source_sha256"].items():
        assert base.file_hash(ROOT / path) == sha
    base.dump(destination / "provenance.json", {"renderer": str(original_renderer.relative_to(ROOT)),
        "renderer_sha256": renderer_hash, "original_renderer_unchanged": True,
        "adapter": str(Path(__file__).resolve().relative_to(ROOT)), "adapter_sha256": base.file_hash(Path(__file__)),
        "adaptations": ["Current Module 05 measurements", "Accurate shared-prefix settings", "Same branch order in both windows", "Common vertical scale", "Vector and two-page exports"],
        "figures": figures, "source_sha256": result["source_sha256"],
        "checks": {"raw_rows": 25000, "shared_prefix_identical": True,
                   "cost_components_reconcile": True, "saved_window_totals_match": True,
                   "plotted_bar_heights_match_reviewed_data": True, "original_source_data_unchanged": True}})
    print(json.dumps({"figures": len(figures), "output": str(destination), "ranking_comparison": result["ranking_comparison"]}, indent=2))


if __name__ == "__main__":
    main()
