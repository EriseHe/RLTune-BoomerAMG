"""Publish audited fresh Module 5 records using the reviewed manuscript style.

No native solves, learning, scans or retiming are performed by this publisher.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from prepare_module4_current import COLORS, WEIGHT_CMAP, sha, style
from paper_figure_io import save
import official_paths


def publish(run, runtime, paper):
    code = run.resolve().parents[2]
    assert run.resolve() == official_paths.module_root(code, "module05").resolve()
    assert runtime.resolve() == official_paths.runtime_root(code).resolve()
    sys.path.insert(0, str(runtime))
    from experiments.paper_final.matched_policy import workflow
    from experiments.paper_final.matched_policy.selection import select_fixed
    from experiments.paper_final.plot_05_matched_policy import read_study, paired_gains
    from experiments.paper_final.matched_policy.configuration import METHODS
    import matplotlib.pyplot as plt

    workflow.verify(run)
    scans = workflow.phase_rows(run, "scan")
    trials = workflow.phase_rows(run, "test")
    selections = json.loads((run / "selections.json").read_text())
    assert select_fixed(scans, json.loads((run / "jobs_scan.json").read_text())) == selections
    summary, seeds, cases, costs, success, weights, sources = read_study(run)
    assert seeds == [1, 2, 3, 4, 5, 6] and len(cases) == 100
    assert summary["audit"]["fresh_scan_trials"] == len(scans) == 29520
    assert success.all() and summary["audit"]["evaluation_failures"] == 0
    assert costs.shape == (6, 5, 100, 4) and weights.shape == (6, 5, 100, 50)
    assert all(row["setup_learning_unchanged"] for row in json.loads((run / "setup_freeze_audit.json").read_text()))
    assert json.loads((run / "preflight.json").read_text())["passed"]
    finite = weights[np.isfinite(weights)]
    assert ((finite >= 1) & (finite <= 3)).all()
    np.testing.assert_allclose((finite - 1) / .05, np.round((finite - 1) / .05), atol=1e-8)
    native = paired_gains(costs, success)
    inclusive = paired_gains(costs, success, 1)
    manifest = json.loads((run / "paper_figures/manifest.json").read_text())
    assert manifest["checks"]["passed"]
    for name, expected in manifest["source_sha256"].items():
        assert sha(official_paths.recorded_path(code, name)) == expected
    for name, expected in manifest["output_sha256"].items():
        assert sha(run / "paper_figures" / name) == expected
    with np.load(run / "paper_figures/plot_data.npz", allow_pickle=False) as arrays:
        np.testing.assert_allclose(arrays["costs_sec"], costs)
        np.testing.assert_allclose(arrays["first_repetition_primary_weights"], weights, equal_nan=True)
        np.testing.assert_allclose(arrays["native_reductions_pct"], native)
        np.testing.assert_allclose(arrays["inclusive_reductions_pct"], inclusive)

    review = paper / "review/module5_fresh_20261012"
    review.mkdir(parents=True, exist_ok=True)
    for name in ("protocol.json", "analysis/summary.json", "analysis/per_seed.csv", "analysis/per_case.csv",
                 "paper_figures/manifest.json", "paper_figures/plot_data.npz", "setup_freeze_audit.json", "preflight.json"):
        destination = review / ("plot_manifest.json" if name == "paper_figures/manifest.json" else Path(name).name)
        shutil.copy2(run / name, destination)
        sources[str(run / name)] = sha(run / name)
    for phase in ("scan", "test"):
        marker = run / f"raw/{phase}/complete.json"
        sources[str(marker)] = sha(marker)
        sources.update({str(run / name): digest for name, digest in json.loads(marker.read_text())["raw_sha256"].items()})

    comparisons, rows = [], []
    for j, (method, label) in enumerate(zip(METHODS[:-1], (r"Fixed $w=1$", "Global best fixed",
                                                        "Per-instance best fixed", r"Periodic $(2.85,1.10)$"))):
        values = []
        for field in (0, 1):
            totals = costs[..., field].sum(axis=2)
            gain = 100 * (1 - totals[:, 4] / totals[:, j])
            mean, sd = float(gain.mean()), float(gain.std(ddof=1))
            expected = next(r for r in summary["rl_comparisons"] if r["comparator"] == method and
                            r["metric"] == ("native_continuation_sec", "inclusive_continuation_sec")[field])
            assert expected["seeds"] == 6
            np.testing.assert_allclose([mean, sd], [expected["mean_pct"], expected["sample_sd_pct"]], atol=1e-10)
            comparisons.append(dict(comparator=method, cost=("native", "inclusive")[field],
                                    values=gain.tolist(), mean=mean, sample_sd=sd))
            values.append(rf"${mean:.1f}\,\pm\,{sd:.1f}$\%")
        rows.append(label + " & " + " & ".join(values) + r" \\")
    (paper / "sections/current_matched_rows.tex").write_text("\n".join(rows) + "\n")

    style()
    methods = ("fixed", "oracle", "periodic", "rl")
    labels = ("Global best fixed", "Per-instance best fixed", "Periodic (2.85, 1.10)", "Frozen RL")
    colors = (COLORS[2], COLORS[1], WEIGHT_CMAP(.63), COLORS[3])
    fig, ax = plt.subplots(figsize=(6.85, 2.35))
    extent, statistics = [], {}
    for i, (method, color) in enumerate(zip(methods, colors)):
        gains = native[:, METHODS.index(method)]
        mean, sd = float(gains.mean()), float(gains.std(ddof=1))
        ax.scatter(gains, i + np.linspace(-.13, .13, 6), s=16, facecolors="white",
                   edgecolors=color, linewidths=.85, zorder=3)
        ax.errorbar(mean, i, xerr=sd, fmt="D", color=color, capsize=2.5,
                    markersize=4, elinewidth=.9, zorder=4)
        extent.extend([gains.min(), gains.max(), mean - sd, mean + sd])
        statistics[method] = dict(values=gains.tolist(), mean=mean, sample_sd=sd)
    padding = .055 * (max(extent) - min(extent))
    ax.set(yticks=range(4), yticklabels=labels, ylim=(3.4, -.4),
           xlim=(min(extent) - padding, max(extent) + padding),
           xlabel=r"Native continuation-time reduction relative to $w=1$ (%)")
    fig.suptitle(r"Matched-policy savings (Diffusion $60^3$, six checkpoints)", y=.99, fontsize=9)
    ax.text(.98, 1.075, "Open circles: checkpoints; diamonds: mean ± SD", transform=ax.transAxes,
            ha="right", fontsize=6.5)
    fig.subplots_adjust(left=.235, right=.985, top=.78, bottom=.22)
    save(fig, paper, "matched_policy_savings")

    selected = int(np.argmax(native[:, 4]))
    seed = seeds[selected]
    assert seed == manifest["illustrative_seed"]
    order = np.lexsort((cases, -np.isfinite(weights[selected, 0]).sum(axis=1)))
    heatmap_methods = ("reference", "oracle", "periodic", "rl")
    indices = [METHODS.index(m) for m in heatmap_methods]
    # Display every recorded primary cycle in the selected illustration.
    cap = int(np.max(np.isfinite(weights[selected, indices]).sum(axis=2)))
    assert 1 <= cap <= summary["protocol"]["max_cycles"] == 50
    shown = {m: weights[selected, METHODS.index(m), order, :cap] for m in heatmap_methods}
    ticks = np.unique(np.r_[np.arange(0, cap, 10), cap - 1])
    fig, axes = plt.subplots(2, 2, figsize=(6.85, 3.6), sharex=True, sharey=True)
    panel_labels = (r"Fixed $w=1$", "Per-instance best fixed", "Periodic (2.85, 1.10)", "Frozen RL")
    for ax, method, title, letter in zip(axes.flat, heatmap_methods, panel_labels, "abcd"):
        im = ax.imshow(shown[method].T, origin="upper", interpolation="nearest", aspect="auto",
                       extent=(.5, 100.5, cap - .5, -.5), cmap=WEIGHT_CMAP, vmin=1, vmax=3, rasterized=True)
        ax.set_title(f"({letter}) {title}", loc="left")
        ax.set(xticks=[1, 25, 50, 75, 100], yticks=ticks)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Cycle index $k$")
    for ax in axes[1]:
        ax.set_xlabel("Test problem rank")
    fig.subplots_adjust(left=.10, right=.855, top=.83, bottom=.14, wspace=.12, hspace=.4)
    color_ax = fig.add_axes([.89, .14, .017, .69])
    fig.colorbar(im, cax=color_ax, label="Relaxation weight", ticks=[1, 1.5, 2, 2.5, 3])
    fig.suptitle(rf"Matched-policy relaxation weights (Diffusion $60^3$, seed {seed})", y=.97, fontsize=9)
    save(fig, paper, "matched_policy_weights")
    np.savez_compressed(review / "displayed_weights.npz", **shown, case_order_0_based=order,
                        case_ids_sorted=np.asarray(cases)[order])

    recovery = [r for r in trials if r["outcome"].get("fallback_used", False)]
    report = dict(passed=True, run=str(run), numerical_rerun=False, audit=summary["audit"],
                  native_savings=statistics, rl_comparisons=comparisons,
                  global_fixed_choices={s: v["global"] for s, v in selections.items()},
                  illustrative_seed=seed, illustrative_rl_native_reduction_pct=float(native[selected, 4]),
                  illustration_selection="Largest qualifying native RL reduction versus weight one across all six seeds",
                  heatmap_methods=heatmap_methods, displayed_cycle_indices=[0, cap - 1], attempt_cycle_cap=50,
                  heatmap_cycle_zero_at_top=True, heatmap_blank="white", recovery_cycles_excluded_from_heatmap=True,
                  heatmap_primary_cycles={m: np.isfinite(shown[m]).sum(axis=1).tolist() for m in heatmap_methods},
                  evaluation_recoveries_by_seed=dict(Counter(str(r["seed"]) for r in recovery)),
                  evaluation_recoveries_by_policy=dict(Counter(r["policy"] for r in recovery)),
                  checks=dict(raw_hashes=True, exact_coverage=True, independent_selection_recomputed=True,
                              prepared_source_environment=True, frozen_setup_controller=True,
                              timing_accounting=True, matches_analysis_and_export=True, actual_action_traces=True),
                  source_sha256=sources,
                  generated_sha256={str(p.relative_to(paper)): sha(p) for p in
                                    [paper / "sections/current_matched_rows.tex", review / "displayed_weights.npz",
                                     *[paper / "figures" / f"{name}.{ext}" for name in
                                       ("matched_policy_savings", "matched_policy_weights") for ext in ("svg", "png")]]})
    report["publisher_sha256"] = sha(Path(__file__))
    (review / "publication_data.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "audit", "native_savings", "illustrative_seed",
                                          "displayed_cycle_indices", "evaluation_recoveries_by_seed")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--paper", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    publish(args.run.resolve(), args.runtime.resolve(), args.paper.resolve())
