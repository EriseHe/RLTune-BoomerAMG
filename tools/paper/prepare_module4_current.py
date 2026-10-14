"""Publish completed Module 4 evidence from saved, audited experiment records.

Read-only with respect to the experiment repository: no solver or learner runs.
The official data registry selects the complete Module 4 study.
Every cached array and original trajectory is checked against its saved hash.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
import numpy as np
import module4_policy_figure
import official_paths

METHODS = ("default_setup_default_solve", "default_setup_lstdq", "bandit_default", "bandit_lstdq")
LABELS = ("Default", "LSTDQ", "LinUCB", "LinUCB + LSTDQ")
TEX_LABELS = ("Default", "LSTDQ", "LinUCB", "LinUCB--LSTDQ")
FIELDS = ("setup_runtime", "native_solve_runtime", "native_runtime", "infer_runtime", "bandit_overhead_runtime", "end_to_end_runtime")
# Restore the previously used Wes Anderson Zissou palette from the GitHub
# reference, with the same interpolation over relaxation weights 1–3.
ZISSOU = ("#3A9AB2", "#6FB2C1", "#91BAB6", "#A5C2A3", "#BDC881", "#DCCB4E", "#E3B710", "#E79805", "#EC7A05", "#EF5703", "#F11B00")
WEIGHT_CMAP = LinearSegmentedColormap.from_list("module4_zissou", ZISSOU, N=256)
WEIGHT_CMAP.set_bad("white")
# Sample the displayed heatmap scale directly: LinUCB at 1, LSTDQ at the
# green anchor near 1.6, and Joint at 3. Default keeps its charcoal dash.
METHOD_WEIGHT_ANCHORS = (1.6, 1.0, 3.0)
COLORS = ("#273046",) + tuple(to_hex(WEIGHT_CMAP((weight - 1) / 2))
                              for weight in METHOD_WEIGHT_ANCHORS)
# User preference: Figure 3 always highlights LinUCB + LSTDQ in this red,
# independently of the heatmap palette (main panels, inset, and legend).
CUMULATIVE_COLORS = COLORS[:3] + ("#F11B00",)
HEATMAP_PROBLEM_RANGE = (1, 2500)


def overhead_tint(color):
    """Keep the phase hue while making recurring overhead visually lighter."""
    return to_hex(.45 * np.asarray(to_rgb(color)) + .55)


BAR_COMPONENTS = ((0, "Native setup", COLORS[2]),
                  (4, "Setup learner overhead", overhead_tint(COLORS[2])),
                  (1, "Native solve", COLORS[1]),
                  (3, "Solve controller overhead", overhead_tint(COLORS[1])))


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def style():
    # Serif typography and method/phase colors from the current heatmap scale.
    plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": "#202020", "axes.labelcolor": "#202020",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.linewidth": .65,
        "axes.grid": False,
        "grid.color": "#dddddd", "grid.linewidth": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "svg.hashsalt": "module4-current",
        "savefig.bbox": "tight", "savefig.pad_inches": .04})


def load(code):
    campaign = official_paths.module_root(code, "module04")
    schedule = read(campaign / "campaign.json")
    cases, sources, components = defaultdict(dict), {}, []
    native_identity = numerical_identity = None
    for job in schedule["jobs"]:
        root = campaign / job["output"]
        if read(root / "status.json")["status"] != "complete":
            continue
        protocol = read(root / "protocol.json")
        assert sha(root / "protocol.json") == job["protocol_sha256"]
        assert read(root / "completion_audit.json")["passed"]
        summary = read(root / "summary.json")
        assert summary["passed"]
        assert tuple(protocol["methods"]) == METHODS
        assert not protocol["shared_prefix"] and not protocol["copied_measurements"]
        assert protocol["feedback_mode"] == "adjacent_delayed"
        assert protocol["learned_attempts"] == 3 and protocol["default_fallback_attempts"] == 1
        assert not protocol["rl_recovery_cost"] and protocol["setup_failure_observation_retention"]
        identity = read(root / "source_manifest.json")
        if numerical_identity is not None:
            assert identity == numerical_identity
            assert protocol["native_sha256"] == native_identity
        numerical_identity, native_identity = identity, protocol["native_sha256"]
        # Follow the campaign's accepted result path for every completed batch.
        # A replacement must never inherit an earlier export's cached arrays.
        cache_manifest_path = root / "paper_figures/manifest.json"
        cache_manifest = read(cache_manifest_path)
        assert cache_manifest["checks"]["passed"]
        assert cache_manifest["checks"]["paired_inputs"]
        cache_sources = {official_paths.recorded_path(code, name).resolve(): value
                         for name, value in cache_manifest["source_sha256"].items()}
        cache = root / "paper_figures/plot_data.npz"
        expected = cache_manifest["output_sha256"]["plot_data.npz"]
        assert sha(cache) == expected
        with np.load(cache, allow_pickle=False) as arrays:
            assert tuple(arrays["methods"].tolist()) == METHODS
            assert tuple(arrays["timing_fields"].tolist()) == FIELDS
            assert arrays["seeds"].tolist() == job["seeds"]
            timing = arrays["timings_sec"].copy()
            weights = arrays["first_primary_weights"].copy()
            residuals = arrays["first_primary_residuals"].copy()
        assert timing.shape == (len(job["seeds"]), 4, 5000, 6)
        assert np.isfinite(timing).all() and (timing >= 0).all()
        np.testing.assert_allclose(timing[..., 0] + timing[..., 1], timing[..., 2], atol=1e-8)
        np.testing.assert_allclose(timing[..., 2] + timing[..., 3] + timing[..., 4], timing[..., 5], atol=1e-8)
        valid_weights = weights[np.isfinite(weights)]
        assert ((valid_weights >= 1) & (valid_weights <= 3)).all()
        np.testing.assert_allclose((valid_weights - 1) / .05, np.round((valid_weights - 1) / .05), atol=1e-8)
        assert (np.diff(np.isfinite(weights).sum(axis=2), axis=2) <= 0).all()
        print("Verifying completed records:", job["experiment"], ",".join(job["seeds"]), flush=True)
        for index, seed in enumerate(job["seeds"]):
            stage = root / f"seed_{seed}"
            cfg = read(stage / "experiment_config.json")
            result = read(stage / "result.json")
            assert result["complete"] and result["passed"] and result["cases"] == 5000
            assert cfg["setup"]["candidate_schedule"]["pool_size"] == 1024
            assert cfg["stream"]["warmup_cases"] == 0
            assert all(m.get("solve_activation_case", 0) == 0 for m in cfg["methods"])
            assert sha(stage / "experiment_config.json") == protocol["stages"][seed]["config_sha256"]
            catalog_path = stage / "aot_candidate_schedules/expanded_no_endpoints/structured512/metadata.json"
            assert read(catalog_path)["catalog"]["action_count"] == 16128000
            for method_index, method in enumerate(METHODS):
                trajectory = stage / "trajectories" / f"{method}.jsonl"
                actual_hash = sha(trajectory)
                assert actual_hash == result["audits"][method]["trajectory_sha256"]
                assert actual_hash == cache_sources[trajectory.resolve()]
                sources[str(trajectory)] = actual_hash
                for window, start in (("full", 0), ("last_1000", 4000)):
                    row = next(r for r in summary["rows"] if r["seed"] == seed and r["method"] == method and r["window"] == window)
                    assert row["cases"] == 5000 - start and row["failures"] == 0
                    totals = timing[index, method_index, start:].sum(axis=0)
                    np.testing.assert_allclose(totals[[2, 5]], [row["native_sec"], row["total_sec"]], atol=1e-7)
            assert seed not in cases[job["experiment"]]
            cases[job["experiment"]][seed] = dict(a=timing[index], w=weights[index], r=residuals[index], cfg=cfg, result=result, stage=stage)
            for name in ("experiment_config.json", "result.json", "inputs.jsonl", "initial_audit.json"):
                sources[str(stage / name)] = sha(stage / name)
        for name in ("protocol.json", "summary.json", "completion_audit.json", "source_manifest.json"):
            sources[str(root / name)] = sha(root / name)
        sources[str(cache)] = expected
        sources[str(cache_manifest_path)] = sha(cache_manifest_path)
        components.append(dict(case=job["experiment"], seeds=job["seeds"], accepted_path=str(root), resolved_path=str(root.resolve())))
    assert len(cases) == 6
    upwind = read(campaign / "upwind_discretization_audit.json")
    assert upwind["verified_discretization"] == "HYPRE upwind (atype=3)"
    assert upwind["native_sha256"] == native_identity["hypre/interfaces/libamg_runtime.dylib"]
    sources[str(campaign / "upwind_discretization_audit.json")] = sha(campaign / "upwind_discretization_audit.json")
    sources[str(campaign / "campaign.json")] = sha(campaign / "campaign.json")
    return cases, sources, components


def statistics(cases):
    aggregates, records, ranking = [], [], []
    for label in sorted(cases, key=lambda v: (v.startswith("advection"), int(v.split("_")[-1]))):
        seeds = sorted(cases[label], key=int)
        a = np.stack([cases[label][seed]["a"] for seed in seeds])
        for window, start in (("full", 0), ("last_1000", 4000)):
            totals = a[:, :, start:].sum(axis=2)
            native = 100 * (1 - totals[:, :, 2] / totals[:, :1, 2])
            measured = 100 * (1 - totals[:, :, 5] / totals[:, :1, 5])
            for j, method in enumerate(METHODS):
                means = totals[:, j] * 1000 / (5000 - start)
                aggregate = dict(case=label, window=window, method=method, n=len(seeds), seeds=seeds,
                    setup_ms=float(means[:, 0].mean()), solve_ms=float(means[:, 1].mean()),
                    native_ms=float(means[:, 2].mean()), total_ms=float(means[:, 5].mean()),
                    total_ms_sd=float(means[:, 5].std(ddof=1)),
                    native_reduction=float(native[:, j].mean()), native_reduction_sd=float(native[:, j].std(ddof=1)),
                    total_reduction=float(measured[:, j].mean()), total_reduction_sd=float(measured[:, j].std(ddof=1)))
                aggregates.append(aggregate)
                for i, seed in enumerate(seeds):
                    record = dict(case=label, window=window, seed=seed, method=method,
                        setup_ms=float(means[i, 0]), solve_ms=float(means[i, 1]), native_ms=float(means[i, 2]),
                        total_ms=float(means[i, 5]), native_reduction=float(native[i, j]), total_reduction=float(measured[i, j]))
                    records.append(record)
                    if window == "full" and j == 3 and label.startswith("diffusion"):
                        ranking.append(record)
    return aggregates, records, max(ranking, key=lambda r: r["total_reduction"]), max(ranking, key=lambda r: r["native_reduction"])


def heatmap_display_indices(chosen):
    """Keep the 50-cycle cap visible while compacting the standalone panel."""
    cap = chosen["w"].shape[-1]
    assert cap == 50
    return (list(range(21)) + [cap - 1], list(range(cap)))


def cycle_heatmap(chosen, title):
    first_problem, last_problem = HEATMAP_PROBLEM_RANGE
    assert 1 <= first_problem <= last_problem <= chosen["w"].shape[1]
    weights = chosen["w"][:, first_problem-1:last_problem, :]
    problem_ticks = np.linspace(first_problem-1, last_problem, 6, dtype=int)
    problem_ticks[0] = first_problem
    displayed = heatmap_display_indices(chosen)
    last = displayed[0][-1]
    cmap = WEIGHT_CMAP
    fig = plt.figure(figsize=(6.85, 3.1))
    grid = fig.add_gridspec(2, 1, height_ratios=[25, 50], hspace=.43,
                          left=.10, right=.87, bottom=.15, top=.84)
    upper = grid[0].subgridspec(2, 1, height_ratios=[21, 1.2], hspace=.45)
    top = fig.add_subplot(upper[0])
    endpoint = fig.add_subplot(upper[1], sharex=top)
    joint = fig.add_subplot(grid[1], sharex=top)
    for ax, q, first, final in [(top, 0, 0, 20), (endpoint, 0, last, last),
                               (joint, 1, 0, last)]:
        im = ax.imshow(weights[q, :, first:final+1].T, origin="upper", aspect="auto",
                       interpolation="nearest", extent=(first_problem-.5, last_problem+.5, final+.5, first-.5),
                       cmap=cmap, vmin=1, vmax=3, rasterized=True)
        ax.set_xlim(first_problem-.5, last_problem+.5)
        ax.set_xticks(problem_ticks)
    top.set(ylabel=r"Cycle index $k$", yticks=[0, 10, 20])
    top.set_title("(a) LSTDQ", loc="left")
    top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    top.spines["bottom"].set_visible(False)
    endpoint.set_yticks([last])
    endpoint.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    endpoint.spines["top"].set_visible(False)
    for ax, y in [(top, 0), (endpoint, 1)]:
        ax.plot([0, 1], [y, y], transform=ax.transAxes, linestyle="none",
                marker=[(-1, -.6), (1, .6)], markersize=4.2,
                markeredgewidth=.65, color="#202020", clip_on=False)
    joint.set(ylabel=r"Cycle index $k$", yticks=[0, 10, 20, 30, 40, last],
              xlabel="Problem index")
    joint.set_title("(b) LinUCB + LSTDQ", loc="left")
    color_ax = fig.add_axes([.90, .15, .017, .69])
    fig.colorbar(im, cax=color_ax, label="Relaxation weight", ticks=[1, 1.5, 2, 2.5, 3])
    fig.suptitle("RL relaxation-weight choices (" + title + ")", y=.965, fontsize=9)
    return fig


def plots(cases, selected, figures, examples):
    label, seed = selected["case"], selected["seed"]
    case = cases[label]
    a = np.stack([case[s]["a"] for s in sorted(case, key=int)])
    chosen = case[seed]
    title = rf'Diffusion ${label.split("_")[-1]}^3$'
    style()
    figures.mkdir(parents=True, exist_ok=True)
    def save(fig, name):
        for ax in fig.axes:
            ax.grid(False, which="both")
            for child in ax.child_axes:
                child.grid(False, which="both")
        for ext, dpi in (("svg", 600), ("png", 180)):
            path = figures / f"{name}.{ext}"
            fig.savefig(path, dpi=dpi, metadata={"Date": None} if ext == "svg" else None)
            if ext == "svg":
                path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.05), sharey=True)
    for ax, start, letter in zip(axes, (0, 4000), ("a", "b")):
        costs = a[:, :, start:].mean(axis=2) * 1000
        means = costs.mean(axis=0)
        left = np.zeros(4)
        for field, name, color in BAR_COMPONENTS:
            ax.barh(np.arange(4), means[:, field], left=left, color=color, height=.64, label=name)
            left += means[:, field]
        for j in range(4):
            ax.text(left[j] + left.max() * .025, j, f'{costs[:, j, 5].mean():.1f} ± {costs[:, j, 5].std(ddof=1):.1f}', va="center", fontsize=6.7)
        ax.set(xlabel="Measured cost (ms/input)", xlim=(0, left.max() * 1.49), yticks=range(4), yticklabels=LABELS)
        ax.set_title(f'({letter}) Problems {start+1}-5000', loc="left")
    axes[0].invert_yaxis()
    fig.suptitle("Runtime components (" + title + f', {len(case)} seeds)', y=.995, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.01), fontsize=6.6, columnspacing=1.2, handlelength=1.5)
    fig.subplots_adjust(left=.17, right=.985, top=.84, bottom=.24, wspace=.18)
    save(fig, "runtime_breakdown")
    fig, axes = plt.subplots(1, 2, figsize=(6.85, 3.15))
    totals = np.column_stack([np.zeros(4), np.cumsum(chosen["a"][:, :, 5], axis=1)])
    x = np.arange(5001)
    for j in range(4):
        axes[0].plot(x, totals[j], color=CUMULATIVE_COLORS[j], ls="--" if j == 0 else "-", lw=1.3, label=LABELS[j] + f' ({totals[j,-1]:.1f} s)')
        if j:
            axes[1].plot(x[1:], 100*(1-totals[j,1:]/totals[0,1:]), color=CUMULATIVE_COLORS[j], lw=1.15)
    axes[1].axhline(0, color=CUMULATIVE_COLORS[0], ls="--", lw=.8)
    for ax in axes:
        ax.set(xlabel="Problem index", xlim=(0,5000), xticks=np.arange(0,5001,1000))
    axes[0].set(ylabel="Cumulative measured time (s)", ylim=(0,totals[0,-1]*1.06))
    axes[0].set_title("(a) Cumulative cost", loc="left")
    zoom = axes[0].inset_axes([.03, .54, .44, .44])
    # A square physical frame; the two data axes retain independent scales.
    zoom.set_box_aspect(1)
    zoom.set_gid("first_1000_square_inset")
    for j in range(4):
        zoom.plot(x[:1001], totals[j, :1001], color=CUMULATIVE_COLORS[j], ls="--" if j == 0 else "-", lw=.6)
    zoom.set(xlim=(0, 1000), ylim=(0, totals[:, :1001].max()*1.04),
             xticks=[0, 500, 1000])
    zoom.tick_params(labelsize=5.5, length=2, pad=1)
    for spine in zoom.spines.values():
        spine.set_visible(True)
        spine.set_color("#888888")
        spine.set_linewidth(.55)
    axes[0].indicate_inset_zoom(zoom, edgecolor="#777777", alpha=.65, linewidth=.6)
    axes[1].set(ylabel="Reduction relative to Default (%)")
    axes[1].set_title("(b) Cumulative reduction", loc="left")
    fig.suptitle("Cumulative solver cost (" + title + f', seed {seed})', y=.995, fontsize=9)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=2,
               frameon=False, fontsize=6.6, bbox_to_anchor=(.53,.94))
    fig.subplots_adjust(left=.10, right=.985, top=.72, bottom=.16, wspace=.45)
    save(fig, "cumulative_times")
    save(cycle_heatmap(chosen, title + f", seed {seed}"), "learned_cycle_weights")
    # Outline only this figure's labels so Inkscape reproduces the preview
    # exactly even when the manuscript host lacks the Matplotlib serif font.
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        save(module4_policy_figure.draw(examples), "individual_cycle_schedules")
    for extension in ("svg", "png"):
        (figures / f"mean_cycle_weights.{extension}").unlink(missing_ok=True)


def write_rows(paper, aggregates, records, selected):
    def get(label, method, window="full"):
        return next(r for r in aggregates if r["case"]==label and r["method"]==method and r["window"]==window)
    def pct(row):
        return rf'\({row["total_reduction"]:.1f}\pm{row["total_reduction_sd"]:.1f}\)\%'
    labels=sorted({r["case"] for r in aggregates},key=lambda v:(v.startswith("advection"),int(v.split("_")[-1])))
    full=[]; reduction=[]
    for label in labels:
        family="Diffusion--advection" if label.startswith("advection") else "Diffusion"
        size=label.split("_")[-1]; n=get(label,METHODS[0])["n"]
        if full: full.append(r"\addlinespace[3pt]")
        full.append(rf'\multicolumn{{5}}{{@{{}}l}}{{\textit{{{family}, \({size}^3\)}}}} \\')
        for method,name in zip(METHODS,TEX_LABELS):
            row=get(label,method)
            reduction_cell=r"---" if method==METHODS[0] else pct(row)
            full.append(f'{name} & {row["setup_ms"]:.1f} & {row["solve_ms"]:.1f} & {row["total_ms"]:.1f} & {reduction_cell} '+r"\\")
        if size=="40": reduction.append(rf'\multicolumn{{7}}{{@{{}}l}}{{\textit{{{family}}}}} \\')
        cells=[pct(get(label,m,w)) for w in ("full","last_1000") for m in METHODS[1:]]
        reduction.append(rf'\({size}^3\) & '+" & ".join(cells)+r" \\")
    chosen=[]
    for method,name in zip(METHODS,TEX_LABELS):
        rows=[next(r for r in records if r["case"]==selected["case"] and r["seed"]==selected["seed"] and r["method"]==method and r["window"]==w) for w in ("full","last_1000")]
        first,last=rows
        p0=r"---" if method==METHODS[0] else rf'{first["total_reduction"]:.1f}\%'
        p1=r"---" if method==METHODS[0] else rf'{last["total_reduction"]:.1f}\%'
        chosen.append(f'{name} & {first["setup_ms"]:.1f} & {first["solve_ms"]:.1f} & {first["total_ms"]:.1f} & {p0} & {last["total_ms"]:.1f} & {p1} '+r"\\")
    for name,rows in (("full",full),("reductions",reduction),("selected",chosen)):
        (paper/"sections"/f"module4_{name}_rows.tex").write_text("% Generated from audited current Module 4 data; unreviewed R99.\n"+"\n".join(rows)+"\n")


def write_values(paper, cases, aggregates, records, selected, examples):
    """Keep caption facts and narrative numbers tied to the plotted records."""
    label, seed = selected["case"], selected["seed"]
    def aggregate(method, field, window="full"):
        return next(r[field] for r in aggregates
                    if r["case"] == label and r["method"] == method and r["window"] == window)
    def record(method, window="full"):
        return next(r for r in records if r["case"] == label and r["seed"] == seed
                    and r["method"] == method and r["window"] == window)
    def number(value):
        return {1:"one", 2:"two", 3:"three", 4:"four", 5:"five", 6:"six"}.get(value, str(value))
    def joined(items):
        return items[0] if len(items) == 1 else (" and ".join(items) if len(items) == 2
               else ", ".join(items[:-1]) + ", and " + items[-1])
    counts = {name: len(rows) for name, rows in cases.items()}
    if len(set(counts.values())) == 1:
        coverage = f"{number(next(iter(counts.values())))} seed settings for every family and grid"
    elif all(counts[f"diffusion_{size}"] == counts[f"advection_{size}"] for size in (40,60,80)):
        groups = defaultdict(list)
        for size in (40,60,80):
            groups[counts[f"diffusion_{size}"]].append(rf"\({size}^3\)")
        coverage = joined([f"{number(n)} seed settings at {joined(grids)}" for n, grids in groups.items()])
        coverage += ", for both problem families"
    else:
        coverage = "; ".join(f"{name.replace('_',' ')}: {n} seed settings" for name,n in counts.items())
    values = dict(ModuleFourSeedCoverage=coverage, ModuleFourGrid=label.split("_")[-1],
                  ModuleFourSeed=seed, ModuleFourSeedCount=str(counts[label]),
                  ModuleFourSeedCountWord=number(counts[label]))
    for family, tex_family in [("diffusion", "Diffusion"), ("advection", "Advection")]:
        for window, tex_window in [("full", "Full"), ("last_1000", "Last")]:
            reductions = [next(r["total_reduction"] for r in aggregates
                          if r["case"] == f"{family}_{size}" and r["window"] == window
                          and r["method"] == METHODS[3]) for size in (40,60,80)]
            values[f"ModuleFour{tex_family}{tex_window}Savings"] = joined([rf"{v:.1f}\%" for v in reductions])
    for name, method, field in [
        ("DefaultSetup",METHODS[0],"setup_ms"), ("BanditSetup",METHODS[2],"setup_ms"),
        ("DefaultSolve",METHODS[0],"solve_ms"), ("BanditSolve",METHODS[2],"solve_ms"),
        ("JointSetup",METHODS[3],"setup_ms"), ("JointSolve",METHODS[3],"solve_ms"),
        ("FixedRLSolve",METHODS[1],"solve_ms")]:
        values["ModuleFour"+name] = f"{aggregate(method,field):.1f}"
    for name, method, field, window in [
        ("SelectedDefaultCost",METHODS[0],"total_ms","full"),
        ("SelectedJointCost",METHODS[3],"total_ms","full"),
        ("SelectedFullReduction",METHODS[3],"total_reduction","full"),
        ("SelectedJointLastCost",METHODS[3],"total_ms","last_1000"),
        ("SelectedLastReduction",METHODS[3],"total_reduction","last_1000")]:
        values["ModuleFour"+name] = f"{record(method,window)[field]:.1f}"
    for window, name in [("full","Full"), ("last_1000","Last")]:
        reduction = 100*(1-record(METHODS[3],window)["total_ms"]/record(METHODS[2],window)["total_ms"])
        values[f"ModuleFourSelectedVsBandit{name}"] = f"{reduction:.1f}"
    metrics = cases[label][seed]["result"]["methods"]
    failures = [str(metrics[m]["first_failures"]) for m in METHODS]
    fallbacks = [str(metrics[m]["fallbacks"]) for m in METHODS]
    values["ModuleFourRetrySummary"] = (
        "In the selected run, Default, LSTDQ, LinUCB, and LinUCB--LSTDQ have "
        + joined(failures) + " first-attempt failures, respectively. All inputs ultimately complete. "
        + "The corresponding numbers of inputs requiring default fallback are " + joined(fallbacks)
        + "; remaining first-attempt failures complete through learned setup retries.")
    values["ModuleFourExampleIDs"] = joined([str(row["problem_index"]) for row in examples["examples"]])
    values["ModuleFourExampleCycles"] = joined([str(row["cycles"]) for row in examples["examples"]])
    values["ModuleFourExampleGrid"] = str(examples["grid"])
    values["ModuleFourExampleSeed"] = str(examples["seed"])
    values["ModuleFourExampleInitialWeights"] = joined([f'{row["weights"][0]:.2f}' for row in examples["examples"]])
    destination = paper / "sections/module4_values.tex"
    destination.write_text("% Generated from the same accepted records as the Module 4 figures.\n"
        + "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name,value in values.items()) + "\n")
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code",type=Path,required=True); parser.add_argument("--paper",type=Path,required=True)
    args=parser.parse_args(); code=args.code.resolve(); paper=args.paper.resolve()
    cases,sources,components=load(code)
    aggregates,records,selected,best_native=statistics(cases)
    examples=module4_policy_figure.prepare(cases[module4_policy_figure.EXAMPLE_CASE][module4_policy_figure.EXAMPLE_SEED])
    sources.update(examples["source_sha256"])
    plots(cases,selected,paper/"figures/module4_20261010",examples)
    write_rows(paper,aggregates,records,selected)
    values=write_values(paper,cases,aggregates,records,selected,examples)
    output=paper/"review/module4_20261010"; output.mkdir(parents=True,exist_ok=True)
    sources[str(Path(__file__).resolve())]=sha(Path(__file__).resolve())
    policy_source=Path(module4_policy_figure.__file__).resolve()
    sources[str(policy_source)]=sha(policy_source)
    with (output/"seed_summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]),lineterminator="\n"); writer.writeheader(); writer.writerows(records)
    result=dict(status="unreviewed",cases_per_seed=5000,methods=METHODS,candidate_pool=1024,activation_case=0,
        completed_seed_counts={label:len(seeds) for label,seeds in cases.items()},
        paired_inputs=sum(len(seeds)*5000 for seeds in cases.values()),final_failures=0,
        aggregates=aggregates,selected_run=selected,best_native_run=best_native,individual_examples=examples, figures_review_status="accepted",
        selection_rule="Largest full-stream paired total-time reduction among completed diffusion runs",
        accepted_components=components,source_sha256=sources,numerical_rerun=False,
        generated_sha256={str(path.relative_to(paper)):sha(path)
            for path in sorted((paper/"figures/module4_20261010").glob("*"))
            if path.is_file() and path.suffix in (".svg", ".png")},
        figure_style="Method curves and native setup/solve bars sampled from the current heatmap; overhead bars use lighter phase-matched tints; Figure 3 always uses red for LinUCB + LSTDQ",
        figure_palette_source="https://github.com/karthik/wesanderson/blob/master/R/colors.R",
        heatmap_color_reference=dict(source_colormap=WEIGHT_CMAP.name,
            restored_from_revision="c8172f0", palette_control_colors=ZISSOU,
            source_weight_range=[1.0,3.0],
            anchors={str(weight):to_hex(WEIGHT_CMAP((weight-1)/2)) for weight in (1.,1.6,2.,3.)}),
        method_labels=LABELS, method_colors=COLORS,
        figure_specific_method_colors={"cumulative_times":dict(zip(LABELS,CUMULATIVE_COLORS))},
        method_color_weight_reference={"LinUCB":1.0, "LSTDQ":1.6, "LinUCB + LSTDQ":3.0},
        runtime_stack=[dict(field=FIELDS[field], label=name, color=color) for field,name,color in BAR_COMPONENTS],
        bar_native_setup_color_reference=dict(colormap=WEIGHT_CMAP.name, relaxation_weight=1.0, vmin=1, vmax=3,
                                              source="Current heatmap colorbar"),
        bar_native_solve_color_reference=dict(colormap=WEIGHT_CMAP.name, relaxation_weight=1.6, vmin=1, vmax=3,
                                              source="Current heatmap colorbar"),
        bar_overhead_tint=dict(native_color_fraction=.45, white_fraction=.55),
        cumulative_zoom=dict(first_problem=1, last_problem=1000, includes_zero_origin=True, box_aspect=1,
                             line_width_pt=.6, title=None, grid=False,
                             source="Exact prefix sums of the plotted measured totals"),
        generated_text_values=values, display_decimal_places=1, output_formats=["svg","png"], grids=False,
        heatmap_orientation="Cycle zero at top; increasing cycle index downward",
        heatmap_problem_window=dict(first_problem=HEATMAP_PROBLEM_RANGE[0],
                                    last_problem=HEATMAP_PROBLEM_RANGE[1]),
        heatmap_display_indices=heatmap_display_indices(cases[selected["case"]][selected["seed"]]))
    (output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("completed_seed_counts","paired_inputs","selected_run","best_native_run","final_failures")},indent=2))


if __name__=="__main__":
    main()
