"""Tables and figures for Module 1."""
from __future__ import annotations
from collections import defaultdict

import matplotlib

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

import numpy as np

import cycle_examples

METHODS = ("default_setup_default_solve", "default_setup_lstdq", "bandit_default", "bandit_lstdq")


LABELS = ("Default", "LSTDQ", "LinUCB", "LinUCB + LSTDQ")


TEX_LABELS = ("Default", "LSTDQ", "LinUCB", "LinUCB--LSTDQ")


ZISSOU = ("#3A9AB2", "#6FB2C1", "#91BAB6", "#A5C2A3", "#BDC881", "#DCCB4E", "#E3B710", "#E79805", "#EC7A05", "#EF5703", "#F11B00")


WEIGHT_CMAP = LinearSegmentedColormap.from_list("module1_zissou", ZISSOU, N=256)
WEIGHT_CMAP.set_bad("white")


METHOD_WEIGHT_ANCHORS = (1.6, 1.0, 3.0)


COLORS = ("#273046",) + tuple(to_hex(WEIGHT_CMAP((weight - 1) / 2))
                              for weight in METHOD_WEIGHT_ANCHORS)


CUMULATIVE_COLORS = COLORS[:3] + ("#F11B00",)


HEATMAP_PROBLEM_RANGE = (1, 2500)


def overhead_tint(color):
    """Keep the phase hue while making recurring overhead visually lighter."""
    return to_hex(.45 * np.asarray(to_rgb(color)) + .55)


BAR_COMPONENTS = ((0, "Native setup", COLORS[2]),
                  (4, "Setup learner overhead", overhead_tint(COLORS[2])),
                  (1, "Native solve", COLORS[1]),
                  (3, "Solve controller overhead", overhead_tint(COLORS[1])))


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
        "svg.hashsalt": "module1-current",
        "savefig.bbox": "tight", "savefig.pad_inches": .04})


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
        for ext, dpi in (("svg", 600), ("png", 260 if name == "individual_cycle_schedules" else 180)):
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
    with matplotlib.rc_context({"svg.fonttype": "path", "axes.titlesize": 8.5}):
        save(cycle_examples.draw(examples), "individual_cycle_schedules")
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
        (paper/"sections"/f"module1_{name}_rows.tex").write_text("% Generated from audited current Module 1 data.\n"+"\n".join(rows)+"\n")


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
    values = dict(ModuleOneSeedCoverage=coverage, ModuleOneGrid=label.split("_")[-1],
                  ModuleOneSeed=seed, ModuleOneSeedCount=str(counts[label]),
                  ModuleOneSeedCountWord=number(counts[label]))
    for family, tex_family in [("diffusion", "Diffusion"), ("advection", "Advection")]:
        for window, tex_window in [("full", "Full"), ("last_1000", "Last")]:
            reductions = [next(r["total_reduction"] for r in aggregates
                          if r["case"] == f"{family}_{size}" and r["window"] == window
                          and r["method"] == METHODS[3]) for size in (40,60,80)]
            values[f"ModuleOne{tex_family}{tex_window}Savings"] = joined([rf"{v:.1f}\%" for v in reductions])
    for name, method, field in [
        ("DefaultSetup",METHODS[0],"setup_ms"), ("BanditSetup",METHODS[2],"setup_ms"),
        ("DefaultSolve",METHODS[0],"solve_ms"), ("BanditSolve",METHODS[2],"solve_ms"),
        ("JointSetup",METHODS[3],"setup_ms"), ("JointSolve",METHODS[3],"solve_ms"),
        ("FixedRLSolve",METHODS[1],"solve_ms")]:
        values["ModuleOne"+name] = f"{aggregate(method,field):.1f}"
    for name, method, field, window in [
        ("SelectedDefaultCost",METHODS[0],"total_ms","full"),
        ("SelectedJointCost",METHODS[3],"total_ms","full"),
        ("SelectedFullReduction",METHODS[3],"total_reduction","full"),
        ("SelectedJointLastCost",METHODS[3],"total_ms","last_1000"),
        ("SelectedLastReduction",METHODS[3],"total_reduction","last_1000")]:
        values["ModuleOne"+name] = f"{record(method,window)[field]:.1f}"
    for window, name in [("full","Full"), ("last_1000","Last")]:
        reduction = 100*(1-record(METHODS[3],window)["total_ms"]/record(METHODS[2],window)["total_ms"])
        values[f"ModuleOneSelectedVsBandit{name}"] = f"{reduction:.1f}"
    metrics = cases[label][seed]["result"]["methods"]
    failures = [str(metrics[m]["first_failures"]) for m in METHODS]
    fallbacks = [str(metrics[m]["fallbacks"]) for m in METHODS]
    values["ModuleOneRetrySummary"] = (
        "In the selected run, Default, LSTDQ, LinUCB, and LinUCB--LSTDQ have "
        + joined(failures) + " first-attempt failures, respectively. All inputs ultimately complete. "
        + "The corresponding numbers of inputs requiring default fallback are " + joined(fallbacks)
        + "; remaining first-attempt failures complete through learned setup retries.")
    values["ModuleOneExampleIDs"] = joined([str(row["problem_index"]) for row in examples["examples"]])
    values["ModuleOneExampleCycles"] = joined([str(row["cycles"]) for row in examples["examples"]])
    values["ModuleOneExampleGrid"] = str(examples["grid"])
    values["ModuleOneExampleSeed"] = str(examples["seed"])
    values["ModuleOneExampleInitialWeights"] = joined([f'{row["weights"][0]:.2f}' for row in examples["examples"]])
    destination = paper / "sections/module1_values.tex"
    destination.write_text("% Generated from the same accepted records as the Module 1 figures.\n"
        + "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name,value in values.items()) + "\n")
    return values
