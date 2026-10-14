"""Recreate the paper's Module 04 figures from a completed four-branch run.

Usage: python -m experiments.paper_final.plot_04_factorial RUN_DIRECTORY
No solver or learner is executed. Figures, plotted arrays, captions and hashes
are saved under RUN_DIRECTORY/paper_figures (or --output).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from experiments.runtime import configure_single_thread

configure_single_thread()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

METHODS = (
    "default_setup_default_solve", "default_setup_lstdq",
    "bandit_default", "bandit_lstdq",
)
LABELS = ("Default", "Default + LSTDQ", "LinUCB", "LinUCB + LSTDQ")
COLORS = ("#333333", "#009E73", "#0072B2", "#D55E00")
FIELDS = (
    "setup_runtime", "native_solve_runtime", "native_runtime",
    "infer_runtime", "bandit_overhead_runtime", "end_to_end_runtime",
)
RL_INDICES = (1, 3)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_run(root):
    protocol = json.loads((root / "protocol.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    completion = json.loads((root / "completion_audit.json").read_text())
    assert summary["passed"] and completion["passed"], "Run has not passed its audits"
    assert tuple(protocol["methods"]) == METHODS
    seeds = list(protocol["seeds"])
    sources = {str(root / name): sha(root / name) for name in
               ("protocol.json", "summary.json", "completion_audit.json")}
    timings, weights, residuals, configs = [], [], [], []
    for seed in seeds:
        stage = root / f"seed_{seed}"
        config_path = stage / "experiment_config.json"
        sources[str(config_path)] = sha(config_path)
        config = json.loads(config_path.read_text())
        result = json.loads((stage / "result.json").read_text())
        configs.append(config)
        n = protocol["stages"][seed]["cases"]
        cap = config["solve"]["max_cycles"]
        a = np.zeros((4, n, len(FIELDS)))
        w = np.full((2, n, cap), np.nan)
        residual = np.full((2, n, cap + 1), np.nan)
        residual[:, :, 0] = 1.
        paired = None
        specs = {m["id"]: m for m in config["methods"]}
        for j, method in enumerate(METHODS):
            source = stage / "trajectories" / f"{method}.jsonl"
            digest = hashlib.sha256()
            input_digest = hashlib.sha256()
            count = 0
            with source.open("rb") as handle:
                for i, line in enumerate(handle):
                    digest.update(line)
                    row = json.loads(line)
                    assert i < n and row["online_index"] == i
                    assert row["problem_index"] == i + 1 and row["method"] == method
                    input_digest.update(json.dumps(
                        [row["mkw"], row["context"]], sort_keys=True,
                        separators=(",", ":"), allow_nan=False).encode())
                    o = row["outcome"]
                    v = np.array([o[k] for k in FIELDS])
                    assert np.isfinite(v).all() and (v >= 0).all()
                    assert abs(v[0] + v[1] - v[2]) < 1e-8
                    assert abs(v[2] + v[3] + v[4] - v[5]) < 1e-8
                    a[j, i] = v
                    if j in RL_INDICES:
                        q = RL_INDICES.index(j)
                        # The outcome-level trace may describe the final learned
                        # attempt after reselection. Use the first attempt explicitly.
                        first = o["primary_attempts"][0]
                        cycles = int(first["cycles"])
                        assert 0 <= cycles <= cap
                        active = i >= specs[method]["solve_activation_case"]
                        assert bool(row["rl_active"]) == active
                        actions = np.asarray(first["cycle_actions"], dtype=float)
                        if not active:
                            assert actions.size == 0
                            actions = np.ones(cycles)
                        assert actions.size == cycles
                        grid = config["solve"]["action_grid"]
                        assert ((actions >= grid["start"] - 1e-9) &
                                (actions <= grid["stop"] + 1e-9)).all()
                        steps = (actions - grid["start"]) / grid["step"]
                        assert np.allclose(steps, np.round(steps), atol=1e-8)
                        w[q, i, :cycles] = actions
                        if active:
                            r = np.asarray(first["cycle_residuals"], dtype=float)
                            assert r.size == cycles
                            assert (r[np.isfinite(r)] >= 0).all()
                            if first["status"] == "success":
                                assert np.isfinite(r).all()
                            r[~np.isfinite(r)] = np.nan
                            if cycles:
                                residual[q, i, 1:] = r[np.minimum(np.arange(cap), cycles - 1)]
                            else:
                                residual[q, i, 1:] = 1. if first["status"] == "success" else np.nan
                    count += 1
            assert count == n
            sources[str(source)] = digest.hexdigest()
            assert sources[str(source)] == result["audits"][method]["trajectory_sha256"]
            if paired is None:
                paired = input_digest.hexdigest()
            assert input_digest.hexdigest() == paired, "Methods must share the exact input order"
            for record in (r for r in summary["rows"] if r["seed"] == seed and
                           r["method"] == method and r["window"] in ("full", "last_1000")):
                start = 0 if record["window"] == "full" else max(0, n - 1000)
                assert record["cases"] == n - start
                np.testing.assert_allclose(a[j, start:, 2].sum(), record["native_sec"], atol=1e-7, rtol=1e-10)
                np.testing.assert_allclose(a[j, start:, 5].sum(), record["total_sec"], atol=1e-7, rtol=1e-10)
        timings.append(a)
        weights.append(w)
        residuals.append(residual)
    assert all(c["problem"] == configs[0]["problem"] and
               c["solve"] == configs[0]["solve"] and
               c["methods"] == configs[0]["methods"] for c in configs)
    return protocol, seeds, np.stack(timings), np.stack(weights), np.stack(residuals), configs[0], sources


def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif", "font.size": 8,
        "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "text.color": "#202020", "axes.labelcolor": "#202020",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.axisbelow": True, "axes.linewidth": .65,
        "grid.color": "#dddddd", "grid.linewidth": .5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": .04,
    })


def index_ticks(n, *, zero=True):
    ticks = np.unique(np.round(np.linspace(0, n, min(6, n + 1))).astype(int))
    if not zero:
        ticks[0] = 1
    return ticks


def cumulative_panels(axes, data, activation, *, prefix="", legend_totals=True):
    n = data.shape[1]
    x = np.arange(n + 1)
    cumulative = np.column_stack([np.zeros(4), np.cumsum(data[:, :, 2], axis=1)])
    for j, label in enumerate(LABELS):
        if legend_totals:
            label += f" ({cumulative[j, -1]:.1f} s)"
        axes[0].plot(x, cumulative[j], label=label, color=COLORS[j],
                     lw=1.25, ls="--" if j == 0 else "-")
        if j:
            axes[1].plot(x[1:], 100 * (1 - cumulative[j, 1:] / cumulative[0, 1:]),
                         color=COLORS[j], lw=1.25)
    axes[0].set(ylim=(0, 1.04 * cumulative[0, -1]), ylabel="Cumulative native time (s)")
    axes[1].axhline(0, color=COLORS[0], ls="--", lw=.75)
    axes[1].set_ylabel("Native-time reduction (%)")
    axes[0].set_title(prefix + "(a) Cumulative native time", loc="left")
    axes[1].set_title(prefix + "(b) Cumulative reduction", loc="left")
    for ax in axes:
        if 0 < activation < n:
            ax.axvline(activation + .5, color="#777777", ls=":", lw=.9)
        ax.set(xlim=(0, n), xticks=index_ticks(n), xlabel="Problem index")
        ax.grid()
    return cumulative


def plot_cumulative(data, seed, title, activation):
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.95))
    cumulative_panels(axes, data, activation)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(f"{title} · seed {seed}", y=1.0, fontsize=9)
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(.53, .945), handlelength=2, columnspacing=1.6)
    fig.subplots_adjust(left=.095, right=.987, top=.73, bottom=.21, wspace=.37)
    note = ("Both LSTDQ branches active from problem 1" if activation == 0 else
            "Dotted line: joint LSTDQ starts at problem " + str(activation + 1))
    fig.text(.53, .018, note,
             ha="center", fontsize=7, color="#555555")
    return fig


def plot_all_seeds(a, seeds, title, activation):
    fig, axes = plt.subplots(len(seeds), 2, figsize=(6.75, 2.45 * len(seeds)), squeeze=False)
    cumulative = np.cumsum(a[:, :, :, 2], axis=2)
    reduction = 100 * (1 - cumulative[:, 1:] / cumulative[:, :1])
    ymin = 25 * np.floor(reduction.min() / 25) - 5
    ymax = 25 * np.ceil(reduction.max() / 25) + 5
    for i, seed in enumerate(seeds):
        cumulative_panels(axes[i], a[i], activation, prefix=f"Seed {seed} · ", legend_totals=False)
        axes[i, 0].set_ylim(0, 1.04 * cumulative[:, 0, -1].max())
        axes[i, 1].set_ylim(ymin, ymax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"{title} · all {len(seeds)} seeds", y=.995, fontsize=10)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.53, .97), columnspacing=1.2)
    fig.subplots_adjust(left=.095, right=.99, top=.915, bottom=.075, hspace=.48, wspace=.37)
    note = ("Exact prefix sums; both LSTDQ branches active from problem 1." if activation == 0 else
            "Exact prefix sums; dotted lines mark joint LSTDQ activation.")
    fig.text(.53, .012, note, ha="center", fontsize=7)
    return fig


def plot_breakdown(a, title, nseeds):
    components = ((0, "Native setup", "#66A99A"), (1, "Native solve", "#6E91C4"),
                  (3, "Solve learner", "#E8B35F"), (4, "Setup learner", "#B5B5B5"))
    n = a.shape[2]
    starts = (0, n - 1000) if n > 1000 else (0,)
    titles = (f"(a) Problems 1–{n}", f"(b) Problems {n - 999}–{n}")
    fig, grid = plt.subplots(1, len(starts), figsize=(7.05 if len(starts) == 2 else 4.8, 3.35),
                             sharey=True, squeeze=False)
    axes = grid[0]
    maximum = float(max(a[:, :, :, 5].mean(axis=2).max(), a[:, :, -1000:, 5].mean(axis=2).max()) * 1000)
    xmax = np.ceil((maximum + 85) / 50) * 50
    y = np.arange(4)
    for ax, start, label in zip(axes, starts, titles):
        seed_means = 1000 * a[:, :, start:].mean(axis=2)
        mean = seed_means.mean(axis=0)
        sd = seed_means[:, :, 5].std(axis=0, ddof=1) if nseeds > 1 else None
        left = np.zeros(4)
        for k, name, color in components:
            ax.barh(y, mean[:, k], left=left, height=.62, color=color,
                    edgecolor="white", linewidth=.3, label=name)
            left += mean[:, k]
        np.testing.assert_allclose(left, mean[:, 5], atol=1e-8, rtol=1e-10)
        for j in range(4):
            label = f"{left[j]:.1f}" + (f" ± {sd[j]:.1f}" if sd is not None else "")
            ax.text(left[j] + 6, j, label, va="center", fontsize=7)
        ax.set_yticks(y, LABELS)
        ax.set(xlim=(0, xmax), xticks=np.arange(0, xmax, 100), xlabel="Mean time (ms/problem)")
        ax.set_title(label, loc="left")
        ax.grid(axis="x")
        ax.tick_params(axis="y", length=0)
    axes[0].invert_yaxis()
    note = f"mean ± sample SD across {nseeds} seeds" if nseeds > 1 else "one seed; sample SD unavailable"
    fig.suptitle(f"{title} · {note}", y=.99, fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.56, .005), handlelength=1.4, columnspacing=1)
    fig.subplots_adjust(left=.2 if len(starts) == 2 else .3, right=.99, top=.84, bottom=.25, wspace=.12)
    return fig


def plot_weights(w, seed, title, activation):
    fig, axes = plt.subplots(2, 1, figsize=(6.9, 4.25), sharex=True, layout="constrained")
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("white")
    n, cap = w.shape[1:]
    for ax, j, matrix, letter in zip(axes, RL_INDICES, w, ("a", "b")):
        im = ax.imshow(matrix.T, origin="lower", aspect="auto", interpolation="none",
                       extent=(.5, n + .5, -.5, cap - .5), cmap=cmap, vmin=1, vmax=3)
        ax.set_title(f"({letter}) {LABELS[j]}", loc="left")
        ax.set(ylabel=r"Cycle index $k$", xticks=index_ticks(n, zero=False),
               yticks=np.unique(np.r_[np.arange(0, cap, 10), cap - 1]))
        if j == 3 and activation > 0:
            ax.axvline(activation + .5, color=COLORS[j], ls="--", lw=.9)
            ax.text(activation / 2, cap - 6, r"Fixed $w=1$", ha="center", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
            ax.text(activation + .05 * n, cap - 6, "RL active", fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        else:
            ax.text(.025, .88, "RL active from problem 1", transform=ax.transAxes, fontsize=7,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
    axes[-1].set_xlabel("Problem index")
    fig.colorbar(im, ax=axes, label="Relaxation weight", ticks=[1, 1.5, 2, 2.5, 3], pad=.02)
    fig.suptitle(f"{title} · seed {seed} · first primary attempt", fontsize=9)
    return fig


def plot_joint_atlas(w, seeds, title, activation):
    fig, grid = plt.subplots(len(seeds), 1, figsize=(6.9, 1.8 * len(seeds) + .55),
                             sharex=True, sharey=True, squeeze=False, layout="constrained")
    axes = grid[:, 0]
    n, cap = w.shape[2:]
    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("white")
    for ax, seed, matrix in zip(axes, seeds, w[:, 1]):
        im = ax.imshow(matrix.T, origin="lower", aspect="auto", interpolation="none",
                       extent=(.5, n + .5, -.5, cap - .5), cmap=cmap, vmin=1, vmax=3)
        ax.set_title(f"Seed {seed}", loc="left")
        ax.set(ylabel=r"Cycle index $k$", xticks=index_ticks(n, zero=False),
               yticks=np.unique(np.r_[np.arange(0, cap, 10), cap - 1]))
        if 0 < activation < n:
            ax.axvline(activation + .5, color=COLORS[3], ls="--", lw=.9)
    axes[-1].set_xlabel("Problem index")
    fig.colorbar(im, ax=list(axes), label="Relaxation weight", ticks=[1, 1.5, 2, 2.5, 3], pad=.02)
    fig.suptitle(f"{title} · LinUCB + LSTDQ · first primary attempt", fontsize=9)
    return fig


def cycle_profiles(w, residuals):
    count = np.isfinite(w).sum(axis=1)
    mean = np.divide(np.nansum(w, axis=1), count,
                     out=np.full(count.shape, np.nan, dtype=float), where=count > 0)
    assert np.all(np.diff(count, axis=1) <= 0)
    return count, mean, residuals.mean(axis=1)


def plot_profile(w, residuals, seed, title, tolerance, activation):
    start = max(0, w.shape[1] - 1000, activation)
    count, mean, residual_mean = cycle_profiles(w[:, start:], residuals[:, start:])
    width = w.shape[2]
    fig = plt.figure(figsize=(6.75, 3.45))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.25, 1.05], wspace=.42, hspace=.06)
    left = fig.add_subplot(gs[0, 0])
    count_ax = fig.add_subplot(gs[1, 0], sharex=left)
    right = fig.add_subplot(gs[:, 1])
    for q, j in enumerate(RL_INDICES):
        left.plot(np.arange(width), mean[q, :width], "o-", color=COLORS[j], ms=2.6, lw=1.15, label=LABELS[j])
        positive_counts = np.where(count[q, :width] > 0, count[q, :width], np.nan)
        count_ax.semilogy(np.arange(width), positive_counts, color=COLORS[j], lw=1.15)
        positive = np.flatnonzero(count[q, :width] > 0)
        if len(positive):
            last = positive[-1]
            count_ax.annotate(str(count[q, last]), (last, count[q, last]),
                              xytext=(4, 2), textcoords="offset points", color=COLORS[j], fontsize=7)
        right.semilogy(np.arange(width + 1), residual_mean[q, :width + 1], "o-", color=COLORS[j], ms=2.6, lw=1.15)
    left.set(ylabel="Mean relaxation weight", ylim=(.95, 3.05), yticks=[1, 1.5, 2, 2.5, 3],
             xlim=(-.3, width - .7), xticks=index_ticks(width - 1))
    left.set_title("(a) Mean selected weights", loc="left")
    left.tick_params(labelbottom=False)
    count_ax.set(ylabel=r"Solves ($n$)", xlabel=r"Cycle index $k$", ylim=(.8, 1600),
                 yticks=[1, 10, 100, 1000], yticklabels=["1", "10", "100", "1000"])
    count_ax.minorticks_off()
    right.axhline(tolerance, color="#777777", ls=":", lw=.9)
    finite_residuals = residual_mean[:, :width + 1][np.isfinite(residual_mean[:, :width + 1])]
    right.set(ylabel="Mean relative residual", xlabel="Completed cycles", xlim=(-.3, width + .3),
              xticks=index_ticks(width), ylim=(tolerance * .4, max(1.8, finite_residuals.max(initial=1.) * 1.2)),
              yticks=[1, 1e-2, 1e-4, 1e-6])
    right.set_title("(b) First-attempt mean residual", loc="left")
    for ax in (left, count_ax, right):
        ax.grid()
    fig.suptitle(f"{title} · seed {seed} · problems {start + 1}–{w.shape[1]}", y=.995, fontsize=9)
    handles, labels = left.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(.53, .94))
    fig.text(.53, .015, "First attempts only; failed attempts retained. Recovery is excluded from these profiles.",
             ha="center", fontsize=6.5, color="#555555")
    fig.subplots_adjust(left=.1, right=.99, top=.79, bottom=.19)
    return fig


def generate(run, *, output=None, illustrative_seed=None):
    root = Path(run).expanduser().resolve()
    output = Path(output or root / "paper_figures").expanduser().resolve()
    protocol, seeds, a, w, residuals, config, sources = read_run(root)
    native_totals = a[:, :, :, 2].sum(axis=2)
    reductions = 100 * (1 - native_totals / native_totals[:, :1])
    selected = seeds.index(str(illustrative_seed)) if illustrative_seed is not None else int(np.argmax(reductions[:, 3]))
    seed = seeds[selected]
    selection = ("Explicit --illustrative-seed override" if illustrative_seed is not None else
                 "Largest full-stream paired native-time reduction of LinUCB + LSTDQ among this run's seeds; same selection rule as the paper")
    activation = next(m["solve_activation_case"] for m in config["methods"] if m["id"] == METHODS[3])
    grid = config["problem"]["grid"]
    kind = config["problem"]["kind"]
    family = "Diffusion–advection" if "advection" in kind or "convection" in kind else "Diffusion"
    title = family + " " + (rf"${grid[0]}^3$" if len(set(grid)) == 1 else " × ".join(map(str, grid)))
    style()
    output.mkdir(parents=True, exist_ok=True)
    figures = [
        ("cumulative_times", plot_cumulative(a[selected], seed, title, activation)),
        ("runtime_breakdown", plot_breakdown(a, title, len(seeds))),
        ("learned_cycle_weights", plot_weights(w[selected], seed, title, activation)),
        ("cumulative_times_all_seeds", plot_all_seeds(a, seeds, title, activation)),
        ("joint_lstdq_heatmaps_all_seeds", plot_joint_atlas(w, seeds, title, activation)),
    ]
    if activation < a.shape[2]:
        figures.append(("mean_cycle_weights_seed" + seed, plot_profile(w[selected], residuals[selected], seed,
                         title, config["solve"]["tolerance"], activation)))
    for i, label in enumerate(seeds):
        figures.append(("cumulative_times_seed" + label, plot_cumulative(a[i], label, title, activation)))
        figures.append(("learned_cycle_weights_seed" + label, plot_weights(w[i], label, title, activation)))
    metadata = {"Title": f"Module 04 {family} {grid}: four branches",
                "Author": "RLTune-BoomerAMG", "Subject": "Figures from saved trajectories; no experiment rerun"}
    with PdfPages(output / "module04_paper_figures.pdf", metadata=metadata) as pdf:
        for name, fig in figures:
            fig.savefig(output / f"{name}.pdf", dpi=600, metadata=metadata)
            fig.savefig(output / f"{name}.png", dpi=260)
            fig.savefig(output / f"{name}.svg", dpi=600)
            pdf.savefig(fig, dpi=600, bbox_inches="tight", pad_inches=.04)
            plt.close(fig)
    np.savez_compressed(output / "plot_data.npz", seeds=np.array(seeds), methods=np.array(METHODS),
                        timing_fields=np.array(FIELDS), timings_sec=a,
                        rl_methods=np.array([METHODS[j] for j in RL_INDICES]),
                        first_primary_weights=w, first_primary_residuals=residuals,
                        illustrative_seed=np.array(seed))
    n = a.shape[2]
    activation_caption = ("Both LSTDQ branches are active from problem 1." if activation == 0 else
                          f"The dotted line marks joint LSTDQ activation at problem {activation + 1}; Default + LSTDQ is active from problem 1.")
    prefix_caption = ("A single measured setup prefix is included once in each setup continuation's budget; the learned setup state is copied at the activation boundary." if protocol.get("shared_prefix") else
                      "All four branches are measured independently from input one.")
    dispersion_caption = (f"mean ± sample SD across {len(seeds)} seed-level mean costs" if len(seeds) > 1 else
                          "the single seed-level mean cost; sample SD is unavailable")
    captions = (
        f"These figures use Module 04 {family} {grid}, {n} inputs per seed, seeds {', '.join(seeds)}, and all four branches. {prefix_caption}\n"
        f"Detailed illustration: seed {seed}. Selection: {selection}. This selected run is not a typical-run estimate.\n\n"
        f"cumulative_times: (a) Cumulative native time; (b) percentage reduction against the paired Default, computed from exact prefix totals without smoothing. All attempted inputs are retained, including final failures if any. Native time includes every setup/solve attempt and recovery, and excludes learner overhead. Legend values are full-stream seconds. {activation_caption}\n\n"
        f"runtime_breakdown: Mean time for the full stream" + (" and final 1000 problems" if n > 1000 else "") +
        f". Stacks show native setup, native solve, solve-learner overhead and setup-learner overhead. Labels are full measured cost as {dispersion_caption} (ms/problem); recovery is included once.\n\n"
        "learned_cycle_weights: Executed relaxation weights for the first primary attempt of each input. Columns are inputs; rows are zero-based action indices. The inactive joint prefix is reconstructed as weight one from recorded first-attempt cycle counts. White cells follow first-attempt termination. Learned reselection and fallback attempts are excluded from this display.\n\n"
        f"mean_cycle_weights_seed{seed}: Full {w.shape[-1]}-cycle cap over inputs {max(0, n - 1000, activation) + 1}–{n}, where both controllers are active. Mean weights are conditional on reaching a cycle; the lower strip shows that count on a logarithmic scale. Mean relative residual retains each terminated first primary attempt's last recorded residual, including failed attempts, so this mean need not reach tolerance. A mean is undefined wherever any residual is missing or nonfinite; no such attempt is silently excluded. Random exploration is included; learned reselection and fallback trajectories are excluded. Final recovered convergence is reported separately in the run summary.\n\n"
        "cumulative_times_all_seeds: Exact cumulative native time and paired percentage reduction for every seed, using the same conventions as cumulative_times.\n"
        "joint_lstdq_heatmaps_all_seeds: Recorded joint-controller first-attempt weights for every seed with identical axes; blanks follow first-attempt termination. Individual learned_cycle_weights_seed* figures also show Default + LSTDQ.\n"
    )
    (output / "captions.txt").write_text(captions)
    start = max(0, a.shape[2] - 1000, activation)
    count, mean, residual_mean = cycle_profiles(w[selected, :, start:], residuals[selected, :, start:]) if start < a.shape[2] else (None, None, None)
    detail = {"run": str(root), "seeds": seeds, "methods": list(METHODS),
              "illustrative_seed": seed, "selection_rule": selection,
              "full_stream_native_reduction_pct": dict(zip(seeds, reductions.tolist())),
              "timing_fields": list(FIELDS),
              "cycle_profiles": {METHODS[j]: {"cycle_counts": count[q].tolist(),
                    "mean_weights": [float(v) if np.isfinite(v) else None for v in mean[q]],
                    "mean_relative_residuals": [float(v) if np.isfinite(v) else None for v in residual_mean[q]],
                    "missing_residual_counts": np.count_nonzero(~np.isfinite(residuals[selected, q, start:]), axis=0).tolist()}
                    for q, j in enumerate(RL_INDICES)} if count is not None else {},
              "checks": {"passed": True, "paired_inputs": True, "native_accounting": True,
                         "learner_accounting": True, "matches_run_summary": True,
                         "action_grid_and_activation": True, "post_termination_blanks": True},
              "source_sha256": sources,
              "reproduce": f"python -m experiments.paper_final.plot_04_factorial {root}",
              "reference_style": "Manuscript prepare_current_results.py: runtime_breakdown, cumulative_times, learned_cycle_weights, mean_cycle_weights_seed2"}
    if Path(__file__).resolve() != (output / "plot_04_factorial.py").resolve():
        shutil.copyfile(__file__, output / "plot_04_factorial.py")
    detail["output_sha256"] = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    (output / "manifest.json").write_text(json.dumps(detail, indent=2, allow_nan=False) + "\n")
    info = {"output": str(output), "illustrative_seed": seed,
                      "selected_native_reductions_pct": dict(zip(LABELS, reductions[selected].tolist())),
                      "figures": [name for name, _ in figures], "checks_passed": True}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--illustrative-seed", help="Override the paper's maximum joint-reduction selection rule")
    args = parser.parse_args()
    print(json.dumps(generate(args.run, output=args.output, illustrative_seed=args.illustrative_seed), indent=2))


if __name__ == "__main__":
    main()
