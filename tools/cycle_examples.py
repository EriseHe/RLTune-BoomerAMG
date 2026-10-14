"""Illustrative executed relaxation schedules."""
from __future__ import annotations
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

import numpy as np

def draw(data):
    fig = plt.figure(figsize=(9.15, 3.0))
    # Two main blocks; the right block contains a tightly paired comparison.
    outer = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.4], wspace=.30)
    schedule = fig.add_subplot(outer[0, 0])
    residual_grid = outer[0, 1].subgridspec(1, 2, wspace=.06)
    convergence = fig.add_subplot(residual_grid[0, 0])
    default = fig.add_subplot(residual_grid[0, 1], sharey=convergence)
    axes = (schedule, convergence, default)
    for ex in data["examples"]:
        default.semilogy(np.arange(ex["default_cycles"]+1), ex["default_relative_residuals"],
            color="#111111", linestyle=ex["linestyle"], marker=ex["marker"],
            lw=.85, ms=2.5, markerfacecolor="white", markeredgewidth=.6, zorder=1)
    schedule.axhline(1.0, color="#111111", ls="--", lw=.9, zorder=0)
    for ex in reversed(data["examples"]):
        style = dict(color=ex["color"], linestyle=ex["linestyle"], marker=ex["marker"],
                     lw=1.05, ms=3.0, markerfacecolor="white", markeredgewidth=.7, zorder=3)
        schedule.plot(np.arange(ex["cycles"]), ex["weights"],
            label=f'{ex["label"]}: input {ex["problem_index"]} ({ex["cycles"]} cycles)', **style)
        convergence.semilogy(np.arange(ex["cycles"]+1), ex["relative_residuals"], **style)
        if ex["exploration_cycles"]:
            k = np.array(ex["exploration_cycles"])
            schedule.scatter(k, np.array(ex["weights"])[k], marker="x", s=30,
                             color="#202020", linewidths=1.0, zorder=6)
    end = max(max(ex["cycles"], ex["default_cycles"]) for ex in data["examples"])
    ticks = np.arange(0, end+1, 2)
    schedule.set(xlabel=r"Cycle index $k$", ylabel=r"Relaxation weight $w_k$",
                 xlim=(-.35, end-.3), ylim=(.88, 3.13), xticks=ticks, yticks=[1, 1.5, 2, 2.5, 3])
    for ax in (convergence, default):
        ax.set(xlabel="Completed cycles", xlim=(-.35, end+.5), ylim=(1e-7, 2.0),
               xticks=np.arange(0, end+1, 4), yticks=[1, 1e-2, 1e-4, 1e-6])
        ax.axhline(1e-6, color="#666666", ls=":", lw=.7, zorder=0)
    convergence.set_ylabel("Relative residual")
    default.tick_params(axis="y", labelleft=False)
    schedule.set_title("(a) Learned relaxation schedules", loc="left", pad=8)
    convergence.set_title("(b) LinUCB + LSTDQ", loc="left", pad=8)
    default.set_title(r"(c) Default ($w=1$)", loc="left", pad=8)
    handles, labels = schedule.get_legend_handles_labels()
    handles, labels = list(reversed(handles)), list(reversed(labels))
    handles.extend([Line2D([], [], color="#111111", ls="--", lw=.9),
                    Line2D([], [], color="#202020", marker="x", ms=5, ls="None", markeredgewidth=1.0)])
    labels.extend([r"Default ($w=1$)", "Exploration"])
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .915), ncol=5,
               frameon=False, handlelength=2.3, columnspacing=1.2, fontsize=7)
    fig.suptitle(r"Learned relaxation schedules (Diffusion $60^3$, seed 1, LinUCB + LSTDQ)",
                 fontsize=9.5, y=.995)
    fig.subplots_adjust(left=.065, right=.992, bottom=.175, top=.755)
    for ax in axes:
        ax.grid(False)
    return fig
