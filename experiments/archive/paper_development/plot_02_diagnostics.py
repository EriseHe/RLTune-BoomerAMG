"""Plot saved stage-02 diagnostics; no training or native solver execution."""
from __future__ import annotations

from experiments.paper_final.common.artifacts import ROOT as REPOSITORY_ROOT


import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from experiments.diagnostics.solve_control.plot_shared_action_rl_study import _read_json_lines, plot_action_trajectory_grid


ROOT = REPOSITORY_ROOT
FAMILIES = ("diffusion", "advection")
SETUPS = ("reference", "learned")
RL_METHODS = ("rl_lcb", "rl_mean")
RL_LABELS = {"rl_lcb": "RL-LCB", "rl_mean": "RL-mean"}
FAMILY_LABELS = {"diffusion": "Diffusion 60³", "advection": "Diffusion–advection 80³"}
SETUP_LABELS = {"reference": "Reference setup", "learned": "Diffusion-pilot setup"}


def generate_plots(result_dir: Path) -> dict:
    summary = json.loads((result_dir / "summary.json").read_text())
    protocol = json.loads((result_dir / "protocol.json").read_text())
    if not summary.get("complete"):
        raise ValueError("Stage 02 must be complete before plotting its held-out comparison")
    rows = _read_json_lines(result_dir / "trajectories.jsonl")
    cases = protocol["evaluation_cases"]
    cells = {(c["family"], c["setup"], c["policy"]): c
             for c in summary["cells"] if c["phase"] == "evaluation"}
    selected = {(s["family"], s["setup"], s["category"]): s["chosen"]
                for s in summary["selected_baselines"]}
    grouped = {key: sorted([r for r in rows if r["phase"] == "evaluation"
                            and (r["family"], r["setup"], r["policy"]) == key],
                           key=lambda r: r["case"])
               for key in cells}
    for key, records in grouped.items():
        if [r["case"] for r in records] != cases:
            raise ValueError(f"Incomplete or duplicate held-out cases: {key}")

    output = result_dir / "figures"
    output.mkdir(exist_ok=True)
    plot_data = []
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    figure.subplots_adjust(top=.81, bottom=.13, hspace=.45, wspace=.23)
    for fi, family in enumerate(FAMILIES):
        for si, setup in enumerate(SETUPS):
            axis = axes[fi, si]
            policies = [selected[(family, setup, category)]
                        for category in ("fixed", "schedule")] + list(RL_METHODS)
            components, labels, failures = [], [], []
            for policy in policies:
                cell = cells[(family, setup, policy)]
                outcomes = [r["outcome"] for r in grouped[(family, setup, policy)]]
                native = sum(o.get("primary_solve_runtime", o["solve_runtime"])
                             for o in outcomes)
                control = sum(o["infer_runtime"] - o.get("fallback_controller_runtime", 0)
                              for o in outcomes)
                recovery = cell["fallback_sec"]
                np.testing.assert_allclose(native + control + recovery,
                                           cell["continuation_sec"], rtol=1e-10, atol=1e-10)
                components.append((native, control, recovery))
                failures.append((cell["primary_failures"], cell["unrecovered_failures"]))
                plot_data.append(dict(cell, primary_native_sec=native,
                                      primary_controller_sec=control))
                if policy.startswith("fixed_"):
                    labels.append(f"Fixed\nw={float(policy.removeprefix('fixed_')):g}")
                elif policy.startswith("schedule_"):
                    weights = protocol["periodic_schedules"][policy.removeprefix("schedule_")]
                    labels.append("Schedule\n" + " → ".join(f"{w:g}" for w in weights))
                else:
                    labels.append(RL_LABELS[policy])
            totals = np.sum(components, axis=1)
            bottom = np.zeros(len(policies))
            for index, (label, color) in enumerate((
                ("Primary native solve", "#4e79a7"),
                ("Primary controller", "#59a14f"),
                ("Recovery: setup + solve + controller", "#f28e2b"),
            )):
                values = np.asarray(components)[:, index]
                axis.bar(range(len(policies)), values, bottom=bottom,
                         label=label, color=color, width=.65)
                bottom += values
            for index, (total, (primary, unrecovered)) in enumerate(zip(totals, failures)):
                annotation = f"{total:.3f}"
                if primary:
                    annotation += f"\n{primary - unrecovered} recovered"
                if unrecovered:
                    annotation += f"\n{unrecovered} unrecovered"
                axis.text(index, total + .025 * max(totals), annotation,
                          ha="center", va="bottom", fontsize=9)
            axis.set(xticks=range(len(policies)), xticklabels=labels,
                     ylabel="Total recorded comparison time (s)",
                     title=f"{FAMILY_LABELS[family]} · {SETUP_LABELS[setup]}",
                     ylim=(0, max(totals) * 1.30))
            axis.grid(axis="y", alpha=.2)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .90),
                  ncol=3, frameon=False)
    figure.suptitle("02 · Frozen-policy diagnostic: held-out costs\n"
                    "6 matched inputs per panel (cases 7–12); baselines selected on cases 1–6",
                    fontsize=14, y=.985)
    figure.text(.5, .025, "Primary setup excluded; failed attempts and all recovery costs included.\n"
                "The diffusion-pilot setup is transferred to advection. Development screen; one trained policy per family.",
                ha="center", fontsize=9)
    figure.savefig(output / "heldout_costs.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    figure.subplots_adjust(top=.76, bottom=.20, wspace=.15)
    saturation = []
    for axis, family in zip(axes, FAMILIES):
        for mi, method in enumerate(RL_METHODS):
            values = []
            for setup in SETUPS:
                records = grouped[(family, setup, method)]
                times = [t for r in records if r["outcome"].get("cycle_times")
                         for t in [0., *r["outcome"]["cycle_times"][:-1]]]
                fraction = float(np.mean(np.asarray(times) >= .01))
                np.testing.assert_allclose(fraction, cells[(family, setup, method)]
                                           ["last_cycle_time_inputs_above_10ms_fraction"])
                values.append(100 * fraction)
                saturation.append(dict(family=family, setup=setup, policy=method,
                                       saturated_fraction=fraction, action_count=len(times)))
            bars = axis.bar(np.arange(2) + (mi - .5) * .34, values, width=.34,
                            label=RL_LABELS[method], color=("#4e79a7", "#e15759")[mi])
            axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        axis.set(title=FAMILY_LABELS[family], xticks=range(2),
                 xticklabels=[SETUP_LABELS[s] for s in SETUPS], ylim=(0, 110))
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Action inputs with time feature at cap (%)")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
                  bbox_to_anchor=(.5, .87), ncol=2, frameon=False)
    figure.suptitle("02 · Previous-cycle time feature saturation\n"
                    "Held-out cases 7–12; cap reached at 10 ms (0.002 s scale × 5)", fontsize=14)
    figure.text(.5, .035, "Includes each solve's initial zero-time input; excludes terminal-cycle time, which has no next action.\n"
                "Primary attempts only; policies frozen, exploration disabled.", ha="center", fontsize=9)
    figure.savefig(output / "time_feature_saturation.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    columns = [(family, setup) for family in FAMILIES for setup in SETUPS]
    with plt.rc_context({"savefig.dpi": 180}):
        plot_action_trajectory_grid(
            {(i, method): grouped[(family, setup, method)]
             for i, (family, setup) in enumerate(columns) for method in RL_METHODS},
            range(len(columns)), protocol["cycle_cap"], output / "cycle_actions.png",
            action_min=1., action_max=3., methods=RL_METHODS, method_labels=RL_LABELS,
            column_labels={i: f"{FAMILY_LABELS[family]}\n{SETUP_LABELS[setup]}"
                           for i, (family, setup) in enumerate(columns)},
            x_label="Held-out input (1–6 = cases 7–12)", panel_width=3.8, panel_height=3.0,
            title="02 · Frozen RL actions on held-out inputs (primary attempts)\n"
                  "White: no recorded cycle / terminated; exploration disabled; cycle index starts at 0",
        )
    manifest = {"development_only": True, "evaluation_cases": cases,
                "figures": ["heldout_costs.png", "time_feature_saturation.png", "cycle_actions.png"],
                "heldout_cells": plot_data, "time_feature_saturation": saturation}
    (result_dir / "plot_summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path,
                        default=ROOT / "results/paper_final/02_diagnostics")
    args = parser.parse_args()
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        manifest = generate_plots(args.result_dir)
    print(json.dumps({"result_dir": str(args.result_dir), "figures": manifest["figures"]}, indent=2))


if __name__ == "__main__":
    main()
