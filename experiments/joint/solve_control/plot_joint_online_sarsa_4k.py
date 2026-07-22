from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from plot_shared_action_rl_study import plot_action_trajectory_grid
from online_td_experiment_common import _write_json
from run_online_methods_2k import _method_comparison
from run_joint_online_sarsa_4k import _window_result, _write_summary_csv


FAMILY_ORDER = (
    "default_setup",
    "default",
    "fixed_w1.6",
    "ppo",
    "sarsa_uniform",
    "sarsa_uncertainty_lcb",
    "stagewise_lsvi_lcb",
    "recursive_mc_lcb",
    "recursive_lstdq_lcb",
    "recursive_lstdq_v2_lcb",
    "structured_model_based",
    "recalibrated_lsvi_lcb",
    "batched_lsvi_lcb",
)
FAMILY_LABELS = {
    "default_setup": "Default setup + default solve",
    "default": "Online LinUCB + default solve",
    "fixed_w1.6": "Online LinUCB + fixed w=1.6",
    "ppo": "Online LinUCB + 2000-instance absolute PPO",
    "sarsa_uniform": "Online LinUCB + SARSA, uniform epsilon",
    "sarsa_uncertainty_lcb": "Online LinUCB + SARSA, uncertainty LCB",
    "stagewise_lsvi_lcb": "Online LinUCB + Stagewise LSVI-LCB",
    "recursive_mc_lcb": "Online LinUCB + Recursive MC-LCB",
    "recursive_lstdq_lcb": "Online LinUCB + Recursive LSTDQ-LCB",
    "recursive_lstdq_v2_lcb": "Online LinUCB + Recursive LSTDQ v2-LCB",
    "structured_model_based": "Online LinUCB + Structured model-based",
    "recalibrated_lsvi_lcb": "Online LinUCB + Recalibrated LSVI-LCB",
    "batched_lsvi_lcb": "Online LinUCB + Batched LSVI-LCB",
}


def _active_families(family_by_method: Dict[str, str]) -> tuple[str, ...]:
    present = set(family_by_method.values())
    return tuple(family for family in FAMILY_ORDER if family in present)


def _read_json_lines(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rolling(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if window <= 0 or window > array.size:
        raise ValueError("rolling window must fit within the trajectory")
    return np.convolve(array, np.ones(window) / float(window), mode="valid")


def _trailing_mean(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("rolling window must be positive")
    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    indices = np.arange(array.size)
    starts = np.maximum(indices + 1 - window, 0)
    totals = cumulative[indices + 1] - cumulative[starts]
    return totals / (indices + 1 - starts)


def _candidate_label(candidate: Dict[str, Any]) -> str:
    return (
        rf"$\alpha={float(candidate['alpha']):g}$, "
        rf"$\lambda={float(candidate['trace_lambda']):g}$"
    )


def _load_records(
    result_dir: Path,
    methods: Sequence[str],
) -> Dict[str, list[Dict[str, Any]]]:
    return {
        method: _read_json_lines(result_dir / "trajectories" / f"{method}.jsonl")
        for method in methods
    }


def _metric(rows: Sequence[Dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray(
        [float(row["outcome"].get(field, 0.0)) for row in rows],
        dtype=float,
    )


def _selected_weights(
    rows: Sequence[Dict[str, Any]],
    *,
    family: str,
    statistic: str,
) -> np.ndarray:
    if family in {"default_setup", "default"}:
        return np.ones(len(rows), dtype=float)
    if family == "fixed_w1.6":
        return np.full(len(rows), 1.6, dtype=float)
    if statistic not in {"first_cycle", "solve_mean"}:
        raise ValueError(f"Unsupported weight statistic: {statistic}")

    selected = []
    for row_index, row in enumerate(rows):
        actions = row["outcome"].get("cycle_actions")
        if not actions:
            raise ValueError(
                f"Missing cycle_actions for {family} record {row_index}"
            )
        action_values = np.asarray(actions, dtype=float)
        selected.append(
            float(action_values[0])
            if statistic == "first_cycle"
            else float(np.mean(action_values))
        )
    return np.asarray(selected, dtype=float)


def _plot_five_family_weight_trajectory(
    *,
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
    candidate_by_method: Dict[str, Dict[str, Any]],
    statistic: str,
    window: int,
) -> None:
    statistic_label = {
        "first_cycle": "first-cycle selected w",
        "solve_mean": "mean selected w across solve cycles",
    }[statistic]
    families = _active_families(family_by_method)
    plotted_values = [
        _selected_weights(
            records[method],
            family=family_by_method[method],
            statistic=statistic,
        )
        for method in family_by_method
    ]
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in plotted_values if values.size]
    )
    value_min = min(1.0, float(np.min(finite_values)))
    value_max = max(2.0, float(np.max(finite_values)))
    value_margin = max(0.05, 0.03 * (value_max - value_min))
    y_lower = value_min - value_margin
    y_upper = value_max + value_margin
    tick_step = 0.25 if value_max > 2.2 else 0.2
    y_ticks = np.arange(
        np.ceil(value_min / tick_step) * tick_step,
        value_max + 0.5 * tick_step,
        tick_step,
    )
    figure, axes_grid = plt.subplots(
        len(families),
        1,
        figsize=(14, 3.2 * len(families)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_grid[:, 0]
    midpoint = len(next(iter(records.values()))) // 2
    palette = plt.get_cmap("tab10")
    for axis, family in zip(axes, families):
        family_methods = [
            method for method, method_family in family_by_method.items()
            if method_family == family
        ]
        for method_index, method in enumerate(family_methods):
            values = _selected_weights(
                records[method],
                family=family,
                statistic=statistic,
            )
            trend = _trailing_mean(values, window)
            if method in candidate_by_method:
                label = _candidate_label(candidate_by_method[method])
            else:
                label = FAMILY_LABELS[family]
            color = palette(method_index % 10)
            axis.plot(
                np.arange(1, values.size + 1),
                trend,
                linewidth=1.55,
                color=color,
                label=label,
            )
            axis.scatter(
                [1],
                [values[0]],
                s=24,
                color=color,
                zorder=3,
            )
        axis.axhline(1.0, color="0.55", linewidth=0.75, linestyle=":")
        axis.axhline(1.6, color="0.45", linewidth=0.75, linestyle="--")
        axis.axvline(midpoint, color="0.35", linewidth=0.8, linestyle="--")
        axis.set_title(FAMILY_LABELS[family], loc="left", fontweight="bold")
        axis.set_ylabel("Selected w")
        axis.set_ylim(y_lower, y_upper)
        axis.set_yticks(y_ticks)
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.legend(
            loc="upper right",
            ncol=3 if len(family_methods) > 3 else 1,
            frameon=False,
        )
    figure.suptitle(
        f"Online weight selection: {statistic_label}\n"
        f"Trailing mean up to {window} instances; dots show the exact first instance",
        fontweight="bold",
    )
    axes[-1].set_xlabel(
        "Persistent online comparison instance "
        "(vertical line: audit checkpoint only, no reset)"
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_ppo_weight_detail(
    *,
    path: Path,
    rows: Sequence[Dict[str, Any]],
    window: int,
) -> None:
    trajectories = [
        np.asarray(row["outcome"]["cycle_actions"], dtype=float)
        for row in rows
    ]
    max_cycles = max(len(actions) for actions in trajectories)
    by_cycle = [
        np.asarray(
            [actions[cycle] for actions in trajectories if len(actions) > cycle],
            dtype=float,
        )
        for cycle in range(max_cycles)
    ]
    retained_cycles = sum(values.size >= 50 for values in by_cycle)
    cycle_axis = np.arange(retained_cycles)
    cycle_mean = np.asarray(
        [np.mean(values) for values in by_cycle[:retained_cycles]], dtype=float
    )
    cycle_p05 = np.asarray(
        [np.percentile(values, 5) for values in by_cycle[:retained_cycles]],
        dtype=float,
    )
    cycle_p95 = np.asarray(
        [np.percentile(values, 95) for values in by_cycle[:retained_cycles]],
        dtype=float,
    )
    solve_means = np.asarray(
        [np.mean(actions) for actions in trajectories], dtype=float
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9),
        constrained_layout=True,
    )
    axes[0].fill_between(
        cycle_axis,
        cycle_p05,
        cycle_p95,
        alpha=0.18,
        label="5th-95th percentile across instances",
    )
    axes[0].plot(
        cycle_axis,
        cycle_mean,
        linewidth=2.0,
        label="Mean physical w",
    )
    axes[0].axhline(1.5, color="0.45", linewidth=0.8, linestyle="--")
    axes[0].set_title(
        "Frozen Exp44 PPO moves within each solve",
        loc="left",
        fontweight="bold",
    )
    axes[0].set_xlabel("AMG cycle")
    axes[0].set_ylabel("Selected w")
    axes[0].grid(alpha=0.22, linewidth=0.7)
    axes[0].legend(frameon=False)

    instance_axis = np.arange(1, solve_means.size + 1)
    axes[1].plot(
        instance_axis,
        solve_means,
        linewidth=0.6,
        alpha=0.22,
        label="Per-instance mean w",
    )
    axes[1].plot(
        instance_axis,
        _trailing_mean(solve_means, window),
        linewidth=1.8,
        label=f"Trailing mean up to {window} instances",
    )
    axes[1].axvline(
        solve_means.size // 2,
        color="0.35",
        linewidth=0.8,
        linestyle="--",
    )
    axes[1].set_title(
        "Across instances the policy is frozen, so there is no learning trend",
        loc="left",
        fontweight="bold",
    )
    axes[1].set_xlabel("Persistent online comparison instance")
    axes[1].set_ylabel("Mean selected w across solve")
    axes[1].grid(alpha=0.22, linewidth=0.7)
    axes[1].legend(frameon=False)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_cycle_action_trajectory_lines(
    *,
    path: Path,
    rows: Sequence[Dict[str, Any]],
    method_label: str,
    start_instance: int,
    stop_instance: int,
    sampled_trajectories: int = 40,
    minimum_cycle_support: int = 50,
) -> None:
    start_index = max(0, int(start_instance) - 1)
    stop_index = min(len(rows), int(stop_instance))
    selected_rows = list(rows[start_index:stop_index])
    if not selected_rows:
        raise ValueError("Cycle-action trajectory window is empty")
    trajectories = [
        np.asarray(row["outcome"]["cycle_actions"], dtype=float)
        for row in selected_rows
    ]
    max_cycles = max(len(actions) for actions in trajectories)
    by_cycle = [
        np.asarray(
            [actions[cycle] for actions in trajectories if len(actions) > cycle],
            dtype=float,
        )
        for cycle in range(max_cycles)
    ]
    retained_cycles = sum(
        values.size >= int(minimum_cycle_support) for values in by_cycle
    )
    cycle_axis = np.arange(retained_cycles)
    cycle_mean = np.asarray(
        [np.mean(values) for values in by_cycle[:retained_cycles]], dtype=float
    )
    cycle_median = np.asarray(
        [np.median(values) for values in by_cycle[:retained_cycles]], dtype=float
    )
    finite_actions = np.concatenate(
        [actions[np.isfinite(actions)] for actions in trajectories if actions.size]
    )
    action_min = min(1.0, float(np.min(finite_actions)))
    action_max = max(2.0, float(np.max(finite_actions)))
    action_margin = max(0.05, 0.03 * (action_max - action_min))
    y_lower = action_min - action_margin
    y_upper = action_max + action_margin
    tick_step = 0.25 if action_max > 2.2 else 0.2
    y_ticks = np.arange(
        np.ceil(action_min / tick_step) * tick_step,
        action_max + 0.5 * tick_step,
        tick_step,
    )

    sample_count = min(int(sampled_trajectories), len(trajectories))
    sample_indices = np.linspace(
        0,
        len(trajectories) - 1,
        num=sample_count,
        dtype=int,
    )

    figure, axis = plt.subplots(figsize=(13.2, 6.8), constrained_layout=True)
    for sample_number, sample_index in enumerate(sample_indices):
        actions = trajectories[int(sample_index)]
        axis.plot(
            np.arange(min(actions.size, retained_cycles)),
            actions[:retained_cycles],
            color="#4C78A8",
            linewidth=0.75,
            alpha=0.13,
            label="Sampled instance trajectories" if sample_number == 0 else None,
        )
    axis.plot(
        cycle_axis,
        cycle_mean,
        color="#D18F00",
        linewidth=2.8,
        marker="o",
        markersize=5.0,
        label="Mean across all available instances",
        zorder=4,
    )
    axis.plot(
        cycle_axis,
        cycle_median,
        color="#222222",
        linewidth=1.35,
        linestyle="--",
        marker="s",
        markersize=3.3,
        label="Median",
        zorder=3,
    )
    axis.axhline(
        1.6,
        color="#6F6F6F",
        linewidth=1.1,
        linestyle=":",
        label="Fixed w=1.6 reference",
    )
    axis.annotate(
        f"Cycle 0 mean: {cycle_mean[0]:.2f}",
        xy=(0, cycle_mean[0]),
        xytext=(1.1, y_upper - 0.04 * (y_upper - y_lower)),
        arrowprops={"arrowstyle": "->", "color": "#7A5600", "linewidth": 1.0},
        color="#7A5600",
        fontsize=10,
    )
    axis.set_title(
        f"{method_label}: per-cycle action trajectories\n"
        f"Online instances {start_index + 1:,}-{stop_index:,}; "
        f"{sample_count} sampled solves shown as thin lines",
        loc="left",
        fontweight="bold",
    )
    axis.set_xlabel("AMG cycle")
    axis.set_ylabel("Selected relaxation weight w")
    axis.set_xlim(-0.35, retained_cycles - 0.65)
    axis.set_ylim(y_lower, y_upper)
    axis.set_xticks(np.arange(0, retained_cycles, 2))
    axis.set_yticks(y_ticks)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="lower right", frameon=False, ncol=2)
    figure.savefig(path, dpi=210, bbox_inches="tight")
    plt.close(figure)


def _best_complete_native_solve_window(
    rows: Sequence[Dict[str, Any]],
    *,
    window: int = 500,
) -> tuple[int, int]:
    if not rows:
        raise ValueError("Cannot select a trajectory window from no records")
    block_size = min(int(window), len(rows))
    candidates = []
    for start in range(0, len(rows) - block_size + 1, block_size):
        stop = start + block_size
        mean_solve = float(
            np.mean(_metric(rows[start:stop], "native_solve_runtime"))
        )
        candidates.append((mean_solve, start + 1, stop))
    _mean_solve, start_instance, stop_instance = min(candidates)
    return int(start_instance), int(stop_instance)


def _plot_cumulative_components(
    *,
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
) -> None:
    fields = (
        ("setup_runtime", "Cumulative setup (s)"),
        ("solve_runtime", "Cumulative native solve (s)"),
        ("end_to_end_runtime", "Cumulative end-to-end (s)"),
    )
    figure, axes = plt.subplots(
        len(fields),
        1,
        figsize=(14, 11),
        sharex=True,
        constrained_layout=True,
    )
    palette = plt.get_cmap("tab10")
    midpoint = len(next(iter(records.values()))) // 2
    for method_index, (method, rows) in enumerate(records.items()):
        family = family_by_method[method]
        x_axis = np.arange(1, len(rows) + 1)
        for axis, (field, ylabel) in zip(axes, fields):
            axis.plot(
                x_axis,
                np.cumsum(_metric(rows, field)),
                color=palette(method_index % 10),
                linewidth=1.45,
                label=FAMILY_LABELS[family],
            )
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22, linewidth=0.7)
            axis.axvline(midpoint, color="0.35", linewidth=0.8, linestyle="--")
    axes[0].legend(ncol=2, frameon=False)
    axes[-1].set_xlabel(
        "Persistent online comparison instance "
        "(vertical line: audit checkpoint only, no reset)"
    )
    figure.suptitle("Cumulative runtime by measured component", fontweight="bold")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_first_last_components(
    *,
    path: Path,
    result: Dict[str, Any],
) -> None:
    components = (
        ("setup_runtime", "Setup"),
        ("native_solve_runtime", "Native solve"),
        ("controller_runtime", "Controller"),
        ("setup_bandit_overhead", "Bandit overhead"),
    )
    windows = (("first_1000", "First 1000"), ("last_1000", "Last 1000"))
    methods = tuple(result["protocol"]["methods"])
    labels = [
        FAMILY_LABELS[result["protocol"]["families"][method]]
        for method in methods
    ]
    colors = plt.get_cmap("Set2")(np.linspace(0.05, 0.95, len(components)))
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        sharey=True,
        constrained_layout=True,
    )
    for axis, (window_key, title) in zip(axes, windows):
        bottoms = np.zeros(len(methods), dtype=float)
        for (component, component_label), color in zip(components, colors):
            values = np.asarray(
                [
                    1_000.0
                    * float(
                        result["windows"][window_key]["methods"][method]
                        ["means_sec"][component]
                    )
                    for method in methods
                ],
                dtype=float,
            )
            axis.bar(
                np.arange(len(methods)),
                values,
                bottom=bottoms,
                color=color,
                label=component_label,
            )
            bottoms += values
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(np.arange(len(methods)), labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0].set_ylabel("Mean end-to-end runtime (ms/case)")
    axes[1].legend(frameon=False, loc="upper right")
    figure.suptitle(
        "First vs last 1000: measured runtime components",
        fontweight="bold",
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_all_runtime_breakdown(
    *,
    path: Path,
    result: Dict[str, Any],
    window_key: str | None = None,
    figure_title: str | None = None,
) -> None:
    methods = tuple(result["protocol"]["methods"])
    labels = [
        FAMILY_LABELS[result["protocol"]["families"][method]]
        for method in methods
    ]
    if window_key is None:
        window_key = next(
            key for key in result["windows"] if str(key).startswith("all_")
        )
    window = result["windows"][window_key]["methods"]
    case_count = int(window[next(iter(methods))]["cases"])
    components = (
        ("setup_runtime", "Setup", "#66c2a5"),
        ("native_solve_runtime", "Native solve", "#8da0cb"),
        ("controller_runtime", "Controller", "#ffd92f"),
        ("setup_bandit_overhead", "Bandit overhead", "#b3b3b3"),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        constrained_layout=True,
    )
    for axis, included, title in (
        (axes[0], components[:2], "Native runtime"),
        (axes[1], components, "End-to-end runtime"),
    ):
        bottoms = np.zeros(len(methods), dtype=float)
        for component, component_label, color in included:
            values = np.asarray(
                [
                    1_000.0 * float(window[method]["means_sec"][component])
                    for method in methods
                ],
                dtype=float,
            )
            axis.bar(
                np.arange(len(methods)),
                values,
                bottom=bottoms,
                color=color,
                label=component_label,
            )
            bottoms += values
        for index, total in enumerate(bottoms):
            axis.text(
                index,
                total,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(np.arange(len(methods)), labels, rotation=24, ha="right")
        axis.set_ylabel("Mean runtime (ms/case)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        if axis is axes[0]:
            axis.legend(frameon=False, loc="lower left")
        else:
            axis.legend(
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
            )
    figure.suptitle(
        figure_title or f"All {case_count:,} online comparison instances",
        fontweight="bold",
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_recovery_outcomes(
    *,
    path: Path,
    result: Dict[str, Any],
) -> None:
    methods = tuple(result["protocol"]["methods"])
    labels = [
        FAMILY_LABELS[result["protocol"]["families"][method]]
        for method in methods
    ]
    all_window_key = next(
        key for key in result["windows"] if str(key).startswith("all_")
    )
    summaries = result["windows"][all_window_key]["methods"]
    processed = np.asarray(
        [float(summaries[method]["cases"]) for method in methods],
        dtype=float,
    )
    recovered = np.asarray(
        [float(summaries[method]["recovered_failure_count"]) for method in methods],
        dtype=float,
    )
    unrecovered = np.asarray(
        [
            float(summaries[method]["unrecovered_failure_count"])
            for method in methods
        ],
        dtype=float,
    )
    primary_success = processed - recovered - unrecovered
    scale = np.divide(
        100.0,
        processed,
        out=np.zeros_like(processed),
        where=processed > 0.0,
    )
    components = (
        (primary_success * scale, "Primary success", "#66c2a5"),
        (recovered * scale, "Recovered by default fallback", "#ffd92f"),
        (unrecovered * scale, "Unrecovered", "#d95f02"),
    )
    figure, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    bottoms = np.zeros(len(methods), dtype=float)
    for values, label, color in components:
        axis.bar(
            np.arange(len(methods)),
            values,
            bottom=bottoms,
            color=color,
            label=label,
        )
        bottoms += values
    for index, (fallback_count, failure_count) in enumerate(
        zip(recovered, unrecovered)
    ):
        axis.text(
            index,
            101.0,
            f"fallback {int(fallback_count):,}\nunrecovered {int(failure_count):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axis.set_ylim(0.0, 113.0)
    axis.set_ylabel("Share of external instances (%)")
    axis.set_xticks(np.arange(len(methods)), labels, rotation=24, ha="right")
    axis.set_title(
        "Primary attempt and default-fallback outcomes",
        fontweight="bold",
    )
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_first_last_improvement_ci(
    *,
    path: Path,
    result: Dict[str, Any],
) -> None:
    family_by_method = result["protocol"]["families"]
    methods = tuple(
        method
        for method in result["protocol"]["methods"]
        if family_by_method[method] not in {"default", "fixed_w1.6"}
    )
    windows = (("first_1000", "First 1000"), ("last_1000", "Last 1000"))
    metrics = (
        ("native_solve_runtime", "Native solve"),
        ("end_to_end_runtime", "End-to-end"),
    )
    labels = [FAMILY_LABELS[family_by_method[method]] for method in methods]
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharey=False,
        constrained_layout=True,
    )
    offsets = np.linspace(-0.16, 0.16, len(windows))
    for axis, (metric, title) in zip(axes, metrics):
        for offset, (window_key, window_label) in zip(offsets, windows):
            comparisons = result["windows"][window_key]["comparisons"][
                "vs_fixed_w1.6"
            ]
            means = []
            lower_errors = []
            upper_errors = []
            for method in methods:
                entry = comparisons[method][metric]
                mean = float(entry["candidate_improvement_pct"])
                lower, upper = map(float, entry["candidate_improvement_95pct"])
                means.append(mean)
                lower_errors.append(mean - lower)
                upper_errors.append(upper - mean)
            axis.errorbar(
                np.arange(len(methods)) + offset,
                means,
                yerr=np.asarray([lower_errors, upper_errors]),
                fmt="o",
                capsize=4,
                linewidth=1.4,
                label=window_label,
            )
        axis.axhline(0.0, color="0.35", linewidth=0.9, linestyle="--")
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(np.arange(len(methods)), labels, rotation=24, ha="right")
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    for axis in axes:
        axis.set_ylabel("Improvement vs online LinUCB + fixed w=1.6 (%)")
    axes[1].legend(frameon=False)
    figure.suptitle("Paired bootstrap 95% confidence intervals", fontweight="bold")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_five_family_trajectory(
    *,
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
    candidate_by_method: Dict[str, Dict[str, Any]],
    field: str,
    ylabel: str,
    window: int,
) -> None:
    families = _active_families(family_by_method)
    figure, axes_grid = plt.subplots(
        len(families),
        1,
        figsize=(14, 3.2 * len(families)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes_grid[:, 0]
    midpoint = len(next(iter(records.values()))) // 2
    palette = plt.get_cmap("tab10")
    all_rolling: list[np.ndarray] = []
    for family_index, (axis, family) in enumerate(zip(axes, families)):
        family_methods = [
            method for method, method_family in family_by_method.items()
            if method_family == family
        ]
        for method_index, method in enumerate(family_methods):
            values = 1_000.0 * _metric(records[method], field)
            rolling = _rolling(values, window)
            all_rolling.append(rolling)
            if method in candidate_by_method:
                label = _candidate_label(candidate_by_method[method])
            else:
                label = FAMILY_LABELS[family]
            axis.plot(
                np.arange(window, values.size + 1),
                rolling,
                linewidth=1.55,
                color=palette(method_index % 10),
                label=label,
            )
        axis.axvline(midpoint, color="0.35", linewidth=0.8, linestyle="--")
        axis.set_title(FAMILY_LABELS[family], loc="left", fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.legend(loc="upper right", ncol=3 if len(family_methods) > 3 else 1, frameon=False)
    finite = np.concatenate([values[np.isfinite(values)] for values in all_rolling])
    lower, upper = np.percentile(finite, [1.0, 99.0])
    margin = max(0.05 * (upper - lower), 0.1)
    for axis in axes:
        axis.set_ylim(lower - margin, upper + margin)
    axes[-1].set_xlabel("Persistent online comparison instance")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _write_markdown_table(
    path: Path,
    rows: Sequence[Dict[str, str]],
) -> None:
    headers = (
        "Method",
        "Setup ms",
        "Native solve ms",
        "Native total ms",
        "Solve improvement vs reference",
        "Controller ms",
        "Bandit overhead ms",
        "End-to-end ms",
        "Primary failures",
        "Recovered",
        "Unrecovered",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["method"],
                    row["setup_ms"],
                    row["solve_ms"],
                    row["native_total_ms"],
                    row["solve_vs_default"],
                    row["controller_ms"],
                    row["bandit_overhead_ms"],
                    row["end_to_end_ms"],
                    row["primary_failures"],
                    row["recovered_failures"],
                    row["unrecovered_failures"],
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_plots(
    *,
    result_dir: Path,
    rolling_window: int = 100,
    analysis_stop: int | None = None,
) -> Dict[str, Any]:
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    methods = tuple(protocol["methods"])
    family_by_method = dict(protocol["families"])
    candidate_by_method = {
        candidate["name"]: candidate for candidate in protocol["candidates"]
    }
    records = _load_records(result_dir, methods)
    lengths = {method: len(rows) for method, rows in records.items()}
    online_window = protocol["stream_partition"]["online"]
    source_cases = int(online_window[1]) - int(online_window[0])
    if set(lengths.values()) != {source_cases}:
        raise ValueError(
            f"Every method must contain {source_cases} records, got {lengths}"
        )
    if analysis_stop is None:
        expected_cases = source_cases
        artifact_dir = result_dir
    else:
        expected_cases = int(analysis_stop)
        if not 1 <= expected_cases <= source_cases:
            raise ValueError(
                f"analysis_stop must be in [1, {source_cases}], got "
                f"{expected_cases}"
            )
        if expected_cases < int(rolling_window):
            raise ValueError("analysis_stop must be at least rolling_window")
        records = {
            method: rows[:expected_cases] for method, rows in records.items()
        }
        windows = {f"all_{expected_cases}": (0, expected_cases)}
        if expected_cases >= 1_000:
            windows.update(
                {
                    "first_1000": (0, 1_000),
                    "last_1000": (expected_cases - 1_000, expected_cases),
                    "last_500": (expected_cases - 500, expected_cases),
                }
            )
        result = dict(result)
        result["windows"] = {
            name: _window_result(
                {
                    method: rows[start:stop]
                    for method, rows in records.items()
                },
                seed=int(9_331_001 + window_index * 100_003),
            )
            for window_index, (name, (start, stop)) in enumerate(windows.items())
        }
        artifact_dir = result_dir / f"analysis_first_{expected_cases}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        analysis_scope = {
            "source_result": str(result_dir / "result.json"),
            "source_stream_sha256": protocol["stream"]["sha256"],
            "included_online_indices": [0, expected_cases],
            "excluded_online_indices": [expected_cases, source_cases],
        }
        _write_json(
            artifact_dir / "analysis_scope.json",
            analysis_scope,
        )
        _write_json(
            artifact_dir / "analysis_result.json",
            {
                "analysis_scope": analysis_scope,
                "protocol": protocol,
                "windows": result["windows"],
            },
        )

    fixed_rows = records["bandit_fixed_w1.6"]
    audit_windows = {
        "all": (0, expected_cases),
        "first_half": (0, expected_cases // 2),
        "last_half": (expected_cases // 2, expected_cases),
        "last_500": (max(0, expected_cases - 500), expected_cases),
    }
    same_setup_audit: Dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        if method in {"bandit_default", "bandit_fixed_w1.6"}:
            continue
        same_setup_audit[method] = {}
        for window_index, (window_name, (start, stop)) in enumerate(
            audit_windows.items()
        ):
            indices = [
                index
                for index in range(start, stop)
                if records[method][index]["params"] == fixed_rows[index]["params"]
            ]
            candidate = [records[method][index] for index in indices]
            baseline = [fixed_rows[index] for index in indices]
            same_setup_audit[method][window_name] = {
                "matching_cases": int(len(indices)),
                "comparison": (
                    _method_comparison(
                        candidate,
                        baseline,
                        seed=int(8_217_001 + method_index * 1009 + window_index),
                    )
                    if indices
                    else None
                ),
            }
    same_setup_path = artifact_dir / "same_setup_audit.json"
    _write_json(same_setup_path, same_setup_audit)

    figures_dir = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    solve_path = figures_dir / "five_family_native_solve_trajectory.png"
    setup_path = figures_dir / "five_family_setup_trajectory.png"
    first_weight_path = figures_dir / "five_family_first_cycle_weight_trajectory.png"
    mean_weight_path = figures_dir / "five_family_mean_weight_trajectory.png"
    ppo_weight_detail_path = figures_dir / "ppo_weight_selection_detail.png"
    action_trajectory_path = figures_dir / "learned_per_cycle_action_trajectories.png"
    lstdq_trajectory_lines_path = (
        figures_dir / "recursive_lstdq_lcb_best_window_cycle_trajectory_lines.png"
    )
    cumulative_path = figures_dir / "five_family_cumulative_components.png"
    all_runtime_path = figures_dir / f"all_{expected_cases}_runtime_breakdown.png"
    last_1000_runtime_path = figures_dir / "last_1000_runtime_breakdown.png"
    recovery_path = figures_dir / "primary_and_fallback_outcomes.png"
    first_last_components_path = figures_dir / "first_vs_last_1000_components.png"
    first_last_ci_path = figures_dir / "first_vs_last_1000_improvement_ci.png"
    _plot_five_family_trajectory(
        path=solve_path,
        records=records,
        family_by_method=family_by_method,
        candidate_by_method=candidate_by_method,
        field="solve_runtime",
        ylabel=f"Rolling {rolling_window} native solve (ms)",
        window=int(rolling_window),
    )
    _plot_five_family_trajectory(
        path=setup_path,
        records=records,
        family_by_method=family_by_method,
        candidate_by_method=candidate_by_method,
        field="setup_runtime",
        ylabel=f"Rolling {rolling_window} setup (ms)",
        window=int(rolling_window),
    )
    _plot_five_family_weight_trajectory(
        path=first_weight_path,
        records=records,
        family_by_method=family_by_method,
        candidate_by_method=candidate_by_method,
        statistic="first_cycle",
        window=int(rolling_window),
    )
    _plot_five_family_weight_trajectory(
        path=mean_weight_path,
        records=records,
        family_by_method=family_by_method,
        candidate_by_method=candidate_by_method,
        statistic="solve_mean",
        window=int(rolling_window),
    )
    if "bandit_ppo" in records:
        _plot_ppo_weight_detail(
            path=ppo_weight_detail_path,
            rows=records["bandit_ppo"],
            window=int(rolling_window),
        )
    learned_action_methods = [
        method
        for method, family in family_by_method.items()
        if family not in {"default_setup", "default", "fixed_w1.6", "ppo"}
    ]
    if learned_action_methods:
        max_cycles = max(
            len(row["outcome"].get("cycle_actions", []))
            for method in learned_action_methods
            for row in records[method]
        )
        action_min = min(
            min(row["outcome"].get("cycle_actions", [np.inf]))
            for method in learned_action_methods
            for row in records[method]
        )
        action_max = max(
            max(row["outcome"].get("cycle_actions", [-np.inf]))
            for method in learned_action_methods
            for row in records[method]
        )
        plot_action_trajectory_grid(
            {(0, method): records[method] for method in learned_action_methods},
            (0,),
            max_cycles,
            action_trajectory_path,
            action_min=float(action_min),
            action_max=float(action_max),
            methods=tuple(learned_action_methods),
            method_labels={
                method: FAMILY_LABELS[family_by_method[method]]
                for method in learned_action_methods
            },
            column_labels={0: f"Persistent {expected_cases:,}-instance online comparison"},
            x_label="Online comparison instance",
            title=(
                "Per-instance, per-cycle learned action trajectories "
                "(white: solve already terminated)"
            ),
            panel_width=10.5,
            panel_height=3.2,
        )
    if "bandit_recursive_lstdq_lcb" in records:
        best_start, best_stop = _best_complete_native_solve_window(
            records["bandit_recursive_lstdq_lcb"],
        )
        _plot_cycle_action_trajectory_lines(
            path=lstdq_trajectory_lines_path,
            rows=records["bandit_recursive_lstdq_lcb"],
            method_label=FAMILY_LABELS["recursive_lstdq_lcb"],
            start_instance=best_start,
            stop_instance=best_stop,
        )
    _plot_cumulative_components(
        path=cumulative_path,
        records=records,
        family_by_method=family_by_method,
    )
    _plot_all_runtime_breakdown(path=all_runtime_path, result=result)
    if "last_1000" in result["windows"]:
        _plot_all_runtime_breakdown(
            path=last_1000_runtime_path,
            result=result,
            window_key="last_1000",
            figure_title="Last 1,000 online comparison instances (3,001-4,000)",
        )
    _plot_recovery_outcomes(path=recovery_path, result=result)
    has_split_windows = {"first_1000", "last_1000"}.issubset(result["windows"])
    if has_split_windows:
        _plot_first_last_components(
            path=first_last_components_path,
            result=result,
        )
        _plot_first_last_improvement_ci(
            path=first_last_ci_path,
            result=result,
        )

    summary_csv_path = artifact_dir / f"summary_{expected_cases}.csv"
    _write_summary_csv(summary_csv_path, records, family_by_method)
    summary_rows = list(csv.DictReader(summary_csv_path.open(encoding="utf-8")))
    reference_method = (
        "bandit_default"
        if any(row["method"] == "bandit_default" for row in summary_rows)
        else "bandit_fixed_w1.6"
    )
    reference_row = next(
        row for row in summary_rows if row["method"] == reference_method
    )
    reference_solve = float(reference_row["mean_native_solve_runtime_sec"])
    table_rows = []
    for row in summary_rows:
        method = row["method"]
        solve = float(row["mean_native_solve_runtime_sec"])
        if method in candidate_by_method:
            display = (
                f"{family_by_method[method]} "
                f"a={float(candidate_by_method[method]['alpha']):g}, "
                f"lambda={float(candidate_by_method[method]['trace_lambda']):g}"
            )
        else:
            display = FAMILY_LABELS[family_by_method[method]]
        table_rows.append(
            {
                "method": display,
                "setup_ms": f"{1_000.0 * float(row['mean_setup_runtime_sec']):.3f}",
                "solve_ms": f"{1_000.0 * solve:.3f}",
                "native_total_ms": f"{1_000.0 * float(row['mean_native_total_runtime_sec']):.3f}",
                "solve_vs_default": (
                    f"{100.0 * (reference_solve - solve) / reference_solve:+.2f}%"
                ),
                "controller_ms": f"{1_000.0 * float(row['mean_controller_runtime_sec']):.3f}",
                "bandit_overhead_ms": f"{1_000.0 * float(row['mean_setup_bandit_overhead_sec']):.3f}",
                "end_to_end_ms": f"{1_000.0 * float(row['mean_end_to_end_runtime_sec']):.3f}",
                "primary_failures": str(int(row["primary_failures"])),
                "recovered_failures": str(int(row["recovered_failures"])),
                "unrecovered_failures": str(int(row["unrecovered_failures"])),
            }
        )
    table_path = artifact_dir / f"summary_{expected_cases}.md"
    _write_markdown_table(table_path, table_rows)
    output = {
        "native_solve_trajectory": str(solve_path),
        "setup_trajectory": str(setup_path),
        "first_cycle_weight_trajectory": str(first_weight_path),
        "mean_weight_trajectory": str(mean_weight_path),
        "cumulative_components": str(cumulative_path),
        f"all_{expected_cases}_runtime_breakdown": str(all_runtime_path),
        "primary_and_fallback_outcomes": str(recovery_path),
        "summary_markdown": str(table_path),
        "summary_csv": str(summary_csv_path),
        "same_setup_audit": str(same_setup_path),
        "rolling_window": int(rolling_window),
    }
    if "bandit_ppo" in records:
        output["ppo_weight_detail"] = str(ppo_weight_detail_path)
    if learned_action_methods:
        output["learned_per_cycle_action_trajectories"] = str(
            action_trajectory_path
        )
    if "bandit_recursive_lstdq_lcb" in records:
        output["recursive_lstdq_lcb_best_window_cycle_trajectory_lines"] = str(
            lstdq_trajectory_lines_path
        )
    if has_split_windows:
        output["last_1000_runtime_breakdown"] = str(last_1000_runtime_path)
        output["first_vs_last_components"] = str(first_last_components_path)
        output["first_vs_last_improvement_ci"] = str(first_last_ci_path)
    if analysis_stop is not None:
        output["analysis_scope"] = str(artifact_dir / "analysis_scope.json")
        output["analysis_result"] = str(artifact_dir / "analysis_result.json")
    _write_json(artifact_dir / "plot_summary.json", output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a locked persistent online solve-controller study."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=100)
    parser.add_argument(
        "--analysis-stop",
        type=int,
        default=None,
        help="Analyze only records [0, analysis_stop) without changing raw results.",
    )
    args = parser.parse_args()
    output = generate_plots(
        result_dir=args.result_dir,
        rolling_window=args.rolling_window,
        analysis_stop=args.analysis_stop,
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
