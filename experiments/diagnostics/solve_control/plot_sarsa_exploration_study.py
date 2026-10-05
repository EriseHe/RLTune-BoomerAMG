from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "fixed_w1.6",
    "ppo",
    "absolute_uniform",
    "absolute_local",
    "absolute_uncertainty_lcb",
    "residual_uniform",
    "residual_local",
    "residual_uncertainty_lcb",
)
SARSA_METHODS = METHODS[2:]
LABELS = {
    "fixed_w1.6": "Fixed w=1.6",
    "ppo": "Exp44 PPO",
    "absolute_uniform": "Absolute + uniform epsilon",
    "absolute_local": "Absolute + local epsilon",
    "absolute_uncertainty_lcb": "Absolute + uncertainty LCB",
    "residual_uniform": "Residual + uniform epsilon",
    "residual_local": "Residual + local epsilon",
    "residual_uncertainty_lcb": "Residual + uncertainty LCB",
}
COLORS = {
    "fixed_w1.6": "#242424",
    "ppo": "#d1495b",
    "absolute_uniform": "#00798c",
    "absolute_local": "#edae49",
    "absolute_uncertainty_lcb": "#30638e",
    "residual_uniform": "#2a9d8f",
    "residual_local": "#f4a261",
    "residual_uncertainty_lcb": "#6a4c93",
}

WINDOW_LABELS = {
    "all_1000": "All 1,000",
    "first_500": "First 500",
    "last_500": "Last 500",
    "last_250": "Last 250",
    "all_2000": "All 2,000",
    "first_1000": "First 1,000",
    "last_1000": "Last 1,000",
}


def _ordered_methods(data: Dict[str, Any]) -> tuple[str, ...]:
    known = [method for method in METHODS if method in data]
    return tuple(known + sorted(set(data) - set(known)))


def _sarsa_methods(methods: Sequence[str]) -> tuple[str, ...]:
    return tuple(method for method in methods if method not in {"fixed_w1.6", "ppo"})


def _subplot_grid(count: int, *, width: float = 5.0, height: float = 4.0) -> tuple[Any, np.ndarray]:
    columns = min(3, max(1, count))
    rows = max(1, math.ceil(count / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(width * columns, height * rows), squeeze=False)
    return figure, axes


def _read_csv(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _group_performance(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, np.ndarray]]:
    grouped: Dict[str, list[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    fields = (
        "setup_runtime_sec",
        "native_solve_runtime_sec",
        "native_total_runtime_sec",
        "controller_runtime_sec",
        "end_to_end_runtime_sec",
        "mean_physical_weight",
        "first_weight",
        "epsilon",
    )
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for method, method_rows in grouped.items():
        ordered = sorted(method_rows, key=lambda row: int(row["case_index"]))
        result[method] = {
            field: np.asarray([float(row[field]) for row in ordered], dtype=float)
            for field in fields
        }
    return result


def _rolling(values: np.ndarray, window: int) -> np.ndarray:
    if values.size < window:
        return np.empty(0, dtype=float)
    return np.convolve(values, np.ones(window, dtype=float) / window, mode="valid")


def _style_axis(axis: Any) -> None:
    axis.grid(axis="y", color="#d8d8d8", linewidth=0.7, alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_cumulative(
    data: Dict[str, Dict[str, np.ndarray]],
    output: Path,
) -> None:
    methods = _ordered_methods(data)
    fields = (
        ("setup_runtime_sec", "Cumulative setup time", "seconds"),
        ("native_solve_runtime_sec", "Cumulative native solve time", "seconds"),
        ("end_to_end_runtime_sec", "Cumulative end-to-end time", "seconds"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for axis, (field, title, ylabel) in zip(axes, fields):
        for method in methods:
            values = data[method][field]
            axis.plot(
                np.arange(1, values.size + 1),
                np.cumsum(values),
                label=LABELS[method],
                color=COLORS[method],
                linewidth=2.1 if method in {"fixed_w1.6", "ppo"} else 1.4,
                alpha=0.95,
            )
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylabel(ylabel)
        _style_axis(axis)
    axes[-1].set_xlabel("Online case")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    case_count = len(data[methods[0]]["setup_runtime_sec"])
    figure.suptitle(f"Direct online performance on the shared {case_count:,}-case stream", fontsize=15)
    figure.tight_layout(rect=(0, 0.08, 1, 0.97))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_rolling_improvement(
    data: Dict[str, Dict[str, np.ndarray]],
    output: Path,
    *,
    window: int = 100,
) -> None:
    methods = _ordered_methods(data)
    fields = (
        ("native_solve_runtime_sec", "Native solve improvement vs fixed w=1.6"),
        ("end_to_end_runtime_sec", "End-to-end improvement vs fixed w=1.6"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for axis, (field, title) in zip(axes, fields):
        reference = _rolling(data["fixed_w1.6"][field], window)
        for method in methods:
            if method == "fixed_w1.6":
                continue
            candidate = _rolling(data[method][field], window)
            improvement = 100.0 * (reference - candidate) / reference
            axis.plot(
                np.arange(window, window + improvement.size),
                improvement,
                label=LABELS[method],
                color=COLORS[method],
                linewidth=2.0 if method == "ppo" else 1.5,
            )
        axis.axhline(0.0, color="#242424", linewidth=1.0)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylabel("improvement (%)")
        _style_axis(axis)
    axes[-1].set_xlabel("Trailing-window endpoint")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle(f"Rolling {window}-case paired performance", fontsize=15)
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_windows(result: Dict[str, Any], output: Path) -> None:
    preferred = (
        ("all_2000", "first_1000", "last_1000", "last_500")
        if "all_2000" in result["windows"]
        else ("all_1000", "first_500", "last_500", "last_250")
    )
    window_names = tuple(name for name in preferred if name in result["windows"])
    methods = _ordered_methods(result["windows"][window_names[0]]["methods"])
    candidates = tuple(method for method in methods if method != "fixed_w1.6")
    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    metrics = (
        ("native_solve_runtime", "Native solve improvement vs fixed w=1.6"),
        ("end_to_end_runtime", "End-to-end improvement vs fixed w=1.6"),
    )
    x = np.arange(len(window_names), dtype=float)
    width = min(0.22, 0.8 / max(1, len(candidates)))
    for axis, (metric, title) in zip(axes, metrics):
        for index, method in enumerate(candidates):
            points = []
            lower = []
            upper = []
            for window_name in window_names:
                row = result["windows"][window_name]["comparisons_vs_fixed_w1.6"][method][metric]
                point = float(row["candidate_improvement_pct"])
                interval = row["candidate_improvement_95pct"]
                points.append(point)
                lower.append(point - float(interval[0]))
                upper.append(float(interval[1]) - point)
            offset = (index - (len(candidates) - 1) / 2.0) * width
            axis.bar(
                x + offset,
                points,
                width=width,
                color=COLORS[method],
                label=LABELS[method],
                alpha=0.9,
            )
            axis.errorbar(
                x + offset,
                points,
                yerr=np.asarray([lower, upper]),
                fmt="none",
                ecolor="#333333",
                elinewidth=0.7,
                capsize=1.5,
            )
        axis.axhline(0.0, color="#242424", linewidth=1.0)
        axis.set_title(title, loc="left", fontsize=12, fontweight="bold")
        axis.set_ylabel("improvement (%)")
        _style_axis(axis)
    axes[-1].set_xticks(x, [WINDOW_LABELS[name] for name in window_names])
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("Online-window performance with paired 95% bootstrap intervals", fontsize=15)
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_weight_trajectories(
    data: Dict[str, Dict[str, np.ndarray]],
    output: Path,
    *,
    window: int = 25,
) -> None:
    methods = _sarsa_methods(_ordered_methods(data))
    figure, axes = plt.subplots(
        2,
        len(methods),
        figsize=(5.2 * len(methods), 8.0),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fields = (
        ("first_weight", "cycle 0 physical w", 1.0),
        ("mean_physical_weight", "mean physical w per solve", 1.6),
    )
    for column, method in enumerate(methods):
        for row, (field, ylabel, reference) in enumerate(fields):
            axis = axes[row, column]
            values = data[method][field]
            x = np.arange(1, values.size + 1)
            axis.scatter(x, values, s=6, color=COLORS[method], alpha=0.13, linewidths=0)
            smoothed = _rolling(values, window)
            axis.plot(
                np.arange(window, window + smoothed.size),
                smoothed,
                color=COLORS[method],
                linewidth=2.0,
            )
            axis.axhline(reference, color="#242424", linewidth=0.9, linestyle="--")
            axis.set_ylim(0.98, 2.02)
            axis.set_ylabel(ylabel if column == 0 else "")
            _style_axis(axis)
        axes[0, column].scatter(
            [1],
            [data[method]["first_weight"][0]],
            s=70,
            marker="*",
            color="#242424",
            zorder=5,
        )
        axes[0, column].set_title(LABELS[method], fontsize=11, fontweight="bold")
        axes[-1, column].set_xlabel("Online case")
    figure.suptitle(
        f"Cycle-0 actions and per-solve means (raw cases + rolling {window}-case mean)",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_cycle_profiles(action_rows: Sequence[Dict[str, str]], output: Path) -> None:
    grouped: Dict[str, Dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in action_rows:
        grouped[row["method"]][int(row["cycle"])].append(float(row["physical_weight"]))
    methods = _ordered_methods(grouped)
    figure, axes = _subplot_grid(len(methods))
    for axis, method in zip(axes.flat, methods):
        cycles = np.asarray(sorted(cycle for cycle in grouped[method] if cycle < 20), dtype=int)
        means = np.asarray([np.mean(grouped[method][int(cycle)]) for cycle in cycles])
        p10 = np.asarray([np.percentile(grouped[method][int(cycle)], 10) for cycle in cycles])
        p90 = np.asarray([np.percentile(grouped[method][int(cycle)], 90) for cycle in cycles])
        axis.fill_between(cycles, p10, p90, color=COLORS[method], alpha=0.18)
        axis.plot(cycles, means, color=COLORS[method], linewidth=2.0)
        axis.axhline(1.6, color="#242424", linewidth=0.9, linestyle="--")
        axis.set_title(LABELS[method], fontsize=11, fontweight="bold")
        axis.set_ylim(0.98, 2.02)
        _style_axis(axis)
    for index, axis in enumerate(axes.flat):
        if index >= len(methods):
            axis.set_visible(False)
            continue
        axis.set_xlabel("AMG cycle")
        if index % axes.shape[1] == 0:
            axis.set_ylabel("physical w")
    figure.suptitle("Per-cycle action profile (mean and 10th-90th percentile)", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_exploration(action_rows: Sequence[Dict[str, str]], output: Path) -> None:
    per_case: Dict[str, Dict[int, list[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in action_rows:
        per_case[row["method"]][int(row["case_index"])].append(row)
    figure, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    for method in _ordered_methods(per_case):
        cases = sorted(per_case[method])
        explored = np.asarray(
            [
                np.mean([entry["explored"] == "True" for entry in per_case[method][case]])
                for case in cases
            ],
            dtype=float,
        )
        uncertainty = np.asarray(
            [
                np.mean(
                    [float(entry["selected_uncertainty_sec"]) for entry in per_case[method][case]]
                )
                for case in cases
            ],
            dtype=float,
        )
        axes[0].plot(
            np.arange(100, 100 + _rolling(explored, 100).size),
            _rolling(explored, 100),
            label=LABELS[method],
            color=COLORS[method],
            linewidth=1.7,
        )
        if "uncertainty_lcb" in method:
            axes[1].plot(
                np.arange(100, 100 + _rolling(uncertainty, 100).size),
                1_000.0 * _rolling(uncertainty, 100),
                label=LABELS[method],
                color=COLORS[method],
                linewidth=2.0,
            )
    axes[0].set_title("Rolling 100-case non-greedy action rate", loc="left", fontweight="bold")
    axes[0].set_ylabel("rate")
    axes[1].set_title("Rolling 100-case selected LCB uncertainty", loc="left", fontweight="bold")
    axes[1].set_ylabel("milliseconds")
    axes[1].set_xlabel("Online case")
    for axis in axes:
        _style_axis(axis)
        axis.legend(frameon=False, ncol=3)
    figure.suptitle("Exploration behavior over the online stream", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(input_dir: Path) -> list[Path]:
    result = json.loads((input_dir / "result.json").read_text(encoding="utf-8"))
    performance_rows = _read_csv(input_dir / "online_performance.csv")
    action_rows = _read_csv(input_dir / "action_trajectory.csv")
    data = _group_performance(performance_rows)
    figures = input_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = [
        figures / "cumulative_times.png",
        figures / "rolling100_improvement.png",
        figures / "window_performance.png",
        figures / "online_weight_trajectories.png",
        figures / "cycle_action_profiles.png",
        figures / "exploration_trajectory.png",
    ]
    plot_cumulative(data, outputs[0])
    plot_rolling_improvement(data, outputs[1])
    plot_windows(result, outputs[2])
    plot_weight_trajectories(data, outputs[3])
    plot_cycle_profiles(action_rows, outputs[4])
    plot_exploration(action_rows, outputs[5])
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the 1K SARSA exploration study.")
    parser.add_argument("--input-dir", type=Path, required=True)
    outputs = run(parser.parse_args().input_dir)
    print(json.dumps({"figures": [str(path) for path in outputs]}, indent=2))


if __name__ == "__main__":
    main()
