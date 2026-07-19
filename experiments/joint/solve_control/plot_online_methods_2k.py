from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from online_td_experiment_common import _write_json


METHOD_LABELS = {
    "default_setup": "Default setup + solve",
    "bandit_default": "Online bandit + default solve",
    "bandit_fixed_w1.6": "Online bandit + fixed w=1.6",
    "bandit_ppo": "Online bandit + frozen PPO",
    "bandit_td": "Online bandit + true-online SARSA",
}
METHOD_COLORS = {
    "default_setup": "#4C4C4C",
    "bandit_default": "#2F6B9A",
    "bandit_fixed_w1.6": "#D28B26",
    "bandit_ppo": "#6F5AA8",
    "bandit_td": "#2C8C6A",
}


def _outcomes(records: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    return [dict(record["outcome"]) for record in records]


def _metric_values(
    records: Sequence[Dict[str, Any]],
    metric: str,
) -> np.ndarray:
    outcomes = _outcomes(records)
    if metric == "setup_runtime":
        return np.asarray([float(row["setup_runtime"]) for row in outcomes])
    if metric == "native_solve_runtime":
        return np.asarray([float(row["solve_runtime"]) for row in outcomes])
    if metric == "native_total_runtime":
        return np.asarray([float(row["runtime"]) for row in outcomes])
    if metric == "controller_runtime":
        return np.asarray([float(row.get("infer_runtime", 0.0)) for row in outcomes])
    if metric == "setup_bandit_overhead":
        return np.asarray(
            [float(row.get("bandit_overhead_runtime", 0.0)) for row in outcomes]
        )
    if metric == "end_to_end_runtime":
        return np.asarray([float(row["end_to_end_runtime"]) for row in outcomes])
    raise ValueError(f"Unknown metric: {metric}")


def _bootstrap_mean_interval(values: np.ndarray, *, seed: int) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    means = np.empty(2000, dtype=float)
    for index in range(means.size):
        sample = rng.integers(0, array.size, size=array.size)
        means[index] = float(np.mean(array[sample]))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def rolling_improvement(
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if baseline.shape != candidate.shape:
        raise ValueError("Rolling comparison streams must align")
    if window <= 0 or window > baseline.size:
        raise ValueError("Rolling window must fit within the stream")
    kernel = np.ones(int(window), dtype=float)
    baseline_sum = np.convolve(baseline, kernel, mode="valid")
    candidate_sum = np.convolve(candidate, kernel, mode="valid")
    return 100.0 * (baseline_sum - candidate_sum) / baseline_sum


def _plot_cumulative(
    records: Dict[str, list[Dict[str, Any]]],
    path: Path,
) -> None:
    metrics = (
        ("setup_runtime", "Cumulative native setup (s)"),
        ("native_solve_runtime", "Cumulative native solve (s)"),
        ("end_to_end_runtime", "Cumulative end-to-end (s)"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        for method, method_records in records.items():
            values = _metric_values(method_records, metric)
            axis.plot(
                np.arange(1, values.size + 1),
                np.cumsum(values),
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                linewidth=1.7,
            )
        axis.set_title(title)
        axis.set_xlabel("Online instance")
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[0].set_ylabel("Seconds")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_rolling(
    records: Dict[str, list[Dict[str, Any]]],
    path: Path,
    *,
    window: int,
) -> None:
    baseline_method = "bandit_default"
    metrics = (
        ("native_solve_runtime", "Native solve improvement"),
        ("end_to_end_runtime", "End-to-end improvement"),
    )
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for axis, (metric, title) in zip(axes, metrics):
        baseline = _metric_values(records[baseline_method], metric)
        x = np.arange(window, baseline.size + 1)
        for method, method_records in records.items():
            if method == baseline_method:
                continue
            values = rolling_improvement(
                baseline,
                _metric_values(method_records, metric),
                window=window,
            )
            axis.plot(
                x,
                values,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                linewidth=1.5,
            )
        axis.axhline(0.0, color="#202020", linewidth=0.9)
        axis.set_title(f"{title} vs online bandit + default solve (rolling {window})")
        axis.set_ylabel("Improvement (%)")
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[-1].set_xlabel("Online instance")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_first_last_components(
    records: Dict[str, list[Dict[str, Any]]],
    path: Path,
    *,
    seed: int,
) -> None:
    methods = list(records)
    windows = {
        "First 1000": slice(0, 1000),
        "Last 1000": slice(-1000, None),
    }
    metrics = (
        ("setup_runtime", "Native setup", 1000.0),
        ("native_solve_runtime", "Native solve", 1000.0),
        ("end_to_end_runtime", "End-to-end", 1000.0),
    )
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    x = np.arange(len(methods), dtype=float)
    width = 0.36
    for metric_index, (axis, (metric, title, scale)) in enumerate(zip(axes, metrics)):
        for window_index, (window_name, window_slice) in enumerate(windows.items()):
            means = []
            lower_errors = []
            upper_errors = []
            for method_index, method in enumerate(methods):
                values = _metric_values(records[method][window_slice], metric) * scale
                mean = float(np.mean(values))
                low, high = _bootstrap_mean_interval(
                    values,
                    seed=int(seed + 100 * metric_index + 10 * window_index + method_index),
                )
                means.append(mean)
                lower_errors.append(mean - low)
                upper_errors.append(high - mean)
            offset = (-0.5 if window_index == 0 else 0.5) * width
            axis.bar(
                x + offset,
                means,
                width=width,
                yerr=np.asarray([lower_errors, upper_errors]),
                capsize=2.5,
                label=window_name,
                color=("#8FB7D8" if window_index == 0 else "#2C8C6A"),
                edgecolor="#333333",
                linewidth=0.5,
            )
        axis.set_title(f"{title} mean with 95% CI")
        axis.set_ylabel("Milliseconds per instance")
        axis.set_xticks(x)
        axis.set_xticklabels(
            [METHOD_LABELS.get(method, method) for method in methods],
            rotation=28,
            ha="right",
        )
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axes[0].legend(frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_td_actions(
    records: Sequence[Dict[str, Any]],
    path: Path,
    *,
    rolling_window: int,
) -> Dict[str, Any]:
    outcomes = _outcomes(records)
    max_cycles = max(len(row.get("cycle_actions", [])) for row in outcomes)
    action_matrix = np.full((len(outcomes), max_cycles), np.nan, dtype=float)
    explored_per_episode = np.full(len(outcomes), np.nan, dtype=float)
    epsilon_per_episode = np.asarray(
        [float(row.get("epsilon", np.nan)) for row in outcomes], dtype=float
    )
    for episode, row in enumerate(outcomes):
        actions = np.asarray(row.get("cycle_actions", []), dtype=float)
        action_matrix[episode, : actions.size] = actions
        explored = row.get("cycle_explored", [])
        if explored:
            explored_per_episode[episode] = float(np.mean(np.asarray(explored, dtype=float)))

    weights = np.round(np.arange(1.0, 2.01, 0.1), 1)
    distribution_cycles = min(max_cycles, 30)
    distribution = np.zeros((weights.size, distribution_cycles), dtype=float)
    for cycle in range(distribution_cycles):
        values = action_matrix[:, cycle]
        values = values[np.isfinite(values)]
        if values.size:
            for weight_index, weight in enumerate(weights):
                distribution[weight_index, cycle] = float(
                    np.mean(np.isclose(values, weight, atol=1.0e-9, rtol=0.0))
                )

    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    image = axes[0].imshow(
        action_matrix.T,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=1.0,
        vmax=2.0,
        cmap="viridis",
    )
    axes[0].set_title("True-online SARSA action trajectory")
    axes[0].set_ylabel("AMG cycle")
    axes[0].set_xlabel("Online instance")
    figure.colorbar(image, ax=axes[0], label="Relaxation weight w")

    distribution_image = axes[1].imshow(
        distribution,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=0.0,
        vmax=max(0.01, float(np.max(distribution))),
        cmap="magma",
    )
    axes[1].set_title("Greedy/exploratory action distribution by cycle")
    axes[1].set_ylabel("Weight")
    axes[1].set_xlabel("AMG cycle")
    axes[1].set_yticks(np.arange(weights.size))
    axes[1].set_yticklabels([f"{weight:.1f}" for weight in weights])
    figure.colorbar(distribution_image, ax=axes[1], label="Fraction of actions")

    kernel = np.ones(int(rolling_window), dtype=float) / float(rolling_window)
    x = np.arange(rolling_window, len(outcomes) + 1)
    axes[2].plot(
        x,
        np.convolve(epsilon_per_episode, kernel, mode="valid"),
        color="#2F6B9A",
        label="Behavior epsilon",
    )
    if np.any(np.isfinite(explored_per_episode)):
        actual = np.nan_to_num(explored_per_episode, nan=0.0)
        axes[2].plot(
            x,
            np.convolve(actual, kernel, mode="valid"),
            color="#D28B26",
            label="Actual exploratory-action fraction",
        )
    axes[2].set_title(f"Exploration trajectory (rolling {rolling_window})")
    axes[2].set_xlabel("Online instance")
    axes[2].set_ylabel("Fraction")
    axes[2].set_ylim(bottom=0.0)
    axes[2].grid(alpha=0.22, linewidth=0.7)
    axes[2].legend(frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return {
        "instances": int(len(outcomes)),
        "max_cycles": int(max_cycles),
        "mean_action": float(np.nanmean(action_matrix)),
        "mean_recorded_exploration_fraction": (
            float(np.nanmean(explored_per_episode))
            if np.any(np.isfinite(explored_per_episode))
            else None
        ),
    }


def generate_plots(
    *,
    result_path: Path,
    output_dir: Path,
    rolling_window: int = 100,
) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    records = dict(payload["online_records"])
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cumulative": output_dir / "cumulative_components.png",
        "rolling": output_dir / "rolling_100_improvements.png",
        "first_last": output_dir / "first_vs_last_1000_components.png",
        "td_actions": output_dir / "true_online_action_trajectory.png",
    }
    _plot_cumulative(records, paths["cumulative"])
    _plot_rolling(records, paths["rolling"], window=int(rolling_window))
    _plot_first_last_components(
        records,
        paths["first_last"],
        seed=int(payload["protocol"]["seed"]),
    )
    action_summary = _plot_td_actions(
        records["bandit_td"],
        paths["td_actions"],
        rolling_window=int(rolling_window),
    )
    summary = {
        "result": str(result_path),
        "figures": {name: str(path) for name, path in paths.items()},
        "rolling_window": int(rolling_window),
        "td_action_summary": action_summary,
    }
    _write_json(output_dir / "visualization_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the five-method online 2K outcome.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=100)
    args = parser.parse_args()
    summary = generate_plots(
        result_path=args.result,
        output_dir=args.output_dir,
        rolling_window=int(args.rolling_window),
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
