from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from online_td_experiment_common import _write_json


METHOD_LABELS = {
    "default_setup": "Default setup + solve",
    "bandit_default": "Bandit + default solve",
    "bandit_fixed_w1.6": "Bandit + fixed w=1.6",
    "bandit_ppo": "Bandit + frozen PPO",
    "bandit_sarsa": "Bandit + online SARSA",
}
METHOD_COLORS = {
    "default_setup": "#4C4C4C",
    "bandit_default": "#3B78A0",
    "bandit_fixed_w1.6": "#D28B26",
    "bandit_ppo": "#755AA8",
    "bandit_sarsa": "#24866B",
}


def _load_seed_results(payload: Dict[str, Any]) -> list[Dict[str, Any]]:
    return [
        json.loads(Path(value["result_path"]).read_text(encoding="utf-8"))
        for _seed, value in sorted(payload["per_seed"].items(), key=lambda item: int(item[0]))
    ]


def _metric(records: Sequence[Dict[str, Any]], name: str) -> np.ndarray:
    outcomes = [record["outcome"] for record in records]
    if name == "setup":
        return np.asarray([float(row["setup_runtime"]) for row in outcomes], dtype=float)
    if name == "solve":
        return np.asarray([float(row["solve_runtime"]) for row in outcomes], dtype=float)
    if name == "native_total":
        return np.asarray([float(row["runtime"]) for row in outcomes], dtype=float)
    if name == "controller":
        return np.asarray([float(row.get("infer_runtime", 0.0)) for row in outcomes], dtype=float)
    if name == "end_to_end":
        return np.asarray([float(row["end_to_end_runtime"]) for row in outcomes], dtype=float)
    raise ValueError(name)


def _rolling_improvement(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    kernel = np.ones(int(window), dtype=float)
    reference_sum = np.convolve(reference, kernel, mode="valid")
    candidate_sum = np.convolve(candidate, kernel, mode="valid")
    return 100.0 * (reference_sum - candidate_sum) / reference_sum


def _paired_improvement_ci(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
    draws: int = 2000,
) -> tuple[float, tuple[float, float]]:
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    if reference.shape != candidate.shape or reference.ndim != 1 or reference.size == 0:
        raise ValueError("Paired comparison requires aligned non-empty vectors")
    reference_mean = float(np.mean(reference))
    improvement = 100.0 * (reference_mean - float(np.mean(candidate))) / reference_mean
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, reference.size, size=(int(draws), reference.size))
    reference_means = np.mean(reference[indices], axis=1)
    candidate_means = np.mean(candidate[indices], axis=1)
    samples = 100.0 * (reference_means - candidate_means) / reference_means
    return float(improvement), (
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def _plot_components(payload: Dict[str, Any], path: Path) -> None:
    methods = list(payload["methods"])
    setup = np.asarray(
        [payload["methods"][method]["means_sec"]["setup_runtime"] for method in methods]
    ) * 1000.0
    solve = np.asarray(
        [payload["methods"][method]["means_sec"]["native_solve_runtime"] for method in methods]
    ) * 1000.0
    controller = np.asarray(
        [payload["methods"][method]["means_sec"]["controller_runtime"] for method in methods]
    ) * 1000.0
    x = np.arange(len(methods))
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)
    colors = [METHOD_COLORS[method] for method in methods]
    axes[0].bar(x, setup, color=colors, alpha=0.48, label="Native setup")
    axes[0].bar(x, solve, bottom=setup, color=colors, label="Native solve")
    axes[0].set_title("Native runtime components, 5,000 cases")
    axes[0].set_ylabel("Milliseconds per instance")
    axes[0].legend(frameon=False)

    native = setup + solve
    axes[1].bar(x, native, color=colors, label="Native total")
    axes[1].bar(
        x,
        controller,
        bottom=native,
        color="#EFEFEF",
        edgecolor=colors,
        hatch="///",
        label="Controller inference/update",
    )
    axes[1].set_title("Controller-inclusive runtime\n(shared bandit overhead excluded)")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(
            [METHOD_LABELS[method] for method in methods], rotation=27, ha="right"
        )
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_forest(payload: Dict[str, Any], path: Path) -> None:
    comparisons = payload["clustered_comparisons"]
    rows = []
    for comparison, metrics in comparisons.items():
        for metric in ("native_solve_runtime", "native_total_runtime", "end_to_end_runtime"):
            value = metrics[metric]
            rows.append(
                (
                    f"{comparison.replace('_', ' ')} | {metric.replace('_runtime', '').replace('_', ' ')}",
                    float(value["candidate_improvement_pct"]),
                    tuple(float(x) for x in value["candidate_improvement_95pct"]),
                )
            )
    figure, axis = plt.subplots(figsize=(11, 8), constrained_layout=True)
    y = np.arange(len(rows))[::-1]
    points = np.asarray([row[1] for row in rows])
    lows = np.asarray([row[2][0] for row in rows])
    highs = np.asarray([row[2][1] for row in rows])
    axis.errorbar(
        points,
        y,
        xerr=np.vstack([points - lows, highs - points]),
        fmt="o",
        color="#24866B",
        ecolor="#555555",
        capsize=3,
    )
    axis.axvline(0.0, color="#202020", linewidth=1.0)
    axis.set_yticks(y)
    axis.set_yticklabels([row[0] for row in rows])
    axis.set_xlabel("Candidate improvement (%)")
    axis.set_title("Clustered paired improvements with 95% CI")
    axis.grid(axis="x", alpha=0.22, linewidth=0.7)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_rolling(
    seed_results: Sequence[Dict[str, Any]],
    path: Path,
    *,
    window: int,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, constrained_layout=True)
    comparisons = (
        ("bandit_fixed_w1.6", "SARSA vs fixed w=1.6"),
        ("bandit_ppo", "SARSA vs frozen PPO"),
    )
    for axis, (reference_method, title) in zip(axes, comparisons):
        for result in seed_results:
            records = result["evaluation"]["records"]
            improvement = _rolling_improvement(
                _metric(records[reference_method], "solve"),
                _metric(records["bandit_sarsa"], "solve"),
                window=int(window),
            )
            axis.plot(
                np.arange(window, window + improvement.size),
                improvement,
                linewidth=1.25,
                label=str(result["forward_seed"]),
            )
        axis.axhline(0.0, color="#202020", linewidth=0.9)
        axis.set_title(f"{title}, rolling {window} native solve")
        axis.set_ylabel("Improvement (%)")
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[-1].set_xlabel("Cases within 1500:2500")
    axes[0].legend(title="Forward seed", ncol=5, frameon=False)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_ppo_per_seed(seed_results: Sequence[Dict[str, Any]], path: Path) -> None:
    seeds = [int(result["forward_seed"]) for result in seed_results]
    metric_specs = (
        ("Native solve", False, "#755AA8"),
        ("Solve + PPO inference", True, "#B6A6D8"),
    )
    points: list[list[float]] = [[] for _ in metric_specs]
    intervals: list[list[tuple[float, float]]] = [[] for _ in metric_specs]
    for result in seed_results:
        records = result["evaluation"]["records"]
        reference = _metric(records["bandit_default"], "solve")
        ppo_solve = _metric(records["bandit_ppo"], "solve")
        ppo_controller = _metric(records["bandit_ppo"], "controller")
        for metric_index, (_label, include_controller, _color) in enumerate(metric_specs):
            candidate = ppo_solve + ppo_controller if include_controller else ppo_solve
            point, interval = _paired_improvement_ci(
                reference,
                candidate,
                seed=int(result["forward_seed"] + 10_000 * metric_index),
            )
            points[metric_index].append(point)
            intervals[metric_index].append(interval)

    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    x = np.arange(len(seeds), dtype=float)
    width = 0.36
    for metric_index, (label, _include_controller, color) in enumerate(metric_specs):
        values = np.asarray(points[metric_index], dtype=float)
        lows = np.asarray([item[0] for item in intervals[metric_index]], dtype=float)
        highs = np.asarray([item[1] for item in intervals[metric_index]], dtype=float)
        positions = x + (metric_index - 0.5) * width
        bars = axis.bar(
            positions,
            values,
            width=width,
            color=color,
            label=label,
            yerr=np.vstack([values - lows, highs - values]),
            capsize=4,
            error_kw={"elinewidth": 1.2, "ecolor": "#454545"},
        )
        axis.bar_label(bars, labels=[f"{value:+.1f}%" for value in values], padding=5)
    axis.axhline(0.0, color="#202020", linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels([str(seed) for seed in seeds])
    axis.set_xlabel("Forward seed (1,000 cases each, window 1500:2500)")
    axis.set_ylabel("PPO improvement vs bandit + default solve (%)")
    axis.set_title("Frozen PPO performance by forward seed\nSame setup on every paired case")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _episode_action_summary(records: Sequence[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    exploration = []
    for record in records:
        outcome = record["outcome"]
        actions = np.asarray(outcome.get("cycle_actions", []), dtype=float)
        explored = np.asarray(outcome.get("cycle_explored", []), dtype=float)
        means.append(float(np.mean(actions)) if actions.size else np.nan)
        exploration.append(float(np.mean(explored)) if explored.size else np.nan)
    return np.asarray(means), np.asarray(exploration)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    finite = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0)
    return np.convolve(finite, np.ones(window) / float(window), mode="valid")


def _by_cycle_mean(records: Sequence[Dict[str, Any]], max_cycles: int = 30) -> np.ndarray:
    buckets: list[list[float]] = [[] for _ in range(max_cycles)]
    for record in records:
        for cycle, weight in enumerate(record["outcome"].get("cycle_actions", [])):
            if cycle >= max_cycles:
                break
            buckets[cycle].append(float(weight))
    return np.asarray([float(np.mean(values)) if values else np.nan for values in buckets])


def _plot_trajectories(
    seed_results: Sequence[Dict[str, Any]],
    path: Path,
    *,
    window: int,
) -> None:
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), constrained_layout=True)
    for result in seed_results:
        adaptation = result["adaptation"]["records"]
        evaluation = result["evaluation"]["records"]["bandit_sarsa"]
        mean_action, explored = _episode_action_summary([*adaptation, *evaluation])
        local_window = min(int(window), int(mean_action.size))
        x = np.arange(local_window, mean_action.size + 1)
        label = str(result["forward_seed"])
        axes[0].plot(
            x,
            _rolling_mean(mean_action, local_window),
            label=label,
            linewidth=1.2,
        )
        axes[1].plot(
            x,
            _rolling_mean(explored, local_window),
            label=label,
            linewidth=1.2,
        )
    axes[0].axvline(1500, color="#202020", linestyle="--", linewidth=1.0)
    axes[1].axvline(1500, color="#202020", linestyle="--", linewidth=1.0)
    axes[0].set_title(f"SARSA mean relaxation weight, rolling {window}")
    axes[1].set_title(f"SARSA exploratory-action fraction, rolling {window}")
    axes[0].set_ylabel("Mean w")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(bottom=0.0)
    axes[1].set_xlabel("Forward-stream instance")
    axes[0].legend(title="Forward seed", ncol=5, frameon=False)

    pooled_sarsa = [
        record
        for result in seed_results
        for record in result["evaluation"]["records"]["bandit_sarsa"]
    ]
    pooled_ppo = [
        record
        for result in seed_results
        for record in result["evaluation"]["records"]["bandit_ppo"]
    ]
    sarsa_cycle = _by_cycle_mean(pooled_sarsa)
    ppo_cycle = _by_cycle_mean(pooled_ppo)
    axes[2].plot(sarsa_cycle, marker="o", markersize=3, color="#24866B", label="Online SARSA")
    axes[2].plot(ppo_cycle, marker="o", markersize=3, color="#755AA8", label="Frozen PPO")
    axes[2].set_title("Mean chosen weight by AMG cycle, 5,000 evaluation cases")
    axes[2].set_xlabel("AMG cycle")
    axes[2].set_ylabel("Mean w")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22, linewidth=0.7)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def generate_plots(result_path: Path, output_dir: Path, *, rolling_window: int) -> Dict[str, Any]:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    seed_results = _load_seed_results(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runtime_components": output_dir / "runtime_components_5000.png",
        "paired_forest": output_dir / "paired_improvement_forest.png",
        "rolling": output_dir / "rolling_100_native_solve.png",
        "controller_trajectories": output_dir / "controller_trajectories.png",
        "ppo_per_seed": output_dir / "ppo_per_seed_vs_bandit_default.png",
    }
    _plot_components(payload, paths["runtime_components"])
    _plot_forest(payload, paths["paired_forest"])
    _plot_rolling(seed_results, paths["rolling"], window=int(rolling_window))
    _plot_trajectories(seed_results, paths["controller_trajectories"], window=20)
    _plot_ppo_per_seed(seed_results, paths["ppo_per_seed"])
    summary = {
        "result": str(result_path),
        "rolling_window": int(rolling_window),
        "figures": {key: str(path) for key, path in paths.items()},
    }
    _write_json(output_dir / "visualization_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Active Exp44 PPO vs online SARSA results.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rolling-window", type=int, default=100)
    args = parser.parse_args()
    print(
        json.dumps(
            generate_plots(
                args.result,
                args.output_dir,
                rolling_window=int(args.rolling_window),
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
