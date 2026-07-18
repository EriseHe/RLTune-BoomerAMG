from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    "independent_sarsa",
    "shared_sarsa",
    "bootstrap_lcb_sarsa",
    "stagewise_lsvi_lcb",
)
DISPLAY_METHODS = ("fixed_w1.6",) + METHODS
LABELS = {
    "fixed_w1.6": "Fixed w=1.6",
    "independent_sarsa": "Independent SARSA",
    "shared_sarsa": "Shared SARSA",
    "bootstrap_lcb_sarsa": "Bootstrap-LCB SARSA",
    "stagewise_lsvi_lcb": "Stagewise LSVI-LCB",
}
COLORS = {
    "fixed_w1.6": "#2f2f2f",
    "independent_sarsa": "#6c757d",
    "shared_sarsa": "#00798c",
    "bootstrap_lcb_sarsa": "#d1495b",
    "stagewise_lsvi_lcb": "#edae49",
}

RUNTIME_FIELDS = {
    "setup_ms": "mean_setup_runtime_sec",
    "native_solve_ms": "mean_native_solve_runtime_sec",
    "native_total_ms": "mean_native_total_runtime_sec",
    "controller_ms": "mean_controller_runtime_sec",
    "end_to_end_ms": "mean_end_to_end_runtime_sec",
}


def _read_json_lines(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.22,
            "figure.dpi": 150,
        }
    )


def plot_improvement_forest(result: Dict[str, Any], output: Path) -> None:
    aggregate = result["aggregate"]
    methods = tuple(method for method in METHODS if method != "independent_sarsa")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    comparisons = (
        ("native_solve_vs_fixed", "Native solve vs fixed w=1.6"),
        ("native_solve_vs_independent_sarsa", "Native solve vs independent SARSA"),
    )
    y = np.arange(len(methods))
    for axis, (key, title) in zip(axes, comparisons):
        for row, method in enumerate(methods):
            metric = aggregate[method][key]
            point = float(metric["improvement_pct"])
            low, high = metric["hierarchical_bootstrap_95pct"]
            axis.errorbar(
                point,
                row,
                xerr=[[point - float(low)], [float(high) - point]],
                fmt="o",
                color=COLORS[method],
                capsize=3,
                linewidth=1.5,
            )
        axis.axvline(0.0, color="#242424", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Improvement (%)")
        axis.set_yticks(y, [LABELS[method] for method in methods])
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _late_action_shares(rows: Sequence[Dict[str, Any]], weights: np.ndarray) -> np.ndarray:
    start = int(0.75 * len(rows))
    actions = [
        float(action)
        for row in rows[start:]
        for action in row["outcome"].get("cycle_actions", [])
    ]
    if not actions:
        return np.zeros(weights.size, dtype=float)
    action_values = np.asarray(actions, dtype=float)
    counts = np.asarray(
        [np.sum(np.isclose(action_values, weight, atol=1.0e-12)) for weight in weights],
        dtype=float,
    )
    return counts / max(float(np.sum(counts)), 1.0)


def plot_late_action_distributions(
    result: Dict[str, Any],
    result_dir: Path,
    output: Path,
) -> None:
    weights = np.asarray(result["protocol"]["actions"], dtype=float)
    controller_seeds = result["protocol"]["controller_seeds"]
    methods = METHODS
    fig, axes = plt.subplots(
        len(controller_seeds),
        1,
        figsize=(10.2, 2.6 * len(controller_seeds)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    width = 0.18
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width
    for row_index, controller_seed in enumerate(controller_seeds):
        axis = axes[row_index, 0]
        training_dir = result_dir / f"controller_seed_{controller_seed}" / "training"
        for method_index, method in enumerate(methods):
            rows = _read_json_lines(training_dir / f"{method}.jsonl")
            shares = _late_action_shares(rows, weights)
            axis.bar(
                np.arange(weights.size) + offsets[method_index],
                shares,
                width=width,
                color=COLORS[method],
                label=LABELS[method],
            )
        axis.set_ylabel("Cycle share")
        axis.set_title(f"Controller seed {controller_seed}, final training quartile")
    axes[-1, 0].set_xticks(np.arange(weights.size), [f"{weight:.1f}" for weight in weights])
    axes[-1, 0].set_xlabel("Relaxation weight")
    axes[0, 0].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _action_matrix(rows: Sequence[Dict[str, Any]], max_cycles: int) -> np.ndarray:
    matrix = np.full((int(max_cycles), len(rows)), np.nan, dtype=float)
    for instance, row in enumerate(rows):
        actions = np.asarray(row["outcome"].get("cycle_actions", []), dtype=float)
        matrix[: min(actions.size, max_cycles), instance] = actions[:max_cycles]
    return matrix


def _load_training_action_rows(
    result: Dict[str, Any],
    result_dir: Path,
) -> tuple[Dict[tuple[int, str], list[Dict[str, Any]]], int]:
    rows_by_seed_method = {}
    max_cycles = 0
    for controller_seed in result["protocol"]["controller_seeds"]:
        training_dir = result_dir / f"controller_seed_{controller_seed}" / "training"
        for method in DISPLAY_METHODS:
            rows = _read_json_lines(training_dir / f"{method}.jsonl")
            rows_by_seed_method[(int(controller_seed), method)] = rows
            max_cycles = max(
                max_cycles,
                max(
                    (len(row["outcome"].get("cycle_actions", [])) for row in rows),
                    default=0,
                ),
            )
    return rows_by_seed_method, max_cycles


def plot_action_trajectory_grid(
    rows_by_seed_method: Dict[tuple[int, str], list[Dict[str, Any]]],
    controller_seeds: Sequence[int],
    max_cycles: int,
    output: Path,
    *,
    action_min: float,
    action_max: float,
    methods: Sequence[str] = DISPLAY_METHODS,
    method_labels: Dict[str, str] | None = None,
    column_labels: Dict[int, str] | None = None,
    x_label: str = "Training instance",
    title: str = "Per-instance, per-cycle action trajectories (white: solve already terminated)",
    panel_width: float = 5.4,
    panel_height: float = 2.2,
) -> None:
    seeds = tuple(int(seed) for seed in controller_seeds)
    plotted_methods = tuple(methods)
    labels = LABELS if method_labels is None else method_labels
    fig, axes = plt.subplots(
        len(plotted_methods),
        len(seeds),
        figsize=(float(panel_width) * len(seeds), float(panel_height) * len(plotted_methods)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    colormap = plt.colormaps["viridis"].copy()
    colormap.set_bad("white")
    image = None
    for row_index, method in enumerate(plotted_methods):
        for column_index, controller_seed in enumerate(seeds):
            axis = axes[row_index, column_index]
            rows = rows_by_seed_method[(controller_seed, method)]
            matrix = _action_matrix(rows, max_cycles)
            image = axis.imshow(
                matrix,
                aspect="auto",
                interpolation="nearest",
                origin="lower",
                vmin=float(action_min),
                vmax=float(action_max),
                cmap=colormap,
                extent=(0.5, len(rows) + 0.5, -0.5, max_cycles - 0.5),
            )
            axis.grid(False)
            if row_index == 0:
                axis.set_title(
                    f"Controller seed {controller_seed}"
                    if column_labels is None
                    else column_labels[int(controller_seed)]
                )
            if column_index == 0:
                axis.set_ylabel(f"{labels.get(method, method)}\nAMG cycle")
            if row_index == len(plotted_methods) - 1:
                axis.set_xlabel(x_label)
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Relaxation weight w",
            shrink=0.92,
        )
    fig.suptitle(title)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _load_validation_action_rows(
    result: Dict[str, Any],
    result_dir: Path,
    methods: Sequence[str],
) -> tuple[Dict[tuple[int, str], list[Dict[str, Any]]], int]:
    rows_by_seed_method = {}
    max_cycles = 0
    validation_seeds = result["trace_manifest"]["validation"]["seeds"]
    for controller_seed in result["protocol"]["controller_seeds"]:
        for method in methods:
            rows = []
            for validation_seed in validation_seeds:
                path = (
                    result_dir
                    / f"controller_seed_{controller_seed}"
                    / "validation"
                    / f"seed_{validation_seed}"
                    / f"{method}.jsonl"
                )
                rows.extend(_read_json_lines(path))
            rows_by_seed_method[(int(controller_seed), method)] = rows
            max_cycles = max(
                max_cycles,
                max(
                    (len(row["outcome"].get("cycle_actions", [])) for row in rows),
                    default=0,
                ),
            )
    return rows_by_seed_method, max_cycles


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    result = np.empty_like(values)
    for index in range(values.size):
        start = max(0, index + 1 - int(window))
        result[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return result


def plot_average_action_trajectories(
    rows_by_seed_method: Dict[tuple[int, str], list[Dict[str, Any]]],
    controller_seeds: Sequence[int],
    output: Path,
    *,
    window: int = 50,
    action_min: float,
    action_max: float,
) -> None:
    seeds = tuple(int(seed) for seed in controller_seeds)
    fig, axes = plt.subplots(
        len(seeds),
        1,
        figsize=(11.5, 3.0 * len(seeds)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    for row_index, controller_seed in enumerate(seeds):
        axis = axes[row_index, 0]
        for method in METHODS:
            rows = rows_by_seed_method[(controller_seed, method)]
            values = np.asarray(
                [
                    np.mean(np.asarray(row["outcome"].get("cycle_actions", []), dtype=float))
                    for row in rows
                ],
                dtype=float,
            )
            axis.plot(
                np.arange(1, values.size + 1),
                _trailing_mean(values, window),
                color=COLORS[method],
                linewidth=1.4,
                label=LABELS[method],
            )
        axis.axhline(1.6, color="#242424", linestyle="--", linewidth=1.0, label="Fixed w=1.6")
        axis.set_title(f"Controller seed {controller_seed}")
        axis.set_ylabel("Mean selected w")
        axis.set_ylim(float(action_min) - 0.05, float(action_max) + 0.05)
        axis.set_yticks(np.arange(float(action_min), float(action_max) + 0.01, 0.2))
    axes[-1, 0].set_xlabel("Training instance")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle(
        f"Mean selected weight per solve, trailing {window}-instance average"
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_uncertainty_coverage(result: Dict[str, Any], output: Path) -> None:
    methods = ("bootstrap_lcb_sarsa", "stagewise_lsvi_lcb")
    levels = ("50", "80", "95")
    target = np.asarray([0.5, 0.8, 0.95], dtype=float)
    fig, axis = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(levels))
    width = 0.32
    for method_index, method in enumerate(methods):
        per_seed = [
            seed_result["calibration"][method]["coverage"]
            for seed_result in result["controller_seed_results"]
            if seed_result["calibration"][method].get("samples", 0) > 0
        ]
        observed = np.asarray(
            [np.mean([float(seed[level]) for seed in per_seed]) for level in levels],
            dtype=float,
        )
        axis.bar(
            x + (method_index - 0.5) * width,
            observed,
            width=width,
            color=COLORS[method],
            label=LABELS[method],
        )
    axis.plot(x, target, color="#242424", marker="o", linewidth=1.2, label="Nominal")
    axis.set_xticks(x, [f"{level}% interval" for level in levels])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Empirical coverage")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def _pooled_runtime_table(result: Dict[str, Any]) -> list[Dict[str, Any]]:
    rows = []
    for method in DISPLAY_METHODS:
        summaries = [
            seed_result["validation"][method]
            for seed_result in result["controller_seed_results"]
        ]
        cases = sum(int(summary["cases"]) for summary in summaries)
        row: Dict[str, Any] = {
            "method": method,
            "label": LABELS[method],
            "cases": cases,
            "failures": sum(int(summary["failed_count"]) for summary in summaries),
        }
        for output_name, input_name in RUNTIME_FIELDS.items():
            weighted_seconds = sum(
                float(summary[input_name]) * int(summary["cases"])
                for summary in summaries
            ) / max(cases, 1)
            row[output_name] = 1000.0 * weighted_seconds
        rows.append(row)
    return rows


def _write_runtime_table(rows: Sequence[Dict[str, Any]], output: Path) -> None:
    fields = (
        "method",
        "label",
        "cases",
        "setup_ms",
        "native_solve_ms",
        "native_total_ms",
        "controller_ms",
        "end_to_end_ms",
        "failures",
    )
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_runtime_breakdown(rows: Sequence[Dict[str, Any]], output: Path) -> None:
    labels = [str(row["label"]) for row in rows]
    setup = np.asarray([float(row["setup_ms"]) for row in rows])
    solve = np.asarray([float(row["native_solve_ms"]) for row in rows])
    native = np.asarray([float(row["native_total_ms"]) for row in rows])
    controller = np.asarray([float(row["controller_ms"]) for row in rows])
    colors = [COLORS[str(row["method"])] for row in rows]
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    axes[0].bar(x, setup, color="#b8bec4", label="Native setup")
    axes[0].bar(x, solve, bottom=setup, color=colors, label="Native solve")
    axes[0].set_title("Native runtime")
    axes[0].set_ylabel("Milliseconds per instance")
    axes[0].legend(frameon=False)

    axes[1].bar(x, native, color=colors, label="Native total")
    axes[1].bar(
        x,
        controller,
        bottom=native,
        color="white",
        edgecolor=colors,
        hatch="///",
        label="Controller",
    )
    axes[1].set_title("End-to-end runtime")
    axes[1].legend(frameon=False)
    for index, value in enumerate(native + controller):
        axes[1].text(index, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    for axis in axes:
        axis.set_xticks(x, labels, rotation=22, ha="right")
        axis.set_ylim(0.0, 1.12 * float(np.max(native + controller)))
    fig.suptitle(
        "Frozen validation: 750 cases x 3 controller seeds (2,250 evaluations per method)"
    )
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def run(result_path: Path, output_dir: Path | None = None) -> Dict[str, str]:
    result_path = result_path.resolve()
    result_dir = result_path.parent
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    figures = result_dir / "figures" if output_dir is None else output_dir
    figures.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = {
        "runtime_breakdown": figures / "runtime_breakdown.png",
        "action_trajectories_all_seeds": figures / "action_trajectories_all_seeds.png",
        "focused_frozen_action_trajectories": figures / "focused_frozen_action_trajectories.png",
        "average_action_trajectory": figures / "average_action_trajectory.png",
        "improvement_forest": figures / "shared_action_improvement_forest.png",
        "late_action_distribution": figures / "late_action_distribution.png",
        "uncertainty_coverage": figures / "uncertainty_coverage.png",
    }
    runtime_rows = _pooled_runtime_table(result)
    runtime_table = result_dir / "main_table.csv"
    _write_runtime_table(runtime_rows, runtime_table)
    plot_runtime_breakdown(runtime_rows, outputs["runtime_breakdown"])
    action_rows, max_cycles = _load_training_action_rows(result, result_dir)
    actions = np.asarray(result["protocol"]["actions"], dtype=float)
    plot_action_trajectory_grid(
        action_rows,
        result["protocol"]["controller_seeds"],
        max_cycles,
        outputs["action_trajectories_all_seeds"],
        action_min=float(np.min(actions)),
        action_max=float(np.max(actions)),
    )
    focused_methods = ("fixed_w1.6", "bootstrap_lcb_sarsa", "stagewise_lsvi_lcb")
    validation_rows, validation_max_cycles = _load_validation_action_rows(
        result,
        result_dir,
        focused_methods,
    )
    plot_action_trajectory_grid(
        validation_rows,
        result["protocol"]["controller_seeds"],
        validation_max_cycles,
        outputs["focused_frozen_action_trajectories"],
        action_min=float(np.min(actions)),
        action_max=float(np.max(actions)),
        methods=focused_methods,
        x_label="Frozen validation instance (3 seeds x 250)",
        title=(
            "Frozen deterministic action trajectories "
            "(epsilon=0, no updates; white: solve already terminated)"
        ),
    )
    plot_average_action_trajectories(
        action_rows,
        result["protocol"]["controller_seeds"],
        outputs["average_action_trajectory"],
        action_min=float(np.min(actions)),
        action_max=float(np.max(actions)),
    )
    for controller_seed in result["protocol"]["controller_seeds"]:
        name = f"action_trajectory_seed_{controller_seed}"
        outputs[name] = figures / f"{name}.png"
        plot_action_trajectory_grid(
            action_rows,
            (int(controller_seed),),
            max_cycles,
            outputs[name],
            action_min=float(np.min(actions)),
            action_max=float(np.max(actions)),
        )
    plot_improvement_forest(result, outputs["improvement_forest"])
    plot_late_action_distributions(
        result,
        result_dir,
        outputs["late_action_distribution"],
    )
    plot_uncertainty_coverage(result, outputs["uncertainty_coverage"])
    rendered = {name: str(path) for name, path in outputs.items()}
    rendered["main_table"] = str(runtime_table)
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the shared-action RL screen.")
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.result, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
