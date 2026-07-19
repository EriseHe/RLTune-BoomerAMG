from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "default_setup_default_solve": "Default setup + solve",
    "mature_bandit_default_solve": "Bandit + default solve",
    "mature_bandit_fixed_w_1.60": "Bandit + fixed w=1.6",
    "mature_bandit_rl": "Bandit + PPO",
    "bandit_only": "Bandit + default solve",
    "fixed_w_1.60": "Bandit + fixed w=1.6",
    "ppo_best": "Bandit + PPO",
}
METHOD_COLORS = {
    "default_setup_default_solve": "#4D4D4D",
    "mature_bandit_default_solve": "#3C78A8",
    "mature_bandit_fixed_w_1.60": "#D18B28",
    "mature_bandit_rl": "#6F5AA8",
    "bandit_only": "#3C78A8",
    "fixed_w_1.60": "#D18B28",
    "ppo_best": "#6F5AA8",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _method_table(methods: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for method, values in methods.items():
        cases = int(values["cases"])
        native_total = float(values["mean_runtime"])
        setup = float(values.get("mean_setup_runtime", 0.0))
        solve = float(values.get("mean_solve_runtime", native_total - setup))
        controller = float(values.get("mean_infer_runtime", 0.0))
        inclusive = float(
            values.get("mean_runtime_with_controller", native_total + controller)
        )
        rows.append(
            {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "cases": cases,
                "native_setup_ms_per_instance": 1000.0 * setup,
                "native_solve_ms_per_instance": 1000.0 * solve,
                "native_total_ms_per_instance": 1000.0 * native_total,
                "controller_ms_per_instance": 1000.0 * controller,
                "controller_inclusive_ms_per_instance": 1000.0 * inclusive,
                "native_total_seconds_all_cases": float(
                    values.get("total_runtime", native_total * cases)
                ),
                "controller_inclusive_seconds_all_cases": float(
                    values.get("total_runtime_with_controller", inclusive * cases)
                ),
                "failures": int(values.get("failed_count", 0)),
            }
        )
    return rows


def _plot_runtime_breakdown(
    methods: Mapping[str, Mapping[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    names = list(methods)
    setup = np.asarray([float(methods[name].get("mean_setup_runtime", 0.0)) for name in names]) * 1000.0
    solve = np.asarray([float(methods[name].get("mean_solve_runtime", 0.0)) for name in names]) * 1000.0
    controller = np.asarray([float(methods[name].get("mean_infer_runtime", 0.0)) for name in names]) * 1000.0
    x = np.arange(len(names))
    colors = [METHOD_COLORS.get(name, "#777777") for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)

    axes[0].bar(x, setup, color=colors, alpha=0.48, label="Native setup")
    axes[0].bar(x, solve, bottom=setup, color=colors, label="Native solve")
    axes[0].set_title("Native runtime")
    axes[0].set_ylabel("Milliseconds per instance")

    native = setup + solve
    axes[1].bar(x, native, color=colors, label="Native total")
    axes[1].bar(
        x,
        controller,
        bottom=native,
        color="#F2F2F2",
        edgecolor=colors,
        hatch="///",
        label="Controller",
    )
    axes[1].set_title("Controller-inclusive runtime")
    figure.suptitle(title, fontsize=13)
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(
            [METHOD_LABELS.get(name, name) for name in names], rotation=24, ha="right"
        )
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_per_seed(
    per_seed: Mapping[str, Mapping[str, Any]],
    path: Path,
    *,
    title: str,
) -> None:
    seeds = sorted(per_seed, key=int)
    method_names = list(next(iter(per_seed.values()))["methods"])
    x = np.arange(len(seeds), dtype=float)
    width = 0.8 / max(1, len(method_names))
    figure, axis = plt.subplots(figsize=(13, 5.8), constrained_layout=True)
    for index, method in enumerate(method_names):
        values = [
            1000.0 * float(per_seed[seed]["methods"][method]["mean_runtime"])
            for seed in seeds
        ]
        positions = x - 0.4 + width / 2.0 + index * width
        axis.bar(
            positions,
            values,
            width=width,
            color=METHOD_COLORS.get(method, "#777777"),
            label=METHOD_LABELS.get(method, method),
        )
    axis.set_xticks(x)
    axis.set_xticklabels(seeds)
    axis.set_xlabel("Evaluation seed")
    axis.set_ylabel("Native total (ms per instance)")
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False, ncol=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_checkpoint_scores(payload: Mapping[str, Any], path: Path) -> None:
    history = payload["checkpoint_selection"].get("checkpoint_history", [])
    requested = [int(row["timesteps"]) for row in history]
    actual = [int(row.get("actual_timesteps", row["timesteps"])) for row in history]
    scores = [float(row["score"]) for row in history]
    figure, axis = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    axis.plot(actual, scores, marker="o", color="#6F5AA8", linewidth=1.5)
    for x, requested_x in zip(actual, requested):
        if x != requested_x:
            axis.annotate(f"requested {requested_x}", (x, scores[actual.index(x)]), xytext=(6, 8), textcoords="offset points")
    axis.set_xlabel("Actual environment transitions")
    axis.set_ylabel("Checkpoint selection score")
    axis.set_title("PPO checkpoint validation")
    axis.grid(alpha=0.22, linewidth=0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _evaluation_per_seed(payload: Mapping[str, Any], window: str) -> dict[str, Any]:
    return {
        str(row["seed"]): {"methods": row["methods"]}
        for row in payload["windows"][window]["rows"]
    }


def _rolling_mean(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    numerator = np.convolve(np.where(finite, values, 0.0), np.ones(window), mode="valid")
    denominator = np.convolve(finite.astype(float), np.ones(window), mode="valid")
    return np.arange(window, values.size + 1), numerator / np.maximum(1.0, denominator)


def _plot_ppo_trajectory(payload: Mapping[str, Any], window: str, path: Path) -> bool:
    rows = payload["windows"][window]["rows"]
    trajectories: list[tuple[str, list[float], list[float]]] = []
    for row in rows:
        source = _load(Path(row["source_path"]))
        ppo = source["windows"][window]["methods"].get("ppo_best")
        if not ppo or not ppo.get("per_case_mean_w"):
            continue
        trajectories.append(
            (
                str(row["seed"]),
                [float(value) for value in ppo["per_case_mean_w"]],
                [float(value) for value in ppo.get("mean_w_by_cycle", [])],
            )
        )
    if not trajectories:
        return False

    figure, axes = plt.subplots(2, 1, figsize=(12.5, 8), constrained_layout=True)
    for seed, per_case, by_cycle in trajectories:
        values = np.asarray(per_case, dtype=float)
        rolling_window = min(50, values.size)
        x, rolling = _rolling_mean(values, rolling_window)
        axes[0].plot(x, rolling, marker="o", markersize=2.5, linewidth=1.2, label=seed)
        axes[1].plot(np.arange(len(by_cycle)), by_cycle, marker="o", markersize=3, linewidth=1.2, label=seed)
    axes[0].set_title("Frozen PPO mean chosen weight by evaluation instance (rolling 50)")
    axes[0].set_xlabel("Instance within formal window")
    axes[0].set_ylabel("Mean w")
    axes[1].set_title("Frozen PPO mean chosen weight by AMG cycle")
    axes[1].set_xlabel("AMG cycle")
    axes[1].set_ylabel("Mean w")
    for axis in axes:
        axis.set_ylim(0.95, 2.05)
        axis.grid(alpha=0.22, linewidth=0.7)
    axes[0].legend(title="Seed", ncol=5, frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return True


def _plot_ppo_relative(methods: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    ppo = methods["ppo_best"]
    comparisons = []
    for reference_name in ("bandit_only", "fixed_w_1.60"):
        if reference_name not in methods:
            continue
        reference = methods[reference_name]
        for metric, label in (
            ("mean_solve_runtime", "Native solve"),
            ("mean_runtime", "Native total"),
            ("mean_runtime_with_controller", "Controller-inclusive"),
        ):
            value = 100.0 * (float(reference[metric]) - float(ppo[metric])) / float(reference[metric])
            comparisons.append((f"vs {METHOD_LABELS[reference_name]} | {label}", value))
    figure, axis = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    y = np.arange(len(comparisons))[::-1]
    values = [row[1] for row in comparisons]
    colors = ["#27816F" if value >= 0.0 else "#B6544D" for value in values]
    axis.barh(y, values, color=colors)
    axis.axvline(0.0, color="#222222", linewidth=1.0)
    axis.set_yticks(y)
    axis.set_yticklabels([row[0] for row in comparisons])
    axis.set_xlabel("PPO improvement (%)")
    axis.set_title("PPO relative runtime")
    axis.grid(axis="x", alpha=0.22, linewidth=0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def report_training(result_path: Path, output_dir: Path) -> dict[str, Any]:
    payload = _load(result_path)
    methods = payload["combined_eval"]["methods"]
    per_seed = payload["per_eval_seed"]
    figures = output_dir / "figures"
    table_path = output_dir / "validation_table.csv"
    runtime_path = figures / "validation_runtime_breakdown.png"
    seed_path = figures / "validation_by_seed.png"
    checkpoint_path = figures / "checkpoint_scores.png"
    _write_csv(table_path, _method_table(methods))
    _plot_runtime_breakdown(methods, runtime_path, title="Exp44 internal validation")
    _plot_per_seed(per_seed, seed_path, title="Exp44 internal validation by seed")
    _plot_checkpoint_scores(payload, checkpoint_path)
    return {
        "mode": "training",
        "result": str(result_path),
        "table": str(table_path),
        "figures": [str(runtime_path), str(seed_path), str(checkpoint_path)],
    }


def report_evaluation(
    result_path: Path,
    output_dir: Path,
    *,
    window: str,
) -> dict[str, Any]:
    payload = _load(result_path)
    if window not in payload["windows"]:
        raise KeyError(f"window {window!r} not found; available={list(payload['windows'])}")
    methods = payload["windows"][window]["combined"]
    figures = output_dir / "figures"
    table_path = output_dir / "main_table.csv"
    runtime_path = figures / "runtime_breakdown.png"
    seed_path = figures / "per_seed_native_total.png"
    relative_path = figures / "ppo_relative_runtime.png"
    trajectory_path = figures / "ppo_weight_trajectory.png"
    _write_csv(table_path, _method_table(methods))
    _plot_runtime_breakdown(
        methods,
        runtime_path,
        title=f"Exp44 held-out evaluation ({window})",
    )
    _plot_per_seed(
        _evaluation_per_seed(payload, window),
        seed_path,
        title=f"Exp44 held-out native total by seed ({window})",
    )
    _plot_ppo_relative(methods, relative_path)
    generated = [runtime_path, seed_path, relative_path]
    if _plot_ppo_trajectory(payload, window, trajectory_path):
        generated.append(trajectory_path)
    return {
        "mode": "evaluation",
        "window": window,
        "result": str(result_path),
        "table": str(table_path),
        "figures": [str(path) for path in generated],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Exp44 tables and figures.")
    parser.add_argument("mode", choices=("training", "evaluation"))
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--window", default="eval_1000")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "training":
        report = report_training(args.result, args.output_dir)
    else:
        report = report_evaluation(
            args.result, args.output_dir, window=str(args.window)
        )
    report_path = args.output_dir / "report_manifest.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "manifest": str(report_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
