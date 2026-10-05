from __future__ import annotations

"Reusable plotting implementation for completed joint AMG experiments."
import csv
import json
from pathlib import Path
from typing import Any, Dict, Sequence
import matplotlib.pyplot as plt
import numpy as np
from .action_trajectory_plot import plot_action_trajectory_grid
from experiments.paper_final.online.io import _write_json
from experiments.paper_final.online.comparison import method_comparison
from experiments.paper_final.online.method_spec import CONTEXT_DISPLAY_LABELS
from experiments.paper_final.online.reporting import (
    _action_summary,
    _comparison_windows,
    _window_result,
    _write_solve_screen_report,
    _write_summary_csv,
)

FAMILY_ORDER = ("default_setup", "default", "recursive_lstdq_v3_lcb")
FAMILY_LABELS = {
    "default_setup": "Default",
    "default": "LinUCB",
    "recursive_lstdq_v3_lcb": "LinUCB–LSTDQ",
}


def _method_label(
    method: str, family_by_method: Dict[str, str], method_labels: Dict[str, str]
) -> str:
    if method in method_labels:
        return str(method_labels[method]).replace(" [recommended; structured-512]", "")
    family = family_by_method[method]
    return FAMILY_LABELS.get(family, family.replace("_", " "))


def _uniform_value(values: Sequence[Any]) -> Any | None:
    unique = list(dict.fromkeys(values))
    return unique[0] if len(unique) == 1 else None


def _solver_setting(protocol: Dict[str, Any], name: str) -> Any | None:
    solver = protocol.get("solve", {})
    if name in solver:
        return solver[name]
    encoder_key = {"tolerance": "tol", "max_cycles": "max_cycles"}.get(name, name)
    values = [
        metadata.get("state_encoder", {}).get(encoder_key)
        for metadata in protocol.get("solve_controllers", {}).values()
        if metadata.get("state_encoder", {}).get(encoder_key) is not None
    ]
    return _uniform_value(values) if values else None


def _compact_setup_label(spec: Dict[str, Any]) -> str:
    if spec["setup_kind"] == "default":
        return "Default setup"
    context = str(spec.get("setup_context", "default"))
    return f"LinUCB ({CONTEXT_DISPLAY_LABELS.get(context, context)})"


def _compact_method_labels(protocol: Dict[str, Any]) -> Dict[str, str]:
    """Build short labels while retaining only per-method deviations."""
    specs = {str(spec["name"]): dict(spec) for spec in protocol.get("method_specs", [])}
    learned = [s for s in specs.values() if s.get("setup_kind") == "linucb"]
    if (
        len(specs) == 3
        and len(learned) == 2
        and (
            {s.get("solve_kind") for s in learned} == {"default", "recursive_lstdq_v3"}
        )
        and (
            len(
                {
                    tuple(
                        (
                            s.get(k)
                            for k in (
                                "setup_context",
                                "setup_space",
                                "candidate_sampling",
                                "seed_offset",
                                "setup_warmup_cases",
                            )
                        )
                    )
                    for s in learned
                }
            )
            == 1
        )
        and all((not s.get("solve_activation") for s in learned))
        and all(
            (
                s.get("solve_kind") == "default"
                or s.get("solve_context") == s.get("setup_context")
                for s in learned
            )
        )
        and any(
            (
                s.get("setup_kind") == s.get("solve_kind") == "default"
                for s in specs.values()
            )
        )
    ):
        return {
            name: "Default"
            if s["setup_kind"] == "default"
            else "LinUCB"
            if s["solve_kind"] == "default"
            else "LinUCB–LSTDQ"
            for (name, s) in specs.items()
        }
    raise ValueError("Plots require the official three-method roster")


def _shared_plot_settings(protocol: Dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return compact code-name/value pairs for a shared-settings panel."""
    stream = dict(protocol.get("stream", {}))
    settings: list[tuple[str, str]] = []
    problem = stream.get("problem")
    if problem is not None:
        settings.append(("problem", str(problem)))
    grid = stream.get("grid")
    if grid:
        settings.append(("grid", " x ".join((str(int(value)) for value in grid))))
    online = protocol.get("stream_partition", {}).get("online")
    if online and len(online) == 2:
        settings.append(("online_cases", f"{int(online[1]) - int(online[0]):,}"))
    tolerance = _solver_setting(protocol, "tolerance")
    if tolerance is not None:
        settings.append(("solve_tolerance", f"{float(tolerance):g}"))
    max_cycles = _solver_setting(protocol, "max_cycles")
    if max_cycles is not None:
        settings.append(("max_cycles", str(int(max_cycles))))
    specs = [dict(spec) for spec in protocol.get("method_specs", [])]
    solve_specs = [
        spec for spec in specs if str(spec.get("solve_kind", "default")) != "default"
    ]
    if solve_specs:
        activation = (
            None
            if any((spec.get("solve_activation") for spec in solve_specs))
            else _uniform_value(
                [int(spec.get("solve_activation_case", 0)) for spec in solve_specs]
            )
        )
        settings.append(
            (
                "solve_activation_case",
                f"{activation} (RL only)" if activation is not None else "varies",
            )
        )
    setup_specs = [
        spec for spec in specs if str(spec.get("setup_kind", "default")) != "default"
    ]
    setup_label = _uniform_value([_compact_setup_label(spec) for spec in setup_specs])
    if setup_label is not None:
        settings.append(("setup_bandit", str(setup_label)))
    setup_space = _uniform_value([spec.get("setup_space") for spec in setup_specs])
    if setup_space is not None:
        settings.append(("setup_space", str(setup_space)))
    candidate_sampling = _uniform_value(
        [spec.get("candidate_sampling") for spec in setup_specs]
    )
    if candidate_sampling is not None:
        settings.append(("candidate_sampling", str(candidate_sampling)))
    return tuple(settings)


def _axes_with_settings_panel(
    *,
    protocol: Dict[str, Any],
    plot_columns: int,
    figsize: tuple[float, float],
    sharey: bool = False,
) -> tuple[Any, tuple[Any, ...]]:
    figure = plt.figure(figsize=figsize, constrained_layout=True)
    grid = figure.add_gridspec(
        1, plot_columns + 1, width_ratios=(*[1.0] * plot_columns, 0.38)
    )
    axes = []
    for index in range(plot_columns):
        axes.append(
            figure.add_subplot(
                grid[0, index], sharey=axes[0] if sharey and axes else None
            )
        )
    panel = figure.add_subplot(grid[0, -1])
    panel.set_axis_off()
    panel.set_title("Shared settings", loc="left", fontweight="bold")
    text = "\n".join(
        (f"{name}:\n  {value}" for (name, value) in _shared_plot_settings(protocol))
    )
    panel.text(
        0.0,
        0.97,
        text,
        transform=panel.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        family="monospace",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#f7f7f7",
            "edgecolor": "#cccccc",
        },
    )
    return (figure, tuple(axes))


def _active_families(family_by_method: Dict[str, str]) -> tuple[str, ...]:
    present = set(family_by_method.values())
    ordered = [family for family in FAMILY_ORDER if family in present]
    ordered.extend(
        (
            family
            for family in dict.fromkeys(family_by_method.values())
            if family not in FAMILY_ORDER
        )
    )
    return tuple(ordered)


def _read_json_lines(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rolling(values: Sequence[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if window <= 0 or window > array.size:
        raise ValueError("rolling window must fit within the trajectory")
    return np.convolve(array, np.ones(window) / float(window), mode="valid")


def _trailing_mean(values: Sequence[float], window: int) -> np.ndarray:
    """Average recorded values in each window, preserving all-missing windows."""
    array = np.asarray(values, dtype=float)
    if window <= 0:
        raise ValueError("rolling window must be positive")
    finite = np.isfinite(array)
    cumulative = np.cumsum(np.insert(np.where(finite, array, 0.0), 0, 0.0))
    counts = np.cumsum(np.insert(finite.astype(int), 0, 0))
    indices = np.arange(array.size)
    starts = np.maximum(indices + 1 - window, 0)
    totals = cumulative[indices + 1] - cumulative[starts]
    available = counts[indices + 1] - counts[starts]
    return np.divide(
        totals, available, out=np.full(array.shape, np.nan), where=available > 0
    )


def _candidate_label(candidate: Dict[str, Any]) -> str:
    return f"$\\alpha={float(candidate['alpha']):g}$, $\\lambda={float(candidate['trace_lambda']):g}$"


def _load_records(
    result_dir: Path, methods: Sequence[str]
) -> Dict[str, list[Dict[str, Any]]]:
    return {
        method: _read_json_lines(result_dir / "trajectories" / f"{method}.jsonl")
        for method in methods
    }


def _same_setup_audit(
    records: Dict[str, list[Dict[str, Any]]],
    methods: Sequence[str],
    expected_cases: int,
) -> Dict[str, Any]:
    if "bandit_fixed_w1.6" not in records:
        return {}
    audit_windows = {
        "all": (0, expected_cases),
        "first_half": (0, expected_cases // 2),
        "last_half": (expected_cases // 2, expected_cases),
        "last_500": (max(0, expected_cases - 500), expected_cases),
    }
    fixed_rows = records["bandit_fixed_w1.6"]
    audit: Dict[str, Any] = {}
    for method_index, method in enumerate(methods):
        if method in {"bandit_default", "bandit_fixed_w1.6"}:
            continue
        audit[method] = {}
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
            audit[method][window_name] = {
                "matching_cases": int(len(indices)),
                "comparison": method_comparison(
                    candidate,
                    baseline,
                    seed=int(8217001 + method_index * 1009 + window_index),
                )
                if indices
                else None,
            }
    return audit


def _summary_reference_method(
    summary_rows: Sequence[Dict[str, str]], methods: Sequence[str]
) -> str:
    available = {row["method"] for row in summary_rows}
    return next(
        (
            method
            for method in (
                "bandit_default",
                "default_setup",
                "bandit_fixed_w1.6",
                *methods,
            )
            if method in available
        )
    )


def _metric(rows: Sequence[Dict[str, Any]], field: str) -> np.ndarray:
    return np.asarray(
        [float(row["outcome"].get(field, 0.0)) for row in rows], dtype=float
    )


def _selected_weights(
    rows: Sequence[Dict[str, Any]], *, family: str, statistic: str
) -> np.ndarray:
    if family in {"default_setup", "default"}:
        return np.ones(len(rows), dtype=float)
    if family == "fixed_w1.6":
        return np.full(len(rows), 1.6, dtype=float)
    if statistic not in {"first_cycle", "solve_mean"}:
        raise ValueError(f"Unsupported weight statistic: {statistic}")
    selected = []
    for row_index, row in enumerate(rows):
        outcome = row["outcome"]
        actions = outcome.get("cycle_actions")
        if not actions:
            final_weight = outcome.get("final_w")
            if final_weight is not None and np.isfinite(float(final_weight)):
                selected.append(float(final_weight))
                continue
            if bool(outcome.get("unrecovered_failure", outcome.get("failed", False))):
                selected.append(float("nan"))
                continue
            native_solve_completed = (
                outcome.get("native_status") is not None
                and int(outcome.get("iterations", 0)) > 0
            )
            if not native_solve_completed:
                raise ValueError(
                    f"Missing cycle_actions and final_w for {family} record {row_index}"
                )
            selected.append(1.0)
            continue
        action_values = np.asarray(actions, dtype=float)
        selected.append(
            float(action_values[0])
            if statistic == "first_cycle"
            else float(np.mean(action_values))
        )
    return np.asarray(selected, dtype=float)


def _plot_weight_trajectory(
    *,
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
    method_labels: Dict[str, str],
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
            records[method], family=family_by_method[method], statistic=statistic
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
    (figure, axes_grid) = plt.subplots(
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
            method
            for (method, method_family) in family_by_method.items()
            if method_family == family
        ]
        for method_index, method in enumerate(family_methods):
            values = _selected_weights(
                records[method], family=family, statistic=statistic
            )
            trend = _trailing_mean(values, window)
            if method in candidate_by_method:
                label = _candidate_label(candidate_by_method[method])
            else:
                label = _method_label(method, family_by_method, method_labels)
            color = palette(method_index % 10)
            axis.plot(
                np.arange(1, values.size + 1),
                trend,
                linewidth=1.55,
                color=color,
                label=label,
            )
            axis.scatter([1], [values[0]], s=24, color=color, zorder=3)
        axis.axhline(1.0, color="0.55", linewidth=0.75, linestyle=":")
        axis.axhline(1.6, color="0.45", linewidth=0.75, linestyle="--")
        axis.axvline(midpoint, color="0.35", linewidth=0.8, linestyle="--")
        axis.set_title(FAMILY_LABELS[family], loc="left", fontweight="bold")
        axis.set_ylabel("Selected w")
        axis.set_ylim(y_lower, y_upper)
        axis.set_yticks(y_ticks)
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.legend(
            loc="upper right", ncol=3 if len(family_methods) > 3 else 1, frameon=False
        )
    figure.suptitle(
        f"Online weight selection: {statistic_label}\nTrailing mean over recorded values in up to {window} instances; dots show the exact first instance",
        fontweight="bold",
    )
    axes[-1].set_xlabel(
        "Persistent online comparison instance (vertical line: stream midpoint, no reset)"
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_cumulative_components(
    *,
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
    method_labels: Dict[str, str],
) -> None:
    fields = (
        ("setup_runtime", "Cumulative setup (s)"),
        ("solve_runtime", "Cumulative native solve (s)"),
        ("end_to_end_runtime", "Cumulative end-to-end (s)"),
    )
    (figure, axes) = plt.subplots(
        len(fields), 1, figsize=(14, 11), sharex=True, constrained_layout=True
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
                label=_method_label(method, family_by_method, method_labels),
            )
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22, linewidth=0.7)
            axis.axvline(midpoint, color="0.35", linewidth=0.8, linestyle="--")
    axes[0].legend(ncol=2, frameon=False)
    axes[-1].set_xlabel(
        "Persistent online comparison instance (vertical line: stream midpoint, no reset)"
    )
    figure.suptitle("Cumulative runtime by measured component", fontweight="bold")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_first_last_components(*, path: Path, result: Dict[str, Any]) -> None:
    components = (
        ("setup_runtime", "Setup"),
        ("native_solve_runtime", "Native solve"),
        ("controller_runtime", "Controller"),
        ("setup_bandit_overhead", "Bandit overhead"),
    )
    windows = (("first_1000", "First 1000"), ("last_1000", "Last 1000"))
    methods = tuple(result["protocol"]["methods"])
    family_by_method = result["protocol"]["families"]
    method_labels = result["protocol"].get("method_labels", {})
    labels = [
        _method_label(method, family_by_method, method_labels) for method in methods
    ]
    colors = plt.get_cmap("Set2")(np.linspace(0.05, 0.95, len(components)))
    (figure, axes) = _axes_with_settings_panel(
        protocol=result["protocol"], plot_columns=2, figsize=(18.5, 7), sharey=True
    )
    for axis, (window_key, title) in zip(axes, windows):
        bottoms = np.zeros(len(methods), dtype=float)
        for (component, component_label), color in zip(components, colors):
            values = np.asarray(
                [
                    1000.0
                    * float(
                        result["windows"][window_key]["methods"][method]["means_sec"][
                            component
                        ]
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
        "First vs last 1000: measured runtime components", fontweight="bold"
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
    if window_key is None:
        window_key = next(
            (key for key in result["windows"] if str(key).startswith("all_"))
        )
    window = result["windows"][window_key]["methods"]
    methods = _methods_by_native_runtime(result["protocol"]["methods"], window)
    family_by_method = result["protocol"]["families"]
    method_labels = result["protocol"].get("method_labels", {})
    labels = [
        _method_label(method, family_by_method, method_labels) for method in methods
    ]
    case_count = int(window[next(iter(methods))]["cases"])
    components = (
        ("setup_runtime", "Setup", "#66c2a5"),
        ("native_solve_runtime", "Native solve", "#8da0cb"),
        ("controller_runtime", "Controller", "#ffd92f"),
        ("setup_bandit_overhead", "Bandit overhead", "#b3b3b3"),
    )
    (figure, axes) = _axes_with_settings_panel(
        protocol=result["protocol"], plot_columns=2, figsize=(18.5, 7)
    )
    for axis, included, title in (
        (axes[0], components[:2], "Native runtime"),
        (axes[1], components, "End-to-end runtime"),
    ):
        bottoms = np.zeros(len(methods), dtype=float)
        for component, component_label, color in included:
            values = np.asarray(
                [
                    1000.0 * float(window[method]["means_sec"][component])
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
                index, total, f"{total:.1f}", ha="center", va="bottom", fontsize=9
            )
        if bottoms.size and float(np.max(bottoms)) > 0.0:
            axis.set_ylim(0.0, 1.22 * float(np.max(bottoms)))
        axis.set_title(title, fontweight="bold")
        axis.set_xticks(np.arange(len(methods)), labels, rotation=24, ha="right")
        axis.set_ylabel("Mean runtime (ms/case)")
        axis.grid(axis="y", alpha=0.22, linewidth=0.7)
        axis.legend(frameon=False, loc="upper center", ncol=2)
    figure.suptitle(
        figure_title or f"All {case_count:,} online comparison instances",
        fontweight="bold",
    )
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _methods_by_native_runtime(
    methods: Sequence[str], window: Dict[str, Any]
) -> tuple[str, ...]:
    """Order methods by setup plus native solve runtime, slowest first."""
    return tuple(
        sorted(
            methods,
            key=lambda method: float(window[method]["means_sec"]["setup_runtime"])
            + float(window[method]["means_sec"]["native_solve_runtime"]),
            reverse=True,
        )
    )


def _plot_recovery_outcomes(*, path: Path, result: Dict[str, Any]) -> None:
    methods = tuple(result["protocol"]["methods"])
    family_by_method = result["protocol"]["families"]
    method_labels = result["protocol"].get("method_labels", {})
    labels = [
        _method_label(method, family_by_method, method_labels) for method in methods
    ]
    all_window_key = next(
        (key for key in result["windows"] if str(key).startswith("all_"))
    )
    summaries = result["windows"][all_window_key]["methods"]
    processed = np.asarray(
        [float(summaries[method]["cases"]) for method in methods], dtype=float
    )
    recovered = np.asarray(
        [float(summaries[method]["recovered_failure_count"]) for method in methods],
        dtype=float,
    )
    unrecovered = np.asarray(
        [float(summaries[method]["unrecovered_failure_count"]) for method in methods],
        dtype=float,
    )
    primary_success = processed - recovered - unrecovered
    scale = np.divide(
        100.0, processed, out=np.zeros_like(processed), where=processed > 0.0
    )
    components = (
        (primary_success * scale, "Primary success", "#66c2a5"),
        (recovered * scale, "Recovered by default fallback", "#ffd92f"),
        (unrecovered * scale, "Unrecovered", "#d95f02"),
    )
    (figure, axes) = _axes_with_settings_panel(
        protocol=result["protocol"], plot_columns=1, figsize=(15.5, 6)
    )
    axis = axes[0]
    bottoms = np.zeros(len(methods), dtype=float)
    for values, label, color in components:
        axis.bar(
            np.arange(len(methods)), values, bottom=bottoms, color=color, label=label
        )
        bottoms += values
    for index, (fallback_count, failure_count) in enumerate(
        zip(recovered, unrecovered)
    ):
        axis.text(
            index,
            101.0,
            f"recovered {int(fallback_count):,}\nunrecovered {int(failure_count):,}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    axis.set_ylim(0.0, 120.0)
    axis.set_yticks(np.arange(0.0, 101.0, 20.0))
    axis.set_ylabel("Share of external instances (%)")
    axis.set_xticks(np.arange(len(methods)), labels, rotation=24, ha="right")
    axis.set_title("Primary attempt and default-fallback outcomes", fontweight="bold")
    axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    axis.legend(frameon=False, loc="upper center", ncol=3)
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_first_last_improvement_ci(
    *,
    path: Path,
    result: Dict[str, Any],
    comparison_key: str,
    reference_method: str,
    reference_label: str,
) -> None:
    family_by_method = result["protocol"]["families"]
    method_labels = result["protocol"].get("method_labels", {})
    methods = tuple(
        (
            method
            for method in result["protocol"]["methods"]
            if method != reference_method
            and method in result["windows"]["first_1000"]["comparisons"][comparison_key]
            and (
                method in result["windows"]["last_1000"]["comparisons"][comparison_key]
            )
        )
    )
    windows = (("first_1000", "First 1000"), ("last_1000", "Last 1000"))
    metrics = (
        ("native_solve_runtime", "Native solve"),
        ("end_to_end_runtime", "End-to-end"),
    )
    labels = [
        _method_label(method, family_by_method, method_labels) for method in methods
    ]
    (figure, axes) = _axes_with_settings_panel(
        protocol=result["protocol"], plot_columns=2, figsize=(17.5, 6)
    )
    offsets = np.linspace(-0.16, 0.16, len(windows))
    for axis, (metric, title) in zip(axes, metrics):
        for offset, (window_key, window_label) in zip(offsets, windows):
            comparisons = result["windows"][window_key]["comparisons"][comparison_key]
            means = []
            lower_errors = []
            upper_errors = []
            for method in methods:
                entry = comparisons[method][metric]
                mean = float(entry["candidate_improvement_pct"])
                (lower, upper) = map(float, entry["candidate_improvement_95pct"])
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
        axis.set_ylabel(f"Improvement vs {reference_label} (%)")
    axes[1].legend(frameon=False)
    figure.suptitle("Paired bootstrap 95% confidence intervals", fontweight="bold")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _plot_runtime_trajectory(
    *,
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
    family_by_method: Dict[str, str],
    method_labels: Dict[str, str],
    candidate_by_method: Dict[str, Dict[str, Any]],
    field: str,
    ylabel: str,
    window: int,
) -> None:
    families = _active_families(family_by_method)
    (figure, axes_grid) = plt.subplots(
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
            method
            for (method, method_family) in family_by_method.items()
            if method_family == family
        ]
        for method_index, method in enumerate(family_methods):
            values = 1000.0 * _metric(records[method], field)
            rolling = _rolling(values, window)
            all_rolling.append(rolling)
            if method in candidate_by_method:
                label = _candidate_label(candidate_by_method[method])
            else:
                label = _method_label(method, family_by_method, method_labels)
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
        axis.legend(
            loc="upper right", ncol=3 if len(family_methods) > 3 else 1, frameon=False
        )
    finite = np.concatenate([values[np.isfinite(values)] for values in all_rolling])
    (lower, upper) = (float(np.min(finite)), float(np.max(finite)))
    margin = max(0.05 * (upper - lower), 0.1)
    for axis in axes:
        axis.set_ylim(lower - margin, upper + margin)
    axes[-1].set_xlabel("Persistent online comparison instance")
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)


def _write_markdown_table(path: Path, rows: Sequence[Dict[str, str]]) -> None:
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
        "|" + "|".join(("---" for _ in headers)) + "|",
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
    *, result_dir: Path, rolling_window: int = 100, analysis_stop: int | None = None
) -> Dict[str, Any]:
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    protocol = result["protocol"]
    protocol["method_labels"] = _compact_method_labels(protocol)
    methods = tuple(protocol["methods"])
    family_by_method = dict(protocol["families"])
    method_labels = dict(protocol.get("method_labels", {}))
    for method, family in family_by_method.items():
        FAMILY_LABELS.setdefault(
            family, method_labels.get(method, family.replace("_", " "))
        )
    candidate_by_method = {
        candidate["name"]: candidate for candidate in protocol["candidates"]
    }
    records = _load_records(result_dir, methods)
    lengths = {method: len(rows) for (method, rows) in records.items()}
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
                f"analysis_stop must be in [1, {source_cases}], got {expected_cases}"
            )
        if expected_cases < int(rolling_window):
            raise ValueError("analysis_stop must be at least rolling_window")
        records = {method: rows[:expected_cases] for (method, rows) in records.items()}
        artifact_dir = result_dir / f"analysis_first_{expected_cases}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        analysis_scope = {
            "source_result": str(result_dir / "result.json"),
            "source_stream_sha256": protocol["stream"]["sha256"],
            "included_online_indices": [0, expected_cases],
            "excluded_online_indices": [expected_cases, source_cases],
        }
        _write_json(artifact_dir / "analysis_scope.json", analysis_scope)
    windows = _comparison_windows(expected_cases)
    result = dict(result)
    result["windows"] = {
        name: _window_result(
            {method: rows[start:stop] for (method, rows) in records.items()},
            seed=int(protocol["method_order_seed"] + start + stop),
        )
        for (name, (start, stop)) in windows.items()
    }
    window_summary_path = artifact_dir / "window_summary.json"
    _write_json(
        window_summary_path,
        {
            "source_result": str(result_dir / "result.json"),
            "protocol": protocol,
            "windows": result["windows"],
        },
    )
    window_actions = {
        name: {
            method: _action_summary(records[method][start:stop])
            for (method, family) in family_by_method.items()
            if family not in {"default_setup", "default"}
        }
        for (name, (start, stop)) in windows.items()
    }
    _write_solve_screen_report(
        artifact_dir / "screen_report.md",
        window_results=result["windows"],
        window_actions=window_actions,
    )
    if analysis_stop is not None:
        _write_json(
            artifact_dir / "analysis_result.json",
            {
                "analysis_scope": analysis_scope,
                "protocol": protocol,
                "windows": result["windows"],
            },
        )
    same_setup_audit = _same_setup_audit(records, methods, expected_cases)
    same_setup_path = artifact_dir / "same_setup_audit.json"
    _write_json(same_setup_path, same_setup_audit)
    figures_dir = artifact_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    solve_path = figures_dir / "online_native_solve_trajectory.png"
    setup_path = figures_dir / "online_setup_trajectory.png"
    first_weight_path = figures_dir / "online_first_cycle_weight_trajectory.png"
    mean_weight_path = figures_dir / "online_mean_weight_trajectory.png"
    action_trajectory_path = figures_dir / "learned_per_cycle_action_trajectories.png"
    cumulative_path = figures_dir / "online_cumulative_components.png"
    all_runtime_path = figures_dir / f"all_{expected_cases}_runtime_breakdown.png"
    last_1000_runtime_path = figures_dir / "last_1000_runtime_breakdown.png"
    recovery_path = figures_dir / "primary_and_fallback_outcomes.png"
    first_last_components_path = figures_dir / "first_vs_last_1000_components.png"
    first_last_ci_path = figures_dir / "first_vs_last_1000_improvement_ci.png"
    _plot_runtime_trajectory(
        path=solve_path,
        records=records,
        family_by_method=family_by_method,
        method_labels=method_labels,
        candidate_by_method=candidate_by_method,
        field="solve_runtime",
        ylabel=f"Rolling {rolling_window} native solve (ms)",
        window=int(rolling_window),
    )
    _plot_runtime_trajectory(
        path=setup_path,
        records=records,
        family_by_method=family_by_method,
        method_labels=method_labels,
        candidate_by_method=candidate_by_method,
        field="setup_runtime",
        ylabel=f"Rolling {rolling_window} setup (ms)",
        window=int(rolling_window),
    )
    _plot_weight_trajectory(
        path=first_weight_path,
        records=records,
        family_by_method=family_by_method,
        method_labels=method_labels,
        candidate_by_method=candidate_by_method,
        statistic="first_cycle",
        window=int(rolling_window),
    )
    _plot_weight_trajectory(
        path=mean_weight_path,
        records=records,
        family_by_method=family_by_method,
        method_labels=method_labels,
        candidate_by_method=candidate_by_method,
        statistic="solve_mean",
        window=int(rolling_window),
    )
    learned_action_methods = [
        method
        for (method, family) in family_by_method.items()
        if family not in {"default_setup", "default"}
    ]
    if learned_action_methods:
        activation_cases = {
            str(spec["name"]): int(spec.get("solve_activation_case", 0))
            for spec in result["protocol"].get("method_specs", [])
            if int(spec.get("solve_activation_case", 0)) > 0
        }
        max_cycles = max(
            (
                len(row["outcome"].get("cycle_actions", []))
                for method in learned_action_methods
                for row in records[method]
            )
        )
        action_min = min(
            (
                min(row["outcome"].get("cycle_actions", [np.inf]))
                for method in learned_action_methods
                for row in records[method]
            )
        )
        action_max = max(
            (
                max(row["outcome"].get("cycle_actions", [-np.inf]))
                for method in learned_action_methods
                for row in records[method]
            )
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
                method: _method_label(method, family_by_method, method_labels).split(
                    "; activate=", 1
                )[0]
                for method in learned_action_methods
            },
            column_labels={
                0: f"Persistent {expected_cases:,}-instance online comparison"
            },
            x_label="Online comparison instance",
            title="Per-instance, per-cycle relaxation weights (white: no recorded cycle / terminated)",
            panel_width=10.5,
            panel_height=3.2,
            native_default_weight=1.0,
            activation_cases=activation_cases,
        )
    _plot_cumulative_components(
        path=cumulative_path,
        records=records,
        family_by_method=family_by_method,
        method_labels=method_labels,
    )
    _plot_all_runtime_breakdown(path=all_runtime_path, result=result)
    if "last_1000" in result["windows"]:
        _plot_all_runtime_breakdown(
            path=last_1000_runtime_path,
            result=result,
            window_key="last_1000",
            figure_title=f"Last 1,000 online comparison instances ({expected_cases - 999:,}-{expected_cases:,})",
        )
    _plot_recovery_outcomes(path=recovery_path, result=result)
    has_split_windows = {"first_1000", "last_1000"}.issubset(result["windows"])
    if has_split_windows:
        _plot_first_last_components(path=first_last_components_path, result=result)
    comparison_options = (
        ("vs_default_setup_default_solve", "default_setup_default_solve", "Default"),
    )
    improvement_reference = next(
        (
            option
            for option in comparison_options
            if has_split_windows
            and all(
                (
                    option[0] in result["windows"][window]["comparisons"]
                    for window in ("first_1000", "last_1000")
                )
            )
        ),
        None,
    )
    if improvement_reference is not None:
        (comparison_key, reference_method, reference_label) = improvement_reference
        _plot_first_last_improvement_ci(
            path=first_last_ci_path,
            result=result,
            comparison_key=comparison_key,
            reference_method=reference_method,
            reference_label=reference_label,
        )
    summary_csv_path = artifact_dir / f"summary_{expected_cases}.csv"
    _write_summary_csv(summary_csv_path, records, family_by_method)
    summary_rows = list(csv.DictReader(summary_csv_path.open(encoding="utf-8")))
    reference_method = _summary_reference_method(summary_rows, methods)
    reference_row = next(
        (row for row in summary_rows if row["method"] == reference_method)
    )
    reference_solve = float(reference_row["mean_native_solve_runtime_sec"])
    table_rows = []
    for row in summary_rows:
        method = row["method"]
        solve = float(row["mean_native_solve_runtime_sec"])
        if method in candidate_by_method:
            display = f"{family_by_method[method]} a={float(candidate_by_method[method]['alpha']):g}, lambda={float(candidate_by_method[method]['trace_lambda']):g}"
        else:
            display = _method_label(method, family_by_method, method_labels)
        table_rows.append(
            {
                "method": display,
                "setup_ms": f"{1000.0 * float(row['mean_setup_runtime_sec']):.3f}",
                "solve_ms": f"{1000.0 * solve:.3f}",
                "native_total_ms": f"{1000.0 * float(row['mean_native_total_runtime_sec']):.3f}",
                "solve_vs_default": f"{100.0 * (reference_solve - solve) / reference_solve:+.2f}%",
                "controller_ms": f"{1000.0 * float(row['mean_controller_runtime_sec']):.3f}",
                "bandit_overhead_ms": f"{1000.0 * float(row['mean_setup_bandit_overhead_sec']):.3f}",
                "end_to_end_ms": f"{1000.0 * float(row['mean_end_to_end_runtime_sec']):.3f}",
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
        "window_summary": str(window_summary_path),
        "rolling_window": int(rolling_window),
    }
    if learned_action_methods:
        output["learned_per_cycle_action_trajectories"] = str(action_trajectory_path)
    if has_split_windows:
        output["last_1000_runtime_breakdown"] = str(last_1000_runtime_path)
        output["first_vs_last_components"] = str(first_last_components_path)
    if improvement_reference is not None:
        output["first_vs_last_improvement_ci"] = str(first_last_ci_path)
    if analysis_stop is not None:
        output["analysis_scope"] = str(artifact_dir / "analysis_scope.json")
        output["analysis_result"] = str(artifact_dir / "analysis_result.json")
    _write_json(artifact_dir / "plot_summary.json", output)
    return output
