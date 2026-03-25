from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "mplconfig_bandits_for_setup"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _short_label(name: str) -> str:
    if name == "default (fixed)":
        return "Default"
    if "Shared LinUCB v2" in name and "tune3" in name:
        return "LinUCB Quad (tune 3)"
    if "Shared LinUCB v2" in name and "tune5" in name:
        return "LinUCB Quad (tune 5)"
    if "Shared LinUCB v3" in name and "tune3" in name:
        return "LinUCB-v3 (tune3)"
    if "Shared LinUCB v3" in name and "tune5" in name:
        return "LinUCB-v3 (tune5)"
    if "Shared LinUCB v4" in name and "tune7" in name:
        return "LinUCB Mixed (tune 7)"
    return name


def _color_for(name: str) -> str:
    if name == "default (fixed)":
        return "#4D4D4D"
    if "Shared LinUCB v2" in name:
        return "#0072B2"
    if "Shared LinUCB v3" in name:
        return "#D55E00"
    if "Shared LinUCB v4" in name:
        return "#009E73"
    return "#CC79A7"


def _line_style_for(name: str) -> str:
    return "--" if name == "default (fixed)" else "-"


def _setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "Times New Roman", "DejaVu Serif", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.labelsize": 19,
            "axes.titlesize": 20,
            "legend.fontsize": 16,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "figure.titlesize": 20,
        }
    )


def _save_svg(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, format="svg", bbox_inches="tight")


def _load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_runtime_matrix(
    csv_path: Path,
    *,
    method_names: Sequence[str],
    T: int,
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {
        name: np.full(int(T), np.nan, dtype=float) for name in method_names
    }
    seen = {name: set() for name in method_names}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row["method"])
            if name not in arrays:
                continue
            t = int(row["t"])
            if t < 1 or t > int(T):
                raise ValueError(f"row has invalid t={t}")
            idx = t - 1
            if idx in seen[name]:
                raise ValueError(f"duplicate (method,t) row for {name} at t={t}")
            seen[name].add(idx)
            arrays[name][idx] = float(row["runtime_sec"])

    for name in method_names:
        if np.isnan(arrays[name]).any():
            missing = np.where(np.isnan(arrays[name]))[0] + 1
            raise ValueError(f"missing runtime rows for {name}: first missing t={int(missing[0])}")

    return arrays


def _load_action_keys(
    csv_path: Path,
    *,
    method_names: Sequence[str],
    T: int,
) -> Dict[str, List[Tuple[str, ...]]]:
    action_cols = [
        "strong_threshold",
        "max_row_sum",
        "trunc_factor",
        "coarsen_type",
        "P_max_elmts",
        "agg_num_levels",
        "interp_type",
        "agg_interp_type",
        "agg_tr",
        "agg_Pmx",
    ]
    arrays: Dict[str, List[Tuple[str, ...] | None]] = {
        name: [None] * int(T) for name in method_names
    }
    seen = {name: set() for name in method_names}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row["method"])
            if name not in arrays:
                continue
            t = int(row["t"])
            idx = t - 1
            if idx in seen[name]:
                raise ValueError(f"duplicate (method,t) row for {name} at t={t}")
            seen[name].add(idx)
            arrays[name][idx] = tuple(str(row[col]) for col in action_cols)

    out: Dict[str, List[Tuple[str, ...]]] = {}
    for name in method_names:
        if any(x is None for x in arrays[name]):
            missing = next(i + 1 for i, x in enumerate(arrays[name]) if x is None)
            raise ValueError(f"missing action rows for {name}: first missing t={missing}")
        out[name] = [x for x in arrays[name] if x is not None]
    return out


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = int(x.size)
    if window <= 0:
        raise ValueError("window must be positive")
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out
    csum = np.cumsum(np.insert(x, 0, 0.0))
    out[window - 1 :] = (csum[window:] - csum[:-window]) / float(window)
    return out


def _rolling_unique_count(keys: Sequence[Tuple[str, ...]], window: int) -> np.ndarray:
    n = len(keys)
    out = np.full(n, np.nan, dtype=float)
    if window <= 0:
        raise ValueError("window must be positive")
    if n < window:
        return out
    for end in range(window - 1, n):
        start = end - window + 1
        out[end] = float(len(set(keys[start : end + 1])))
    return out


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xs = np.sort(np.asarray(x, dtype=float).reshape(-1))
    ys = np.arange(1, xs.size + 1, dtype=float) / float(xs.size)
    return xs, ys


def _tail_mean(x: np.ndarray, window: int) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    w = min(int(window), int(x.size))
    return float(np.mean(x[-w:])) if w > 0 else float("nan")


def _diagnostic_rows(
    summary: Dict[str, Any],
    runtime: Dict[str, np.ndarray],
    default_method: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    default_rt = runtime[default_method]
    retry_total = summary.get("retry_count_total", {})
    for name in summary["method_names"]:
        rt = runtime[name]
        row = {
            "method": name,
            "short_label": _short_label(name),
            "total_runtime_sec": float(np.sum(rt)),
            "mean_runtime_sec": float(np.mean(rt)),
            "median_runtime_sec": float(np.median(rt)),
            "p90_runtime_sec": float(np.quantile(rt, 0.90)),
            "p95_runtime_sec": float(np.quantile(rt, 0.95)),
            "slope_50": _tail_mean(rt, 50),
            "slope_100": _tail_mean(rt, 100),
            "slope_200": _tail_mean(rt, 200),
            "slope_500": _tail_mean(rt, 500),
            "retry_count_total": int(retry_total.get(name, 0)),
        }
        if name == default_method:
            row["total_savings_vs_default_sec"] = 0.0
            row["win_fraction_vs_default"] = 0.0
            row["mean_runtime_ratio_vs_default"] = 1.0
        else:
            diff = default_rt - rt
            row["total_savings_vs_default_sec"] = float(np.sum(diff))
            row["win_fraction_vs_default"] = float(np.mean(rt < default_rt))
            safe_ratio = np.divide(
                rt,
                default_rt,
                out=np.full_like(rt, np.nan, dtype=float),
                where=default_rt > 0.0,
            )
            row["mean_runtime_ratio_vs_default"] = float(np.nanmean(safe_ratio))
        rows.append(row)
    return rows


def _write_diagnostics(
    out_dir: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    plot_paths: Dict[str, str],
) -> None:
    json_path = out_dir / "siam_runtime_only_diagnostics.json"
    csv_path = out_dir / "siam_runtime_only_diagnostics.csv"

    payload = {
        "plots": plot_paths,
        "rows": list(rows),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_cumulative(
    out_path: Path,
    *,
    summary: Dict[str, Any],
    runtime: Dict[str, np.ndarray],
    diagnostics: Sequence[Dict[str, Any]],
) -> None:
    methods = list(summary["method_names"])
    t_axis = np.arange(1, int(summary["T"]) + 1)
    diag_map = {str(row["method"]): row for row in diagnostics}
    zoom_T = min(400, int(summary["T"]))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.7, 1.0]},
    )
    ax = axes[0]
    for name in methods:
        rt = runtime[name]
        row = diag_map[name]
        label = f"{_short_label(name)}  ({row['total_runtime_sec']:.0f} s)"
        ax.plot(
            t_axis,
            np.cumsum(rt),
            color=_color_for(name),
            linestyle=_line_style_for(name),
            linewidth=2.8,
            label=label,
        )
    ax.set_xlabel("Instance index")
    ax.set_ylabel("Cumulative HYPRE runtime (s)")
    ax.set_title("Cumulative Runtime")
    ax.legend(frameon=False, loc="upper left")

    ax_zoom = axes[1]
    for name in methods:
        rt = runtime[name]
        ax_zoom.plot(
            t_axis[:zoom_T],
            np.cumsum(rt[:zoom_T]),
            color=_color_for(name),
            linestyle=_line_style_for(name),
            linewidth=2.6,
        )
    ax_zoom.set_xlabel("Instance index")
    ax_zoom.set_ylabel("Cumulative HYPRE runtime (s)")
    ax_zoom.set_title(f"Zoom: first {zoom_T} instances")
    _save_svg(fig, out_path)
    plt.close(fig)


def _plot_savings_vs_default(
    out_path: Path,
    *,
    summary: Dict[str, Any],
    runtime: Dict[str, np.ndarray],
    diagnostics: Sequence[Dict[str, Any]],
    default_method: str,
) -> None:
    methods = list(summary["method_names"])
    t_axis = np.arange(1, int(summary["T"]) + 1)
    default_rt = runtime[default_method]
    diag_map = {str(row["method"]): row for row in diagnostics}
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    ax.axhline(0.0, color="#666666", linewidth=1.1, linestyle=":")
    for name in methods:
        if name == default_method:
            continue
        savings = np.cumsum(default_rt - runtime[name])
        row = diag_map[name]
        label = f"{_short_label(name)}  (final +{row['total_savings_vs_default_sec']:.0f} s)"
        ax.plot(
            t_axis,
            savings,
            color=_color_for(name),
            linewidth=2.8,
            label=label,
        )
    ax.set_xlabel("Instance index")
    ax.set_ylabel("Cumulative savings vs default (s)")
    ax.set_title("Cumulative Runtime Savings vs Default")
    ax.legend(frameon=False, loc="upper left")
    _save_svg(fig, out_path)
    plt.close(fig)


def _plot_slope(
    out_path: Path,
    *,
    summary: Dict[str, Any],
    runtime: Dict[str, np.ndarray],
    window: int,
) -> None:
    methods = list(summary["method_names"])
    t_axis = np.arange(1, int(summary["T"]) + 1)
    series: Dict[str, np.ndarray] = {
        name: _rolling_mean(runtime[name], int(window)) for name in methods
    }
    finite_values: List[np.ndarray] = []
    for name in methods:
        arr = series[name]
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_values.append(arr)
    y_min = float(min(np.min(x) for x in finite_values))
    y_max = float(max(np.max(x) for x in finite_values))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 0.02 * max(1.0, y_max)
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    for name in methods:
        label = f"{_short_label(name)}  ({_tail_mean(runtime[name], window):.3f} s)"
        ax.plot(
            t_axis,
            series[name],
            color=_color_for(name),
            linestyle=_line_style_for(name),
            linewidth=2.7,
            label=label,
        )
    ax.set_title(f"Trailing Mean Runtime (window = {window})")
    ax.set_xlabel("Instance index")
    ax.set_ylabel("Slope = trailing mean runtime (s)")
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.legend(frameon=False, loc="upper right")
    _save_svg(fig, out_path)
    plt.close(fig)


def _plot_ecdf(
    out_path: Path,
    *,
    summary: Dict[str, Any],
    runtime: Dict[str, np.ndarray],
) -> None:
    methods = list(summary["method_names"])
    default_method = str(summary.get("default_method_name", "default (fixed)"))
    default_rt = runtime[default_method]
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    for name in methods:
        xs, ys = _ecdf(runtime[name])
        label = f"{_short_label(name)}"
        ax.plot(
            xs,
            ys,
            color=_color_for(name),
            linestyle=_line_style_for(name),
            linewidth=2.7,
            label=label,
        )
    if np.min(default_rt) > 0.0:
        ax.set_xscale("log")
    ax.set_xlabel("Per-instance HYPRE runtime (s)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Per-instance Runtime Distribution")
    ax.legend(frameon=False, loc="lower right")
    _save_svg(fig, out_path)
    plt.close(fig)


def _plot_advantage_vs_default(
    out_path: Path,
    *,
    summary: Dict[str, Any],
    runtime: Dict[str, np.ndarray],
    diagnostics: Sequence[Dict[str, Any]],
    default_method: str,
    window: int,
) -> None:
    methods = list(summary["method_names"])
    t_axis = np.arange(1, int(summary["T"]) + 1)
    plot_T = min(1500, int(summary["T"]))
    default_rt = runtime[default_method]
    diag_map = {str(row["method"]): row for row in diagnostics}
    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    ax.axhline(0.0, color="#666666", linewidth=1.1, linestyle=":")
    for name in methods:
        if name == default_method:
            continue
        advantage = _rolling_mean(default_rt - runtime[name], window)
        row = diag_map[name]
        label = (
            f"{_short_label(name)}  "
            f"(wins {100.0 * row['win_fraction_vs_default']:.1f}%)"
        )
        ax.plot(
            t_axis[:plot_T],
            advantage[:plot_T],
            color=_color_for(name),
            linewidth=2.7,
            label=label,
        )
    ax.set_xlabel("Instance index")
    ax.set_ylabel("Trailing mean savings vs default (s)")
    ax.set_title(f"Trailing Mean Runtime Savings vs Default (window = {window}, first {plot_T})")
    ax.legend(frameon=False, loc="lower right")
    _save_svg(fig, out_path)
    plt.close(fig)


def _plot_unique_actions(
    out_path: Path,
    *,
    summary: Dict[str, Any],
    action_keys: Dict[str, List[Tuple[str, ...]]],
    window: int,
) -> None:
    methods = list(summary["method_names"])
    t_axis = np.arange(1, int(summary["T"]) + 1)
    series: Dict[str, np.ndarray] = {
        name: _rolling_unique_count(action_keys[name], int(window)) for name in methods
    }
    finite_values: List[np.ndarray] = []
    for name in methods:
        arr = series[name]
        arr = arr[np.isfinite(arr)]
        if arr.size:
            finite_values.append(arr)
    y_min = float(min(np.min(x) for x in finite_values))
    y_max = float(max(np.max(x) for x in finite_values))
    pad = 0.05 * (y_max - y_min) if y_max > y_min else 1.0

    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)
    for name in methods:
        tail_val = series[name][np.isfinite(series[name])]
        tail_text = f"{tail_val[-1]:.0f}" if tail_val.size else "n/a"
        label = f"{_short_label(name)}  ({tail_text})"
        ax.plot(
            t_axis,
            series[name],
            color=_color_for(name),
            linestyle=_line_style_for(name),
            linewidth=2.7,
            label=label,
        )
    ax.set_xlabel("Instance index")
    ax.set_ylabel(f"Unique actions in trailing window ({window})")
    ax.set_title(f"Action-space stability (window = {window})")
    ax.set_ylim(max(0.0, y_min - pad), y_max + pad)
    ax.legend(frameon=False, loc="upper right")
    _save_svg(fig, out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create runtime-only SIAM-style figures from a test_final run folder.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory containing summary JSON and CSV.")
    parser.add_argument(
        "--include-methods",
        type=str,
        default="",
        help="Comma-separated subset of methods to include. Default: all methods in summary.",
    )
    args = parser.parse_args()

    _setup_matplotlib()

    run_dir = Path(args.run_dir).resolve()
    summary_path = run_dir / "final_unified_tuning_runtime_summary.json"
    csv_path = run_dir / "per_instance_runtime_data.csv"

    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    summary = _load_summary(summary_path)
    method_names = list(summary["method_names"])
    if args.include_methods.strip():
        include = [x.strip() for x in args.include_methods.split(",") if x.strip()]
        missing = [x for x in include if x not in method_names]
        if missing:
            raise ValueError(f"include-methods contains unknown methods: {missing}")
        method_names = include
        summary = dict(summary)
        summary["method_names"] = list(method_names)
    T = int(summary["T"])
    default_method = str(summary.get("default_method_name", "default (fixed)"))
    runtime = _load_runtime_matrix(csv_path, method_names=method_names, T=T)
    action_keys = _load_action_keys(csv_path, method_names=method_names, T=T)
    diagnostics = _diagnostic_rows(summary, runtime, default_method)

    for stale_path in run_dir.glob("siam_runtime_only*.png"):
        stale_path.unlink()
    for stale_path in run_dir.glob("siam_runtime_only*.svg"):
        stale_path.unlink()

    cumulative_plot = run_dir / "siam_runtime_only_cumulative.svg"
    savings_plot = run_dir / "siam_runtime_only_savings_vs_default.svg"
    slope_50_plot = run_dir / "siam_runtime_only_slope_w50.svg"
    slope_200_plot = run_dir / "siam_runtime_only_slope_w200.svg"
    slope_500_plot = run_dir / "siam_runtime_only_slope_w500.svg"
    ecdf_plot = run_dir / "siam_runtime_only_ecdf.svg"
    advantage_plot = run_dir / "siam_runtime_only_advantage_w100.svg"
    unique_actions_plot = run_dir / "siam_runtime_only_unique_actions_w200.svg"

    _plot_cumulative(
        cumulative_plot,
        summary=summary,
        runtime=runtime,
        diagnostics=diagnostics,
    )
    _plot_savings_vs_default(
        savings_plot,
        summary=summary,
        runtime=runtime,
        diagnostics=diagnostics,
        default_method=default_method,
    )
    _plot_slope(
        slope_50_plot,
        summary=summary,
        runtime=runtime,
        window=50,
    )
    _plot_slope(
        slope_200_plot,
        summary=summary,
        runtime=runtime,
        window=200,
    )
    _plot_slope(
        slope_500_plot,
        summary=summary,
        runtime=runtime,
        window=500,
    )
    _plot_ecdf(
        ecdf_plot,
        summary=summary,
        runtime=runtime,
    )
    _plot_advantage_vs_default(
        advantage_plot,
        summary=summary,
        runtime=runtime,
        diagnostics=diagnostics,
        default_method=default_method,
        window=100,
    )
    _plot_unique_actions(
        unique_actions_plot,
        summary=summary,
        action_keys=action_keys,
        window=200,
    )

    _write_diagnostics(
        run_dir,
        diagnostics,
        plot_paths={
            "siam_runtime_only_cumulative": str(cumulative_plot),
            "siam_runtime_only_savings_vs_default": str(savings_plot),
            "siam_runtime_only_slope_w50": str(slope_50_plot),
            "siam_runtime_only_slope_w200": str(slope_200_plot),
            "siam_runtime_only_slope_w500": str(slope_500_plot),
            "siam_runtime_only_ecdf": str(ecdf_plot),
            "siam_runtime_only_advantage_w100": str(advantage_plot),
            "siam_runtime_only_unique_actions_w200": str(unique_actions_plot),
        },
    )

    print("=== Runtime-only paper figures written ===")
    print(cumulative_plot)
    print(savings_plot)
    print(slope_50_plot)
    print(slope_200_plot)
    print(slope_500_plot)
    print(ecdf_plot)
    print(advantage_plot)
    print(unique_actions_plot)


if __name__ == "__main__":
    main()
