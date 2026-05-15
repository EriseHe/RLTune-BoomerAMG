from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np


def create_run_output_dir(
    *,
    base_dir: Path,
    script_name: str,
    problem_name: str,
    size_tag: str,
    T: int,
    seed: int,
) -> Path:
    """
    Create a deterministic per-run folder:
      <base>/<script>_<problem>_<size>_T<...>_<seed>/
    """

    def _slug(text: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip()).strip("-").lower()

    base_name = (
        f"{_slug(script_name)}_{_slug(problem_name)}_{_slug(size_tag)}_T{int(T)}_{int(seed)}"
    )

    # Avoid overwriting a previous run: if the deterministic name already exists,
    # append a monotonically increasing suffix:  _1, _2, ...
    run_dir = Path(base_dir) / base_name
    if run_dir.exists():
        suffix = 1
        while True:
            candidate = Path(base_dir) / f"{base_name}_{suffix}"
            if not candidate.exists():
                run_dir = candidate
                break
            suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _last_window_default_diagnostics(
    *,
    traces: Dict[str, Dict[str, np.ndarray]],
    trace_keys: Sequence[str],
    method_names: Sequence[str],
    default_params: Dict[str, Any] | None,
    window: int,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Compute last-window diagnostics against default action from traces:
      - fraction of rounds exactly equal to default
      - mode action tuple/count in that window
    """
    frac_map: Dict[str, float] = {}
    mode_map: Dict[str, Dict[str, Any]] = {}

    if not trace_keys:
        return frac_map, mode_map

    for name in method_names:
        trace = traces.get(name)
        if not trace:
            frac_map[name] = 0.0
            mode_map[name] = {"action": None, "count": 0, "window": int(window)}
            continue

        t_len = len(next(iter(trace.values())))
        start = max(0, int(t_len) - int(window))

        if default_params is not None:
            mask = np.ones(t_len - start, dtype=bool)
            for key in trace_keys:
                if key not in trace:
                    mask &= False
                    continue
                mask &= np.isclose(
                    trace[key][start:],
                    float(default_params[key]),
                    rtol=0.0,
                    atol=1e-12,
                )
            frac_map[name] = float(np.mean(mask)) if mask.size else 0.0
        else:
            frac_map[name] = 0.0

        keys = [tuple(round(float(trace[key][idx]), 6) for key in trace_keys) for idx in range(start, t_len)]
        if keys:
            action, count = Counter(keys).most_common(1)[0]
            mode_map[name] = {"action": list(action), "count": int(count), "window": int(window)}
        else:
            mode_map[name] = {"action": None, "count": 0, "window": int(window)}

    return frac_map, mode_map


def save_runtime_artifacts(
    *,
    run_dir: Path,
    run_prefix: str,
    method_names: Sequence[str],
    runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    T: int,
    title: str,
    default_method_name: str = "default (fixed)",
    traces: Dict[str, Dict[str, np.ndarray]] | None = None,
    trace_keys: Sequence[str] = (),
    trace_title: str | None = None,
    default_params: Dict[str, Any] | None = None,
    diagnostics_window: int = 500,
    summary_extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Save standard runtime artifacts:
      - cumulative end-to-end runtime plot
      - optional parameter trace plot (bandit methods only)
      - summary JSON with totals + last-window diagnostics
    """
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Plot as "instance index" (left: 0, right: T) vs cumulative end-to-end runtime.
    # Include the natural start point (t=0, 0 runtime).
    x = np.arange(0, int(T) + 1, dtype=int)  # 0..T
    plt.figure(figsize=(10, 5))
    method_colors: Dict[str, Any] = {}
    for name in method_names:
        per_round = np.asarray(runtime_sec[name], dtype=float) + np.asarray(overhead_sec[name], dtype=float)
        cum = np.concatenate(([0.0], np.cumsum(per_round)))
        if name == default_method_name:
            line, = plt.plot(x, cum, "--", linewidth=2.0, label=name)
        else:
            line, = plt.plot(x, cum, linewidth=2.3, label=name)
        method_colors[name] = line.get_color()

    plt.xlabel("instance")
    plt.ylabel("cumulative runtime (seconds)")
    plt.title(title, fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plot_path = run_dir / f"{run_prefix}_runtime_cumulative.png"
    plt.savefig(plot_path, dpi=256)
    plt.close()

    # Also save a runtime-only cumulative plot so raw solver/runtime trends are
    # visible separately from tuner overhead.
    plt.figure(figsize=(10, 5))
    for name in method_names:
        per_round = np.asarray(runtime_sec[name], dtype=float)
        cum = np.concatenate(([0.0], np.cumsum(per_round)))
        if name == default_method_name:
            plt.plot(x, cum, "--", linewidth=2.0, label=name)
        else:
            plt.plot(x, cum, linewidth=2.3, label=name)

    plt.xlabel("instance")
    plt.ylabel("cumulative runtime (seconds)")
    plt.title(f"{title}  [runtime only]", fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()
    runtime_only_plot_path = run_dir / f"{run_prefix}_runtime_only_cumulative.png"
    plt.savefig(runtime_only_plot_path, dpi=256)
    plt.close()

    trace_plot_path: Path | None = None
    traces = traces or {}
    bandit_names = [name for name in method_names if name in traces and traces[name]]
    if bandit_names and trace_keys:
        def _trace_ylim(key: str) -> Tuple[float, float]:
            values = []
            for bandit_name in bandit_names:
                trace = traces.get(bandit_name, {})
                if key not in trace:
                    continue
                arr = np.asarray(trace[key], dtype=float).reshape(-1)
                finite = arr[np.isfinite(arr)]
                if finite.size:
                    values.append(finite)

            if default_params is not None and key in default_params:
                dv = float(default_params[key])
                if np.isfinite(dv):
                    values.append(np.asarray([dv], dtype=float))

            if not values:
                return (-0.02, 1.02)

            data = np.concatenate(values, axis=0)
            lo = float(np.min(data))
            hi = float(np.max(data))
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return (-0.02, 1.02)

            if np.isclose(lo, hi, rtol=0.0, atol=1e-12):
                pad = 0.05 * max(1.0, abs(lo))
                if pad <= 0.0 or not np.isfinite(pad):
                    pad = 0.05
            else:
                pad = 0.05 * (hi - lo)

            return (lo - pad, hi + pad)

        nrows = len(bandit_names)
        ncols = len(trace_keys)
        ylims = {str(k): _trace_ylim(str(k)) for k in trace_keys}
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows), sharex=True, sharey=False)
        axes = np.atleast_2d(axes)
        t_axis = np.arange(1, int(T) + 1)

        for row_idx, name in enumerate(bandit_names):
            bandit_color = method_colors.get(name, "C0")
            for col_idx, key in enumerate(trace_keys):
                ax = axes[row_idx, col_idx]
                series = traces[name][key]
                ax.scatter(
                    t_axis,
                    series,
                    s=14,
                    marker=".",
                    color=bandit_color,
                    alpha=0.9,
                    linewidths=0.0,
                    rasterized=True,
                )
                if row_idx == 0:
                    ax.set_title(str(key), fontsize=10)
                if col_idx == 0:
                    ax.set_ylabel(name, fontsize=8)
                ax.set_ylim(*ylims[str(key)])
                ax.grid(True, alpha=0.25)
                if row_idx == nrows - 1:
                    ax.set_xlabel("t")

        fig.suptitle(trace_title or f"Bandit parameter traces  T={int(T)}", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        trace_plot_path = run_dir / f"{run_prefix}_param_trace.png"
        fig.savefig(trace_plot_path, dpi=256)
        plt.close(fig)

    frac_default, mode_action = _last_window_default_diagnostics(
        traces=traces,
        trace_keys=trace_keys,
        method_names=method_names,
        default_params=default_params,
        window=int(diagnostics_window),
    )

    total_hypre = {name: float(np.sum(np.asarray(runtime_sec[name], dtype=float))) for name in method_names}
    total_overhead = {name: float(np.sum(np.asarray(overhead_sec[name], dtype=float))) for name in method_names}
    total_end_to_end = {name: float(total_hypre[name] + total_overhead[name]) for name in method_names}

    summary: Dict[str, Any] = {
        "T": int(T),
        "method_names": list(method_names),
        "default_method_name": str(default_method_name),
        "total_hypre_runtime_sec": total_hypre,
        "total_bandit_overhead_sec": total_overhead,
        "total_end_to_end_sec": total_end_to_end,
        "plot": str(plot_path),
        "runtime_only_plot": str(runtime_only_plot_path),
        "trace_plot": str(trace_plot_path) if trace_plot_path is not None else None,
        "trace_keys": list(trace_keys),
        "last500_fraction_equal_default": frac_default,
        "last500_mode_action": mode_action,
    }

    if summary_extra:
        summary.update(summary_extra)

    summary_path = run_dir / f"{run_prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["summary_path"] = str(summary_path)
    return summary
