from __future__ import annotations

import csv
import time
import numpy as np

from typing import Any, Dict, Iterable, List, Sequence
from pathlib import Path

from setup.learners.common import (
    ParameterSpaceSpec,
    action_key_from_parameter_space_spec,
    default_action_from_parameter_space_spec,
    iter_actions_from_parameter_space_spec,
)
# ============================================================================
# Helper Functions
# ============================================================================

def init_param_trace(keys: Sequence[str], T: int) -> Dict[str, np.ndarray]:
    """
    Initialize a numeric trace for chosen parameters.

    Returns a dict mapping each key -> np.ndarray of length T (float),
    filled with NaN until recorded.
    """
    return {str(k): np.full(int(T), np.nan, dtype=float) for k in keys}


def record_param_trace(trace: Dict[str, np.ndarray], *, t: int, params: Dict[str, Any], keys: Sequence[str]) -> None:
    """
    Record params[key] into trace[key][t].

    Missing keys are left as NaN.
    """
    for k in keys:
        if k in params and k in trace:
            trace[k][int(t)] = float(params[k])


def _format_duration(seconds: float) -> str:
    seconds_i = max(0, int(round(float(seconds))))
    hours, rem = divmod(seconds_i, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    every: int | None = None,
    start_time: float | None = None,
) -> None:
    width = 30
    if total <= 0:
        return
    if every is None:
        every = max(1, total // 100)
    if current not in (1, total) and current % every != 0:
        return
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    timing = ""
    if start_time is not None and current > 0:
        elapsed = max(0.0, float(time.perf_counter() - start_time))
        rate = elapsed / float(current)
        eta = max(0.0, rate * float(total - current))
        timing = f" elapsed {_format_duration(elapsed)} eta {_format_duration(eta)}"
    print(f"\r{prefix} [{bar}] {current}/{total}{timing}", end="", flush=True)
    if current == total:
        print()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """
    Compute a simple centered moving average.
    """
    if window <= 1 or len(values) <= 1:
        return values.copy()
    w = min(int(window), len(values))
    kernel = np.ones(w, dtype=float) / float(w)
    return np.convolve(values, kernel, mode="same")


def summarize_totals(result: Dict[str, Any], metric: str) -> Dict[str, float]:
    """
    Compute mean total metric over rounds for bandit and baselines.
    """
    bandit_arr = result[f"bandit_{metric}"]
    baseline_arr = result[f"baseline_{metric}"]
    totals: Dict[str, float] = {}
    for i, name in enumerate(result["baseline_names"]):
        totals[name] = float(np.mean(np.sum(baseline_arr[:, :, i], axis=0)))
    totals["bandit"] = float(np.mean(np.sum(bandit_arr, axis=0)))
    return totals


def save_logs(logs: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save list-of-dicts logs to CSV.
    """
    if not logs:
        return
    keys = sorted({k for row in logs for k in row.keys()})
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(logs)


def build_actions_th_coarsen_interp(
    thresholds: Iterable[float],
    coarsen_types: Iterable[int],
    interp_types: Iterable[int],
    *,
    fixed_params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
    """
    Build cartesian-product action space for
    (strong_threshold, coarsen_type, interp_type).
    """
    fixed = dict(fixed_params or {})
    actions: List[Dict[str, Any]] = []
    for th in thresholds:
        for coarsen in coarsen_types:
            for interp in interp_types:
                params = dict(fixed)
                params.update(
                    {
                        "strong_threshold": float(th),
                        "coarsen_type": int(coarsen),
                        "interp_type": int(interp),
                    }
                )
                actions.append(params)
    return actions


def build_actions_th_mxrs_tr(
    thresholds: Iterable[float],
    max_row_sums: Iterable[float],
    trunc_factors: Iterable[float],
    *,
    fixed_params: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:
    """
    Build cartesian-product action space for
    (strong_threshold, max_row_sum, trunc_factor).
    """
    fixed = dict(fixed_params or {})
    actions: List[Dict[str, Any]] = []
    for th in thresholds:
        for max_row_sum in max_row_sums:
            for trunc_factor in trunc_factors:
                params = dict(fixed)
                params.update(
                    {
                        "strong_threshold": float(th),
                        "max_row_sum": float(max_row_sum),
                        "trunc_factor": float(trunc_factor),
                    }
                )
                actions.append(params)
    return actions


def default_action_from_spec(
    parameter_spec: ParameterSpaceSpec,
    *,
    fixed_params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build the default action implied by a generic parameter spec.

    `fixed_params` are merged after the tuned-parameter defaults.
    """
    params = default_action_from_parameter_space_spec(parameter_spec)
    out = dict(fixed_params or {})
    out.update(params)
    return out


def build_actions_from_spec(
    parameter_spec: ParameterSpaceSpec,
    *,
    fixed_params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Build a stable action list from a generic parameter spec.

    Parameter enumeration follows the spec order. Conditional inactive knobs
    are pinned to defaults. The spec-implied default action is appended if the
    enumerated grid misses it.
    """
    fixed = dict(fixed_params or {})
    actions = []
    seen = set()
    for tuned_action in iter_actions_from_parameter_space_spec(parameter_spec):
        key = action_key_from_parameter_space_spec(tuned_action, parameter_spec)
        if key in seen:
            continue
        seen.add(key)
        out = dict(fixed)
        out.update(tuned_action)
        actions.append(out)

    default_tuned = default_action_from_parameter_space_spec(parameter_spec)
    default_key = action_key_from_parameter_space_spec(default_tuned, parameter_spec)
    if default_key not in seen:
        out = dict(fixed)
        out.update(default_tuned)
        actions.append(out)
    return actions
