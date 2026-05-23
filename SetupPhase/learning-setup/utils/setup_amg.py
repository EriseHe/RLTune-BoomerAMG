from __future__ import annotations

import csv
import time
import sys
import numpy as np

from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple
from pathlib import Path

<<<<<<< HEAD
from learners._amg_action_features import (
    ParameterSpaceSpec,
    action_key_from_parameter_space_spec,
    default_action_from_parameter_space_spec,
    enumerate_actions_from_parameter_space_spec,
)
=======
>>>>>>> dda295d259f1308ca92fd1591a033af1b6a0ab50
from solver import solve

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


<<<<<<< HEAD
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
=======
def progress_bar(current: int, total: int, prefix: str = "", every: int | None = None) -> None:
>>>>>>> dda295d259f1308ca92fd1591a033af1b6a0ab50
    width = 30
    if total <= 0:
        return
    if every is None:
        every = max(1, total // 100)
    if current not in (1, total) and current % every != 0:
        return
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
<<<<<<< HEAD
    timing = ""
    if start_time is not None and current > 0:
        elapsed = max(0.0, float(time.perf_counter() - start_time))
        rate = elapsed / float(current)
        eta = max(0.0, rate * float(total - current))
        timing = f" elapsed {_format_duration(elapsed)} eta {_format_duration(eta)}"
    print(f"\r{prefix} [{bar}] {current}/{total}{timing}", end="", flush=True)
=======
    print(f"\r{prefix} [{bar}] {current}/{total}", end="", flush=True)
>>>>>>> dda295d259f1308ca92fd1591a033af1b6a0ab50
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


<<<<<<< HEAD
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
    for tuned_action in enumerate_actions_from_parameter_space_spec(parameter_spec):
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


=======
>>>>>>> dda295d259f1308ca92fd1591a033af1b6a0ab50
def _sample_problem(
    problem_set: Any, 
    rng: np.random.Generator, 
    t: int, 
    trial: int
    ) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, Any]]:
    """
    Sample a problem from the problem set.
    
    Returns:
        Tuple of (mkw, context, meta)
    """
    sampler = problem_set.sample if hasattr(problem_set, "sample") else problem_set
    payload = sampler(rng=rng, t=t, trial=trial)
    
    # Parse payload - support dict or tuple format
    if isinstance(payload, dict):
        mkw = payload["mkw"]
        context = np.asarray(payload.get("context", []), dtype=float)
        meta = payload.get("meta", {})
        return mkw, context, meta
    
    # Tuple format: (mkw, context) or (mkw, context, meta)
    if len(payload) == 2:
        mkw, context = payload
        return mkw, np.asarray(context, dtype=float), {}
    
    mkw, context, meta = payload
    return mkw, np.asarray(context, dtype=float), meta or {}


def _solve(params: Dict[str, Any], mkw: Dict[str, Any], fail_penalty: float) -> Dict[str, float]:
    """
    Run the solver and measure work units and hypre-only runtime.
    """
    try:
        res = solve(params=params, **mkw)
        return {"wu": float(res.work_units), "runtime": float(res.runtime_sec)}
    
    except Exception:
        # Penalize failures heavily under runtime loss too (avoid "fast failure")
        return {"wu": float(fail_penalty), "runtime": 1e9}


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_amg_setup_experiment(
    problem_set: Any,
    parameter_space: Dict[str, Any],
    bandit: Any,
    loss: Callable[..., float],
    *,
    T: int,
    trials: int,
    seed: int,
    baselines: Sequence[Tuple[str, Dict[str, Any]]],
    fail_penalty: float = 1e6,
) -> Dict[str, Any]:
    """
    Run multi-armed bandit experiment for AMG setup parameter tuning.
    
    Args:
        problem_set: Problem generator with .sample(rng, t, trial) method
        parameter_space: Space of possible parameters
        bandit: Bandit algorithm with .select(context, parameter_space) 
                and optional .update(loss, context, params, outcome)
        loss: Loss function(outcome, context, params, meta, mkw, t, trial)
        T: Number of rounds per trial
        trials: Number of independent trials
        seed: Random seed
        baselines: List of (name, params) baseline configurations
        fail_penalty: Penalty value for solver failures
    
    Returns:
        Dictionary containing results and logs
    """
    # ========================================================================
    # Initialize result arrays
    # ========================================================================
    num_baselines = len(baselines)
    baseline_names = [name for name, _ in baselines]
    
    bandit_wu = np.zeros((T, trials), dtype=float)
    bandit_rt = np.zeros((T, trials), dtype=float)
    bandit_overhead_rt = np.zeros((T, trials), dtype=float)
    base_wu = np.zeros((T, trials, num_baselines), dtype=float)
    base_rt = np.zeros((T, trials, num_baselines), dtype=float)
    
    logs: List[Dict[str, Any]] = []
    trial_policies: List[Any] = []

    # ========================================================================
    # Run trials
    # ========================================================================
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        
        # Initialize policy for this trial
        if hasattr(bandit, "new_trial"):
            policy = bandit.new_trial(
                parameter_space=parameter_space,
                T=T,
                seed=seed + 10000 + trial,
                trial=trial,
            )
        else:
            policy = bandit
        
        trial_policies.append(policy)
        
        # ====================================================================
        # Run rounds within this trial
        # ====================================================================
        for t in range(T):
            # Sample problem instance
            mkw, context, meta = _sample_problem(problem_set, rng, t + 1, trial + 1)
            
            # Select parameters using bandit policy
            sel_start = time.perf_counter_ns()
            selected = policy.select(context=context, parameter_space=parameter_space)
            sel_sec = (time.perf_counter_ns() - sel_start) / 1e9
            
            # Parse selection result
            if isinstance(selected, tuple):
                params, info = selected[0], selected[1] or {}
            else:
                params, info = selected, {}
            
            # ================================================================
            # Evaluate bandit selection
            # ================================================================
            out_bandit = _solve(params, mkw, fail_penalty)
            bandit_wu[t, trial] = out_bandit["wu"]
            bandit_rt[t, trial] = out_bandit["runtime"]
            
            # ================================================================
            # Evaluate all baselines
            # ================================================================
            for i, (_name, baseline_params) in enumerate(baselines):
                out_baseline = _solve(baseline_params, mkw, fail_penalty)
                base_wu[t, trial, i] = out_baseline["wu"]
                base_rt[t, trial, i] = out_baseline["runtime"]
            
            # ================================================================
            # Compute loss and update policy
            # ================================================================
            loss_start = time.perf_counter_ns()
            loss_value = float(
                loss(
                    outcome=out_bandit,
                    context=context,
                    params=params,
                    meta=meta,
                    mkw=mkw,
                    t=t + 1,
                    trial=trial + 1,
                )
            )
            loss_sec = (time.perf_counter_ns() - loss_start) / 1e9
            
            upd_sec = 0.0
            if hasattr(policy, "update"):
                upd_start = time.perf_counter_ns()
                policy.update(
                    loss=loss_value,
                    context=context,
                    params=params,
                    outcome=out_bandit,
                )
                upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

            bandit_overhead_rt[t, trial] = float(sel_sec + loss_sec + upd_sec)
            
            # ================================================================
            # Log first trial details
            # ================================================================
            if trial == 0:
                row = {
                    "trial": trial + 1,
                    "t": t + 1,
                    "loss_used": loss_value,
                    "overhead_sec": float(sel_sec + loss_sec + upd_sec),
                    "select_sec": float(sel_sec),
                    "loss_eval_sec": float(loss_sec),
                    "update_sec": float(upd_sec),
                }
                
                # Add chosen parameters
                row.update({f"chosen_{k}": v for k, v in params.items()})
                
                # Add selection info
                row.update({str(k): v for k, v in info.items()})
                
                # Add context features
                for i_ctx, val in enumerate(np.asarray(context).reshape(-1)):
                    row[f"context_{i_ctx}"] = float(val)
                
                logs.append(row)

    # ========================================================================
    # Return results
    # ========================================================================
    return {
        "T": T,
        "trials": trials,
        "seed": seed,
        "baseline_names": baseline_names,
        "bandit_wu": bandit_wu,
        "bandit_runtime": bandit_rt,
        "bandit_overhead_runtime": bandit_overhead_rt,
        "bandit_total_runtime": bandit_rt + bandit_overhead_rt,
        "baseline_wu": base_wu,
        "baseline_runtime": base_rt,
        "logs": logs,
        "trial_policies": trial_policies,
    }
