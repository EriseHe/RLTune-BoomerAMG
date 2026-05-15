"""
Final non-RL setup-phase comparison harness.

Protocol
--------
All branches are evaluated on the same pre-generated scalar anisotropic diffusion instance stream.
At each step, execution order is randomly permuted over all branches:
  - default (fixed)
  - Shared LinUCB v2 | tune3
  - Shared LinUCB v2 | tune5
  - Shared LinUCB v3 | tune3
  - Shared LinUCB v3 | tune5

Fairness rules
--------------
1) Same instance for every branch at step t
2) One within-step permutation over all branches
3) Same family seed across tuning dimensions
   (for example, v2 uses the same seed for tune3 and tune5)

Outputs
-------
1) One cumulative runtime plot
2) One summary JSON
3) One per-instance CSV
"""

from __future__ import annotations

from collections import deque
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learners.SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from learners.SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from learners.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from learners._amg_action_features import ParameterSpaceSpec, ParameterSpec
from solver import solve
from utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
from utils.scalar_anisotropic_diffusion import (
    SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM,
    stencil_0_scalar_anisotropic_diffusion_rl,
)
from utils.setup_amg import (
    build_actions_from_spec,
    build_actions_th_mxrs_tr,
    init_param_trace,
    progress_bar,
    record_param_trace,
)


T = int(os.environ.get("T", "1000"))
if T <= 0:
    raise ValueError("T must be positive")

SEED = int(os.environ.get("SEED", "39393939"))

ALPHA = float(os.environ.get("ALPHA", "1.0"))
L2 = float(os.environ.get("L2", "1.0"))
TUNE7_ALPHA_OVERRIDE = os.environ.get("TUNE7_ALPHA_OVERRIDE", "").strip()

FIXED_N = int(os.environ.get("FIXED_N", "60"))
FIXED_NX = int(os.environ.get("FIXED_NX", str(FIXED_N)))
FIXED_NY = int(os.environ.get("FIXED_NY", str(FIXED_N)))
FIXED_NZ = int(os.environ.get("FIXED_NZ", str(FIXED_N)))
C_MIN = float(os.environ.get("C_MIN", "1.0"))
C_MAX = float(os.environ.get("C_MAX", "1000.0"))

if FIXED_NX <= 0 or FIXED_NY <= 0 or FIXED_NZ <= 0:
    raise ValueError("FIXED_NX/FIXED_NY/FIXED_NZ must all be positive")

CANDIDATE_POOL_SIZE = int(os.environ.get("CANDIDATE_POOL_SIZE", "512"))
ELITE_CACHE_SIZE = int(os.environ.get("ELITE_CACHE_SIZE", "64"))

SOLVER_TOL = float(os.environ.get("SOLVER_TOL", "1e-8"))
SOLVER_MAX_ITER = int(os.environ.get("SOLVER_MAX_ITER", "10000"))
RETRY_MAX_ATTEMPTS = int(os.environ.get("RETRY_MAX_ATTEMPTS", "1000"))
LEGACY_SOLVER_TOL = float(os.environ.get("LEGACY_SOLVER_TOL", "1e-8"))
LEGACY_SOLVER_MAX_ITER = int(os.environ.get("LEGACY_SOLVER_MAX_ITER", "10000"))
FINAL_COMPARE_V2_TUNE3_SOLVER_PROFILES = os.environ.get(
    "FINAL_COMPARE_V2_TUNE3_SOLVER_PROFILES", ""
).strip().lower() in {"1", "true", "yes", "on"}
FAILURE_PENALTY_MULTIPLIER = float(os.environ.get("FAILURE_PENALTY_MULTIPLIER", "2.0"))
FAILURE_SCALE_WINDOW = int(os.environ.get("FAILURE_SCALE_WINDOW", "200"))
FAILURE_SEVERITY_CAP = float(os.environ.get("FAILURE_SEVERITY_CAP", "6.0"))
FAILURE_SCALE_MIN_RUNTIME_SEC = float(os.environ.get("FAILURE_SCALE_MIN_RUNTIME_SEC", "0.0"))
STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER = float(
    os.environ.get("STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER", "3.0")
)

if RETRY_MAX_ATTEMPTS <= 0:
    raise ValueError("RETRY_MAX_ATTEMPTS must be positive")
if FAILURE_SCALE_WINDOW <= 0:
    raise ValueError("FAILURE_SCALE_WINDOW must be positive")
if FAILURE_PENALTY_MULTIPLIER < 0.0:
    raise ValueError("FAILURE_PENALTY_MULTIPLIER must be >= 0")
if FAILURE_SEVERITY_CAP < 0.0:
    raise ValueError("FAILURE_SEVERITY_CAP must be >= 0")
if FAILURE_SCALE_MIN_RUNTIME_SEC < 0.0:
    raise ValueError("FAILURE_SCALE_MIN_RUNTIME_SEC must be >= 0")
if STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER < 0.0:
    raise ValueError("STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER must be >= 0")

plots_base_dir = Path(__file__).resolve().parent.parent / "plots" / "Combined"
plots_base_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_PARAMS = {
    "strong_threshold": 0.25,
    "max_row_sum": 0.90,
    "trunc_factor": 0.00,
    "coarsen_type": 10,
    "interp_type": 6,
    "P_max_elmts": 4,
    "agg_num_levels": 0,
    "agg_interp_type": 4,
    "agg_tr": 0.0,
    "agg_Pmx": 0,
}

DEFAULT_COARSEN_TYPE_VALUES = (0, 2, 6, 8, 10)
DEFAULT_P_MAX_ELMTS_VALUES = (2, 4, 6, 8, 12, 16)
DEFAULT_AGG_NUM_LEVELS_VALUES = (0, 1, 2, 3, 4, 5)
DEFAULT_AGG_TR_VALUES = (0.0, 0.1)
DEFAULT_AGG_PMX_VALUES = (0, 4)
DEFAULT_TUNE7_INTERP_TYPES = (6, 8)
DEFAULT_TUNE7_AGG_INTERP_TYPES = (4, 6)
TRACE_KEYS_FINAL = (
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
)
TUNE7_VARIANT = os.environ.get("TUNE7_VARIANT", "categorical").strip().lower()
TUNE7_CANDIDATE_STRATEGY = os.environ.get("TUNE7_CANDIDATE_STRATEGY", "").strip().lower()
TUNE7_CANDIDATE_POOL_SIZE = int(os.environ.get("TUNE7_CANDIDATE_POOL_SIZE", "1024"))
TUNE7_CANDIDATE_POOL_SIZE_BURNIN = int(os.environ.get("TUNE7_CANDIDATE_POOL_SIZE_BURNIN", "4096"))
TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS = int(os.environ.get("TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS", "200"))
TUNE7_ALPHA_DECAY_BURNIN_ROUNDS = int(os.environ.get("TUNE7_ALPHA_DECAY_BURNIN_ROUNDS", "250"))
TUNE7_LOCAL_NEIGHBOR_RADIUS = int(os.environ.get("TUNE7_LOCAL_NEIGHBOR_RADIUS", "1"))
TUNE7_CANDIDATE_LOCAL_FRACTION = float(os.environ.get("TUNE7_CANDIDATE_LOCAL_FRACTION", "0.60"))
TUNE7_CANDIDATE_ELITE_FRACTION = float(os.environ.get("TUNE7_CANDIDATE_ELITE_FRACTION", "0.20"))

FINAL_TUNE_DIMS = tuple(
    sorted(
        {
            int(x.strip())
            for x in os.environ.get("FINAL_TUNE_DIMS", "3,5").split(",")
            if x.strip()
        }
    )
)
FINAL_BRANCH_FILTER = tuple(
    x.strip()
    for x in os.environ.get("FINAL_BRANCH_FILTER", "").split(",")
    if x.strip()
)
FINAL_INCLUDE_DEFAULT = os.environ.get("FINAL_INCLUDE_DEFAULT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
FINAL_SHARED_BRANCH_SEED = os.environ.get("FINAL_SHARED_BRANCH_SEED", "").strip()


def _rolling_success_scale_sec(
    success_runtime_history: deque[float],
    *,
    b_min_runtime_sec: float,
) -> float:
    vals = np.asarray(list(success_runtime_history), dtype=float)
    finite = vals[np.isfinite(vals) & (vals > 0.0)]
    if finite.size:
        return float(max(float(np.median(finite)), float(b_min_runtime_sec)))
    return float(b_min_runtime_sec)


def runtime_loss_sec(
    *,
    outcome: Dict[str, Any],
    fail_runtime_sec: float,
    solver_tol: float,
    success_runtime_scale_sec: float,
    failure_penalty_multiplier: float,
    failure_severity_cap: float,
    structural_failure_surcharge_multiplier: float,
    **_,
) -> float:
    rt = float(outcome["runtime"])
    if not np.isfinite(rt):
        return float(fail_runtime_sec)
    if bool(outcome.get("failed", False)) or rt >= 0.999 * float(fail_runtime_sec):
        res_norm = float(outcome.get("residual_norm", float("inf")))
        tol = max(float(solver_tol), 1e-300)
        if np.isfinite(res_norm):
            severity = max(0.0, float(np.log10(max(res_norm, tol) / tol)))
        else:
            severity = float(failure_severity_cap)
        severity = min(float(severity), float(failure_severity_cap))
        scale = max(float(success_runtime_scale_sec), 0.0)
        structural_fail = bool(
            bool(outcome.get("structural_fail", False))
            or not np.isfinite(float(outcome.get("residual_norm", float("inf"))))
            or not np.isfinite(float(outcome.get("runtime", float("inf"))))
        )
        surcharge = float(structural_failure_surcharge_multiplier) if structural_fail else 0.0
        return float(rt + scale * (float(failure_penalty_multiplier) + severity + surcharge))
    return rt


def _compute_failure_scale_min_runtime_sec(
    *,
    mkw: Dict[str, Any],
    solver_tol: float,
    solver_max_iter: int,
) -> float:
    if FAILURE_SCALE_MIN_RUNTIME_SEC > 0.0:
        return float(FAILURE_SCALE_MIN_RUNTIME_SEC)
    try:
        res = solve(
            params=DEFAULT_PARAMS,
            tol=float(solver_tol),
            max_iter=int(solver_max_iter),
            **mkw,
        )
        rt = float(res.runtime_sec)
        if np.isfinite(rt) and rt > 0.0:
            return float(rt)
    except Exception:
        pass
    return 1e-3


class FixedPolicy:
    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)

    def select(self, context, **_):
        return dict(self._params), {}

    def update(self, loss, **_):
        return None


class SharedFactory:
    def __init__(self, model_cls, alpha: float, l2: float, *, model_kwargs=None):
        self.model_cls = model_cls
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.model_kwargs = dict(model_kwargs or {})

    def new_trial(self, *, parameter_space, seed: int, **_):
        class _Policy:
            def __init__(self, actions, context_dim, alpha, l2, seed):
                self.m = model_cls(
                    actions,
                    context_dim=int(context_dim),
                    alpha=float(alpha),
                    l2_reg=float(l2),
                    seed=int(seed),
                    **model_kwargs,
                )

            def select(self, context, **_):
                params = self.m.predict(context)
                if getattr(self.m, "history", None):
                    step = self.m.history[-1]
                    info = {
                        "pred_mean": float(getattr(step, "pred_mean", np.nan)),
                        "pred_uncert": float(getattr(step, "pred_uncert", np.nan)),
                    }
                else:
                    info = {}
                return params, info

            def update(self, loss, **_):
                self.m.update(float(loss))

        model_cls = self.model_cls
        model_kwargs = self.model_kwargs
        return _Policy(
            parameter_space["actions"],
            parameter_space["context_dim"],
            self.alpha,
            self.l2,
            seed,
        )


class SharedFactoryV4:
    def __init__(self, alpha: float, l2: float, *, model_kwargs=None):
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.model_kwargs = dict(model_kwargs or {})

    def new_trial(self, *, parameter_space, seed: int, **_):
        class _Policy:
            def __init__(self, actions, context_dim, alpha, l2, seed):
                self.m = SharedLinUCB_AMG_v4(
                    actions,
                    context_dim=int(context_dim),
                    alpha=float(alpha),
                    l2_reg=float(l2),
                    seed=int(seed),
                    **model_kwargs,
                )

            def select(self, context, **_):
                params = self.m.predict(context)
                if getattr(self.m, "history", None):
                    step = self.m.history[-1]
                    info = {
                        "pred_mean": float(getattr(step, "pred_mean", np.nan)),
                        "pred_uncert": float(getattr(step, "pred_uncert", np.nan)),
                    }
                else:
                    info = {}
                return params, info

            def update(self, loss, **_):
                self.m.update(float(loss))

        model_kwargs = self.model_kwargs
        return _Policy(
            parameter_space["actions"],
            parameter_space["context_dim"],
            self.alpha,
            self.l2,
            seed,
        )


@dataclass(frozen=True)
class BranchRun:
    label: str
    family: str
    tune_set: str
    seed: int | None
    policy: Any
    parameter_space: Dict[str, Any]
    solver_tol: float
    solver_max_iter: int


def _same_action(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        np.isclose(float(a["strong_threshold"]), float(b["strong_threshold"]), rtol=0.0, atol=1e-12)
        and np.isclose(float(a["max_row_sum"]), float(b["max_row_sum"]), rtol=0.0, atol=1e-12)
        and np.isclose(float(a["trunc_factor"]), float(b["trunc_factor"]), rtol=0.0, atol=1e-12)
        and int(a["coarsen_type"]) == int(b["coarsen_type"])
        and int(a["interp_type"]) == int(b["interp_type"])
        and int(a["P_max_elmts"]) == int(b["P_max_elmts"])
        and int(a["agg_num_levels"]) == int(b["agg_num_levels"])
        and int(a["agg_interp_type"]) == int(b["agg_interp_type"])
        and np.isclose(float(a["agg_tr"]), float(b["agg_tr"]), rtol=0.0, atol=1e-12)
        and int(a["agg_Pmx"]) == int(b["agg_Pmx"])
    )


def _parse_int_list_env(name: str, default_values: Sequence[int]) -> List[int]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [int(v) for v in default_values]


def _parse_float_list_env(name: str, default_values: Sequence[float]) -> List[float]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [float(v) for v in default_values]


def _select_trace_keys(
    traces: Dict[str, Dict[str, np.ndarray]],
    keys: Sequence[str],
) -> List[str]:
    active: List[str] = []
    for key in keys:
        arrays = []
        for trace in traces.values():
            if key not in trace:
                continue
            arr = np.asarray(trace[key], dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                arrays.append(finite)
        if not arrays:
            continue
        data = np.concatenate(arrays, axis=0)
        uniq = np.unique(np.round(data, 12))
        if uniq.size > 1:
            active.append(str(key))
    return active


def _late_window_diagnostics(
    *,
    method_names: Sequence[str],
    traces: Dict[str, Dict[str, np.ndarray]],
    runtime_sec: Dict[str, np.ndarray],
    trace_keys: Sequence[str],
    windows: Sequence[int] = (50, 100, 200, 500),
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not trace_keys:
        return out

    for name in method_names:
        out[name] = {}
        trace = traces.get(name, {})
        if not trace:
            continue
        t_len = len(next(iter(trace.values()))) if trace else 0
        for window in windows:
            start = max(0, t_len - int(window))
            keys = [
                tuple(round(float(trace[k][idx]), 6) for k in trace_keys)
                for idx in range(start, t_len)
            ]
            if keys:
                from collections import Counter

                action, count = Counter(keys).most_common(1)[0]
                mode_action = {str(k): float(v) for k, v in zip(trace_keys, action)}
                unique_count = int(len(set(keys)))
            else:
                mode_action = {}
                count = 0
                unique_count = 0
            rt_tail = np.asarray(runtime_sec[name], dtype=float)[start:t_len]
            out[name][str(int(window))] = {
                "mean_runtime_sec": float(np.mean(rt_tail)) if rt_tail.size else float("nan"),
                "unique_action_count": int(unique_count),
                "mode_action_count": int(count),
                "mode_action": mode_action,
            }
    return out


def _late_window_flat_fields(
    late_window_summary: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    windows: Sequence[int] = (50, 100, 200, 500),
) -> Dict[str, Dict[str, float | int]]:
    flat: Dict[str, Dict[str, float | int]] = {}
    for window in windows:
        wkey = str(int(window))
        slope_key = f"slope_{int(window)}"
        mode_frac_key = f"mode_fraction_{int(window)}"
        unique_key = f"unique_action_count_{int(window)}"
        flat[slope_key] = {}
        flat[mode_frac_key] = {}
        flat[unique_key] = {}
        for method, per_window in late_window_summary.items():
            info = per_window.get(wkey, {})
            flat[slope_key][method] = float(info.get("mean_runtime_sec", float("nan")))
            mode_count = int(info.get("mode_action_count", 0))
            flat[mode_frac_key][method] = float(mode_count / float(window)) if int(window) > 0 else float("nan")
            flat[unique_key][method] = int(info.get("unique_action_count", 0))
    return flat


def _build_grids() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    grid_n = int(os.environ.get("GRID_N", "20"))
    grid_max = float(os.environ.get("GRID_MAX", "0.95"))
    th_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, grid_n)
    return grid_n, th_grid, mxrs_grid, tr_grid


def _build_actions_tune3(*, th_grid, mxrs_grid, tr_grid) -> List[Dict[str, Any]]:
    base_actions = build_actions_th_mxrs_tr(
        th_grid,
        mxrs_grid,
        tr_grid,
        fixed_params={
            "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
            "interp_type": DEFAULT_PARAMS["interp_type"],
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        params = dict(base)
        params["P_max_elmts"] = int(DEFAULT_PARAMS["P_max_elmts"])
        params["agg_num_levels"] = int(DEFAULT_PARAMS["agg_num_levels"])
        params["agg_interp_type"] = int(DEFAULT_PARAMS["agg_interp_type"])
        params["agg_tr"] = float(DEFAULT_PARAMS["agg_tr"])
        params["agg_Pmx"] = int(DEFAULT_PARAMS["agg_Pmx"])
        actions.append(params)
    return actions


def _build_actions_tune5(*, th_grid, mxrs_grid, tr_grid) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    p_max_values = _parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = _parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)

    base_actions = build_actions_th_mxrs_tr(
        th_grid,
        mxrs_grid,
        tr_grid,
        fixed_params={
            "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
            "interp_type": DEFAULT_PARAMS["interp_type"],
            "agg_interp_type": DEFAULT_PARAMS["agg_interp_type"],
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        for p_max in p_max_values:
            for agg_nl in agg_nl_values:
                params = dict(base)
                params["P_max_elmts"] = int(p_max)
                params["agg_num_levels"] = int(agg_nl)
                params["agg_interp_type"] = int(DEFAULT_PARAMS["agg_interp_type"])
                params["agg_tr"] = float(DEFAULT_PARAMS["agg_tr"])
                params["agg_Pmx"] = int(DEFAULT_PARAMS["agg_Pmx"])
                actions.append(params)
    return actions, p_max_values, agg_nl_values


def _build_actions_tune7_categorical(
    *,
    th_grid,
    mxrs_grid,
    tr_grid,
) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[int], List[int], ParameterSpaceSpec]:
    p_max_values = _parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = _parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)
    coarsen_type_values = _parse_int_list_env("COARSEN_TYPE_VALUES", DEFAULT_COARSEN_TYPE_VALUES)
    interp_values = _parse_int_list_env("TUNE7_INTERP_TYPES", DEFAULT_TUNE7_INTERP_TYPES)

    parameter_spec = ParameterSpaceSpec(
        (
            ParameterSpec(
                name="strong_threshold",
                kind="continuous",
                values=tuple(float(v) for v in th_grid),
                default=float(DEFAULT_PARAMS["strong_threshold"]),
                center=float(DEFAULT_PARAMS["strong_threshold"]),
                scale=0.25,
            ),
            ParameterSpec(
                name="max_row_sum",
                kind="continuous",
                values=tuple(float(v) for v in mxrs_grid),
                default=float(DEFAULT_PARAMS["max_row_sum"]),
                center=float(DEFAULT_PARAMS["max_row_sum"]),
                scale=0.10,
            ),
            ParameterSpec(
                name="trunc_factor",
                kind="continuous",
                values=tuple(float(v) for v in tr_grid),
                default=float(DEFAULT_PARAMS["trunc_factor"]),
                center=float(DEFAULT_PARAMS["trunc_factor"]),
                scale=0.20,
            ),
            ParameterSpec(
                name="P_max_elmts",
                kind="integer",
                values=tuple(int(v) for v in p_max_values),
                default=int(DEFAULT_PARAMS["P_max_elmts"]),
                center=float(DEFAULT_PARAMS["P_max_elmts"]),
                scale=4.0,
            ),
            ParameterSpec(
                name="agg_num_levels",
                kind="integer",
                values=tuple(int(v) for v in agg_nl_values),
                default=int(DEFAULT_PARAMS["agg_num_levels"]),
                center=float(DEFAULT_PARAMS["agg_num_levels"]),
                scale=1.0,
            ),
            ParameterSpec(
                name="coarsen_type",
                kind="categorical",
                values=tuple(int(v) for v in coarsen_type_values),
                default=int(DEFAULT_PARAMS["coarsen_type"]),
            ),
            ParameterSpec(
                name="interp_type",
                kind="categorical",
                values=tuple(int(v) for v in interp_values),
                default=int(DEFAULT_PARAMS["interp_type"]),
            ),
        )
    )

    actions = build_actions_from_spec(
        parameter_spec,
        fixed_params={
            "agg_interp_type": int(DEFAULT_PARAMS["agg_interp_type"]),
            "agg_tr": float(DEFAULT_PARAMS["agg_tr"]),
            "agg_Pmx": int(DEFAULT_PARAMS["agg_Pmx"]),
        },
    )
    return actions, p_max_values, agg_nl_values, coarsen_type_values, interp_values, parameter_spec


def _build_actions_tune7_agg_conditional(
    *,
    th_grid,
    mxrs_grid,
    tr_grid,
) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[float], List[int], ParameterSpaceSpec]:
    p_max_values = _parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = _parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)
    agg_tr_values = _parse_float_list_env("TUNE7_AGG_TR_VALUES", DEFAULT_AGG_TR_VALUES)
    agg_pmx_values = _parse_int_list_env("TUNE7_AGG_PMX_VALUES", DEFAULT_AGG_PMX_VALUES)

    active_agg_levels = tuple(int(v) for v in agg_nl_values if int(v) != int(DEFAULT_PARAMS["agg_num_levels"]))
    parameter_spec = ParameterSpaceSpec(
        (
            ParameterSpec(
                name="strong_threshold",
                kind="continuous",
                values=tuple(float(v) for v in th_grid),
                default=float(DEFAULT_PARAMS["strong_threshold"]),
                center=float(DEFAULT_PARAMS["strong_threshold"]),
                scale=0.25,
            ),
            ParameterSpec(
                name="max_row_sum",
                kind="continuous",
                values=tuple(float(v) for v in mxrs_grid),
                default=float(DEFAULT_PARAMS["max_row_sum"]),
                center=float(DEFAULT_PARAMS["max_row_sum"]),
                scale=0.10,
            ),
            ParameterSpec(
                name="trunc_factor",
                kind="continuous",
                values=tuple(float(v) for v in tr_grid),
                default=float(DEFAULT_PARAMS["trunc_factor"]),
                center=float(DEFAULT_PARAMS["trunc_factor"]),
                scale=0.20,
            ),
            ParameterSpec(
                name="P_max_elmts",
                kind="integer",
                values=tuple(int(v) for v in p_max_values),
                default=int(DEFAULT_PARAMS["P_max_elmts"]),
                center=float(DEFAULT_PARAMS["P_max_elmts"]),
                scale=4.0,
            ),
            ParameterSpec(
                name="agg_num_levels",
                kind="integer",
                values=tuple(int(v) for v in agg_nl_values),
                default=int(DEFAULT_PARAMS["agg_num_levels"]),
                center=float(DEFAULT_PARAMS["agg_num_levels"]),
                scale=1.0,
            ),
            ParameterSpec(
                name="agg_tr",
                kind="continuous",
                values=tuple(float(v) for v in agg_tr_values),
                default=float(DEFAULT_PARAMS["agg_tr"]),
                center=float(DEFAULT_PARAMS["agg_tr"]),
                scale=0.10,
                active_if={"agg_num_levels": active_agg_levels},
            ),
            ParameterSpec(
                name="agg_Pmx",
                kind="integer",
                values=tuple(int(v) for v in agg_pmx_values),
                default=int(DEFAULT_PARAMS["agg_Pmx"]),
                center=float(DEFAULT_PARAMS["agg_Pmx"]),
                scale=4.0,
                active_if={"agg_num_levels": active_agg_levels},
            ),
        )
    )

    actions = build_actions_from_spec(
        parameter_spec,
        fixed_params={
            "coarsen_type": int(DEFAULT_PARAMS["coarsen_type"]),
            "interp_type": int(DEFAULT_PARAMS["interp_type"]),
            "agg_interp_type": int(DEFAULT_PARAMS["agg_interp_type"]),
        },
    )
    return actions, p_max_values, agg_nl_values, agg_tr_values, agg_pmx_values, parameter_spec


def _ensure_default_arm(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    default_arm_index = next((i for i, a in enumerate(actions) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        return [*actions, dict(DEFAULT_PARAMS)], int(len(actions))
    return actions, int(default_arm_index)


def _generate_instances(*, T: int, seed: int, sampler_kwargs: Dict[str, Any]):
    rng = np.random.default_rng(seed)
    instances = []
    for t in range(int(T)):
        mkw, context, _meta = stencil_0_scalar_anisotropic_diffusion_rl(rng=rng, t=t, trial=0, **sampler_kwargs)
        instances.append((mkw, np.asarray(context, dtype=float)))
    return instances


def _safe_solve(
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    *,
    fail_runtime_sec: float,
    solver_tol: float,
    solver_max_iter: int,
) -> Dict[str, Any]:
    try:
        res = solve(params=params, tol=float(solver_tol), max_iter=int(solver_max_iter), **mkw)
        res_norm = float(res.residual_norm)
        iters = int(res.iterations)
        converged = bool(
            np.isfinite(res_norm)
            and res_norm <= float(solver_tol)
            and iters < int(solver_max_iter)
        )
        return {
            "runtime": float(res.runtime_sec),
            "failed": (not converged),
            "residual_norm": res_norm,
            "iterations": iters,
            "structural_fail": False,
        }
    except Exception:
        return {
            "runtime": float(fail_runtime_sec),
            "failed": True,
            "residual_norm": float("inf"),
            "iterations": int(solver_max_iter),
            "structural_fail": True,
        }


def _run_one_step(
    *,
    policy: Any,
    parameter_space: Dict[str, Any],
    context: np.ndarray,
    mkw: Dict[str, Any],
    prev_update_est: float,
    solver_tol: float,
    solver_max_iter: int,
    success_runtime_history: deque[float],
    b_min_runtime_sec: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, int]:
    fail_runtime_sec = 1e9
    total_runtime = 0.0
    total_overhead = 0.0
    failed_attempts = 0
    local_prev_update_est = float(prev_update_est)

    last_params: Dict[str, Any] | None = None
    last_out: Dict[str, Any] | None = None
    last_upd_sec = 0.0

    for _attempt in range(int(RETRY_MAX_ATTEMPTS)):
        sel_start = time.perf_counter_ns()
        selected = policy.select(context=context, parameter_space=parameter_space)
        sel_sec = (time.perf_counter_ns() - sel_start) / 1e9
        params = selected[0] if isinstance(selected, tuple) else selected

        out = _safe_solve(
            params,
            mkw,
            fail_runtime_sec=fail_runtime_sec,
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
        )
        total_runtime += float(out["runtime"])

        loss_start = time.perf_counter_ns()
        success_runtime_scale_sec = _rolling_success_scale_sec(
            success_runtime_history,
            b_min_runtime_sec=float(b_min_runtime_sec),
        )
        base_loss_sec = float(
            runtime_loss_sec(
                outcome=out,
                fail_runtime_sec=fail_runtime_sec,
                solver_tol=float(solver_tol),
                success_runtime_scale_sec=float(success_runtime_scale_sec),
                failure_penalty_multiplier=float(FAILURE_PENALTY_MULTIPLIER),
                failure_severity_cap=float(FAILURE_SEVERITY_CAP),
                structural_failure_surcharge_multiplier=float(STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER),
            )
        )
        loss_eval_sec = (time.perf_counter_ns() - loss_start) / 1e9
        end_to_end_loss_sec = base_loss_sec + float(sel_sec) + float(loss_eval_sec) + float(local_prev_update_est)

        upd_sec = 0.0
        if hasattr(policy, "update"):
            upd_start = time.perf_counter_ns()
            policy.update(loss=end_to_end_loss_sec, context=context, params=params, outcome=out)
            upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

        total_overhead += float(sel_sec + loss_eval_sec + upd_sec)
        last_params = dict(params)
        last_out = dict(out)
        last_upd_sec = float(upd_sec)
        local_prev_update_est = float(upd_sec)

        if not bool(out.get("failed", False)):
            rt_success = float(out["runtime"])
            if np.isfinite(rt_success) and rt_success > 0.0:
                success_runtime_history.append(float(rt_success))
            final_out = dict(out)
            final_out["runtime"] = float(total_runtime)
            final_out["failed"] = False
            return last_params, final_out, float(total_overhead), float(last_upd_sec), int(failed_attempts)

        failed_attempts += 1

    final_out = dict(
        last_out
        or {
            "runtime": fail_runtime_sec,
            "failed": True,
            "residual_norm": float("inf"),
            "iterations": int(solver_max_iter),
        }
    )
    final_out["runtime"] = float(total_runtime if total_runtime > 0.0 else fail_runtime_sec)
    final_out["failed"] = True
    return dict(last_params or DEFAULT_PARAMS), final_out, float(total_overhead), float(last_upd_sec), int(failed_attempts)


def _save_per_instance_csv(
    *,
    out_csv: Path,
    branches: Sequence[BranchRun],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    failed_flags: Dict[str, np.ndarray],
    retry_counts: Dict[str, np.ndarray],
    traces: Dict[str, Dict[str, np.ndarray]],
) -> None:
    fieldnames = [
        "t",
        "method",
        "family",
        "tune_set",
        "seed",
        "nx",
        "ny",
        "nz",
        "k",
        "c",
        "a0",
        "a1",
        "a2",
        "a3",
        "runtime_sec",
        "overhead_sec",
        "end_to_end_sec",
        "failed",
        "retry_count",
        "attempt_count",
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
    branch_map = {branch.label: branch for branch in branches}

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for t, (mkw, _context) in enumerate(instances):
            for label, trace in traces.items():
                branch = branch_map[label]
                writer.writerow(
                    {
                        "t": int(t + 1),
                        "method": str(label),
                        "family": str(branch.family),
                        "tune_set": str(branch.tune_set),
                        "seed": (int(branch.seed) if branch.seed is not None else ""),
                        "nx": int(mkw["nx"]),
                        "ny": int(mkw["ny"]),
                        "nz": int(mkw["nz"]),
                        "k": float(mkw["k"]),
                        "c": float(mkw["c"]),
                        "a0": float(mkw["a0"]),
                        "a1": float(mkw["a1"]),
                        "a2": float(mkw["a2"]),
                        "a3": float(mkw["a3"]),
                        "runtime_sec": float(runtime_sec[label][t]),
                        "overhead_sec": float(overhead_sec[label][t]),
                        "end_to_end_sec": float(runtime_sec[label][t] + overhead_sec[label][t]),
                        "failed": int(bool(failed_flags[label][t])),
                        "retry_count": int(retry_counts[label][t]),
                        "attempt_count": int(retry_counts[label][t] + 1),
                        "strong_threshold": float(trace["strong_threshold"][t]),
                        "max_row_sum": float(trace["max_row_sum"][t]),
                        "trunc_factor": float(trace["trunc_factor"][t]),
                        "coarsen_type": float(trace["coarsen_type"][t]),
                        "P_max_elmts": float(trace["P_max_elmts"][t]),
                        "agg_num_levels": float(trace["agg_num_levels"][t]),
                        "interp_type": float(trace["interp_type"][t]),
                        "agg_interp_type": float(trace["agg_interp_type"][t]),
                        "agg_tr": float(trace["agg_tr"][t]),
                        "agg_Pmx": float(trace["agg_Pmx"][t]),
                    }
                )


def main() -> None:
    sampler_kwargs = {
        "nx": int(FIXED_NX),
        "ny": int(FIXED_NY),
        "nz": int(FIXED_NZ),
        "n_min": int(max(FIXED_NX, FIXED_NY, FIXED_NZ)),
        "n_max": int(max(FIXED_NX, FIXED_NY, FIXED_NZ)),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
    }

    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_0_scalar_anisotropic_diffusion_rl(rng=warm_rng, t=0, trial=0, **sampler_kwargs)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    grid_n, th_grid, mxrs_grid, tr_grid = _build_grids()
    actions_tune3 = _build_actions_tune3(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune3, default_arm_index_tune3 = _ensure_default_arm(actions_tune3)
    actions_tune5 = []
    p_max_values: List[int] = []
    agg_nl_values: List[int] = []
    default_arm_index_tune5 = -1
    if 5 in FINAL_TUNE_DIMS:
        actions_tune5, p_max_values, agg_nl_values = _build_actions_tune5(
            th_grid=th_grid,
            mxrs_grid=mxrs_grid,
            tr_grid=tr_grid,
        )
        actions_tune5, default_arm_index_tune5 = _ensure_default_arm(actions_tune5)

    actions_tune7 = []
    coarsen_type_values_tune7: List[int] = []
    interp_values_tune7: List[int] = []
    agg_interp_values_tune7: List[int] = []
    agg_tr_values_tune7: List[float] = []
    agg_pmx_values_tune7: List[int] = []
    parameter_spec_tune7 = None
    default_arm_index_tune7 = -1
    if 7 in FINAL_TUNE_DIMS:
        if TUNE7_VARIANT == "categorical":
            (
                actions_tune7,
                p_max_values,
                agg_nl_values,
                coarsen_type_values_tune7,
                interp_values_tune7,
                parameter_spec_tune7,
            ) = _build_actions_tune7_categorical(
                th_grid=th_grid,
                mxrs_grid=mxrs_grid,
                tr_grid=tr_grid,
            )
        elif TUNE7_VARIANT == "agg_conditional":
            (
                actions_tune7,
                p_max_values,
                agg_nl_values,
                agg_tr_values_tune7,
                agg_pmx_values_tune7,
                parameter_spec_tune7,
            ) = _build_actions_tune7_agg_conditional(
                th_grid=th_grid,
                mxrs_grid=mxrs_grid,
                tr_grid=tr_grid,
            )
        else:
            raise RuntimeError(f"Unsupported TUNE7_VARIANT: {TUNE7_VARIANT}")
        actions_tune7, default_arm_index_tune7 = _ensure_default_arm(actions_tune7)

    instances = _generate_instances(T=T, seed=SEED, sampler_kwargs=sampler_kwargs)

    parameter_space_tune3 = {"actions": actions_tune3, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM}
    parameter_space_tune5 = {"actions": actions_tune5, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM} if actions_tune5 else None
    parameter_space_tune7 = {"actions": actions_tune7, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM} if actions_tune7 else None

    seed_map_by_family = {
        "Shared LinUCB v2": int(SEED + 11003),
        "Shared LinUCB v3": int(SEED + 12003),
        "Shared LinUCB v4": int(SEED + 13003),
    }
    if FINAL_SHARED_BRANCH_SEED:
        shared_branch_seed = int(FINAL_SHARED_BRANCH_SEED)
        seed_map_by_family = {k: shared_branch_seed for k in seed_map_by_family}

    family_specs = [
        ("Shared LinUCB v2", SharedLinUCB_AMG_v2),
        ("Shared LinUCB v3", SharedLinUCB_AMG_v3),
    ]

    branches: List[BranchRun] = []
    if FINAL_INCLUDE_DEFAULT:
        branches.append(
            BranchRun(
                label="default (fixed)",
                family="default",
                tune_set="default",
                seed=None,
                policy=FixedPolicy(DEFAULT_PARAMS),
                parameter_space=parameter_space_tune3,
                solver_tol=float(SOLVER_TOL),
                solver_max_iter=int(SOLVER_MAX_ITER),
            )
        )

    if FINAL_COMPARE_V2_TUNE3_SOLVER_PROFILES:
        branches = []
        family_name = "Shared LinUCB v2"
        model_cls = SharedLinUCB_AMG_v2
        family_seed = int(seed_map_by_family[family_name])
        compare_specs = [
            (
                "Shared LinUCB v2 | tune3 | tol=1e-8,max_iter=10000",
                float(LEGACY_SOLVER_TOL),
                int(LEGACY_SOLVER_MAX_ITER),
            ),
            (
                "Shared LinUCB v2 | tune3 | tol=1e-6,max_iter=50",
                float(SOLVER_TOL),
                int(SOLVER_MAX_ITER),
            ),
        ]
        for label, branch_solver_tol, branch_solver_max_iter in compare_specs:
            policy = SharedFactory(
                model_cls,
                ALPHA,
                L2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index_tune3)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space_tune3, seed=family_seed, T=T, trial=0)
            branches.append(
                BranchRun(
                    label=label,
                    family=family_name,
                    tune_set="tune3",
                    seed=family_seed,
                    policy=policy,
                    parameter_space=parameter_space_tune3,
                    solver_tol=float(branch_solver_tol),
                    solver_max_iter=int(branch_solver_max_iter),
                )
            )
    else:
        for family_name, model_cls in family_specs:
            family_seed = int(seed_map_by_family[family_name])
            tune_specs = [("tune3", parameter_space_tune3, int(default_arm_index_tune3))]
            if 5 in FINAL_TUNE_DIMS and parameter_space_tune5 is not None:
                tune_specs.append(("tune5", parameter_space_tune5, int(default_arm_index_tune5)))
            for tune_set, parameter_space, default_arm_index in tune_specs:
                policy = SharedFactory(
                    model_cls,
                    ALPHA,
                    L2,
                    model_kwargs={
                        "action_center": DEFAULT_PARAMS,
                        "alpha_decay": True,
                        "candidate_pool_size": CANDIDATE_POOL_SIZE,
                        "always_include_arms": [int(default_arm_index)],
                        "elite_cache_size": ELITE_CACHE_SIZE,
                    },
                ).new_trial(parameter_space=parameter_space, seed=family_seed, T=T, trial=0)
                branches.append(
                    BranchRun(
                        label=f"{family_name} | {tune_set}",
                        family=family_name,
                        tune_set=tune_set,
                        seed=family_seed,
                        policy=policy,
                        parameter_space=parameter_space,
                        solver_tol=float(SOLVER_TOL),
                        solver_max_iter=int(SOLVER_MAX_ITER),
                    )
                )

    if 7 in FINAL_TUNE_DIMS and parameter_space_tune7 is not None and parameter_spec_tune7 is not None:
        family_name = "Shared LinUCB v4"
        family_seed = int(seed_map_by_family[family_name])
        tune7_label = f"{family_name} | tune7" if TUNE7_VARIANT == "agg_conditional" else f"{family_name} | tune7-categorical"
        tune7_model_kwargs: Dict[str, Any] = {
            "parameter_spec": parameter_spec_tune7,
            "context_interaction_indices": (1, 2, 3, 4),
            "always_include_arms": [int(default_arm_index_tune7)],
            "elite_cache_size": ELITE_CACHE_SIZE,
            "initial_guess": [DEFAULT_PARAMS[param.name] for param in parameter_spec_tune7.parameters],
            "initial_guess_rounds": 1,
        }
        tune7_candidate_strategy = str(TUNE7_CANDIDATE_STRATEGY)
        if not tune7_candidate_strategy:
            tune7_candidate_strategy = "adaptive_local" if TUNE7_VARIANT == "agg_conditional" else "uniform"
        if tune7_candidate_strategy == "adaptive_local":
            tune7_model_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": int(TUNE7_CANDIDATE_POOL_SIZE),
                    "candidate_strategy": "adaptive_local",
                    "candidate_pool_size_burnin": int(TUNE7_CANDIDATE_POOL_SIZE_BURNIN),
                    "candidate_pool_burnin_rounds": int(TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS),
                    "alpha_decay_burnin_rounds": int(TUNE7_ALPHA_DECAY_BURNIN_ROUNDS),
                    "elite_rank_metric": "mean_loss",
                    "local_neighbor_radius": int(TUNE7_LOCAL_NEIGHBOR_RADIUS),
                    "candidate_local_fraction": float(TUNE7_CANDIDATE_LOCAL_FRACTION),
                    "candidate_elite_fraction": float(TUNE7_CANDIDATE_ELITE_FRACTION),
                }
            )
        else:
            tune7_model_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                }
            )
        policy = SharedFactoryV4(
            (float(TUNE7_ALPHA_OVERRIDE) if TUNE7_ALPHA_OVERRIDE else ALPHA),
            L2,
            model_kwargs=tune7_model_kwargs,
        ).new_trial(parameter_space=parameter_space_tune7, seed=family_seed, T=T, trial=0)
        branches.append(
            BranchRun(
                label=tune7_label,
                family=family_name,
                tune_set="tune7",
                seed=family_seed,
                policy=policy,
                parameter_space=parameter_space_tune7,
                solver_tol=float(SOLVER_TOL),
                solver_max_iter=int(SOLVER_MAX_ITER),
            )
        )

    if FINAL_BRANCH_FILTER:
        keep = set(FINAL_BRANCH_FILTER)
        branches = [branch for branch in branches if branch.label in keep]

    if not branches:
        raise RuntimeError("No branches selected for test_final.py")

    branch_failure_scale_min: Dict[str, float] = {}
    profile_failure_scale_cache: Dict[Tuple[float, int], float] = {}
    for branch in branches:
        profile_key = (float(branch.solver_tol), int(branch.solver_max_iter))
        if profile_key not in profile_failure_scale_cache:
            profile_failure_scale_cache[profile_key] = _compute_failure_scale_min_runtime_sec(
                mkw=warm_mkw,
                solver_tol=float(branch.solver_tol),
                solver_max_iter=int(branch.solver_max_iter),
            )
        branch_failure_scale_min[branch.label] = float(profile_failure_scale_cache[profile_key])

    run_dir = create_run_output_dir(
        base_dir=plots_base_dir,
        script_name=Path(__file__).stem,
        problem_name="scalar_anisotropic_diffusion",
        size_tag=f"{FIXED_NX}x{FIXED_NY}x{FIXED_NZ}",
        T=T,
        seed=SEED,
    )

    permutation_seed = int(SEED ^ 0x1A2B3C4D)
    rng_order = np.random.default_rng(permutation_seed)
    print("Within-step permutation over all branches: enabled")

    branch_labels = [branch.label for branch in branches]
    runtime_sec = {label: np.zeros(T, dtype=float) for label in branch_labels}
    overhead_sec = {label: np.zeros(T, dtype=float) for label in branch_labels}
    failed_flags = {label: np.zeros(T, dtype=bool) for label in branch_labels}
    retry_counts = {label: np.zeros(T, dtype=int) for label in branch_labels}
    traces = {label: init_param_trace(TRACE_KEYS_FINAL, T) for label in branch_labels}
    prev_update_est = {label: 0.0 for label in branch_labels}
    success_runtime_history = {
        label: deque(maxlen=int(FAILURE_SCALE_WINDOW))
        for label in branch_labels
    }

    print(f"Single run: unified tuning non-RL comparison, T={T}")
    for t, (mkw, context) in enumerate(instances):
        order = rng_order.permutation(len(branches))
        for i in order:
            branch = branches[int(i)]
            params, out, step_overhead, upd_sec, retry_count = _run_one_step(
                policy=branch.policy,
                parameter_space=branch.parameter_space,
                context=context,
                mkw=mkw,
                prev_update_est=float(prev_update_est[branch.label]),
                solver_tol=float(branch.solver_tol),
                solver_max_iter=int(branch.solver_max_iter),
                success_runtime_history=success_runtime_history[branch.label],
                b_min_runtime_sec=float(branch_failure_scale_min[branch.label]),
            )
            runtime_sec[branch.label][t] = float(out["runtime"])
            overhead_sec[branch.label][t] = float(step_overhead)
            failed_flags[branch.label][t] = bool(out.get("failed", False))
            retry_counts[branch.label][t] = int(retry_count)
            record_param_trace(traces[branch.label], t=t, params=params, keys=TRACE_KEYS_FINAL)
            prev_update_est[branch.label] = float(upd_sec)

        progress_bar(t + 1, T, prefix="  final run")

    active_trace_keys = _select_trace_keys(traces, TRACE_KEYS_FINAL)
    late_window_summary = _late_window_diagnostics(
        method_names=branch_labels,
        traces=traces,
        runtime_sec=runtime_sec,
        trace_keys=active_trace_keys,
    )
    late_window_flat = _late_window_flat_fields(late_window_summary)

    summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="final_unified_tuning_runtime",
        method_names=branch_labels,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        T=T,
        title=(
            f"BoomerAMG setup cumulative runtime (final unified tuning, non-RL)  "
            f"T={T}  n={FIXED_NX}x{FIXED_NY}x{FIXED_NZ}  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name=("default (fixed)" if FINAL_INCLUDE_DEFAULT else branch_labels[0]),
        trace_keys=(),
        default_params=DEFAULT_PARAMS,
        summary_extra={
            "script": Path(__file__).name,
            "seed": int(SEED),
            "alpha": float(ALPHA),
            "l2": float(L2),
            "fixed_n": int(FIXED_N) if FIXED_NX == FIXED_NY == FIXED_NZ == FIXED_N else None,
            "fixed_nx": int(FIXED_NX),
            "fixed_ny": int(FIXED_NY),
            "fixed_nz": int(FIXED_NZ),
            "c_min": float(C_MIN),
            "c_max": float(C_MAX),
        "context_dim": int(SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM),
            "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(ELITE_CACHE_SIZE),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "legacy_solver_tol": float(LEGACY_SOLVER_TOL),
            "legacy_solver_max_iter": int(LEGACY_SOLVER_MAX_ITER),
            "compare_v2_tune3_solver_profiles": bool(FINAL_COMPARE_V2_TUNE3_SOLVER_PROFILES),
            "failure_penalty_multiplier": float(FAILURE_PENALTY_MULTIPLIER),
            "failure_scale_window": int(FAILURE_SCALE_WINDOW),
            "failure_severity_cap": float(FAILURE_SEVERITY_CAP),
            "failure_scale_min_runtime_sec_override": float(FAILURE_SCALE_MIN_RUNTIME_SEC),
            "structural_failure_surcharge_multiplier": float(STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER),
            "shared_instance_stream": True,
            "within_step_branch_permutation": True,
            "permutation_seed": int(permutation_seed),
            "seed_map_by_family": {k: int(v) for k, v in seed_map_by_family.items()},
            "branch_seed_map": {branch.label: int(branch.seed) for branch in branches if branch.seed is not None},
            "branch_solver_profile_map": {
                branch.label: {
                    "solver_tol": float(branch.solver_tol),
                    "solver_max_iter": int(branch.solver_max_iter),
                }
                for branch in branches
            },
            "branch_failure_scale_min_runtime_sec": {
                branch.label: float(branch_failure_scale_min[branch.label])
                for branch in branches
            },
            "tune_dimensions": [int(v) for v in FINAL_TUNE_DIMS],
            "default_baseline": ("default (fixed)" if FINAL_INCLUDE_DEFAULT else None),
            "actions_count_tune3": int(len(actions_tune3)),
            "actions_count_tune5": int(len(actions_tune5)),
            "actions_count_tune7": int(len(actions_tune7)),
            "grid_n": int(grid_n),
            "coarsen_type_values": [int(v) for v in DEFAULT_COARSEN_TYPE_VALUES],
            "p_max_values": [int(v) for v in p_max_values],
            "agg_num_levels_values": [int(v) for v in agg_nl_values],
            "tune7_variant": str(TUNE7_VARIANT),
            "tune7_alpha": (float(TUNE7_ALPHA_OVERRIDE) if TUNE7_ALPHA_OVERRIDE else float(ALPHA)),
            "tune7_candidate_strategy": (
                str(TUNE7_CANDIDATE_STRATEGY)
                if TUNE7_CANDIDATE_STRATEGY
                else ("adaptive_local" if TUNE7_VARIANT == "agg_conditional" else "uniform")
            ),
            "tune7_coarsen_types": [int(v) for v in coarsen_type_values_tune7],
            "tune7_interp_types": [int(v) for v in interp_values_tune7],
            "tune7_agg_interp_types": [int(v) for v in agg_interp_values_tune7],
            "tune7_agg_tr_values": [float(v) for v in agg_tr_values_tune7],
            "tune7_agg_pmx_values": [int(v) for v in agg_pmx_values_tune7],
            "tune7_candidate_pool_size": int(TUNE7_CANDIDATE_POOL_SIZE),
            "tune7_candidate_pool_size_burnin": int(TUNE7_CANDIDATE_POOL_SIZE_BURNIN),
            "tune7_candidate_pool_burnin_rounds": int(TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS),
            "tune7_alpha_decay_burnin_rounds": int(TUNE7_ALPHA_DECAY_BURNIN_ROUNDS),
            "tune7_local_neighbor_radius": int(TUNE7_LOCAL_NEIGHBOR_RADIUS),
            "tune7_candidate_local_fraction": float(TUNE7_CANDIDATE_LOCAL_FRACTION),
            "tune7_candidate_elite_fraction": float(TUNE7_CANDIDATE_ELITE_FRACTION),
            "branch_filter": list(FINAL_BRANCH_FILTER),
            "failed_count_total": {k: int(np.sum(v.astype(int))) for k, v in failed_flags.items()},
            "retry_count_total": {k: int(np.sum(v.astype(int))) for k, v in retry_counts.items()},
            "retry_event_count_total": {k: int(np.sum(v > 0)) for k, v in retry_counts.items()},
            "mean_retry_count": {k: float(np.mean(v)) for k, v in retry_counts.items()},
            "mean_test_problem_runtime_sec": {k: float(np.mean(v)) for k, v in runtime_sec.items()},
            "late_window_summary": late_window_summary,
            **late_window_flat,
        },
    )

    data_csv_path = run_dir / "per_instance_runtime_data.csv"
    _save_per_instance_csv(
        out_csv=data_csv_path,
        branches=branches,
        instances=instances,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        failed_flags=failed_flags,
        retry_counts=retry_counts,
        traces=traces,
    )

    print("RUNTIME PLOT:", summary["plot"])
    print("RUNTIME SUMMARY:", summary["summary_path"])
    print("DATA CSV:", data_csv_path)


if __name__ == "__main__":
    main()
