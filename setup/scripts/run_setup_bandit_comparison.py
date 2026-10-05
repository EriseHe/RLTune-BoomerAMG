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

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from setup.learners import SharedLinUCB_AMG_v2
from setup.learners import SharedLinUCB_AMG_v3
from setup.learners import SharedLinUCB_AMG_v4
from setup.learners.linucb import run_same_context_setup_reselection
from setup.learners.common import ParameterSpaceSpec, ParameterSpec
from setup.learners.common import resolve_tune7_candidate_strategy
from hypre.bindings import SolveStatus, solve
from setup.utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
from problems.scalar_anisotropic_diffusion import (
    SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM,
    stencil_0_scalar_anisotropic_diffusion_rl,
)
from setup.utils.setup_amg import (
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
LEGACY_SOLVER_TOL = float(os.environ.get("LEGACY_SOLVER_TOL", "1e-8"))
LEGACY_SOLVER_MAX_ITER = int(os.environ.get("LEGACY_SOLVER_MAX_ITER", "10000"))
FINAL_COMPARE_V2_TUNE3_SOLVER_PROFILES = os.environ.get(
    "FINAL_COMPARE_V2_TUNE3_SOLVER_PROFILES", ""
).strip().lower() in {"1", "true", "yes", "on"}

plots_base_dir = REPO_ROOT / "results" / "setup"
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


def _solve_once(
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    *,
    solver_tol: float,
    solver_max_iter: int,
) -> Dict[str, Any]:
    res = solve(
        params=params,
        tol=float(solver_tol),
        max_iter=int(solver_max_iter),
        **mkw,
    )
    res_norm = float(res.residual_norm)
    iters = int(res.iterations)
    converged = bool(res.status is SolveStatus.CONVERGED)
    return {
        "runtime": float(res.runtime_sec),
        "setup_runtime": float(res.setup_runtime_sec),
        "solve_runtime": float(res.solve_runtime_sec),
        "infer_runtime": 0.0,
        "failed": not converged,
        "failure_reason": "" if converged else "max_iter_reached_without_convergence",
        "attempt_status": "success" if converged else "nonconvergence",
        "native_status": res.status.name.lower(),
        "failure_stage": "" if converged else "solve",
        "residual_norm": res_norm,
        "iterations": iters,
        "structural_fail": False,
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
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float, int]:
    result = run_same_context_setup_reselection(
        policy=policy,
        parameter_space=parameter_space,
        context=np.asarray(context, dtype=float),
        solver_fn=lambda params: _solve_once(
            dict(params),
            dict(mkw),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
        ),
        fallback_solver_fn=lambda _params: _solve_once(
            dict(DEFAULT_PARAMS),
            dict(mkw),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
        ),
        default_params=DEFAULT_PARAMS,
        prev_update_est=float(prev_update_est),
        primary_is_default=isinstance(policy, FixedPolicy),
        max_learned_attempts=3,
    )
    return (
        dict(result.params),
        dict(result.outcome),
        float(result.timing["overhead_sec"]),
        float(result.update_runtime_sec),
        int(result.fallback_used),
    )


def _save_per_instance_csv(
    *,
    out_csv: Path,
    branches: Sequence[BranchRun],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    failed_flags: Dict[str, np.ndarray],
    fallback_counts: Dict[str, np.ndarray],
    outcomes: Dict[str, Sequence[Dict[str, Any]]],
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
        "fallback_used",
        "attempt_count",
        "primary_status",
        "primary_failure_reason",
        "primary_setup_runtime_sec",
        "primary_solve_runtime_sec",
        "primary_residual_norm",
        "primary_cycles",
        "fallback_status",
        "fallback_setup_runtime_sec",
        "fallback_solve_runtime_sec",
        "recovered",
        "unrecovered_failure",
        "bandit_update_committed",
        "controller_update_committed",
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
                outcome = outcomes[label][t]
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
                        "fallback_used": bool(fallback_counts[label][t]),
                        "attempt_count": int(
                            outcome.get("primary_attempt_count", 1)
                            + fallback_counts[label][t]
                        ),
                        "primary_status": str(outcome.get("primary_status", "success")),
                        "primary_failure_reason": str(
                            outcome.get("primary_failure_reason", "")
                        ),
                        "primary_setup_runtime_sec": float(
                            outcome.get("primary_setup_runtime", 0.0)
                        ),
                        "primary_solve_runtime_sec": float(
                            outcome.get("primary_solve_runtime", 0.0)
                        ),
                        "primary_residual_norm": float(
                            outcome.get("primary_residual_norm", float("nan"))
                        ),
                        "primary_cycles": int(outcome.get("primary_cycles", 0)),
                        "fallback_status": str(
                            outcome.get("fallback_status", "not_run")
                        ),
                        "fallback_setup_runtime_sec": float(
                            outcome.get("fallback_setup_runtime", 0.0)
                        ),
                        "fallback_solve_runtime_sec": float(
                            outcome.get("fallback_solve_runtime", 0.0)
                        ),
                        "recovered": bool(outcome.get("recovered", False)),
                        "unrecovered_failure": bool(
                            outcome.get("unrecovered_failure", False)
                        ),
                        "bandit_update_committed": bool(
                            outcome.get("bandit_update_committed", False)
                        ),
                        "controller_update_committed": bool(
                            outcome.get("controller_update_committed", False)
                        ),
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


@dataclass(frozen=True)
class _ComparisonActions:
    grid_n: int
    actions_tune3: List[Dict[str, Any]]
    actions_tune5: List[Dict[str, Any]]
    actions_tune7: List[Dict[str, Any]]
    default_arm_index_tune3: int
    default_arm_index_tune5: int
    default_arm_index_tune7: int
    p_max_values: List[int]
    agg_nl_values: List[int]
    coarsen_type_values_tune7: List[int]
    interp_values_tune7: List[int]
    agg_interp_values_tune7: List[int]
    agg_tr_values_tune7: List[float]
    agg_pmx_values_tune7: List[int]
    parameter_spec_tune7: ParameterSpaceSpec | None


@dataclass(frozen=True)
class _ComparisonResults:
    permutation_seed: int
    branch_labels: List[str]
    runtime_sec: Dict[str, np.ndarray]
    overhead_sec: Dict[str, np.ndarray]
    failed_flags: Dict[str, np.ndarray]
    fallback_counts: Dict[str, np.ndarray]
    outcomes: Dict[str, List[Dict[str, Any]]]
    traces: Dict[str, Dict[str, np.ndarray]]


def _build_comparison_actions():
    """Build the requested action grids and retain reporting metadata."""
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

    return _ComparisonActions(
        grid_n=grid_n,
        actions_tune3=actions_tune3,
        actions_tune5=actions_tune5,
        actions_tune7=actions_tune7,
        default_arm_index_tune3=default_arm_index_tune3,
        default_arm_index_tune5=default_arm_index_tune5,
        default_arm_index_tune7=default_arm_index_tune7,
        p_max_values=p_max_values,
        agg_nl_values=agg_nl_values,
        coarsen_type_values_tune7=coarsen_type_values_tune7,
        interp_values_tune7=interp_values_tune7,
        agg_interp_values_tune7=agg_interp_values_tune7,
        agg_tr_values_tune7=agg_tr_values_tune7,
        agg_pmx_values_tune7=agg_pmx_values_tune7,
        parameter_spec_tune7=parameter_spec_tune7,
    )


def _prepare_comparison_stream():
    """Warm the native solver, then prepare actions and the paired stream."""
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

    actions = _build_comparison_actions()
    instances = _generate_instances(T=T, seed=SEED, sampler_kwargs=sampler_kwargs)

    return actions, instances


def _build_comparison_branches(actions: _ComparisonActions):
    """Construct independent branches using the prescribed family seeds."""
    parameter_space_tune3 = {"actions": actions.actions_tune3, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM}
    parameter_space_tune5 = {"actions": actions.actions_tune5, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM} if actions.actions_tune5 else None
    parameter_space_tune7 = {"actions": actions.actions_tune7, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM} if actions.actions_tune7 else None

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
                    "always_include_arms": [int(actions.default_arm_index_tune3)],
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
            tune_specs = [("tune3", parameter_space_tune3, int(actions.default_arm_index_tune3))]
            if 5 in FINAL_TUNE_DIMS and parameter_space_tune5 is not None:
                tune_specs.append(("tune5", parameter_space_tune5, int(actions.default_arm_index_tune5)))
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

    if 7 in FINAL_TUNE_DIMS and parameter_space_tune7 is not None and actions.parameter_spec_tune7 is not None:
        family_name = "Shared LinUCB v4"
        family_seed = int(seed_map_by_family[family_name])
        tune7_label = f"{family_name} | tune7" if TUNE7_VARIANT == "agg_conditional" else f"{family_name} | tune7-categorical"
        tune7_model_kwargs: Dict[str, Any] = {
            "parameter_spec": actions.parameter_spec_tune7,
            "context_interaction_indices": (1, 2, 3, 4),
            "always_include_arms": [int(actions.default_arm_index_tune7)],
            "elite_cache_size": ELITE_CACHE_SIZE,
            "initial_guess": [DEFAULT_PARAMS[param.name] for param in actions.parameter_spec_tune7.parameters],
            "initial_guess_rounds": 1,
        }
        tune7_candidate_strategy = resolve_tune7_candidate_strategy(
            tune7_variant=TUNE7_VARIANT,
            configured_strategy=TUNE7_CANDIDATE_STRATEGY,
        )
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
        raise RuntimeError("No branches selected for run_setup_bandit_comparison.py")

    return branches, seed_map_by_family


def _evaluate_comparison(branches: Sequence[BranchRun], instances):
    """Execute paired instances with the original within-step permutation."""
    permutation_seed = int(SEED ^ 0x1A2B3C4D)
    rng_order = np.random.default_rng(permutation_seed)
    print("Within-step permutation over all branches: enabled")

    branch_labels = [branch.label for branch in branches]
    runtime_sec = {label: np.zeros(T, dtype=float) for label in branch_labels}
    overhead_sec = {label: np.zeros(T, dtype=float) for label in branch_labels}
    failed_flags = {label: np.zeros(T, dtype=bool) for label in branch_labels}
    fallback_counts = {label: np.zeros(T, dtype=int) for label in branch_labels}
    outcomes: Dict[str, List[Dict[str, Any]]] = {
        label: [] for label in branch_labels
    }
    traces = {label: init_param_trace(TRACE_KEYS_FINAL, T) for label in branch_labels}
    prev_update_est = {label: 0.0 for label in branch_labels}
    print(f"Single run: unified tuning non-RL comparison, T={T}")
    for t, (mkw, context) in enumerate(instances):
        order = rng_order.permutation(len(branches))
        for i in order:
            branch = branches[int(i)]
            params, out, step_overhead, upd_sec, fallback_used = _run_one_step(
                policy=branch.policy,
                parameter_space=branch.parameter_space,
                context=context,
                mkw=mkw,
                prev_update_est=float(prev_update_est[branch.label]),
                solver_tol=float(branch.solver_tol),
                solver_max_iter=int(branch.solver_max_iter),
            )
            runtime_sec[branch.label][t] = float(out["runtime"])
            overhead_sec[branch.label][t] = float(step_overhead)
            failed_flags[branch.label][t] = bool(out.get("failed", False))
            fallback_counts[branch.label][t] = int(fallback_used)
            outcomes[branch.label].append(dict(out))
            record_param_trace(traces[branch.label], t=t, params=params, keys=TRACE_KEYS_FINAL)
            prev_update_est[branch.label] = float(upd_sec)

        progress_bar(t + 1, T, prefix="  final run")

    return _ComparisonResults(
        permutation_seed=permutation_seed,
        branch_labels=branch_labels,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        failed_flags=failed_flags,
        fallback_counts=fallback_counts,
        outcomes=outcomes,
        traces=traces,
    )


def _audit_comparison_recovery(branches: Sequence[BranchRun], results: _ComparisonResults):
    """Verify committed feedback and summarize attempts and recovery."""
    recovery_audit: Dict[str, Dict[str, int]] = {}
    for branch in branches:
        branch_outcomes = results.outcomes[branch.label]
        update_count = sum(
            bool(outcome.get("bandit_update_committed", False))
            for outcome in branch_outcomes
        )
        unrecovered_count = sum(
            bool(outcome.get("unrecovered_failure", False))
            for outcome in branch_outcomes
        )
        if (
            not isinstance(branch.policy, FixedPolicy)
            and update_count + unrecovered_count != T
        ):
            raise AssertionError(
                f"{branch.label}: updates plus unrecovered failures must equal {T}"
            )
        recovery_audit[branch.label] = {
            "processed_instances": int(T),
            "primary_selections": int(
                sum(
                    int(outcome.get("primary_attempt_count", 1))
                    for outcome in branch_outcomes
                )
            ),
            "learned_reselections": int(
                sum(
                    int(outcome.get("learned_reselection_count", 0))
                    for outcome in branch_outcomes
                )
            ),
            "native_attempts": int(
                sum(
                    int(outcome.get("primary_attempt_count", 1))
                    for outcome in branch_outcomes
                )
                + np.sum(results.fallback_counts[branch.label])
            ),
            "fallback_uses": int(np.sum(results.fallback_counts[branch.label])),
            "recovered_failures": int(
                sum(
                    bool(outcome.get("recovered", False))
                    for outcome in branch_outcomes
                )
            ),
            "unrecovered_failures": int(unrecovered_count),
            "bandit_updates": int(update_count),
            "bandit_observations": int(
                sum(
                    int(outcome.get("bandit_observation_count", 0))
                    for outcome in branch_outcomes
                )
            ),
        }

    return recovery_audit


def _save_comparison_results(actions: _ComparisonActions, branches, instances, seed_map_by_family, run_dir, results: _ComparisonResults, recovery_audit):
    """Write runtime summaries, parameter diagnostics, and per-instance data."""
    active_trace_keys = _select_trace_keys(results.traces, TRACE_KEYS_FINAL)
    late_window_summary = _late_window_diagnostics(
        method_names=results.branch_labels,
        traces=results.traces,
        runtime_sec=results.runtime_sec,
        trace_keys=active_trace_keys,
    )
    late_window_flat = _late_window_flat_fields(late_window_summary)

    summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="final_unified_tuning_runtime",
        method_names=results.branch_labels,
        runtime_sec=results.runtime_sec,
        overhead_sec=results.overhead_sec,
        T=T,
        title=(
            f"BoomerAMG setup cumulative runtime (final unified tuning, non-RL)  "
            f"T={T}  n={FIXED_NX}x{FIXED_NY}x{FIXED_NZ}  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name=("default (fixed)" if FINAL_INCLUDE_DEFAULT else results.branch_labels[0]),
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
            "failure_protocol": "single_default_fallback",
            "shared_instance_stream": True,
            "within_step_branch_permutation": True,
            "permutation_seed": int(results.permutation_seed),
            "seed_map_by_family": {k: int(v) for k, v in seed_map_by_family.items()},
            "branch_seed_map": {branch.label: int(branch.seed) for branch in branches if branch.seed is not None},
            "branch_solver_profile_map": {
                branch.label: {
                    "solver_tol": float(branch.solver_tol),
                    "solver_max_iter": int(branch.solver_max_iter),
                }
                for branch in branches
            },
            "tune_dimensions": [int(v) for v in FINAL_TUNE_DIMS],
            "default_baseline": ("default (fixed)" if FINAL_INCLUDE_DEFAULT else None),
            "actions_count_tune3": int(len(actions.actions_tune3)),
            "actions_count_tune5": int(len(actions.actions_tune5)),
            "actions_count_tune7": int(len(actions.actions_tune7)),
            "grid_n": int(actions.grid_n),
            "coarsen_type_values": [int(v) for v in DEFAULT_COARSEN_TYPE_VALUES],
            "p_max_values": [int(v) for v in actions.p_max_values],
            "agg_num_levels_values": [int(v) for v in actions.agg_nl_values],
            "tune7_variant": str(TUNE7_VARIANT),
            "tune7_alpha": (float(TUNE7_ALPHA_OVERRIDE) if TUNE7_ALPHA_OVERRIDE else float(ALPHA)),
            "tune7_candidate_strategy": resolve_tune7_candidate_strategy(
                tune7_variant=TUNE7_VARIANT,
                configured_strategy=TUNE7_CANDIDATE_STRATEGY,
            ),
            "tune7_coarsen_types": [int(v) for v in actions.coarsen_type_values_tune7],
            "tune7_interp_types": [int(v) for v in actions.interp_values_tune7],
            "tune7_agg_interp_types": [int(v) for v in actions.agg_interp_values_tune7],
            "tune7_agg_tr_values": [float(v) for v in actions.agg_tr_values_tune7],
            "tune7_agg_pmx_values": [int(v) for v in actions.agg_pmx_values_tune7],
            "tune7_candidate_pool_size": int(TUNE7_CANDIDATE_POOL_SIZE),
            "tune7_candidate_pool_size_burnin": int(TUNE7_CANDIDATE_POOL_SIZE_BURNIN),
            "tune7_candidate_pool_burnin_rounds": int(TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS),
            "tune7_alpha_decay_burnin_rounds": int(TUNE7_ALPHA_DECAY_BURNIN_ROUNDS),
            "tune7_local_neighbor_radius": int(TUNE7_LOCAL_NEIGHBOR_RADIUS),
            "tune7_candidate_local_fraction": float(TUNE7_CANDIDATE_LOCAL_FRACTION),
            "tune7_candidate_elite_fraction": float(TUNE7_CANDIDATE_ELITE_FRACTION),
            "branch_filter": list(FINAL_BRANCH_FILTER),
            "failed_count_total": {k: int(np.sum(v.astype(int))) for k, v in results.failed_flags.items()},
            "fallback_count_total": {k: int(np.sum(v.astype(int))) for k, v in results.fallback_counts.items()},
            "recovery_audit": recovery_audit,
            "mean_test_problem_runtime_sec": {k: float(np.mean(v)) for k, v in results.runtime_sec.items()},
            "late_window_summary": late_window_summary,
            **late_window_flat,
        },
    )

    data_csv_path = run_dir / "per_instance_runtime_data.csv"
    _save_per_instance_csv(
        out_csv=data_csv_path,
        branches=branches,
        instances=instances,
        runtime_sec=results.runtime_sec,
        overhead_sec=results.overhead_sec,
        failed_flags=results.failed_flags,
        fallback_counts=results.fallback_counts,
        outcomes=results.outcomes,
        traces=results.traces,
    )

    print("RUNTIME PLOT:", summary["plot"])
    print("RUNTIME SUMMARY:", summary["summary_path"])
    print("DATA CSV:", data_csv_path)


def main() -> None:
    actions, instances = _prepare_comparison_stream()
    branches, seed_map_by_family = _build_comparison_branches(actions)
    run_dir = create_run_output_dir(
        base_dir=plots_base_dir,
        script_name=Path(__file__).stem,
        problem_name="scalar_anisotropic_diffusion",
        size_tag=f"{FIXED_NX}x{FIXED_NY}x{FIXED_NZ}",
        T=T,
        seed=SEED,
    )

    results = _evaluate_comparison(branches, instances)
    recovery_audit = _audit_comparison_recovery(branches, results)
    _save_comparison_results(
        actions, branches, instances, seed_map_by_family, run_dir, results, recovery_audit,
    )


if __name__ == "__main__":
    main()
