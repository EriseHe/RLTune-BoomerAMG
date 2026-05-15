from __future__ import annotations

import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from learners.HierarchicalLinUCB_AMG_v5 import HierarchicalLinUCB_AMG_v5
from learners.HierarchicalSquareCB_AMG_v4 import HierarchicalSquareCB_AMG_v4
from learners.NystromKernelUCB_AMG_v4 import NystromKernelUCB_AMG_v4
from learners.SharedLinTS_AMG_v4 import SharedLinTS_AMG_v4
from learners.SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from learners.SquareCB_AMG_v4 import SquareCB_AMG_v4
from learners._amg_action_features import ParameterSpaceSpec, ParameterSpec
from learners._tune7_online_features import Tune7FeatureBuilder
from solver import solve
from utils.scalar_anisotropic_diffusion import (
    SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM,
    stencil_0_scalar_anisotropic_diffusion_rl,
)
from utils.setup_amg import build_actions_from_spec, build_actions_th_mxrs_tr


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

DEFAULT_COARSEN_TYPE_VALUES = (0, 3, 6, 7, 8, 10, 22)
DEFAULT_P_MAX_ELMTS_VALUES = (2, 4, 6, 8, 12, 16)
DEFAULT_AGG_NUM_LEVELS_VALUES = (0, 1, 2, 3, 4, 5)
DEFAULT_TUNE7_INTERP_TYPES = (3, 6, 8, 14, 16, 17, 18)
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

NUMERIC_SCALES_TUNE7 = {
    "strong_threshold": (float(DEFAULT_PARAMS["strong_threshold"]), 0.25),
    "max_row_sum": (float(DEFAULT_PARAMS["max_row_sum"]), 0.10),
    "trunc_factor": (float(DEFAULT_PARAMS["trunc_factor"]), 0.20),
    "P_max_elmts": (float(DEFAULT_PARAMS["P_max_elmts"]), 4.0),
    "agg_num_levels": (float(DEFAULT_PARAMS["agg_num_levels"]), 1.0),
}


@dataclass(frozen=True)
class BenchmarkBranch:
    label: str
    family: str
    tune_set: str
    seed: int | None
    policy: Any
    solver_tol: float
    solver_max_iter: int


def _same_action(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
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


def parse_int_list_env(name: str, default_values: Sequence[int], env: Mapping[str, str]) -> List[int]:
    raw = env.get(name, ",".join(str(v) for v in default_values))
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [int(v) for v in default_values]


def build_grids(*, grid_n: int, grid_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if int(grid_n) <= 0:
        raise ValueError("grid_n must be positive")
    if not np.isfinite(float(grid_max)) or float(grid_max) <= 0.0:
        raise ValueError("grid_max must be finite and positive")
    th_grid = np.linspace(0.0, float(grid_max), int(grid_n))
    mxrs_grid = np.linspace(0.0, float(grid_max), int(grid_n))
    mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, float(grid_max), int(grid_n))
    return th_grid, mxrs_grid, tr_grid


def ensure_default_arm(actions: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    out = list(actions)
    default_arm_index = next((i for i, a in enumerate(out) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        out.append(dict(DEFAULT_PARAMS))
        default_arm_index = len(out) - 1
    return out, int(default_arm_index)


def build_actions_tune3(
    *,
    th_grid: Iterable[float],
    mxrs_grid: Iterable[float],
    tr_grid: Iterable[float],
) -> List[Dict[str, Any]]:
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


def build_actions_tune7(
    *,
    th_grid: Iterable[float],
    mxrs_grid: Iterable[float],
    tr_grid: Iterable[float],
    p_max_values: Sequence[int] = DEFAULT_P_MAX_ELMTS_VALUES,
    agg_nl_values: Sequence[int] = DEFAULT_AGG_NUM_LEVELS_VALUES,
    coarsen_type_values: Sequence[int] = DEFAULT_COARSEN_TYPE_VALUES,
    interp_values: Sequence[int] = DEFAULT_TUNE7_INTERP_TYPES,
) -> Tuple[List[Dict[str, Any]], ParameterSpaceSpec]:
    parameter_spec = ParameterSpaceSpec(
        (
            ParameterSpec(
                name="strong_threshold",
                kind="continuous",
                values=tuple(float(v) for v in th_grid),
                default=float(DEFAULT_PARAMS["strong_threshold"]),
                center=float(DEFAULT_PARAMS["strong_threshold"]),
                scale=NUMERIC_SCALES_TUNE7["strong_threshold"][1],
            ),
            ParameterSpec(
                name="max_row_sum",
                kind="continuous",
                values=tuple(float(v) for v in mxrs_grid),
                default=float(DEFAULT_PARAMS["max_row_sum"]),
                center=float(DEFAULT_PARAMS["max_row_sum"]),
                scale=NUMERIC_SCALES_TUNE7["max_row_sum"][1],
            ),
            ParameterSpec(
                name="trunc_factor",
                kind="continuous",
                values=tuple(float(v) for v in tr_grid),
                default=float(DEFAULT_PARAMS["trunc_factor"]),
                center=float(DEFAULT_PARAMS["trunc_factor"]),
                scale=NUMERIC_SCALES_TUNE7["trunc_factor"][1],
            ),
            ParameterSpec(
                name="P_max_elmts",
                kind="integer",
                values=tuple(int(v) for v in p_max_values),
                default=int(DEFAULT_PARAMS["P_max_elmts"]),
                center=float(DEFAULT_PARAMS["P_max_elmts"]),
                scale=NUMERIC_SCALES_TUNE7["P_max_elmts"][1],
            ),
            ParameterSpec(
                name="agg_num_levels",
                kind="integer",
                values=tuple(int(v) for v in agg_nl_values),
                default=int(DEFAULT_PARAMS["agg_num_levels"]),
                center=float(DEFAULT_PARAMS["agg_num_levels"]),
                scale=NUMERIC_SCALES_TUNE7["agg_num_levels"][1],
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
    return actions, parameter_spec


def generate_instances(
    *,
    T: int,
    seed: int,
    sampler_kwargs: Dict[str, Any],
    repeat_same_instance: bool = False,
) -> List[Tuple[Dict[str, Any], np.ndarray]]:
    rng = np.random.default_rng(int(seed))
    instances: List[Tuple[Dict[str, Any], np.ndarray]] = []
    if bool(repeat_same_instance):
        mkw, context, _meta = stencil_0_scalar_anisotropic_diffusion_rl(
            rng=rng,
            t=0,
            trial=0,
            **sampler_kwargs,
        )
        base_mkw = dict(mkw)
        base_context = np.asarray(context, dtype=float)
        for _t in range(int(T)):
            instances.append((dict(base_mkw), np.asarray(base_context, dtype=float).copy()))
        return instances
    for t in range(int(T)):
        mkw, context, _meta = stencil_0_scalar_anisotropic_diffusion_rl(rng=rng, t=t, trial=0, **sampler_kwargs)
        instances.append((mkw, np.asarray(context, dtype=float)))
    return instances


def rolling_success_scale_sec(
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
    return float(rt)


def compute_failure_scale_min_runtime_sec(
    *,
    mkw: Dict[str, Any],
    solver_tol: float,
    solver_max_iter: int,
    failure_scale_min_runtime_sec: float,
) -> float:
    if float(failure_scale_min_runtime_sec) > 0.0:
        return float(failure_scale_min_runtime_sec)
    try:
        res = solve(params=DEFAULT_PARAMS, tol=float(solver_tol), max_iter=int(solver_max_iter), **mkw)
        rt = float(res.runtime_sec)
        if np.isfinite(rt) and rt > 0.0:
            return float(rt)
    except Exception:
        pass
    return 1e-3


def safe_solve(
    params: Mapping[str, Any],
    mkw: Mapping[str, Any],
    *,
    fail_runtime_sec: float,
    solver_tol: float,
    solver_max_iter: int,
) -> Dict[str, Any]:
    try:
        res = solve(params=dict(params), tol=float(solver_tol), max_iter=int(solver_max_iter), **dict(mkw))
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


def evaluate_action(
    *,
    params: Mapping[str, Any],
    mkw: Mapping[str, Any],
    success_runtime_history: deque[float],
    b_min_runtime_sec: float,
    solver_tol: float,
    solver_max_iter: int,
    fail_runtime_sec: float,
    failure_penalty_multiplier: float,
    failure_severity_cap: float,
    structural_failure_surcharge_multiplier: float,
) -> Tuple[Dict[str, Any], float, float]:
    outcome = safe_solve(
        params=params,
        mkw=mkw,
        fail_runtime_sec=float(fail_runtime_sec),
        solver_tol=float(solver_tol),
        solver_max_iter=int(solver_max_iter),
    )
    loss_start = time.perf_counter_ns()
    success_runtime_scale_sec = rolling_success_scale_sec(
        success_runtime_history,
        b_min_runtime_sec=float(b_min_runtime_sec),
    )
    raw_loss_sec = runtime_loss_sec(
        outcome=outcome,
        fail_runtime_sec=float(fail_runtime_sec),
        solver_tol=float(solver_tol),
        success_runtime_scale_sec=float(success_runtime_scale_sec),
        failure_penalty_multiplier=float(failure_penalty_multiplier),
        failure_severity_cap=float(failure_severity_cap),
        structural_failure_surcharge_multiplier=float(structural_failure_surcharge_multiplier),
    )
    loss_eval_sec = float((time.perf_counter_ns() - loss_start) / 1e9)
    if not bool(outcome.get("failed", False)):
        runtime_sec = float(outcome["runtime"])
        if np.isfinite(runtime_sec) and runtime_sec > 0.0:
            success_runtime_history.append(float(runtime_sec))
    return dict(outcome), float(raw_loss_sec), float(loss_eval_sec)


def _maximin_order(points: np.ndarray, *, seed: int, count: Optional[int] = None) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2:
        raise ValueError("points must have shape (n, d)")
    n = int(pts.shape[0])
    if n == 0:
        return np.zeros(0, dtype=int)
    target = n if count is None else min(int(count), n)
    rng = np.random.default_rng(int(seed))
    centroid = np.mean(pts, axis=0)
    radii = np.linalg.norm(pts - centroid, axis=1)
    max_radius = float(np.max(radii))
    first_candidates = np.flatnonzero(np.isclose(radii, max_radius, rtol=0.0, atol=1e-12))
    first = int(rng.choice(first_candidates))

    selected = [first]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[first] = True
    min_dists = np.linalg.norm(pts - pts[first], axis=1)
    min_dists[first] = -np.inf

    while len(selected) < target:
        remaining = np.flatnonzero(~selected_mask)
        if remaining.size == 0:
            break
        best_dist = float(np.max(min_dists[remaining]))
        best_candidates = remaining[np.isclose(min_dists[remaining], best_dist, rtol=0.0, atol=1e-12)]
        nxt = int(rng.choice(best_candidates))
        selected.append(nxt)
        selected_mask[nxt] = True
        d = np.linalg.norm(pts - pts[nxt], axis=1)
        min_dists = np.minimum(min_dists, d)
        min_dists[selected_mask] = -np.inf
    return np.asarray(selected, dtype=int)


def _pairwise_distance_median(
    values: np.ndarray,
    *,
    max_points: Optional[int] = None,
    seed: Optional[int] = None,
) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 1.0
    if max_points is not None and int(max_points) > 1 and arr.shape[0] > int(max_points):
        rng = np.random.default_rng(seed)
        keep = np.asarray(rng.choice(arr.shape[0], size=int(max_points), replace=False), dtype=int)
        arr = arr[keep]
    diffs = arr[:, None, :] - arr[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    triu = dists[np.triu_indices(arr.shape[0], k=1)]
    finite = triu[np.isfinite(triu) & (triu > 0.0)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def _median_random_pair_distance(
    values: np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] <= 1:
        return 1.0
    pair_count = max(1, int(pair_count))
    rng = np.random.default_rng(int(seed))
    left = np.asarray(rng.integers(0, arr.shape[0], size=pair_count), dtype=int)
    right = np.asarray(rng.integers(0, arr.shape[0], size=pair_count), dtype=int)
    same = left == right
    if np.any(same):
        right[same] = (right[same] + 1) % arr.shape[0]
    dists = np.linalg.norm(arr[left] - arr[right], axis=1)
    finite = dists[np.isfinite(dists) & (dists > 0.0)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def _stratified_sample_indices(
    group_ids: np.ndarray,
    *,
    sample_size: int,
    seed: int,
) -> np.ndarray:
    ids = np.asarray(group_ids, dtype=int).reshape(-1)
    if ids.size == 0 or int(sample_size) <= 0:
        return np.zeros(0, dtype=int)
    if ids.size <= int(sample_size):
        return np.arange(ids.size, dtype=int)

    rng = np.random.default_rng(int(seed))
    groups = [np.flatnonzero(ids == group).astype(int) for group in sorted(int(v) for v in np.unique(ids))]
    selected: List[int] = []
    selected_set = set()
    order = np.asarray(rng.permutation(len(groups)), dtype=int)

    while len(selected) < int(sample_size):
        added = False
        for group_idx in order:
            pool = groups[int(group_idx)]
            available = pool[~np.isin(pool, np.asarray(list(selected_set), dtype=int), assume_unique=False)]
            if available.size == 0:
                continue
            pick = int(available[int(rng.integers(0, available.size))])
            selected.append(pick)
            selected_set.add(pick)
            added = True
            if len(selected) >= int(sample_size):
                break
        if not added:
            break
    return np.asarray(selected[: int(sample_size)], dtype=int)


def select_nystrom_landmarks(
    *,
    contexts: Sequence[Sequence[float]],
    feature_builder: Tune7FeatureBuilder,
    m: int,
    seed: int,
    sigma_z_sample_size: int = 512,
    sigma_z_pair_count: int = 10_000,
) -> Dict[str, np.ndarray | float]:
    contexts_arr = np.asarray([np.asarray(x, dtype=float) for x in contexts], dtype=float)
    if contexts_arr.ndim != 2 or contexts_arr.shape[0] == 0:
        raise ValueError("contexts must be a non-empty sequence of vectors")
    if int(m) <= 0:
        raise ValueError("m must be positive")

    x_mean = np.mean(contexts_arr, axis=0)
    x_std = np.std(contexts_arr, axis=0)
    x_std = np.where(x_std > 1e-8, x_std, 1.0)
    x_scaled = (contexts_arr - x_mean) / x_std

    sigma_z_arms = _stratified_sample_indices(
        feature_builder.regime_indices,
        sample_size=min(int(sigma_z_sample_size), int(feature_builder.K)),
        seed=int(seed) ^ 0x13579BDF,
    )
    if sigma_z_arms.size == 0:
        sigma_z_arms = np.arange(min(int(m), int(feature_builder.K)), dtype=int)

    landmark_arms = _stratified_sample_indices(
        feature_builder.regime_indices,
        sample_size=min(int(m), int(feature_builder.K)),
        seed=int(seed) ^ 0x0F1E2D3C,
    )
    if landmark_arms.size == 0:
        landmark_arms = sigma_z_arms[: min(int(m), int(sigma_z_arms.size))]

    context_order = _maximin_order(
        x_scaled,
        seed=int(seed) ^ 0x2468ACE0,
        count=min(int(m), int(x_scaled.shape[0])),
    )
    landmark_count = min(int(m), int(landmark_arms.size))
    if landmark_count <= 0:
        raise ValueError("unable to choose Nyström landmarks")

    context_idx = np.resize(context_order, landmark_count)
    action_idx = landmark_arms[:landmark_count]
    landmark_x = contexts_arr[context_idx]
    landmark_z = feature_builder.z_matrix(action_idx)
    landmark_regime = feature_builder.regime_indices[action_idx]

    return {
        "landmark_x": np.asarray(landmark_x, dtype=float),
        "landmark_z": np.asarray(landmark_z, dtype=float),
        "landmark_regime": np.asarray(landmark_regime, dtype=int),
        "sigma_x": float(_pairwise_distance_median(contexts_arr)),
        "sigma_z": float(
            _median_random_pair_distance(
                feature_builder.z_matrix(sigma_z_arms),
                pair_count=int(sigma_z_pair_count),
                seed=int(seed) ^ 0x10293847,
            )
        ),
    }

class DirectPolicyAdapter:
    def __init__(
        self,
        learner: Any,
        *,
        initial_params: Optional[Mapping[str, Any]] = None,
        initial_guess_rounds: int = 0,
    ) -> None:
        self.learner = learner
        self.initial_params = (dict(initial_params) if initial_params is not None else None)
        self.initial_guess_rounds = int(initial_guess_rounds)
        self._pending_initial_context: Optional[np.ndarray] = None
        self._pending_initial_params: Optional[Dict[str, Any]] = None

    def _history_info(self) -> Dict[str, float]:
        info: Dict[str, float] = {}
        if getattr(self.learner, "history", None):
            step = self.learner.history[-1]
            info.update(
                {
                "pred_mean": float(getattr(step, "pred_mean", np.nan)),
                "pred_uncert": float(getattr(step, "pred_uncert", np.nan)),
                }
            )
        if getattr(self.learner, "candidate_stats_history", None):
            stats = self.learner.candidate_stats_history[-1]
            for key, value in dict(stats).items():
                if isinstance(value, (int, float, np.integer, np.floating)):
                    info[str(key)] = float(value)
        return info

    def _should_force_initial_guess(self) -> bool:
        return (
            self.initial_params is not None
            and self.initial_guess_rounds > 0
            and int(getattr(self.learner, "t", 0)) < self.initial_guess_rounds
        )

    def _consume_forced_initial_update(self, *, loss: float) -> Optional[Dict[str, Any]]:
        if self._pending_initial_context is None or self._pending_initial_params is None:
            return None
        params = dict(self._pending_initial_params)
        context = np.asarray(self._pending_initial_context, dtype=float)
        self._pending_initial_context = None
        self._pending_initial_params = None
        self.learner.observe(
            context=context,
            params=params,
            loss=float(loss),
            bounded_loss=None,
        )
        return params

    def select(self, context: Iterable[float], **_kwargs: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
        if self._should_force_initial_guess():
            self._pending_initial_context = np.asarray(list(context), dtype=float)
            self._pending_initial_params = dict(self.initial_params)
            return dict(self.initial_params), {"pred_mean": float("nan"), "pred_uncert": float("nan")}
        params = self.learner.predict(context)
        return params, self._history_info()

    def update(
        self,
        *,
        loss: float,
        **_kwargs: Any,
    ) -> None:
        if self._consume_forced_initial_update(loss=float(loss)) is not None:
            return
        self.learner.update(float(loss), bounded_loss=None)

    def observe(
        self,
        *,
        context: Iterable[float],
        params: Mapping[str, Any],
        loss: float,
        **_kwargs: Any,
    ) -> None:
        self.learner.observe(
            context=context,
            params=params,
            loss=float(loss),
            bounded_loss=None,
        )
def build_online_benchmark_branches(
    *,
    seed: int,
    actions_tune3: Sequence[Mapping[str, Any]],
    actions_tune7: Sequence[Mapping[str, Any]],
    parameter_spec_tune7: ParameterSpaceSpec,
    instance_contexts: Sequence[Sequence[float]],
    solver_tol: float,
    solver_max_iter: int,
    legacy_solver_tol: float,
    legacy_solver_max_iter: int,
    kernel_context_count: int = 64,
    kernel_sigma_z_sample_size: int = 512,
    kernel_sigma_z_pair_count: int = 10_000,
    kernel_landmark_count: int = 32,
) -> Tuple[List[BenchmarkBranch], Dict[str, Any]]:
    context_dim = int(SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM)
    actions_tune3_list = list(actions_tune3)
    actions_tune7_list = tuple(actions_tune7)
    default_arm_index_tune7 = next(
        i for i, action in enumerate(actions_tune7_list) if _same_action(action, DEFAULT_PARAMS)
    )

    tune7_feature_builder = Tune7FeatureBuilder(tuple(actions_tune7_list), parameter_spec_tune7)
    kernel_contexts = [
        np.asarray(context, dtype=float)
        for context in instance_contexts[: max(1, min(int(kernel_context_count), len(instance_contexts)))]
    ]
    landmark_data = select_nystrom_landmarks(
        contexts=kernel_contexts,
        feature_builder=tune7_feature_builder,
        m=int(kernel_landmark_count),
        seed=int(seed + 17003),
        sigma_z_sample_size=int(kernel_sigma_z_sample_size),
        sigma_z_pair_count=int(kernel_sigma_z_pair_count),
    )

    branches: List[BenchmarkBranch] = []
    hyperparams: Dict[str, Dict[str, Any]] = {}

    v2_seed = int(seed + 11003)
    v2_policy = DirectPolicyAdapter(
        SharedLinUCB_AMG_v2(
            actions_tune3_list,
            context_dim=context_dim,
            alpha=1.0,
            l2_reg=1.0,
            seed=v2_seed,
        )
    )
    branches.append(
        BenchmarkBranch(
            label="Shared LinUCB v2 | tune3",
            family="Shared LinUCB v2",
            tune_set="tune3",
            seed=v2_seed,
            policy=v2_policy,
            solver_tol=float(legacy_solver_tol),
            solver_max_iter=int(legacy_solver_max_iter),
        )
    )
    hyperparams["Shared LinUCB v2 | tune3"] = {"alpha": 1.0, "l2_reg": 1.0}

    def _native_branch(
        *,
        label: str,
        family: str,
        branch_seed: int,
        learner: Any,
    ) -> None:
        policy = DirectPolicyAdapter(
            learner,
            initial_params=actions_tune7_list[int(default_arm_index_tune7)],
            initial_guess_rounds=1,
        )
        branches.append(
            BenchmarkBranch(
                label=label,
                family=family,
                tune_set="tune7",
                seed=branch_seed,
                policy=policy,
                solver_tol=float(solver_tol),
                solver_max_iter=int(solver_max_iter),
            )
        )

    lints_seed = int(seed + 14003)
    _native_branch(
        label="Shared LinTS v4 | tune7",
        family="Shared LinTS v4",
        branch_seed=lints_seed,
        learner=SharedLinTS_AMG_v4(
            actions_tune7_list,
            context_dim=context_dim,
            parameter_spec=parameter_spec_tune7,
            l2_reg=1.0,
            nu=0.5,
            candidate_pool_size=512,
            always_include_arms=[int(default_arm_index_tune7)],
            elite_cache_size=128,
            candidate_elite_size=128,
            feature_builder=tune7_feature_builder,
            seed=lints_seed,
        ),
    )
    hyperparams["Shared LinTS v4 | tune7"] = {
        "l2_reg": 1.0,
        "nu": 0.5,
        "initial_guess_rounds": 1,
        "selector_type": "native_uniform_elite",
        "candidate_pool_size": 512,
        "candidate_elite_size": 128,
        "candidate_random_size": 384,
        "elite_cache_size": 128,
    }

    square_linear_seed = int(seed + 15003)
    _native_branch(
        label="SquareCB linear | tune7",
        family="SquareCB linear",
        branch_seed=square_linear_seed,
        learner=SquareCB_AMG_v4(
            actions_tune7_list,
            context_dim=context_dim,
            parameter_spec=parameter_spec_tune7,
            oracle_kind="linear",
            l2_reg=1.0,
            gamma_scale=10.0,
            gamma_exponent=0.5,
            candidate_pool_size=512,
            always_include_arms=[int(default_arm_index_tune7)],
            elite_cache_size=128,
            candidate_local_size=0,
            candidate_elite_size=128,
            candidate_random_mode="uniform",
            feature_builder=tune7_feature_builder,
            seed=square_linear_seed,
        ),
    )
    hyperparams["SquareCB linear | tune7"] = {
        "l2_reg": 1.0,
        "gamma_scale": 10.0,
        "gamma_exponent": 0.5,
        "initial_guess_rounds": 1,
        "selector_type": "native_uniform_elite",
        "candidate_pool_size": 512,
        "candidate_elite_size": 128,
        "candidate_random_size": 384,
        "elite_cache_size": 128,
    }

    square_quad_seed = int(seed + 16003)
    _native_branch(
        label="SquareCB quadratic | tune7",
        family="SquareCB quadratic",
        branch_seed=square_quad_seed,
        learner=SquareCB_AMG_v4(
            actions_tune7_list,
            context_dim=context_dim,
            parameter_spec=parameter_spec_tune7,
            oracle_kind="quadratic",
            l2_reg=1.0,
            gamma_scale=10.0,
            gamma_exponent=0.5,
            candidate_pool_size=512,
            always_include_arms=[int(default_arm_index_tune7)],
            elite_cache_size=128,
            candidate_local_size=128,
            candidate_elite_size=128,
            candidate_random_mode="regime_stratified",
            feature_builder=tune7_feature_builder,
            seed=square_quad_seed,
        ),
    )
    hyperparams["SquareCB quadratic | tune7"] = {
        "l2_reg": 1.0,
        "gamma_scale": 10.0,
        "gamma_exponent": 0.5,
        "initial_guess_rounds": 1,
        "selector_type": "native_lattice_mixed",
        "candidate_pool_size": 512,
        "candidate_local_size": 128,
        "candidate_elite_size": 128,
        "candidate_random_size": 256,
        "candidate_random_mode": "regime_stratified",
        "elite_cache_size": 128,
    }

    kernel_seed = int(seed + 17003)
    _native_branch(
        label="Nyström KernelUCB | tune7",
        family="Nyström KernelUCB",
        branch_seed=kernel_seed,
        learner=NystromKernelUCB_AMG_v4(
            actions_tune7_list,
            context_dim=context_dim,
            parameter_spec=parameter_spec_tune7,
            landmark_x=np.asarray(landmark_data["landmark_x"], dtype=float),
            landmark_z=np.asarray(landmark_data["landmark_z"], dtype=float),
            landmark_regime=np.asarray(landmark_data["landmark_regime"], dtype=int),
            sigma_x=float(landmark_data["sigma_x"]),
            sigma_z=float(landmark_data["sigma_z"]),
            l2_reg=1e-2,
            beta0=1.0,
            candidate_pool_size=384,
            always_include_arms=[int(default_arm_index_tune7)],
            elite_cache_size=128,
            candidate_local_size=192,
            candidate_elite_size=64,
            feature_builder=tune7_feature_builder,
            seed=kernel_seed,
        ),
    )
    hyperparams["Nyström KernelUCB | tune7"] = {
        "l2_reg": 1e-2,
        "beta0": 1.0,
        "m": int(np.asarray(landmark_data["landmark_x"]).shape[0]),
        "sigma_x": float(landmark_data["sigma_x"]),
        "sigma_z": float(landmark_data["sigma_z"]),
        "kernel_context_count": int(len(kernel_contexts)),
        "kernel_sigma_z_sample_size": int(min(int(kernel_sigma_z_sample_size), int(len(actions_tune7_list)))),
        "kernel_sigma_z_pair_count": int(kernel_sigma_z_pair_count),
        "initial_guess_rounds": 1,
        "selector_type": "native_kernel_local",
        "candidate_pool_size": 384,
        "candidate_local_size": 192,
        "candidate_elite_size": 64,
        "candidate_random_size": 128,
        "candidate_random_mode": "regime_stratified",
        "elite_cache_size": 128,
    }

    hierarchical_seed = int(seed + 18003)
    _native_branch(
        label="Hierarchical SquareCB | tune7",
        family="Hierarchical SquareCB",
        branch_seed=hierarchical_seed,
        learner=HierarchicalSquareCB_AMG_v4(
            actions_tune7_list,
            context_dim=context_dim,
            parameter_spec=parameter_spec_tune7,
            top_l2_reg=0.3,
            top_gamma_scale=10.0,
            second_l2_reg=1.0,
            second_gamma_scale=200.0,
            gamma_exponent=0.5,
            min_regime_support=4,
            second_candidate_pool_size=512,
            always_include_arms=[int(default_arm_index_tune7)],
            elite_cache_size=128,
            second_local_size=320,
            second_elite_size=160,
            feature_builder=tune7_feature_builder,
            seed=hierarchical_seed,
        ),
    )
    hyperparams["Hierarchical SquareCB | tune7"] = {
        "top_l2_reg": 0.3,
        "top_gamma_scale": 10.0,
        "second_l2_reg": 1.0,
        "second_gamma_scale": 200.0,
        "gamma_exponent": 0.5,
        "min_regime_support": 4,
        "initial_guess_rounds": 1,
        "selector_type": "native_hierarchical_regime_local",
        "second_candidate_pool_size": 512,
        "second_local_size": 320,
        "second_elite_size": 160,
        "second_random_size": 32,
        "elite_cache_size": 128,
    }

    hierarchical_lincb_seed = int(seed + 19003)
    _native_branch(
        label="Hierarchical LinUCB v5 | tune7",
        family="Hierarchical LinUCB v5",
        branch_seed=hierarchical_lincb_seed,
        learner=HierarchicalLinUCB_AMG_v5(
            actions_tune7_list,
            context_dim=context_dim,
            parameter_spec=parameter_spec_tune7,
            alpha=1.0,
            l2_reg=1.0,
            candidate_pool_size=512,
            always_include_arms=[int(default_arm_index_tune7)],
            elite_cache_size=128,
            local_size=320,
            elite_size=160,
            local_seed_count=4,
            feature_builder=tune7_feature_builder,
            seed=hierarchical_lincb_seed,
        ),
    )
    hyperparams["Hierarchical LinUCB v5 | tune7"] = {
        "alpha": 1.0,
        "l2_reg": 1.0,
        "initial_guess_rounds": 1,
        "selector_type": "native_hierarchical_regime_value_lincb",
        "candidate_pool_size": 512,
        "candidate_local_size": 320,
        "candidate_elite_size": 160,
        "candidate_random_size": 32,
        "elite_cache_size": 128,
        "local_seed_count": 4,
    }

    metadata = {
        "branch_hyperparams": hyperparams,
    }
    return branches, metadata


def save_per_instance_csv(
    *,
    out_csv: Path,
    branches: Sequence[BenchmarkBranch],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    runtime_sec: Dict[str, np.ndarray],
    select_sec: Dict[str, np.ndarray],
    loss_eval_sec: Dict[str, np.ndarray],
    update_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    penalized_loss_sec: Dict[str, np.ndarray],
    failed_flags: Dict[str, np.ndarray],
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
        "select_sec",
        "loss_eval_sec",
        "update_sec",
        "overhead_sec",
        "end_to_end_sec",
        "penalized_loss_sec",
        "failed",
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
    out_csv.parent.mkdir(parents=True, exist_ok=True)
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
                        "select_sec": float(select_sec[label][t]),
                        "loss_eval_sec": float(loss_eval_sec[label][t]),
                        "update_sec": float(update_sec[label][t]),
                        "overhead_sec": float(overhead_sec[label][t]),
                        "end_to_end_sec": float(runtime_sec[label][t] + overhead_sec[label][t]),
                        "penalized_loss_sec": float(penalized_loss_sec[label][t]),
                        "failed": int(bool(failed_flags[label][t])),
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
