"""
Test 9: tune3 and tune5 run separately, then evaluate both solve modes.

Protocol
--------
All methods are evaluated on the same instance stream.
At each step, execution order is randomly permuted over all branches:
  (method x tune_set), where tune_set in {tune3, tune5}.
For each selected setup-parameter action, we:
  1) build the BoomerAMG hierarchy using the setup-phase parameters
  2) solve the same system with the trained solve-phase RL policy
  3) also solve the same system without solve-phase RL

Outputs
-------
1) One cumulative runtime plot for RL solve
2) One cumulative runtime plot for non-RL solve
3) One per-instance CSV export for each solve mode
4) One JSON summary for paths + run metadata
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Render plots headlessly by default (safe for local runs too).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

REPO_ROOT = Path(__file__).resolve().parents[3]
SETUP_ROOT = REPO_ROOT / "SetupPhase"
SOLVE_CORE_DIR = REPO_ROOT / "SolvePhase" / "core"
for path in (REPO_ROOT, SETUP_ROOT, SOLVE_CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from learners.SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from learners.SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from learners.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from learners._amg_action_features import ParameterSpaceSpec, ParameterSpec
from amg_gym_env import BoomerAMGRelaxEnv, build_policy_obs, decode_policy_action
from hypre.bindings import create_env, solve
from problems.amg import DIFCONV_CONTEXT_DIM, stencil_0_difconv_rl
from utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
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

PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "0"))

SEED = int(os.environ.get("SEED", "39393939"))

ALPHA = float(os.environ.get("ALPHA", "1.0"))
L2 = float(os.environ.get("L2", "1.0"))
SIGMA = float(os.environ.get("SIGMA", "0.1"))

FIXED_N = int(os.environ.get("FIXED_N", "80"))
C_MIN = float(os.environ.get("C_MIN", "1.0"))
C_MAX = float(os.environ.get("C_MAX", "1000.0"))

CANDIDATE_POOL_SIZE = int(os.environ.get("CANDIDATE_POOL_SIZE", "512"))
ELITE_CACHE_SIZE = int(os.environ.get("ELITE_CACHE_SIZE", "64"))
METHOD_FILTER = os.environ.get("METHOD_FILTER", "").strip().lower()
BRANCH_FILTER = os.environ.get("BRANCH_FILTER", "").strip().lower()

DEFAULT_HYPRE_TOL = 1.0e-6
DEFAULT_HYPRE_MAX_ITER = 50

_SOLVER_TOL_RAW = os.environ.get("SOLVER_TOL", "").strip()
SOLVER_TOL = float(_SOLVER_TOL_RAW) if _SOLVER_TOL_RAW else float(DEFAULT_HYPRE_TOL)
_SOLVER_MAX_ITER_RAW = os.environ.get("SOLVER_MAX_ITER", "").strip()
SOLVER_MAX_ITER = int(_SOLVER_MAX_ITER_RAW) if _SOLVER_MAX_ITER_RAW else int(DEFAULT_HYPRE_MAX_ITER)
TEST9_RELAX_TYPE = os.environ.get("TEST9_RELAX_TYPE", "").strip()
SOLVE_MODEL_TYPE = os.environ.get("SOLVE_MODEL_TYPE", os.environ.get("MODEL_TYPE", "mlp")).strip().lower()
SOLVE_MODEL_PATH = Path(os.environ.get("SOLVE_MODEL_PATH", str(SOLVE_TEST_DIR / "ppo_boomeramg_setup_random")))
SOLVE_VEC_PATH = Path(os.environ.get("SOLVE_VEC_PATH", str(SOLVE_TEST_DIR / "vecnormalize_setup_random.pkl")))
SOLVE_LIB_PATH = Path(os.environ.get("SOLVE_LIB_PATH", str(SOLVE_TEST_DIR / "libamg_env.dylib")))
SOLVE_W_CENTER = float(os.environ.get("SOLVE_W_CENTER", "1.25"))
SOLVE_W_SCALE = float(os.environ.get("SOLVE_W_SCALE", "0.75"))
SOLVE_SWEEPS_MIN = int(os.environ.get("SOLVE_SWEEPS_MIN", "1"))
SOLVE_SWEEPS_MAX = int(os.environ.get("SOLVE_SWEEPS_MAX", "1"))
SOLVE_W_INIT = os.environ.get("SOLVE_W_INIT", "").strip()
SOLVE_SWEEPS_INIT = os.environ.get("SOLVE_SWEEPS_INIT", "").strip()
SOLVE_TOL = float(os.environ.get("SOLVE_TOL", str(DEFAULT_HYPRE_TOL)))
SOLVE_MAX_CYCLES = int(os.environ.get("SOLVE_MAX_CYCLES", str(DEFAULT_HYPRE_MAX_ITER)))
SOLVE_GRID_NORM_DIV = float(os.environ.get("SOLVE_GRID_NORM_DIV", "100"))
SOLVE_C_NORM_DIV = float(os.environ.get("SOLVE_C_NORM_DIV", str(C_MAX)))

plots_base_dir = REPO_ROOT / "results" / "joint" / "legacy_setup_solve"
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
HYPRE_DEFAULT_PARAMS: Dict[str, Any] = {}
DEFAULT_COARSEN_TYPE_VALUES = (0, 2, 6, 8, 10)
DEFAULT_P_MAX_ELMTS_VALUES = (2, 4, 6, 8, 12, 16)
DEFAULT_AGG_NUM_LEVELS_VALUES = (0, 1, 2, 3, 4, 5)
DEFAULT_TUNE7_INTERP_TYPES = (6, 8)
TRACE_KEYS_FINAL = (
    "strong_threshold",
    "max_row_sum",
    "trunc_factor",
    "coarsen_type",
    "interp_type",
    "P_max_elmts",
    "agg_num_levels",
    "agg_interp_type",
    "agg_tr",
    "agg_Pmx",
)
FINAL_TUNE_DIMS = tuple(
    sorted(
        {
            int(x.strip())
            for x in os.environ.get("FINAL_TUNE_DIMS", "3,5").split(",")
            if x.strip()
        }
    )
)
TUNE7_VARIANT = os.environ.get("TUNE7_VARIANT", "categorical").strip().lower()
SOLVE_MODE_RL = "rl_solve"
SOLVE_MODE_NO_RL = "no_rl_solve"


def runtime_loss_sec(*, outcome: Dict[str, Any], fail_runtime_sec: float, **_) -> float:
    rt = float(outcome["runtime"])
    if not np.isfinite(rt):
        return float(fail_runtime_sec)
    if bool(outcome.get("failed", False)) or rt >= 0.999 * float(fail_runtime_sec):
        return rt + 10.0
    return rt


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _compose_failure_reason(*, reasons: Sequence[str]) -> str:
    unique: List[str] = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(str(reason))
    return ";".join(unique) if unique else ""


def _classify_rl_failure(*, residual_norm: float, iterations: int) -> str:
    reasons: List[str] = []
    if not np.isfinite(residual_norm):
        reasons.append("non_finite_residual_norm")
    elif residual_norm > float(SOLVE_TOL):
        reasons.append("residual_above_solve_tol")
    if int(iterations) >= int(SOLVE_MAX_CYCLES):
        if np.isfinite(residual_norm) and residual_norm <= float(SOLVE_TOL):
            reasons.append("hit_or_exceeded_solve_max_cycles")
        else:
            reasons.append("max_cycles_reached_without_convergence")
    return _compose_failure_reason(reasons=reasons)


def _classify_no_rl_failure(*, residual_norm: float, iterations: int) -> str:
    reasons: List[str] = []
    if not np.isfinite(residual_norm):
        reasons.append("non_finite_residual_norm")
    elif residual_norm > float(SOLVER_TOL):
        reasons.append("residual_above_solver_tol")
    if int(iterations) >= int(SOLVER_MAX_ITER):
        reasons.append("max_iter_reached_without_convergence")
    return _compose_failure_reason(reasons=reasons)


def _summarize_failure_reasons(reason_map: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for label, arr in reason_map.items():
        counts: Dict[str, int] = {}
        for raw in arr.tolist():
            reason = str(raw or "").strip()
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
        out[label] = counts
    return out


def _write_failure_records(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_json_safe(rec), sort_keys=True) + "\n")


def _augment_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if TEST9_RELAX_TYPE:
        out["relax_type"] = int(TEST9_RELAX_TYPE)
    return out


class FixedPolicy:
    def __init__(self, params: Dict[str, Any]):
        self._params = _augment_params(params)

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


class SolvePolicyRunner:
    def __init__(self) -> None:
        if SOLVE_MODEL_TYPE == "lstm":
            self.model = RecurrentPPO.load(str(SOLVE_MODEL_PATH))
            self.use_lstm = True
        else:
            self.model = PPO.load(str(SOLVE_MODEL_PATH))
            self.use_lstm = False

        action_shape = getattr(self.model.action_space, "shape", ())
        self.w_only = bool(action_shape and int(action_shape[0]) == 1)
        self.w_init = float(SOLVE_W_INIT) if SOLVE_W_INIT else None
        self.sweeps_init = int(SOLVE_SWEEPS_INIT) if SOLVE_SWEEPS_INIT else None
        self.sweeps_center = 0.5 * (SOLVE_SWEEPS_MIN + SOLVE_SWEEPS_MAX)
        self.sweeps_half = 0.5 * (SOLVE_SWEEPS_MAX - SOLVE_SWEEPS_MIN)
        self.sweeps_default = int(np.clip(np.round(self.sweeps_center), SOLVE_SWEEPS_MIN, SOLVE_SWEEPS_MAX))

        self.vec_norm = None
        if SOLVE_VEC_PATH.exists():
            raw_env = DummyVecEnv(
                [lambda: BoomerAMGRelaxEnv(
                    lib_path=str(SOLVE_LIB_PATH),
                    fixed_grid=(FIXED_N, FIXED_N, FIXED_N),
                    randomize_A=False,
                    randomize_b=False,
                    fixed_stencil=0,
                    difconv_c=(1.0, 1.0, 1.0),
                    difconv_c_range=(1.0, max(1.0, SOLVE_C_NORM_DIV)),
                    difconv_a=(0.0, 0.0, 0.0),
                    w_center=SOLVE_W_CENTER,
                    w_scale=SOLVE_W_SCALE,
                    sweeps_min=SOLVE_SWEEPS_MIN,
                    sweeps_max=SOLVE_SWEEPS_MAX,
                    w_only=self.w_only,
                )]
            )
            self.vec_norm = VecNormalize.load(str(SOLVE_VEC_PATH), raw_env)
            self.vec_norm.training = False
            self.vec_norm.norm_reward = False

    def _make_obs(self, *, r: float, r_prev: float, cycle: int, last_w: float, mkw: Dict[str, Any]) -> np.ndarray:
        eps = 1e-30
        c_denom = max(float(np.log(float(SOLVE_C_NORM_DIV) + eps)), eps)
        s1 = float(np.log(float(mkw["k"]) + eps) / c_denom)
        s2 = float(np.log(float(mkw["c"]) + eps) / c_denom)
        s3 = float(np.log(float(mkw["a0"]) + eps) / c_denom)

        n_denom = max(float(np.log(float(SOLVE_GRID_NORM_DIV) + eps)), eps)
        nx_norm = float(np.log(float(mkw["nx"]) + eps) / n_denom)
        ny_norm = float(np.log(float(mkw["ny"]) + eps) / n_denom)
        nz_norm = float(np.log(float(mkw["nz"]) + eps) / n_denom)

        obs = build_policy_obs(
            r=float(r),
            r_prev=float(r_prev),
            cycle=cycle,
            max_cycles=SOLVE_MAX_CYCLES,
            coeff_triplet=(s1, s2, s3),
            grid_triplet=(nx_norm, ny_norm, nz_norm),
            last_w=float(last_w),
        )
        if self.vec_norm is not None:
            obs = self.vec_norm.normalize_obs(obs)
        return obs

    def _map_action(self, action: np.ndarray, cycle: int, last_w: float) -> Tuple[float, int, int]:
        return decode_policy_action(
            action,
            w_only=self.w_only,
            w_center=SOLVE_W_CENTER,
            w_scale=SOLVE_W_SCALE,
            sweeps_min=SOLVE_SWEEPS_MIN,
            sweeps_max=SOLVE_SWEEPS_MAX,
            cycle=cycle,
            last_w=last_w,
            w_init=self.w_init,
            sweeps_init=self.sweeps_init,
            w_smooth_alpha=0.0,
        )

    def run(self, env, *, mkw: Dict[str, Any]) -> Dict[str, Any]:
        r_prev_obs = float(env.r0)
        r_curr = float(env.r0)
        last_w = float(self.w_init) if self.w_init is not None else float(SOLVE_W_CENTER)
        solve_runtime = 0.0
        cycles = 0
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        last_sd = self.sweeps_default
        last_su = self.sweeps_default

        for cycle in range(SOLVE_MAX_CYCLES):
            obs = self._make_obs(
                r=r_curr,
                r_prev=r_prev_obs,
                cycle=cycle,
                last_w=last_w,
                mkw=mkw,
            )
            if self.use_lstm:
                action, lstm_state = self.model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            w, sd, su = self._map_action(action, cycle, last_w)
            r_new, dt = env.step_rl(relax_weight=w, sweeps_down=sd, sweeps_up=su)
            solve_runtime += float(dt)
            cycles = cycle + 1
            last_w = float(w)
            last_sd = int(sd)
            last_su = int(su)
            if r_new <= SOLVE_TOL:
                r_curr = float(r_new)
                break
            r_prev_obs = float(r_curr)
            r_curr = float(r_new)
            episode_start[...] = False
        else:
            r_curr = float(r_new)

        return {
            "solve_runtime": float(solve_runtime),
            "residual_norm": float(r_curr),
            "iterations": int(cycles),
            "failed": not (np.isfinite(r_curr) and r_curr <= float(SOLVE_TOL) and cycles < int(SOLVE_MAX_CYCLES)),
            "final_w": float(last_w),
            "final_sweeps_down": int(last_sd),
            "final_sweeps_up": int(last_su),
        }


def _same_action(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        np.isclose(float(a["strong_threshold"]), float(b["strong_threshold"]), rtol=0.0, atol=1e-12)
        and np.isclose(float(a["max_row_sum"]), float(b["max_row_sum"]), rtol=0.0, atol=1e-12)
        and np.isclose(float(a["trunc_factor"]), float(b["trunc_factor"]), rtol=0.0, atol=1e-12)
        and int(a["coarsen_type"]) == int(b["coarsen_type"])
        and int(a["interp_type"]) == int(b["interp_type"])
        and int(a["P_max_elmts"]) == int(b["P_max_elmts"])
        and int(a["agg_num_levels"]) == int(b["agg_num_levels"])
    )


def _parse_int_list_env(name: str, default_values: Sequence[int]) -> List[int]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [int(v) for v in default_values]


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
            "agg_interp_type": DEFAULT_PARAMS["agg_interp_type"],
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


def _ensure_default_arm(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    default_arm_index = next((i for i, a in enumerate(actions) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        return [*actions, dict(DEFAULT_PARAMS)], int(len(actions))
    return actions, int(default_arm_index)


def _make_methods(*, parameter_space: Dict[str, Any], default_arm_index: int, seed_base: int):
    return [
        ("default (fixed)", FixedPolicy(DEFAULT_PARAMS)),
        (
            "Shared LinUCB v2",
            SharedFactory(
                SharedLinUCB_AMG_v2,
                ALPHA,
                L2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 11003, T=T, trial=0),
        ),
        (
            "Shared LinUCB v3",
            SharedFactory(
                SharedLinUCB_AMG_v3,
                ALPHA,
                L2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 12003, T=T, trial=0),
        ),
    ]


def _make_methods_tune7(*, parameter_space: Dict[str, Any], parameter_spec: ParameterSpaceSpec, default_arm_index: int, seed_base: int):
    return [
        (
            "Shared LinUCB v4",
            SharedFactoryV4(
                ALPHA,
                L2,
                model_kwargs={
                    "parameter_spec": parameter_spec,
                    "context_interaction_indices": (1, 2, 3, 4),
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                    "initial_guess": [DEFAULT_PARAMS[param.name] for param in parameter_spec.parameters],
                    "initial_guess_rounds": 1,
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 13003, T=T, trial=0),
        ),
    ]


def _normalize_method_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _filter_methods(methods: Sequence[Tuple[str, Any]]) -> List[Tuple[str, Any]]:
    if not METHOD_FILTER:
        return list(methods)

    filt = _normalize_method_name(METHOD_FILTER)
    kept = [(name, policy) for name, policy in methods if filt in _normalize_method_name(name)]
    if not kept:
        available = ", ".join(name for name, _ in methods)
        raise ValueError(f"METHOD_FILTER={METHOD_FILTER!r} matched no methods. Available: {available}")
    return kept


def _filter_branch_entries(
    branch_entries: Sequence[Tuple[str, str, str, Any, Dict[str, Any]]],
    label_meta: Dict[str, Dict[str, Any]],
) -> Tuple[List[Tuple[str, str, str, Any, Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    if not BRANCH_FILTER:
        return list(branch_entries), dict(label_meta)

    filters = [_normalize_method_name(part) for part in BRANCH_FILTER.split(",") if part.strip()]
    kept = [
        entry
        for entry in branch_entries
        if any(filt in _normalize_method_name(entry[0]) for filt in filters)
    ]
    if not kept:
        available = ", ".join(entry[0] for entry in branch_entries)
        raise ValueError(f"BRANCH_FILTER={BRANCH_FILTER!r} matched no branches. Available: {available}")
    kept_labels = {entry[0] for entry in kept}
    return kept, {label: meta for label, meta in label_meta.items() if label in kept_labels}


def _generate_instances(*, T: int, seed: int, sampler_kwargs: Dict[str, Any]):
    rng = np.random.default_rng(seed)
    instances = []
    for t in range(int(T)):
        mkw, context, _meta = stencil_0_difconv_rl(rng=rng, t=t, trial=0, **sampler_kwargs)
        instances.append((mkw, np.asarray(context, dtype=float)))
    return instances


def _progress_every() -> int | None:
    if PROGRESS_EVERY <= 0:
        return None
    return int(PROGRESS_EVERY)


def _safe_solve_rl(params: Dict[str, Any], mkw: Dict[str, Any], *, solve_policy: SolvePolicyRunner, fail_runtime_sec: float) -> Dict[str, Any]:
    try:
        params = _augment_params(params)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            rl_out = solve_policy.run(env, mkw=mkw)
        res_norm = float(rl_out["residual_norm"])
        iters = int(rl_out["iterations"])
        total_runtime = float(prep.setup_runtime_sec + rl_out["solve_runtime"])
        failure_reason = _classify_rl_failure(residual_norm=res_norm, iterations=iters)
        converged = not bool(failure_reason)
        return {
            "runtime": total_runtime,
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(rl_out["solve_runtime"]),
            "failed": (not converged),
            "failure_reason": failure_reason,
            "residual_norm": res_norm,
            "iterations": iters,
            "final_w": float(rl_out["final_w"]),
            "final_sweeps_down": int(rl_out["final_sweeps_down"]),
            "final_sweeps_up": int(rl_out["final_sweeps_up"]),
        }
    except Exception as exc:
        return {
            "runtime": float(fail_runtime_sec),
            "setup_runtime": float(fail_runtime_sec),
            "solve_runtime": 0.0,
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(SOLVE_MAX_CYCLES),
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }


def _safe_solve_no_rl(params: Dict[str, Any], mkw: Dict[str, Any], *, fail_runtime_sec: float) -> Dict[str, Any]:
    try:
        params = _augment_params(params)
        out = solve(
            params=params,
            tol=(float(_SOLVER_TOL_RAW) if _SOLVER_TOL_RAW else None),
            max_iter=int(SOLVER_MAX_ITER),
            **mkw,
        )
        total_runtime = float(out.runtime_sec)
        failure_reason = _classify_no_rl_failure(
            residual_norm=float(out.residual_norm),
            iterations=int(out.iterations),
        )
        converged = not bool(failure_reason)
        return {
            "runtime": total_runtime,
            "setup_runtime": float(out.setup_runtime_sec),
            "solve_runtime": float(out.solve_runtime_sec),
            "failed": (not converged),
            "failure_reason": failure_reason,
            "residual_norm": float(out.residual_norm),
            "iterations": int(out.iterations),
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }
    except Exception as exc:
        return {
            "runtime": float(fail_runtime_sec),
            "setup_runtime": float("nan"),
            "solve_runtime": float("nan"),
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(SOLVER_MAX_ITER),
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }


def _run_one_step(
    *,
    policy: Any,
    solve_policy: SolvePolicyRunner,
    parameter_space: Dict[str, Any],
    context: np.ndarray,
    mkw: Dict[str, Any],
    prev_update_est: float,
    solve_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float]]:
    fail_runtime_sec = 1e9

    sel_start = time.perf_counter_ns()
    selected = policy.select(context=context, parameter_space=parameter_space)
    sel_sec = (time.perf_counter_ns() - sel_start) / 1e9
    params = selected[0] if isinstance(selected, tuple) else selected

    if solve_mode == SOLVE_MODE_RL:
        out = _safe_solve_rl(params, mkw, solve_policy=solve_policy, fail_runtime_sec=fail_runtime_sec)
    elif solve_mode == SOLVE_MODE_NO_RL:
        out = _safe_solve_no_rl(params, mkw, fail_runtime_sec=fail_runtime_sec)
    else:
        raise ValueError(f"Unknown solve_mode: {solve_mode}")

    loss_start = time.perf_counter_ns()
    base_loss_sec = float(runtime_loss_sec(outcome=out, fail_runtime_sec=fail_runtime_sec))
    loss_eval_sec = (time.perf_counter_ns() - loss_start) / 1e9
    end_to_end_loss_sec = base_loss_sec + float(sel_sec) + float(loss_eval_sec) + float(prev_update_est)

    upd_sec = 0.0
    if hasattr(policy, "update"):
        upd_start = time.perf_counter_ns()
        policy.update(loss=end_to_end_loss_sec, context=context, params=params, outcome=out)
        upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

    step_overhead = float(sel_sec + loss_eval_sec + upd_sec)
    return params, out, {
        "select_sec": float(sel_sec),
        "loss_eval_sec": float(loss_eval_sec),
        "update_sec": float(upd_sec),
        "overhead_sec": float(step_overhead),
    }


def _run_phase_dual_interleaved(
    *,
    phase_label: str,
    branch_entries: Sequence[Tuple[str, str, str, Any, Dict[str, Any]]],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    solve_policy: SolvePolicyRunner,
    solve_mode: str,
    runtime_sec: Dict[str, np.ndarray],
    setup_runtime_sec: Dict[str, np.ndarray],
    solve_runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    select_sec: Dict[str, np.ndarray],
    loss_eval_sec: Dict[str, np.ndarray],
    update_sec: Dict[str, np.ndarray],
    failed_flags: Dict[str, np.ndarray],
    failure_reason: Dict[str, np.ndarray],
    traces: Dict[str, Dict[str, np.ndarray]],
    prev_update_est: Dict[str, float],
    label_meta: Dict[str, Dict[str, Any]],
    failure_records: List[Dict[str, Any]],
    rng_order: np.random.Generator,
) -> None:
    phase_start_time = time.perf_counter()
    for local_t, (mkw, context) in enumerate(instances):
        order = rng_order.permutation(len(branch_entries))
        for i in order:
            label, base_name, tune_set, policy, parameter_space = branch_entries[int(i)]
            params, out, timing = _run_one_step(
                policy=policy,
                solve_policy=solve_policy,
                parameter_space=parameter_space,
                context=context,
                mkw=mkw,
                prev_update_est=float(prev_update_est[label]),
                solve_mode=solve_mode,
            )
            runtime_sec[label][local_t] = float(out["runtime"])
            setup_runtime_sec[label][local_t] = float(out["setup_runtime"])
            solve_runtime_sec[label][local_t] = float(out["solve_runtime"])
            overhead_sec[label][local_t] = float(timing["overhead_sec"])
            select_sec[label][local_t] = float(timing["select_sec"])
            loss_eval_sec[label][local_t] = float(timing["loss_eval_sec"])
            update_sec[label][local_t] = float(timing["update_sec"])
            failed_flags[label][local_t] = bool(out.get("failed", False))
            failure_reason[label][local_t] = str(out.get("failure_reason", ""))
            if label in traces:
                record_param_trace(traces[label], t=local_t, params=params, keys=TRACE_KEYS_FINAL)
            prev_update_est[label] = float(timing["update_sec"])
            if bool(out.get("failed", False)):
                failure_records.append(
                    {
                        "phase": "dual_permuted",
                        "solve_mode": str(solve_mode),
                        "t": int(local_t + 1),
                        "method": str(base_name),
                        "label": str(label),
                        "tune_set": str(tune_set),
                        "failure_reason": str(out.get("failure_reason", "")),
                        "residual_norm": float(out.get("residual_norm", np.nan)),
                        "iterations": int(out.get("iterations", -1)),
                        "setup_runtime_sec": float(out.get("setup_runtime", np.nan)),
                        "solve_runtime_sec": float(out.get("solve_runtime", np.nan)),
                        "runtime_sec": float(out.get("runtime", np.nan)),
                        "select_sec": float(timing["select_sec"]),
                        "loss_eval_sec": float(timing["loss_eval_sec"]),
                        "update_sec": float(timing["update_sec"]),
                        "overhead_sec": float(timing["overhead_sec"]),
                        "end_to_end_sec": float(out.get("runtime", np.nan) + timing["overhead_sec"]),
                        "final_w": float(out.get("final_w", np.nan)),
                        "final_sweeps_down": int(out.get("final_sweeps_down", -1)),
                        "final_sweeps_up": int(out.get("final_sweeps_up", -1)),
                        "params": dict(params),
                        "fixed_params": dict(label_meta.get(label, {}).get("fixed_params", {})),
                        "problem": {
                            "nx": int(mkw["nx"]),
                            "ny": int(mkw["ny"]),
                            "nz": int(mkw["nz"]),
                            "k": float(mkw["k"]),
                            "c": float(mkw["c"]),
                            "a0": float(mkw["a0"]),
                        },
                    }
                )

        progress_bar(
            local_t + 1,
            len(instances),
            prefix=phase_label,
            every=_progress_every(),
            start_time=phase_start_time,
        )


def _save_dual_csv(
    *,
    out_csv: Path,
    solve_mode: str,
    labels: Sequence[str],
    label_meta: Dict[str, Dict[str, str]],
    runtime_sec: Dict[str, np.ndarray],
    setup_runtime_sec: Dict[str, np.ndarray],
    solve_runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    select_sec: Dict[str, np.ndarray],
    loss_eval_sec: Dict[str, np.ndarray],
    update_sec: Dict[str, np.ndarray],
    failed: Dict[str, np.ndarray],
    failure_reason: Dict[str, np.ndarray],
    traces: Dict[str, Dict[str, np.ndarray]],
    t_total: int,
) -> None:
    fieldnames = [
        "phase",
        "solve_mode",
        "t",
        "t_phase",
        "method",
        "tune_set",
        "setup_runtime_sec",
        "solve_runtime_sec",
        "runtime_sec",
        "select_sec",
        "loss_eval_sec",
        "update_sec",
        "overhead_sec",
        "end_to_end_sec",
        "failed",
        "failure_reason",
        "strong_threshold",
        "max_row_sum",
        "trunc_factor",
        "P_max_elmts",
        "agg_num_levels",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for local_t in range(int(t_total)):
            global_t = int(local_t + 1)
            for label in labels:
                trace = traces.get(label, {})
                fixed_params = dict(label_meta[label].get("fixed_params", DEFAULT_PARAMS))

                def _param_value(key: str) -> float:
                    arr = trace.get(key)
                    if arr is None:
                        if key in fixed_params:
                            return float(fixed_params[key])
                        return float("nan")
                    val = float(arr[local_t])
                    if np.isfinite(val):
                        return val
                    if key in fixed_params:
                        return float(fixed_params[key])
                    return float("nan")

                rt = float(runtime_sec[label][local_t])
                setup_rt = float(setup_runtime_sec[label][local_t])
                solve_rt = float(solve_runtime_sec[label][local_t])
                sel = float(select_sec[label][local_t])
                loss_eval = float(loss_eval_sec[label][local_t])
                upd = float(update_sec[label][local_t])
                ov = float(overhead_sec[label][local_t])
                writer.writerow(
                    {
                        "phase": "dual_permuted",
                        "solve_mode": str(solve_mode),
                        "t": global_t,
                        "t_phase": int(local_t + 1),
                        "method": str(label_meta[label]["method"]),
                        "tune_set": str(label_meta[label]["tune_set"]),
                        "setup_runtime_sec": setup_rt,
                        "solve_runtime_sec": solve_rt,
                        "runtime_sec": rt,
                        "select_sec": sel,
                        "loss_eval_sec": loss_eval,
                        "update_sec": upd,
                        "overhead_sec": ov,
                        "end_to_end_sec": float(rt + ov),
                        "failed": int(bool(failed[label][local_t])),
                        "failure_reason": str(failure_reason[label][local_t]),
                        "strong_threshold": _param_value("strong_threshold"),
                        "max_row_sum": _param_value("max_row_sum"),
                        "trunc_factor": _param_value("trunc_factor"),
                        "P_max_elmts": _param_value("P_max_elmts"),
                        "agg_num_levels": _param_value("agg_num_levels"),
                    }
                )


def _subset_label_meta(
    branch_entries: Sequence[Tuple[str, str, str, Any, Dict[str, Any]]],
    label_meta: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {label: dict(label_meta[label]) for label, *_rest in branch_entries}


def _alloc_metrics_for_entries(
    branch_entries: Sequence[Tuple[str, str, str, Any, Dict[str, Any]]],
) -> Tuple[
    List[str],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, float],
]:
    labels = [entry[0] for entry in branch_entries]
    return (
        labels,
        {label: np.zeros(T, dtype=float) for label in labels},
        {label: np.zeros(T, dtype=float) for label in labels},
        {label: np.zeros(T, dtype=float) for label in labels},
        {label: np.zeros(T, dtype=float) for label in labels},
        {label: np.zeros(T, dtype=float) for label in labels},
        {label: np.zeros(T, dtype=float) for label in labels},
        {label: np.zeros(T, dtype=float) for label in labels},
        {label: np.zeros(T, dtype=bool) for label in labels},
        {label: np.full(T, "", dtype=object) for label in labels},
        {
            label: init_param_trace(TRACE_KEYS_FINAL, T)
            for label, _base, _set, policy, _ps in branch_entries
            if not isinstance(policy, FixedPolicy)
        },
        {label: 0.0 for label in labels},
    )


def _run_scenario(
    *,
    scenario_name: str,
    scenario_desc: str,
    phase_label: str,
    branch_entries: Sequence[Tuple[str, str, str, Any, Dict[str, Any]]],
    label_meta: Dict[str, Dict[str, Any]],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    solve_policy: SolvePolicyRunner,
    solve_mode: str,
    run_dir: Path,
    permutation_seed: int,
    grid_n: int,
    actions_tune3: Sequence[Dict[str, Any]],
    actions_tune5: Sequence[Dict[str, Any]],
    p_max_values: Sequence[int],
    agg_nl_values: Sequence[int],
) -> Dict[str, Any] | None:
    if not branch_entries:
        return None
    scenario_wall_start = time.perf_counter()

    (
        branch_labels,
        runtime_sec,
        setup_runtime_sec,
        solve_runtime_sec,
        overhead_sec,
        select_sec,
        loss_eval_sec,
        update_sec,
        failed_flags,
        failure_reason,
        traces,
        prev_update_est,
    ) = _alloc_metrics_for_entries(branch_entries)
    failure_records: List[Dict[str, Any]] = []

    print(f"Single run: {scenario_desc}, T={T}")
    _run_phase_dual_interleaved(
        phase_label=phase_label,
        branch_entries=branch_entries,
        instances=instances,
        solve_policy=solve_policy,
        solve_mode=solve_mode,
        runtime_sec=runtime_sec,
        setup_runtime_sec=setup_runtime_sec,
        solve_runtime_sec=solve_runtime_sec,
        overhead_sec=overhead_sec,
        select_sec=select_sec,
        loss_eval_sec=loss_eval_sec,
        update_sec=update_sec,
        failed_flags=failed_flags,
        failure_reason=failure_reason,
        traces=traces,
        prev_update_est=prev_update_est,
        label_meta=label_meta,
        failure_records=failure_records,
        rng_order=np.random.default_rng(permutation_seed),
    )

    failure_reason_counts = _summarize_failure_reasons(failure_reason)
    run_prefix = f"{scenario_name}_interleaved_runtime"
    solve_mode_title = "RL solve" if solve_mode == SOLVE_MODE_RL else "non-RL solve"
    default_method_name = "default (fixed)" if "default (fixed)" in branch_labels else branch_labels[0]
    summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix=run_prefix,
        method_names=branch_labels,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        T=T,
        title=(
            f"{scenario_desc} cumulative runtime ({solve_mode_title})  "
            f"T={T}  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name=default_method_name,
        traces=traces,
        trace_keys=(),
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "script": Path(__file__).name,
            "scenario_name": str(scenario_name),
            "scenario_desc": str(scenario_desc),
            "seed": int(SEED),
            "alpha": float(ALPHA),
            "l2": float(L2),
            "sigma": float(SIGMA),
            "fixed_n": int(FIXED_N),
            "c_min": float(C_MIN),
            "c_max": float(C_MAX),
            "context_dim": int(DIFCONV_CONTEXT_DIM),
            "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(ELITE_CACHE_SIZE),
            "method_filter": str(METHOD_FILTER),
            "branch_filter": str(BRANCH_FILTER),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "test9_relax_type": (int(TEST9_RELAX_TYPE) if TEST9_RELAX_TYPE else None),
            "solve_model_type": str(SOLVE_MODEL_TYPE),
            "solve_model_path": str(SOLVE_MODEL_PATH),
            "solve_vec_path": str(SOLVE_VEC_PATH),
            "solve_lib_path": str(SOLVE_LIB_PATH),
            "solve_tol": float(SOLVE_TOL),
            "solve_max_cycles": int(SOLVE_MAX_CYCLES),
            "solve_w_center": float(SOLVE_W_CENTER),
            "solve_w_scale": float(SOLVE_W_SCALE),
            "solve_sweeps_min": int(SOLVE_SWEEPS_MIN),
            "solve_sweeps_max": int(SOLVE_SWEEPS_MAX),
            "actions_grid_n": int(grid_n),
            "actions_count_tune3": int(len(actions_tune3)),
            "actions_count_tune5": int(len(actions_tune5)),
            "p_max_values": [int(v) for v in p_max_values],
            "agg_num_levels_values": [int(v) for v in agg_nl_values],
            "T": int(T),
            "continuation": False,
            "bias_mitigation": "within-step permutation on identical instance stream across scenario branches",
            "within_step_method_permutation": True,
            "within_step_tune_set_permutation": True,
            "permutation_seed": int(permutation_seed),
            "solve_mode": str(solve_mode),
            "failed_count_total": {k: int(np.sum(v.astype(int))) for k, v in failed_flags.items()},
            "failed_reason_counts": failure_reason_counts,
            "total_setup_runtime_sec": {k: float(np.sum(v)) for k, v in setup_runtime_sec.items()},
            "mean_setup_runtime_sec": {k: float(np.mean(v)) for k, v in setup_runtime_sec.items()},
            "total_solve_runtime_sec": {k: float(np.sum(v)) for k, v in solve_runtime_sec.items()},
            "mean_solve_runtime_sec": {k: float(np.mean(v)) for k, v in solve_runtime_sec.items()},
            "total_select_sec": {k: float(np.sum(v)) for k, v in select_sec.items()},
            "mean_select_sec": {k: float(np.mean(v)) for k, v in select_sec.items()},
            "total_loss_eval_sec": {k: float(np.sum(v)) for k, v in loss_eval_sec.items()},
            "mean_loss_eval_sec": {k: float(np.mean(v)) for k, v in loss_eval_sec.items()},
            "total_update_sec": {k: float(np.sum(v)) for k, v in update_sec.items()},
            "mean_update_sec": {k: float(np.mean(v)) for k, v in update_sec.items()},
            "total_overhead_sec": {k: float(np.sum(v)) for k, v in overhead_sec.items()},
            "mean_overhead_sec": {k: float(np.mean(v)) for k, v in overhead_sec.items()},
            "total_end_to_end_sec": {k: float(np.sum(runtime_sec[k] + overhead_sec[k])) for k in branch_labels},
            "mean_end_to_end_sec": {k: float(np.mean(runtime_sec[k] + overhead_sec[k])) for k in branch_labels},
            "mean_test_problem_runtime_sec": {k: float(np.mean(v)) for k, v in runtime_sec.items()},
        },
    )

    data_csv_path = run_dir / f"per_instance_runtime_data_{scenario_name}.csv"
    _save_dual_csv(
        out_csv=data_csv_path,
        solve_mode=solve_mode,
        labels=branch_labels,
        label_meta=label_meta,
        runtime_sec=runtime_sec,
        setup_runtime_sec=setup_runtime_sec,
        solve_runtime_sec=solve_runtime_sec,
        overhead_sec=overhead_sec,
        select_sec=select_sec,
        loss_eval_sec=loss_eval_sec,
        update_sec=update_sec,
        failed=failed_flags,
        failure_reason=failure_reason,
        traces=traces,
        t_total=T,
    )

    failure_log_path = run_dir / f"failed_cases_{scenario_name}.jsonl"
    _write_failure_records(failure_log_path, failure_records)
    scenario_wall_time_sec = float(time.perf_counter() - scenario_wall_start)
    summary["scenario_wall_time_sec"] = float(scenario_wall_time_sec)
    summary["mean_scenario_wall_time_per_instance_sec"] = float(scenario_wall_time_sec / max(1, int(T)))
    summary_path = Path(summary["summary_path"])
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return {
        "scenario_name": scenario_name,
        "scenario_desc": scenario_desc,
        "solve_mode": solve_mode,
        "branch_labels": branch_labels,
        "summary": summary,
        "data_csv_path": data_csv_path,
        "failure_log_path": failure_log_path,
        "runtime_sec": runtime_sec,
        "setup_runtime_sec": setup_runtime_sec,
        "solve_runtime_sec": solve_runtime_sec,
        "overhead_sec": overhead_sec,
        "select_sec": select_sec,
        "update_sec": update_sec,
        "failed_flags": failed_flags,
        "failure_reason_counts": failure_reason_counts,
        "scenario_wall_time_sec": float(scenario_wall_time_sec),
    }


def _print_scenario_summary(result: Dict[str, Any]) -> None:
    labels = result["branch_labels"]
    runtime_sec = result["runtime_sec"]
    setup_runtime_sec = result["setup_runtime_sec"]
    solve_runtime_sec = result["solve_runtime_sec"]
    overhead_sec = result["overhead_sec"]
    select_sec = result["select_sec"]
    update_sec = result["update_sec"]
    failed_flags = result["failed_flags"]
    scenario_desc = result["scenario_desc"]
    summary = result["summary"]

    print(f"RUNTIME PLOT ({scenario_desc}):", summary["plot"])
    print(f"RUNTIME SUMMARY ({scenario_desc}):", summary["summary_path"])
    print(f"DATA CSV ({scenario_desc}):", result["data_csv_path"])
    print(f"FAILURE LOG ({scenario_desc}):", result["failure_log_path"])
    print(f"TOTAL SETUP RUNTIME ({scenario_desc}) [sec]:", {k: float(np.sum(v)) for k, v in setup_runtime_sec.items()})
    print(f"MEAN SETUP RUNTIME ({scenario_desc}) [sec]:", {k: float(np.mean(v)) for k, v in setup_runtime_sec.items()})
    print(f"TOTAL SOLVE RUNTIME ({scenario_desc}) [sec]:", {k: float(np.sum(v)) for k, v in solve_runtime_sec.items()})
    print(f"MEAN SOLVE RUNTIME ({scenario_desc}) [sec]:", {k: float(np.mean(v)) for k, v in solve_runtime_sec.items()})
    print(f"TOTAL BANDIT SELECT TIME ({scenario_desc}) [sec]:", {k: float(np.sum(v)) for k, v in select_sec.items()})
    print(f"MEAN BANDIT SELECT TIME ({scenario_desc}) [sec]:", {k: float(np.mean(v)) for k, v in select_sec.items()})
    print(f"TOTAL BANDIT UPDATE TIME ({scenario_desc}) [sec]:", {k: float(np.sum(v)) for k, v in update_sec.items()})
    print(f"MEAN BANDIT UPDATE TIME ({scenario_desc}) [sec]:", {k: float(np.mean(v)) for k, v in update_sec.items()})
    print(f"TOTAL OVERHEAD TIME ({scenario_desc}) [sec]:", {k: float(np.sum(v)) for k, v in overhead_sec.items()})
    print(f"MEAN OVERHEAD TIME ({scenario_desc}) [sec]:", {k: float(np.mean(v)) for k, v in overhead_sec.items()})
    print(f"TOTAL TEST PROBLEM RUNTIME ({scenario_desc}) [sec]:", summary["total_hypre_runtime_sec"])
    print(f"MEAN TEST PROBLEM RUNTIME ({scenario_desc}) [sec]:", summary["mean_test_problem_runtime_sec"])
    print(f"TOTAL END-TO-END RUNTIME ({scenario_desc}) [sec]:", {k: float(np.sum(runtime_sec[k] + overhead_sec[k])) for k in labels})
    print(f"MEAN END-TO-END RUNTIME ({scenario_desc}) [sec]:", {k: float(np.mean(runtime_sec[k] + overhead_sec[k])) for k in labels})
    print(f"SCENARIO WALL TIME ({scenario_desc}) [sec]:", float(result["scenario_wall_time_sec"]))
    print(f"FAILED COUNT ({scenario_desc}):", {k: int(np.sum(v.astype(int))) for k, v in failed_flags.items()})
    print(f"FAILED REASONS ({scenario_desc}):", result["failure_reason_counts"])


def _best_mean_end_to_end(result: Dict[str, Any]) -> Tuple[str, float]:
    means = {
        str(k): float(v)
        for k, v in result["summary"].get("mean_end_to_end_sec", {}).items()
    }
    if not means:
        raise ValueError(f"No mean_end_to_end_sec found for scenario {result.get('scenario_name')}")
    best_label = min(means, key=means.get)
    return str(best_label), float(means[best_label])


def _print_overall_comparison(scenario_results: Sequence[Dict[str, Any]]) -> None:
    by_name = {str(result["scenario_name"]): result for result in scenario_results}
    if not by_name:
        return

    lines: List[Tuple[str, str, float, str]] = []
    fixed_value = None
    bandit_only_value = None

    fixed_result = by_name.get("fixed_only")
    if fixed_result is not None:
        fixed_label, fixed_value = _best_mean_end_to_end(fixed_result)
        lines.append(("fixed_only", fixed_label, fixed_value, "reference"))

    rl_only_result = by_name.get("rl_only")
    if rl_only_result is not None:
        label, value = _best_mean_end_to_end(rl_only_result)
        note = f"delta_vs_fixed={float(value - fixed_value):+.6f}" if fixed_value is not None else ""
        lines.append(("rl_only", label, value, note))

    bandit_only_result = by_name.get("bandit_only")
    if bandit_only_result is not None:
        label, bandit_only_value = _best_mean_end_to_end(bandit_only_result)
        note = f"delta_vs_fixed={float(bandit_only_value - fixed_value):+.6f}" if fixed_value is not None else ""
        lines.append(("bandit_only", label, bandit_only_value, note))

    rl_bandit_result = by_name.get("rl_bandit")
    if rl_bandit_result is not None:
        label, value = _best_mean_end_to_end(rl_bandit_result)
        notes: List[str] = []
        if fixed_value is not None:
            notes.append(f"delta_vs_fixed={float(value - fixed_value):+.6f}")
        if bandit_only_value is not None:
            notes.append(f"delta_vs_bandit_only={float(value - bandit_only_value):+.6f}")
        lines.append(("rl_bandit", label, value, ", ".join(notes)))

    print("OVERALL COMPARISON (mean end-to-end runtime [sec]):")
    for scenario_name, label, value, note in lines:
        suffix = f"  {note}" if note else ""
        print(f"  {scenario_name}: {value:.6f}  [{label}]{suffix}")


def main() -> None:
    sampler_kwargs = {
        "nx": int(FIXED_N),
        "ny": int(FIXED_N),
        "nz": int(FIXED_N),
        "n_min": int(FIXED_N),
        "n_max": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
    }

    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_0_difconv_rl(rng=warm_rng, t=0, trial=0, **sampler_kwargs)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)
    solve_policy = SolvePolicyRunner()

    grid_n, th_grid, mxrs_grid, tr_grid = _build_grids()
    actions_tune3 = _build_actions_tune3(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune3, default_arm_index_tune3 = _ensure_default_arm(actions_tune3)
    actions_tune5: List[Dict[str, Any]] = []
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

    actions_tune7: List[Dict[str, Any]] = []
    coarsen_type_values_tune7: List[int] = []
    interp_values_tune7: List[int] = []
    parameter_spec_tune7: ParameterSpaceSpec | None = None
    default_arm_index_tune7 = -1
    if 7 in FINAL_TUNE_DIMS:
        if TUNE7_VARIANT != "categorical":
            raise RuntimeError("test_9.py currently supports only TUNE7_VARIANT=categorical")
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
        actions_tune7, default_arm_index_tune7 = _ensure_default_arm(actions_tune7)

    instances = _generate_instances(T=T, seed=SEED, sampler_kwargs=sampler_kwargs)

    parameter_space_tune3 = {"actions": actions_tune3, "context_dim": DIFCONV_CONTEXT_DIM}
    parameter_space_tune5 = {"actions": actions_tune5, "context_dim": DIFCONV_CONTEXT_DIM} if actions_tune5 else None
    parameter_space_tune7 = {"actions": actions_tune7, "context_dim": DIFCONV_CONTEXT_DIM} if actions_tune7 else None

    methods_tune3_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune3,
        default_arm_index=int(default_arm_index_tune3),
        seed_base=SEED + 10_000,
    ))
    methods_tune5_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune5,
        default_arm_index=int(default_arm_index_tune5),
        seed_base=SEED + 20_000,
    )) if parameter_space_tune5 is not None else []
    methods_tune7_rl = _filter_methods(_make_methods_tune7(
        parameter_space=parameter_space_tune7,
        parameter_spec=parameter_spec_tune7,
        default_arm_index=int(default_arm_index_tune7),
        seed_base=SEED + 30_000,
    )) if parameter_space_tune7 is not None and parameter_spec_tune7 is not None else []
    methods_tune3_no_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune3,
        default_arm_index=int(default_arm_index_tune3),
        seed_base=SEED + 10_000,
    ))
    methods_tune5_no_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune5,
        default_arm_index=int(default_arm_index_tune5),
        seed_base=SEED + 20_000,
    )) if parameter_space_tune5 is not None else []
    methods_tune7_no_rl = _filter_methods(_make_methods_tune7(
        parameter_space=parameter_space_tune7,
        parameter_spec=parameter_spec_tune7,
        default_arm_index=int(default_arm_index_tune7),
        seed_base=SEED + 30_000,
    )) if parameter_space_tune7 is not None and parameter_spec_tune7 is not None else []

    run_dir = create_run_output_dir(
        base_dir=plots_base_dir,
        script_name=Path(__file__).stem,
        problem_name="difconv",
        size_tag=f"{FIXED_N}x{FIXED_N}x{FIXED_N}",
        T=T,
        seed=SEED,
    )

    def _build_branch_entries(
        methods_tune3: Sequence[Tuple[str, Any]],
        methods_tune5: Sequence[Tuple[str, Any]],
        methods_tune7: Sequence[Tuple[str, Any]],
    ) -> Tuple[List[Tuple[str, str, str, Any, Dict[str, Any]]], Dict[str, Dict[str, str]]]:
        branch_entries: List[Tuple[str, str, str, Any, Dict[str, Any]]] = []
        label_meta: Dict[str, Dict[str, Any]] = {}
        for name, policy in methods_tune3:
            if isinstance(policy, FixedPolicy):
                label = str(name)
                fixed_params = dict(getattr(policy, "_params", {}))
                branch_entries.append((label, name, "default", policy, parameter_space_tune3))
                label_meta[label] = {"method": str(name), "tune_set": "default", "fixed_params": fixed_params}
                continue
            label = f"{name} | tune3"
            branch_entries.append((label, name, "tune3", policy, parameter_space_tune3))
            label_meta[label] = {"method": str(name), "tune_set": "tune3", "fixed_params": {}}
        for name, policy in methods_tune5:
            if isinstance(policy, FixedPolicy):
                continue
            label = f"{name} | tune5"
            branch_entries.append((label, name, "tune5", policy, parameter_space_tune5))
            label_meta[label] = {"method": str(name), "tune_set": "tune5", "fixed_params": {}}
        for name, policy in methods_tune7:
            if isinstance(policy, FixedPolicy):
                continue
            label = f"{name} | tune7-categorical"
            branch_entries.append((label, name, "tune7", policy, parameter_space_tune7))
            label_meta[label] = {"method": str(name), "tune_set": "tune7", "fixed_params": {}}
        return branch_entries, label_meta

    branch_entries_rl, label_meta_rl = _build_branch_entries(methods_tune3_rl, methods_tune5_rl, methods_tune7_rl)
    branch_entries_no_rl, label_meta_no_rl = _build_branch_entries(methods_tune3_no_rl, methods_tune5_no_rl, methods_tune7_no_rl)
    branch_entries_rl, label_meta_rl = _filter_branch_entries(branch_entries_rl, label_meta_rl)
    branch_entries_no_rl, label_meta_no_rl = _filter_branch_entries(branch_entries_no_rl, label_meta_no_rl)
    if [entry[0] for entry in branch_entries_rl] != [entry[0] for entry in branch_entries_no_rl]:
        raise RuntimeError("RL / non-RL branch labels do not match")

    permutation_seed = int(SEED ^ 0x1A2B3C4D)
    print("Within-step permutation over all branches: enabled")

    fixed_entries_rl = [entry for entry in branch_entries_rl if isinstance(entry[3], FixedPolicy)]
    fixed_entries_no_rl = [entry for entry in branch_entries_no_rl if isinstance(entry[3], FixedPolicy)]
    bandit_entries_rl = [entry for entry in branch_entries_rl if not isinstance(entry[3], FixedPolicy)]
    bandit_entries_no_rl = [entry for entry in branch_entries_no_rl if not isinstance(entry[3], FixedPolicy)]

    scenario_results: List[Dict[str, Any]] = []
    for result in [
        _run_scenario(
            scenario_name="rl_only",
            scenario_desc="RL only (fixed setup + RL solve)",
            phase_label="  rl only",
            branch_entries=fixed_entries_rl,
            label_meta=_subset_label_meta(fixed_entries_rl, label_meta_rl),
            instances=instances,
            solve_policy=solve_policy,
            solve_mode=SOLVE_MODE_RL,
            run_dir=run_dir,
            permutation_seed=permutation_seed,
            grid_n=grid_n,
            actions_tune3=actions_tune3,
            actions_tune5=actions_tune5,
            p_max_values=p_max_values,
            agg_nl_values=agg_nl_values,
        ),
        _run_scenario(
            scenario_name="rl_bandit",
            scenario_desc="RL + Bandit (bandit setup + RL solve)",
            phase_label="  rl+bandit",
            branch_entries=bandit_entries_rl,
            label_meta=_subset_label_meta(bandit_entries_rl, label_meta_rl),
            instances=instances,
            solve_policy=solve_policy,
            solve_mode=SOLVE_MODE_RL,
            run_dir=run_dir,
            permutation_seed=permutation_seed,
            grid_n=grid_n,
            actions_tune3=actions_tune3,
            actions_tune5=actions_tune5,
            p_max_values=p_max_values,
            agg_nl_values=agg_nl_values,
        ),
        _run_scenario(
            scenario_name="bandit_only",
            scenario_desc="Bandit only (bandit setup + non-RL solve)",
            phase_label="  bandit only",
            branch_entries=bandit_entries_no_rl,
            label_meta=_subset_label_meta(bandit_entries_no_rl, label_meta_no_rl),
            instances=instances,
            solve_policy=solve_policy,
            solve_mode=SOLVE_MODE_NO_RL,
            run_dir=run_dir,
            permutation_seed=permutation_seed,
            grid_n=grid_n,
            actions_tune3=actions_tune3,
            actions_tune5=actions_tune5,
            p_max_values=p_max_values,
            agg_nl_values=agg_nl_values,
        ),
        _run_scenario(
            scenario_name="fixed_only",
            scenario_desc="Fixed only (fixed setup + non-RL solve)",
            phase_label="  fixed only",
            branch_entries=fixed_entries_no_rl,
            label_meta=_subset_label_meta(fixed_entries_no_rl, label_meta_no_rl),
            instances=instances,
            solve_policy=solve_policy,
            solve_mode=SOLVE_MODE_NO_RL,
            run_dir=run_dir,
            permutation_seed=permutation_seed,
            grid_n=grid_n,
            actions_tune3=actions_tune3,
            actions_tune5=actions_tune5,
            p_max_values=p_max_values,
            agg_nl_values=agg_nl_values,
        ),
    ]:
        if result is not None:
            scenario_results.append(result)

    bundle_summary = {
        "script": Path(__file__).name,
        "seed": int(SEED),
        "T": int(T),
        "fixed_n": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
        "method_filter": str(METHOD_FILTER),
        "branch_filter": str(BRANCH_FILTER),
        "test9_relax_type": (int(TEST9_RELAX_TYPE) if TEST9_RELAX_TYPE else None),
        "continuation": False,
        "bias_mitigation": "within-step permutation on identical instance stream across scenario branches",
        "within_step_method_permutation": True,
        "within_step_tune_set_permutation": True,
        "permutation_seed": permutation_seed,
        "scenarios": {
            result["scenario_name"]: {
                "scenario_desc": result["scenario_desc"],
                "solve_mode": result["solve_mode"],
                "runtime_plot": str(result["summary"]["plot"]),
                "runtime_summary": str(result["summary"]["summary_path"]),
                "data_csv": str(result["data_csv_path"]),
                "failure_log": str(result["failure_log_path"]),
            }
            for result in scenario_results
        },
    }
    bundle_summary_path = run_dir / "test_9_export_summary.json"
    bundle_summary_path.write_text(json.dumps(bundle_summary, indent=2) + "\n")

    for result in scenario_results:
        _print_scenario_summary(result)
    _print_overall_comparison(scenario_results)
    print("EXPORT SUMMARY:", bundle_summary_path)


if __name__ == "__main__":
    main()
