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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
REPO_ROOT = Path(__file__).resolve().parents[3]
SOLVE_TEST_DIR = REPO_ROOT / "SolvePhase" / "hypre" / "src" / "test"
sys.path.insert(0, str(SOLVE_TEST_DIR))

from learners.Bayesianbandits_AMG_v2 import Bayesianbandits_AMG_v2
from learners.SharedLinTS_AMG import SharedLinTS_AMG
from learners.SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from learners.SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from amg_gym_env import BoomerAMGRelaxEnv
from solver import create_env, solve
from utils.problem_amg import DIFCONV_CONTEXT_DIM, stencil_0_difconv_rl
from utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
from utils.setup_amg import build_actions_th_mxrs_tr, init_param_trace, progress_bar, record_param_trace


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

SOLVER_TOL = float(os.environ.get("SOLVER_TOL", "1e-8"))
SOLVER_MAX_ITER = int(os.environ.get("SOLVER_MAX_ITER", "10000"))
TEST9_RELAX_TYPE = os.environ.get("TEST9_RELAX_TYPE", "").strip()
SOLVE_MODEL_TYPE = os.environ.get("SOLVE_MODEL_TYPE", os.environ.get("MODEL_TYPE", "mlp")).strip().lower()
SOLVE_MODEL_PATH = Path(os.environ.get("SOLVE_MODEL_PATH", str(SOLVE_TEST_DIR / "ppo_boomeramg_gen")))
SOLVE_VEC_PATH = Path(os.environ.get("SOLVE_VEC_PATH", str(SOLVE_TEST_DIR / "vecnormalize_gen.pkl")))
SOLVE_LIB_PATH = Path(os.environ.get("SOLVE_LIB_PATH", str(SOLVE_TEST_DIR / "libamg_env.dylib")))
SOLVE_W_CENTER = float(os.environ.get("SOLVE_W_CENTER", "1.25"))
SOLVE_W_SCALE = float(os.environ.get("SOLVE_W_SCALE", "0.75"))
SOLVE_SWEEPS_MIN = int(os.environ.get("SOLVE_SWEEPS_MIN", "1"))
SOLVE_SWEEPS_MAX = int(os.environ.get("SOLVE_SWEEPS_MAX", "1"))
SOLVE_W_INIT = os.environ.get("SOLVE_W_INIT", "").strip()
SOLVE_SWEEPS_INIT = os.environ.get("SOLVE_SWEEPS_INIT", "").strip()
SOLVE_TOL = float(os.environ.get("SOLVE_TOL", "1e-8"))
SOLVE_MAX_CYCLES = int(os.environ.get("SOLVE_MAX_CYCLES", "20"))
SOLVE_GRID_NORM_DIV = float(os.environ.get("SOLVE_GRID_NORM_DIV", "100"))
SOLVE_C_NORM_DIV = float(os.environ.get("SOLVE_C_NORM_DIV", str(C_MAX)))

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
}
HYPRE_DEFAULT_PARAMS: Dict[str, Any] = {}
DEFAULT_P_MAX_ELMTS_VALUES = (2, 4, 6, 8, 12, 16)
DEFAULT_AGG_NUM_LEVELS_VALUES = (0, 1, 2, 3, 4, 5)
TRACE_KEYS_5 = ("strong_threshold", "max_row_sum", "trunc_factor", "P_max_elmts", "agg_num_levels")
SOLVE_MODE_RL = "rl_solve"
SOLVE_MODE_NO_RL = "no_rl_solve"


def runtime_loss_sec(*, outcome: Dict[str, Any], fail_runtime_sec: float, **_) -> float:
    rt = float(outcome["runtime"])
    if not np.isfinite(rt):
        return float(fail_runtime_sec)
    if bool(outcome.get("failed", False)) or rt >= 0.999 * float(fail_runtime_sec):
        return rt + 10.0
    return rt


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
        ratio = (float(r) + eps) / (float(r_prev) + eps)
        ratio = float(np.clip(ratio, 1e-30, 1e30))
        log_ratio = float(np.log(ratio))
        dr = float(r_prev) - float(r)
        sgn = 1.0 if dr > 0 else (-1.0 if dr < 0 else 0.0)
        cycle_frac = float(cycle) / float(SOLVE_MAX_CYCLES)

        c_denom = max(float(np.log(float(SOLVE_C_NORM_DIV) + eps)), eps)
        s1 = float(np.log(float(mkw["k"]) + eps) / c_denom)
        s2 = float(np.log(float(mkw["c"]) + eps) / c_denom)
        s3 = float(np.log(float(mkw["a0"]) + eps) / c_denom)

        n_denom = max(float(np.log(float(SOLVE_GRID_NORM_DIV) + eps)), eps)
        nx_norm = float(np.log(float(mkw["nx"]) + eps) / n_denom)
        ny_norm = float(np.log(float(mkw["ny"]) + eps) / n_denom)
        nz_norm = float(np.log(float(mkw["nz"]) + eps) / n_denom)

        obs = np.array(
            [log_ratio, sgn, cycle_frac, s1, s2, s3, nx_norm, ny_norm, nz_norm, float(last_w)],
            dtype=np.float32,
        )
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        if self.vec_norm is not None:
            obs = self.vec_norm.normalize_obs(obs)
        return obs

    def _map_action(self, action: np.ndarray, cycle: int) -> Tuple[float, int, int]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        a_w = float(np.clip(action[0], -1.0, 1.0))
        if self.w_only or action.shape[0] == 1:
            a_d = 0.0
            a_u = 0.0
        else:
            a_d = float(np.clip(action[1], -1.0, 1.0))
            a_u = float(np.clip(action[2], -1.0, 1.0))

        w = float(np.clip(
            SOLVE_W_CENTER + SOLVE_W_SCALE * a_w,
            SOLVE_W_CENTER - SOLVE_W_SCALE,
            SOLVE_W_CENTER + SOLVE_W_SCALE,
        ))

        if self.w_only:
            sweeps_down = self.sweeps_default
            sweeps_up = self.sweeps_default
        elif self.sweeps_half <= 0.0:
            sweeps_down = SOLVE_SWEEPS_MIN
            sweeps_up = SOLVE_SWEEPS_MIN
        else:
            sweeps_down = int(np.clip(np.round(self.sweeps_center + self.sweeps_half * a_d),
                                      SOLVE_SWEEPS_MIN, SOLVE_SWEEPS_MAX))
            sweeps_up = int(np.clip(np.round(self.sweeps_center + self.sweeps_half * a_u),
                                    SOLVE_SWEEPS_MIN, SOLVE_SWEEPS_MAX))

        if cycle == 0:
            if self.w_init is not None:
                w = float(np.clip(self.w_init, SOLVE_W_CENTER - SOLVE_W_SCALE, SOLVE_W_CENTER + SOLVE_W_SCALE))
            if self.sweeps_init is not None:
                init_s = int(np.clip(int(self.sweeps_init), SOLVE_SWEEPS_MIN, SOLVE_SWEEPS_MAX))
                sweeps_down = init_s
                sweeps_up = init_s

        return w, int(sweeps_down), int(sweeps_up)

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
            obs = self._make_obs(r=r_curr, r_prev=r_prev_obs, cycle=cycle, last_w=last_w, mkw=mkw)
            if self.use_lstm:
                action, lstm_state = self.model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = self.model.predict(obs, deterministic=True)

            w, sd, su = self._map_action(action, cycle)
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
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        params = dict(base)
        params["P_max_elmts"] = int(DEFAULT_PARAMS["P_max_elmts"])
        params["agg_num_levels"] = int(DEFAULT_PARAMS["agg_num_levels"])
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
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        for p_max in p_max_values:
            for agg_nl in agg_nl_values:
                params = dict(base)
                params["P_max_elmts"] = int(p_max)
                params["agg_num_levels"] = int(agg_nl)
                actions.append(params)
    return actions, p_max_values, agg_nl_values


def _ensure_default_arm(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    default_arm_index = next((i for i, a in enumerate(actions) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        return [*actions, dict(DEFAULT_PARAMS)], int(len(actions))
    return actions, int(default_arm_index)


def _make_methods(*, parameter_space: Dict[str, Any], default_arm_index: int, seed_base: int):
    return [
        ("default (fixed)", FixedPolicy(DEFAULT_PARAMS)),
        ("hypre default (fixed)", FixedPolicy(HYPRE_DEFAULT_PARAMS)),
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
        (
            "Shared LinTS",
            SharedFactory(
                SharedLinTS_AMG,
                ALPHA,
                L2,
                model_kwargs={
                    "sigma": SIGMA,
                    "action_center": DEFAULT_PARAMS,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 13003, T=T, trial=0),
        ),
        (
            "Bayesianbandits v2",
            SharedFactory(
                Bayesianbandits_AMG_v2,
                ALPHA,
                L2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 15003, T=T, trial=0),
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
        converged = bool(np.isfinite(res_norm) and res_norm <= float(SOLVE_TOL) and iters < int(SOLVE_MAX_CYCLES))
        return {
            "runtime": total_runtime,
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(rl_out["solve_runtime"]),
            "failed": (not converged),
            "residual_norm": res_norm,
            "iterations": iters,
            "final_w": float(rl_out["final_w"]),
            "final_sweeps_down": int(rl_out["final_sweeps_down"]),
            "final_sweeps_up": int(rl_out["final_sweeps_up"]),
        }
    except Exception:
        return {
            "runtime": float(fail_runtime_sec),
            "setup_runtime": float(fail_runtime_sec),
            "solve_runtime": 0.0,
            "failed": True,
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
            tol=SOLVER_TOL,
            max_iter=SOLVER_MAX_ITER,
            **mkw,
        )
        total_runtime = float(out.runtime_sec)
        converged = bool(np.isfinite(out.residual_norm) and out.residual_norm <= float(SOLVER_TOL))
        return {
            "runtime": total_runtime,
            "setup_runtime": float("nan"),
            "solve_runtime": float("nan"),
            "failed": (not converged),
            "residual_norm": float(out.residual_norm),
            "iterations": int(out.iterations),
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }
    except Exception:
        return {
            "runtime": float(fail_runtime_sec),
            "setup_runtime": float("nan"),
            "solve_runtime": float("nan"),
            "failed": True,
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
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
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
    return params, out, step_overhead, float(upd_sec)


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
    failed_flags: Dict[str, np.ndarray],
    traces: Dict[str, Dict[str, np.ndarray]],
    prev_update_est: Dict[str, float],
    rng_order: np.random.Generator,
) -> None:
    for local_t, (mkw, context) in enumerate(instances):
        order = rng_order.permutation(len(branch_entries))
        for i in order:
            label, _base_name, _tune_set, policy, parameter_space = branch_entries[int(i)]
            params, out, step_overhead, upd_sec = _run_one_step(
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
            overhead_sec[label][local_t] = float(step_overhead)
            failed_flags[label][local_t] = bool(out.get("failed", False))
            if label in traces:
                record_param_trace(traces[label], t=local_t, params=params, keys=TRACE_KEYS_5)
            prev_update_est[label] = float(upd_sec)

        progress_bar(local_t + 1, len(instances), prefix=phase_label, every=_progress_every())


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
    failed: Dict[str, np.ndarray],
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
        "overhead_sec",
        "end_to_end_sec",
        "failed",
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
                        "overhead_sec": ov,
                        "end_to_end_sec": float(rt + ov),
                        "failed": int(bool(failed[label][local_t])),
                        "strong_threshold": _param_value("strong_threshold"),
                        "max_row_sum": _param_value("max_row_sum"),
                        "trunc_factor": _param_value("trunc_factor"),
                        "P_max_elmts": _param_value("P_max_elmts"),
                        "agg_num_levels": _param_value("agg_num_levels"),
                    }
                )


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
    actions_tune5, p_max_values, agg_nl_values = _build_actions_tune5(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune5, default_arm_index_tune5 = _ensure_default_arm(actions_tune5)

    instances = _generate_instances(T=T, seed=SEED, sampler_kwargs=sampler_kwargs)

    parameter_space_tune3 = {"actions": actions_tune3, "context_dim": DIFCONV_CONTEXT_DIM}
    parameter_space_tune5 = {"actions": actions_tune5, "context_dim": DIFCONV_CONTEXT_DIM}

    methods_tune3_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune3,
        default_arm_index=int(default_arm_index_tune3),
        seed_base=SEED + 10_000,
    ))
    methods_tune5_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune5,
        default_arm_index=int(default_arm_index_tune5),
        seed_base=SEED + 20_000,
    ))
    methods_tune3_no_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune3,
        default_arm_index=int(default_arm_index_tune3),
        seed_base=SEED + 10_000,
    ))
    methods_tune5_no_rl = _filter_methods(_make_methods(
        parameter_space=parameter_space_tune5,
        default_arm_index=int(default_arm_index_tune5),
        seed_base=SEED + 20_000,
    ))
    method_names = [name for name, _ in methods_tune3_rl]
    if method_names != [name for name, _ in methods_tune5_rl]:
        raise RuntimeError("Method lists for tune3 and tune5 do not match")

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
            if str(name) == "Shared LinTS":
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
        return branch_entries, label_meta

    branch_entries_rl, label_meta_rl = _build_branch_entries(methods_tune3_rl, methods_tune5_rl)
    branch_entries_no_rl, label_meta_no_rl = _build_branch_entries(methods_tune3_no_rl, methods_tune5_no_rl)
    if [entry[0] for entry in branch_entries_rl] != [entry[0] for entry in branch_entries_no_rl]:
        raise RuntimeError("RL / non-RL branch labels do not match")
    branch_labels = [entry[0] for entry in branch_entries_rl]

    permutation_seed = int(SEED ^ 0x1A2B3C4D)
    print("Within-step permutation over all branches: enabled")

    def _alloc_metrics():
        return (
            {label: np.zeros(T, dtype=float) for label in branch_labels},
            {label: np.zeros(T, dtype=float) for label in branch_labels},
            {label: np.zeros(T, dtype=float) for label in branch_labels},
            {label: np.zeros(T, dtype=float) for label in branch_labels},
            {label: np.zeros(T, dtype=bool) for label in branch_labels},
            {label: init_param_trace(TRACE_KEYS_5, T)
             for label, _base, _set, policy, _ps in branch_entries_rl
             if not isinstance(policy, FixedPolicy)},
            {label: 0.0 for label in branch_labels},
        )

    runtime_sec_rl, setup_runtime_sec_rl, solve_runtime_sec_rl, overhead_sec_rl, failed_flags_rl, traces_rl, prev_update_est_rl = _alloc_metrics()
    runtime_sec_no_rl, setup_runtime_sec_no_rl, solve_runtime_sec_no_rl, overhead_sec_no_rl, failed_flags_no_rl, traces_no_rl, prev_update_est_no_rl = _alloc_metrics()

    print(f"Single run: tune3 + tune5 separate branches with RL solve, T={T}")
    _run_phase_dual_interleaved(
        phase_label="  dual run rl",
        branch_entries=branch_entries_rl,
        instances=instances,
        solve_policy=solve_policy,
        solve_mode=SOLVE_MODE_RL,
        runtime_sec=runtime_sec_rl,
        setup_runtime_sec=setup_runtime_sec_rl,
        solve_runtime_sec=solve_runtime_sec_rl,
        overhead_sec=overhead_sec_rl,
        failed_flags=failed_flags_rl,
        traces=traces_rl,
        prev_update_est=prev_update_est_rl,
        rng_order=np.random.default_rng(permutation_seed),
    )

    print(f"Single run: tune3 + tune5 separate branches with non-RL solve, T={T}")
    _run_phase_dual_interleaved(
        phase_label="  dual run no-rl",
        branch_entries=branch_entries_no_rl,
        instances=instances,
        solve_policy=solve_policy,
        solve_mode=SOLVE_MODE_NO_RL,
        runtime_sec=runtime_sec_no_rl,
        setup_runtime_sec=setup_runtime_sec_no_rl,
        solve_runtime_sec=solve_runtime_sec_no_rl,
        overhead_sec=overhead_sec_no_rl,
        failed_flags=failed_flags_no_rl,
        traces=traces_no_rl,
        prev_update_est=prev_update_est_no_rl,
        rng_order=np.random.default_rng(permutation_seed),
    )

    summary_rl = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="dual_tune3_tune5_interleaved_permuted_runtime_rlsolve",
        method_names=branch_labels,
        runtime_sec=runtime_sec_rl,
        overhead_sec=overhead_sec_rl,
        T=T,
        title=(
            f"BoomerAMG setup + RL solve cumulative runtime (test 9 tune3/tune5 separate, fully permuted)  "
            f"T={T}  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=traces_rl,
        trace_keys=(),
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "script": Path(__file__).name,
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
            "bias_mitigation": "within-step permutation on identical instance stream across method+tune_set branches",
            "within_step_method_permutation": True,
            "within_step_tune_set_permutation": True,
            "permutation_seed": permutation_seed,
            "solve_mode": SOLVE_MODE_RL,
            "failed_count_total": {k: int(np.sum(v.astype(int))) for k, v in failed_flags_rl.items()},
            "total_setup_runtime_sec": {k: float(np.sum(v)) for k, v in setup_runtime_sec_rl.items()},
            "mean_setup_runtime_sec": {k: float(np.mean(v)) for k, v in setup_runtime_sec_rl.items()},
            "total_rl_solve_runtime_sec": {k: float(np.sum(v)) for k, v in solve_runtime_sec_rl.items()},
            "mean_rl_solve_runtime_sec": {k: float(np.mean(v)) for k, v in solve_runtime_sec_rl.items()},
            "mean_test_problem_runtime_sec": {k: float(np.mean(v)) for k, v in runtime_sec_rl.items()},
        },
    )

    summary_no_rl = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="dual_tune3_tune5_interleaved_permuted_runtime_norlsolve",
        method_names=branch_labels,
        runtime_sec=runtime_sec_no_rl,
        overhead_sec=overhead_sec_no_rl,
        T=T,
        title=(
            f"BoomerAMG setup + non-RL solve cumulative runtime (test 9 tune3/tune5 separate, fully permuted)  "
            f"T={T}  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=traces_no_rl,
        trace_keys=(),
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "script": Path(__file__).name,
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
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "test9_relax_type": (int(TEST9_RELAX_TYPE) if TEST9_RELAX_TYPE else None),
            "solve_mode": SOLVE_MODE_NO_RL,
            "within_step_method_permutation": True,
            "within_step_tune_set_permutation": True,
            "permutation_seed": permutation_seed,
            "failed_count_total": {k: int(np.sum(v.astype(int))) for k, v in failed_flags_no_rl.items()},
            "mean_test_problem_runtime_sec": {k: float(np.mean(v)) for k, v in runtime_sec_no_rl.items()},
        },
    )

    data_csv_path_rl = run_dir / "per_instance_runtime_data_rlsolve.csv"
    _save_dual_csv(
        out_csv=data_csv_path_rl,
        solve_mode=SOLVE_MODE_RL,
        labels=branch_labels,
        label_meta=label_meta_rl,
        runtime_sec=runtime_sec_rl,
        setup_runtime_sec=setup_runtime_sec_rl,
        solve_runtime_sec=solve_runtime_sec_rl,
        overhead_sec=overhead_sec_rl,
        failed=failed_flags_rl,
        traces=traces_rl,
        t_total=T,
    )
    data_csv_path_no_rl = run_dir / "per_instance_runtime_data_norlsolve.csv"
    _save_dual_csv(
        out_csv=data_csv_path_no_rl,
        solve_mode=SOLVE_MODE_NO_RL,
        labels=branch_labels,
        label_meta=label_meta_no_rl,
        runtime_sec=runtime_sec_no_rl,
        setup_runtime_sec=setup_runtime_sec_no_rl,
        solve_runtime_sec=solve_runtime_sec_no_rl,
        overhead_sec=overhead_sec_no_rl,
        failed=failed_flags_no_rl,
        traces=traces_no_rl,
        t_total=T,
    )

    bundle_summary = {
        "script": Path(__file__).name,
        "seed": int(SEED),
        "T": int(T),
        "fixed_n": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
        "method_filter": str(METHOD_FILTER),
        "test9_relax_type": (int(TEST9_RELAX_TYPE) if TEST9_RELAX_TYPE else None),
        "continuation": False,
        "bias_mitigation": "within-step permutation on identical instance stream across method+tune_set branches",
        "within_step_method_permutation": True,
        "within_step_tune_set_permutation": True,
        "permutation_seed": permutation_seed,
        "runtime_plot_rlsolve": str(summary_rl["plot"]),
        "runtime_summary_rlsolve": str(summary_rl["summary_path"]),
        "data_csv_rlsolve": str(data_csv_path_rl),
        "runtime_plot_norlsolve": str(summary_no_rl["plot"]),
        "runtime_summary_norlsolve": str(summary_no_rl["summary_path"]),
        "data_csv_norlsolve": str(data_csv_path_no_rl),
    }
    bundle_summary_path = run_dir / "test_9_export_summary.json"
    bundle_summary_path.write_text(json.dumps(bundle_summary, indent=2) + "\n")

    print("RUNTIME PLOT (RL SOLVE):", summary_rl["plot"])
    print("RUNTIME SUMMARY (RL SOLVE):", summary_rl["summary_path"])
    print("DATA CSV (RL SOLVE):", data_csv_path_rl)
    print("RUNTIME PLOT (NO RL SOLVE):", summary_no_rl["plot"])
    print("RUNTIME SUMMARY (NO RL SOLVE):", summary_no_rl["summary_path"])
    print("DATA CSV (NO RL SOLVE):", data_csv_path_no_rl)
    print("EXPORT SUMMARY:", bundle_summary_path)
    print("TOTAL TEST PROBLEM RUNTIME (setup + RL solve) [sec]:", summary_rl["total_hypre_runtime_sec"])
    print("MEAN TEST PROBLEM RUNTIME (setup + RL solve) [sec]:", summary_rl["mean_test_problem_runtime_sec"])
    print("TOTAL TEST PROBLEM RUNTIME (setup + no-RL solve) [sec]:", summary_no_rl["total_hypre_runtime_sec"])
    print("MEAN TEST PROBLEM RUNTIME (setup + no-RL solve) [sec]:", summary_no_rl["mean_test_problem_runtime_sec"])


if __name__ == "__main__":
    main()
