from __future__ import annotations

import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
_SETUP_ROOT = _REPO_ROOT / "SetupPhase" / "learning-setup"
_SETUP_SCRIPTS = _SETUP_ROOT / "scripts"
for _path in (str(_SETUP_ROOT), str(_SETUP_SCRIPTS)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from amg_gym_env import (
    build_policy_obs,
    decode_policy_action,
    decode_policy_action_hierarchical,
    decode_policy_action_residual,
)
from amg_setup_gym_env import BoomerAMGSetupRelaxEnv, SetupObsEncoder, build_setup_parameter_spec, build_setup_param_space
from learners.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from learners._amg_action_features import ParameterSpaceSpec, ParameterSpec
from solver import create_env, solve
from utils.problem_amg import DIFCONV_CONTEXT_DIM, stencil_0_difconv_rl
from utils.setup_amg import build_actions_from_spec, build_actions_th_mxrs_tr, init_param_trace, progress_bar, record_param_trace


FAIL_RUNTIME_SEC = 1e9
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
DEFAULT_SETUP_PARAMS: Dict[str, Any] = {
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


@dataclass(frozen=True)
class ActionSpaceBundle:
    grid_n: int
    actions_tune3: Sequence[Dict[str, Any]]
    actions_tune5: Sequence[Dict[str, Any]]
    actions_tune7: Sequence[Dict[str, Any]]
    p_max_values: Sequence[int]
    agg_nl_values: Sequence[int]
    agg_tr_values: Sequence[float]
    agg_pmx_values: Sequence[int]
    coarsen_type_values_tune7: Sequence[int]
    interp_values_tune7: Sequence[int]
    parameter_space_tune3: Dict[str, Any]
    parameter_space_tune5: Optional[Dict[str, Any]]
    parameter_space_tune7: Optional[Dict[str, Any]]
    parameter_spec_tune7: Optional[ParameterSpaceSpec]
    default_arm_index_tune3: int
    default_arm_index_tune5: int
    default_arm_index_tune7: int


@dataclass(frozen=True)
class TestFinalBanditConfig:
    alpha: float
    l2: float
    candidate_pool_size: int
    elite_cache_size: int
    retry_max_attempts: int
    failure_penalty_multiplier: float
    failure_severity_cap: float
    failure_scale_window: int
    failure_scale_min_runtime_sec_override: float
    structural_failure_surcharge_multiplier: float
    tune7_candidate_pool_size: int
    tune7_candidate_pool_size_burnin: int
    tune7_candidate_pool_burnin_rounds: int
    tune7_alpha_decay_burnin_rounds: int
    tune7_candidate_strategy: str
    tune7_local_neighbor_radius: int
    tune7_candidate_local_fraction: float
    tune7_candidate_elite_fraction: float


@dataclass(frozen=True)
class SetupAwareRLConfig:
    tune_dim: int
    tune7_variant: str
    algo: str
    model_type: str
    model_path: Path
    vec_path: Path
    fixed_grid: Tuple[int, int, int]
    difconv_c_range: Tuple[float, float]
    w_only: bool
    w_center: float
    w_scale: float
    sweeps_min: int
    sweeps_max: int
    w_init: Optional[float]
    sweeps_init: Optional[int]
    solve_max_cycles: int
    solve_tol: float
    default_setup_params: Dict[str, Any]
    w_global_min: float = 1.0
    w_global_max: float = 2.0
    obs_mode: str = "full"
    obs_include_trace_progress: bool = False
    action_mode: str = "continuous"
    discrete_w_values: Tuple[float, ...] = ()
    discrete_joint_actions: Tuple[Tuple[float, int, int], ...] = ()
    discrete_extended_actions: Tuple[Tuple[float, int, int, int, int, int, float, float], ...] = ()
    discrete_blend_alphas: Tuple[float, ...] = ()
    blend_safe_action: Tuple[float, int, int, int, int, int, float, float] = (1.6, 1, 1, 1, 1, 18, -1.0, -1.0)
    blend_aggr_action: Tuple[float, int, int, int, int, int, float, float] = (1.6, 1, 1, 2, 1, 18, -1.0, 0.1)
    blend_decay_tau: float = 0.0
    blend_cutoff_cycles: int = -1
    blend_progress_start: float = 0.0
    blend_progress_end: float = 0.0


def default_test_final_bandit_config_from_env() -> TestFinalBanditConfig:
    return TestFinalBanditConfig(
        alpha=float(os.environ.get("ALPHA", "1.0")),
        l2=float(os.environ.get("L2", "1.0")),
        candidate_pool_size=int(os.environ.get("CANDIDATE_POOL_SIZE", "512")),
        elite_cache_size=int(os.environ.get("ELITE_CACHE_SIZE", "64")),
        retry_max_attempts=int(os.environ.get("RETRY_MAX_ATTEMPTS", "1000")),
        failure_penalty_multiplier=float(os.environ.get("FAILURE_PENALTY_MULTIPLIER", "2.0")),
        failure_severity_cap=float(os.environ.get("FAILURE_SEVERITY_CAP", "6.0")),
        failure_scale_window=int(os.environ.get("FAILURE_SCALE_WINDOW", "200")),
        failure_scale_min_runtime_sec_override=float(os.environ.get("FAILURE_SCALE_MIN_RUNTIME_SEC", "0.0")),
        structural_failure_surcharge_multiplier=float(
            os.environ.get("STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER", "3.0")
        ),
        tune7_candidate_pool_size=int(os.environ.get("TUNE7_CANDIDATE_POOL_SIZE", "1024")),
        tune7_candidate_pool_size_burnin=int(os.environ.get("TUNE7_CANDIDATE_POOL_SIZE_BURNIN", "4096")),
        tune7_candidate_pool_burnin_rounds=int(os.environ.get("TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS", "200")),
        tune7_alpha_decay_burnin_rounds=int(os.environ.get("TUNE7_ALPHA_DECAY_BURNIN_ROUNDS", "250")),
        tune7_candidate_strategy=os.environ.get("TUNE7_CANDIDATE_STRATEGY", "").strip().lower(),
        tune7_local_neighbor_radius=int(os.environ.get("TUNE7_LOCAL_NEIGHBOR_RADIUS", "1")),
        tune7_candidate_local_fraction=float(os.environ.get("TUNE7_CANDIDATE_LOCAL_FRACTION", "0.60")),
        tune7_candidate_elite_fraction=float(os.environ.get("TUNE7_CANDIDATE_ELITE_FRACTION", "0.20")),
    )


def normalize_method_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def parse_int_list_env(name: str, default_values: Sequence[int]) -> List[int]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [int(v) for v in default_values]


def parse_float_list_env(name: str, default_values: Sequence[float]) -> List[float]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [float(v) for v in default_values]


def augment_setup_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply repo-level setup overrides before solve-time setup construction."""
    out = dict(params)
    relax_type_raw = os.environ.get("SETUP_RELAX_TYPE", "").strip()
    if relax_type_raw:
        out["relax_type"] = int(relax_type_raw)
    num_sweeps_raw = os.environ.get("SETUP_NUM_SWEEPS", "").strip()
    if num_sweeps_raw:
        out["num_sweeps"] = int(num_sweeps_raw)
    cycle_type_raw = os.environ.get("SETUP_CYCLE_TYPE", "").strip()
    if cycle_type_raw:
        out["cycle_type"] = int(cycle_type_raw)
    max_levels_raw = os.environ.get("SETUP_MAX_LEVELS", "").strip()
    if max_levels_raw:
        out["max_levels"] = int(max_levels_raw)
    return out


def relative_residual(residual_norm: float, r0: float) -> float:
    """||r|| / ||r0||; matches HYPRE relative residual when x0 = 0 (r0 = ||b||)."""
    return float(residual_norm) / max(float(r0), 1e-300)


def converged_relative(residual_norm: float, r0: float, tol: float) -> bool:
    return bool(np.isfinite(float(residual_norm)) and relative_residual(residual_norm, r0) <= float(tol))


def classify_rl_failure(*, residual_norm: float, iterations: int, solve_tol: float, solve_max_cycles: int) -> str:
    # residual_norm here is expected to already be relative (||r||/||r0||).
    if not np.isfinite(float(residual_norm)):
        return "non_finite_residual_norm"
    if float(residual_norm) > float(solve_tol) and int(iterations) >= int(solve_max_cycles):
        return "residual_above_solve_tol;max_cycles_reached_without_convergence"
    if float(residual_norm) > float(solve_tol):
        return "residual_above_solve_tol"
    if int(iterations) >= int(solve_max_cycles):
        return "hit_or_exceeded_solve_max_cycles"
    return ""


# The active Exp44 path used to reach these helpers through the old
# explore_* entrypoints. They now live here so train/eval/pipeline all depend
# on one common module.


def same_action(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
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


def build_grids_from_env() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    grid_n = int(os.environ.get("GRID_N", "20"))
    grid_max = float(os.environ.get("GRID_MAX", "0.95"))
    th_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, grid_n)
    return grid_n, th_grid, mxrs_grid, tr_grid


def build_actions_tune3(*, th_grid, mxrs_grid, tr_grid) -> List[Dict[str, Any]]:
    base_actions = build_actions_th_mxrs_tr(
        th_grid,
        mxrs_grid,
        tr_grid,
        fixed_params={
            "coarsen_type": DEFAULT_SETUP_PARAMS["coarsen_type"],
            "interp_type": DEFAULT_SETUP_PARAMS["interp_type"],
            "agg_interp_type": DEFAULT_SETUP_PARAMS["agg_interp_type"],
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        params = dict(base)
        params["P_max_elmts"] = int(DEFAULT_SETUP_PARAMS["P_max_elmts"])
        params["agg_num_levels"] = int(DEFAULT_SETUP_PARAMS["agg_num_levels"])
        params["agg_interp_type"] = int(DEFAULT_SETUP_PARAMS["agg_interp_type"])
        params["agg_tr"] = float(DEFAULT_SETUP_PARAMS["agg_tr"])
        params["agg_Pmx"] = int(DEFAULT_SETUP_PARAMS["agg_Pmx"])
        actions.append(params)
    return actions


def build_actions_tune5(*, th_grid, mxrs_grid, tr_grid) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    p_max_values = parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)

    base_actions = build_actions_th_mxrs_tr(
        th_grid,
        mxrs_grid,
        tr_grid,
        fixed_params={
            "coarsen_type": DEFAULT_SETUP_PARAMS["coarsen_type"],
            "interp_type": DEFAULT_SETUP_PARAMS["interp_type"],
            "agg_interp_type": DEFAULT_SETUP_PARAMS["agg_interp_type"],
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        for p_max in p_max_values:
            for agg_nl in agg_nl_values:
                params = dict(base)
                params["P_max_elmts"] = int(p_max)
                params["agg_num_levels"] = int(agg_nl)
                params["agg_interp_type"] = int(DEFAULT_SETUP_PARAMS["agg_interp_type"])
                params["agg_tr"] = float(DEFAULT_SETUP_PARAMS["agg_tr"])
                params["agg_Pmx"] = int(DEFAULT_SETUP_PARAMS["agg_Pmx"])
                actions.append(params)
    return actions, p_max_values, agg_nl_values


def build_actions_tune7_categorical(
    *,
    th_grid,
    mxrs_grid,
    tr_grid,
) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[int], List[int], ParameterSpaceSpec]:
    p_max_values = parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)
    coarsen_type_values = parse_int_list_env("COARSEN_TYPE_VALUES", DEFAULT_COARSEN_TYPE_VALUES)
    interp_values = parse_int_list_env("TUNE7_INTERP_TYPES", DEFAULT_TUNE7_INTERP_TYPES)

    parameter_spec = ParameterSpaceSpec(
        (
            ParameterSpec(
                name="strong_threshold",
                kind="continuous",
                values=tuple(float(v) for v in th_grid),
                default=float(DEFAULT_SETUP_PARAMS["strong_threshold"]),
                center=float(DEFAULT_SETUP_PARAMS["strong_threshold"]),
                scale=0.25,
            ),
            ParameterSpec(
                name="max_row_sum",
                kind="continuous",
                values=tuple(float(v) for v in mxrs_grid),
                default=float(DEFAULT_SETUP_PARAMS["max_row_sum"]),
                center=float(DEFAULT_SETUP_PARAMS["max_row_sum"]),
                scale=0.10,
            ),
            ParameterSpec(
                name="trunc_factor",
                kind="continuous",
                values=tuple(float(v) for v in tr_grid),
                default=float(DEFAULT_SETUP_PARAMS["trunc_factor"]),
                center=float(DEFAULT_SETUP_PARAMS["trunc_factor"]),
                scale=0.20,
            ),
            ParameterSpec(
                name="P_max_elmts",
                kind="integer",
                values=tuple(int(v) for v in p_max_values),
                default=int(DEFAULT_SETUP_PARAMS["P_max_elmts"]),
                center=float(DEFAULT_SETUP_PARAMS["P_max_elmts"]),
                scale=4.0,
            ),
            ParameterSpec(
                name="agg_num_levels",
                kind="integer",
                values=tuple(int(v) for v in agg_nl_values),
                default=int(DEFAULT_SETUP_PARAMS["agg_num_levels"]),
                center=float(DEFAULT_SETUP_PARAMS["agg_num_levels"]),
                scale=1.0,
            ),
            ParameterSpec(
                name="coarsen_type",
                kind="categorical",
                values=tuple(int(v) for v in coarsen_type_values),
                default=int(DEFAULT_SETUP_PARAMS["coarsen_type"]),
            ),
            ParameterSpec(
                name="interp_type",
                kind="categorical",
                values=tuple(int(v) for v in interp_values),
                default=int(DEFAULT_SETUP_PARAMS["interp_type"]),
            ),
        )
    )

    actions = build_actions_from_spec(
        parameter_spec,
        fixed_params={
            "agg_interp_type": int(DEFAULT_SETUP_PARAMS["agg_interp_type"]),
            "agg_tr": float(DEFAULT_SETUP_PARAMS["agg_tr"]),
            "agg_Pmx": int(DEFAULT_SETUP_PARAMS["agg_Pmx"]),
        },
    )
    return actions, p_max_values, agg_nl_values, coarsen_type_values, interp_values, parameter_spec


def build_actions_tune7_agg_conditional(
    *,
    th_grid,
    mxrs_grid,
    tr_grid,
) -> Tuple[List[Dict[str, Any]], List[int], List[int], List[float], List[int], ParameterSpaceSpec]:
    p_max_values = parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)
    agg_tr_values = parse_float_list_env("TUNE7_AGG_TR_VALUES", DEFAULT_AGG_TR_VALUES)
    agg_pmx_values = parse_int_list_env("TUNE7_AGG_PMX_VALUES", DEFAULT_AGG_PMX_VALUES)

    active_agg_levels = tuple(int(v) for v in agg_nl_values if int(v) != int(DEFAULT_SETUP_PARAMS["agg_num_levels"]))
    parameter_spec = ParameterSpaceSpec(
        (
            ParameterSpec(
                name="strong_threshold",
                kind="continuous",
                values=tuple(float(v) for v in th_grid),
                default=float(DEFAULT_SETUP_PARAMS["strong_threshold"]),
                center=float(DEFAULT_SETUP_PARAMS["strong_threshold"]),
                scale=0.25,
            ),
            ParameterSpec(
                name="max_row_sum",
                kind="continuous",
                values=tuple(float(v) for v in mxrs_grid),
                default=float(DEFAULT_SETUP_PARAMS["max_row_sum"]),
                center=float(DEFAULT_SETUP_PARAMS["max_row_sum"]),
                scale=0.10,
            ),
            ParameterSpec(
                name="trunc_factor",
                kind="continuous",
                values=tuple(float(v) for v in tr_grid),
                default=float(DEFAULT_SETUP_PARAMS["trunc_factor"]),
                center=float(DEFAULT_SETUP_PARAMS["trunc_factor"]),
                scale=0.20,
            ),
            ParameterSpec(
                name="P_max_elmts",
                kind="integer",
                values=tuple(int(v) for v in p_max_values),
                default=int(DEFAULT_SETUP_PARAMS["P_max_elmts"]),
                center=float(DEFAULT_SETUP_PARAMS["P_max_elmts"]),
                scale=4.0,
            ),
            ParameterSpec(
                name="agg_num_levels",
                kind="integer",
                values=tuple(int(v) for v in agg_nl_values),
                default=int(DEFAULT_SETUP_PARAMS["agg_num_levels"]),
                center=float(DEFAULT_SETUP_PARAMS["agg_num_levels"]),
                scale=1.0,
            ),
            ParameterSpec(
                name="agg_tr",
                kind="continuous",
                values=tuple(float(v) for v in agg_tr_values),
                default=float(DEFAULT_SETUP_PARAMS["agg_tr"]),
                center=float(DEFAULT_SETUP_PARAMS["agg_tr"]),
                scale=0.10,
                active_if={"agg_num_levels": active_agg_levels},
            ),
            ParameterSpec(
                name="agg_Pmx",
                kind="integer",
                values=tuple(int(v) for v in agg_pmx_values),
                default=int(DEFAULT_SETUP_PARAMS["agg_Pmx"]),
                center=float(DEFAULT_SETUP_PARAMS["agg_Pmx"]),
                scale=4.0,
                active_if={"agg_num_levels": active_agg_levels},
            ),
        )
    )

    actions = build_actions_from_spec(
        parameter_spec,
        fixed_params={
            "coarsen_type": int(DEFAULT_SETUP_PARAMS["coarsen_type"]),
            "interp_type": int(DEFAULT_SETUP_PARAMS["interp_type"]),
            "agg_interp_type": int(DEFAULT_SETUP_PARAMS["agg_interp_type"]),
        },
    )
    return actions, p_max_values, agg_nl_values, agg_tr_values, agg_pmx_values, parameter_spec


def ensure_default_arm(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    default_arm_index = next((i for i, a in enumerate(actions) if same_action(a, DEFAULT_SETUP_PARAMS)), None)
    if default_arm_index is None:
        return [*actions, dict(DEFAULT_SETUP_PARAMS)], int(len(actions))
    return actions, int(default_arm_index)


def build_action_space_bundle(*, final_tune_dims: Sequence[int], tune7_variant: str) -> ActionSpaceBundle:
    grid_n, th_grid, mxrs_grid, tr_grid = build_grids_from_env()
    actions_tune3 = build_actions_tune3(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune3, default_arm_index_tune3 = ensure_default_arm(actions_tune3)

    actions_tune5: List[Dict[str, Any]] = []
    p_max_values: List[int] = []
    agg_nl_values: List[int] = []
    default_arm_index_tune5 = -1
    if 5 in {int(v) for v in final_tune_dims}:
        actions_tune5, p_max_values, agg_nl_values = build_actions_tune5(
            th_grid=th_grid,
            mxrs_grid=mxrs_grid,
            tr_grid=tr_grid,
        )
        actions_tune5, default_arm_index_tune5 = ensure_default_arm(actions_tune5)

    actions_tune7: List[Dict[str, Any]] = []
    agg_tr_values: List[float] = []
    agg_pmx_values: List[int] = []
    coarsen_type_values_tune7: List[int] = []
    interp_values_tune7: List[int] = []
    parameter_spec_tune7: Optional[ParameterSpaceSpec] = None
    default_arm_index_tune7 = -1
    if 7 in {int(v) for v in final_tune_dims}:
        if str(tune7_variant).strip().lower() == "agg_conditional":
            (
                actions_tune7,
                p_max_values,
                agg_nl_values,
                agg_tr_values,
                agg_pmx_values,
                parameter_spec_tune7,
            ) = build_actions_tune7_agg_conditional(
                th_grid=th_grid,
                mxrs_grid=mxrs_grid,
                tr_grid=tr_grid,
            )
        else:
            (
                actions_tune7,
                p_max_values,
                agg_nl_values,
                coarsen_type_values_tune7,
                interp_values_tune7,
                parameter_spec_tune7,
            ) = build_actions_tune7_categorical(
                th_grid=th_grid,
                mxrs_grid=mxrs_grid,
                tr_grid=tr_grid,
            )
        actions_tune7, default_arm_index_tune7 = ensure_default_arm(actions_tune7)

    return ActionSpaceBundle(
        grid_n=int(grid_n),
        actions_tune3=actions_tune3,
        actions_tune5=actions_tune5,
        actions_tune7=actions_tune7,
        p_max_values=p_max_values,
        agg_nl_values=agg_nl_values,
        agg_tr_values=agg_tr_values,
        agg_pmx_values=agg_pmx_values,
        coarsen_type_values_tune7=coarsen_type_values_tune7,
        interp_values_tune7=interp_values_tune7,
        parameter_space_tune3={"actions": actions_tune3, "context_dim": DIFCONV_CONTEXT_DIM},
        parameter_space_tune5={"actions": actions_tune5, "context_dim": DIFCONV_CONTEXT_DIM} if actions_tune5 else None,
        parameter_space_tune7={"actions": actions_tune7, "context_dim": DIFCONV_CONTEXT_DIM} if actions_tune7 else None,
        parameter_spec_tune7=parameter_spec_tune7,
        default_arm_index_tune3=int(default_arm_index_tune3),
        default_arm_index_tune5=int(default_arm_index_tune5),
        default_arm_index_tune7=int(default_arm_index_tune7),
    )


class RandomPolicy:
    def __init__(self, actions: Sequence[Dict[str, Any]], seed: int) -> None:
        self._actions = [dict(a) for a in actions]
        self._rng = np.random.default_rng(int(seed))

    def select(self, context, **_):
        idx = int(self._rng.integers(0, len(self._actions)))
        return dict(self._actions[idx]), {"arm_index": idx}

    def update(self, loss, **_):
        return None


def family_seed_map(*, seed: int) -> Dict[str, int]:
    return {
        "Shared LinUCB v4": int(seed + 13003),
    }


def default_branch_label(*, method: str, tune_dim: int, tune7_variant: str) -> str:
    if str(method).strip().lower() == "default":
        return "default (fixed)"
    family = {
        "linucbv4": "Shared LinUCB v4",
    }.get(str(method).strip().lower(), str(method))
    if int(tune_dim) == 7:
        suffix = "tune7" if str(tune7_variant).strip().lower() == "agg_conditional" else "tune7-categorical"
        return f"{family} | {suffix}"
    return f"{family} | tune{int(tune_dim)}"


def build_single_branch(
    *,
    method: str,
    tune_dim: int,
    tune7_variant: str,
    seed: int,
    solver_tol: float,
    solver_max_iter: int,
    bandit_cfg: TestFinalBanditConfig,
    bundle: ActionSpaceBundle,
) -> BranchRun:
    method_key = str(method).strip().lower()
    label = default_branch_label(method=method_key, tune_dim=int(tune_dim), tune7_variant=str(tune7_variant))
    if method_key == "default":
        parameter_space = bundle.parameter_space_tune3 if int(tune_dim) in {3, 0} else (
            bundle.parameter_space_tune5 if int(tune_dim) == 5 else bundle.parameter_space_tune7
        )
        if parameter_space is None:
            raise ValueError(f"No parameter space available for tune_dim={tune_dim}")
        return BranchRun(
            label=label,
            family="default",
            tune_set="default",
            seed=None,
            policy=FixedPolicy(DEFAULT_SETUP_PARAMS),
            parameter_space=parameter_space,
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
        )

    seed_map = family_seed_map(seed=int(seed))
    family = {
        "linucbv4": "Shared LinUCB v4",
    }.get(method_key)
    if family is None:
        raise ValueError(f"Unsupported method: {method}")

    if int(tune_dim) == 3:
        parameter_space = bundle.parameter_space_tune3
        default_arm_index = int(bundle.default_arm_index_tune3)
        parameter_spec = None
        tune_set = "tune3"
    elif int(tune_dim) == 5:
        parameter_space = bundle.parameter_space_tune5
        default_arm_index = int(bundle.default_arm_index_tune5)
        parameter_spec = None
        tune_set = "tune5"
    elif int(tune_dim) == 7:
        parameter_space = bundle.parameter_space_tune7
        default_arm_index = int(bundle.default_arm_index_tune7)
        parameter_spec = bundle.parameter_spec_tune7
        tune_set = "tune7"
    else:
        raise ValueError(f"Unsupported tune_dim: {tune_dim}")
    if parameter_space is None:
        raise ValueError(f"No parameter space available for tune_dim={tune_dim}")

    policy = build_test_final_bandit_policy(
        method=method_key,
        tune_dim=int(tune_dim),
        actions=parameter_space["actions"],
        context_dim=int(parameter_space["context_dim"]),
        seed=int(seed_map[family]),
        default_params=DEFAULT_SETUP_PARAMS,
        default_arm_index=int(default_arm_index),
        parameter_spec=parameter_spec,
        cfg=bandit_cfg,
    )
    return BranchRun(
        label=label,
        family=family,
        tune_set=tune_set,
        seed=int(seed_map[family]),
        policy=policy,
        parameter_space=parameter_space,
        solver_tol=float(solver_tol),
        solver_max_iter=int(solver_max_iter),
    )

def generate_difconv_instances(
    *,
    T: int,
    seed: int,
    grid_choices: Sequence[Tuple[int, int, int]],
    c_min: float,
    c_max: float,
    difconv_a: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    grids = [tuple(int(x) for x in grid) for grid in grid_choices]
    if not grids:
        raise ValueError("grid_choices must be non-empty")
    seed_seq = np.random.SeedSequence(int(seed))
    child_seeds = seed_seq.spawn(int(T))
    instances = []
    for t in range(int(T)):
        grid = grids[t % len(grids)]
        rng = np.random.default_rng(child_seeds[t])
        sampler_kwargs = {
            "nx": int(grid[0]),
            "ny": int(grid[1]),
            "nz": int(grid[2]),
            "n_min": int(max(grid)),
            "n_max": int(max(grid)),
            "c_min": float(c_min),
            "c_max": float(c_max),
            "ax": float(difconv_a[0]),
            "ay": float(difconv_a[1]),
            "az": float(difconv_a[2]),
        }
        mkw, context, _meta = stencil_0_difconv_rl(rng=rng, t=t, trial=0, **sampler_kwargs)
        instances.append((mkw, np.asarray(context, dtype=float)))
    return instances


class FixedPolicy:
    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)

    def select(self, context, **_):
        return dict(self._params), {}

    def update(self, loss, **_):
        return None


class GenericBanditPolicy:
    def __init__(self, model):
        self.model = model

    def select(self, context, **_):
        params = self.model.predict(np.asarray(context, dtype=float))
        info = {}
        if getattr(self.model, "history", None):
            step = self.model.history[-1]
            info = {
                "pred_mean": float(getattr(step, "pred_mean", np.nan)),
                "pred_uncert": float(getattr(step, "pred_uncert", np.nan)),
            }
        return dict(params), info

    def update(self, loss, **_):
        self.model.update(float(loss))


def build_test10_branches(
    *,
    final_tune_dims: Sequence[int],
    tune7_variant: str,
    seed: int,
    solver_tol: float,
    solver_max_iter: int,
    include_default: bool,
    method_filter: str,
    branch_filter: str,
    bandit_cfg: TestFinalBanditConfig,
) -> Tuple[List[BranchRun], ActionSpaceBundle]:
    bundle = build_action_space_bundle(
        final_tune_dims=final_tune_dims,
        tune7_variant=tune7_variant,
    )

    branches: List[BranchRun] = []
    if include_default:
        branches.append(
            build_single_branch(
                method="default",
                tune_dim=3,
                tune7_variant=tune7_variant,
                seed=int(seed),
                solver_tol=float(solver_tol),
                solver_max_iter=int(solver_max_iter),
                bandit_cfg=bandit_cfg,
                bundle=bundle,
            )
        )

    method_specs: List[Tuple[str, int]] = []
    for tune_dim in sorted(int(v) for v in final_tune_dims):
        if tune_dim == 7:
            method_specs.append(("linucbv4", tune_dim))
        else:
            raise ValueError(
                f"The retained Exp44 active path only supports tune_dim=7, got tune_dim={tune_dim}"
            )

    for method, tune_dim in method_specs:
        branches.append(
            build_single_branch(
                method=method,
                tune_dim=int(tune_dim),
                tune7_variant=tune7_variant,
                seed=int(seed),
                solver_tol=float(solver_tol),
                solver_max_iter=int(solver_max_iter),
                bandit_cfg=bandit_cfg,
                bundle=bundle,
            )
        )

    if str(method_filter).strip():
        method_key = normalize_method_name(method_filter)
        branches = [
            branch
            for branch in branches
            if method_key in normalize_method_name(branch.family)
            or method_key in normalize_method_name(branch.label)
        ]
        if not branches:
            raise ValueError(f"METHOD_FILTER={method_filter!r} matched no branches")

    if str(branch_filter).strip():
        filters = [normalize_method_name(part) for part in str(branch_filter).split(",") if part.strip()]
        branches = [
            branch
            for branch in branches
            if any(filt in normalize_method_name(branch.label) for filt in filters)
        ]
        if not branches:
            raise ValueError(f"BRANCH_FILTER={branch_filter!r} matched no branches")

    return branches, bundle


def build_test_final_bandit_policy(
    *,
    method: str,
    tune_dim: int,
    actions: Sequence[Dict[str, Any]],
    context_dim: int,
    seed: int,
    default_params: Dict[str, Any],
    default_arm_index: int,
    parameter_spec: Optional[ParameterSpaceSpec],
    cfg: TestFinalBanditConfig,
) -> GenericBanditPolicy:
    method_key = str(method).strip().lower()
    if int(tune_dim) == 7:
        if parameter_spec is None:
            raise ValueError("parameter_spec is required for tune_dim=7")
        tune7_kwargs: Dict[str, Any] = {
            "parameter_spec": parameter_spec,
            "context_interaction_indices": (1, 2, 3, 4),
            "always_include_arms": [int(default_arm_index)],
            "elite_cache_size": int(cfg.elite_cache_size),
            "initial_guess": [default_params[param.name] for param in parameter_spec.parameters],
            "initial_guess_rounds": 1,
        }
        strategy = str(cfg.tune7_candidate_strategy or "").strip().lower()
        if not strategy:
            strategy = "adaptive_local"
        if strategy == "adaptive_local":
            tune7_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": int(cfg.tune7_candidate_pool_size),
                    "candidate_strategy": "adaptive_local",
                    "candidate_pool_size_burnin": int(cfg.tune7_candidate_pool_size_burnin),
                    "candidate_pool_burnin_rounds": int(cfg.tune7_candidate_pool_burnin_rounds),
                    "alpha_decay_burnin_rounds": int(cfg.tune7_alpha_decay_burnin_rounds),
                    "elite_rank_metric": "mean_loss",
                    "local_neighbor_radius": int(cfg.tune7_local_neighbor_radius),
                    "candidate_local_fraction": float(cfg.tune7_candidate_local_fraction),
                    "candidate_elite_fraction": float(cfg.tune7_candidate_elite_fraction),
                }
            )
        else:
            tune7_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": int(cfg.candidate_pool_size),
                }
            )
        model = SharedLinUCB_AMG_v4(
            actions,
            context_dim=int(context_dim),
            alpha=float(cfg.alpha),
            l2_reg=float(cfg.l2),
            seed=int(seed),
            **tune7_kwargs,
        )
        return GenericBanditPolicy(model)

    raise ValueError(f"The retained Exp44 active path only supports linucbv4, got method={method!r}")


def rolling_success_scale_sec(
    success_runtime_history: Deque[float],
    *,
    b_min_runtime_sec: float,
) -> float:
    vals = np.asarray(list(success_runtime_history), dtype=float)
    finite = vals[np.isfinite(vals) & (vals > 0.0)]
    if finite.size:
        return float(max(float(np.median(finite)), float(b_min_runtime_sec)))
    return float(b_min_runtime_sec)


def runtime_loss_sec_test_final(
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
    solver_fn: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    override_value: float,
) -> float:
    if float(override_value) > 0.0:
        return float(override_value)
    try:
        out = solver_fn(dict(params), dict(mkw))
        rt = float(out.get("runtime", np.nan))
        if np.isfinite(rt) and rt > 0.0:
            return float(rt)
    except Exception:
        pass
    return 1e-3


def run_bandit_step_test_final(
    *,
    policy: Any,
    parameter_space: Dict[str, Any],
    context: np.ndarray,
    solver_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    prev_update_est: float,
    success_runtime_history: Deque[float],
    b_min_runtime_sec: float,
    solver_tol: float,
    cfg: TestFinalBanditConfig,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], int, float]:
    total_runtime = 0.0
    total_overhead = 0.0
    failed_attempts = 0
    local_prev_update_est = float(prev_update_est)
    last_params: Dict[str, Any] | None = None
    last_out: Dict[str, Any] | None = None
    last_upd_sec = 0.0
    last_select_sec = 0.0
    last_loss_eval_sec = 0.0

    for _attempt in range(int(cfg.retry_max_attempts)):
        sel_start = time.perf_counter_ns()
        selected = policy.select(context=context, parameter_space=parameter_space)
        sel_sec = (time.perf_counter_ns() - sel_start) / 1e9
        params = selected[0] if isinstance(selected, tuple) else selected

        out = solver_fn(dict(params))
        total_runtime += float(out["runtime"])

        loss_start = time.perf_counter_ns()
        success_runtime_scale_sec = rolling_success_scale_sec(
            success_runtime_history,
            b_min_runtime_sec=float(b_min_runtime_sec),
        )
        base_loss_sec = float(
            runtime_loss_sec_test_final(
                outcome=out,
                fail_runtime_sec=float(FAIL_RUNTIME_SEC),
                solver_tol=float(solver_tol),
                success_runtime_scale_sec=float(success_runtime_scale_sec),
                failure_penalty_multiplier=float(cfg.failure_penalty_multiplier),
                failure_severity_cap=float(cfg.failure_severity_cap),
                structural_failure_surcharge_multiplier=float(cfg.structural_failure_surcharge_multiplier),
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
        last_select_sec = float(sel_sec)
        last_loss_eval_sec = float(loss_eval_sec)
        local_prev_update_est = float(upd_sec)

        if not bool(out.get("failed", False)):
            rt_success = float(out["runtime"])
            if np.isfinite(rt_success) and rt_success > 0.0:
                success_runtime_history.append(float(rt_success))
            final_out = dict(out)
            final_out["runtime"] = float(total_runtime)
            final_out["failed"] = False
            return (
                last_params,
                final_out,
                {
                    "select_sec": float(last_select_sec),
                    "loss_eval_sec": float(last_loss_eval_sec),
                    "update_sec": float(last_upd_sec),
                    "overhead_sec": float(total_overhead),
                },
                int(failed_attempts),
                float(last_upd_sec),
            )

        failed_attempts += 1

    final_out = dict(
        last_out
        or {
            "runtime": float(FAIL_RUNTIME_SEC),
            "failed": True,
            "residual_norm": float("inf"),
            "iterations": 0,
        }
    )
    final_out["runtime"] = float(total_runtime if total_runtime > 0.0 else FAIL_RUNTIME_SEC)
    final_out["failed"] = True
    return (
        dict(last_params or {}),
        final_out,
        {
            "select_sec": float(last_select_sec),
            "loss_eval_sec": float(last_loss_eval_sec),
            "update_sec": float(last_upd_sec),
            "overhead_sec": float(total_overhead),
        },
        int(failed_attempts),
        float(last_upd_sec),
    )


class SetupAwareSolvePolicyRunner:
    def __init__(self, cfg: SetupAwareRLConfig) -> None:
        self.cfg = cfg
        self.algo = str(cfg.algo).strip().lower()
        self.model_type = str(cfg.model_type).strip().lower()
        self.use_lstm = self.algo == "ppo" and self.model_type == "lstm"
        if self.algo == "dqn":
            self.model = DQN.load(str(cfg.model_path))
        else:
            self.model = RecurrentPPO.load(str(cfg.model_path)) if self.use_lstm else PPO.load(str(cfg.model_path))
        action_shape = getattr(self.model.policy.action_space, "shape", ())
        self.w_only = bool(cfg.w_only or (action_shape and int(action_shape[0]) == 1))
        self.w_init = cfg.w_init
        self.sweeps_init = cfg.sweeps_init
        self.action_mode = str(cfg.action_mode).strip().lower()
        self.discrete_w_values = tuple(float(x) for x in cfg.discrete_w_values)
        self.discrete_joint_actions = tuple((float(w), int(sd), int(su)) for (w, sd, su) in cfg.discrete_joint_actions)
        self.discrete_extended_actions = tuple(
            (
                float(spec[0]),
                int(spec[1]),
                int(spec[2]),
                int(spec[3]),
                int(spec[4]),
                int(spec[5]),
                (-1.0 if len(spec) < 7 else float(spec[6])),
                (-1.0 if len(spec) < 8 else float(spec[7])),
            )
            for spec in cfg.discrete_extended_actions
        )
        self.discrete_blend_alphas = tuple(float(x) for x in cfg.discrete_blend_alphas)
        self.blend_safe_action = (
            float(cfg.blend_safe_action[0]),
            int(cfg.blend_safe_action[1]),
            int(cfg.blend_safe_action[2]),
            int(cfg.blend_safe_action[3]),
            int(cfg.blend_safe_action[4]),
            int(cfg.blend_safe_action[5]),
            float(cfg.blend_safe_action[6]),
            float(cfg.blend_safe_action[7]),
        )
        self.blend_aggr_action = (
            float(cfg.blend_aggr_action[0]),
            int(cfg.blend_aggr_action[1]),
            int(cfg.blend_aggr_action[2]),
            int(cfg.blend_aggr_action[3]),
            int(cfg.blend_aggr_action[4]),
            int(cfg.blend_aggr_action[5]),
            float(cfg.blend_aggr_action[6]),
            float(cfg.blend_aggr_action[7]),
        )
        self.blend_decay_tau = float(cfg.blend_decay_tau)
        self.blend_cutoff_cycles = int(cfg.blend_cutoff_cycles)
        self.sweeps_center = 0.5 * (int(cfg.sweeps_min) + int(cfg.sweeps_max))
        self.sweeps_half = 0.5 * (int(cfg.sweeps_max) - int(cfg.sweeps_min))
        self.sweeps_default = int(
            np.clip(np.round(self.sweeps_center), int(cfg.sweeps_min), int(cfg.sweeps_max))
        )
        self.obs_mode = str(cfg.obs_mode).strip().lower()
        if self.action_mode == "discrete_switch" and len(self.discrete_extended_actions) < 2:
            raise ValueError("discrete_switch requires at least two discrete_extended_actions")
        if self.action_mode == "discrete_blend" and not self.discrete_blend_alphas:
            raise ValueError("discrete_blend requires discrete_blend_alphas")

        self.setup_parameter_spec, _fixed_setup_params = build_setup_parameter_spec(
            tune_dim=int(cfg.tune_dim),
            tune7_variant=str(cfg.tune7_variant),
        )
        self.setup_obs_keys = tuple(self.setup_parameter_spec.parameter_names)
        self.setup_obs_encoder = SetupObsEncoder(
            self.setup_parameter_spec,
            dict(cfg.default_setup_params),
            self.setup_obs_keys,
        )
        self.vec_norm = None
        if cfg.vec_path.exists():
            observe_setup_params = str(cfg.obs_mode).strip().lower() not in {"solve_only"}
            raw_env = DummyVecEnv(
                [
                    lambda: BoomerAMGSetupRelaxEnv(
                        setup_mode="random",
                        tune_dim=int(cfg.tune_dim),
                        tune7_variant=str(cfg.tune7_variant),
                        fixed_grid=tuple(int(x) for x in cfg.fixed_grid),
                        randomize_A=False,
                        randomize_b=False,
                        randomize_grid=False,
                        difconv_c=(1.0, 1.0, 1.0),
                        difconv_c_range=tuple(float(x) for x in cfg.difconv_c_range),
                        difconv_a=(0.0, 0.0, 0.0),
                        w_center=float(cfg.w_center),
                        w_scale=float(cfg.w_scale),
                        sweeps_min=int(cfg.sweeps_min),
                        sweeps_max=int(cfg.sweeps_max),
                        w_only=bool(self.w_only),
                        observe_setup_params=observe_setup_params,
                        bandit_update=False,
                    )
                ]
            )
            self.vec_norm = VecNormalize.load(str(cfg.vec_path), raw_env)
            self.vec_norm.training = False
            self.vec_norm.norm_reward = False

    def _make_obs(
        self,
        *,
        r: float,
        r_prev: float,
        cycle: int,
        case_progress: float,
        last_w: float,
        last_coarse_sweeps: int,
        last_cycle_type: int,
        last_relax_type: int,
        last_outer_weight: float,
        last_add_relax_weight: float,
        switched_to_safe: bool,
        mkw: Dict[str, Any],
        setup_params: Dict[str, Any],
    ) -> np.ndarray:
        eps = 1e-30
        c_norm_div = max(1.0, float(self.cfg.difconv_c_range[1]))
        c_denom = max(float(np.log(float(c_norm_div) + eps)), eps)
        s1 = float(np.log(float(mkw["k"]) + eps) / c_denom)
        s2 = float(np.log(float(mkw["c"]) + eps) / c_denom)
        s3 = float(np.log(float(mkw["a0"]) + eps) / c_denom)

        grid_norm_div = max(float(mkw["nx"]), float(mkw["ny"]), float(mkw["nz"]), 1.0)
        n_denom = max(float(np.log(float(grid_norm_div) + eps)), eps)
        nx_norm = float(np.log(float(mkw["nx"]) + eps) / n_denom)
        ny_norm = float(np.log(float(mkw["ny"]) + eps) / n_denom)
        nz_norm = float(np.log(float(mkw["nz"]) + eps) / n_denom)

        solve_obs = build_policy_obs(
            r=float(r),
            r_prev=float(r_prev),
            cycle=int(cycle),
            max_cycles=int(self.cfg.solve_max_cycles),
            coeff_triplet=(s1, s2, s3),
            grid_triplet=(nx_norm, ny_norm, nz_norm),
            last_w=float(last_w),
        ).astype(np.float32)
        setup_obs = self.setup_obs_encoder.encode(dict(setup_params)).astype(np.float32)
        action_state = np.array(
            [
                float(last_coarse_sweeps),
                float(last_cycle_type),
                float(last_relax_type),
                float(last_outer_weight),
                float(last_add_relax_weight),
                1.0 if switched_to_safe else 0.0,
            ],
            dtype=np.float32,
        )
        compact_solve_obs = solve_obs[[0, 1, 2, 9]]
        if self.obs_mode == "full":
            obs = np.concatenate([solve_obs, setup_obs], axis=0).astype(np.float32)
        elif self.obs_mode == "solve_only":
            obs = solve_obs.astype(np.float32)
        elif self.obs_mode == "full_action":
            obs = np.concatenate([solve_obs, action_state, setup_obs], axis=0).astype(np.float32)
        elif self.obs_mode == "cycle_setup":
            obs = np.concatenate([compact_solve_obs, setup_obs], axis=0).astype(np.float32)
        elif self.obs_mode == "cycle_action_setup":
            obs = np.concatenate([compact_solve_obs, action_state, setup_obs], axis=0).astype(np.float32)
        elif self.obs_mode == "cycle_only":
            obs = compact_solve_obs.astype(np.float32)
        else:
            raise ValueError(f"Unknown obs_mode: {self.obs_mode}")
        if bool(self.cfg.obs_include_trace_progress):
            obs = np.concatenate([obs, np.asarray([float(case_progress)], dtype=np.float32)], axis=0).astype(np.float32)
        if self.vec_norm is not None:
            obs = self.vec_norm.normalize_obs(obs)
        return obs

    def _map_action(
        self,
        action: np.ndarray,
        cycle: int,
        last_w: float,
        switched_to_safe: bool,
        case_progress: float = 0.0,
    ) -> Tuple[float, int, int, int, int, int, float, float]:
        if self.action_mode == "discrete_switch":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, 1))
            if int(idx) == 1:
                switched_to_safe = True
            chosen_idx = 1 if switched_to_safe else 0
            return self.discrete_extended_actions[chosen_idx]
        if self.action_mode == "discrete_extended":
            if not self.discrete_extended_actions:
                raise ValueError("discrete_extended_actions must be non-empty when action_mode='discrete_extended'")
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_extended_actions) - 1))
            return self.discrete_extended_actions[idx]
        if self.action_mode == "discrete_joint":
            if not self.discrete_joint_actions:
                raise ValueError("discrete_joint_actions must be non-empty when action_mode='discrete_joint'")
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_joint_actions) - 1))
            w, sd, su = self.discrete_joint_actions[idx]
            return float(w), int(sd), int(su), 1, -1, -1, -1.0, -1.0
        if self.action_mode == "discrete_w":
            if not self.discrete_w_values:
                raise ValueError("discrete_w_values must be non-empty when action_mode='discrete_w'")
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_w_values) - 1))
            w = float(self.discrete_w_values[idx])
            return w, self.sweeps_default, self.sweeps_default, 1, -1, -1, -1.0, -1.0
        if self.action_mode == "discrete_blend":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = int(np.clip(idx, 0, len(self.discrete_blend_alphas) - 1))
            alpha = float(self.discrete_blend_alphas[idx])
            if self.blend_cutoff_cycles >= 0 and int(cycle) >= int(self.blend_cutoff_cycles):
                alpha = 0.0
            if self.blend_decay_tau > 0.0:
                alpha *= float(np.exp(-float(cycle) / float(self.blend_decay_tau)))
            if self.cfg.blend_progress_end > self.cfg.blend_progress_start:
                if float(case_progress) <= float(self.cfg.blend_progress_start):
                    alpha = 0.0
                else:
                    ramp = (float(case_progress) - float(self.cfg.blend_progress_start)) / (
                        float(self.cfg.blend_progress_end) - float(self.cfg.blend_progress_start)
                    )
                    alpha *= float(np.clip(ramp, 0.0, 1.0))
            alpha = float(np.clip(alpha, 0.0, 1.0))
            sw, sdown, sup, ssc, sct, srt, sow, sarw = self.blend_safe_action
            aw, adown, aup, asc, act, art, aow, aarw = self.blend_aggr_action
            w = float(sw + alpha * (aw - sw))
            sd = int(np.clip(np.rint(sdown + alpha * (adown - sdown)), self.cfg.sweeps_min, self.cfg.sweeps_max))
            su = int(np.clip(np.rint(sup + alpha * (aup - sup)), self.cfg.sweeps_min, self.cfg.sweeps_max))
            sc = max(1, int(np.rint(ssc + alpha * (asc - ssc))))
            ct = int(sct if alpha < 0.5 or int(act) < 0 else act)
            rt = int(srt if alpha < 0.5 or int(art) < 0 else art)
            if float(sow) < 0.0 and float(aow) < 0.0:
                ow = -1.0
            elif float(sow) < 0.0 or float(aow) < 0.0:
                ow = float(sow if alpha < 0.5 else aow)
            else:
                ow = float(sow + alpha * (aow - sow))
            if float(sarw) < 0.0 and float(aarw) < 0.0:
                arw = -1.0
            elif float(sarw) < 0.0 or float(aarw) < 0.0:
                arw = float(sarw if alpha < 0.5 else aarw)
            else:
                arw = float(sarw + alpha * (aarw - sarw))
            return w, sd, su, sc, ct, rt, ow, arw
        if self.action_mode == "continuous_residual":
            w, sd, su = decode_policy_action_residual(
                action,
                w_only=self.w_only,
                w_center=float(self.cfg.w_center),
                w_scale=float(self.cfg.w_scale),
                w_min=float(self.cfg.w_global_min),
                w_max=float(self.cfg.w_global_max),
                sweeps_min=int(self.cfg.sweeps_min),
                sweeps_max=int(self.cfg.sweeps_max),
                cycle=int(cycle),
                last_w=float(last_w),
                w_init=self.w_init,
                sweeps_init=self.sweeps_init,
            )
            return float(w), int(sd), int(su), 1, -1, -1, -1.0, -1.0
        if self.action_mode == "continuous_hierarchical":
            w, sd, su = decode_policy_action_hierarchical(
                action,
                w_only=self.w_only,
                sweeps_min=int(self.cfg.sweeps_min),
                sweeps_max=int(self.cfg.sweeps_max),
            )
            return float(w), int(sd), int(su), 1, -1, -1, -1.0, -1.0
        w, sd, su = decode_policy_action(
            action,
            w_only=self.w_only,
            w_center=float(self.cfg.w_center),
            w_scale=float(self.cfg.w_scale),
            sweeps_min=int(self.cfg.sweeps_min),
            sweeps_max=int(self.cfg.sweeps_max),
            cycle=int(cycle),
            last_w=float(last_w),
            w_init=self.w_init,
            sweeps_init=self.sweeps_init,
            w_smooth_alpha=0.0,
        )
        return float(w), int(sd), int(su), 1, -1, -1, -1.0, -1.0

    def run(self, env, *, mkw: Dict[str, Any], setup_params: Dict[str, Any], case_progress: float = 0.0) -> Dict[str, Any]:
        r0 = float(env.r0)
        r_prev_obs = float(env.r0)
        r_curr = float(env.r0)
        last_w = float(self.w_init) if self.w_init is not None else float(self.cfg.w_center)
        solve_runtime = 0.0
        infer_runtime = 0.0
        cycles = 0
        action_counts: Dict[int, int] = {}
        lstm_state = None
        episode_start = np.ones((1,), dtype=bool)
        last_sd = self.sweeps_default
        last_su = self.sweeps_default
        last_sc = 1
        last_ct = -1
        last_rt = -1
        last_ow = -1.0
        last_arw = -1.0
        switched_to_safe = False
        r_new = float(r_curr)

        for cycle in range(int(self.cfg.solve_max_cycles)):
            obs = self._make_obs(
                r=r_curr,
                r_prev=r_prev_obs,
                cycle=cycle,
                case_progress=float(case_progress),
                last_w=last_w,
                last_coarse_sweeps=last_sc,
                last_cycle_type=last_ct,
                last_relax_type=last_rt,
                last_outer_weight=last_ow,
                last_add_relax_weight=last_arw,
                switched_to_safe=switched_to_safe,
                mkw=mkw,
                setup_params=setup_params,
            )
            infer_start = time.perf_counter()
            if self.use_lstm:
                action, lstm_state = self.model.predict(
                    obs,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=True,
                )
            else:
                action, _ = self.model.predict(obs, deterministic=True)
            infer_runtime += float(time.perf_counter() - infer_start)

            action_idx: Optional[int] = None
            if self.action_mode == "discrete_switch":
                chosen = int(np.asarray(action).reshape(-1)[0])
                chosen = int(np.clip(chosen, 0, 1))
                if chosen == 1:
                    switched_to_safe = True
                action_idx = 1 if switched_to_safe else 0
                w, sd, su, sc, ct, rt, ow, arw = self.discrete_extended_actions[action_idx]
            else:
                w, sd, su, sc, ct, rt, ow, arw = self._map_action(
                    action,
                    cycle,
                    last_w,
                    switched_to_safe,
                    case_progress=float(case_progress),
                )
                if self.action_mode in {"discrete_extended", "discrete_joint", "discrete_w", "discrete_blend"}:
                    action_idx = int(np.asarray(action).reshape(-1)[0])
            if action_idx is not None:
                action_idx = int(action_idx)
                action_counts[action_idx] = int(action_counts.get(action_idx, 0)) + 1
            r_new, dt = env.step_rl(
                relax_weight=w,
                sweeps_down=sd,
                sweeps_up=su,
                coarse_sweeps=sc,
                cycle_type=(None if int(ct) < 0 else int(ct)),
                relax_type=(None if int(rt) < 0 else int(rt)),
                outer_weight=(None if float(ow) < 0.0 else float(ow)),
                add_relax_weight=(None if float(arw) < 0.0 else float(arw)),
            )
            solve_runtime += float(dt)
            cycles = cycle + 1
            last_w = float(w)
            last_sd = int(sd)
            last_su = int(su)
            last_sc = int(sc)
            last_ct = int(ct)
            last_rt = int(rt)
            last_ow = float(ow)
            last_arw = float(arw)
            if converged_relative(r_new, r0, self.cfg.solve_tol):
                r_curr = float(r_new)
                break
            r_prev_obs = float(r_curr)
            r_curr = float(r_new)
            episode_start[...] = False
        else:
            r_curr = float(r_new)

        rel_res = relative_residual(r_curr, r0)
        return {
            "solve_runtime": float(solve_runtime),
            "infer_runtime": float(infer_runtime),
            "residual_norm": float(rel_res),
            "iterations": int(cycles),
            "failed": not (
                np.isfinite(rel_res)
                and rel_res <= float(self.cfg.solve_tol)
                and cycles < int(self.cfg.solve_max_cycles)
            ),
            "final_w": float(last_w),
            "final_sweeps_down": int(last_sd),
            "final_sweeps_up": int(last_su),
            "final_coarse_sweeps": int(last_sc),
            "final_cycle_type": int(last_ct),
            "final_relax_type": int(last_rt),
            "final_outer_weight": float(last_ow),
            "final_add_relax_weight": float(last_arw),
            "action_counts": dict(action_counts),
        }


def solve_fixed_w_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    w: float,
    sweeps_down: int,
    sweeps_up: int,
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    try:
        params = augment_setup_params(params)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            solve_runtime = 0.0
            r0 = float(env.r0)
            residual_norm = float(env.r0)
            iterations = 0
            for cycle in range(int(solve_max_cycles)):
                residual_norm, dt = env.step_rl(
                    relax_weight=float(w),
                    sweeps_down=int(sweeps_down),
                    sweeps_up=int(sweeps_up),
                )
                solve_runtime += float(dt)
                iterations = cycle + 1
                if converged_relative(residual_norm, r0, solve_tol):
                    break
        rel_res = relative_residual(residual_norm, r0)
        failure_reason = classify_rl_failure(
            residual_norm=float(rel_res),
            iterations=int(iterations),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
        )
        return {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": 0.0,
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "residual_norm": float(rel_res),
            "iterations": int(iterations),
            "final_w": float(w),
            "final_sweeps_down": int(sweeps_down),
            "final_sweeps_up": int(sweeps_up),
        }
    except Exception as exc:
        return {
            "runtime": float(FAIL_RUNTIME_SEC),
            "setup_runtime": float(FAIL_RUNTIME_SEC),
            "solve_runtime": 0.0,
            "infer_runtime": 0.0,
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(solve_max_cycles),
            "final_w": float(w),
            "final_sweeps_down": int(sweeps_down),
            "final_sweeps_up": int(sweeps_up),
        }


def solve_schedule_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    schedule: Sequence[Tuple[int, float, int, int]],
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    try:
        params = augment_setup_params(params)
        schedule_sorted = sorted((int(end), float(w), int(sd), int(su)) for end, w, sd, su in schedule)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            solve_runtime = 0.0
            r0 = float(env.r0)
            residual_norm = float(env.r0)
            iterations = 0
            last_w = float("nan")
            last_sd = -1
            last_su = -1
            for cycle in range(int(solve_max_cycles)):
                chosen = schedule_sorted[-1]
                for end_cycle, w, sd, su in schedule_sorted:
                    if cycle < int(end_cycle):
                        chosen = (end_cycle, w, sd, su)
                        break
                _end, w, sd, su = chosen
                residual_norm, dt = env.step_rl(
                    relax_weight=float(w),
                    sweeps_down=int(sd),
                    sweeps_up=int(su),
                )
                solve_runtime += float(dt)
                iterations = cycle + 1
                last_w = float(w)
                last_sd = int(sd)
                last_su = int(su)
                if converged_relative(residual_norm, r0, solve_tol):
                    break
        rel_res = relative_residual(residual_norm, r0)
        failure_reason = classify_rl_failure(
            residual_norm=float(rel_res),
            iterations=int(iterations),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
        )
        return {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(solve_runtime),
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "residual_norm": float(rel_res),
            "iterations": int(iterations),
            "final_w": float(last_w),
            "final_sweeps_down": int(last_sd),
            "final_sweeps_up": int(last_su),
        }
    except Exception as exc:
        return {
            "runtime": float(FAIL_RUNTIME_SEC),
            "setup_runtime": float(FAIL_RUNTIME_SEC),
            "solve_runtime": 0.0,
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(solve_max_cycles),
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }


def solve_setup_aware_rl_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    solve_policy: SetupAwareSolvePolicyRunner,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]],
    classify_rl_failure: Callable[..., str],
    solve_max_cycles: int,
    case_progress: float = 0.0,
) -> Dict[str, Any]:
    # Bandit and RL are combined sequentially here:
    # 1) setup params have already been chosen upstream by the mature bandit
    # 2) prepare_rl(params=...) builds that setup in Hypre
    # 3) solve_policy.run(...) lets RL control solve-phase weights/sweeps on
    #    top of the fixed setup
    try:
        params = augment_params(params)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            rl_out = solve_policy.run(env, mkw=mkw, setup_params=params, case_progress=float(case_progress))
        res_norm = float(rl_out["residual_norm"])
        iters = int(rl_out["iterations"])
        total_runtime = float(prep.setup_runtime_sec + rl_out["solve_runtime"])
        failure_reason = classify_rl_failure(residual_norm=res_norm, iterations=iters)
        converged = not bool(failure_reason)
        return {
            "runtime": total_runtime,
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(rl_out["solve_runtime"]),
            "infer_runtime": float(rl_out["infer_runtime"]),
            "failed": (not converged),
            "failure_reason": failure_reason,
            "residual_norm": res_norm,
            "iterations": iters,
            "final_w": float(rl_out["final_w"]),
            "final_sweeps_down": int(rl_out["final_sweeps_down"]),
            "final_sweeps_up": int(rl_out["final_sweeps_up"]),
            "action_counts": dict(rl_out.get("action_counts", {})),
        }
    except Exception as exc:
        return {
            "runtime": float(FAIL_RUNTIME_SEC),
            "setup_runtime": float(FAIL_RUNTIME_SEC),
            "solve_runtime": 0.0,
            "infer_runtime": 0.0,
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(solve_max_cycles),
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
            "action_counts": {},
            "structural_fail": True,
        }


def classify_no_rl_failure(*, residual_norm: float, iterations: int, solver_tol: float, solver_max_iter: int) -> str:
    reasons: List[str] = []
    if not np.isfinite(float(residual_norm)):
        reasons.append("non_finite_residual_norm")
    elif float(residual_norm) > float(solver_tol):
        reasons.append("residual_above_solver_tol")
    if int(iterations) >= int(solver_max_iter):
        reasons.append("max_iter_reached_without_convergence")
    return ";".join(reasons)


def solve_no_rl_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    solver_tol: float,
    solver_max_iter: int,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    try:
        res = solve(
            params=augment_params(dict(params)),
            tol=float(solver_tol),
            max_iter=int(solver_max_iter),
            **mkw,
        )
        residual_norm = float(res.residual_norm)
        iterations = int(res.iterations)
        failure_reason = classify_no_rl_failure(
            residual_norm=float(residual_norm),
            iterations=int(iterations),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
        )
        return {
            "runtime": float(res.runtime_sec),
            "setup_runtime": float(res.setup_runtime_sec),
            "solve_runtime": float(res.solve_runtime_sec),
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "structural_fail": False,
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }
    except Exception as exc:
        return {
            "runtime": float(FAIL_RUNTIME_SEC),
            "setup_runtime": float(FAIL_RUNTIME_SEC),
            "solve_runtime": 0.0,
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(solver_max_iter),
            "structural_fail": True,
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }


def fixed_trace(
    *,
    T: int,
    grid: Tuple[int, int, int],
    seed: int,
    tune_dim: int,
    bandit_method: str,
    solve_mode: str = "no_rl",
    solve_policy: Optional[SetupAwareSolvePolicyRunner] = None,
) -> Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]:
    difconv_a = tuple(float(x) for x in os.environ.get("DIFCONV_A", "0,0,0").split(","))
    instances = generate_difconv_instances(
        T=int(T),
        seed=int(seed),
        grid_choices=[tuple(int(x) for x in grid)],
        c_min=float(os.environ.get("C_MIN", "1.0")),
        c_max=float(os.environ.get("C_MAX", "1000.0")),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )
    bandit_cfg = default_test_final_bandit_config_from_env()
    branches, _bundle = build_test10_branches(
        final_tune_dims=[int(tune_dim)],
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        seed=int(seed),
        solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
        solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
        include_default=False,
        method_filter=str(bandit_method),
        branch_filter=os.environ.get(
            "BRANCH_FILTER",
            default_branch_label(
                method=str(bandit_method),
                tune_dim=int(tune_dim),
                tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
            ),
        ),
        bandit_cfg=bandit_cfg,
    )
    if len(branches) != 1:
        labels = ", ".join(branch.label for branch in branches)
        raise ValueError(f"Expected one branch, got {len(branches)}: {labels}")
    branch = branches[0]
    prev_update_est = 0.0
    success_runtime_history = deque(maxlen=max(1, int(bandit_cfg.failure_scale_window)))
    failure_scale_min_runtime_sec = None
    trace: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for mkw, context in instances:
        if str(solve_mode).strip().lower() == "no_rl":
            def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_no_rl_case(
                    params=selected_params,
                    mkw=dict(mkw),
                    solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
                    solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
                    augment_params=augment_setup_params,
                )
            solver_tol = float(os.environ.get("SOLVER_TOL", "1e-6"))
        elif str(solve_mode).strip().lower() == "rl":
            if solve_policy is None:
                raise ValueError("solve_policy is required when solve_mode='rl'")

            def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_setup_aware_rl_case(
                    params=selected_params,
                    mkw=dict(mkw),
                    solve_policy=solve_policy,
                    augment_params=augment_setup_params,
                    classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
                        residual_norm=float(residual_norm),
                        iterations=int(iterations),
                        solve_tol=float(solve_policy.cfg.solve_tol),
                        solve_max_cycles=int(solve_policy.cfg.solve_max_cycles),
                    ),
                    solve_max_cycles=int(solve_policy.cfg.solve_max_cycles),
                )
            solver_tol = float(solve_policy.cfg.solve_tol)
        else:
            raise ValueError(f"Unsupported solve_mode: {solve_mode}")

        if failure_scale_min_runtime_sec is None:
            failure_scale_min_runtime_sec = compute_failure_scale_min_runtime_sec(
                solver_fn=solver_fn,
                params=DEFAULT_SETUP_PARAMS,
                mkw=dict(mkw),
                override_value=float(bandit_cfg.failure_scale_min_runtime_sec_override),
            )

        params, _out, _timing, _failed_attempts, prev_update_est = run_bandit_step_test_final(
            policy=branch.policy,
            parameter_space=branch.parameter_space,
            context=np.asarray(context, dtype=float),
            solver_fn=solver_fn,
            prev_update_est=float(prev_update_est),
            success_runtime_history=success_runtime_history,
            b_min_runtime_sec=float(failure_scale_min_runtime_sec),
            solver_tol=float(solver_tol),
            cfg=bandit_cfg,
        )
        trace.append((dict(mkw), dict(params)))
    return trace


def eval_runner(runner: SetupAwareSolvePolicyRunner, trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    solve_tol = float(os.environ.get("SOLVE_TOL", "1e-6"))
    solve_max_cycles = int(os.environ.get("SOLVE_MAX_CYCLES", "50"))
    vals = []
    fails = 0
    action_hist: Dict[int, int] = {}
    denom = max(1, len(trace) - 1)
    for idx_case, (mkw, params) in enumerate(trace):
        out = solve_setup_aware_rl_case(
            params=params,
            mkw=dict(mkw),
            solve_policy=runner,
            augment_params=augment_setup_params,
            classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
                residual_norm=float(residual_norm),
                iterations=int(iterations),
                solve_tol=float(solve_tol),
                solve_max_cycles=int(solve_max_cycles),
            ),
            solve_max_cycles=int(solve_max_cycles),
            case_progress=float(idx_case) / float(denom),
        )
        vals.append(out)
        fails += int(bool(out["failed"]))
        for idx, count in dict(out.get("action_counts", {})).items():
            key = int(idx)
            action_hist[key] = int(action_hist.get(key, 0)) + int(count)
    return {
        "mean_runtime": float(np.mean([v["runtime"] for v in vals])),
        "mean_setup_runtime": float(np.mean([v["setup_runtime"] for v in vals])),
        "mean_solve_runtime": float(np.mean([v["solve_runtime"] for v in vals])),
        "failed_count": int(fails),
        "mean_iterations": float(np.mean([v["iterations"] for v in vals])),
        "mean_final_w": float(np.mean([v["final_w"] for v in vals if np.isfinite(v["final_w"])])),
        "action_hist": {int(k): int(action_hist[k]) for k in sorted(action_hist)},
    }


def alloc_branch_metrics(*, labels: Sequence[str], T: int) -> Dict[str, Any]:
    return {
        "runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "setup_runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "solve_runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "infer_runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "overhead_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "select_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "loss_eval_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "update_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "failed_flags": {label: np.zeros(T, dtype=bool) for label in labels},
        "failure_reason": {label: np.full(T, "", dtype=object) for label in labels},
        "iterations": {label: np.zeros(T, dtype=int) for label in labels},
        "residual_norm": {label: np.full(T, np.nan, dtype=float) for label in labels},
        "final_w": {label: np.full(T, np.nan, dtype=float) for label in labels},
        "final_sweeps_down": {label: np.full(T, -1, dtype=int) for label in labels},
        "final_sweeps_up": {label: np.full(T, -1, dtype=int) for label in labels},
        "traces": {label: init_param_trace(TRACE_KEYS_FINAL, T) for label in labels},
        "prev_update_est": {label: 0.0 for label in labels},
        "success_runtime_history": {
            label: deque(maxlen=max(1, int(default_test_final_bandit_config_from_env().failure_scale_window)))
            for label in labels
        },
    }


def run_interleaved_branch_scenario(
    *,
    phase_label: str,
    branches: Sequence[BranchRun],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    solve_mode: str,
    solve_policy: Optional[SetupAwareSolvePolicyRunner],
    permutation_seed: int,
    solve_max_cycles: int,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]],
    classify_rl_failure: Callable[..., str],
    bandit_cfg: TestFinalBanditConfig,
    progress_every: Optional[int],
    solve_policy_start_case: int = 0,
) -> Dict[str, Any]:
    labels = [branch.label for branch in branches]
    metrics = alloc_branch_metrics(labels=labels, T=len(instances))
    phase_start_time = time.perf_counter()
    rng_order = np.random.default_rng(int(permutation_seed))
    failure_records: List[Dict[str, Any]] = []
    failure_scale_min_runtime_sec: Dict[str, Optional[float]] = {label: None for label in labels}

    branch_by_label = {branch.label: branch for branch in branches}
    branch_meta = {
        branch.label: {
            "method": str(branch.family),
            "tune_set": str(branch.tune_set),
            "fixed_params": dict(getattr(branch.policy, "_params", {})),
        }
        for branch in branches
    }

    for local_t, (mkw, context) in enumerate(instances):
        order = rng_order.permutation(len(branches))
        for idx in order:
            branch = branches[int(idx)]
            label = branch.label
            if solve_mode == "rl":
                if solve_policy is None:
                    raise ValueError("solve_policy is required for solve_mode='rl'")

                if int(local_t) < int(solve_policy_start_case):

                    def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                        return solve_no_rl_case(
                            params=selected_params,
                            mkw=dict(mkw),
                            solver_tol=float(branch.solver_tol),
                            solver_max_iter=int(branch.solver_max_iter),
                            augment_params=augment_params,
                        )

                else:

                    def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                        return solve_setup_aware_rl_case(
                            params=selected_params,
                            mkw=dict(mkw),
                            solve_policy=solve_policy,
                            augment_params=augment_params,
                            classify_rl_failure=classify_rl_failure,
                            solve_max_cycles=int(solve_max_cycles),
                        )

            elif solve_mode == "no_rl":

                def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                    return solve_no_rl_case(
                        params=selected_params,
                        mkw=dict(mkw),
                        solver_tol=float(branch.solver_tol),
                        solver_max_iter=int(branch.solver_max_iter),
                        augment_params=augment_params,
                    )

            else:
                raise ValueError(f"Unsupported solve_mode: {solve_mode}")

            if failure_scale_min_runtime_sec[label] is None:
                failure_scale_min_runtime_sec[label] = compute_failure_scale_min_runtime_sec(
                    solver_fn=solver_fn,
                    params=DEFAULT_SETUP_PARAMS,
                    mkw=dict(mkw),
                    override_value=float(bandit_cfg.failure_scale_min_runtime_sec_override),
                )

            params, out, timing, _failed_attempts, last_update_sec = run_bandit_step_test_final(
                policy=branch.policy,
                parameter_space=branch.parameter_space,
                context=np.asarray(context, dtype=float),
                solver_fn=solver_fn,
                prev_update_est=float(metrics["prev_update_est"][label]),
                success_runtime_history=metrics["success_runtime_history"][label],
                b_min_runtime_sec=float(failure_scale_min_runtime_sec[label]),
                solver_tol=float(branch.solver_tol if solve_mode == "no_rl" else solve_policy.cfg.solve_tol),
                cfg=bandit_cfg,
            )

            metrics["runtime_sec"][label][local_t] = float(out["runtime"])
            metrics["setup_runtime_sec"][label][local_t] = float(out["setup_runtime"])
            metrics["solve_runtime_sec"][label][local_t] = float(out["solve_runtime"])
            metrics["infer_runtime_sec"][label][local_t] = float(out.get("infer_runtime", 0.0))
            metrics["overhead_sec"][label][local_t] = float(timing["overhead_sec"])
            metrics["select_sec"][label][local_t] = float(timing["select_sec"])
            metrics["loss_eval_sec"][label][local_t] = float(timing["loss_eval_sec"])
            metrics["update_sec"][label][local_t] = float(timing["update_sec"])
            metrics["failed_flags"][label][local_t] = bool(out.get("failed", False))
            metrics["failure_reason"][label][local_t] = str(out.get("failure_reason", ""))
            metrics["iterations"][label][local_t] = int(out.get("iterations", 0))
            metrics["residual_norm"][label][local_t] = float(out.get("residual_norm", np.nan))
            metrics["final_w"][label][local_t] = float(out.get("final_w", np.nan))
            metrics["final_sweeps_down"][label][local_t] = int(out.get("final_sweeps_down", -1))
            metrics["final_sweeps_up"][label][local_t] = int(out.get("final_sweeps_up", -1))
            record_param_trace(metrics["traces"][label], t=local_t, params=params, keys=TRACE_KEYS_FINAL)
            metrics["prev_update_est"][label] = float(last_update_sec)

            if bool(out.get("failed", False)):
                failure_records.append(
                    {
                        "phase": "dual_permuted",
                        "solve_mode": str(solve_mode),
                        "t": int(local_t + 1),
                        "method": str(branch.family),
                        "label": str(label),
                        "tune_set": str(branch.tune_set),
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
                        "fixed_params": dict(branch_meta.get(label, {}).get("fixed_params", {})),
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
            prefix=str(phase_label),
            every=progress_every,
            start_time=phase_start_time,
        )

    metrics["failure_records"] = failure_records
    metrics["branch_meta"] = branch_meta
    metrics["labels"] = labels
    metrics["branches"] = {label: branch_by_label[label] for label in labels}
    return metrics
