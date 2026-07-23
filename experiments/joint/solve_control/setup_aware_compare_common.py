from __future__ import annotations

import _project_paths  # noqa: F401

import copy
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from solve.controllers.ppo import (
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    _SpaceOnlyEnv,
)
from setup.space import (
    SetupConfigurationSpace,
    SetupObsEncoder,
    build_setup_parameter_spec,
    build_setup_param_space,
)
from setup.learners.linucb import run_same_context_setup_reselection
from setup.learners.common import (
    AOTCandidateSchedule,
    FactorizedActionFeatureCache,
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    ParameterSpec,
    SharedSetupLearnerSpec,
    resolve_tune7_candidate_strategy,
)
from setup.registry import (
    build_online_setup_learner,
    make_setup_learner_spec,
)
from hypre.bindings import (
    AMGNativeError,
    SolveStatus,
    augment_setup_params,
    create_env,
    run_with_default_fallback,
    solve,
)
from solve.core.outcomes import classify_rl_failure
from problems.amg import DIFCONV_CONTEXT_DIM
from problems.streams import generate_difconv_instances as _generate_difconv_instances
from setup.utils.setup_amg import (
    build_actions_from_spec,
    build_actions_th_mxrs_tr,
    init_param_trace,
    progress_bar,
    record_param_trace,
)


EXP44_MATRIX_GRID_N = 40
EXP44_SETUP_PARAM_RESOLUTION = 20
EXP44_TUNE7_CATEGORICAL_ACTION_COUNT = 2_880_000
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


def clone_branch_for_independent_updates(branch: BranchRun) -> BranchRun:
    """Clone online learner state while sharing immutable action-space storage."""
    source_policy = branch.policy
    source_model = getattr(source_policy, "model", None)
    if source_model is not None and hasattr(
        source_model, "clone_for_independent_updates"
    ):
        cloned_policy = type(source_policy)(
            source_model.clone_for_independent_updates()
        )
    else:
        cloned_policy = copy.deepcopy(source_policy)
    return BranchRun(
        label=str(branch.label),
        family=str(branch.family),
        tune_set=str(branch.tune_set),
        seed=branch.seed,
        policy=cloned_policy,
        parameter_space=branch.parameter_space,
        solver_tol=float(branch.solver_tol),
        solver_max_iter=int(branch.solver_max_iter),
    )


@dataclass(frozen=True)
class ActionSpaceBundle:
    param_resolution: int
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


def validate_expected_setup_action_count(branch: BranchRun) -> int:
    model = getattr(branch.policy, "model", branch.policy)
    observed = int(model.K)
    raw_expected = os.environ.get("EXPECTED_SETUP_ACTION_COUNT", "").strip()
    if raw_expected:
        expected = int(raw_expected)
        if observed != expected:
            raise ValueError(
                "Setup action-count mismatch: "
                f"expected {expected:,}, observed {observed:,}. "
                "MATRIX_GRID_N and SETUP_PARAM_RESOLUTION must remain independent."
            )
    return observed


@dataclass(frozen=True)
class TestFinalBanditConfig:
    alpha: float
    l2: float
    candidate_pool_size: int
    elite_cache_size: int
    tune7_candidate_pool_size: int
    tune7_candidate_pool_size_burnin: int
    tune7_candidate_pool_burnin_rounds: int
    tune7_alpha_decay_burnin_rounds: int
    tune7_candidate_strategy: str
    tune7_local_neighbor_radius: int
    tune7_candidate_local_fraction: float
    tune7_candidate_elite_fraction: float


def default_test_final_bandit_config_from_env() -> TestFinalBanditConfig:
    return TestFinalBanditConfig(
        alpha=float(os.environ.get("ALPHA", "1.0")),
        l2=float(os.environ.get("L2", "1.0")),
        candidate_pool_size=int(os.environ.get("CANDIDATE_POOL_SIZE", "512")),
        elite_cache_size=int(os.environ.get("ELITE_CACHE_SIZE", "64")),
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
    param_resolution = int(
        os.environ.get("SETUP_PARAM_RESOLUTION", str(EXP44_SETUP_PARAM_RESOLUTION))
    )
    grid_max = float(os.environ.get("GRID_MAX", "0.95"))
    th_grid = np.linspace(0.0, grid_max, param_resolution)
    mxrs_grid = np.linspace(0.0, grid_max, param_resolution)
    mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, param_resolution)
    return param_resolution, th_grid, mxrs_grid, tr_grid


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
    param_resolution, th_grid, mxrs_grid, tr_grid = build_grids_from_env()
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
        param_resolution=int(param_resolution),
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
        # Candidate/tie RNGs are deliberately paired across the two learners.
        # LinTS uses an independent posterior RNG internally.
        "Shared LinTS v2": int(seed + 13003),
    }


def default_branch_label(*, method: str, tune_dim: int, tune7_variant: str) -> str:
    if str(method).strip().lower() == "default":
        return "default (fixed)"
    family = {
        "linucbv4": "Shared LinUCB v4",
        "lints_v2": "Shared LinTS v2",
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
        "lints_v2": "Shared LinTS v2",
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
        tune7_variant=tune7_variant,
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
    return _generate_difconv_instances(
        count=int(T),
        seed=int(seed),
        grid_choices=grid_choices,
        c_min=float(c_min),
        c_max=float(c_max),
        advection=difconv_a,
    )


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
                "arm_index": int(getattr(step, "arm_index", -1)),
                "pred_mean": float(getattr(step, "pred_mean", np.nan)),
                "pred_uncert": float(getattr(step, "pred_uncert", np.nan)),
            }
        return dict(params), info

    def update(self, loss, **_):
        self.model.update(float(loss))

    def cancel_pending(self) -> None:
        self.model.cancel_pending()

    def recommend(self, context, *, candidate_arms, alpha: float = 0.0, **_):
        return self.model.recommend(
            np.asarray(context, dtype=float),
            candidate_arms=candidate_arms,
            alpha=float(alpha),
        )


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

    for branch in branches:
        if normalize_method_name(branch.family) == normalize_method_name("Shared LinUCB v4"):
            validate_expected_setup_action_count(branch)

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
    tune7_variant: str,
    cfg: TestFinalBanditConfig,
    candidate_schedule: AOTCandidateSchedule | None = None,
    action_feature_cache: FactorizedActionFeatureCache | None = None,
    lin_ts_relative_sampling_scale: float = 0.15,
    lin_ts_loss_scale_prior: float = 0.1,
    candidate_sampling: str = "uniform512",
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
            "initial_guess_rounds": int(os.environ.get("SETUP_INITIAL_GUESS_ROUNDS", "1")),
        }
        strategy = resolve_tune7_candidate_strategy(
            tune7_variant=tune7_variant,
            configured_strategy=cfg.tune7_candidate_strategy,
        )
        sampling = str(candidate_sampling).strip().lower().replace("-", "")
        if sampling not in AOTCandidateSchedule.SAMPLING_METHODS:
            raise ValueError(
                "candidate_sampling must be uniform512 or structured512"
            )
        if sampling == "structured512":
            if candidate_schedule is None:
                raise ValueError("structured512 requires an AOT candidate schedule")
            tune7_kwargs.update(
                {
                    "alpha_decay": True,
                    "candidate_pool_size": int(cfg.candidate_pool_size),
                    "candidate_strategy": "structured",
                    "elite_rank_metric": "mean_loss",
                    "local_neighbor_radius": 2,
                }
            )
        elif strategy == "adaptive_local":
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
        learner_kwargs: Dict[str, Any] = {}
        if method_key == "lints_v2":
            learner_kwargs.update(
                {
                    "relative_sampling_scale": float(
                        lin_ts_relative_sampling_scale
                    ),
                    "loss_scale_prior": float(lin_ts_loss_scale_prior),
                }
            )
        model = build_online_setup_learner(
            make_setup_learner_spec(
                kind=method_key,
                shared=SharedSetupLearnerSpec(
                    actions=actions,
                    context_dim=int(context_dim),
                    seed=int(seed),
                    alpha=float(cfg.alpha),
                    l2_reg=float(cfg.l2),
                    candidate_schedule=candidate_schedule,
                    action_feature_cache=action_feature_cache,
                    **tune7_kwargs,
                ),
                algorithm_parameters=learner_kwargs,
            )
        )
        return GenericBanditPolicy(model)

    raise ValueError(
        "The retained Exp44 active path only supports generic tune7 setup "
        f"learners, got method={method!r}"
    )


def _safe_one_at_a_time_actions(parameter_spec: Any) -> list[Dict[str, Any]]:
    values = {
        "strong_threshold": (0.15, 0.35),
        "max_row_sum": (0.8, 0.95),
        "trunc_factor": (0.05, 0.1),
        "P_max_elmts": (2, 6, 8),
        "agg_num_levels": (1,),
        "coarsen_type": (6, 8),
        "interp_type": (8,),
    }
    actions = [dict(DEFAULT_SETUP_PARAMS)]
    seen = {tuple(sorted(DEFAULT_SETUP_PARAMS.items()))}
    params_by_name = {param.name: param for param in parameter_spec.parameters}
    for name, candidates in values.items():
        parameter = params_by_name[name]
        for candidate in candidates:
            value = candidate
            if value not in parameter.values:
                if parameter.kind == "categorical":
                    raise ValueError(
                        f"{name}={value!r} is not in {tuple(parameter.values)!r}"
                    )
                value = min(
                    parameter.values,
                    key=lambda item: abs(float(item) - float(value)),
                )
            params = dict(DEFAULT_SETUP_PARAMS)
            params[name] = value
            key = tuple(sorted(params.items()))
            if key not in seen:
                actions.append(params)
                seen.add(key)
    return actions


def build_online_linucb_branch(
    *,
    seed: int,
    learner_kind: str = "linucb",
    tune_dim: int = 7,
    tune7_variant: str = "categorical",
    action_space_mode: str | None = None,
    solver_tol: float | None = None,
    solver_max_iter: int | None = None,
    parameter_resolution: int | None = None,
    configuration_space: SetupConfigurationSpace | None = None,
    candidate_schedule_dir: Path | None = None,
    candidate_schedule_rounds: int | None = None,
    candidate_schedule_seed: int | None = None,
    candidate_schedule_chunk_rounds: int = 256,
    lin_ts_relative_sampling_scale: float = 0.15,
    lin_ts_loss_scale_prior: float = 0.1,
    candidate_sampling: str = "uniform512",
) -> tuple[BranchRun, TestFinalBanditConfig]:
    """Build a canonical online setup-bandit branch without a Gym dependency."""

    learner_token = str(learner_kind).strip().lower()
    learner_method = {
        "linucb": "linucbv4",
        "lints": "lints_v2",
    }.get(learner_token)
    if learner_method is None:
        raise ValueError(f"Unsupported setup learner kind: {learner_kind!r}")
    family = {
        "linucb": "Shared LinUCB v4",
        "lints": "Shared LinTS v2",
    }[learner_token]

    cfg = default_test_final_bandit_config_from_env()
    mode = str(
        os.environ.get("SETUP_ACTION_SPACE", "safe_one_at_a_time")
        if action_space_mode is None
        else action_space_mode
    ).strip().lower()
    resolved_tol = float(
        os.environ.get("SOLVE_TOL", "1e-6")
        if solver_tol is None
        else solver_tol
    )
    resolved_max_iter = int(
        os.environ.get("SOLVE_MAX_CYCLES", "50")
        if solver_max_iter is None
        else solver_max_iter
    )
    if mode == "full_cartesian" and configuration_space is not None:
        if int(tune_dim) != 7 or str(tune7_variant).strip().lower() != "categorical":
            raise ValueError(
                "Named setup configuration spaces require tune_dim=7 and "
                "tune7_variant='categorical'"
            )
        setup_space = build_setup_param_space(
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            parameter_resolution=parameter_resolution,
            configuration_space=configuration_space,
            materialize=candidate_schedule_dir is None,
        )
        parameter_space = {
            "actions": setup_space.actions,
            "context_dim": int(DIFCONV_CONTEXT_DIM),
        }
        family_seed = int(family_seed_map(seed=int(seed))[family])
        candidate_schedule = None
        action_feature_cache = None
        if candidate_schedule_dir is not None:
            if candidate_schedule_rounds is None or candidate_schedule_rounds <= 0:
                raise ValueError(
                    "candidate_schedule_rounds must be positive for AOT mode"
                )
            encoder = GenericActionFeatureEncoder(setup_space.parameter_spec)
            action_feature_cache = FactorizedActionFeatureCache(
                setup_space.actions, encoder
            )
            candidate_schedule = AOTCandidateSchedule(
                directory=Path(candidate_schedule_dir),
                catalog=setup_space.actions,
                rounds=int(candidate_schedule_rounds),
                pool_size=int(cfg.candidate_pool_size),
                seed=int(
                    family_seed
                    if candidate_schedule_seed is None
                    else candidate_schedule_seed
                ),
                chunk_rounds=int(candidate_schedule_chunk_rounds),
                sampling_method=str(candidate_sampling),
                factorized_cache=action_feature_cache,
            )
        policy = build_test_final_bandit_policy(
            method=learner_method,
            tune_dim=int(tune_dim),
            actions=setup_space.actions,
            context_dim=int(DIFCONV_CONTEXT_DIM),
            seed=family_seed,
            default_params=dict(DEFAULT_SETUP_PARAMS),
            default_arm_index=int(setup_space.default_arm_index),
            parameter_spec=setup_space.parameter_spec,
            tune7_variant=tune7_variant,
            cfg=cfg,
            candidate_schedule=candidate_schedule,
            action_feature_cache=action_feature_cache,
            lin_ts_relative_sampling_scale=float(
                lin_ts_relative_sampling_scale
            ),
            lin_ts_loss_scale_prior=float(lin_ts_loss_scale_prior),
            candidate_sampling=str(candidate_sampling),
        )
        branch = BranchRun(
            label=(
                f"{family} | {configuration_space.name} | "
                f"{str(candidate_sampling)}"
            ),
            family=family,
            tune_set=configuration_space.name,
            seed=family_seed,
            policy=policy,
            parameter_space=parameter_space,
            solver_tol=resolved_tol,
            solver_max_iter=resolved_max_iter,
        )
    elif mode == "full_cartesian":
        bundle = build_action_space_bundle(
            final_tune_dims=[int(tune_dim)],
            tune7_variant=tune7_variant,
        )
        branch = build_single_branch(
            method=learner_method,
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            seed=int(seed),
            solver_tol=resolved_tol,
            solver_max_iter=resolved_max_iter,
            bandit_cfg=cfg,
            bundle=bundle,
        )
    elif mode == "safe_one_at_a_time":
        parameter_spec, _fixed_params = build_setup_parameter_spec(
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
        )
        actions = _safe_one_at_a_time_actions(parameter_spec)
        parameter_space = {
            "actions": actions,
            "context_dim": int(DIFCONV_CONTEXT_DIM),
        }
        family_seed = int(family_seed_map(seed=int(seed))[family])
        policy = build_test_final_bandit_policy(
            method=learner_method,
            tune_dim=int(tune_dim),
            actions=actions,
            context_dim=int(DIFCONV_CONTEXT_DIM),
            seed=family_seed,
            default_params=dict(DEFAULT_SETUP_PARAMS),
            default_arm_index=0,
            parameter_spec=parameter_spec,
            tune7_variant=tune7_variant,
            cfg=cfg,
            lin_ts_relative_sampling_scale=float(
                lin_ts_relative_sampling_scale
            ),
            lin_ts_loss_scale_prior=float(lin_ts_loss_scale_prior),
        )
        branch = BranchRun(
            label=default_branch_label(
                method=learner_method,
                tune_dim=int(tune_dim),
                tune7_variant=tune7_variant,
            ),
            family=family,
            tune_set="tune7",
            seed=family_seed,
            policy=policy,
            parameter_space=parameter_space,
            solver_tol=resolved_tol,
            solver_max_iter=resolved_max_iter,
        )
    else:
        raise ValueError(
            "SETUP_ACTION_SPACE must be safe_one_at_a_time or full_cartesian"
        )
    if configuration_space is None:
        validate_expected_setup_action_count(branch)
    return branch, cfg


def run_bandit_step_test_final(
    *,
    policy: Any,
    parameter_space: Dict[str, Any],
    context: np.ndarray,
    solver_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
    fallback_solver_fn: Callable[[Dict[str, Any]], Dict[str, Any]] | None,
    prev_update_est: float,
    primary_is_default: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, float], int, float]:
    result = run_same_context_setup_reselection(
        policy=policy,
        parameter_space=parameter_space,
        context=np.asarray(context, dtype=float),
        solver_fn=solver_fn,
        fallback_solver_fn=fallback_solver_fn,
        default_params=DEFAULT_SETUP_PARAMS,
        prev_update_est=float(prev_update_est),
        primary_is_default=bool(primary_is_default),
        max_learned_attempts=3,
    )
    return (
        dict(result.params),
        dict(result.outcome),
        dict(result.timing),
        int(result.fallback_used),
        float(result.update_runtime_sec),
    )


def _actual_failure_result(
    exc: Exception,
    *,
    started_at: float,
    setup_runtime: float = 0.0,
    solve_runtime: float = 0.0,
    controller_runtime: float = 0.0,
    iterations: int = 0,
    residual_norm: float = float("nan"),
) -> Dict[str, Any]:
    setup_sec = max(0.0, float(setup_runtime))
    solve_sec = max(0.0, float(solve_runtime))
    controller_sec = max(0.0, float(controller_runtime))
    if isinstance(exc, AMGNativeError):
        setup_sec += float(exc.setup_runtime_sec)
        solve_sec += float(exc.solve_runtime_sec)
        stage = "setup" if exc.operation in {"create", "setup"} else "solve"
    else:
        stage = "setup" if setup_sec <= 0.0 else "solve"
        elapsed = max(0.0, float(time.perf_counter() - started_at))
        unaccounted = max(0.0, elapsed - setup_sec - solve_sec - controller_sec)
        if stage == "setup":
            setup_sec += unaccounted
        else:
            controller_sec += unaccounted
    reason = f"exception:{type(exc).__name__}:{exc}"
    return {
        "runtime": float(setup_sec + solve_sec),
        "setup_runtime": setup_sec,
        "solve_runtime": solve_sec,
        "infer_runtime": controller_sec,
        "failed": True,
        "failure_reason": reason,
        "failure_stage": stage,
        "residual_norm": float(residual_norm),
        "iterations": int(iterations),
        "structural_fail": stage == "setup",
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
    started_at = time.perf_counter()
    prep = None
    solve_runtime = 0.0
    residual_norm = float("nan")
    iterations = 0
    cycle_residuals: List[float] = []
    cycle_times: List[float] = []
    native_status = SolveStatus.CONTINUE
    try:
        params = augment_setup_params(params)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            residual_norm = float(env.r0)
            for cycle in range(int(solve_max_cycles)):
                residual_norm, dt = env.step_rl(
                    relax_weight=float(w),
                    sweeps_down=int(sweeps_down),
                    sweeps_up=int(sweeps_up),
                    tol=float(solve_tol),
                    max_cycles=int(solve_max_cycles),
                )
                solve_runtime += float(dt)
                iterations = cycle + 1
                cycle_residuals.append(float(residual_norm))
                cycle_times.append(float(dt))
                native_status = env.last_step.status
                if native_status is not SolveStatus.CONTINUE:
                    break
        failure_reason = (
            "" if native_status is SolveStatus.CONVERGED
            else "max_cycles_reached_without_convergence"
        )
        return {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": 0.0,
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "attempt_status": "success" if not failure_reason else "nonconvergence",
            "native_status": native_status.name.lower(),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "final_w": float(w),
            "final_sweeps_down": int(sweeps_down),
            "final_sweeps_up": int(sweeps_up),
            "cycle_actions": [float(w)] * int(iterations),
            "cycle_residuals": cycle_residuals,
            "cycle_times": cycle_times,
        }
    except Exception as exc:
        result = _actual_failure_result(
            exc,
            started_at=started_at,
            setup_runtime=0.0 if prep is None else prep.setup_runtime_sec,
            solve_runtime=solve_runtime,
            iterations=iterations,
            residual_norm=residual_norm,
        )
        result.update({
            "final_w": float(w),
            "final_sweeps_down": int(sweeps_down),
            "final_sweeps_up": int(sweeps_up),
            "cycle_actions": [],
            "cycle_residuals": [],
            "cycle_times": [],
        })
        return result


def solve_schedule_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    schedule: Sequence[Tuple[int, float, int, int]],
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    prep = None
    solve_runtime = 0.0
    residual_norm = float("nan")
    iterations = 0
    native_status = SolveStatus.CONTINUE
    try:
        params = augment_setup_params(params)
        schedule_sorted = sorted((int(end), float(w), int(sd), int(su)) for end, w, sd, su in schedule)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            residual_norm = float(env.r0)
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
                    tol=float(solve_tol),
                    max_cycles=int(solve_max_cycles),
                )
                solve_runtime += float(dt)
                iterations = cycle + 1
                last_w = float(w)
                last_sd = int(sd)
                last_su = int(su)
                native_status = env.last_step.status
                if native_status is not SolveStatus.CONTINUE:
                    break
        failure_reason = (
            "" if native_status is SolveStatus.CONVERGED
            else "max_cycles_reached_without_convergence"
        )
        return {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(solve_runtime),
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "attempt_status": "success" if not failure_reason else "nonconvergence",
            "native_status": native_status.name.lower(),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "final_w": float(last_w),
            "final_sweeps_down": int(last_sd),
            "final_sweeps_up": int(last_su),
        }
    except Exception as exc:
        result = _actual_failure_result(
            exc,
            started_at=started_at,
            setup_runtime=0.0 if prep is None else prep.setup_runtime_sec,
            solve_runtime=solve_runtime,
            iterations=iterations,
            residual_norm=residual_norm,
        )
        result.update({
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        })
        return result


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
    started_at = time.perf_counter()
    prep = None
    rl_out: Dict[str, Any] = {}
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
            "cycle_actions": list(rl_out.get("cycle_actions", [])),
            "cycle_residuals": list(rl_out.get("cycle_residuals", [])),
            "cycle_times": list(rl_out.get("cycle_times", [])),
        }
    except Exception as exc:
        result = _actual_failure_result(
            exc,
            started_at=started_at,
            setup_runtime=0.0 if prep is None else prep.setup_runtime_sec,
            solve_runtime=float(rl_out.get("solve_runtime", 0.0)),
            controller_runtime=float(rl_out.get("infer_runtime", 0.0)),
            iterations=int(rl_out.get("iterations", 0)),
            residual_norm=float(rl_out.get("residual_norm", float("nan"))),
        )
        result.update({
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
            "action_counts": {},
            "cycle_actions": [],
            "cycle_residuals": [],
            "cycle_times": [],
        })
        return result


def classify_no_rl_failure(*, residual_norm: float, iterations: int, solver_tol: float, solver_max_iter: int) -> str:
    reasons: List[str] = []
    if not np.isfinite(float(residual_norm)):
        reasons.append("non_finite_residual_norm")
    elif float(residual_norm) > float(solver_tol):
        reasons.append("residual_above_solver_tol")
    if float(residual_norm) > float(solver_tol) and int(iterations) >= int(solver_max_iter):
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
    started_at = time.perf_counter()
    try:
        res = solve(
            params=augment_params(dict(params)),
            tol=float(solver_tol),
            max_iter=int(solver_max_iter),
            **mkw,
        )
        residual_norm = float(res.residual_norm)
        iterations = int(res.iterations)
        failure_reason = (
            "" if res.status is SolveStatus.CONVERGED
            else "max_iter_reached_without_convergence"
        )
        return {
            "runtime": float(res.runtime_sec),
            "setup_runtime": float(res.setup_runtime_sec),
            "solve_runtime": float(res.solve_runtime_sec),
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "attempt_status": "success" if not failure_reason else "nonconvergence",
            "native_status": res.status.name.lower(),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "structural_fail": False,
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }
    except Exception as exc:
        result = _actual_failure_result(exc, started_at=started_at)
        result.update({
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        })
        return result


def solve_default_baseline_case(
    *,
    mkw: Dict[str, Any],
    solver_tol: float,
    solver_max_iter: int,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]] = augment_setup_params,
) -> Dict[str, Any]:
    """Run the default baseline once, without retrying the same attempt."""

    recovery = run_with_default_fallback(
        lambda: solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(mkw),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
            augment_params=augment_params,
        ),
        None,
        primary_is_default=True,
    )
    result = recovery.to_result()
    result.update(
        {
            "recovery_protocol_applied": True,
            "bandit_update_committed": False,
            "controller_update_committed": False,
        }
    )
    return result


def fixed_trace(
    *,
    T: int,
    grid: Tuple[int, int, int],
    seed: int,
    tune_dim: int,
    bandit_method: str,
    solve_mode: str = "no_rl",
    solve_policy: Optional[SetupAwareSolvePolicyRunner] = None,
    trace_records: Optional[List[Dict[str, Any]]] = None,
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
    trace: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for case_index, (mkw, context) in enumerate(instances):
        if str(solve_mode).strip().lower() == "no_rl":
            def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_no_rl_case(
                    params=selected_params,
                    mkw=dict(mkw),
                    solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
                    solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
                    augment_params=augment_setup_params,
                )
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
        else:
            raise ValueError(f"Unsupported solve_mode: {solve_mode}")

        def fallback_solver_fn(_params: Dict[str, Any]) -> Dict[str, Any]:
            return solve_no_rl_case(
                params=dict(DEFAULT_SETUP_PARAMS),
                mkw=dict(mkw),
                solver_tol=float(
                    os.environ.get("SOLVER_TOL", "1e-6")
                    if solve_policy is None
                    else solve_policy.cfg.solve_tol
                ),
                solver_max_iter=int(
                    os.environ.get("SOLVER_MAX_ITER", "50")
                    if solve_policy is None
                    else solve_policy.cfg.solve_max_cycles
                ),
                augment_params=augment_setup_params,
            )

        params, out, timing, fallback_used, prev_update_est = run_bandit_step_test_final(
            policy=branch.policy,
            parameter_space=branch.parameter_space,
            context=np.asarray(context, dtype=float),
            solver_fn=solver_fn,
            fallback_solver_fn=fallback_solver_fn,
            prev_update_est=float(prev_update_est),
        )
        trace.append((dict(mkw), dict(params)))
        if trace_records is not None:
            trace_records.append(
                {
                    "instance_index": int(case_index),
                    "mkw": dict(mkw),
                    "context": np.asarray(context, dtype=float).tolist(),
                    "params": dict(params),
                    "fallback_used": int(fallback_used),
                    "bandit_timing": dict(timing),
                    "feedback_outcome": dict(out),
                }
            )
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
    runtimes = [float(v["runtime"]) for v in vals]
    setup_runtimes = [float(v["setup_runtime"]) for v in vals]
    solve_runtimes = [float(v["solve_runtime"]) for v in vals]
    infer_runtimes = [float(v.get("infer_runtime", 0.0)) for v in vals]
    per_case_mean_w = [
        float(np.mean(actions)) if actions else float("nan")
        for actions in (list(v.get("cycle_actions", ())) for v in vals)
    ]
    max_cycles = max((len(v.get("cycle_actions", ())) for v in vals), default=0)
    mean_w_by_cycle = []
    for cycle in range(max_cycles):
        cycle_values = [
            float(v["cycle_actions"][cycle])
            for v in vals
            if cycle < len(v.get("cycle_actions", ()))
        ]
        mean_w_by_cycle.append(float(np.mean(cycle_values)))
    return {
        "cases": int(len(vals)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
        "mean_setup_runtime": float(np.mean(setup_runtimes)),
        "total_setup_runtime": float(np.sum(setup_runtimes)),
        "mean_solve_runtime": float(np.mean(solve_runtimes)),
        "total_solve_runtime": float(np.sum(solve_runtimes)),
        "mean_infer_runtime": float(np.mean(infer_runtimes)),
        "total_infer_runtime": float(np.sum(infer_runtimes)),
        "mean_runtime_with_controller": float(
            np.mean(np.asarray(runtimes) + np.asarray(infer_runtimes))
        ),
        "total_runtime_with_controller": float(
            np.sum(np.asarray(runtimes) + np.asarray(infer_runtimes))
        ),
        "failed_count": int(fails),
        "mean_iterations": float(np.mean([v["iterations"] for v in vals])),
        "mean_final_w": float(np.mean([v["final_w"] for v in vals if np.isfinite(v["final_w"])])),
        "action_hist": {int(k): int(action_hist[k]) for k in sorted(action_hist)},
        "per_case_mean_w": per_case_mean_w,
        "mean_w_by_cycle": mean_w_by_cycle,
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

            def fallback_solver_fn(_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=float(
                        branch.solver_tol
                        if solve_mode == "no_rl"
                        else solve_policy.cfg.solve_tol
                    ),
                    solver_max_iter=int(
                        branch.solver_max_iter
                        if solve_mode == "no_rl"
                        else solve_policy.cfg.solve_max_cycles
                    ),
                    augment_params=augment_params,
                )

            params, out, timing, _fallback_used, last_update_sec = run_bandit_step_test_final(
                policy=branch.policy,
                parameter_space=branch.parameter_space,
                context=np.asarray(context, dtype=float),
                solver_fn=solver_fn,
                fallback_solver_fn=fallback_solver_fn,
                prev_update_est=float(metrics["prev_update_est"][label]),
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
