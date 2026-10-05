"""Experiment action grids and immutable action-space descriptions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from setup.learners.common import ParameterSpaceSpec, ParameterSpec
from problems.amg import DIFCONV_CONTEXT_DIM
from setup.utils.setup_amg import build_actions_from_spec, build_actions_th_mxrs_tr
from setup.space import (
    DEFAULT_SETUP_PARAMS,
    DEFAULT_COARSEN_TYPE_VALUES,
    DEFAULT_P_MAX_ELMTS_VALUES,
    DEFAULT_AGG_NUM_LEVELS_VALUES,
    DEFAULT_TUNE7_AGG_TR_VALUES as DEFAULT_AGG_TR_VALUES,
    DEFAULT_TUNE7_AGG_PMX_VALUES as DEFAULT_AGG_PMX_VALUES,
    DEFAULT_TUNE7_INTERP_TYPES,
)

DEFAULT_TUNE7_AGG_INTERP_TYPES = (4, 6)


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


def parse_int_list_env(name: str, default_values: Sequence[int]) -> List[int]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [int(v) for v in default_values]


def parse_float_list_env(name: str, default_values: Sequence[float]) -> List[float]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [float(v) for v in default_values]


def same_action(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        np.isclose(
            float(a["strong_threshold"]),
            float(b["strong_threshold"]),
            rtol=0.0,
            atol=1e-12,
        )
        and np.isclose(
            float(a["max_row_sum"]), float(b["max_row_sum"]), rtol=0.0, atol=1e-12
        )
        and np.isclose(
            float(a["trunc_factor"]), float(b["trunc_factor"]), rtol=0.0, atol=1e-12
        )
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


def build_actions_tune5(
    *, th_grid, mxrs_grid, tr_grid
) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    p_max_values = parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = parse_int_list_env(
        "AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES
    )

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
) -> Tuple[
    List[Dict[str, Any]], List[int], List[int], List[int], List[int], ParameterSpaceSpec
]:
    p_max_values = parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = parse_int_list_env(
        "AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES
    )
    coarsen_type_values = parse_int_list_env(
        "COARSEN_TYPE_VALUES", DEFAULT_COARSEN_TYPE_VALUES
    )
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
    return (
        actions,
        p_max_values,
        agg_nl_values,
        coarsen_type_values,
        interp_values,
        parameter_spec,
    )


def build_actions_tune7_agg_conditional(
    *,
    th_grid,
    mxrs_grid,
    tr_grid,
) -> Tuple[
    List[Dict[str, Any]],
    List[int],
    List[int],
    List[float],
    List[int],
    ParameterSpaceSpec,
]:
    p_max_values = parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = parse_int_list_env(
        "AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES
    )
    agg_tr_values = parse_float_list_env("TUNE7_AGG_TR_VALUES", DEFAULT_AGG_TR_VALUES)
    agg_pmx_values = parse_int_list_env("TUNE7_AGG_PMX_VALUES", DEFAULT_AGG_PMX_VALUES)

    active_agg_levels = tuple(
        int(v)
        for v in agg_nl_values
        if int(v) != int(DEFAULT_SETUP_PARAMS["agg_num_levels"])
    )
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
    return (
        actions,
        p_max_values,
        agg_nl_values,
        agg_tr_values,
        agg_pmx_values,
        parameter_spec,
    )


def ensure_default_arm(
    actions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    default_arm_index = next(
        (i for i, a in enumerate(actions) if same_action(a, DEFAULT_SETUP_PARAMS)), None
    )
    if default_arm_index is None:
        return [*actions, dict(DEFAULT_SETUP_PARAMS)], int(len(actions))
    return actions, int(default_arm_index)


def build_action_space_bundle(
    *,
    final_tune_dims: Sequence[int],
    tune7_variant: str,
    context_dim: int = DIFCONV_CONTEXT_DIM,
) -> ActionSpaceBundle:
    param_resolution, th_grid, mxrs_grid, tr_grid = build_grids_from_env()
    actions_tune3 = build_actions_tune3(
        th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid
    )
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
        parameter_space_tune3={
            "actions": actions_tune3,
            "context_dim": int(context_dim),
        },
        parameter_space_tune5=(
            {"actions": actions_tune5, "context_dim": int(context_dim)}
            if actions_tune5
            else None
        ),
        parameter_space_tune7=(
            {"actions": actions_tune7, "context_dim": int(context_dim)}
            if actions_tune7
            else None
        ),
        parameter_spec_tune7=parameter_spec_tune7,
        default_arm_index_tune3=int(default_arm_index_tune3),
        default_arm_index_tune5=int(default_arm_index_tune5),
        default_arm_index_tune7=int(default_arm_index_tune7),
    )
