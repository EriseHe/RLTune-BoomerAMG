from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from setup.learners.common import (
    CompactActionCatalog,
    ParameterSpaceSpec,
    ParameterSpec,
    SharedSetupLearnerSpec,
)
from setup.registry import (
    build_online_setup_learner,
    make_setup_learner_spec,
)
from problems.amg import DIFCONV_CONTEXT_DIM
from setup.utils.setup_amg import build_actions_from_spec

DEFAULT_SETUP_PARAMS = {
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
DEFAULT_TUNE7_INTERP_TYPES = (6, 8)
DEFAULT_TUNE7_AGG_TR_VALUES = (0.0, 0.1)
DEFAULT_TUNE7_AGG_PMX_VALUES = (0, 4)

ALL_SETUP_PARAM_KEYS = (
    "strong_threshold",
    "max_row_sum",
    "trunc_factor",
    "P_max_elmts",
    "agg_num_levels",
    "coarsen_type",
    "interp_type",
    "agg_interp_type",
    "agg_tr",
    "agg_Pmx",
)
SETUP_OBS_KEYS = ALL_SETUP_PARAM_KEYS

_MANUAL_SETUP_BOUNDS = {
    "strong_threshold": ("continuous", float(DEFAULT_SETUP_PARAMS["strong_threshold"]), 0.25, None),
    "max_row_sum": ("continuous", float(DEFAULT_SETUP_PARAMS["max_row_sum"]), 0.10, None),
    "trunc_factor": ("continuous", float(DEFAULT_SETUP_PARAMS["trunc_factor"]), 0.20, None),
    "P_max_elmts": ("integer", float(DEFAULT_SETUP_PARAMS["P_max_elmts"]), 4.0, tuple(int(v) for v in DEFAULT_P_MAX_ELMTS_VALUES)),
    "agg_num_levels": ("integer", float(DEFAULT_SETUP_PARAMS["agg_num_levels"]), 1.0, tuple(int(v) for v in DEFAULT_AGG_NUM_LEVELS_VALUES)),
    "coarsen_type": ("categorical", float(DEFAULT_SETUP_PARAMS["coarsen_type"]), 1.0, tuple(int(v) for v in DEFAULT_COARSEN_TYPE_VALUES)),
    "interp_type": ("categorical", float(DEFAULT_SETUP_PARAMS["interp_type"]), 1.0, tuple(int(v) for v in DEFAULT_TUNE7_INTERP_TYPES)),
    "agg_interp_type": ("categorical", float(DEFAULT_SETUP_PARAMS["agg_interp_type"]), 1.0, (4, 6)),
    "agg_tr": ("continuous", float(DEFAULT_SETUP_PARAMS["agg_tr"]), 0.10, None),
    "agg_Pmx": ("integer", float(DEFAULT_SETUP_PARAMS["agg_Pmx"]), 4.0, tuple(int(v) for v in DEFAULT_TUNE7_AGG_PMX_VALUES)),
}


@dataclass(frozen=True)
class SetupParamSpace:
    actions: Sequence[Dict[str, Any]]
    parameter_spec: ParameterSpaceSpec
    default_arm_index: int


@dataclass(frozen=True)
class SetupConfigurationSpace:
    """Named categorical choices for one setup-bandit branch."""

    name: str
    coarsen_types: Tuple[int, ...]
    interp_types: Tuple[int, ...]
    agg_interp_types: Tuple[int, ...] = (
        int(DEFAULT_SETUP_PARAMS["agg_interp_type"]),
    )

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name or any(character in name for character in "/\\:@"):
            raise ValueError(
                "Setup configuration-space names must be non-empty and cannot "
                "contain path separators, ':' or '@'"
            )
        object.__setattr__(self, "name", name)
        for field_name in (
            "coarsen_types",
            "interp_types",
            "agg_interp_types",
        ):
            values = tuple(int(value) for value in getattr(self, field_name))
            if not values:
                raise ValueError(f"{field_name} cannot be empty")
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} cannot contain duplicate values")
            object.__setattr__(self, field_name, values)

    @classmethod
    def from_mapping(
        cls,
        name: str,
        raw: Mapping[str, Any],
    ) -> "SetupConfigurationSpace":
        allowed = {"coarsen_types", "interp_types", "agg_interp_types"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(
                f"Unknown keys in setup configuration space {name!r}: "
                f"{sorted(unknown)}"
            )
        missing = {"coarsen_types", "interp_types"} - set(raw)
        if missing:
            raise ValueError(
                f"Setup configuration space {name!r} is missing "
                f"{sorted(missing)}"
            )
        return cls(
            name=str(name),
            coarsen_types=tuple(int(value) for value in raw["coarsen_types"]),
            interp_types=tuple(int(value) for value in raw["interp_types"]),
            agg_interp_types=tuple(
                int(value)
                for value in raw.get(
                    "agg_interp_types",
                    (DEFAULT_SETUP_PARAMS["agg_interp_type"],),
                )
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "coarsen_types": [int(value) for value in self.coarsen_types],
            "interp_types": [int(value) for value in self.interp_types],
            "agg_interp_types": [
                int(value) for value in self.agg_interp_types
            ],
        }


class SetupObsEncoder:
    def __init__(
        self,
        parameter_spec: ParameterSpaceSpec,
        defaults: Dict[str, Any],
        observed_keys: Sequence[str],
    ) -> None:
        self.parameter_spec = parameter_spec
        self.defaults = dict(defaults)
        self.spec_by_name = {param.name: param for param in parameter_spec.parameters}
        self.observed_keys = tuple(str(key) for key in observed_keys)

    def encode(self, params: Dict[str, Any]) -> np.ndarray:
        encoded = []
        for key in self.observed_keys:
            encoded.append(self._encode_one(key, params.get(key, self.defaults.get(key, 0.0))))
        arr = np.asarray(encoded, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

    def _encode_one(self, key: str, value: Any) -> float:
        param = self.spec_by_name.get(key)
        if param is not None:
            if param.kind in {"continuous", "integer"}:
                center = float(param.center if param.center is not None else param.default)
                scale = float(param.scale if param.scale is not None else 1.0)
                return float((float(value) - center) / max(scale, 1e-12))
            levels = tuple(param.values)
            return self._encode_categorical(value=value, default=param.default, levels=levels)

        kind, center, scale, levels = _MANUAL_SETUP_BOUNDS[key]
        if kind in {"continuous", "integer"}:
            return float((float(value) - float(center)) / max(float(scale), 1e-12))
        return self._encode_categorical(value=value, default=center, levels=levels or ())

    @staticmethod
    def _encode_categorical(*, value: Any, default: Any, levels: Sequence[Any]) -> float:
        levels = tuple(levels)
        if not levels:
            return 0.0
        if value not in levels:
            value = default
        try:
            idx = levels.index(value)
        except ValueError:
            idx = 0
        try:
            default_idx = levels.index(default)
        except ValueError:
            default_idx = 0
        denom = max(1, len(levels) - 1)
        return float(idx - default_idx) / float(denom)


class FixedPolicy:
    def __init__(self, params: Dict[str, Any]) -> None:
        self._params = dict(params)

    def select(self, context: np.ndarray, **_: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return dict(self._params), {}

    def update(self, loss: float, **_: Any) -> None:
        return None


class SetupBanditPolicy:
    def __init__(
        self,
        *,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        tune_dim: int,
        parameter_spec: ParameterSpaceSpec,
        seed: int,
        alpha: float,
        l2: float,
        method: str,
        default_arm_index: int,
        candidate_pool_size: int,
        elite_cache_size: int,
    ) -> None:
        self.tune_dim = int(tune_dim)
        self.parameter_spec = parameter_spec
        self.method = str(method).strip().lower()
        if self.tune_dim == 7:
            self.model = build_online_setup_learner(
                make_setup_learner_spec(
                    kind="linucb",
                    shared=SharedSetupLearnerSpec(
                        actions=actions,
                        context_dim=int(context_dim),
                        alpha=float(alpha),
                        l2_reg=float(l2),
                        seed=int(seed),
                        parameter_spec=parameter_spec,
                        context_interaction_indices=(1, 2, 3, 4),
                        always_include_arms=(int(default_arm_index),),
                        elite_cache_size=int(elite_cache_size),
                        initial_guess=[
                            DEFAULT_SETUP_PARAMS[param.name]
                            for param in parameter_spec.parameters
                        ],
                        initial_guess_rounds=1,
                        alpha_decay=True,
                        candidate_pool_size=int(candidate_pool_size),
                    ),
                )
            )
        else:
            raise ValueError(
                "The retained Exp44 active path only supports tune_dim=7 with SharedLinUCB_AMG_v4"
            )

    def select(self, context: np.ndarray, **_: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        params = self.model.predict(np.asarray(context, dtype=float))
        return dict(params), {}

    def update(self, loss: float, **_: Any) -> None:
        self.model.update(float(loss))


class RandomSetupParamProvider:
    def __init__(self, *, action_space: SetupParamSpace, seed: int) -> None:
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def select(self, *, context: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        idx = int(self.rng.integers(0, len(self.action_space.actions)))
        return dict(self.action_space.actions[idx]), {"arm_index": idx}

    def update(self, *, loss: float, context: np.ndarray, params: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        return None


class BanditSetupParamProvider:
    def __init__(self, *, policy: Any) -> None:
        self.policy = policy

    def select(self, *, context: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self.policy.select(context=context)

    def update(self, *, loss: float, context: np.ndarray, params: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        self.policy.update(loss=float(loss), context=context, params=params, outcome=outcome)


def _default_bandit_seed_like_test10(*, seed: int, tune_dim: int, method: str) -> int:
    method_key = str(method).strip().lower()
    tune_key = int(tune_dim)
    tune_offset = {3: 10_000, 5: 20_000, 7: 30_000}.get(tune_key, 20_000)
    method_offset = {
        "linucbv2": 11_003,
        "linucbv3": 12_003,
        "linucbv4": 13_003,
    }.get(method_key, 12_003)
    return int(seed + tune_offset + method_offset)


def _parse_int_list_env(name: str, default_values: Sequence[int]) -> Tuple[int, ...]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return tuple(vals) if vals else tuple(int(v) for v in default_values)


def _parse_float_list_env(name: str, default_values: Sequence[float]) -> Tuple[float, ...]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    return tuple(vals) if vals else tuple(float(v) for v in default_values)


def _same_action(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = set(a) | set(b)
    for key in keys:
        av = a.get(key)
        bv = b.get(key)
        if isinstance(av, float) or isinstance(bv, float):
            if not np.isclose(float(av), float(bv), rtol=0.0, atol=1e-12):
                return False
        else:
            if av != bv:
                return False
    return True


def _ensure_default_arm(actions: Sequence[Dict[str, Any]]) -> Tuple[Tuple[Dict[str, Any], ...], int]:
    for idx, action in enumerate(actions):
        if _same_action(action, DEFAULT_SETUP_PARAMS):
            return tuple(dict(a) for a in actions), int(idx)
    out = tuple([*(dict(a) for a in actions), dict(DEFAULT_SETUP_PARAMS)])
    return out, int(len(out) - 1)


def build_setup_parameter_spec(
    *,
    tune_dim: int,
    tune7_variant: str = "categorical",
    parameter_resolution: int | None = None,
    coarsen_type_values: Sequence[int] | None = None,
    interp_type_values: Sequence[int] | None = None,
    agg_interp_type_values: Sequence[int] | None = None,
) -> Tuple[ParameterSpaceSpec, Dict[str, Any]]:
    param_resolution = int(
        os.environ.get("SETUP_PARAM_RESOLUTION", "10")
        if parameter_resolution is None
        else parameter_resolution
    )
    if param_resolution <= 0:
        raise ValueError("parameter_resolution must be positive")
    grid_max = float(os.environ.get("SETUP_GRID_MAX", os.environ.get("GRID_MAX", "0.95")))
    th_grid = np.linspace(0.0, grid_max, param_resolution)
    mxrs_grid = np.linspace(0.0, grid_max, param_resolution)
    if len(mxrs_grid) > 0:
        mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, param_resolution)
    p_max_values = _parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = _parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)
    coarsen_values = (
        _parse_int_list_env("COARSEN_TYPE_VALUES", DEFAULT_COARSEN_TYPE_VALUES)
        if coarsen_type_values is None
        else [int(value) for value in coarsen_type_values]
    )
    interp_values = (
        _parse_int_list_env("TUNE7_INTERP_TYPES", DEFAULT_TUNE7_INTERP_TYPES)
        if interp_type_values is None
        else [int(value) for value in interp_type_values]
    )
    agg_interp_values = (
        None
        if agg_interp_type_values is None
        else tuple(int(value) for value in agg_interp_type_values)
    )
    for name, values in (
        ("coarsen_type_values", coarsen_values),
        ("interp_type_values", interp_values),
        ("agg_interp_type_values", agg_interp_values),
    ):
        if values is not None and not values:
            raise ValueError(f"{name} cannot be empty")
    agg_tr_values = _parse_float_list_env("TUNE7_AGG_TR_VALUES", DEFAULT_TUNE7_AGG_TR_VALUES)
    agg_pmx_values = _parse_int_list_env("TUNE7_AGG_PMX_VALUES", DEFAULT_TUNE7_AGG_PMX_VALUES)

    base_specs = [
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
    ]

    fixed_params: Dict[str, Any] = {
        "coarsen_type": int(DEFAULT_SETUP_PARAMS["coarsen_type"]),
        "interp_type": int(DEFAULT_SETUP_PARAMS["interp_type"]),
        "agg_interp_type": int(DEFAULT_SETUP_PARAMS["agg_interp_type"]),
        "agg_tr": float(DEFAULT_SETUP_PARAMS["agg_tr"]),
        "agg_Pmx": int(DEFAULT_SETUP_PARAMS["agg_Pmx"]),
        "P_max_elmts": int(DEFAULT_SETUP_PARAMS["P_max_elmts"]),
        "agg_num_levels": int(DEFAULT_SETUP_PARAMS["agg_num_levels"]),
    }

    if int(tune_dim) >= 5:
        base_specs.extend(
            [
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
            ]
        )
        fixed_params.pop("P_max_elmts", None)
        fixed_params.pop("agg_num_levels", None)

    if int(tune_dim) == 7:
        if str(tune7_variant).strip().lower() == "agg_conditional":
            active_agg_levels = tuple(int(v) for v in agg_nl_values if int(v) != int(DEFAULT_SETUP_PARAMS["agg_num_levels"]))
            base_specs.extend(
                [
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
                ]
            )
            fixed_params.pop("agg_tr", None)
            fixed_params.pop("agg_Pmx", None)
        else:
            base_specs.extend(
                [
                    ParameterSpec(
                        name="coarsen_type",
                        kind="categorical",
                        values=tuple(int(v) for v in coarsen_values),
                        default=int(DEFAULT_SETUP_PARAMS["coarsen_type"]),
                    ),
                    ParameterSpec(
                        name="interp_type",
                        kind="categorical",
                        values=tuple(int(v) for v in interp_values),
                        default=int(DEFAULT_SETUP_PARAMS["interp_type"]),
                    ),
                ]
            )
            fixed_params.pop("coarsen_type", None)
            fixed_params.pop("interp_type", None)

            if agg_interp_values is not None and agg_interp_values != (
                int(DEFAULT_SETUP_PARAMS["agg_interp_type"]),
            ):
                active_agg_levels = tuple(
                    int(value)
                    for value in agg_nl_values
                    if int(value)
                    != int(DEFAULT_SETUP_PARAMS["agg_num_levels"])
                )
                base_specs.append(
                    ParameterSpec(
                        name="agg_interp_type",
                        kind="categorical",
                        values=tuple(int(value) for value in agg_interp_values),
                        default=int(DEFAULT_SETUP_PARAMS["agg_interp_type"]),
                        active_if={"agg_num_levels": active_agg_levels},
                    )
                )
                fixed_params.pop("agg_interp_type", None)

    return ParameterSpaceSpec(tuple(base_specs)), fixed_params


def build_setup_param_space(
    *,
    tune_dim: int,
    tune7_variant: str = "categorical",
    parameter_resolution: int | None = None,
    configuration_space: SetupConfigurationSpace | None = None,
    materialize: bool = True,
) -> SetupParamSpace:
    parameter_spec, fixed_params = build_setup_parameter_spec(
        tune_dim=tune_dim,
        tune7_variant=tune7_variant,
        parameter_resolution=parameter_resolution,
        coarsen_type_values=(
            None
            if configuration_space is None
            else configuration_space.coarsen_types
        ),
        interp_type_values=(
            None
            if configuration_space is None
            else configuration_space.interp_types
        ),
        agg_interp_type_values=(
            None
            if configuration_space is None
            else configuration_space.agg_interp_types
        ),
    )
    if materialize:
        actions = build_actions_from_spec(
            parameter_spec, fixed_params=fixed_params
        )
        actions, default_arm_index = _ensure_default_arm(actions)
    else:
        actions = CompactActionCatalog(
            parameter_spec, fixed_params=fixed_params
        )
        default_arm_index = int(actions.default_arm_index)
    return SetupParamSpace(actions=actions, parameter_spec=parameter_spec, default_arm_index=default_arm_index)


def build_setup_param_provider(*, mode: str, seed: int, tune_dim: int, tune7_variant: str = "categorical"):
    action_space = build_setup_param_space(tune_dim=tune_dim, tune7_variant=tune7_variant)
    if str(mode).strip().lower() == "random":
        provider = RandomSetupParamProvider(action_space=action_space, seed=seed)
    elif str(mode).strip().lower() == "bandit":
        method = os.environ.get("SETUP_BANDIT_METHOD", "").strip().lower()
        if not method:
            method = "linucbv4" if int(tune_dim) == 7 else "linucbv3"
        bandit_seed = int(
            os.environ.get(
                "SETUP_BANDIT_SEED",
                str(_default_bandit_seed_like_test10(seed=int(seed), tune_dim=int(tune_dim), method=method)),
            )
        )
        provider = BanditSetupParamProvider(
            policy=SetupBanditPolicy(
                actions=action_space.actions,
                context_dim=int(DIFCONV_CONTEXT_DIM),
                tune_dim=int(tune_dim),
                parameter_spec=action_space.parameter_spec,
                seed=int(bandit_seed),
                alpha=float(os.environ.get("SETUP_BANDIT_ALPHA", os.environ.get("ALPHA", "1.0"))),
                l2=float(os.environ.get("SETUP_BANDIT_L2", os.environ.get("L2", "1.0"))),
                method=method,
                default_arm_index=int(action_space.default_arm_index),
                candidate_pool_size=int(
                    os.environ.get("SETUP_BANDIT_CANDIDATE_POOL_SIZE", os.environ.get("CANDIDATE_POOL_SIZE", "512"))
                ),
                elite_cache_size=int(
                    os.environ.get("SETUP_BANDIT_ELITE_CACHE_SIZE", os.environ.get("ELITE_CACHE_SIZE", "64"))
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported setup mode: {mode}")
    return provider, action_space
