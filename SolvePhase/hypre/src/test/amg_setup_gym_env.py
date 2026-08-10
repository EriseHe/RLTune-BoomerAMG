from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[4]
_SETUP_ROOT = _REPO_ROOT / "SetupPhase" / "learning-setup"
if str(_SETUP_ROOT) not in sys.path:
    sys.path.insert(0, str(_SETUP_ROOT))

from amg_gym_env import build_policy_obs, decode_policy_action
from learners.SharedLinUCB_AMG_v4 import SharedLinUCB_AMG_v4
from learners._amg_action_features import ParameterSpaceSpec, ParameterSpec
from solver import PreparedAMGEnv, create_env
from utils.problem_amg import (
    DIFCONV_CONTEXT_DIM,
    build_context_difconv,
    build_matrix_kwargs_difconv,
)
from utils.setup_amg import build_actions_from_spec

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
    actions: Tuple[Dict[str, Any], ...]
    parameter_spec: ParameterSpaceSpec
    default_arm_index: int


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
            self.model = SharedLinUCB_AMG_v4(
                actions,
                context_dim=int(context_dim),
                alpha=float(alpha),
                l2_reg=float(l2),
                seed=int(seed),
                parameter_spec=parameter_spec,
                context_interaction_indices=(1, 2, 3, 4),
                always_include_arms=[int(default_arm_index)],
                elite_cache_size=int(elite_cache_size),
                initial_guess=[DEFAULT_SETUP_PARAMS[p.name] for p in parameter_spec.parameters],
                initial_guess_rounds=1,
                alpha_decay=True,
                candidate_pool_size=int(candidate_pool_size),
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


def build_setup_parameter_spec(*, tune_dim: int, tune7_variant: str = "categorical") -> Tuple[ParameterSpaceSpec, Dict[str, Any]]:
    grid_n = int(os.environ.get("SETUP_GRID_N", os.environ.get("GRID_N", "10")))
    grid_max = float(os.environ.get("SETUP_GRID_MAX", os.environ.get("GRID_MAX", "0.95")))
    th_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid = np.linspace(0.0, grid_max, grid_n)
    if len(mxrs_grid) > 0:
        mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, grid_n)
    p_max_values = _parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = _parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)
    coarsen_type_values = _parse_int_list_env("COARSEN_TYPE_VALUES", DEFAULT_COARSEN_TYPE_VALUES)
    interp_values = _parse_int_list_env("TUNE7_INTERP_TYPES", DEFAULT_TUNE7_INTERP_TYPES)
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
                        values=tuple(int(v) for v in coarsen_type_values),
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

    return ParameterSpaceSpec(tuple(base_specs)), fixed_params


def build_setup_param_space(*, tune_dim: int, tune7_variant: str = "categorical") -> SetupParamSpace:
    parameter_spec, fixed_params = build_setup_parameter_spec(tune_dim=tune_dim, tune7_variant=tune7_variant)
    actions = build_actions_from_spec(parameter_spec, fixed_params=fixed_params)
    actions, default_arm_index = _ensure_default_arm(actions)
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


class BoomerAMGSetupRelaxEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        setup_mode: str,
        tune_dim: int = 5,
        tune7_variant: str = "categorical",
        fixed_grid: Optional[Sequence[Tuple[int, int, int]] | Tuple[int, int, int]] = (60, 60, 60),
        tol: float = 1.0e-6,
        max_cycles: int = 50,
        seed: int = 0,
        w_center: float = 1.05,
        w_scale: float = 0.25,
        w_init: Optional[float] = None,
        w_only: bool = False,
        sweeps_min: int = 1,
        sweeps_max: int = 5,
        sweeps_init: Optional[int] = None,
        w_smooth_alpha: float = 0.0,
        w_change_penalty: float = 0.0,
        sweeps_change_penalty: float = 0.0,
        cycle_penalty: float = 0.0,
        sweep_penalty: float = 0.0,
        randomize_A: bool = True,
        randomize_b: bool = True,
        fixed_rhs_seed: int = 123456789,
        randomize_grid: bool = False,
        grid_min: int = 10,
        grid_max: int = 80,
        use_grid_bias: bool = False,
        grid_bias: float = 1.0,
        difconv_c: Tuple[float, float, float] = (1.0, 100.0, 100.0),
        difconv_c_range: Tuple[float, float] = (1.0, 1000.0),
        difconv_a: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        observe_setup_params: bool = True,
        bandit_update: bool = True,
        setup_failure_penalty_multiplier: float = 2.0,
        setup_failure_min_runtime_sec: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.setup_mode = str(setup_mode).strip().lower()
        self.tune_dim = int(tune_dim)
        self.tune7_variant = str(tune7_variant).strip().lower()
        self.provider, self.setup_action_space = build_setup_param_provider(
            mode=self.setup_mode,
            seed=int(seed),
            tune_dim=self.tune_dim,
            tune7_variant=self.tune7_variant,
        )
        # Only expose setup knobs that the setup policy is actually changing.
        self.setup_obs_keys = tuple(self.setup_action_space.parameter_spec.parameter_names)
        self.setup_obs_encoder = SetupObsEncoder(
            self.setup_action_space.parameter_spec,
            DEFAULT_SETUP_PARAMS,
            self.setup_obs_keys,
        )
        self.observe_setup_params = bool(observe_setup_params)
        self.setup_obs_dim = len(self.setup_obs_keys) if self.observe_setup_params else 0
        self.base_obs_dim = 10
        self.obs_dim = self.base_obs_dim + self.setup_obs_dim
        self.setup_param_overrides: Dict[str, Any] = {}
        relax_type_raw = os.environ.get("SETUP_RELAX_TYPE", "").strip()
        if relax_type_raw:
            self.setup_param_overrides["relax_type"] = int(relax_type_raw)
        num_sweeps_raw = os.environ.get("SETUP_NUM_SWEEPS", "").strip()
        if num_sweeps_raw:
            self.setup_param_overrides["num_sweeps"] = int(num_sweeps_raw)
        cycle_type_raw = os.environ.get("SETUP_CYCLE_TYPE", "").strip()
        if cycle_type_raw:
            self.setup_param_overrides["cycle_type"] = int(cycle_type_raw)
        max_levels_raw = os.environ.get("SETUP_MAX_LEVELS", "").strip()
        if max_levels_raw:
            self.setup_param_overrides["max_levels"] = int(max_levels_raw)

        self.randomize_grid = bool(randomize_grid)
        self.grid_min = int(grid_min)
        self.grid_max = int(grid_max)
        self.use_grid_bias = bool(use_grid_bias)
        self.grid_bias = float(grid_bias)
        self.grid_choices = self._normalize_grid_choices(fixed_grid)
        self.fixed_grid = self.grid_choices[0]
        self.grid_norm_div = float(self.grid_max if self.randomize_grid else max(max(g) for g in self.grid_choices))
        if self.grid_norm_div <= 0.0:
            self.grid_norm_div = 1.0

        self.tol = float(tol)
        self.max_cycles = int(max_cycles)
        self.w_center = float(w_center)
        self.w_scale = float(w_scale)
        self.w_init = None if w_init is None else float(w_init)
        self.w_only = bool(w_only)
        self.sweeps_min = int(sweeps_min)
        self.sweeps_max = int(sweeps_max)
        self.sweeps_center = 0.5 * (self.sweeps_min + self.sweeps_max)
        self.sweeps_half = 0.5 * (self.sweeps_max - self.sweeps_min)
        self.sweeps_default = int(np.clip(np.round(self.sweeps_center), self.sweeps_min, self.sweeps_max))
        self.sweeps_init = None if sweeps_init is None else int(sweeps_init)
        self.w_smooth_alpha = float(w_smooth_alpha)
        self.w_change_penalty = float(w_change_penalty)
        self.sweeps_change_penalty = float(sweeps_change_penalty)
        self.cycle_penalty = float(cycle_penalty)
        self.sweep_penalty = float(sweep_penalty)

        self.randomize_A = bool(randomize_A)
        self.randomize_b = bool(randomize_b)
        self.fixed_rhs_seed = int(fixed_rhs_seed)
        self.difconv_c = tuple(float(x) for x in difconv_c)
        self.difconv_c_range = tuple(float(x) for x in difconv_c_range)
        self.difconv_a = tuple(float(x) for x in difconv_a)
        self.c_norm_div = float(max(self.difconv_c_range[1], 1.0))

        self.bandit_update = bool(bandit_update)
        self.setup_failure_penalty_multiplier = float(setup_failure_penalty_multiplier)
        self.setup_failure_min_runtime_sec = float(setup_failure_min_runtime_sec)
        self.setup_bandit_failure_add = float(os.environ.get("SETUP_BANDIT_FAILURE_ADD", "10.0"))
        self.setup_bandit_loss_mode = os.environ.get("SETUP_BANDIT_LOSS_MODE", "test10").strip().lower()

        self.reward_mode = int(os.environ.get("REWARD_MODE", "0"))
        self.reward_alpha = float(os.environ.get("REWARD_ALPHA", "1.3"))
        self.term_bonus = float(os.environ.get("TERM_BONUS", "2.0"))
        self.trunc_penalty = float(os.environ.get("TRUNC_PENALTY", "2.0"))
        self.dt_penalty = float(os.environ.get("DT_PENALTY", "0.0"))

        if self.w_only:
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        self.episode_id = 0
        self.prepared_env: Optional[PreparedAMGEnv] = None
        self.mkw: Dict[str, Any] = {}
        self.context = np.zeros(DIFCONV_CONTEXT_DIM, dtype=float)
        self.setup_params = dict(DEFAULT_SETUP_PARAMS)
        self.setup_time = 0.0
        self.solve_runtime = 0.0
        self.r0 = 0.0
        self.r_prev = 0.0
        self.cycle = 0
        self.last_w = self.w_init if self.w_init is not None else self.w_center
        self.last_sweeps_down = self.sweeps_default
        self.last_sweeps_up = self.sweeps_default
        self.cx, self.cy, self.cz = self.difconv_c
        self.ax, self.ay, self.az = self.difconv_a
        self.nx_norm = 0.0
        self.ny_norm = 0.0
        self.nz_norm = 0.0
        self.residual_curve = []
        self._selection_info: Dict[str, Any] = {}

    def _normalize_grid_choices(self, fixed_grid):
        if fixed_grid is None:
            return [(self.grid_min, self.grid_min, self.grid_min)]
        if isinstance(fixed_grid, (list, tuple)) and fixed_grid:
            first = fixed_grid[0]
            if isinstance(first, (list, tuple)) and len(first) == 3:
                return [tuple(int(x) for x in g) for g in fixed_grid]
            if len(fixed_grid) == 3 and all(isinstance(x, (int, np.integer)) for x in fixed_grid):
                return [tuple(int(x) for x in fixed_grid)]
        raise ValueError("fixed_grid must be (nx,ny,nz) or list of such tuples")

    def _sample_grid(self) -> Tuple[int, int, int]:
        if self.randomize_grid:
            u = float(self.rng.random())
            if self.use_grid_bias and abs(self.grid_bias - 1.0) > 1e-12:
                u = u ** (1.0 / self.grid_bias)
            n = self.grid_min + int(np.floor(u * (self.grid_max - self.grid_min + 1)))
            n = min(max(n, self.grid_min), self.grid_max)
            return (n, n, n)
        idx = int(self.rng.integers(0, len(self.grid_choices)))
        return tuple(int(x) for x in self.grid_choices[idx])

    def _sample_problem(self) -> Tuple[Dict[str, Any], np.ndarray]:
        nx, ny, nz = self._sample_grid()
        eps = 1e-30
        denom = max(math.log(self.grid_norm_div + eps), eps)
        self.nx_norm = math.log(float(nx) + eps) / denom
        self.ny_norm = math.log(float(ny) + eps) / denom
        self.nz_norm = math.log(float(nz) + eps) / denom

        if self.randomize_A:
            c_min, c_max = self.difconv_c_range
            self.cx = float(self.rng.uniform(c_min, c_max))
            self.cy = float(self.rng.uniform(c_min, c_max))
            self.cz = float(self.rng.uniform(c_min, c_max))
        else:
            self.cx, self.cy, self.cz = self.difconv_c
        self.ax, self.ay, self.az = self.difconv_a
        rhs_seed = int(self.rng.integers(0, 2**63 - 1)) if self.randomize_b else int(self.fixed_rhs_seed)
        mkw = build_matrix_kwargs_difconv(
            nx=nx,
            ny=ny,
            nz=nz,
            cx=self.cx,
            cy=self.cy,
            cz=self.cz,
            ax=self.ax,
            ay=self.ay,
            az=self.az,
            rhs_seed=rhs_seed,
            rhs_type=1,
        )
        context = build_context_difconv(
            cx=self.cx,
            cy=self.cy,
            cz=self.cz,
            nx=nx,
            ny=ny,
            nz=nz,
            grid_norm_div=float(self.grid_norm_div),
            c_norm_div=float(self.c_norm_div),
        )
        return mkw, np.asarray(context, dtype=float)

    def _make_obs(self, r: float, r_prev: float) -> np.ndarray:
        base_obs = build_policy_obs(
            r=float(r),
            r_prev=float(r_prev),
            cycle=int(self.cycle),
            max_cycles=int(self.max_cycles),
            coeff_triplet=(float(self.context[1]), float(self.context[2]), float(self.context[3])),
            grid_triplet=(self.nx_norm, self.ny_norm, self.nz_norm),
            last_w=float(self.last_w),
        )
        if not self.observe_setup_params:
            return base_obs.astype(np.float32)
        setup_obs = self.setup_obs_encoder.encode(self.setup_params)
        return np.concatenate([base_obs.astype(np.float32), setup_obs], axis=0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.close()
        self.episode_id += 1
        self.mkw, self.context = self._sample_problem()
        self.setup_params, self._selection_info = self.provider.select(context=self.context)
        prepared_params = dict(self.setup_params)
        prepared_params.update(self.setup_param_overrides)
        self.prepared_env = create_env(**self.mkw)
        prep = self.prepared_env.prepare_rl(params=prepared_params)
        self.setup_time = float(prep.setup_runtime_sec)
        self.solve_runtime = 0.0
        self.r0 = float(prep.initial_residual_norm)
        self.r_prev = float(self.r0)
        self.cycle = 0
        self.last_w = self.w_init if self.w_init is not None else self.w_center
        init_sweeps = self.sweeps_default if self.sweeps_init is None else int(self.sweeps_init)
        init_sweeps = int(np.clip(init_sweeps, self.sweeps_min, self.sweeps_max))
        self.last_sweeps_down = init_sweeps
        self.last_sweeps_up = init_sweeps
        self.residual_curve = [float(self.r0)]

        obs = self._make_obs(self.r0, self.r_prev)
        info = {
            "episode_id": self.episode_id,
            "setup_time": float(self.setup_time),
            "r0": float(self.r0),
            "nx": int(self.mkw["nx"]),
            "ny": int(self.mkw["ny"]),
            "nz": int(self.mkw["nz"]),
            "rhs_seed": int(self.mkw["rhs_seed"]),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "cz": float(self.cz),
            "ax": float(self.ax),
            "ay": float(self.ay),
            "az": float(self.az),
            "setup_obs_keys": self.setup_obs_keys,
        }
        for key, value in self.setup_params.items():
            info[key] = value
        for key, value in self.setup_param_overrides.items():
            info[f"override_{key}"] = value
        info.update({f"setup_{k}": v for k, v in self._selection_info.items()})
        return obs, info

    def step(self, action):
        if self.prepared_env is None:
            raise RuntimeError("Environment must be reset before step()")

        prev_w = self.last_w
        prev_sd = self.last_sweeps_down
        prev_su = self.last_sweeps_up
        w, sweeps_down, sweeps_up = decode_policy_action(
            action,
            w_only=self.w_only,
            w_center=self.w_center,
            w_scale=self.w_scale,
            sweeps_min=self.sweeps_min,
            sweeps_max=self.sweeps_max,
            cycle=self.cycle,
            last_w=self.last_w,
            w_init=self.w_init,
            sweeps_init=self.sweeps_init,
            w_smooth_alpha=self.w_smooth_alpha,
        )

        self.last_w = float(w)
        self.last_sweeps_down = int(sweeps_down)
        self.last_sweeps_up = int(sweeps_up)
        t0 = time.perf_counter()
        r_solver, dt_solver = self.prepared_env.step_rl(
            relax_weight=float(w),
            sweeps_down=int(sweeps_down),
            sweeps_up=int(sweeps_up),
        )
        dt_wall = time.perf_counter() - t0
        self.solve_runtime += float(dt_solver)
        self.cycle += 1

        if (
            (not np.isfinite(r_solver)) or (r_solver <= 0.0)
            or (not np.isfinite(dt_solver)) or (dt_solver <= 0.0)
            or (not np.isfinite(dt_wall)) or (dt_wall <= 0.0)
        ):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -10.0, False, True, {"bad_step": True, "r": r_solver, "dt_wall": dt_wall, "dt_solver": dt_solver}

        eps = 1e-30
        r_prev = max(float(self.r_prev), eps)
        r_cur = max(float(r_solver), eps)
        r0 = max(float(getattr(self, "r0", r_prev)), 1e-300)
        log_prev = math.log(r_prev + eps)
        log_cur = math.log(r_cur + eps)
        log_tol = math.log(float(self.tol) * r0 + eps)
        log_drop = max(0.0, log_prev - log_cur)
        rel_drop = (r_prev - r_cur) / r_prev
        dt_eff = max(float(dt_solver), 1.0e-6)

        if self.reward_mode == 0:
            reward = self.reward_alpha * (log_drop / dt_eff)
        elif self.reward_mode == 1:
            reward = self.reward_alpha * (rel_drop / dt_eff)
        elif self.reward_mode == 2:
            remaining = max(log_prev - log_tol, 1.0e-6)
            reward = self.reward_alpha * ((log_drop / remaining) / dt_eff)
        elif self.reward_mode == 3:
            reward = -dt_eff
        else:
            reward = self.reward_alpha * (log_drop / dt_eff)

        reward -= self.dt_penalty * dt_eff
        terminated = bool(np.isfinite(r_cur) and (r_cur / r0) <= float(self.tol))
        truncated = bool((not terminated) and self.cycle >= int(self.max_cycles))
        if terminated:
            reward += self.term_bonus
        if truncated:
            reward -= self.trunc_penalty
        if self.cycle_penalty > 0.0:
            reward -= self.cycle_penalty
        if self.sweep_penalty > 0.0:
            reward -= self.sweep_penalty * float(sweeps_down + sweeps_up)
        if self.w_change_penalty > 0.0:
            reward -= self.w_change_penalty * abs(float(w) - float(prev_w))
        if self.sweeps_change_penalty > 0.0:
            reward -= self.sweeps_change_penalty * (
                abs(int(sweeps_down) - int(prev_sd)) + abs(int(sweeps_up) - int(prev_su))
            )

        obs = self._make_obs(r_cur, self.r_prev)
        self.r_prev = float(r_cur)
        self.residual_curve.append(float(r_cur))
        info = {
            "r": float(r_cur),
            "r_prev": float(r_prev),
            "rel_drop": float(rel_drop),
            "dt": float(dt_solver),
            "dt_wall": float(dt_wall),
            "dt_solver": float(dt_solver),
            "setup_time": float(self.setup_time),
            "solve_time": float(self.solve_runtime),
            "time_with_setup": float(self.setup_time + self.solve_runtime),
            "w": float(w),
            "sweeps_down": int(sweeps_down),
            "sweeps_up": int(sweeps_up),
            "cycle": int(self.cycle),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "cz": float(self.cz),
            "setup_obs_keys": self.setup_obs_keys,
        }
        for key, value in self.setup_params.items():
            info[key] = value

        if terminated or truncated:
            info["residual_curve"] = np.asarray(self.residual_curve, dtype=np.float64)
            info["cycles_to_stop"] = int(len(self.residual_curve) - 1)
            failed = not terminated
            total_runtime = float(self.setup_time + self.solve_runtime)
            info["failed"] = bool(failed)
            if self.bandit_update and self.setup_mode == "bandit":
                loss = total_runtime
                if failed:
                    if self.setup_bandit_loss_mode == "test10":
                        loss += float(self.setup_bandit_failure_add)
                    else:
                        loss += float(max(total_runtime, self.setup_failure_min_runtime_sec) * self.setup_failure_penalty_multiplier)
                self.provider.update(
                    loss=float(loss),
                    context=self.context,
                    params=self.setup_params,
                    outcome={
                        "failed": bool(failed),
                        "runtime": float(total_runtime),
                        "setup_runtime": float(self.setup_time),
                        "solve_runtime": float(self.solve_runtime),
                        "residual_norm": float(r_cur),
                        "iterations": int(self.cycle),
                    },
                )

        return obs, float(reward), terminated, truncated, info

    def close(self):
        if self.prepared_env is not None:
            self.prepared_env.close()
            self.prepared_env = None
