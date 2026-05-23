from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ParameterKind = str
_NUMERIC_KINDS = {"continuous", "integer"}
_CATEGORICAL_KIND = "categorical"


def _as_tuple(values: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(values) if not isinstance(values, tuple) else values


def _normalize_active_if(active_if: Optional[Mapping[str, Sequence[Any]] | Sequence[Tuple[str, Sequence[Any]]]]) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
    if active_if is None:
        return ()
    if isinstance(active_if, Mapping):
        items = active_if.items()
    else:
        items = active_if
    out = []
    for name, allowed in items:
        dep = str(name).strip()
        vals = _as_tuple(tuple(allowed))
        if not dep:
            raise ValueError("active_if dependency names must be non-empty")
        if not vals:
            raise ValueError(f"active_if[{dep!r}] must contain at least one allowed value")
        out.append((dep, vals))
    return tuple(out)


def _canonical_numeric(value: Any) -> float:
    return float(round(float(value), 12))


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    kind: ParameterKind
    values: Tuple[Any, ...]
    default: Any
    center: Optional[float] = None
    scale: Optional[float] = None
    active_if: Tuple[Tuple[str, Tuple[Any, ...]], ...] = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("ParameterSpec.name must be non-empty")

        kind = str(self.kind).strip().lower()
        if kind not in (_NUMERIC_KINDS | {_CATEGORICAL_KIND}):
            raise ValueError(f"unsupported parameter kind: {self.kind!r}")

        values = _as_tuple(self.values)
        if not values:
            raise ValueError(f"ParameterSpec[{name!r}] must define at least one candidate value")

        active_if = _normalize_active_if(self.active_if)

        if kind in _NUMERIC_KINDS:
            norm_vals = tuple(float(v) for v in values)
            if not np.all(np.isfinite(np.asarray(norm_vals, dtype=float))):
                raise ValueError(f"ParameterSpec[{name!r}] numeric values must be finite")
            default = float(self.default)
            if not np.isfinite(default):
                raise ValueError(f"ParameterSpec[{name!r}] default must be finite")
            center = float(default if self.center is None else self.center)
            scale = float(1.0 if self.scale is None else self.scale)
            if not np.isfinite(center):
                raise ValueError(f"ParameterSpec[{name!r}] center must be finite")
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"ParameterSpec[{name!r}] scale must be finite and > 0")

            object.__setattr__(self, "values", norm_vals)
            object.__setattr__(self, "default", default)
            object.__setattr__(self, "center", center)
            object.__setattr__(self, "scale", scale)
        else:
            default = self.default
            object.__setattr__(self, "values", values)
            object.__setattr__(self, "default", default)
            object.__setattr__(self, "center", None)
            object.__setattr__(self, "scale", None)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "active_if", active_if)


@dataclass(frozen=True)
class ParameterSpaceSpec:
    parameters: Tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        params = tuple(self.parameters) if not isinstance(self.parameters, tuple) else self.parameters
        if not params:
            raise ValueError("ParameterSpaceSpec.parameters must be non-empty")
        names = [param.name for param in params]
        if len(names) != len(set(names)):
            raise ValueError("ParameterSpaceSpec parameter names must be unique")

        seen = set()
        for param in params:
            for dep_name, _allowed in param.active_if:
                if dep_name not in seen:
                    raise ValueError(
                        f"ParameterSpec[{param.name!r}] active_if dependency {dep_name!r} must refer to an earlier parameter"
                    )
            seen.add(param.name)

        object.__setattr__(self, "parameters", params)

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        return tuple(param.name for param in self.parameters)


def _is_parameter_active(param: ParameterSpec, effective_action: Mapping[str, Any]) -> bool:
    for dep_name, allowed in param.active_if:
        if effective_action.get(dep_name, None) not in allowed:
            return False
    return True


def canonicalize_action_from_spec(
    params: Mapping[str, Any],
    parameter_spec: ParameterSpaceSpec,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for param in parameter_spec.parameters:
        active = _is_parameter_active(param, out)
        if not active:
            value = param.default
        else:
            value = params.get(param.name, param.default)
            if param.kind in _NUMERIC_KINDS:
                value = float(value)
                if not np.isfinite(value):
                    raise ValueError(f"action[{param.name!r}] must be finite")
            allowed = tuple(param.values)
            if value not in allowed and value != param.default:
                raise ValueError(
                    f"action[{param.name!r}]={value!r} is not compatible with the parameter spec"
                )
        if param.kind == "integer":
            value = int(round(float(value)))
        elif param.kind == "continuous":
            value = float(value)
        out[param.name] = value
    return out


def default_action_from_parameter_space_spec(parameter_spec: ParameterSpaceSpec) -> Dict[str, Any]:
    raw = {param.name: param.default for param in parameter_spec.parameters}
    return canonicalize_action_from_spec(raw, parameter_spec)


def action_from_ordered_values(
    values: Sequence[Any],
    parameter_spec: ParameterSpaceSpec,
) -> Dict[str, Any]:
    ordered = tuple(values)
    params = parameter_spec.parameters
    if len(ordered) != len(params):
        raise ValueError(
            f"expected {len(params)} initial-guess values in parameter-spec order, got {len(ordered)}"
        )
    raw = {param.name: value for param, value in zip(params, ordered)}
    return canonicalize_action_from_spec(raw, parameter_spec)


def action_key_from_parameter_space_spec(
    params: Mapping[str, Any],
    parameter_spec: ParameterSpaceSpec,
) -> Tuple[Any, ...]:
    action = canonicalize_action_from_spec(params, parameter_spec)
    key = []
    for param in parameter_spec.parameters:
        value = action[param.name]
        if param.kind in _NUMERIC_KINDS:
            key.append(_canonical_numeric(value))
        else:
            key.append(value)
    return tuple(key)


def enumerate_actions_from_parameter_space_spec(parameter_spec: ParameterSpaceSpec) -> Tuple[Dict[str, Any], ...]:
    params = parameter_spec.parameters
    seen = set()
    actions = []

    def rec(i: int, current: Dict[str, Any]) -> None:
        if i >= len(params):
            canonical = canonicalize_action_from_spec(current, parameter_spec)
            key = action_key_from_parameter_space_spec(canonical, parameter_spec)
            if key not in seen:
                seen.add(key)
                actions.append(canonical)
            return

        param = params[i]
        active = _is_parameter_active(param, current)
        options = (param.default,) if not active else tuple(param.values)
        for raw_value in options:
            next_current = dict(current)
            if param.kind == "integer":
                next_current[param.name] = int(round(float(raw_value)))
            elif param.kind == "continuous":
                next_current[param.name] = float(raw_value)
            else:
                next_current[param.name] = raw_value
            rec(i + 1, next_current)

    rec(0, {})
    return tuple(actions)


def _pairwise_products(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        return np.zeros(0, dtype=float)
    out = []
    for i in range(values.size):
        for j in range(i + 1, values.size):
            out.append(float(values[i] * values[j]))
    return np.asarray(out, dtype=float)


class GenericActionFeatureEncoder:
    def __init__(self, parameter_spec: ParameterSpaceSpec) -> None:
        self.parameter_spec = parameter_spec
        self.numeric_params = tuple(
            param for param in self.parameter_spec.parameters if param.kind in _NUMERIC_KINDS
        )
        self.categorical_params = tuple(
            param for param in self.parameter_spec.parameters if param.kind == _CATEGORICAL_KIND
        )
        self.numeric_names = tuple(param.name for param in self.numeric_params)
        self.categorical_levels = {
            param.name: tuple(value for value in param.values if value != param.default)
            for param in self.categorical_params
        }
        self.numeric_dim = len(self.numeric_params)
        self.categorical_dim = sum(len(levels) for levels in self.categorical_levels.values())
        self.mixed_dim = self.numeric_dim * self.categorical_dim
        self.feature_dim = (
            self.numeric_dim
            + self.numeric_dim
            + (self.numeric_dim * (self.numeric_dim - 1)) // 2
            + self.categorical_dim
            + self.mixed_dim
        )

    def encode_action(self, params: Mapping[str, Any]) -> np.ndarray:
        action = canonicalize_action_from_spec(params, self.parameter_spec)

        numeric_vals = []
        categorical_blocks = []
        for param in self.parameter_spec.parameters:
            active = _is_parameter_active(param, action)
            if param.kind in _NUMERIC_KINDS:
                if not active:
                    numeric_vals.append(0.0)
                else:
                    value = float(action[param.name])
                    numeric_vals.append((value - float(param.center)) / float(param.scale))
            else:
                levels = self.categorical_levels[param.name]
                if not active:
                    categorical_blocks.append(np.zeros(len(levels), dtype=float))
                else:
                    value = action[param.name]
                    categorical_blocks.append(
                        np.asarray([1.0 if value == level else 0.0 for level in levels], dtype=float)
                    )

        numeric = np.asarray(numeric_vals, dtype=float)
        numeric_sq = numeric * numeric
        numeric_pairwise = _pairwise_products(numeric)
        categorical = (
            np.concatenate(categorical_blocks, axis=0)
            if categorical_blocks
            else np.zeros(0, dtype=float)
        )

        mixed_blocks = []
        for cat_block in categorical_blocks:
            for indicator in cat_block:
                mixed_blocks.append(float(indicator) * numeric)
        mixed = np.concatenate(mixed_blocks, axis=0) if mixed_blocks else np.zeros(0, dtype=float)

        return np.concatenate([numeric, numeric_sq, numeric_pairwise, categorical, mixed], axis=0)

    def encode_actions(self, actions: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not actions:
            raise ValueError("actions must be non-empty")
        return np.vstack([self.encode_action(action) for action in actions])


def action_feature_dimension_from_spec(parameter_spec: ParameterSpaceSpec) -> int:
    return int(GenericActionFeatureEncoder(parameter_spec).feature_dim)


def make_tune3_parameter_space_spec(
    *,
    thresholds: Sequence[float],
    max_row_sums: Sequence[float],
    trunc_factors: Sequence[float],
    default_params: Optional[Dict[str, Any]] = None,
    scales: Sequence[float] = (0.25, 0.10, 0.20),
) -> ParameterSpaceSpec:
    defaults = dict(default_params or {})
    scale_vals = tuple(float(v) for v in scales)
    if len(scale_vals) != 3:
        raise ValueError("tune3 scales must have length 3")
    return ParameterSpaceSpec(
        (
            ParameterSpec(
                name="strong_threshold",
                kind="continuous",
                values=tuple(float(v) for v in thresholds),
                default=float(defaults.get("strong_threshold", 0.25)),
                center=float(defaults.get("strong_threshold", 0.25)),
                scale=float(scale_vals[0]),
            ),
            ParameterSpec(
                name="max_row_sum",
                kind="continuous",
                values=tuple(float(v) for v in max_row_sums),
                default=float(defaults.get("max_row_sum", 0.90)),
                center=float(defaults.get("max_row_sum", 0.90)),
                scale=float(scale_vals[1]),
            ),
            ParameterSpec(
                name="trunc_factor",
                kind="continuous",
                values=tuple(float(v) for v in trunc_factors),
                default=float(defaults.get("trunc_factor", 0.00)),
                center=float(defaults.get("trunc_factor", 0.00)),
                scale=float(scale_vals[2]),
            ),
        )
    )


def make_tune5_parameter_space_spec(
    *,
    thresholds: Sequence[float],
    max_row_sums: Sequence[float],
    trunc_factors: Sequence[float],
    p_max_elmts_values: Sequence[int],
    agg_num_levels_values: Sequence[int],
    default_params: Optional[Dict[str, Any]] = None,
    scales: Sequence[float] = (0.25, 0.10, 0.20, 4.0, 1.0),
) -> ParameterSpaceSpec:
    defaults = dict(default_params or {})
    scale_vals = tuple(float(v) for v in scales)
    if len(scale_vals) != 5:
        raise ValueError("tune5 scales must have length 5")
    return ParameterSpaceSpec(
        (
            ParameterSpec(
                name="strong_threshold",
                kind="continuous",
                values=tuple(float(v) for v in thresholds),
                default=float(defaults.get("strong_threshold", 0.25)),
                center=float(defaults.get("strong_threshold", 0.25)),
                scale=float(scale_vals[0]),
            ),
            ParameterSpec(
                name="max_row_sum",
                kind="continuous",
                values=tuple(float(v) for v in max_row_sums),
                default=float(defaults.get("max_row_sum", 0.90)),
                center=float(defaults.get("max_row_sum", 0.90)),
                scale=float(scale_vals[1]),
            ),
            ParameterSpec(
                name="trunc_factor",
                kind="continuous",
                values=tuple(float(v) for v in trunc_factors),
                default=float(defaults.get("trunc_factor", 0.00)),
                center=float(defaults.get("trunc_factor", 0.00)),
                scale=float(scale_vals[2]),
            ),
            ParameterSpec(
                name="P_max_elmts",
                kind="integer",
                values=tuple(int(v) for v in p_max_elmts_values),
                default=int(defaults.get("P_max_elmts", 4)),
                center=float(defaults.get("P_max_elmts", 4)),
                scale=float(scale_vals[3]),
            ),
            ParameterSpec(
                name="agg_num_levels",
                kind="integer",
                values=tuple(int(v) for v in agg_num_levels_values),
                default=int(defaults.get("agg_num_levels", 0)),
                center=float(defaults.get("agg_num_levels", 0)),
                scale=float(scale_vals[4]),
            ),
        )
    )
ACTION_FEATURE_KEYS = (
    "strong_threshold",
    "max_row_sum",
    "trunc_factor",
    "P_max_elmts",
    "agg_num_levels",
)

ACTION_FEATURE_DEFAULTS = {
    "P_max_elmts": 4.0,
    "agg_num_levels": 0.0,
}

# The original learners used scales for the first 3 continuous knobs.
# For backward compatibility, callers may still pass only those 3 values;
# the last 2 discrete knobs will use these defaults.
ACTION_FEATURE_DEFAULT_SCALES = (0.25, 0.10, 0.20, 4.0, 1.0)

ACTION_FEATURE_DIM = (
    len(ACTION_FEATURE_KEYS)  # linear
    + len(ACTION_FEATURE_KEYS)  # squares
    + (len(ACTION_FEATURE_KEYS) * (len(ACTION_FEATURE_KEYS) - 1)) // 2  # pairwise products
)


def normalize_action_scales(action_scales: Sequence[float]) -> np.ndarray:
    scales = np.asarray(list(action_scales), dtype=float).reshape(-1)
    if scales.size == 3:
        scales = np.concatenate(
            [scales, np.asarray(ACTION_FEATURE_DEFAULT_SCALES[3:], dtype=float)],
            axis=0,
        )
    if scales.size != len(ACTION_FEATURE_KEYS):
        raise ValueError(
            f"action_scales must have length 3 or {len(ACTION_FEATURE_KEYS)}"
        )
    if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("action_scales must be finite and > 0")
    return scales


def action_center_from_actions(
    actions: Sequence[Dict[str, Any]],
    action_center: Optional[Dict[str, Any]],
) -> np.ndarray:
    means: Dict[str, float] = {}
    for key in ACTION_FEATURE_KEYS:
        vals = [float(a[key]) for a in actions if key in a]
        if vals:
            m = float(np.mean(np.asarray(vals, dtype=float)))
            if not np.isfinite(m):
                raise ValueError(f"non-finite mean for action key {key}")
            means[key] = m

    out = []
    for key in ACTION_FEATURE_KEYS:
        if action_center is not None and key in action_center:
            v = float(action_center[key])
        elif key in means:
            v = float(means[key])
        elif key in ACTION_FEATURE_DEFAULTS:
            v = float(ACTION_FEATURE_DEFAULTS[key])
        else:
            raise KeyError(f"Missing required action-center key: {key}")

        if not np.isfinite(v):
            raise ValueError(f"action_center[{key!r}] must be finite")
        out.append(v)

    return np.asarray(out, dtype=float)


def action_param_vector(params: Dict[str, Any], *, err_prefix: str) -> np.ndarray:
    try:
        th = float(params["strong_threshold"])
        mxrs = float(params["max_row_sum"])
        tr = float(params["trunc_factor"])
    except KeyError as e:
        raise KeyError(f"{err_prefix} action missing required key: {e}") from e

    p_max = float(params.get("P_max_elmts", ACTION_FEATURE_DEFAULTS["P_max_elmts"]))
    agg_nl = float(params.get("agg_num_levels", ACTION_FEATURE_DEFAULTS["agg_num_levels"]))
    out = np.asarray([th, mxrs, tr, p_max, agg_nl], dtype=float)
    if not np.all(np.isfinite(out)):
        raise ValueError("action parameters must be finite")
    return out


def poly2_features(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(list(values), dtype=float).reshape(-1)
    if x.size != len(ACTION_FEATURE_KEYS):
        raise ValueError(f"expected {len(ACTION_FEATURE_KEYS)} action values, got {x.size}")

    feats = [x, x * x]
    cross_terms = []
    for i in range(x.size):
        for j in range(i + 1, x.size):
            cross_terms.append(x[i] * x[j])
    feats.append(np.asarray(cross_terms, dtype=float))
    return np.concatenate(feats, axis=0)
