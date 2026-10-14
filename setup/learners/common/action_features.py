from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np


ParameterKind = str
_NUMERIC_KINDS = {"continuous", "integer"}
_CATEGORICAL_KIND = "categorical"


def _as_tuple(values: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(values) if not isinstance(values, tuple) else values


def _normalize_active_if(
    active_if: Optional[
        Mapping[str, Sequence[Any]] | Sequence[Tuple[str, Sequence[Any]]]
    ],
) -> Tuple[Tuple[str, Tuple[Any, ...]], ...]:
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
            raise ValueError(
                f"active_if[{dep!r}] must contain at least one allowed value"
            )
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
            raise ValueError(
                f"ParameterSpec[{name!r}] must define at least one candidate value"
            )

        active_if = _normalize_active_if(self.active_if)

        if kind in _NUMERIC_KINDS:
            norm_vals = tuple(float(v) for v in values)
            if not np.all(np.isfinite(np.asarray(norm_vals, dtype=float))):
                raise ValueError(
                    f"ParameterSpec[{name!r}] numeric values must be finite"
                )
            default = float(self.default)
            if not np.isfinite(default):
                raise ValueError(f"ParameterSpec[{name!r}] default must be finite")
            center = float(default if self.center is None else self.center)
            scale = float(1.0 if self.scale is None else self.scale)
            if not np.isfinite(center):
                raise ValueError(f"ParameterSpec[{name!r}] center must be finite")
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    f"ParameterSpec[{name!r}] scale must be finite and > 0"
                )

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
        params = (
            tuple(self.parameters)
            if not isinstance(self.parameters, tuple)
            else self.parameters
        )
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


def _is_parameter_active(
    param: ParameterSpec, effective_action: Mapping[str, Any]
) -> bool:
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


def default_action_from_parameter_space_spec(
    parameter_spec: ParameterSpaceSpec,
) -> Dict[str, Any]:
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


def iter_actions_from_parameter_space_spec(parameter_spec: ParameterSpaceSpec):
    """Yield canonical actions in the stable parameter-spec enumeration order."""
    params = parameter_spec.parameters
    seen = set()

    def rec(i: int, current: Dict[str, Any]):
        if i >= len(params):
            canonical = canonicalize_action_from_spec(current, parameter_spec)
            key = action_key_from_parameter_space_spec(canonical, parameter_spec)
            if key not in seen:
                seen.add(key)
                yield canonical
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
            yield from rec(i + 1, next_current)

    yield from rec(0, {})


def enumerate_actions_from_parameter_space_spec(
    parameter_spec: ParameterSpaceSpec,
) -> Tuple[Dict[str, Any], ...]:
    return tuple(iter_actions_from_parameter_space_spec(parameter_spec))


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
            param
            for param in self.parameter_spec.parameters
            if param.kind in _NUMERIC_KINDS
        )
        self.categorical_params = tuple(
            param
            for param in self.parameter_spec.parameters
            if param.kind == _CATEGORICAL_KIND
        )
        self.numeric_names = tuple(param.name for param in self.numeric_params)
        self.categorical_levels = {
            param.name: tuple(value for value in param.values if value != param.default)
            for param in self.categorical_params
        }
        self.numeric_dim = len(self.numeric_params)
        self.categorical_dim = sum(
            len(levels) for levels in self.categorical_levels.values()
        )
        self.mixed_dim = self.numeric_dim * self.categorical_dim
        self.numeric_feature_dim = (
            self.numeric_dim
            + self.numeric_dim
            + (self.numeric_dim * (self.numeric_dim - 1)) // 2
        )
        self.feature_dim = (
            self.numeric_feature_dim + self.categorical_dim + self.mixed_dim
        )

    def _normalized_numeric_table(
        self,
        values: np.ndarray,
        *,
        active: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        numeric_values = np.asarray(values, dtype=float)
        if numeric_values.ndim == 1:
            numeric_values = numeric_values.reshape(1, -1)
        expected_shape = (numeric_values.shape[0], self.numeric_dim)
        if numeric_values.shape != expected_shape:
            raise ValueError(
                "numeric values must have shape "
                f"(rows, {self.numeric_dim}), got {numeric_values.shape}"
            )
        centers = np.asarray(
            [float(param.center) for param in self.numeric_params],
            dtype=float,
        )
        scales = np.asarray(
            [float(param.scale) for param in self.numeric_params],
            dtype=float,
        )
        normalized = (numeric_values - centers[None, :]) / scales[None, :]
        if active is not None:
            active_mask = np.asarray(active, dtype=bool)
            if active_mask.shape != numeric_values.shape:
                raise ValueError(
                    "numeric active mask must match the numeric-value shape"
                )
            normalized = np.where(active_mask, normalized, 0.0)
        return normalized

    def encode_numeric_table(
        self,
        values: np.ndarray,
        *,
        active: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode a table containing only the numeric parameter prefix."""

        normalized = self._normalized_numeric_table(values, active=active)
        pairwise = (
            np.column_stack(
                [
                    normalized[:, i] * normalized[:, j]
                    for i in range(self.numeric_dim)
                    for j in range(i + 1, self.numeric_dim)
                ]
            )
            if self.numeric_dim > 1
            else np.zeros((normalized.shape[0], 0), dtype=float)
        )
        return np.concatenate(
            [normalized, normalized * normalized, pairwise],
            axis=1,
        )

    def encode_action(self, params: Mapping[str, Any]) -> np.ndarray:
        action = canonicalize_action_from_spec(params, self.parameter_spec)

        numeric_vals = []
        numeric_active = []
        categorical_blocks = []
        for param in self.parameter_spec.parameters:
            active = _is_parameter_active(param, action)
            if param.kind in _NUMERIC_KINDS:
                numeric_vals.append(float(action[param.name]))
                numeric_active.append(bool(active))
            else:
                levels = self.categorical_levels[param.name]
                if not active:
                    categorical_blocks.append(np.zeros(len(levels), dtype=float))
                else:
                    value = action[param.name]
                    categorical_blocks.append(
                        np.asarray(
                            [1.0 if value == level else 0.0 for level in levels],
                            dtype=float,
                        )
                    )

        numeric_features = self.encode_numeric_table(
            np.asarray(numeric_vals, dtype=float).reshape(1, -1),
            active=np.asarray(numeric_active, dtype=bool).reshape(1, -1),
        )[0]
        normalized_numeric = numeric_features[: self.numeric_dim]
        categorical = (
            np.concatenate(categorical_blocks, axis=0)
            if categorical_blocks
            else np.zeros(0, dtype=float)
        )

        mixed_blocks = []
        for cat_block in categorical_blocks:
            for indicator in cat_block:
                mixed_blocks.append(float(indicator) * normalized_numeric)
        mixed = (
            np.concatenate(mixed_blocks, axis=0)
            if mixed_blocks
            else np.zeros(0, dtype=float)
        )

        return np.concatenate(
            [numeric_features, categorical, mixed],
            axis=0,
        )

    def encode_actions(self, actions: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if not actions:
            raise ValueError("actions must be non-empty")
        encoded = np.empty((len(actions), self.feature_dim), dtype=float)
        for index, action in enumerate(actions):
            encoded[index] = self.encode_action(action)
        return encoded
