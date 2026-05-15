from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from ._amg_action_features import (
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    canonicalize_action_from_spec,
)


def _repeat_context(x: np.ndarray, n: int) -> np.ndarray:
    return np.repeat(np.asarray(x, dtype=float).reshape(1, -1), int(n), axis=0)


def _pairwise_numeric_products(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ValueError("values must have shape (n, d)")
    if values.shape[1] <= 1:
        return np.zeros((values.shape[0], 0), dtype=float)
    cols = []
    for left in range(values.shape[1]):
        for right in range(left + 1, values.shape[1]):
            cols.append((values[:, left] * values[:, right]).reshape(-1, 1))
    return np.concatenate(cols, axis=1) if cols else np.zeros((values.shape[0], 0), dtype=float)


@dataclass(frozen=True)
class Tune7FeatureBuilder:
    actions: Tuple[Mapping[str, Any], ...]
    parameter_spec: ParameterSpaceSpec
    context_interaction_indices: Tuple[int, ...] = (1, 2, 3, 4)

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions:
            raise ValueError("actions must be non-empty")
        spec = self.parameter_spec
        numeric_params = tuple(p for p in spec.parameters if p.kind in {"continuous", "integer"})
        categorical_params = tuple(p for p in spec.parameters if p.kind == "categorical")
        if len(numeric_params) != 5:
            raise ValueError("tune7 feature builder expects exactly 5 numeric parameters")
        if len(categorical_params) != 2:
            raise ValueError("tune7 feature builder expects exactly 2 categorical parameters")

        action_encoder = GenericActionFeatureEncoder(spec)
        numeric_name_to_index: Dict[str, int] = {}
        numeric_value_to_index: Dict[str, Dict[Any, int]] = {}
        grid_shape = []
        standardized_numeric_tables = []
        for idx, param in enumerate(numeric_params):
            numeric_name_to_index[param.name] = int(idx)
            numeric_value_to_index[param.name] = {value: int(i) for i, value in enumerate(param.values)}
            grid_shape.append(int(len(param.values)))
            standardized_numeric_tables.append(
                np.asarray(
                    [
                        (float(value) - float(param.center)) / float(param.scale)
                        for value in param.values
                    ],
                    dtype=float,
                )
            )

        lattice_indices = np.zeros((len(actions), len(numeric_params)), dtype=np.uint16)
        for row, action in enumerate(actions):
            for idx, param in enumerate(numeric_params):
                raw_value = action[param.name]
                value_idx = numeric_value_to_index[param.name].get(raw_value)
                if value_idx is None:
                    param_vals = np.asarray(param.values, dtype=float)
                    value_idx = int(np.argmin(np.abs(param_vals - float(raw_value))))
                lattice_indices[row, idx] = np.uint16(value_idx)

        coars_param, interp_param = categorical_params
        coars_levels = tuple(coars_param.values)
        interp_levels = tuple(interp_param.values)
        coars_level_to_index = {value: int(i) for i, value in enumerate(coars_levels)}
        interp_level_to_index = {value: int(i) for i, value in enumerate(interp_levels)}
        coars_default_index = int(coars_level_to_index[coars_param.default])
        interp_default_index = int(interp_level_to_index[interp_param.default])
        coars_level_indices = np.zeros(len(actions), dtype=np.uint8)
        interp_level_indices = np.zeros(len(actions), dtype=np.uint8)
        regime_keys = []
        for row, action in enumerate(actions):
            coars_idx = int(coars_level_to_index[action[coars_param.name]])
            interp_idx = int(interp_level_to_index[action[interp_param.name]])
            coars_level_indices[row] = np.uint8(coars_idx)
            interp_level_indices[row] = np.uint8(interp_idx)
            regime_keys.append((int(action[coars_param.name]), int(action[interp_param.name])))
        regime_key_to_index = {key: idx for idx, key in enumerate(sorted(set(regime_keys)))}
        regime_indices = np.asarray([regime_key_to_index[key] for key in regime_keys], dtype=np.uint8)

        lookup = np.full((len(regime_key_to_index), *tuple(grid_shape)), -1, dtype=np.int32)
        for arm in range(len(actions)):
            idx = tuple(int(v) for v in lattice_indices[arm])
            lookup[(int(regime_indices[arm]), *idx)] = int(arm)

        pair_names = (
            ("strong_threshold", "max_row_sum"),
            ("strong_threshold", "trunc_factor"),
            ("max_row_sum", "trunc_factor"),
        )
        pair_index_tuples = tuple(
            (int(numeric_name_to_index[left]), int(numeric_name_to_index[right]))
            for left, right in pair_names
        )

        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "numeric_params", numeric_params)
        object.__setattr__(self, "categorical_params", categorical_params)
        object.__setattr__(self, "numeric_name_to_index", numeric_name_to_index)
        object.__setattr__(self, "numeric_value_to_index", numeric_value_to_index)
        object.__setattr__(self, "grid_shape", tuple(grid_shape))
        object.__setattr__(self, "standardized_numeric_tables", tuple(standardized_numeric_tables))
        object.__setattr__(self, "lattice_indices", lattice_indices)
        object.__setattr__(self, "coars_levels", coars_levels)
        object.__setattr__(self, "interp_levels", interp_levels)
        object.__setattr__(self, "coars_level_indices", coars_level_indices)
        object.__setattr__(self, "interp_level_indices", interp_level_indices)
        object.__setattr__(self, "coars_level_to_index", coars_level_to_index)
        object.__setattr__(self, "interp_level_to_index", interp_level_to_index)
        object.__setattr__(self, "coars_default_index", coars_default_index)
        object.__setattr__(self, "interp_default_index", interp_default_index)
        object.__setattr__(self, "coars_encoded_dim", len(coars_levels) - 1)
        object.__setattr__(self, "interp_encoded_dim", len(interp_levels) - 1)
        object.__setattr__(self, "regime_keys", tuple(regime_keys))
        object.__setattr__(self, "regime_key_to_index", regime_key_to_index)
        object.__setattr__(self, "regime_indices", regime_indices)
        object.__setattr__(self, "lookup", lookup)
        object.__setattr__(self, "pair_index_tuples", pair_index_tuples)
        object.__setattr__(self, "action_encoder", action_encoder)

    @property
    def K(self) -> int:
        return len(self.actions)

    @property
    def z_dim(self) -> int:
        return len(self.numeric_params)

    @property
    def g_dim(self) -> int:
        return int(self.action_encoder.feature_dim)

    @property
    def regime_dim(self) -> int:
        return len(self.regime_key_to_index)

    @property
    def e_c_dim(self) -> int:
        return int(self.coars_encoded_dim)

    @property
    def e_i_dim(self) -> int:
        return int(self.interp_encoded_dim)

    def _arm_indices(self, arms: np.ndarray | Sequence[int] | None) -> np.ndarray:
        if arms is None:
            return np.arange(self.K, dtype=int)
        return np.asarray(arms, dtype=int).reshape(-1)

    def _categorical_block(
        self,
        full_indices: np.ndarray,
        *,
        total_levels: int,
        default_index: int,
    ) -> np.ndarray:
        encoded_dim = int(total_levels) - 1
        out = np.zeros((full_indices.size, encoded_dim), dtype=float)
        if encoded_dim <= 0:
            return out
        nondefault = np.flatnonzero(full_indices != int(default_index))
        if nondefault.size == 0:
            return out
        columns = np.asarray(full_indices[nondefault], dtype=int)
        columns = columns - (columns > int(default_index)).astype(int)
        out[nondefault, columns] = 1.0
        return out

    def z_matrix(self, arms: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        arm_idx = self._arm_indices(arms)
        lattice = self.lattice_indices[arm_idx]
        cols = [
            np.asarray(table[np.asarray(lattice[:, dim], dtype=int)], dtype=float)
            for dim, table in enumerate(self.standardized_numeric_tables)
        ]
        return np.stack(cols, axis=1)

    def e_c_matrix(self, arms: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        arm_idx = self._arm_indices(arms)
        return self._categorical_block(
            self.coars_level_indices[arm_idx],
            total_levels=len(self.coars_levels),
            default_index=self.coars_default_index,
        )

    def e_i_matrix(self, arms: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        arm_idx = self._arm_indices(arms)
        return self._categorical_block(
            self.interp_level_indices[arm_idx],
            total_levels=len(self.interp_levels),
            default_index=self.interp_default_index,
        )

    def e_ci_matrix(self, regime_indices: Sequence[int]) -> np.ndarray:
        reg_idx = np.asarray(regime_indices, dtype=int).reshape(-1)
        out = np.zeros((reg_idx.size, self.regime_dim), dtype=float)
        if reg_idx.size > 0:
            out[np.arange(reg_idx.size, dtype=int), reg_idx] = 1.0
        return out

    def g_matrix(self, arms: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        arm_idx = self._arm_indices(arms)
        z = self.z_matrix(arm_idx)
        e_c = self.e_c_matrix(arm_idx)
        e_i = self.e_i_matrix(arm_idx)
        categorical = np.concatenate([e_c, e_i], axis=1)
        mixed = (
            (categorical[:, :, None] * z[:, None, :]).reshape(arm_idx.size, -1)
            if categorical.shape[1] > 0
            else np.zeros((arm_idx.size, 0), dtype=float)
        )
        return np.concatenate(
            [
                z,
                z * z,
                _pairwise_numeric_products(z),
                categorical,
                mixed,
            ],
            axis=1,
        )

    def arm_for_params(self, params: Mapping[str, Any]) -> int:
        action = canonicalize_action_from_spec(params, self.parameter_spec)
        numeric_idx = []
        for param in self.numeric_params:
            raw_value = action[param.name]
            value_idx = self.numeric_value_to_index[param.name].get(raw_value)
            if value_idx is None:
                param_vals = np.asarray(param.values, dtype=float)
                value_idx = int(np.argmin(np.abs(param_vals - float(raw_value))))
            numeric_idx.append(int(value_idx))
        regime_key = (
            int(action[self.categorical_params[0].name]),
            int(action[self.categorical_params[1].name]),
        )
        regime_idx = self.regime_key_to_index.get(regime_key)
        if regime_idx is None:
            raise ValueError("params do not match any regime in the provided action set")
        arm = int(self.lookup[(int(regime_idx), *numeric_idx)])
        if arm < 0:
            raise ValueError("params do not match any action in the provided action set")
        return arm

    def phi_v4_matrix(self, x: Sequence[float], arms: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        arm_idx = self._arm_indices(arms)
        g = self.g_matrix(arm_idx)
        blocks = [_repeat_context(x_arr, arm_idx.size), g]
        for ctx_idx in self.context_interaction_indices:
            blocks.append(float(x_arr[int(ctx_idx)]) * g)
        return np.concatenate(blocks, axis=1)

    def quadratic_oracle_matrix(self, x: Sequence[float], arms: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        arm_idx = self._arm_indices(arms)
        z = self.z_matrix(arm_idx)
        e_c = self.e_c_matrix(arm_idx)
        e_i = self.e_i_matrix(arm_idx)
        e_ci = self.e_ci_matrix(self.regime_indices[arm_idx])

        pairs = [
            (z[:, li] * z[:, ri]).reshape(-1, 1)
            for li, ri in self.pair_index_tuples
        ]

        blocks = [_repeat_context(x_arr, arm_idx.size), z, z * z]
        if pairs:
            blocks.append(np.concatenate(pairs, axis=1))
        blocks.extend([e_c, e_i, e_ci])
        for ctx_idx in self.context_interaction_indices:
            blocks.append(float(x_arr[int(ctx_idx)]) * z)
        for ctx_idx in self.context_interaction_indices:
            blocks.append(float(x_arr[int(ctx_idx)]) * e_c)
        for ctx_idx in self.context_interaction_indices:
            blocks.append(float(x_arr[int(ctx_idx)]) * e_i)
        return np.concatenate(blocks, axis=1)

    def hierarchical_regime_matrix(self, x: Sequence[float], regime_indices: Sequence[int]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        reg_idx = np.asarray(regime_indices, dtype=int).reshape(-1)
        e_ci = self.e_ci_matrix(reg_idx)
        blocks = [_repeat_context(x_arr, reg_idx.size), e_ci]
        for ctx_idx in self.context_interaction_indices:
            blocks.append(float(x_arr[int(ctx_idx)]) * e_ci)
        return np.concatenate(blocks, axis=1)

    def hierarchical_numeric_matrix(self, x: Sequence[float], arms: np.ndarray | Sequence[int] | None = None) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        arm_idx = self._arm_indices(arms)
        z = self.z_matrix(arm_idx)
        pairs = [
            (z[:, li] * z[:, ri]).reshape(-1, 1)
            for li, ri in self.pair_index_tuples
        ]
        blocks = [_repeat_context(x_arr, arm_idx.size), z, z * z]
        if pairs:
            blocks.append(np.concatenate(pairs, axis=1))
        for ctx_idx in self.context_interaction_indices:
            blocks.append(float(x_arr[int(ctx_idx)]) * z)
        return np.concatenate(blocks, axis=1)
