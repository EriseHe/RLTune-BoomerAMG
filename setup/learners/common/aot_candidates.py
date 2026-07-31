"""Compact action catalogs and ahead-of-time candidate schedules.

The catalog preserves the stable enumeration order used by
``iter_actions_from_parameter_space_spec`` without materializing one Python
dictionary per action.  The schedule fixes candidate IDs before an online run;
a small factorized cache reconstructs one candidate feature block in RAM.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from .action_features import (
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    action_key_from_parameter_space_spec,
    canonicalize_action_from_spec,
    default_action_from_parameter_space_spec,
)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class CompactActionCatalog(Sequence[Dict[str, Any]]):
    """Random-access view of a conditional Cartesian parameter space."""

    is_compact_action_catalog = True

    def __init__(
        self,
        parameter_spec: ParameterSpaceSpec,
        *,
        fixed_params: Mapping[str, Any] | None = None,
    ) -> None:
        self.parameter_spec = parameter_spec
        self.fixed_params = dict(fixed_params or {})
        self._params = tuple(parameter_spec.parameters)
        for param in self._params:
            if len(param.values) != len(set(param.values)):
                raise ValueError(
                    f"Compact catalogs require unique values for {param.name!r}"
                )

        dependency_names = {
            dependency
            for param in self._params
            for dependency, _allowed in param.active_if
        }
        self._dependency_names = tuple(
            param.name for param in self._params if param.name in dependency_names
        )
        self._dependency_indices = {
            param.name: index for index, param in enumerate(self._params)
        }
        self._count_cache: Dict[Tuple[int, Tuple[Any, ...]], int] = {}
        self._product_count = int(self._count_suffix(0, {}))

        self._default_tuned = default_action_from_parameter_space_spec(
            self.parameter_spec
        )
        default_rank = self._rank_tuned(self._default_tuned)
        self._default_in_product = default_rank is not None
        self.default_arm_index = (
            int(default_rank)
            if default_rank is not None
            else int(self._product_count)
        )
        self._length = self._product_count + int(not self._default_in_product)

    @property
    def product_count(self) -> int:
        return int(self._product_count)

    @property
    def has_appended_default(self) -> bool:
        return not bool(self._default_in_product)

    def __len__(self) -> int:
        return int(self._length)

    def _state_key(self, depth: int, state: Mapping[str, Any]) -> Tuple[Any, ...]:
        return tuple(
            state[name]
            for name in self._dependency_names
            if self._dependency_indices[name] < int(depth)
        )

    def _state_from_key(self, depth: int, key: Tuple[Any, ...]) -> Dict[str, Any]:
        names = tuple(
            name
            for name in self._dependency_names
            if self._dependency_indices[name] < int(depth)
        )
        return dict(zip(names, key))

    @staticmethod
    def _is_active(param: Any, state: Mapping[str, Any]) -> bool:
        return all(
            state.get(dependency) in allowed
            for dependency, allowed in param.active_if
        )

    def _next_state(
        self,
        state: Mapping[str, Any],
        *,
        param_name: str,
        value: Any,
    ) -> Dict[str, Any]:
        if param_name not in self._dependency_names:
            return dict(state)
        out = dict(state)
        out[param_name] = value
        return out

    def _count_suffix(self, depth: int, state: Mapping[str, Any]) -> int:
        if depth >= len(self._params):
            return 1
        key = (int(depth), self._state_key(depth, state))
        cached = self._count_cache.get(key)
        if cached is not None:
            return int(cached)

        param = self._params[depth]
        options = (
            tuple(param.values)
            if self._is_active(param, state)
            else (param.default,)
        )
        count = sum(
            self._count_suffix(
                depth + 1,
                self._next_state(
                    state, param_name=param.name, value=option
                ),
            )
            for option in options
        )
        self._count_cache[key] = int(count)
        return int(count)

    def _rank_tuned(self, tuned_action: Mapping[str, Any]) -> int | None:
        canonical = canonicalize_action_from_spec(
            tuned_action, self.parameter_spec
        )
        rank = 0
        state: Dict[str, Any] = {}
        for depth, param in enumerate(self._params):
            options = (
                tuple(param.values)
                if self._is_active(param, state)
                else (param.default,)
            )
            value = canonical[param.name]
            if param.kind in {"continuous", "integer"}:
                canonical_value = round(float(value), 12)
                option_index = next(
                    (
                        index
                        for index, option in enumerate(options)
                        if round(float(option), 12) == canonical_value
                    ),
                    None,
                )
            else:
                option_index = (
                    options.index(value) if value in options else None
                )
            if option_index is None:
                return None
            for option in options[:option_index]:
                rank += self._count_suffix(
                    depth + 1,
                    self._next_state(
                        state, param_name=param.name, value=option
                    ),
                )
            state = self._next_state(
                state, param_name=param.name, value=options[option_index]
            )
        return int(rank)

    def index_of(self, action: Mapping[str, Any]) -> int:
        canonical = canonicalize_action_from_spec(action, self.parameter_spec)
        rank = self._rank_tuned(canonical)
        if rank is not None:
            return int(rank)
        if (
            action_key_from_parameter_space_spec(canonical, self.parameter_spec)
            == action_key_from_parameter_space_spec(
                self._default_tuned, self.parameter_spec
            )
            and not self._default_in_product
        ):
            return int(self.default_arm_index)
        raise ValueError("action is not present in the compact catalog")

    def decode_parameter_arrays(
        self, arm_indices: Sequence[int] | np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """Decode arm IDs in vectorized mixed-radix batches."""

        arms = np.asarray(arm_indices, dtype=np.int64).reshape(-1)
        if arms.size == 0:
            empty_values = {
                param.name: np.empty(0, dtype=np.asarray(param.values).dtype)
                for param in self._params
            }
            empty_active = {
                param.name: np.empty(0, dtype=bool) for param in self._params
            }
            return empty_values, empty_active
        if np.any(arms < 0) or np.any(arms >= len(self)):
            raise IndexError("compact action arm index is out of range")

        appended_default = (not self._default_in_product) & (
            arms == self._product_count
        )
        product_rows = np.flatnonzero(~appended_default)
        remainder = arms.copy()
        remainder[appended_default] = 0
        values_by_name: Dict[str, np.ndarray] = {}
        active_by_name: Dict[str, np.ndarray] = {}
        groups: Dict[Tuple[Any, ...], np.ndarray] = {
            (): product_rows.astype(np.int64, copy=False)
        }

        default_state: Dict[str, Any] = {}
        for depth, param in enumerate(self._params):
            dtype = np.asarray((*param.values, param.default)).dtype
            values = np.empty(arms.size, dtype=dtype)
            active = np.zeros(arms.size, dtype=bool)
            next_group_parts: Dict[Tuple[Any, ...], list[np.ndarray]] = {}

            for state_key, rows in groups.items():
                if rows.size == 0:
                    continue
                state = self._state_from_key(depth, state_key)
                is_active = self._is_active(param, state)
                options = tuple(param.values) if is_active else (param.default,)
                widths = np.asarray(
                    [
                        self._count_suffix(
                            depth + 1,
                            self._next_state(
                                state,
                                param_name=param.name,
                                value=option,
                            ),
                        )
                        for option in options
                    ],
                    dtype=np.int64,
                )
                boundaries = np.cumsum(widths, dtype=np.int64)
                choices = np.searchsorted(
                    boundaries, remainder[rows], side="right"
                )
                previous = np.concatenate(
                    [np.zeros(1, dtype=np.int64), boundaries[:-1]]
                )
                remainder[rows] -= previous[choices]
                option_array = np.asarray(options, dtype=dtype)
                values[rows] = option_array[choices]
                active[rows] = bool(is_active)

                if param.name in self._dependency_names:
                    for choice in np.unique(choices):
                        chosen_rows = rows[choices == choice]
                        chosen_state = self._next_state(
                            state,
                            param_name=param.name,
                            value=options[int(choice)],
                        )
                        key = self._state_key(depth + 1, chosen_state)
                        next_group_parts.setdefault(key, []).append(chosen_rows)
                else:
                    key = self._state_key(depth + 1, state)
                    next_group_parts.setdefault(key, []).append(rows)

            default_active = self._is_active(param, default_state)
            if np.any(appended_default):
                values[appended_default] = param.default
                active[appended_default] = bool(default_active)
            if param.name in self._dependency_names:
                default_state[param.name] = param.default

            values_by_name[param.name] = values
            active_by_name[param.name] = active
            groups = {
                key: (
                    parts[0]
                    if len(parts) == 1
                    else np.concatenate(parts).astype(np.int64, copy=False)
                )
                for key, parts in next_group_parts.items()
            }

        if product_rows.size and np.any(remainder[product_rows] != 0):
            raise RuntimeError("mixed-radix decoding did not consume each arm index")
        return values_by_name, active_by_name

    def __getitem__(self, index: int | slice) -> Dict[str, Any] | Tuple[Dict[str, Any], ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        arm = int(index)
        if arm < 0:
            arm += len(self)
        values, _active = self.decode_parameter_arrays([arm])
        action = dict(self.fixed_params)
        for param in self._params:
            value = _python_scalar(values[param.name][0])
            if param.kind == "integer":
                value = int(round(float(value)))
            elif param.kind == "continuous":
                value = float(value)
            action[param.name] = value
        return action

    def description(self) -> Dict[str, Any]:
        return {
            "fixed_params": self.fixed_params,
            "parameters": [
                {
                    "name": param.name,
                    "kind": param.kind,
                    "values": list(param.values),
                    "default": param.default,
                    "center": param.center,
                    "scale": param.scale,
                    "active_if": [
                        [dependency, list(allowed)]
                        for dependency, allowed in param.active_if
                    ],
                }
                for param in self._params
            ],
            "product_count": int(self.product_count),
            "action_count": int(len(self)),
            "default_arm_index": int(self.default_arm_index),
        }


_SHARED_NUMERIC_TABLES: Dict[
    Tuple[Any, ...], Tuple[np.ndarray, np.ndarray]
] = {}


def _numeric_table_key(params: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(
        (
            param.name,
            param.kind,
            tuple(param.values),
            param.default,
            param.center,
            param.scale,
        )
        for param in params
    )


def _shared_numeric_tables(
    params: Sequence[Any],
) -> Tuple[np.ndarray, np.ndarray]:
    key = _numeric_table_key(params)
    cached = _SHARED_NUMERIC_TABLES.get(key)
    if cached is not None:
        return cached

    values = np.asarray(
        list(itertools.product(*(tuple(param.values) for param in params))),
        dtype=float,
    )
    normalized = np.column_stack(
        [
            (values[:, index] - float(param.center)) / float(param.scale)
            for index, param in enumerate(params)
        ]
    )
    pairwise = (
        np.column_stack(
            [
                normalized[:, i] * normalized[:, j]
                for i in range(normalized.shape[1])
                for j in range(i + 1, normalized.shape[1])
            ]
        )
        if normalized.shape[1] > 1
        else np.zeros((normalized.shape[0], 0), dtype=float)
    )
    features = np.concatenate(
        [normalized, normalized * normalized, pairwise], axis=1
    )
    values.setflags(write=False)
    features.setflags(write=False)
    cached = (values, features)
    _SHARED_NUMERIC_TABLES[key] = cached
    return cached


class FactorizedActionFeatureCache:
    """Small RAM cache for a numeric-prefix/categorical-suffix action space."""

    def __init__(
        self,
        catalog: CompactActionCatalog,
        encoder: GenericActionFeatureEncoder,
    ) -> None:
        self.catalog = catalog
        self.encoder = encoder
        params = tuple(catalog.parameter_spec.parameters)
        split = next(
            (index for index, param in enumerate(params) if param.kind == "categorical"),
            len(params),
        )
        self.numeric_params = params[:split]
        self.categorical_params = params[split:]
        if not self.numeric_params or not self.categorical_params:
            raise ValueError(
                "Factorized cache requires numeric parameters followed by categorical parameters"
            )
        if any(param.kind == "categorical" for param in self.numeric_params) or any(
            param.kind != "categorical" for param in self.categorical_params
        ):
            raise ValueError(
                "Factorized cache requires a numeric prefix and categorical suffix"
            )
        if any(param.active_if for param in self.numeric_params):
            raise ValueError(
                "Factorized cache does not support conditional numeric parameters"
            )
        if tuple(param.name for param in self.numeric_params) != tuple(
            param.name for param in encoder.numeric_params
        ) or tuple(param.name for param in self.categorical_params) != tuple(
            param.name for param in encoder.categorical_params
        ):
            raise ValueError("catalog and action-feature encoder parameter order differs")

        self.numeric_values, generic_numeric_features = _shared_numeric_tables(
            self.numeric_params
        )
        if type(encoder) is GenericActionFeatureEncoder:
            self.numeric_features = generic_numeric_features
        else:
            self.numeric_features = encoder.encode_numeric_table(
                self.numeric_values
            )
            self.numeric_features.setflags(write=False)
        numeric_names = {param.name for param in self.numeric_params}
        dependency_names = tuple(
            param.name
            for param in self.numeric_params
            if any(
                dependency == param.name
                for categorical in self.categorical_params
                for dependency, _allowed in categorical.active_if
            )
        )
        dependency_columns = tuple(
            index
            for index, param in enumerate(self.numeric_params)
            if param.name in dependency_names
        )
        if dependency_columns:
            signatures, inverse = np.unique(
                self.numeric_values[:, dependency_columns],
                axis=0,
                return_inverse=True,
            )
        else:
            signatures = np.zeros((1, 0), dtype=float)
            inverse = np.zeros(self.numeric_values.shape[0], dtype=np.int64)

        self._categorical_tables = []
        for signature in signatures:
            state = {
                name: _python_scalar(value)
                for name, value in zip(dependency_names, signature)
            }
            table = self._build_categorical_table(state)
            table.setflags(write=False)
            self._categorical_tables.append(table)
        table_dtype = (
            np.uint8
            if len(self._categorical_tables) <= np.iinfo(np.uint8).max
            else np.uint16
        )
        self._table_ids = np.asarray(inverse, dtype=table_dtype)
        counts = np.asarray(
            [len(self._categorical_tables[int(table)]) for table in self._table_ids],
            dtype=np.int64,
        )
        self._offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
        )
        if int(self._offsets[-1]) != int(catalog.product_count):
            raise RuntimeError(
                "Factorized feature layout does not match compact catalog enumeration"
            )
        self._table_ids.setflags(write=False)
        self._offsets.setflags(write=False)
        self._default_feature = encoder.encode_action(
            catalog[catalog.default_arm_index]
        )
        self._default_feature.setflags(write=False)

    @property
    def storage_bytes(self) -> int:
        return int(
            self.numeric_values.nbytes
            + self.numeric_features.nbytes
            + self._table_ids.nbytes
            + self._offsets.nbytes
            + sum(table.nbytes for table in self._categorical_tables)
        )

    def _build_categorical_table(
        self, initial_state: Mapping[str, Any]
    ) -> np.ndarray:
        rows = []

        def recurse(
            depth: int,
            state: Dict[str, Any],
            values: Dict[str, Any],
            active: Dict[str, bool],
        ) -> None:
            if depth >= len(self.categorical_params):
                row = np.zeros(self.encoder.categorical_dim, dtype=float)
                offset = 0
                for param in self.categorical_params:
                    levels = self.encoder.categorical_levels[param.name]
                    if active[param.name] and values[param.name] in levels:
                        row[offset + levels.index(values[param.name])] = 1.0
                    offset += len(levels)
                rows.append(row)
                return

            param = self.categorical_params[depth]
            is_active = CompactActionCatalog._is_active(param, state)
            options = tuple(param.values) if is_active else (param.default,)
            for option in options:
                next_state = dict(state)
                next_state[param.name] = option
                next_values = dict(values)
                next_values[param.name] = option
                next_active = dict(active)
                next_active[param.name] = bool(is_active)
                recurse(depth + 1, next_state, next_values, next_active)

        recurse(0, dict(initial_state), {}, {})
        return np.asarray(rows, dtype=float)

    def features(self, arm_indices: Sequence[int] | np.ndarray) -> np.ndarray:
        arms = np.asarray(arm_indices, dtype=np.int64).reshape(-1)
        if arms.size == 0:
            return np.empty((0, self.encoder.feature_dim), dtype=float)
        if np.any(arms < 0) or np.any(arms >= len(self.catalog)):
            raise IndexError("factorized feature arm index is out of range")

        appended_default = self.catalog.has_appended_default & (
            arms == self.catalog.product_count
        )
        product_arms = np.where(appended_default, 0, arms)
        numeric_ids = (
            np.searchsorted(self._offsets, product_arms, side="right") - 1
        )
        local_ids = product_arms - self._offsets[numeric_ids]
        numeric = self.numeric_features[numeric_ids]
        categorical = np.empty(
            (arms.size, self.encoder.categorical_dim), dtype=float
        )
        table_ids = self._table_ids[numeric_ids]
        for table_id in np.unique(table_ids):
            mask = table_ids == table_id
            categorical[mask] = self._categorical_tables[int(table_id)][
                local_ids[mask]
            ]
        z = numeric[:, : self.encoder.numeric_dim]
        mixed = (
            categorical[:, :, None] * z[:, None, :]
        ).reshape(arms.size, -1)
        encoded = np.concatenate([numeric, categorical, mixed], axis=1)
        if np.any(appended_default):
            encoded[appended_default] = self._default_feature
        return encoded

    def feature(self, arm_index: int) -> np.ndarray:
        return self.features([int(arm_index)])[0]


class AOTCandidateSchedule:
    """Disk-backed candidate IDs prepared before online timing."""

    SCHEMA_VERSION = 2
    SAMPLING_METHODS = ("uniform512", "structured512")

    def __init__(
        self,
        *,
        directory: Path,
        catalog: CompactActionCatalog,
        rounds: int,
        pool_size: int,
        seed: int,
        chunk_rounds: int = 256,
        sampling_method: str = "uniform512",
        factorized_cache: FactorizedActionFeatureCache | None = None,
        structured_anchor_size: int = 64,
        structured_global_size: int = 192,
        structured_local_size: int = 192,
        structured_sobol_size: int = 64,
    ) -> None:
        if rounds <= 0:
            raise ValueError("AOT schedule rounds must be positive")
        if pool_size <= 0 or pool_size >= len(catalog):
            raise ValueError("AOT pool size must be positive and smaller than K")
        if chunk_rounds <= 0:
            raise ValueError("AOT chunk_rounds must be positive")
        method = str(sampling_method).strip().lower().replace("-", "")
        if method not in self.SAMPLING_METHODS:
            raise ValueError(
                f"sampling_method must be one of {self.SAMPLING_METHODS}"
            )

        structured_sizes = (
            int(structured_anchor_size),
            int(structured_global_size),
            int(structured_local_size),
            int(structured_sobol_size),
        )
        if any(size < 0 for size in structured_sizes):
            raise ValueError("structured candidate quotas must be non-negative")
        if method == "structured512":
            if sum(structured_sizes) != int(pool_size):
                raise ValueError(
                    "structured candidate quotas must sum to pool_size"
                )
            if factorized_cache is None:
                raise ValueError(
                    "structured schedules require a factorized action cache"
                )
            if factorized_cache.catalog is not catalog:
                raise ValueError(
                    "structured schedule and factorized cache must share a catalog"
                )

        self.directory = Path(directory)
        self.catalog = catalog
        self.rounds = int(rounds)
        self.pool_size = int(pool_size)
        self.seed = int(seed)
        self.chunk_rounds = int(chunk_rounds)
        self.sampling_method = method
        self.factorized_cache = factorized_cache
        (
            self.structured_anchor_size,
            self.structured_global_size,
            self.structured_local_size,
            self.structured_sobol_size,
        ) = structured_sizes
        self.cursor = 0
        self._case_start_cursor = 0
        self.directory.mkdir(parents=True, exist_ok=True)

        expected = self._metadata()
        if not self._can_reuse(expected):
            self._prepare(expected)
        self._open()

    @property
    def storage_bytes(self) -> int:
        return int(self._ids.nbytes)

    def _metadata(self) -> Dict[str, Any]:
        return {
            "schema_version": int(self.SCHEMA_VERSION),
            "catalog": self.catalog.description(),
            "rounds": int(self.rounds),
            "pool_size": int(self.pool_size),
            "seed": int(self.seed),
            "sampling_method": self.sampling_method,
            "structured_quotas": {
                "anchor": int(self.structured_anchor_size),
                "balanced_global": int(self.structured_global_size),
                "coordinate_local": int(self.structured_local_size),
                "sobol": int(self.structured_sobol_size),
            },
        }

    @property
    def _metadata_path(self) -> Path:
        return self.directory / "metadata.json"

    @property
    def _ids_path(self) -> Path:
        return self.directory / "candidate_ids.npy"

    def _can_reuse(self, expected: Mapping[str, Any]) -> bool:
        paths = (self._metadata_path, self._ids_path)
        if not all(path.exists() for path in paths):
            return False
        try:
            actual = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            if actual != dict(expected):
                return False
            ids = np.load(self._ids_path, mmap_mode="r", allow_pickle=False)
            return ids.shape == (self.rounds, self.pool_size)
        except (OSError, ValueError, json.JSONDecodeError):
            return False

    def _estimated_storage_bytes(self) -> int:
        candidates = int(self.rounds) * int(self.pool_size)
        id_bytes = 4 if len(self.catalog) <= np.iinfo(np.int32).max else 8
        return candidates * id_bytes

    def _prepare(self, metadata: Mapping[str, Any]) -> None:
        required = self._estimated_storage_bytes()
        free = shutil.disk_usage(self.directory).free
        reserve = 256 * 1024 * 1024
        if required + reserve > free:
            raise OSError(
                "Insufficient disk space for AOT candidate schedule: "
                f"need about {(required + reserve) / 2**30:.2f} GiB, "
                f"have {free / 2**30:.2f} GiB"
            )

        id_dtype = np.int32 if len(self.catalog) <= np.iinfo(np.int32).max else np.int64
        temp_paths = {
            self._ids_path: self.directory / "candidate_ids.tmp.npy",
        }
        for path in temp_paths.values():
            if path.exists():
                path.unlink()

        ids = np.lib.format.open_memmap(
            temp_paths[self._ids_path],
            mode="w+",
            dtype=id_dtype,
            shape=(self.rounds, self.pool_size),
        )
        rng = np.random.default_rng(self.seed)
        sobol_engine = None
        if self.sampling_method == "structured512":
            cache = self.factorized_cache
            assert cache is not None
            sobol_engine = qmc.Sobol(
                d=len(cache.numeric_params),
                scramble=True,
                seed=int(self.seed + 104729),
            )
        progress_interval = max(self.chunk_rounds, self.rounds // 20)
        next_progress = progress_interval
        print(
            json.dumps(
                {
                    "stage": "aot_candidate_schedule",
                    "status": "preparing",
                    "directory": str(self.directory),
                    "rounds": int(self.rounds),
                    "pool_size": int(self.pool_size),
                    "estimated_MiB": required / 2**20,
                }
            ),
            flush=True,
        )
        for start in range(0, self.rounds, self.chunk_rounds):
            stop = min(self.rounds, start + self.chunk_rounds)
            row_count = stop - start
            if self.sampling_method == "structured512":
                assert sobol_engine is not None
                block = self._structured_block(
                    start_round=int(start),
                    row_count=int(row_count),
                    rng=rng,
                    sobol_engine=sobol_engine,
                )
            else:
                block = rng.integers(
                    0,
                    len(self.catalog),
                    size=(row_count, self.pool_size),
                    dtype=np.int64,
                )
                sorted_block = np.sort(block, axis=1)
                duplicate_rows = np.flatnonzero(
                    np.any(np.diff(sorted_block, axis=1) == 0, axis=1)
                )
                for row in duplicate_rows:
                    block[row] = rng.choice(
                        len(self.catalog), size=self.pool_size, replace=False
                    )

            ids[start:stop] = block.astype(id_dtype, copy=False)
            if stop >= next_progress or stop == self.rounds:
                print(
                    json.dumps(
                        {
                            "stage": "aot_candidate_schedule",
                            "status": "preparing",
                            "directory": str(self.directory),
                            "completed_rounds": int(stop),
                            "total_rounds": int(self.rounds),
                        }
                    ),
                    flush=True,
                )
                next_progress += progress_interval

        ids.flush()
        del ids
        for final_path, temp_path in temp_paths.items():
            os.replace(temp_path, final_path)
        self._metadata_path.write_text(
            json.dumps(dict(metadata), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @staticmethod
    def _split_branch_quota(
        total: int,
        *,
        inactive_available: bool,
        active_available: bool,
    ) -> Tuple[int, int]:
        """Split one fixed budget evenly across aggressive-state branches."""

        if inactive_available and active_available:
            inactive = int(total) // 2
            return inactive, int(total) - inactive
        if inactive_available:
            return int(total), 0
        if active_available:
            return 0, int(total)
        raise RuntimeError("structured schedule has no numeric action rows")

    def _numeric_branch_layout(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...], int | None]:
        cache = self.factorized_cache
        if cache is None:
            raise RuntimeError("structured schedule lacks a factorized cache")
        numeric_values = np.asarray(cache.numeric_values)
        radices = tuple(len(param.values) for param in cache.numeric_params)
        agg_column = next(
            (
                index
                for index, param in enumerate(cache.numeric_params)
                if param.name == "agg_num_levels"
            ),
            None,
        )
        if agg_column is None:
            inactive = np.arange(numeric_values.shape[0], dtype=np.int64)
            active = np.empty(0, dtype=np.int64)
        else:
            default = float(cache.numeric_params[int(agg_column)].default)
            inactive_mask = np.isclose(
                numeric_values[:, int(agg_column)],
                default,
                rtol=0.0,
                atol=1.0e-12,
            )
            inactive = np.flatnonzero(inactive_mask).astype(np.int64)
            active = np.flatnonzero(~inactive_mask).astype(np.int64)
        return inactive, active, radices, agg_column

    def _categorical_cycle_arms(
        self,
        numeric_ids: np.ndarray,
        *,
        cycle_offset: int,
    ) -> np.ndarray:
        cache = self.factorized_cache
        if cache is None:
            raise RuntimeError("structured schedule lacks a factorized cache")
        numeric = np.asarray(numeric_ids, dtype=np.int64)
        counts = cache._offsets[numeric + 1] - cache._offsets[numeric]
        positions = np.arange(numeric.size, dtype=np.int64).reshape(numeric.shape)
        local = (positions + int(cycle_offset)) % counts
        return cache._offsets[numeric] + local

    def _balanced_global_segment(
        self,
        *,
        start_round: int,
        row_count: int,
        size: int,
        rng: np.random.Generator,
        inactive_rows: np.ndarray,
        active_rows: np.ndarray,
    ) -> np.ndarray:
        inactive_count, active_count = self._split_branch_quota(
            int(size),
            inactive_available=bool(inactive_rows.size),
            active_available=bool(active_rows.size),
        )
        parts = []
        for branch_index, (pool, count) in enumerate(
            ((inactive_rows, inactive_count), (active_rows, active_count))
        ):
            if count <= 0:
                continue
            numeric_ids = rng.choice(
                pool,
                size=(int(row_count), int(count)),
                replace=True,
            ).astype(np.int64, copy=False)
            parts.append(
                self._categorical_cycle_arms(
                    numeric_ids,
                    cycle_offset=(
                        (int(start_round) * int(count))
                        + branch_index * 15485863
                    ),
                )
            )
        return (
            np.concatenate(parts, axis=1)
            if parts
            else np.empty((int(row_count), 0), dtype=np.int64)
        )

    def _sobol_numeric_ids(
        self,
        *,
        row_count: int,
        size: int,
        sobol_engine: qmc.Sobol,
        radices: Tuple[int, ...],
        agg_column: int | None,
        inactive_count: int,
        active_count: int,
    ) -> np.ndarray:
        point_count = int(row_count) * int(size)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            points = sobol_engine.random(point_count)
        codes = np.floor(
            points * np.asarray(radices, dtype=float)[None, :]
        ).astype(np.int64)
        np.minimum(
            codes,
            np.asarray(radices, dtype=np.int64)[None, :] - 1,
            out=codes,
        )
        codes = codes.reshape(int(row_count), int(size), len(radices))
        if agg_column is not None:
            cache = self.factorized_cache
            assert cache is not None
            param = cache.numeric_params[int(agg_column)]
            default_index = tuple(param.values).index(param.default)
            active_indices = np.asarray(
                [
                    index
                    for index in range(len(param.values))
                    if index != int(default_index)
                ],
                dtype=np.int64,
            )
            if inactive_count:
                codes[:, :inactive_count, int(agg_column)] = int(default_index)
            if active_count:
                raw = codes[:, inactive_count:, int(agg_column)]
                codes[:, inactive_count:, int(agg_column)] = active_indices[
                    raw % active_indices.size
                ]
        return np.ravel_multi_index(
            np.moveaxis(codes, -1, 0).reshape(len(radices), -1),
            dims=radices,
        ).reshape(int(row_count), int(size))

    def _sobol_segment(
        self,
        *,
        start_round: int,
        row_count: int,
        size: int,
        sobol_engine: qmc.Sobol,
        inactive_rows: np.ndarray,
        active_rows: np.ndarray,
        radices: Tuple[int, ...],
        agg_column: int | None,
    ) -> np.ndarray:
        inactive_count, active_count = self._split_branch_quota(
            int(size),
            inactive_available=bool(inactive_rows.size),
            active_available=bool(active_rows.size),
        )
        numeric_ids = self._sobol_numeric_ids(
            row_count=int(row_count),
            size=int(size),
            sobol_engine=sobol_engine,
            radices=radices,
            agg_column=agg_column,
            inactive_count=inactive_count,
            active_count=active_count,
        )
        return self._categorical_cycle_arms(
            numeric_ids,
            cycle_offset=int(start_round) * int(size) + 32452843,
        )

    def _structured_block(
        self,
        *,
        start_round: int,
        row_count: int,
        rng: np.random.Generator,
        sobol_engine: qmc.Sobol,
    ) -> np.ndarray:
        """Build balanced/Sobol cores plus unique AOT filler IDs."""

        inactive, active, radices, agg_column = self._numeric_branch_layout()
        balanced = self._balanced_global_segment(
            start_round=int(start_round),
            row_count=int(row_count),
            size=int(self.structured_global_size),
            rng=rng,
            inactive_rows=inactive,
            active_rows=active,
        )
        sobol = self._sobol_segment(
            start_round=int(start_round),
            row_count=int(row_count),
            size=int(self.structured_sobol_size),
            sobol_engine=sobol_engine,
            inactive_rows=inactive,
            active_rows=active,
            radices=radices,
            agg_column=agg_column,
        )
        core = np.concatenate([balanced, sobol], axis=1)
        block = np.empty(
            (int(row_count), int(self.pool_size)), dtype=np.int64
        )
        for row_index in range(int(row_count)):
            seen: set[int] = set()
            write = 0
            for raw_arm in core[row_index]:
                arm = int(raw_arm)
                while arm in seen:
                    arm = int(rng.integers(0, len(self.catalog)))
                block[row_index, write] = arm
                seen.add(arm)
                write += 1
            while write < int(self.pool_size):
                arm = int(rng.integers(0, len(self.catalog)))
                if arm in seen:
                    continue
                block[row_index, write] = arm
                seen.add(arm)
                write += 1
        return block

    def _open(self) -> None:
        self._ids = np.load(self._ids_path, mmap_mode="r", allow_pickle=False)

    def next_arms(self) -> np.ndarray:
        if self.cursor >= self.rounds:
            raise RuntimeError(
                "AOT candidate schedule exhausted; increase scheduled rounds"
            )
        row = int(self.cursor)
        self.cursor += 1
        return np.asarray(self._ids[row], dtype=np.int64)

    def finish_case(self, *, max_selections: int) -> None:
        """Advance to the next fixed per-instance schedule block.

        A setup failure can consume extra candidate rows.  Reserving the same
        number of rows for every instance keeps the primary row paired across
        learners even when their recovery paths differ.
        """

        width = int(max_selections)
        if width <= 0:
            raise ValueError("max_selections must be positive")
        target = int(self._case_start_cursor) + width
        if self.cursor > target:
            raise RuntimeError(
                "AOT schedule consumed more selections than reserved for one case"
            )
        if target > self.rounds:
            raise RuntimeError(
                "AOT candidate schedule exhausted while finishing a case"
            )
        self.cursor = target
        self._case_start_cursor = target

    def set_cursor(self, cursor: int) -> None:
        value = int(cursor)
        if not (0 <= value <= self.rounds):
            raise ValueError("AOT schedule cursor is out of range")
        self.cursor = value
        self._case_start_cursor = value
