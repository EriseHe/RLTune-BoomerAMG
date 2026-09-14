from __future__ import annotations

import math
from typing import Any, Dict, Sequence

import numpy as np

from problems.amg import (
    COMPACT_DIFFUSION_CONTEXT_INDICES,
    DIFFUSION_ADVECTION_CONTEXT_DIM,
    DIFFUSION_ADVECTION_CONTEXT_FIELDS,
    DIFFUSION_ADVECTION_PHYSICS_COORDINATE_FIELDS,
    build_context_diffusion_advection_from_matrix_kwargs,
    build_diffusion_advection_physics_context,
    compact_diffusion_context,
    normalize_diffusion_advection_context as normalize_problem_context,
)


CANONICAL_PROBLEM_CONTEXT = "canonical"
LEGACY_DIFFUSION_ONLY_CONTEXT = "legacy"
PHYSICS_LINEAR_PROBLEM_CONTEXT = "physics_linear"
PROBLEM_CONTEXT_MODES = (
    CANONICAL_PROBLEM_CONTEXT,
    LEGACY_DIFFUSION_ONLY_CONTEXT,
    PHYSICS_LINEAR_PROBLEM_CONTEXT,
    *COMPACT_DIFFUSION_CONTEXT_INDICES,
)
STATE_ENCODING_VERSIONS = ("legacy_v1", "space_aware_v2")


class SolveStateEncoder:
    """Small bounded feature map for a stream of short AMG solve episodes."""

    _CYCLE_BOUNDS = np.asarray([1, 2, 3, 5, 8, 12], dtype=int)

    def __init__(
        self,
        *,
        tol: float,
        max_cycles: int,
        c_max: float,
        time_scale_sec: float = 2.0e-3,
        mode: str = "full",
        setup_obs_encoder: Any | None = None,
        problem_context_mode: str = CANONICAL_PROBLEM_CONTEXT,
        encoding_version: str = "legacy_v1",
        weight_bounds: tuple[float, float] | None = None,
    ) -> None:
        self.tol = float(tol)
        self.max_cycles = int(max_cycles)
        self.c_max = float(c_max)
        self.time_scale_sec = float(time_scale_sec)
        self.mode = str(mode).strip().lower()
        self.setup_obs_encoder = setup_obs_encoder
        self.encoding_version = str(encoding_version)
        if self.encoding_version not in STATE_ENCODING_VERSIONS:
            raise ValueError(f"encoding_version must be one of {STATE_ENCODING_VERSIONS}")
        self.weight_bounds = weight_bounds
        if self.encoding_version == "space_aware_v2":
            if weight_bounds is None or len(weight_bounds) != 2:
                raise ValueError("space_aware_v2 requires actual action weight bounds")
            low, high = map(float, weight_bounds)
            if not all(math.isfinite(v) for v in (low, high)) or high < low:
                raise ValueError("action weight bounds must be finite and ordered")
            self.weight_bounds = (low, high)
            self._weight_center = (low + high) / 2.0
            self._weight_scale = (high - low) / 2.0 if high > low else 1.0
        self.problem_context_mode = (
            str(problem_context_mode).strip().lower()
        )
        if self.problem_context_mode not in PROBLEM_CONTEXT_MODES:
            raise ValueError(
                "problem_context_mode must be one of "
                f"{PROBLEM_CONTEXT_MODES}"
            )
        self.cycle_bins = int(self._CYCLE_BOUNDS.size + 1)
        if self.problem_context_mode == CANONICAL_PROBLEM_CONTEXT:
            self.problem_context_fields = DIFFUSION_ADVECTION_CONTEXT_FIELDS
            self.problem_feature_dim = DIFFUSION_ADVECTION_CONTEXT_DIM - 1
        elif self.problem_context_mode in COMPACT_DIFFUSION_CONTEXT_INDICES:
            self.problem_context_fields = tuple(
                DIFFUSION_ADVECTION_CONTEXT_FIELDS[index]
                for index in COMPACT_DIFFUSION_CONTEXT_INDICES[self.problem_context_mode]
            )
            self.problem_feature_dim = len(self.problem_context_fields) - 1
        elif self.problem_context_mode == PHYSICS_LINEAR_PROBLEM_CONTEXT:
            self.problem_context_fields = (
                "bias",
                *DIFFUSION_ADVECTION_PHYSICS_COORDINATE_FIELDS,
            )
            self.problem_feature_dim = len(
                DIFFUSION_ADVECTION_PHYSICS_COORDINATE_FIELDS
            )
        else:
            self.problem_context_fields = (
                "c_x",
                "c_y",
                "c_z",
                "diffusion_spread",
            )
            self.problem_feature_dim = 4
        self._problem_offset = 7
        self._cycle_offset = (
            self._problem_offset + self.problem_feature_dim
        )
        self._setup_offset = self._cycle_offset + 2 * self.cycle_bins
        self._cached_problem_key: tuple[float, ...] | None = None
        self._cached_problem_features = np.zeros(
            self.problem_feature_dim,
            dtype=float,
        )
        self._cached_initial_residual: float | None = None
        self._cached_initial_gap = 1.0
        self._cached_setup_key: tuple[Any, ...] | None = None
        self._cached_setup_features = np.empty(0, dtype=float)
        if self.mode in {"full", "setup_full"}:
            setup_dim = 0
            if self.mode == "setup_full":
                if self.setup_obs_encoder is None:
                    raise ValueError("setup_full mode requires a setup observation encoder")
                setup_dim = len(self.setup_obs_encoder.observed_keys)
            self.feature_dim = self._setup_offset + setup_dim
            self._cycle_features = None
        elif self.mode == "cycle_tabular":
            self.feature_dim = int(self.max_cycles + 1)
            self._cycle_features = np.eye(self.feature_dim, dtype=float)
        else:
            raise ValueError(f"Unknown solve state mode: {self.mode}")

    def encode(
        self,
        *,
        mkw: Dict[str, Any],
        initial_residual: float,
        residual: float,
        previous_residual: float,
        cycle: int,
        last_weight: float,
        last_cycle_time: float,
        setup_params: Dict[str, Any] | None = None,
        problem_context: Sequence[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        if self.mode == "cycle_tabular":
            assert self._cycle_features is not None
            return self._cycle_features[int(np.clip(cycle, 0, self.max_cycles))]

        eps = 1.0e-30
        initial_residual_value = float(initial_residual)
        if self._cached_initial_residual != initial_residual_value:
            self._cached_initial_residual = initial_residual_value
            self._cached_initial_gap = max(
                math.log(max(initial_residual_value, self.tol) / self.tol),
                eps,
            )
        initial_gap = self._cached_initial_gap
        gap = math.log(max(float(residual), self.tol) / self.tol)
        gap_fraction = min(max(gap / initial_gap, 0.0), 1.5)
        log_ratio = math.log((float(residual) + eps) / (float(previous_residual) + eps))
        log_ratio = min(max(log_ratio / 5.0, -1.0), 1.0)
        cycle_fraction = min(
            max(float(cycle) / max(1.0, float(self.max_cycles)), 0.0),
            1.0,
        )
        cycle_time = min(
            max(float(last_cycle_time) / max(self.time_scale_sec, eps), 0.0),
            5.0,
        ) / 5.0
        if self.encoding_version == "legacy_v1":
            weight = min(max((float(last_weight) - 1.4) / 0.4, -1.0), 1.5)
        else:
            weight = min(max(
                (float(last_weight) - self._weight_center) / self._weight_scale,
                -1.0,
            ), 1.0)

        if self.problem_context_mode in {
            CANONICAL_PROBLEM_CONTEXT,
            PHYSICS_LINEAR_PROBLEM_CONTEXT,
            *COMPACT_DIFFUSION_CONTEXT_INDICES,
        }:
            canonical_context = (
                self._context_from_matrix_kwargs(mkw)
                if problem_context is None
                else normalize_problem_context(problem_context)
            )
            resolved_context = (
                build_diffusion_advection_physics_context(
                    canonical_context=canonical_context,
                    matrix_kwargs=mkw,
                )
                if self.problem_context_mode
                == PHYSICS_LINEAR_PROBLEM_CONTEXT
                else canonical_context
            )
            if self.problem_context_mode in COMPACT_DIFFUSION_CONTEXT_INDICES:
                resolved_context = compact_diffusion_context(
                    canonical_context, mode=self.problem_context_mode
                )
            problem_key = tuple(float(value) for value in resolved_context)
            if self._cached_problem_key != problem_key:
                self._cached_problem_key = problem_key
                self._cached_problem_features[:] = resolved_context[1:]
        else:
            problem_key = (
                float(mkw["k"]),
                float(mkw["c"]),
                float(mkw["a0"]),
            )
            if self._cached_problem_key != problem_key:
                self._cached_problem_key = problem_key
                c_denom = max(math.log(max(1.0, self.c_max)), eps)
                coeffs = np.asarray(
                    [
                        math.log(max(value, eps)) / c_denom
                        for value in problem_key
                    ],
                    dtype=float,
                )
                coeffs = np.clip(coeffs, -1.0, 1.5)
                self._cached_problem_features[:3] = coeffs
                self._cached_problem_features[3] = float(
                    np.max(coeffs) - np.min(coeffs)
                )

        cycle_bin = int(np.searchsorted(self._CYCLE_BOUNDS, int(cycle), side="right"))
        features = np.zeros(self.feature_dim, dtype=float)
        features[:7] = (
            1.0,
            cycle_fraction,
            gap_fraction,
            gap_fraction * gap_fraction,
            log_ratio,
            cycle_time,
            weight,
        )
        features[
            self._problem_offset : self._cycle_offset
        ] = self._cached_problem_features
        features[self._cycle_offset + cycle_bin] = 1.0
        features[
            self._cycle_offset + self.cycle_bins + cycle_bin
        ] = gap_fraction
        if self.mode == "setup_full":
            assert self.setup_obs_encoder is not None
            params = {} if setup_params is None else setup_params
            setup_defaults = getattr(self.setup_obs_encoder, "defaults", {})
            setup_key = tuple(
                params.get(key, setup_defaults.get(key, 0.0))
                for key in self.setup_obs_encoder.observed_keys
            )
            if self._cached_setup_key != setup_key:
                setup_features = np.asarray(
                    self.setup_obs_encoder.encode(params),
                    dtype=float,
                )
                self._cached_setup_features = np.clip(setup_features, -2.0, 2.0)
                self._cached_setup_key = setup_key
            features[self._setup_offset :] = self._cached_setup_features
        if features.size != self.feature_dim:
            raise RuntimeError(f"Expected {self.feature_dim} state features, got {features.size}")
        return features

    def constant_value_parameters(self, value: float) -> np.ndarray:
        parameters = np.zeros(self.feature_dim, dtype=float)
        if self.mode == "cycle_tabular":
            parameters.fill(float(value))
        else:
            parameters[0] = float(value)
        return parameters

    def _context_from_matrix_kwargs(
        self,
        mkw: Dict[str, Any],
    ) -> np.ndarray:
        """Compatibility path for callers that predate explicit context wiring."""

        nx = int(mkw.get("nx", 1))
        ny = int(mkw.get("ny", 1))
        nz = int(mkw.get("nz", 1))
        advection = tuple(
            float(mkw.get(key, 0.0)) for key in ("a1", "a2", "a3")
        )
        return build_context_diffusion_advection_from_matrix_kwargs(
            mkw,
            grid_norm_div=float(max(nx, ny, nz, 1)),
            c_norm_div=float(self.c_max),
            a_norm_div=float(
                max(self.c_max, *(abs(value) for value in advection), 1.0)
            ),
        )


__all__ = [
    "CANONICAL_PROBLEM_CONTEXT",
    "LEGACY_DIFFUSION_ONLY_CONTEXT",
    "PHYSICS_LINEAR_PROBLEM_CONTEXT",
    "PROBLEM_CONTEXT_MODES",
    "SolveStateEncoder",
    "normalize_problem_context",
]
