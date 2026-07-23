from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np


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
    ) -> None:
        self.tol = float(tol)
        self.max_cycles = int(max_cycles)
        self.c_max = float(c_max)
        self.time_scale_sec = float(time_scale_sec)
        self.mode = str(mode).strip().lower()
        self.setup_obs_encoder = setup_obs_encoder
        self.cycle_bins = int(self._CYCLE_BOUNDS.size + 1)
        self._cached_problem_key: tuple[float, float, float] | None = None
        self._cached_problem_features = np.zeros(4, dtype=float)
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
            self.feature_dim = 11 + 2 * self.cycle_bins + setup_dim
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
        weight = min(max((float(last_weight) - 1.4) / 0.4, -1.0), 1.5)

        problem_key = (float(mkw["k"]), float(mkw["c"]), float(mkw["a0"]))
        if self._cached_problem_key != problem_key:
            self._cached_problem_key = problem_key
            c_denom = max(math.log(max(1.0, self.c_max)), eps)
            coeffs = np.asarray(
                [math.log(max(value, eps)) / c_denom for value in problem_key],
                dtype=float,
            )
            coeffs = np.clip(coeffs, -1.0, 1.5)
            self._cached_problem_features[:3] = coeffs
            self._cached_problem_features[3] = float(np.max(coeffs) - np.min(coeffs))

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
        features[7:11] = self._cached_problem_features
        features[11 + cycle_bin] = 1.0
        features[11 + self.cycle_bins + cycle_bin] = gap_fraction
        if self.mode == "setup_full":
            assert self.setup_obs_encoder is not None
            params = {} if setup_params is None else setup_params
            setup_defaults = getattr(self.setup_obs_encoder, "defaults", {})
            setup_key = tuple(
                params.get(key, setup_defaults.get(key, 0.0))
                for key in self.setup_obs_encoder.observed_keys
            )
            if self._cached_setup_key != setup_key:
                self._cached_setup_key = setup_key
                setup_features = np.asarray(
                    self.setup_obs_encoder.encode(params),
                    dtype=float,
                )
                self._cached_setup_features = np.clip(setup_features, -2.0, 2.0)
            features[11 + 2 * self.cycle_bins :] = self._cached_setup_features
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


__all__ = ["SolveStateEncoder"]
