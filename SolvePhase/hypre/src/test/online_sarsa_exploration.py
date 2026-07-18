from __future__ import annotations

import json
import math
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from online_td_lambda import ExpectedSarsaLambda


@dataclass(frozen=True)
class SarsaBehaviorSpec:
    action_mode: str
    behavior_mode: str
    initial_weight: float = 1.0
    force_default_first_action: bool = True
    residual_min: float = 1.0
    residual_max: float = 2.0
    uncertainty_beta: float = 1.0
    uncertainty_ridge: float = 1.0
    uncertainty_td_floor_sec: float = 1.0e-3

    @property
    def name(self) -> str:
        return f"{self.action_mode}_{self.behavior_mode}"


class BehaviorPolicySarsaController(ExpectedSarsaLambda):
    """True-online SARSA with configurable action mapping and exploration."""

    def __init__(self, *, behavior_spec: SarsaBehaviorSpec, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if behavior_spec.action_mode not in {"absolute", "residual"}:
            raise ValueError("action_mode must be 'absolute' or 'residual'")
        if behavior_spec.behavior_mode not in {
            "uniform",
            "local",
            "uncertainty_lcb",
        }:
            raise ValueError(
                "behavior_mode must be 'uniform', 'local', or 'uncertainty_lcb'"
            )
        if behavior_spec.uncertainty_ridge <= 0.0:
            raise ValueError("uncertainty_ridge must be positive")
        if behavior_spec.uncertainty_beta < 0.0:
            raise ValueError("uncertainty_beta must be non-negative")
        self.behavior_spec = behavior_spec
        inverse_scale = 1.0 / float(behavior_spec.uncertainty_ridge)
        self.coverage_inverse = np.repeat(
            (inverse_scale * np.eye(self.feature_dim, dtype=float))[None, :, :],
            self.weights.size,
            axis=0,
        )
        self.coverage_counts = np.zeros(self.weights.size, dtype=np.int64)
        self.td_error_sum_squares = 0.0
        self.td_error_count = 0
        self._default_first_action_pending = bool(
            behavior_spec.force_default_first_action
        )
        self._initial_environment_weight = float(behavior_spec.initial_weight)
        self.forced_initial_action_count = 0

    def environment_weight(self, action_index: int, last_weight: float) -> float:
        if self.behavior_spec.action_mode == "absolute":
            return super().environment_weight(action_index, last_weight)
        return float(
            np.clip(
                float(last_weight) + float(self.weights[int(action_index)]),
                float(self.behavior_spec.residual_min),
                float(self.behavior_spec.residual_max),
            )
        )

    @property
    def uncertainty_td_scale(self) -> float:
        empirical = (
            math.sqrt(self.td_error_sum_squares / float(self.td_error_count))
            if self.td_error_count > 0
            else 0.0
        )
        return float(
            max(empirical, self.behavior_spec.uncertainty_td_floor_sec)
        )

    def _coverage(self, features: np.ndarray) -> np.ndarray:
        feature_values = np.asarray(features, dtype=float)
        quadratic = np.einsum(
            "i,aij,j->a",
            feature_values,
            self.coverage_inverse,
            feature_values,
            optimize=True,
        )
        return np.sqrt(np.maximum(quadratic, 0.0))

    def _record_selected_features(
        self,
        action_index: int,
        features: np.ndarray,
    ) -> None:
        action = int(action_index)
        feature_values = np.asarray(features, dtype=float)
        inverse = self.coverage_inverse[action]
        projected = inverse @ feature_values
        denominator = 1.0 + float(feature_values @ projected)
        inverse -= np.outer(projected, projected) / max(denominator, 1.0e-15)
        self.coverage_counts[action] += 1

    def _local_indices(
        self,
        greedy_index: int,
        allowed_indices: np.ndarray,
    ) -> np.ndarray:
        allowed = set(int(index) for index in np.asarray(allowed_indices, dtype=int))
        neighbors = [
            index
            for index in (int(greedy_index) - 1, int(greedy_index) + 1)
            if index in allowed
        ]
        if not neighbors:
            return np.asarray([int(greedy_index)], dtype=int)
        return np.asarray(neighbors, dtype=int)

    def select_action(
        self,
        features: np.ndarray,
        *,
        explore: bool,
        epsilon: float | None = None,
        cycle: int | None = None,
        include_metadata: bool = True,
    ) -> tuple[int, float, Dict[str, Any]]:
        feature_values = np.asarray(features, dtype=float)
        q_values = self.q_values(feature_values)
        allowed = self.allowed_action_indices(cycle)
        greedy_index = self._greedy_index(q_values, allowed_indices=allowed)
        if explore and self._default_first_action_pending:
            forced_value = (
                float(self.initial_environment_weight)
                if self.behavior_spec.action_mode == "absolute"
                else 0.0
            )
            action_index = int(np.argmin(np.abs(self.weights - forced_value)))
            if not np.isclose(
                self.weights[action_index],
                forced_value,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "The action space cannot represent the prepared solver default"
                )
            self._default_first_action_pending = False
            self.forced_initial_action_count += 1
            if self.behavior_spec.behavior_mode == "uncertainty_lcb":
                self._record_selected_features(action_index, feature_values)
            metadata: Dict[str, Any] = {}
            if include_metadata:
                metadata = {
                    "epsilon": 0.0,
                    "greedy_index": int(greedy_index),
                    "explored": False,
                    "allowed_indices": allowed.tolist(),
                    "behavior_mode": self.behavior_spec.behavior_mode,
                    "action_mode": self.behavior_spec.action_mode,
                    "uncertainty": np.zeros(self.weights.size, dtype=float).tolist(),
                    "selection_scores": q_values.tolist(),
                    "uncertainty_td_scale": float(self.uncertainty_td_scale),
                    "forced_default_first_action": True,
                }
            return action_index, float(self.weights[action_index]), metadata

        effective_epsilon = (
            (self.epsilon if epsilon is None else float(epsilon)) if explore else 0.0
        )
        if self.behavior_spec.behavior_mode == "uncertainty_lcb":
            effective_epsilon = 0.0
        uncertainty = np.zeros(self.weights.size, dtype=float)
        scores = q_values.copy()
        explored = False

        if explore and self.behavior_spec.behavior_mode == "uncertainty_lcb":
            uncertainty = self.uncertainty_td_scale * self._coverage(feature_values)
            scores = (
                q_values
                - float(self.behavior_spec.uncertainty_beta) * uncertainty
            )
            allowed_scores = scores[allowed]
            best_score = float(np.min(allowed_scores))
            candidates = allowed[
                np.isclose(allowed_scores, best_score, atol=1.0e-12, rtol=0.0)
            ]
            action_index = int(self.rng.choice(candidates))
            explored = bool(action_index != greedy_index)
            self._record_selected_features(action_index, feature_values)
        else:
            epsilon_event = bool(
                effective_epsilon > 0.0
                and float(self.rng.random()) < effective_epsilon
            )
            if epsilon_event:
                candidates = allowed
                if self.behavior_spec.behavior_mode == "local":
                    candidates = self._local_indices(greedy_index, allowed)
                action_index = int(self.rng.choice(candidates))
                explored = bool(action_index != greedy_index)
            else:
                action_index = int(greedy_index)

        metadata: Dict[str, Any] = {}
        if include_metadata:
            metadata = {
                "epsilon": float(effective_epsilon),
                "greedy_index": int(greedy_index),
                "explored": bool(explored),
                "allowed_indices": allowed.tolist(),
                "behavior_mode": self.behavior_spec.behavior_mode,
                "action_mode": self.behavior_spec.action_mode,
                "uncertainty": uncertainty.tolist(),
                "selection_scores": scores.tolist(),
                "uncertainty_td_scale": float(self.uncertainty_td_scale),
                "forced_default_first_action": False,
            }
        return action_index, float(self.weights[action_index]), metadata

    def update(self, **kwargs: Any) -> float:
        td_error = float(super().update(**kwargs))
        if self.behavior_spec.behavior_mode == "uncertainty_lcb":
            self.td_error_sum_squares += td_error * td_error
            self.td_error_count += 1
        return td_error

    def behavior_summary(self) -> Dict[str, Any]:
        return {
            "spec": self.behavior_spec.__dict__,
            "coverage_counts": self.coverage_counts.tolist(),
            "uncertainty_td_scale_sec": float(self.uncertainty_td_scale),
            "td_error_count": int(self.td_error_count),
            "forced_initial_action_count": int(self.forced_initial_action_count),
            "default_first_action_pending": bool(
                self._default_first_action_pending
            ),
        }

    def _extra_checkpoint_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "behavior_spec": np.asarray(json.dumps(asdict(self.behavior_spec))),
            "coverage_inverse": self.coverage_inverse,
            "coverage_counts": self.coverage_counts,
            "td_error_sum_squares": np.asarray(
                self.td_error_sum_squares,
                dtype=float,
            ),
            "td_error_count": np.asarray(self.td_error_count, dtype=np.int64),
        }

    def _load_extra_checkpoint_arrays(self, payload: Any) -> None:
        if "behavior_spec" not in payload:
            return
        saved_spec = json.loads(str(payload["behavior_spec"].item()))
        if saved_spec != asdict(self.behavior_spec):
            raise ValueError("Checkpoint SARSA behavior specification does not match")
        coverage_inverse = np.asarray(payload["coverage_inverse"], dtype=float)
        if coverage_inverse.shape != self.coverage_inverse.shape:
            raise ValueError("Checkpoint uncertainty coverage shape does not match")
        coverage_counts = np.asarray(payload["coverage_counts"], dtype=np.int64)
        if coverage_counts.shape != self.coverage_counts.shape:
            raise ValueError("Checkpoint uncertainty count shape does not match")
        self.coverage_inverse[:] = coverage_inverse
        self.coverage_counts[:] = coverage_counts
        self.td_error_sum_squares = float(
            payload["td_error_sum_squares"].item()
        )
        self.td_error_count = int(payload["td_error_count"].item())
