from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common import (
    _SharedActionLcbController,
    _json_dataclass,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig


@dataclass(frozen=True)
class StructuredModelBasedSpec:
    """Shared RLS controls for physical cycle-cost/progress prediction."""

    ridge: float = 1.0
    minimum_samples: int = 32
    scale_window: int = 2048
    cost_floor_fraction: float = 0.1
    progress_floor_fraction: float = 0.1
    numerical_floor: float = 1.0e-12

_STRUCTURED_MODEL_BASED_CHECKPOINT_VERSION = 1


class StructuredModelBasedController(_SharedActionLcbController):
    """Receding-horizon controller for native time per residual progress."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: StructuredModelBasedSpec,
        seed: int,
    ) -> None:
        if float(spec.ridge) <= 0.0:
            raise ValueError("model-based ridge must be positive")
        if int(spec.minimum_samples) <= 0:
            raise ValueError("model-based minimum samples must be positive")
        if int(spec.scale_window) < int(spec.minimum_samples):
            raise ValueError("model-based scale window is shorter than warmup")
        if not 0.0 < float(spec.cost_floor_fraction) <= 1.0:
            raise ValueError("model-based cost floor fraction must be in (0, 1]")
        if not 0.0 < float(spec.progress_floor_fraction) <= 1.0:
            raise ValueError("model-based progress floor fraction must be in (0, 1]")
        if float(spec.numerical_floor) <= 0.0:
            raise ValueError("model-based numerical floor must be positive")
        super().__init__(feature_dim=feature_dim, config=config, seed=seed)
        self.spec = spec
        ridge = float(spec.ridge)
        self.design_matrix = ridge * np.eye(self.joint_dim, dtype=float)
        self.design_inverse = np.eye(self.joint_dim, dtype=float) / ridge
        self.cost_b = np.zeros(self.joint_dim, dtype=float)
        self.progress_b = np.zeros(self.joint_dim, dtype=float)
        self.cost_theta = np.zeros(self.joint_dim, dtype=float)
        self.progress_theta = np.zeros(self.joint_dim, dtype=float)
        self.native_costs: deque[float] = deque(maxlen=int(spec.scale_window))
        self.signed_progress: deque[float] = deque(maxlen=int(spec.scale_window))
        self.sample_count = 0
        self.skipped_transition_count = 0
        self.inverse_rebuild_count = 0
        self._last_predicted_cost = np.zeros(self.weights.size, dtype=float)
        self._last_predicted_progress = np.zeros(self.weights.size, dtype=float)

    @property
    def requires_next_action(self) -> bool:
        return False

    @property
    def cost_floor(self) -> float:
        if not self.native_costs:
            return float(self.spec.numerical_floor)
        return max(
            float(self.spec.cost_floor_fraction)
            * float(np.median(np.asarray(self.native_costs, dtype=float))),
            float(self.spec.numerical_floor),
        )

    @property
    def progress_floor(self) -> float:
        if not self.signed_progress:
            return float(self.spec.numerical_floor)
        scale = float(
            np.median(np.abs(np.asarray(self.signed_progress, dtype=float)))
        )
        return max(
            float(self.spec.progress_floor_fraction) * scale,
            float(self.spec.numerical_floor),
        )

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del cycle
        joint = self.state_action_features(features)
        predicted_cost = joint @ self.cost_theta
        predicted_progress = joint @ self.progress_theta
        effective_cost = np.maximum(predicted_cost, float(self.cost_floor))
        effective_progress = np.maximum(
            predicted_progress, float(self.progress_floor)
        )
        ratios = effective_cost / effective_progress
        scores = ratios.copy()
        if self.sample_count < int(self.spec.minimum_samples):
            scores.fill(float(np.finfo(float).max))
            scores[self.anchor_index] = 0.0
        self._last_predicted_cost = predicted_cost
        self._last_predicted_progress = predicted_progress
        return ratios, np.zeros_like(ratios), scores

    def select_action(
        self,
        features: np.ndarray,
        *,
        explore: bool,
        epsilon: float | None = None,
        cycle: int | None = None,
        include_metadata: bool = True,
    ) -> tuple[int, float, Dict[str, Any]]:
        action, weight, metadata = super().select_action(
            features,
            explore=explore,
            epsilon=epsilon,
            cycle=cycle,
            include_metadata=include_metadata,
        )
        if include_metadata:
            metadata.update(
                {
                    "predicted_native_cycle_costs": self._last_predicted_cost.tolist(),
                    "predicted_signed_log_progress": self._last_predicted_progress.tolist(),
                    "cost_floor": float(self.cost_floor),
                    "progress_floor": float(self.progress_floor),
                    "model_warmup": bool(
                        self.sample_count < int(self.spec.minimum_samples)
                    ),
                }
            )
        return action, weight, metadata

    def update(
        self,
        *,
        features: np.ndarray,
        action_index: int,
        cost: float,
        next_features: np.ndarray,
        terminal: bool,
        next_cycle: int | None = None,
        next_action_index: int | None = None,
        native_cycle_cost: float | None = None,
        residual_ratio: float | None = None,
    ) -> float:
        del cost, next_features, terminal, next_cycle, next_action_index
        self.steps += 1
        if (
            native_cycle_cost is None
            or residual_ratio is None
            or not np.isfinite(float(native_cycle_cost))
            or float(native_cycle_cost) < 0.0
            or not np.isfinite(float(residual_ratio))
            or float(residual_ratio) < 0.0
        ):
            self.skipped_transition_count += 1
            return 0.0

        native_cost = float(native_cycle_cost)
        safe_ratio = max(float(residual_ratio), float(self.spec.numerical_floor))
        progress = float(-math.log(safe_ratio))
        joint = self.state_action_feature(features, int(action_index))
        cost_error = float(native_cost - joint @ self.cost_theta)
        progress_error = float(progress - joint @ self.progress_theta)
        projected = self.design_inverse @ joint
        denominator = 1.0 + float(joint @ projected)
        self.design_matrix += np.outer(joint, joint)
        self.cost_b += native_cost * joint
        self.progress_b += progress * joint
        if (
            not np.isfinite(denominator)
            or denominator <= float(self.spec.numerical_floor)
        ):
            self.design_inverse[:] = np.linalg.pinv(self.design_matrix)
            self.cost_theta[:] = self.design_inverse @ self.cost_b
            self.progress_theta[:] = self.design_inverse @ self.progress_b
            self.inverse_rebuild_count += 1
        else:
            gain = projected / denominator
            self.design_inverse -= np.outer(projected, projected) / denominator
            self.cost_theta += gain * cost_error
            self.progress_theta += gain * progress_error
        if not (
            np.all(np.isfinite(self.cost_theta))
            and np.all(np.isfinite(self.progress_theta))
        ):
            raise FloatingPointError("model-based RLS parameters became non-finite")
        self.native_costs.append(native_cost)
        self.signed_progress.append(progress)
        self.sample_count += 1
        return cost_error

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            self.episodes += 1

    def snapshot_learning_state(self) -> Dict[str, Any]:
        state = self._snapshot_common_learning_state()
        state.update(
            {
                "design_matrix": self.design_matrix.copy(),
                "design_inverse": self.design_inverse.copy(),
                "cost_b": self.cost_b.copy(),
                "progress_b": self.progress_b.copy(),
                "cost_theta": self.cost_theta.copy(),
                "progress_theta": self.progress_theta.copy(),
                "native_costs": list(self.native_costs),
                "signed_progress": list(self.signed_progress),
                "sample_count": int(self.sample_count),
                "skipped_transition_count": int(self.skipped_transition_count),
                "inverse_rebuild_count": int(self.inverse_rebuild_count),
            }
        )
        return state

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self._restore_common_learning_state(state)
        for name in (
            "design_matrix",
            "design_inverse",
            "cost_b",
            "progress_b",
            "cost_theta",
            "progress_theta",
        ):
            getattr(self, name)[:] = np.asarray(state[name], dtype=float)
        self.native_costs.clear()
        self.native_costs.extend(float(value) for value in state["native_costs"])
        self.signed_progress.clear()
        self.signed_progress.extend(
            float(value) for value in state["signed_progress"]
        )
        self.sample_count = int(state["sample_count"])
        self.skipped_transition_count = int(state["skipped_transition_count"])
        self.inverse_rebuild_count = int(state["inverse_rebuild_count"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            design_matrix=self.design_matrix,
            design_inverse=self.design_inverse,
            cost_b=self.cost_b,
            progress_b=self.progress_b,
            cost_theta=self.cost_theta,
            progress_theta=self.progress_theta,
            native_costs=np.asarray(self.native_costs, dtype=float),
            signed_progress=np.asarray(self.signed_progress, dtype=float),
            sample_count=np.asarray(self.sample_count, dtype=np.int64),
            skipped_transition_count=np.asarray(
                self.skipped_transition_count, dtype=np.int64
            ),
            inverse_rebuild_count=np.asarray(
                self.inverse_rebuild_count, dtype=np.int64
            ),
            checkpoint_version=np.asarray(
                _STRUCTURED_MODEL_BASED_CHECKPOINT_VERSION, dtype=np.int64
            ),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["checkpoint_version"].item()) != int(
                _STRUCTURED_MODEL_BASED_CHECKPOINT_VERSION
            ):
                raise ValueError("model-based checkpoint version does not match")
            if json.loads(str(payload["config"].item())) != _json_dataclass(
                self.config
            ):
                raise ValueError("Checkpoint model-based config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(
                self.spec
            ):
                raise ValueError("Checkpoint model-based spec does not match")
            for name in (
                "design_matrix",
                "design_inverse",
                "cost_b",
                "progress_b",
                "cost_theta",
                "progress_theta",
            ):
                getattr(self, name)[:] = np.asarray(payload[name], dtype=float)
            self.native_costs.clear()
            self.native_costs.extend(
                np.asarray(payload["native_costs"], dtype=float).tolist()
            )
            self.signed_progress.clear()
            self.signed_progress.extend(
                np.asarray(payload["signed_progress"], dtype=float).tolist()
            )
            self.sample_count = int(payload["sample_count"].item())
            self.skipped_transition_count = int(
                payload["skipped_transition_count"].item()
            )
            self.inverse_rebuild_count = int(
                payload["inverse_rebuild_count"].item()
            )
            self._load_common(payload, restore_counters=restore_counters)
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "joint_feature_dim": int(self.joint_dim),
            "sample_count": int(self.sample_count),
            "skipped_transition_count": int(self.skipped_transition_count),
            "cost_floor_sec": float(self.cost_floor),
            "progress_floor": float(self.progress_floor),
            "inverse_rebuild_count": int(self.inverse_rebuild_count),
            "stored_transition_count": 0,
        }
