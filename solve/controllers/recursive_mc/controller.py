from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common import (
    _SharedActionLcbController,
    _json_dataclass,
    _sandwich_quadratic,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig

from .config import RecursiveMonteCarloLcbSpec

_RECURSIVE_MC_CHECKPOINT_VERSION = 2


class RecursiveMonteCarloLcbController(_SharedActionLcbController):
    """Exponentially weighted recursive MC control with clustered uncertainty."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: RecursiveMonteCarloLcbSpec,
        seed: int,
    ) -> None:
        if float(spec.ridge) <= 0.0:
            raise ValueError("recursive MC ridge must be positive")
        if float(spec.uncertainty_beta) < 0.0:
            raise ValueError("recursive MC uncertainty beta must be non-negative")
        if float(spec.residual_floor_sec) <= 0.0:
            raise ValueError("recursive MC residual floor must be positive")
        if float(spec.episode_half_life) <= 0.0:
            raise ValueError("recursive MC episode half-life must be positive")
        super().__init__(feature_dim=feature_dim, config=config, seed=seed)
        self.spec = spec
        self.design_inverse = np.eye(self.joint_dim, dtype=float) / float(spec.ridge)
        self.b = np.zeros(self.joint_dim, dtype=float)
        self.theta = np.zeros(self.joint_dim, dtype=float)
        self.moment_covariance = np.zeros(
            (self.joint_dim, self.joint_dim), dtype=float
        )
        self.postfit_residual_sum_squares = 0.0
        self.postfit_residual_weight = 0.0
        self.sample_count = 0
        self._pending: list[tuple[np.ndarray, float]] = []

    @property
    def requires_next_action(self) -> bool:
        return False

    @property
    def residual_scale(self) -> float:
        if self.postfit_residual_weight <= 0.0:
            return float(self.spec.residual_floor_sec)
        return max(
            math.sqrt(
                self.postfit_residual_sum_squares
                / float(self.postfit_residual_weight)
            ),
            float(self.spec.residual_floor_sec),
        )

    @property
    def episode_forgetting_factor(self) -> float:
        return float(2.0 ** (-1.0 / float(self.spec.episode_half_life)))

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        self._pending.clear()
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
        means = joint @ self.theta
        projected = joint @ self.design_inverse
        coverage_quadratic = np.sum(projected * joint, axis=1)
        sandwich_quadratic = _sandwich_quadratic(
            projected,
            self.moment_covariance,
        )
        floor_quadratic = (
            float(self.spec.residual_floor_sec) ** 2
            * np.maximum(coverage_quadratic, 0.0)
        )
        uncertainty = np.sqrt(
            np.maximum.reduce(
                [
                    sandwich_quadratic,
                    floor_quadratic,
                    np.zeros_like(sandwich_quadratic),
                ]
            )
        )
        scores = means - float(self.spec.uncertainty_beta) * uncertainty
        return means, uncertainty, scores

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
        del (
            next_features,
            next_cycle,
            next_action_index,
            native_cycle_cost,
            residual_ratio,
        )
        joint = self.state_action_feature(features, int(action_index))
        self._pending.append((joint, float(cost)))
        self.steps += 1
        return float(cost - joint @ self.theta) if terminal else 0.0

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            observations: list[tuple[np.ndarray, float]] = []
            return_to_go = 0.0
            for joint, cost in reversed(self._pending):
                return_to_go = float(cost) + float(self.config.gamma) * return_to_go
                observations.append((joint, return_to_go))

            observations.reverse()
            forgetting = self.episode_forgetting_factor
            self.design_inverse /= forgetting
            self.b *= forgetting
            for joint, return_value in observations:
                projected = self.design_inverse @ joint
                denominator = 1.0 + float(joint @ projected)
                if denominator <= 1.0e-12 or not np.isfinite(denominator):
                    raise FloatingPointError("recursive MC inverse update became singular")
                self.design_inverse -= np.outer(projected, projected) / denominator
                self.b += float(return_value) * joint
                self.sample_count += 1

            self.theta = self.design_inverse @ self.b
            if not np.all(np.isfinite(self.theta)):
                raise FloatingPointError("recursive MC parameters became non-finite")

            cluster_score = np.zeros(self.joint_dim, dtype=float)
            episode_residual_sum_squares = 0.0
            for joint, return_value in observations:
                residual = float(return_value - joint @ self.theta)
                cluster_score += joint * residual
                episode_residual_sum_squares += residual * residual
            self.moment_covariance *= forgetting * forgetting
            self.moment_covariance += np.outer(cluster_score, cluster_score)
            self.postfit_residual_sum_squares = (
                forgetting * self.postfit_residual_sum_squares
                + episode_residual_sum_squares
            )
            self.postfit_residual_weight = (
                forgetting * self.postfit_residual_weight
                + float(len(observations))
            )
            self.episodes += 1
        self._pending.clear()

    def snapshot_learning_state(self) -> Dict[str, Any]:
        state = self._snapshot_common_learning_state()
        state.update(
            {
                "design_inverse": self.design_inverse.copy(),
                "b": self.b.copy(),
                "theta": self.theta.copy(),
                "moment_covariance": self.moment_covariance.copy(),
                "postfit_residual_sum_squares": float(
                    self.postfit_residual_sum_squares
                ),
                "postfit_residual_weight": float(
                    self.postfit_residual_weight
                ),
                "sample_count": int(self.sample_count),
            }
        )
        return state

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self._restore_common_learning_state(state)
        self.design_inverse[:] = np.asarray(state["design_inverse"], dtype=float)
        self.b[:] = np.asarray(state["b"], dtype=float)
        self.theta[:] = np.asarray(state["theta"], dtype=float)
        self.moment_covariance[:] = np.asarray(
            state["moment_covariance"], dtype=float
        )
        self.postfit_residual_sum_squares = float(
            state["postfit_residual_sum_squares"]
        )
        self.postfit_residual_weight = float(
            state["postfit_residual_weight"]
        )
        self.sample_count = int(state["sample_count"])
        self._pending.clear()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            design_inverse=self.design_inverse,
            b=self.b,
            moment_covariance=self.moment_covariance,
            postfit_residual_sum_squares=np.asarray(
                self.postfit_residual_sum_squares, dtype=float
            ),
            postfit_residual_weight=np.asarray(
                self.postfit_residual_weight, dtype=float
            ),
            sample_count=np.asarray(self.sample_count, dtype=np.int64),
            checkpoint_version=np.asarray(
                _RECURSIVE_MC_CHECKPOINT_VERSION, dtype=np.int64
            ),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            version = int(payload.get("checkpoint_version", np.asarray(1)).item())
            if version != _RECURSIVE_MC_CHECKPOINT_VERSION:
                raise ValueError(
                    "Legacy recursive MC checkpoint is incompatible with "
                    "EW-RLS clustered covariance"
                )
            if json.loads(str(payload["config"].item())) != _json_dataclass(self.config):
                raise ValueError("Checkpoint recursive MC config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(self.spec):
                raise ValueError("Checkpoint recursive MC spec does not match")
            self.theta[:] = np.asarray(payload["theta"], dtype=float)
            self.design_inverse[:] = np.asarray(
                payload["design_inverse"], dtype=float
            )
            self.b[:] = np.asarray(payload["b"], dtype=float)
            self.moment_covariance[:] = np.asarray(
                payload["moment_covariance"], dtype=float
            )
            self.postfit_residual_sum_squares = float(
                payload["postfit_residual_sum_squares"].item()
            )
            self.postfit_residual_weight = float(
                payload["postfit_residual_weight"].item()
            )
            self.sample_count = int(payload["sample_count"].item())
            self._load_common(payload, restore_counters=restore_counters)
        self._pending.clear()
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "joint_feature_dim": int(self.joint_dim),
            "sample_count": int(self.sample_count),
            "residual_scale_sec": float(self.residual_scale),
            "episode_half_life": float(self.spec.episode_half_life),
            "episode_forgetting_factor": float(self.episode_forgetting_factor),
            "uncertainty_beta": float(self.spec.uncertainty_beta),
            "stored_transition_count": 0,
        }
