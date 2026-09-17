from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common import (
    _blas_dger,
    _json_dataclass,
    _rank_one_accumulate,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig

from .common import _FactorizedLstdqScoring
from .config import RecursiveLstdqV3LcbSpec
from .v1 import RecursiveLstdqLcbController

_RECURSIVE_LSTDQ_V3_CHECKPOINT_VERSION = 2


class RecursiveLstdqV3LcbController(RecursiveLstdqLcbController):
    """Recursive LSTDQ with episode-cluster sandwich uncertainty."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: RecursiveLstdqV3LcbSpec,
        seed: int,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            config=config,
            spec=spec,
            seed=seed,
        )
        use_rank_one_blas = _blas_dger is not None
        if use_rank_one_blas:
            self.a_matrix = np.asfortranarray(self.a_matrix)
            self.a_inverse = np.asfortranarray(self.a_inverse)

        # V1 accumulates one moment per transition.  V3 owns a separate
        # cluster covariance and commits exactly one estimating-equation
        # moment for each successfully committed solve episode.
        del self.moment_covariance
        ridge = float(spec.ridge)
        self.episode_moment_covariance = (
            float(spec.residual_floor_sec) ** 2
            * ridge
            * np.eye(self.joint_dim, dtype=float)
        )
        if use_rank_one_blas:
            self.episode_moment_covariance = np.asfortranarray(
                self.episode_moment_covariance
            )
        self.episode_moment_count = 0
        self._episode_a_start = self.a_matrix.copy()
        self._episode_b_start = self.b.copy()
        self._episode_score = np.empty(self.joint_dim, dtype=float)
        self._episode_active = False
        self._factorized_scoring = _FactorizedLstdqScoring(
            action_basis=self.action_basis,
            feature_dim=self.feature_dim,
        )

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        super().start_episode(
            initial_environment_weight=initial_environment_weight
        )
        self._episode_a_start[:] = self.a_matrix
        self._episode_b_start[:] = self.b
        self._episode_active = True

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del cycle
        state = self._factorized_scoring.prepare_state(features)
        state_lift = self._factorized_scoring.state_lift
        projected_lift = state_lift @ self.a_inverse
        reduced_covariance = (
            projected_lift @ self.episode_moment_covariance
        ) @ projected_lift.T
        reduced_covariance = 0.5 * (
            reduced_covariance + reduced_covariance.T
        )
        means = self._factorized_scoring.means(self.theta, state)
        quadratic = self._factorized_scoring.action_quadratic(
            reduced_covariance
        )
        if not (
            np.all(np.isfinite(means))
            and np.all(np.isfinite(reduced_covariance))
            and np.all(np.isfinite(quadratic))
        ):
            raise FloatingPointError(
                "LSTDQ v3 sandwich scoring became non-finite"
            )
        roundoff_tolerance = (
            100.0
            * np.finfo(float).eps
            * max(
                1.0,
                float(np.linalg.norm(reduced_covariance, ord=np.inf)),
            )
        )
        if np.any(quadratic < -roundoff_tolerance):
            raise FloatingPointError(
                "LSTDQ v3 sandwich covariance produced a materially "
                "negative action variance"
            )
        uncertainty = np.sqrt(np.maximum(quadratic, 0.0))
        scores = means - float(self.spec.uncertainty_beta) * uncertainty
        if self.spec.lcb_lower_bound_sec is not None:
            scores = np.maximum(
                scores,
                float(self.spec.lcb_lower_bound_sec),
            )
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
        del native_cycle_cost, residual_ratio
        td_error, _joint = self._update_mean(
            features=features,
            action_index=action_index,
            cost=cost,
            next_features=next_features,
            terminal=terminal,
            next_cycle=next_cycle,
            next_action_index=next_action_index,
        )
        return float(td_error)

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            if not self._episode_active:
                raise RuntimeError(
                    "LSTDQ v3 cannot commit an episode that was not started"
                )
            self._resync_theta(require_inverse=True)
            # g_e = b_e - A_e theta, evaluated at the current post-fit
            # parameter.  The equivalent expanded expression avoids storing a
            # second joint_dim x joint_dim episode-delta matrix.
            self._episode_score[:] = self.b - self._episode_b_start
            self._episode_score -= self.a_matrix @ self.theta
            self._episode_score += self._episode_a_start @ self.theta
            if not np.all(np.isfinite(self._episode_score)):
                raise FloatingPointError(
                    "LSTDQ v3 episode estimating-equation moment became "
                    "non-finite"
                )
            _rank_one_accumulate(
                self.episode_moment_covariance,
                self._episode_score,
                self._episode_score,
            )
            if not np.all(np.isfinite(self.episode_moment_covariance)):
                raise FloatingPointError(
                    "LSTDQ v3 episode moment covariance became non-finite"
                )
            self.episode_moment_count += 1
            self.episodes += 1
        self.trace.fill(0.0)
        self._episode_active = False

    def snapshot_learning_state(self) -> Dict[str, Any]:
        state = self._snapshot_common_learning_state()
        state.update(
            {
                "a_matrix": self.a_matrix.copy(),
                "a_inverse": self.a_inverse.copy(),
                "b": self.b.copy(),
                "theta": self.theta.copy(),
                "episode_moment_covariance": (
                    self.episode_moment_covariance.copy()
                ),
                "episode_moment_count": int(self.episode_moment_count),
                "trace": self.trace.copy(),
                "sample_count": int(self.sample_count),
                "inverse_rebuild_count": int(self.inverse_rebuild_count),
                "inverse_is_valid": bool(self.inverse_is_valid),
                "last_postfit_td_error": float(self.last_postfit_td_error),
                "episode_a_start": self._episode_a_start.copy(),
                "episode_b_start": self._episode_b_start.copy(),
                "episode_active": bool(self._episode_active),
            }
        )
        return state

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self._restore_common_learning_state(state)
        self.a_matrix[:] = np.asarray(state["a_matrix"], dtype=float)
        self.a_inverse[:] = np.asarray(state["a_inverse"], dtype=float)
        self.b[:] = np.asarray(state["b"], dtype=float)
        self.theta[:] = np.asarray(state["theta"], dtype=float)
        self.episode_moment_covariance[:] = np.asarray(
            state["episode_moment_covariance"],
            dtype=float,
        )
        self.episode_moment_count = int(state["episode_moment_count"])
        self.trace[:] = np.asarray(state["trace"], dtype=float)
        self.sample_count = int(state["sample_count"])
        self.inverse_rebuild_count = int(state["inverse_rebuild_count"])
        self.last_postfit_td_error = float(state["last_postfit_td_error"])
        self._episode_a_start[:] = np.asarray(
            state["episode_a_start"],
            dtype=float,
        )
        self._episode_b_start[:] = np.asarray(
            state["episode_b_start"],
            dtype=float,
        )
        self._episode_active = bool(state["episode_active"])
        self._restore_inverse_validity(state)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            a_matrix=self.a_matrix,
            a_inverse=self.a_inverse,
            b=self.b,
            episode_moment_covariance=self.episode_moment_covariance,
            episode_moment_count=np.asarray(
                self.episode_moment_count,
                dtype=np.int64,
            ),
            sample_count=np.asarray(self.sample_count, dtype=np.int64),
            inverse_rebuild_count=np.asarray(
                self.inverse_rebuild_count,
                dtype=np.int64,
            ),
            inverse_is_valid=np.asarray(self.inverse_is_valid, dtype=bool),
            last_postfit_td_error=np.asarray(
                self.last_postfit_td_error,
                dtype=float,
            ),
            checkpoint_version=np.asarray(
                _RECURSIVE_LSTDQ_V3_CHECKPOINT_VERSION,
                dtype=np.int64,
            ),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            version = int(payload["checkpoint_version"].item())
            if version not in (1, _RECURSIVE_LSTDQ_V3_CHECKPOINT_VERSION):
                raise ValueError("LSTDQ v3 checkpoint version does not match")
            if json.loads(str(payload["config"].item())) != _json_dataclass(
                self.config
            ):
                raise ValueError("Checkpoint LSTDQ v3 config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(
                self.spec
            ):
                raise ValueError("Checkpoint LSTDQ v3 spec does not match")
            self.theta[:] = np.asarray(payload["theta"], dtype=float)
            self.a_matrix[:] = np.asarray(payload["a_matrix"], dtype=float)
            self.a_inverse[:] = np.asarray(payload["a_inverse"], dtype=float)
            self.b[:] = np.asarray(payload["b"], dtype=float)
            self.episode_moment_covariance[:] = np.asarray(
                payload["episode_moment_covariance"],
                dtype=float,
            )
            self.episode_moment_count = int(
                payload["episode_moment_count"].item()
            )
            self.sample_count = int(payload["sample_count"].item())
            self.inverse_rebuild_count = int(
                payload["inverse_rebuild_count"].item()
            )
            self.last_postfit_td_error = float(
                payload["last_postfit_td_error"].item()
            )
            self._load_common(payload, restore_counters=restore_counters)
            self._restore_inverse_validity(payload)
        self.trace.fill(0.0)
        self._episode_a_start[:] = self.a_matrix
        self._episode_b_start[:] = self.b
        self._episode_active = False
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "joint_feature_dim": int(self.joint_dim),
            "sample_count": int(self.sample_count),
            "episode_moment_count": int(self.episode_moment_count),
            "uncertainty_beta": float(self.spec.uncertainty_beta),
            "lcb_lower_bound_sec": self.spec.lcb_lower_bound_sec,
            "uncertainty": (
                "episode-cluster post-fit sandwich covariance"
            ),
            "inverse_rebuild_count": int(self.inverse_rebuild_count),
            "inverse_is_valid": bool(self.inverse_is_valid),
            "stored_transition_count": 0,
        }


__all__ = ["RecursiveLstdqV3LcbController"]
