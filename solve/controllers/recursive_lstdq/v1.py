from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common import (
    _SharedActionLcbController,
    _json_dataclass,
    _rank_one_accumulate,
    _rank_one_inverse_accumulate,
    _sandwich_quadratic,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig

from .config import RecursiveLstdqLcbSpec

_RECURSIVE_LSTDQ_CHECKPOINT_VERSION = 3
_INVERSE_DENOMINATOR_CEILING = 1.0 / np.sqrt(np.finfo(float).eps)


class RecursiveLstdqLcbController(_SharedActionLcbController):
    """Recursive LSTDQ(lambda) with post-fit sandwich uncertainty."""

    prefer_mean_on_score_ties = True

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: RecursiveLstdqLcbSpec,
        seed: int,
    ) -> None:
        if float(spec.ridge) <= 0.0:
            raise ValueError("recursive LSTDQ ridge must be positive")
        if float(spec.uncertainty_beta) < 0.0:
            raise ValueError("recursive LSTDQ uncertainty beta must be non-negative")
        if float(spec.residual_floor_sec) <= 0.0:
            raise ValueError("recursive LSTDQ residual floor must be positive")
        if float(spec.inverse_denominator_floor) <= 0.0:
            raise ValueError("inverse denominator floor must be positive")
        if (
            spec.lcb_lower_bound_sec is not None
            and float(spec.lcb_lower_bound_sec) < 0.0
        ):
            raise ValueError("recursive LSTDQ LCB lower bound must be non-negative")
        super().__init__(feature_dim=feature_dim, config=config, seed=seed)
        self.spec = spec
        ridge = float(spec.ridge)
        self.a_matrix = ridge * np.eye(self.joint_dim, dtype=float)
        self.a_inverse = np.eye(self.joint_dim, dtype=float) / ridge
        self.b = np.zeros(self.joint_dim, dtype=float)
        self.theta = np.zeros(self.joint_dim, dtype=float)
        self.moment_covariance = (
            float(spec.residual_floor_sec) ** 2
            * ridge
            * np.eye(self.joint_dim, dtype=float)
        )
        self.trace = np.zeros(self.joint_dim, dtype=float)
        self._joint_feature_work = np.empty(self.joint_dim, dtype=float)
        self._next_joint_feature_work = np.empty(self.joint_dim, dtype=float)
        self._difference_work = np.empty(self.joint_dim, dtype=float)
        self.sample_count = 0
        self.inverse_rebuild_count = 0
        self.inverse_is_valid = True
        self.last_postfit_td_error = 0.0

    @property
    def requires_next_action(self) -> bool:
        return True

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        self.trace.fill(0.0)
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)

    def _inverse_residual_is_small(self, inverse: np.ndarray) -> bool:
        """Audit a rebuilt/legacy inverse, never on the normal update path."""

        if not np.all(np.isfinite(inverse)):
            return False
        identity = np.eye(self.joint_dim)
        return bool(
            np.linalg.norm(self.a_matrix @ inverse - identity, ord=np.inf) <= 1.0e-8
            and np.linalg.norm(inverse @ self.a_matrix - identity, ord=np.inf) <= 1.0e-8
        )

    def _rebuild_inverse(self) -> None:
        """Keep a singular-prefix pseudoinverse out of inverse recursions."""

        if not np.all(np.isfinite(self.a_matrix)):
            raise FloatingPointError("LSTDQ estimating matrix became non-finite")
        self.inverse_is_valid = False
        try:
            inverse = np.linalg.inv(self.a_matrix)
            self.inverse_is_valid = self._inverse_residual_is_small(inverse)
        except np.linalg.LinAlgError:
            pass
        if not self.inverse_is_valid:
            inverse = np.linalg.pinv(self.a_matrix)
        if not np.all(np.isfinite(inverse)):
            raise FloatingPointError("LSTDQ rebuilt inverse became non-finite")
        # Preserve the Fortran layout used by v2/v3's BLAS updates.
        self.a_inverse[:] = inverse
        self.inverse_rebuild_count += 1

    def _restore_inverse_validity(self, state: Any) -> None:
        """Read the validity flag, or audit a checkpoint predating the flag."""

        validity = state.get("inverse_is_valid")
        if validity is not None:
            self.inverse_is_valid = bool(np.asarray(validity).item())
            return
        self.inverse_is_valid = self._inverse_residual_is_small(self.a_inverse)
        if not self.inverse_is_valid:
            self._rebuild_inverse()
            self._resync_theta()

    def _resync_theta(self, *, require_inverse: bool = False) -> None:
        """Use the stored prefix solve, requiring a true inverse at termination."""

        if require_inverse and not self.inverse_is_valid:
            self._rebuild_inverse()
            if not self.inverse_is_valid:
                raise FloatingPointError(
                    "Could not restore a valid LSTDQ inverse at episode boundary"
                )

        self.theta[:] = self.a_inverse @ self.b
        if not np.all(np.isfinite(self.theta)):
            raise FloatingPointError("recursive LSTDQ parameters became non-finite")

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del cycle
        joint = self.state_action_features(features)
        means = joint @ self.theta
        projected = joint @ self.a_inverse
        quadratic = _sandwich_quadratic(projected, self.moment_covariance)
        uncertainty = np.sqrt(np.maximum(quadratic, 0.0))
        scores = means - float(self.spec.uncertainty_beta) * uncertainty
        if self.spec.lcb_lower_bound_sec is not None:
            scores = np.maximum(scores, float(self.spec.lcb_lower_bound_sec))
        return means, uncertainty, scores

    def _update_mean(
        self,
        *,
        features: np.ndarray,
        action_index: int,
        cost: float,
        next_features: np.ndarray,
        terminal: bool,
        next_cycle: int | None = None,
        next_action_index: int | None = None,
    ) -> tuple[float, np.ndarray]:
        """Apply the recursive LSTDQ mean update shared by v1 and v2."""

        del next_cycle
        joint = self.state_action_feature(
            features,
            int(action_index),
            out=self._joint_feature_work,
        )
        if terminal:
            next_joint = self._next_joint_feature_work
            next_joint.fill(0.0)
        else:
            if next_action_index is None:
                raise ValueError("recursive LSTDQ requires the executed next action")
            next_joint = self.state_action_feature(
                next_features,
                int(next_action_index),
                out=self._next_joint_feature_work,
            )
        difference = self._difference_work
        np.multiply(next_joint, float(self.config.gamma), out=difference)
        np.subtract(joint, difference, out=difference)
        td_error = float(cost + self.config.gamma * (next_joint @ self.theta) - joint @ self.theta)
        np.multiply(
            self.trace,
            float(self.config.gamma) * float(self.config.trace_lambda),
            out=self.trace,
        )
        self.trace += joint
        denominator = float("nan")
        if self.inverse_is_valid:
            projected_trace = self.a_inverse @ self.trace
            projected_difference = difference @ self.a_inverse
            denominator = 1.0 + float(difference @ projected_trace)
        _rank_one_accumulate(self.a_matrix, self.trace, difference)
        self.b += float(cost) * self.trace
        if (
            not np.isfinite(denominator)
            or abs(denominator) < float(self.spec.inverse_denominator_floor)
            # Leaving a nearly singular prefix can subtract two huge inverse
            # entries to obtain a small one. Rebuild instead of losing digits.
            or abs(denominator) > _INVERSE_DENOMINATOR_CEILING
        ):
            self._rebuild_inverse()
            self._resync_theta(require_inverse=terminal)
        else:
            _rank_one_inverse_accumulate(
                self.a_inverse,
                projected_trace,
                projected_difference,
                denominator=denominator,
            )
            # Algebraically identical to theta = A_new^-1 b_new, without a
            # dense matrix-vector product on every cycle.
            self.theta += projected_trace * (td_error / denominator)
            if terminal:
                self._resync_theta(require_inverse=True)
        if not np.all(np.isfinite(self.theta)):
            raise FloatingPointError("recursive LSTDQ parameters became non-finite")
        self.last_postfit_td_error = float(cost - difference @ self.theta)
        self.steps += 1
        self.sample_count += 1
        return td_error, joint

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
        moment = self.trace * self.last_postfit_td_error
        self.moment_covariance += np.outer(moment, moment)
        if not np.all(np.isfinite(self.moment_covariance)):
            raise FloatingPointError(
                "recursive LSTDQ moment covariance became non-finite"
            )
        return td_error

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            self._resync_theta(require_inverse=True)
            self.episodes += 1
        self.trace.fill(0.0)

    def snapshot_learning_state(self) -> Dict[str, Any]:
        state = self._snapshot_common_learning_state()
        state.update(
            {
                "a_matrix": self.a_matrix.copy(),
                "a_inverse": self.a_inverse.copy(),
                "b": self.b.copy(),
                "theta": self.theta.copy(),
                "moment_covariance": self.moment_covariance.copy(),
                "trace": self.trace.copy(),
                "sample_count": int(self.sample_count),
                "inverse_rebuild_count": int(self.inverse_rebuild_count),
                "inverse_is_valid": bool(self.inverse_is_valid),
                "last_postfit_td_error": float(self.last_postfit_td_error),
            }
        )
        return state

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self._restore_common_learning_state(state)
        self.a_matrix[:] = np.asarray(state["a_matrix"], dtype=float)
        self.a_inverse[:] = np.asarray(state["a_inverse"], dtype=float)
        self.b[:] = np.asarray(state["b"], dtype=float)
        self.theta[:] = np.asarray(state["theta"], dtype=float)
        self.moment_covariance[:] = np.asarray(
            state["moment_covariance"], dtype=float
        )
        self.trace[:] = np.asarray(state["trace"], dtype=float)
        self.sample_count = int(state["sample_count"])
        self.inverse_rebuild_count = int(state["inverse_rebuild_count"])
        self.last_postfit_td_error = float(state["last_postfit_td_error"])
        self._restore_inverse_validity(state)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            a_matrix=self.a_matrix,
            a_inverse=self.a_inverse,
            b=self.b,
            moment_covariance=self.moment_covariance,
            sample_count=np.asarray(self.sample_count, dtype=np.int64),
            inverse_rebuild_count=np.asarray(
                self.inverse_rebuild_count, dtype=np.int64
            ),
            inverse_is_valid=np.asarray(self.inverse_is_valid, dtype=bool),
            last_postfit_td_error=np.asarray(
                self.last_postfit_td_error, dtype=float
            ),
            checkpoint_version=np.asarray(
                _RECURSIVE_LSTDQ_CHECKPOINT_VERSION, dtype=np.int64
            ),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            version = int(payload.get("checkpoint_version", np.asarray(1)).item())
            if version not in (2, _RECURSIVE_LSTDQ_CHECKPOINT_VERSION):
                raise ValueError(
                    "Legacy recursive LSTDQ checkpoint is incompatible with "
                    "post-fit covariance"
                )
            if json.loads(str(payload["config"].item())) != _json_dataclass(self.config):
                raise ValueError("Checkpoint recursive LSTDQ config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(self.spec):
                raise ValueError("Checkpoint recursive LSTDQ spec does not match")
            self.theta[:] = np.asarray(payload["theta"], dtype=float)
            self.a_matrix[:] = np.asarray(payload["a_matrix"], dtype=float)
            self.a_inverse[:] = np.asarray(payload["a_inverse"], dtype=float)
            self.b[:] = np.asarray(payload["b"], dtype=float)
            self.moment_covariance[:] = np.asarray(
                payload["moment_covariance"], dtype=float
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
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "joint_feature_dim": int(self.joint_dim),
            "sample_count": int(self.sample_count),
            "uncertainty_beta": float(self.spec.uncertainty_beta),
            "lcb_lower_bound_sec": self.spec.lcb_lower_bound_sec,
            "covariance_residual": "postfit_unclipped",
            "inverse_rebuild_count": int(self.inverse_rebuild_count),
            "inverse_is_valid": bool(self.inverse_is_valid),
            "stored_transition_count": 0,
        }
