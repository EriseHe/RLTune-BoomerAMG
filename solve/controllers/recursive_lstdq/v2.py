from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common import (
    _RollingFloatWindow,
    _blas_dger,
    _json_dataclass,
    _mad_scale,
    _rank_one_accumulate,
    _rank_one_inverse_update,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig

from .common import _FactorizedLstdqScoring
from .config import RecursiveLstdqV2LcbSpec
from .v1 import RecursiveLstdqLcbController

_RECURSIVE_LSTDQ_V2_CHECKPOINT_VERSION = 2


class RecursiveLstdqV2LcbController(RecursiveLstdqLcbController):
    """Recursive LSTDQ mean with MAD-scaled feature-coverage confidence."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: RecursiveLstdqV2LcbSpec,
        seed: int,
    ) -> None:
        if float(spec.coverage_ridge) <= 0.0:
            raise ValueError("LSTDQ v2 coverage ridge must be positive")
        if int(spec.residual_scale_window) <= 0:
            raise ValueError("LSTDQ v2 residual window must be positive")
        if not 1 <= int(spec.residual_scale_min_samples) <= int(
            spec.residual_scale_window
        ):
            raise ValueError("LSTDQ v2 MAD warmup must fit inside its window")
        super().__init__(
            feature_dim=feature_dim,
            config=config,
            spec=spec,
            seed=seed,
        )
        # SciPy's in-place BLAS rank-one update requires column-major storage.
        # Keep NumPy's native row-major layout when SciPy is unavailable so the
        # exact fallback equations do not incur a column-major matvec penalty.
        use_rank_one_blas = _blas_dger is not None
        if use_rank_one_blas:
            self.a_matrix = np.asfortranarray(self.a_matrix)
            self.a_inverse = np.asfortranarray(self.a_inverse)
        # V2 deliberately does not maintain the directional residual covariance
        # allocated by the v1 constructor.
        del self.moment_covariance
        coverage_ridge = float(spec.coverage_ridge)
        self.coverage_matrix = coverage_ridge * np.eye(
            self.joint_dim,
            dtype=float,
        )
        self.coverage_inverse = (
            np.eye(self.joint_dim, dtype=float) / coverage_ridge
        )
        if use_rank_one_blas:
            self.coverage_matrix = np.asfortranarray(self.coverage_matrix)
            self.coverage_inverse = np.asfortranarray(self.coverage_inverse)
        self.coverage_inverse_rebuild_count = 0
        self._factorized_scoring = _FactorizedLstdqScoring(
            action_basis=self.action_basis,
            feature_dim=self.feature_dim,
        )
        self.postfit_td_residuals = _RollingFloatWindow(
            int(spec.residual_scale_window)
        )
        self._cached_residual_scale = float(spec.residual_floor_sec)
        self._scale_cache_sample_count = -1

    @property
    def residual_scale(self) -> float:
        if self._scale_cache_sample_count == self.sample_count:
            return float(self._cached_residual_scale)
        if len(self.postfit_td_residuals) < int(
            self.spec.residual_scale_min_samples
        ):
            scale = float(self.spec.residual_floor_sec)
        else:
            scale = _mad_scale(
                self.postfit_td_residuals.as_array(chronological=False),
                floor=float(self.spec.residual_floor_sec),
            )
        self._cached_residual_scale = float(scale)
        self._scale_cache_sample_count = int(self.sample_count)
        return float(scale)

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del cycle
        state = self._factorized_scoring.prepare_state(features)

        # Every joint feature is z_a = b_a (x) state.  Contracting the
        # 288-dimensional quadratic through the 9-dimensional action basis is
        # algebraically identical to forming all 41 joint feature vectors, but
        # avoids the 41 x 288 by 288 dense product on every solver cycle.
        state_lift = self._factorized_scoring.state_lift
        reduced_coverage = (
            state_lift @ self.coverage_inverse
        ) @ state_lift.T
        means = self._factorized_scoring.means(self.theta, state)
        quadratic = self._factorized_scoring.action_quadratic(
            reduced_coverage
        )
        uncertainty = float(self.residual_scale) * np.sqrt(
            np.maximum(quadratic, 0.0)
        )
        scores = means - float(self.spec.uncertainty_beta) * uncertainty
        if self.spec.lcb_lower_bound_sec is not None:
            scores = np.maximum(scores, float(self.spec.lcb_lower_bound_sec))
        return means, uncertainty, scores

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
            metadata["uncertainty_td_scale"] = float(self.residual_scale)
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
        del native_cycle_cost, residual_ratio
        td_error, joint = self._update_mean(
            features=features,
            action_index=action_index,
            cost=cost,
            next_features=next_features,
            terminal=terminal,
            next_cycle=next_cycle,
            next_action_index=next_action_index,
        )
        _rank_one_accumulate(self.coverage_matrix, joint, joint)
        if not _rank_one_inverse_update(
            self.coverage_inverse,
            joint,
            denominator_floor=float(self.spec.inverse_denominator_floor),
        ):
            self.coverage_inverse[:] = np.linalg.pinv(self.coverage_matrix)
            self.coverage_inverse_rebuild_count += 1
        self.postfit_td_residuals.append(float(self.last_postfit_td_error))
        self._scale_cache_sample_count = -1
        if not np.all(np.isfinite(self.coverage_inverse)):
            raise FloatingPointError("LSTDQ v2 coverage inverse became non-finite")
        return float(td_error)

    def snapshot_learning_state(self) -> Dict[str, Any]:
        state = self._snapshot_common_learning_state()
        state.update(
            {
                "a_matrix": self.a_matrix.copy(),
                "a_inverse": self.a_inverse.copy(),
                "b": self.b.copy(),
                "theta": self.theta.copy(),
                "trace": self.trace.copy(),
                "sample_count": int(self.sample_count),
                "inverse_rebuild_count": int(self.inverse_rebuild_count),
                "inverse_is_valid": bool(self.inverse_is_valid),
                "last_postfit_td_error": float(self.last_postfit_td_error),
                "coverage_matrix": self.coverage_matrix.copy(),
                "coverage_inverse": self.coverage_inverse.copy(),
                "coverage_inverse_rebuild_count": int(
                    self.coverage_inverse_rebuild_count
                ),
                "postfit_td_residuals": self.postfit_td_residuals.as_array(
                    chronological=True
                ).copy(),
            }
        )
        return state

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self._restore_common_learning_state(state)
        self.a_matrix[:] = np.asarray(state["a_matrix"], dtype=float)
        self.a_inverse[:] = np.asarray(state["a_inverse"], dtype=float)
        self.b[:] = np.asarray(state["b"], dtype=float)
        self.theta[:] = np.asarray(state["theta"], dtype=float)
        self.trace[:] = np.asarray(state["trace"], dtype=float)
        self.sample_count = int(state["sample_count"])
        self.inverse_rebuild_count = int(state["inverse_rebuild_count"])
        self.last_postfit_td_error = float(state["last_postfit_td_error"])
        self.coverage_matrix[:] = np.asarray(
            state["coverage_matrix"], dtype=float
        )
        self.coverage_inverse[:] = np.asarray(
            state["coverage_inverse"], dtype=float
        )
        self.coverage_inverse_rebuild_count = int(
            state["coverage_inverse_rebuild_count"]
        )
        self.postfit_td_residuals.clear()
        self.postfit_td_residuals.extend(
            float(value) for value in state["postfit_td_residuals"]
        )
        self._scale_cache_sample_count = -1

        self._restore_inverse_validity(state)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            a_matrix=self.a_matrix,
            a_inverse=self.a_inverse,
            b=self.b,
            coverage_matrix=self.coverage_matrix,
            coverage_inverse=self.coverage_inverse,
            postfit_td_residuals=np.asarray(
                self.postfit_td_residuals.as_array(chronological=True),
                dtype=float,
            ),
            sample_count=np.asarray(self.sample_count, dtype=np.int64),
            inverse_rebuild_count=np.asarray(
                self.inverse_rebuild_count, dtype=np.int64
            ),
            inverse_is_valid=np.asarray(self.inverse_is_valid, dtype=bool),
            coverage_inverse_rebuild_count=np.asarray(
                self.coverage_inverse_rebuild_count, dtype=np.int64
            ),
            last_postfit_td_error=np.asarray(
                self.last_postfit_td_error, dtype=float
            ),
            checkpoint_version=np.asarray(
                _RECURSIVE_LSTDQ_V2_CHECKPOINT_VERSION, dtype=np.int64
            ),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            version = int(payload["checkpoint_version"].item())
            if version not in (1, _RECURSIVE_LSTDQ_V2_CHECKPOINT_VERSION):
                raise ValueError("LSTDQ v2 checkpoint version does not match")
            if json.loads(str(payload["config"].item())) != _json_dataclass(
                self.config
            ):
                raise ValueError("Checkpoint LSTDQ v2 config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(
                self.spec
            ):
                raise ValueError("Checkpoint LSTDQ v2 spec does not match")
            self.theta[:] = np.asarray(payload["theta"], dtype=float)
            self.a_matrix[:] = np.asarray(payload["a_matrix"], dtype=float)
            self.a_inverse[:] = np.asarray(payload["a_inverse"], dtype=float)
            self.b[:] = np.asarray(payload["b"], dtype=float)
            self.coverage_matrix[:] = np.asarray(
                payload["coverage_matrix"], dtype=float
            )
            self.coverage_inverse[:] = np.asarray(
                payload["coverage_inverse"], dtype=float
            )
            self.sample_count = int(payload["sample_count"].item())
            self.inverse_rebuild_count = int(
                payload["inverse_rebuild_count"].item()
            )
            self.coverage_inverse_rebuild_count = int(
                payload["coverage_inverse_rebuild_count"].item()
            )
            self.last_postfit_td_error = float(
                payload["last_postfit_td_error"].item()
            )
            self.postfit_td_residuals.clear()
            self.postfit_td_residuals.extend(
                np.asarray(payload["postfit_td_residuals"], dtype=float).tolist()
            )
            self._load_common(payload, restore_counters=restore_counters)
            self._restore_inverse_validity(payload)
        self.trace.fill(0.0)
        self._scale_cache_sample_count = -1
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
            "uncertainty": "rolling-MAD-scaled feature coverage",
            "residual_scale_sec": float(self.residual_scale),
            "residual_scale_window": int(self.spec.residual_scale_window),
            "residual_scale_samples": int(len(self.postfit_td_residuals)),
            "inverse_rebuild_count": int(self.inverse_rebuild_count),
            "inverse_is_valid": bool(self.inverse_is_valid),
            "coverage_inverse_rebuild_count": int(
                self.coverage_inverse_rebuild_count
            ),
            "stored_transition_count": 0,
        }
