"""Generic shared linear Thompson sampling for BoomerAMG setup tuning.

This version uses the same parameter-space feature map, compact catalog, AOT
candidate schedule, failure head, and recovery protocol as SharedLinUCB v4.
It maintains a Cholesky factor of the precision matrix so one exact posterior
sample costs O(d^2), rather than recomputing an O(d^3) factorization per case.
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, Mapping, Optional

import numpy as np
from scipy.linalg import solve_triangular

from ..linucb.SharedLinUCB_AMG_v4 import (
    SharedLinUCB_AMG_v4,
    SharedLinUCBv4DeferredObservation,
)


def _cholesky_rank_one_update(
    lower: np.ndarray,
    vector: np.ndarray,
) -> None:
    """Update ``lower`` in place so it factors ``A + vector vector.T``."""

    work = np.asarray(vector, dtype=float).reshape(-1).copy()
    if lower.shape != (work.size, work.size):
        raise ValueError("Cholesky factor and update vector dimensions differ")
    for index in range(work.size):
        diagonal = float(lower[index, index])
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            raise np.linalg.LinAlgError(
                "precision Cholesky factor has a non-positive diagonal"
            )
        radius = float(np.hypot(diagonal, work[index]))
        cosine = radius / diagonal
        sine = float(work[index]) / diagonal
        lower[index, index] = radius
        if index + 1 >= work.size:
            continue
        column = (
            lower[index + 1 :, index]
            + sine * work[index + 1 :]
        ) / cosine
        lower[index + 1 :, index] = column
        work[index + 1 :] = (
            cosine * work[index + 1 :] - sine * column
        )


class SharedLinTS_AMG_v2(SharedLinUCB_AMG_v4):
    """Exact Gaussian linear TS over the generic v4 setup feature map.

    ``relative_sampling_scale`` is dimensionless.  It multiplies the first
    successful observed end-to-end loss, removing the old learner's dependence
    on a hard-coded absolute runtime unit.  ``loss_scale_prior`` is used only
    before such an observation exists.
    """

    CHECKPOINT_SCHEMA_VERSION = 1

    def __init__(
        self,
        *args: Any,
        relative_sampling_scale: float = 0.15,
        loss_scale_prior: float = 0.1,
        posterior_seed: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        relative_scale = float(relative_sampling_scale)
        prior = float(loss_scale_prior)
        if not np.isfinite(relative_scale) or relative_scale <= 0.0:
            raise ValueError("relative_sampling_scale must be finite and positive")
        if not np.isfinite(prior) or prior <= 0.0:
            raise ValueError("loss_scale_prior must be finite and positive")

        super().__init__(*args, seed=seed, **kwargs)
        self.relative_sampling_scale = relative_scale
        self.loss_scale_prior = prior
        self.reference_loss: Optional[float] = None
        self._precision_cholesky = (
            np.eye(self.d_phi, dtype=float) * np.sqrt(self.l2_reg)
        )

        if posterior_seed is None:
            posterior_sequence = np.random.SeedSequence(seed).spawn(2)[1]
            self.posterior_rng = np.random.default_rng(posterior_sequence)
        else:
            self.posterior_rng = np.random.default_rng(int(posterior_seed))
        self.last_sampling_scale = self._effective_sampling_scale()

    def _effective_sampling_scale(self) -> float:
        reference = (
            self.loss_scale_prior
            if self.reference_loss is None
            else float(self.reference_loss)
        )
        return float(self.relative_sampling_scale * reference)

    def _sample_parameter(self) -> tuple[np.ndarray, np.ndarray, float]:
        mean = self.A_inv @ self.b
        standard_normal = self.posterior_rng.standard_normal(self.d_phi)
        innovation = solve_triangular(
            self._precision_cholesky.T,
            standard_normal,
            lower=False,
            check_finite=False,
        )
        scale = self._effective_sampling_scale()
        self.last_sampling_scale = scale
        return mean + scale * innovation, mean, scale

    def _selection_score_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        action_features: Optional[np.ndarray] = None,
        alpha: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del alpha
        sampled_theta, mean_theta, _scale = self._sample_parameter()
        sampled_loss = self._linear_prediction_subset(
            x,
            theta=sampled_theta,
            arms=arms,
            action_features=action_features,
        )
        mean_loss = self._linear_prediction_subset(
            x,
            theta=mean_theta,
            arms=arms,
            action_features=action_features,
        )
        # Selection does not require a 512-arm uncertainty contraction.  The
        # shared predict path computes the chosen arm's diagnostic uncertainty.
        uncertainty = np.full(mean_loss.shape, np.nan, dtype=float)
        return sampled_loss, mean_loss, uncertainty

    def predict(self, context):
        selected = super().predict(context)
        if self.candidate_stats_history:
            self.candidate_stats_history[-1]["posterior_scale_sec"] = float(
                self.last_sampling_scale
            )
        return selected

    def _after_runtime_precision_update(self, phi: np.ndarray) -> None:
        _cholesky_rank_one_update(self._precision_cholesky, phi)

    def _record_reference_loss(
        self,
        loss: float,
        *,
        failure_label: float,
    ) -> None:
        if self.reference_loss is not None or float(failure_label) > 0.5:
            return
        value = float(loss)
        if np.isfinite(value) and value > 0.0:
            self.reference_loss = value

    def update(self, loss: float, *, failure_label: float = 0.0) -> None:
        super().update(loss, failure_label=failure_label)
        self._record_reference_loss(loss, failure_label=failure_label)

    def commit_deferred_observation(
        self,
        observation: SharedLinUCBv4DeferredObservation,
        *,
        loss: float,
    ) -> None:
        failure_label = float(
            self.history[int(observation.history_index)].failure_label
        )
        super().commit_deferred_observation(observation, loss=loss)
        self._record_reference_loss(loss, failure_label=failure_label)

    def begin_recovery_transaction(self) -> None:
        super().begin_recovery_transaction()
        self._recovery_transaction["lints_precision_cholesky"] = (
            self._precision_cholesky.copy()
        )

    def rollback_recovery_transaction(self) -> None:
        state = self._recovery_transaction
        saved_cholesky = (
            None
            if state is None
            else np.asarray(
                state["lints_precision_cholesky"], dtype=float
            ).copy()
        )
        super().rollback_recovery_transaction()
        if saved_cholesky is not None:
            self._precision_cholesky[:] = saved_cholesky

    def clone_for_independent_updates(self) -> "SharedLinTS_AMG_v2":
        clone = super().clone_for_independent_updates()
        clone._precision_cholesky = self._precision_cholesky.copy()
        clone.posterior_rng = np.random.default_rng()
        clone.posterior_rng.bit_generator.state = copy.deepcopy(
            self.posterior_rng.bit_generator.state
        )
        return clone

    def _mutable_state_extra_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "lints_v2_checkpoint_version": np.asarray(
                self.CHECKPOINT_SCHEMA_VERSION, dtype=np.int64
            ),
            "lints_v2_precision_cholesky": self._precision_cholesky,
            "lints_v2_relative_sampling_scale": np.asarray(
                self.relative_sampling_scale, dtype=float
            ),
            "lints_v2_loss_scale_prior": np.asarray(
                self.loss_scale_prior, dtype=float
            ),
            "lints_v2_reference_loss": np.asarray(
                np.nan if self.reference_loss is None else self.reference_loss,
                dtype=float,
            ),
            "lints_v2_posterior_rng_state": np.asarray(
                json.dumps(self.posterior_rng.bit_generator.state)
            ),
        }

    def _load_mutable_state_extra_arrays(
        self,
        payload: Mapping[str, np.ndarray],
        *,
        checkpoint_version: int,
    ) -> None:
        del checkpoint_version
        required = {
            "lints_v2_checkpoint_version",
            "lints_v2_precision_cholesky",
            "lints_v2_relative_sampling_scale",
            "lints_v2_loss_scale_prior",
            "lints_v2_reference_loss",
            "lints_v2_posterior_rng_state",
        }
        missing = required - set(payload.files)
        if missing:
            raise ValueError(
                f"LinTS v2 checkpoint is missing {sorted(missing)}"
            )
        version = int(payload["lints_v2_checkpoint_version"].item())
        if version != self.CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported LinTS v2 checkpoint version: {version}")
        for name, configured, saved in (
            (
                "relative sampling scale",
                self.relative_sampling_scale,
                float(payload["lints_v2_relative_sampling_scale"].item()),
            ),
            (
                "loss-scale prior",
                self.loss_scale_prior,
                float(payload["lints_v2_loss_scale_prior"].item()),
            ),
        ):
            if not np.isclose(configured, saved, rtol=0.0, atol=1.0e-15):
                raise ValueError(f"LinTS v2 checkpoint {name} does not match")
        cholesky = np.asarray(
            payload["lints_v2_precision_cholesky"], dtype=float
        )
        if cholesky.shape != self._precision_cholesky.shape:
            raise ValueError("LinTS v2 checkpoint Cholesky shape does not match")
        if not np.all(np.isfinite(cholesky)) or np.any(
            np.diag(cholesky) <= 0.0
        ):
            raise ValueError("LinTS v2 checkpoint Cholesky factor is invalid")
        self._precision_cholesky[:] = cholesky
        reference = float(payload["lints_v2_reference_loss"].item())
        self.reference_loss = None if np.isnan(reference) else reference
        self.posterior_rng.bit_generator.state = json.loads(
            str(payload["lints_v2_posterior_rng_state"].item())
        )
        self.last_sampling_scale = self._effective_sampling_scale()


__all__ = ["SharedLinTS_AMG_v2"]
