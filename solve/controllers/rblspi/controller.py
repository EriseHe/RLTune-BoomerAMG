from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from solve.controllers.common import (
    _SharedActionLcbController,
    _greedy_cost_index,
    _json_dataclass,
    _rank_one_accumulate,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig


@dataclass(frozen=True)
class RecursiveBlstdqSpec:
    """Bayesian LSTDQ posterior controls for RBLSPI-style exploration."""

    prior_precision: float = 1.0e4
    noise_precision: float = 1.0e6
    gram_ridge: float = 1.0e-6

_RECURSIVE_BLSTDQ_CHECKPOINT_VERSION = 1


class RecursiveBlstdqController(_SharedActionLcbController):
    """Recursive BLSTDQ with one coherent RBLSPI sample per solve.

    The controller retains only the fixed-size empirical Bellman statistics
    ``A``, ``C`` and ``b``.  Its posterior is the BLSTD posterior

    ``S^-1 = alpha I + beta A.T C^-1 A`` and
    ``m = beta S A.T C^-1 b``.

    Unlike a heuristic randomization of recursive LSTDQ, the sampled
    covariance is therefore symmetric positive definite even though ``A`` is
    generally nonsymmetric.  Actions minimize cost, so all greedy operations
    are the cost-minimizing counterpart of the reward-maximizing RBLSPI
    algorithm.
    """

    prefer_mean_on_score_ties = True

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: RecursiveBlstdqSpec,
        seed: int,
    ) -> None:
        if float(spec.prior_precision) <= 0.0:
            raise ValueError("BLSTDQ prior precision must be positive")
        if float(spec.noise_precision) <= 0.0:
            raise ValueError("BLSTDQ noise precision must be positive")
        if float(spec.gram_ridge) <= 0.0:
            raise ValueError("BLSTDQ Gram ridge must be positive")
        if not np.isclose(float(config.trace_lambda), 0.0):
            raise ValueError("RBLSPI uses the standard lambda=0 BLSTDQ system")
        super().__init__(feature_dim=feature_dim, config=config, seed=seed)
        self.spec = spec
        self.a_matrix = np.zeros(
            (self.joint_dim, self.joint_dim), dtype=float, order="F"
        )
        self.c_matrix = np.zeros(
            (self.joint_dim, self.joint_dim), dtype=float, order="F"
        )
        self.b = np.zeros(self.joint_dim, dtype=float)
        self.posterior_mean = np.zeros(self.joint_dim, dtype=float)
        self.sampled_theta = np.zeros(self.joint_dim, dtype=float)
        self.posterior_precision_cholesky = (
            np.eye(self.joint_dim, dtype=float)
            * math.sqrt(float(spec.prior_precision))
        )
        self._joint_feature_work = np.empty(self.joint_dim, dtype=float)
        self._next_joint_feature_work = np.empty(self.joint_dim, dtype=float)
        self._difference_work = np.empty(self.joint_dim, dtype=float)
        self._posterior_dirty = True
        self._episode_sample_pending = True
        self.sample_count = 0
        self.posterior_factorization_count = 0
        self.posterior_sample_count = 0
        self.last_postfit_td_error = 0.0

    @property
    def requires_next_action(self) -> bool:
        return False

    def start_episode(
        self, *, initial_environment_weight: float | None = None
    ) -> None:
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)
        # Sampling is delayed until select_action(), where the runner accounts
        # for it as decision overhead.
        self._episode_sample_pending = True

    def _refresh_posterior(self) -> None:
        if not self._posterior_dirty:
            return
        dimension = int(self.joint_dim)
        c_regularized = self.c_matrix.copy(order="F")
        c_regularized.flat[:: dimension + 1] += float(self.spec.gram_ridge)
        c_regularized = 0.5 * (c_regularized + c_regularized.T)
        c_cholesky = np.linalg.cholesky(c_regularized)
        whitened_a = solve_triangular(
            c_cholesky,
            self.a_matrix,
            lower=True,
            check_finite=False,
        )
        whitened_b = solve_triangular(
            c_cholesky,
            self.b,
            lower=True,
            check_finite=False,
        )
        beta = float(self.spec.noise_precision)
        precision = beta * (whitened_a.T @ whitened_a)
        precision.flat[:: dimension + 1] += float(
            self.spec.prior_precision
        )
        precision = 0.5 * (precision + precision.T)
        precision_cholesky = np.linalg.cholesky(precision)
        natural = beta * (whitened_a.T @ whitened_b)
        self.posterior_mean[:] = cho_solve(
            (precision_cholesky, True),
            natural,
            check_finite=False,
        )
        self.posterior_precision_cholesky[:] = precision_cholesky
        if not np.all(np.isfinite(self.posterior_mean)):
            raise FloatingPointError("BLSTDQ posterior mean became non-finite")
        self.posterior_factorization_count += 1
        self._posterior_dirty = False

    def _ensure_episode_sample(self) -> None:
        if not self._episode_sample_pending:
            return
        self._refresh_posterior()
        standard_normal = self.rng.standard_normal(self.joint_dim)
        innovation = solve_triangular(
            self.posterior_precision_cholesky.T,
            standard_normal,
            lower=False,
            check_finite=False,
        )
        self.sampled_theta[:] = self.posterior_mean + innovation
        if not np.all(np.isfinite(self.sampled_theta)):
            raise FloatingPointError("BLSTDQ posterior sample became non-finite")
        self.posterior_sample_count += 1
        self._episode_sample_pending = False

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del cycle
        self._ensure_episode_sample()
        joint = self.state_action_features(features)
        means = joint @ self.posterior_mean
        sampled_scores = joint @ self.sampled_theta
        # RBLSPI selects using the sampled value function; an explicit UCB
        # contraction is neither needed nor paid on each multigrid cycle.
        uncertainty = np.zeros_like(means)
        return means, uncertainty, sampled_scores

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
                    "posterior_sampling": "BLSTD/RBLSPI",
                    "posterior_sample_count": int(self.posterior_sample_count),
                    "posterior_factorization_count": int(
                        self.posterior_factorization_count
                    ),
                }
            )
        return action, weight, metadata

    def _mean_policy_action(self, features: np.ndarray) -> int:
        joint = self.state_action_features(features)
        means = joint @ self.posterior_mean
        return _greedy_cost_index(
            means,
            allowed_indices=self._all_action_indices,
            anchor_index=self.anchor_index,
        )

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
        del next_cycle, next_action_index, native_cycle_cost, residual_ratio
        if self._episode_sample_pending:
            # Normal runner flow selects before updating.  This fallback keeps
            # direct controller use well defined without refreshing mid-episode.
            self._ensure_episode_sample()
        joint = self.state_action_feature(
            features,
            int(action_index),
            out=self._joint_feature_work,
        )
        if terminal:
            next_joint = self._next_joint_feature_work
            next_joint.fill(0.0)
        else:
            target_action = self._mean_policy_action(next_features)
            next_joint = self.state_action_feature(
                next_features,
                target_action,
                out=self._next_joint_feature_work,
            )
        difference = self._difference_work
        np.multiply(next_joint, float(self.config.gamma), out=difference)
        np.subtract(joint, difference, out=difference)

        td_error = float(
            cost
            + float(self.config.gamma) * float(next_joint @ self.posterior_mean)
            - float(joint @ self.posterior_mean)
        )
        _rank_one_accumulate(self.a_matrix, joint, difference)
        _rank_one_accumulate(self.c_matrix, joint, joint)
        self.b += float(cost) * joint
        self.steps += 1
        self.sample_count += 1
        self._posterior_dirty = True
        self.last_postfit_td_error = td_error
        return td_error

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            self.episodes += 1

    def snapshot_learning_state(self) -> Dict[str, Any]:
        state = self._snapshot_common_learning_state()
        state.update(
            {
                "a_matrix": self.a_matrix.copy(),
                "c_matrix": self.c_matrix.copy(),
                "b": self.b.copy(),
                "posterior_mean": self.posterior_mean.copy(),
                "sampled_theta": self.sampled_theta.copy(),
                "posterior_precision_cholesky": (
                    self.posterior_precision_cholesky.copy()
                ),
                "posterior_dirty": bool(self._posterior_dirty),
                "episode_sample_pending": bool(self._episode_sample_pending),
                "sample_count": int(self.sample_count),
                "posterior_factorization_count": int(
                    self.posterior_factorization_count
                ),
                "posterior_sample_count": int(self.posterior_sample_count),
                "last_postfit_td_error": float(self.last_postfit_td_error),
            }
        )
        return state

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self._restore_common_learning_state(state)
        for name in (
            "a_matrix",
            "c_matrix",
            "b",
            "posterior_mean",
            "sampled_theta",
            "posterior_precision_cholesky",
        ):
            getattr(self, name)[:] = np.asarray(state[name], dtype=float)
        self._posterior_dirty = bool(state["posterior_dirty"])
        self._episode_sample_pending = bool(state["episode_sample_pending"])
        self.sample_count = int(state["sample_count"])
        self.posterior_factorization_count = int(
            state["posterior_factorization_count"]
        )
        self.posterior_sample_count = int(state["posterior_sample_count"])
        self.last_postfit_td_error = float(state["last_postfit_td_error"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            a_matrix=self.a_matrix,
            c_matrix=self.c_matrix,
            b=self.b,
            posterior_mean=self.posterior_mean,
            sampled_theta=self.sampled_theta,
            posterior_precision_cholesky=(
                self.posterior_precision_cholesky
            ),
            posterior_dirty=np.asarray(self._posterior_dirty, dtype=np.bool_),
            episode_sample_pending=np.asarray(
                self._episode_sample_pending, dtype=np.bool_
            ),
            sample_count=np.asarray(self.sample_count, dtype=np.int64),
            posterior_factorization_count=np.asarray(
                self.posterior_factorization_count, dtype=np.int64
            ),
            posterior_sample_count=np.asarray(
                self.posterior_sample_count, dtype=np.int64
            ),
            last_postfit_td_error=np.asarray(
                self.last_postfit_td_error, dtype=float
            ),
            checkpoint_version=np.asarray(
                _RECURSIVE_BLSTDQ_CHECKPOINT_VERSION, dtype=np.int64
            ),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["checkpoint_version"].item()) != int(
                _RECURSIVE_BLSTDQ_CHECKPOINT_VERSION
            ):
                raise ValueError("Unsupported recursive BLSTDQ checkpoint")
            if json.loads(str(payload["config"].item())) != _json_dataclass(
                self.config
            ):
                raise ValueError("Checkpoint recursive BLSTDQ config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(
                self.spec
            ):
                raise ValueError("Checkpoint recursive BLSTDQ spec does not match")
            for name in (
                "a_matrix",
                "c_matrix",
                "b",
                "posterior_mean",
                "sampled_theta",
                "posterior_precision_cholesky",
            ):
                target = getattr(self, name)
                saved = np.asarray(payload[name], dtype=float)
                if saved.shape != target.shape:
                    raise ValueError(
                        f"Checkpoint recursive BLSTDQ {name} shape does not match"
                    )
                target[:] = saved
            self._posterior_dirty = bool(payload["posterior_dirty"].item())
            self._episode_sample_pending = bool(
                payload["episode_sample_pending"].item()
            )
            self.sample_count = int(payload["sample_count"].item())
            self.posterior_factorization_count = int(
                payload["posterior_factorization_count"].item()
            )
            self.posterior_sample_count = int(
                payload["posterior_sample_count"].item()
            )
            self.last_postfit_td_error = float(
                payload["last_postfit_td_error"].item()
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
            "prior_precision": float(self.spec.prior_precision),
            "noise_precision": float(self.spec.noise_precision),
            "gram_ridge": float(self.spec.gram_ridge),
            "posterior": "BLSTD empirical-Bellman Gaussian",
            "exploration": "one RBLSPI value-function sample per solve",
            "posterior_factorization_count": int(
                self.posterior_factorization_count
            ),
            "posterior_sample_count": int(self.posterior_sample_count),
            "stored_transition_count": 0,
        }
