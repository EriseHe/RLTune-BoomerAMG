"""
Shared LinUCB for BoomerAMG setup tuning with a generic n-parameter action spec.

v4 keeps the LinUCB v3 architecture:
- shared linear model over context-action features
- candidate subset support
- elite cache support
- optional alpha decay

The difference is that action encoding is driven by a ParameterSpaceSpec
instead of a hardcoded 3- or 5-knob feature map.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..common.action_features import (
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    action_from_ordered_values,
    action_key_from_parameter_space_spec,
)
from ..common.aot_candidates import (
    AOTCandidateSchedule,
    FactorizedActionFeatureCache,
)
from ..common.candidate_subset import CandidateSelector


@dataclass(frozen=True)
class SharedLinUCBv4Step:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float
    failure_label: float = float("nan")
    pred_failure_mean: float = float("nan")
    pred_failure_uncert: float = float("nan")


@dataclass(frozen=True)
class SharedLinUCBv4DeferredObservation:
    """One selected setup whose suffix runtime is not known yet."""

    arm_index: int
    phi: np.ndarray
    provisional_loss: float
    history_index: int


class SharedLinUCB_AMG_v4:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        parameter_spec: ParameterSpaceSpec,
        alpha: float = 1.0,
        alpha_decay: bool = False,
        l2_reg: float = 1.0,
        context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
        candidate_pool_size: Optional[int] = None,
        candidate_strategy: str = "uniform",
        candidate_pool_size_burnin: Optional[int] = None,
        candidate_pool_burnin_rounds: int = 0,
        alpha_decay_burnin_rounds: int = 0,
        elite_rank_metric: str = "best_loss",
        local_neighbor_radius: int = 1,
        candidate_local_fraction: float = 0.0,
        candidate_elite_fraction: float = 0.0,
        always_include_arms: Optional[Sequence[int]] = None,
        elite_cache_size: int = 0,
        failure_beta: float = 2.0,
        failure_l2_reg: float = 1.0,
        initial_guess: Optional[Sequence[Any] | Mapping[str, Any]] = None,
        initial_guess_rounds: int = 0,
        candidate_schedule: Optional[AOTCandidateSchedule] = None,
        action_feature_cache: Optional[FactorizedActionFeatureCache] = None,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")
        if l2_reg <= 0.0:
            raise ValueError("l2_reg must be > 0")
        if failure_beta < 0.0:
            raise ValueError("failure_beta must be non-negative")
        if failure_l2_reg <= 0.0:
            raise ValueError("failure_l2_reg must be > 0")
        if candidate_strategy not in {"uniform", "structured", "adaptive_local"}:
            raise ValueError(
                "candidate_strategy must be 'uniform', 'structured', or "
                "'adaptive_local'"
            )
        if candidate_pool_burnin_rounds < 0:
            raise ValueError("candidate_pool_burnin_rounds must be >= 0")
        if alpha_decay_burnin_rounds < 0:
            raise ValueError("alpha_decay_burnin_rounds must be >= 0")
        if local_neighbor_radius < 0:
            raise ValueError("local_neighbor_radius must be >= 0")
        if not (0.0 <= float(candidate_local_fraction) <= 1.0):
            raise ValueError("candidate_local_fraction must be in [0, 1]")
        if not (0.0 <= float(candidate_elite_fraction) <= 1.0):
            raise ValueError("candidate_elite_fraction must be in [0, 1]")
        if float(candidate_local_fraction) + float(candidate_elite_fraction) > 1.0 + 1e-12:
            raise ValueError("candidate_local_fraction + candidate_elite_fraction must be <= 1")
        if initial_guess_rounds < 0:
            raise ValueError("initial_guess_rounds must be >= 0")

        # Compact catalogs provide random access without materializing one dict
        # per arm. Ordinary iterables retain the historical concrete-list path.
        self.actions: Sequence[Dict[str, Any]] = (
            actions
            if isinstance(actions, list)
            or bool(getattr(actions, "is_compact_action_catalog", False))
            else list(actions)
        )
        self.parameter_spec = parameter_spec
        self._encoder = GenericActionFeatureEncoder(parameter_spec)
        self._candidate_schedule = candidate_schedule
        self._action_feature_cache = action_feature_cache

        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.g_dim = int(self._encoder.feature_dim)
        self.context_interaction_indices = tuple(int(i) for i in context_interaction_indices)
        for idx in self.context_interaction_indices:
            if not (0 <= idx < self.d_x):
                raise ValueError("context_interaction_indices must be within [0, context_dim)")
        self.d_phi = self.d_x + (1 + len(self.context_interaction_indices)) * self.g_dim

        self.alpha = float(alpha)
        self.alpha_decay = bool(alpha_decay)
        self.l2_reg = float(l2_reg)
        self.failure_beta = float(failure_beta)
        self.failure_l2_reg = float(failure_l2_reg)

        self.candidate_strategy = str(candidate_strategy)
        self.candidate_pool_size_burnin = (
            int(candidate_pool_size_burnin) if candidate_pool_size_burnin is not None else None
        )
        self.candidate_pool_burnin_rounds = int(candidate_pool_burnin_rounds)
        self.alpha_decay_burnin_rounds = int(alpha_decay_burnin_rounds)
        self.local_neighbor_radius = int(local_neighbor_radius)
        self.candidate_local_fraction = float(candidate_local_fraction)
        self.candidate_elite_fraction = float(candidate_elite_fraction)
        if (candidate_schedule is None) != (action_feature_cache is None):
            raise ValueError(
                "candidate_schedule and action_feature_cache must be provided together"
            )
        if candidate_schedule is not None:
            expected_schedule_method = {
                "uniform": "uniform512",
                "structured": "structured512",
            }.get(self.candidate_strategy)
            if expected_schedule_method is None:
                raise ValueError(
                    "AOT candidates support uniform and structured strategies"
                )
            if candidate_schedule.sampling_method != expected_schedule_method:
                raise ValueError(
                    "AOT schedule sampling method does not match candidate_strategy"
                )
            if candidate_schedule.catalog is not self.actions:
                raise ValueError("AOT schedule must use the model action catalog")
            if action_feature_cache.catalog is not self.actions:
                raise ValueError("factorized cache must use the model action catalog")
            if candidate_schedule.pool_size < int(candidate_pool_size or self.K):
                raise ValueError("AOT schedule pool is smaller than candidate_pool_size")

        self.rng = np.random.default_rng(seed)
        self._cand = CandidateSelector(
            self.K,
            candidate_pool_size=candidate_pool_size,
            always_include_arms=always_include_arms,
            elite_cache_size=int(elite_cache_size),
            elite_rank_metric=str(elite_rank_metric),
            sparse_statistics=candidate_schedule is not None,
            rng=self.rng,
        )

        self.A_inv = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.b = np.zeros(self.d_phi, dtype=float)
        if np.isclose(
            self.failure_l2_reg,
            self.l2_reg,
            rtol=0.0,
            atol=1.0e-15,
        ):
            self.failure_A_inv = self.A_inv
        else:
            self.failure_A_inv = (
                np.eye(self.d_phi, dtype=float) / self.failure_l2_reg
            )
        self.failure_b = np.zeros(self.d_phi, dtype=float)
        self.failure_observation_count = 0

        self._g_actions = (
            None
            if self._action_feature_cache is not None
            else self._encoder.encode_actions(self.actions)
        )
        if self._g_actions is not None:
            self._g_actions.setflags(write=False)

        self._param_by_name = {param.name: param for param in self.parameter_spec.parameters}
        self._param_value_to_index = {
            param.name: {value: idx for idx, value in enumerate(param.values)}
            for param in self.parameter_spec.parameters
        }
        # Structured/local candidates repeatedly expand the default, incumbent,
        # and a small elite set.  Their coordinate topology is immutable, so
        # cache arm IDs rather than rebuilding action dictionaries every round.
        self._local_neighbor_cache: Dict[
            Tuple[int, bool, int], np.ndarray
        ] = {}
        self._local_neighbor_cache_max_size = 4096
        self.initial_guess_rounds = int(initial_guess_rounds)
        self._initial_guess_arm: Optional[int] = None
        self._initial_guess_action: Optional[Dict[str, Any]] = None
        if initial_guess is not None:
            if isinstance(initial_guess, Mapping):
                guess_action = dict(initial_guess)
            else:
                guess_action = action_from_ordered_values(initial_guess, self.parameter_spec)
            self._initial_guess_action = guess_action
            initial_key = action_key_from_parameter_space_spec(guess_action, self.parameter_spec)
            index_of = getattr(self.actions, "index_of", None)
            if callable(index_of):
                try:
                    self._initial_guess_arm = int(index_of(guess_action))
                except ValueError:
                    self._initial_guess_arm = None
            else:
                for idx, action in enumerate(self.actions):
                    action_key = action_key_from_parameter_space_spec(action, self.parameter_spec)
                    if action_key == initial_key:
                        self._initial_guess_arm = int(idx)
                        break
            if self._initial_guess_arm is None:
                raise ValueError("initial_guess does not match any action in the provided action set")
        self._action_key_to_arm = None
        if self.candidate_strategy == "adaptive_local":
            if self._action_feature_cache is not None:
                raise ValueError(
                    "adaptive_local requires an explicit materialized action catalog"
                )
            self._action_key_to_arm = {
                action_key_from_parameter_space_spec(action, self.parameter_spec): int(i)
                for i, action in enumerate(self.actions)
            }

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[SharedLinUCBv4Step] = []
        self.candidate_stats_history: List[Dict[str, int | str]] = []
        self._recovery_transaction: Optional[Dict[str, Any]] = None

    def clone_for_independent_updates(self) -> "SharedLinUCB_AMG_v4":
        """Clone mutable online state while sharing the immutable action catalog."""
        if self._candidate_schedule is not None:
            raise RuntimeError(
                "AOT schedules are branch-specific and cannot be cloned"
            )
        self._g_actions.setflags(write=False)
        clone = object.__new__(type(self))
        clone.__dict__ = self.__dict__.copy()
        clone.rng = np.random.default_rng()
        clone.rng.bit_generator.state = copy.deepcopy(self.rng.bit_generator.state)
        clone._cand = self._cand.clone_for_independent_updates(rng=clone.rng)
        clone.A_inv = self.A_inv.copy()
        clone.b = self.b.copy()
        clone.failure_A_inv = (
            clone.A_inv
            if self.failure_A_inv is self.A_inv
            else self.failure_A_inv.copy()
        )
        clone.failure_b = self.failure_b.copy()
        clone._last_phi = None if self._last_phi is None else self._last_phi.copy()
        clone.history = list(self.history)
        clone.candidate_stats_history = [dict(row) for row in self.candidate_stats_history]
        clone._local_neighbor_cache = {}
        clone._recovery_transaction = None
        return clone

    def _mutable_state_extra_arrays(self) -> Dict[str, np.ndarray]:
        """Subclass checkpoint payload stored beside the shared online state."""

        return {}

    def _load_mutable_state_extra_arrays(
        self,
        payload: Mapping[str, np.ndarray],
        *,
        checkpoint_version: int,
    ) -> None:
        """Restore subclass state from a shared mutable-state checkpoint."""

        del payload, checkpoint_version

    def save_mutable_state(
        self,
        path: Path,
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Persist online state without serializing the immutable action catalog."""
        if self._last_phi is not None or self._last_arm is not None:
            raise RuntimeError("Cannot checkpoint LinUCB with a pending action")
        if self._recovery_transaction is not None:
            raise RuntimeError("Cannot checkpoint LinUCB during recovery")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        candidate = self._cand
        candidate_state = candidate.export_statistics()
        extra_arrays = self._mutable_state_extra_arrays()
        np.savez_compressed(
            path,
            action_count=np.asarray(self.K, dtype=np.int64),
            parameter_dim=np.asarray(self.d_phi, dtype=np.int64),
            A_inv=self.A_inv,
            b=self.b,
            failure_A_inv=self.failure_A_inv,
            failure_b=self.failure_b,
            failure_observation_count=np.asarray(
                self.failure_observation_count, dtype=np.int64
            ),
            failure_beta=np.asarray(self.failure_beta, dtype=float),
            failure_l2_reg=np.asarray(self.failure_l2_reg, dtype=float),
            checkpoint_version=np.asarray(3, dtype=np.int64),
            t=np.asarray(self.t, dtype=np.int64),
            elite_arms=candidate._elite_arms,
            candidate_stats_sparse=np.asarray(
                bool(candidate_state["sparse"]), dtype=np.bool_
            ),
            candidate_stat_arms=candidate_state["arms"],
            arm_best_loss=candidate_state["best_loss"],
            arm_loss_sum=candidate_state["loss_sum"],
            arm_obs_count=candidate_state["obs_count"],
            candidate_schedule_cursor=np.asarray(
                0
                if self._candidate_schedule is None
                else self._candidate_schedule.cursor,
                dtype=np.int64,
            ),
            rng_state=np.asarray(json.dumps(self.rng.bit_generator.state)),
            metadata=np.asarray(json.dumps(dict(metadata or {}))),
            **extra_arrays,
        )

    def load_mutable_state(self, path: Path) -> Dict[str, Any]:
        """Restore a checkpoint created by :meth:`save_mutable_state`."""
        with np.load(Path(path), allow_pickle=False) as payload:
            if int(payload["action_count"].item()) != self.K:
                raise ValueError("LinUCB checkpoint action count does not match")
            if int(payload["parameter_dim"].item()) != self.d_phi:
                raise ValueError("LinUCB checkpoint feature dimension does not match")
            inverse = np.asarray(payload["A_inv"], dtype=float)
            b = np.asarray(payload["b"], dtype=float)
            if inverse.shape != self.A_inv.shape or b.shape != self.b.shape:
                raise ValueError("LinUCB checkpoint parameter shape does not match")
            self.A_inv[:] = inverse
            self.b[:] = b
            checkpoint_version = int(
                payload.get("checkpoint_version", np.asarray(1)).item()
            )
            if checkpoint_version >= 2:
                if not np.isclose(
                    float(payload["failure_beta"].item()),
                    self.failure_beta,
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    raise ValueError("LinUCB checkpoint failure beta does not match")
                if not np.isclose(
                    float(payload["failure_l2_reg"].item()),
                    self.failure_l2_reg,
                    rtol=0.0,
                    atol=1.0e-15,
                ):
                    raise ValueError("LinUCB checkpoint failure ridge does not match")
                failure_inverse = np.asarray(payload["failure_A_inv"], dtype=float)
                failure_b = np.asarray(payload["failure_b"], dtype=float)
                if (
                    failure_inverse.shape != self.failure_A_inv.shape
                    or failure_b.shape != self.failure_b.shape
                ):
                    raise ValueError(
                        "LinUCB checkpoint failure-head shape does not match"
                    )
                if self.failure_A_inv is not self.A_inv:
                    self.failure_A_inv[:] = failure_inverse
                self.failure_b[:] = failure_b
                self.failure_observation_count = int(
                    payload["failure_observation_count"].item()
                )
            else:
                if self.failure_A_inv is not self.A_inv:
                    self.failure_A_inv[:] = (
                        np.eye(self.d_phi, dtype=float) / self.failure_l2_reg
                    )
                self.failure_b.fill(0.0)
                self.failure_observation_count = 0
            self.t = int(payload["t"].item())
            self._cand._elite_arms = np.asarray(
                payload["elite_arms"], dtype=int
            ).copy()
            if checkpoint_version >= 3:
                self._cand.restore_statistics(
                    {
                        "sparse": bool(payload["candidate_stats_sparse"].item()),
                        "arms": np.asarray(payload["candidate_stat_arms"]),
                        "best_loss": np.asarray(payload["arm_best_loss"]),
                        "loss_sum": np.asarray(payload["arm_loss_sum"]),
                        "obs_count": np.asarray(payload["arm_obs_count"]),
                    }
                )
                saved_cursor = int(payload["candidate_schedule_cursor"].item())
                if self._candidate_schedule is None:
                    if saved_cursor:
                        raise ValueError(
                            "checkpoint contains an unexpected AOT schedule cursor"
                        )
                else:
                    self._candidate_schedule.set_cursor(saved_cursor)
            else:
                for name, attribute in (
                    ("arm_best_loss", "_arm_best_loss"),
                    ("arm_loss_sum", "_arm_loss_sum"),
                    ("arm_obs_count", "_arm_obs_count"),
                ):
                    saved = np.asarray(payload[name])
                    current = getattr(self._cand, attribute)
                    if current is None:
                        if saved.size:
                            raise ValueError(
                                f"LinUCB checkpoint unexpectedly contains {name}"
                            )
                    else:
                        if saved.shape != current.shape:
                            raise ValueError(
                                f"LinUCB checkpoint {name} shape does not match"
                            )
                        current[:] = saved
            self.rng.bit_generator.state = json.loads(
                str(payload["rng_state"].item())
            )
            self._load_mutable_state_extra_arrays(
                payload,
                checkpoint_version=checkpoint_version,
            )
            metadata = json.loads(str(payload["metadata"].item()))
        self._last_phi = None
        self._last_arm = None
        self.history = []
        self.candidate_stats_history = []
        self._recovery_transaction = None
        if self._g_actions is not None:
            self._g_actions.setflags(write=False)
        return dict(metadata)

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _features_for_arms(self, arms: np.ndarray) -> np.ndarray:
        indices = np.asarray(arms, dtype=int)
        if self._g_actions is not None:
            return self._g_actions[indices]
        return self._action_feature_cache.features(indices)

    def _feature_for_arm(self, arm: int) -> np.ndarray:
        if self._g_actions is not None:
            return self._g_actions[int(arm)]
        return self._action_feature_cache.feature(int(arm))

    def _phi_from_feature(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        phi = np.empty(self.d_phi, dtype=float)
        phi[: self.d_x] = x
        offset = self.d_x
        phi[offset : offset + self.g_dim] = g
        offset += self.g_dim
        for idx in self.context_interaction_indices:
            phi[offset : offset + self.g_dim] = float(x[idx]) * g
            offset += self.g_dim
        return phi

    def _phi(self, x: np.ndarray, arm: int) -> np.ndarray:
        return self._phi_from_feature(x, self._feature_for_arm(int(arm)))

    def _linear_prediction_subset(
        self,
        x: np.ndarray,
        *,
        theta: np.ndarray,
        arms: Optional[np.ndarray],
        action_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if action_features is not None:
            features = np.asarray(action_features, dtype=float)
        elif arms is None:
            if self._g_actions is None:
                raise RuntimeError(
                    "full-space scoring is unavailable for factorized catalogs"
                )
            features = self._g_actions
        else:
            features = self._features_for_arms(np.asarray(arms, dtype=int))

        theta = np.asarray(theta, dtype=float).reshape(-1)
        if theta.size != self.d_phi:
            raise ValueError("theta dimension does not match the feature map")
        theta_x = theta[: self.d_x]
        offset = self.d_x
        theta_g = theta[offset : offset + self.g_dim]
        offset += self.g_dim
        prediction = np.asarray(
            float(theta_x @ x)
            + np.einsum("ij,j->i", features, theta_g, optimize=False),
            dtype=float,
        )
        for idx in self.context_interaction_indices:
            theta_block = theta[offset : offset + self.g_dim]
            offset += self.g_dim
            prediction = prediction + float(x[idx]) * np.einsum(
                "ij,j->i", features, theta_block, optimize=False
            )
        return prediction

    def _mean_uncertainty_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        action_features: Optional[np.ndarray] = None,
        precomputed_uncertainty: Optional[np.ndarray] = None,
        inverse: np.ndarray,
        target_b: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if action_features is not None:
            G = np.asarray(action_features, dtype=float)
        elif arms is None:
            if self._g_actions is None:
                raise RuntimeError(
                    "full-space scoring is unavailable for factorized catalogs"
                )
            G = self._g_actions
        else:
            G = self._features_for_arms(np.asarray(arms, dtype=int))

        theta = inverse @ target_b
        mean = self._linear_prediction_subset(
            x,
            theta=theta,
            arms=arms,
            action_features=G,
        )

        if precomputed_uncertainty is not None:
            uncert = np.asarray(precomputed_uncertainty, dtype=float).reshape(-1)
            if uncert.shape != mean.shape:
                raise ValueError("precomputed uncertainty shape does not match candidates")
            return mean, uncert

        c = np.concatenate(
            [x, np.zeros((1 + len(self.context_interaction_indices)) * self.g_dim, dtype=float)],
            axis=0,
        )
        I = np.eye(self.g_dim, dtype=float)
        P_blocks = [np.zeros((self.d_x, self.g_dim), dtype=float), I]
        for idx in self.context_interaction_indices:
            P_blocks.append(float(x[idx]) * I)
        P = np.vstack(P_blocks)

        Ac = inverse @ c
        q0 = float(c @ Ac)
        AP = inverse @ P
        u = P.T @ Ac
        M = P.T @ AP

        quad = q0 + 2.0 * np.einsum("ij,j->i", G, u, optimize=False) + np.einsum("ij,jk,ik->i", G, M, G, optimize=False)
        uncert = np.sqrt(np.maximum(0.0, quad))
        return mean, uncert

    def _score_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        action_features: Optional[np.ndarray] = None,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, uncert = self._mean_uncertainty_subset(
            x,
            arms=arms,
            action_features=action_features,
            inverse=self.A_inv,
            target_b=self.b,
        )
        score = mean - float(alpha) * uncert
        return score, mean, uncert

    def _selection_score_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        action_features: Optional[np.ndarray] = None,
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Score the normal online choice; subclasses can change exploration."""

        return self._score_subset(
            x,
            arms=arms,
            action_features=action_features,
            alpha=alpha,
        )

    def _failure_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        action_features: Optional[np.ndarray] = None,
        precomputed_uncertainty: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean, uncert = self._mean_uncertainty_subset(
            x,
            arms=arms,
            action_features=action_features,
            precomputed_uncertainty=precomputed_uncertainty,
            inverse=self.failure_A_inv,
            target_b=self.failure_b,
        )
        upper = mean + self.failure_beta * uncert
        return upper, mean, uncert

    def _effective_alpha(self) -> float:
        if not self.alpha_decay:
            return float(self.alpha)
        if self.t < self.alpha_decay_burnin_rounds:
            return float(self.alpha)
        decay_t = max(0, self.t - self.alpha_decay_burnin_rounds)
        return float(self.alpha) / np.sqrt(decay_t + 1.0)

    def _effective_candidate_pool_size(self) -> Optional[int]:
        if self.candidate_pool_size_burnin is not None and self.t < self.candidate_pool_burnin_rounds:
            return int(self.candidate_pool_size_burnin)
        return self._cand.candidate_pool_size

    def _scheduled_candidate_subset(
        self, pool_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._candidate_schedule is None:
            raise RuntimeError("AOT candidate schedule is not configured")
        base = self._candidate_schedule.next_arms()
        include = self._cand._merged_include_arms()
        if include.size > int(pool_size):
            include = include[: int(pool_size)]
        include_set = {int(arm) for arm in include}
        needed = int(pool_size) - int(include.size)
        remaining = np.fromiter(
            (
                int(arm)
                for arm in base
                if int(arm) not in include_set
            ),
            dtype=np.int64,
            count=-1,
        )[:needed]
        candidate = np.concatenate(
            [include.astype(np.int64, copy=False), remaining]
        )
        if candidate.size != int(pool_size):
            raise RuntimeError("AOT candidate row could not fill the requested pool")
        return candidate, self._features_for_arms(candidate)

    @staticmethod
    def _stable_unique_arms(
        *groups: np.ndarray | Sequence[int],
        limit: int | None = None,
    ) -> np.ndarray:
        selected: List[int] = []
        seen: set[int] = set()
        for group in groups:
            for raw_arm in np.asarray(group, dtype=np.int64).reshape(-1):
                arm = int(raw_arm)
                if arm in seen:
                    continue
                selected.append(arm)
                seen.add(arm)
                if limit is not None and len(selected) >= int(limit):
                    return np.asarray(selected, dtype=np.int64)
        return np.asarray(selected, dtype=np.int64)

    def _structured_anchor_arms(self, target: int) -> np.ndarray:
        previous = (
            np.asarray([self.history[-1].arm_index], dtype=np.int64)
            if self.history
            else np.empty(0, dtype=np.int64)
        )
        return self._stable_unique_arms(
            self._cand.always_include_arms,
            previous,
            self._cand.elite_arms,
            limit=int(target),
        )

    def _structured_candidate_subset(
        self, pool_size: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int | str]]:
        schedule = self._candidate_schedule
        if schedule is None or schedule.sampling_method != "structured512":
            raise RuntimeError("structured AOT candidate schedule is not configured")
        if int(pool_size) != int(schedule.pool_size):
            raise ValueError("structured schedule requires its configured pool size")

        base = schedule.next_arms()
        global_stop = int(schedule.structured_global_size)
        sobol_stop = global_stop + int(schedule.structured_sobol_size)
        balanced_global = base[:global_stop]
        sobol = base[global_stop:sobol_stop]

        anchors = self._structured_anchor_arms(
            int(schedule.structured_anchor_size)
        )
        seed_count = min(8, int(anchors.size))
        local_pool = self._local_neighbor_arms(
            anchors[:seed_count],
            include_categorical=True,
            radius=max(2, int(self.local_neighbor_radius)),
        )
        local = self._sample_without_replacement(
            local_pool,
            int(schedule.structured_local_size),
        )

        core = self._stable_unique_arms(
            anchors,
            balanced_global,
            local,
            sobol,
        )
        candidate = self._stable_unique_arms(
            core,
            base,
            limit=int(pool_size),
        )
        if candidate.size != int(pool_size):
            raise RuntimeError(
                "structured AOT candidate row could not fill the requested pool"
            )
        stats: Dict[str, int | str] = {
            "strategy": "structured_aot",
            "candidate_count": int(candidate.size),
            "anchor": int(anchors.size),
            "elite": int(self._cand.elite_arms.size),
            "balanced_global": int(balanced_global.size),
            "local": int(local.size),
            "sobol": int(sobol.size),
            "global": int(balanced_global.size + sobol.size),
        }
        return candidate, self._features_for_arms(candidate), stats

    def finish_candidate_schedule_case(self, *, max_selections: int) -> None:
        """Close one fixed-width AOT block after an online instance."""

        if self._candidate_schedule is not None:
            self._candidate_schedule.finish_case(
                max_selections=int(max_selections)
            )

    def _is_param_active(self, param, action: Dict[str, Any]) -> bool:
        for dep_name, allowed in param.active_if:
            if action.get(dep_name, None) not in allowed:
                return False
        return True

    def _incumbent_arm(self) -> int:
        elite = self._cand.elite_arms
        if elite.size:
            return int(elite[0])
        always_include = self._cand.always_include_arms
        if always_include.size:
            return int(always_include[0])
        return 0

    def _sample_without_replacement(self, pool: np.ndarray, need: int) -> np.ndarray:
        pool = np.asarray(pool, dtype=int)
        if need <= 0 or pool.size == 0:
            return np.zeros(0, dtype=int)
        if pool.size <= need:
            return np.unique(pool)
        choice = self.rng.choice(pool, size=int(need), replace=False)
        return np.asarray(np.unique(choice), dtype=int)

    def _sample_global_excluding(self, excluded: np.ndarray, need: int) -> np.ndarray:
        if need <= 0:
            return np.zeros(0, dtype=int)
        excluded_set = {int(v) for v in np.asarray(excluded, dtype=int)}
        out: List[int] = []
        seen = set(excluded_set)
        while len(out) < int(need) and len(seen) < self.K:
            draw = int(self.rng.integers(0, self.K))
            if draw in seen:
                continue
            out.append(draw)
            seen.add(draw)
        return np.asarray(out, dtype=int)

    def _local_neighbor_arms(
        self,
        center_arms: np.ndarray,
        *,
        include_categorical: bool = False,
        radius: int | None = None,
    ) -> np.ndarray:
        effective_radius = (
            int(self.local_neighbor_radius) if radius is None else int(radius)
        )
        if effective_radius <= 0:
            return np.zeros(0, dtype=int)

        index_of = getattr(self.actions, "index_of", None)
        if self._action_key_to_arm is None and not callable(index_of):
            return np.zeros(0, dtype=int)

        def _arm_for_action(candidate: Dict[str, Any]) -> int | None:
            if callable(index_of):
                try:
                    return int(index_of(candidate))
                except ValueError:
                    return None
            key = action_key_from_parameter_space_spec(
                candidate, self.parameter_spec
            )
            return self._action_key_to_arm.get(key)

        neighbor_parts: List[np.ndarray] = []
        for arm in np.asarray(center_arms, dtype=int):
            cache_key = (
                int(arm),
                bool(include_categorical),
                int(effective_radius),
            )
            cached = self._local_neighbor_cache.get(cache_key)
            if cached is not None:
                neighbor_parts.append(cached)
                continue

            action = dict(self.actions[int(arm)])
            neighbors: set[int] = set()
            for param in self.parameter_spec.parameters:
                if not self._is_param_active(param, action):
                    continue
                current_value = action.get(param.name, param.default)
                if param.kind == "categorical":
                    if not include_categorical:
                        continue
                    for value in param.values:
                        if value == current_value:
                            continue
                        candidate = dict(action)
                        candidate[param.name] = value
                        neighbor = _arm_for_action(candidate)
                        if neighbor is not None and int(neighbor) != int(arm):
                            neighbors.add(int(neighbor))
                    continue

                value_to_index = self._param_value_to_index[param.name]
                idx = value_to_index.get(current_value)
                if idx is None:
                    continue
                candidate_indices = set()
                for delta in range(1, effective_radius + 1):
                    for sign in (-1, 1):
                        j = int(idx) + sign * int(delta)
                        if not (0 <= j < len(param.values)):
                            continue
                        candidate_indices.add(j)
                if (
                    include_categorical
                    and param.name == "agg_num_levels"
                    and current_value != param.default
                ):
                    candidate_indices.add(
                        self._param_value_to_index[param.name][param.default]
                    )
                for candidate_index in sorted(candidate_indices):
                    candidate = dict(action)
                    candidate[param.name] = param.values[candidate_index]
                    neighbor = _arm_for_action(candidate)
                    if neighbor is not None and int(neighbor) != int(arm):
                        neighbors.add(int(neighbor))
            arm_neighbors = np.asarray(sorted(neighbors), dtype=int)
            arm_neighbors.setflags(write=False)
            if len(self._local_neighbor_cache) >= int(
                self._local_neighbor_cache_max_size
            ):
                self._local_neighbor_cache.pop(
                    next(iter(self._local_neighbor_cache))
                )
            self._local_neighbor_cache[cache_key] = arm_neighbors
            neighbor_parts.append(arm_neighbors)
        if not neighbor_parts:
            return np.zeros(0, dtype=int)
        nonempty = [part for part in neighbor_parts if part.size]
        if not nonempty:
            return np.zeros(0, dtype=int)
        return np.unique(np.concatenate(nonempty)).astype(int, copy=False)

    def _adaptive_candidate_subset(self, M: int) -> Tuple[np.ndarray, Dict[str, int | str]]:
        if M >= self.K:
            all_arms = np.arange(self.K, dtype=int)
            return all_arms, {"strategy": "adaptive_local", "candidate_count": int(self.K), "elite": 0, "local": 0, "global": int(self.K)}

        selected: List[int] = []
        selected_set = set()

        def _append_unique(values: np.ndarray | Sequence[int]) -> None:
            for value in np.asarray(values, dtype=int).reshape(-1):
                arm = int(value)
                if arm not in selected_set:
                    selected.append(arm)
                    selected_set.add(arm)

        always_include = self._cand.always_include_arms
        incumbent = int(self._incumbent_arm())
        elite = self._cand.elite_arms

        _append_unique(always_include)
        _append_unique([incumbent])

        local_seed = np.asarray(selected + [int(a) for a in elite[: max(1, min(4, elite.size))]], dtype=int)
        local_pool = self._local_neighbor_arms(local_seed)

        target_elite = int(round(float(M) * self.candidate_elite_fraction))
        target_local = int(round(float(M) * self.candidate_local_fraction))

        elite_extra = np.setdiff1d(elite, np.asarray(selected, dtype=int), assume_unique=False)
        elite_pick = self._sample_without_replacement(elite_extra, max(0, target_elite))
        _append_unique(elite_pick)

        local_extra = np.setdiff1d(local_pool, np.asarray(selected, dtype=int), assume_unique=False)
        local_pick = self._sample_without_replacement(local_extra, max(0, target_local))
        _append_unique(local_pick)

        remaining = int(M) - len(selected)
        global_pick = self._sample_global_excluding(np.asarray(selected, dtype=int), remaining)
        _append_unique(global_pick)

        candidate = np.asarray(selected, dtype=int)
        if candidate.size > int(M):
            candidate = candidate[: int(M)]

        stats = {
            "strategy": "adaptive_local",
            "candidate_count": int(candidate.size),
            "elite": int(elite_pick.size),
            "local": int(local_pick.size),
            "global": int(global_pick.size),
        }
        return candidate, stats

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        if self._initial_guess_arm is not None and self.t < self.initial_guess_rounds:
            arm = int(self._initial_guess_arm)
            self._last_phi = self._phi(x, arm)
            self._last_arm = arm
            self.history.append(
                SharedLinUCBv4Step(
                    t=self.t + 1,
                    arm_index=arm,
                    loss=float("nan"),
                    pred_mean=float("nan"),
                    pred_uncert=float("nan"),
                )
            )
            self.candidate_stats_history.append(
                {"strategy": "initial_guess", "candidate_count": 1, "elite": 0, "local": 0, "global": 1}
            )
            return dict(self.actions[arm])

        alpha_eff = self._effective_alpha()
        M = self._effective_candidate_pool_size()
        candidate_features: Optional[np.ndarray] = None
        if M is None or int(M) >= self.K:
            score, mean, uncert = self._selection_score_subset(
                x, arms=None, alpha=alpha_eff
            )
            best_score = float(np.min(score))
            best_arms = np.flatnonzero(score <= best_score + 1e-12)
            chosen = int(self.rng.choice(best_arms))
            arm = chosen
            best_mean = float(mean[chosen])
            best_unc = float(uncert[chosen])
            self.candidate_stats_history.append(
                {"strategy": "full", "candidate_count": int(self.K), "elite": 0, "local": 0, "global": int(self.K)}
            )
        else:
            candidate_features = None
            if self._candidate_schedule is not None:
                if self.candidate_strategy == "structured":
                    cand, candidate_features, cand_stats = (
                        self._structured_candidate_subset(int(M))
                    )
                else:
                    cand, candidate_features = self._scheduled_candidate_subset(
                        int(M)
                    )
                    cand_stats = {
                        "strategy": "uniform_aot",
                        "candidate_count": int(cand.size),
                        "elite": int(self._cand.elite_arms.size),
                        "local": 0,
                        "global": int(cand.size),
                    }
            elif self.candidate_strategy == "adaptive_local":
                cand, cand_stats = self._adaptive_candidate_subset(int(M))
            else:
                cand = self._cand.candidate_subset(pool_size=int(M))
                cand_stats = {
                    "strategy": "uniform",
                    "candidate_count": int(cand.size),
                    "elite": int(self._cand.elite_arms.size),
                    "local": 0,
                    "global": int(cand.size),
                }

            score, mean, uncert = self._selection_score_subset(
                x,
                arms=cand,
                action_features=candidate_features,
                alpha=alpha_eff,
            )
            best_score = float(np.min(score))
            best_loc = np.flatnonzero(score <= best_score + 1e-12)
            loc = int(self.rng.choice(best_loc))
            arm = int(cand[loc])
            best_mean = float(mean[loc])
            best_unc = float(uncert[loc])
            self.candidate_stats_history.append(cand_stats)

        self._last_phi = (
            self._phi_from_feature(x, candidate_features[loc])
            if M is not None
            and int(M) < self.K
            and self._candidate_schedule is not None
            else self._phi(x, arm)
        )
        if not np.isfinite(best_unc):
            best_unc = float(
                np.sqrt(
                    max(
                        0.0,
                        float(self._last_phi @ (self.A_inv @ self._last_phi)),
                    )
                )
            )
        self._last_arm = arm

        self.history.append(
            SharedLinUCBv4Step(
                t=self.t + 1,
                arm_index=arm,
                loss=float("nan"),
                pred_mean=best_mean,
                pred_uncert=best_unc,
            )
        )
        return dict(self.actions[arm])

    def recommend(
        self,
        context: Iterable[float],
        *,
        candidate_arms: Sequence[int],
        alpha: float = 0.0,
    ) -> Tuple[Dict[str, Any], Dict[str, float | int]]:
        """Score a fixed arm set without creating a pending online update."""
        x = self._validate_x(np.asarray(list(context), dtype=float))
        arms = np.asarray(candidate_arms, dtype=int).reshape(-1)
        if arms.size == 0:
            raise ValueError("candidate_arms must be non-empty")
        if np.any(arms < 0) or np.any(arms >= self.K):
            raise ValueError("candidate_arms contains an out-of-range arm")
        arms = np.unique(arms)
        score, mean, uncert = self._score_subset(x, arms=arms, alpha=float(alpha))
        location = int(np.argmin(score))
        arm = int(arms[location])
        return dict(self.actions[arm]), {
            "arm_index": arm,
            "pred_mean": float(mean[location]),
            "pred_uncert": float(uncert[location]),
        }

    def select_recovery(
        self,
        context: Iterable[float],
        *,
        excluded_arms: Sequence[int],
    ) -> Tuple[Dict[str, Any], Dict[str, float | int]]:
        """Select another setup using failure-UCB then runtime-LCB.

        This method is only used after a setup construction failure. Exact
        failed arms are excluded for the current instance; the exclusion is
        supplied by the caller and is never retained by the model.
        """

        if self._last_phi is not None or self._last_arm is not None:
            raise RuntimeError("select_recovery() called with a pending action")
        x = self._validate_x(np.asarray(list(context), dtype=float))
        excluded = np.unique(np.asarray(excluded_arms, dtype=int).reshape(-1))
        if np.any(excluded < 0) or np.any(excluded >= self.K):
            raise ValueError("excluded_arms contains an out-of-range arm")

        M = self._effective_candidate_pool_size()
        candidate_features: Optional[np.ndarray] = None
        if M is None or int(M) >= self.K:
            candidates = np.setdiff1d(
                np.arange(self.K, dtype=int), excluded, assume_unique=True
            )
            candidate_stats: Dict[str, int | str] = {
                "strategy": "failure_lcb_reselection_full",
                "candidate_count": int(candidates.size),
                "elite": 0,
                "local": 0,
                "global": int(candidates.size),
                "excluded": int(excluded.size),
            }
        else:
            if self._candidate_schedule is not None:
                if self.candidate_strategy == "structured":
                    sampled, candidate_features, candidate_stats = (
                        self._structured_candidate_subset(int(M))
                    )
                    candidate_stats = dict(candidate_stats)
                    candidate_stats["strategy"] = (
                        "failure_lcb_reselection_structured_aot"
                    )
                else:
                    sampled, candidate_features = self._scheduled_candidate_subset(
                        int(M)
                    )
                    candidate_stats = {
                        "strategy": "failure_lcb_reselection_uniform_aot",
                        "candidate_count": int(sampled.size),
                        "elite": int(self._cand.elite_arms.size),
                        "local": 0,
                        "global": int(sampled.size),
                    }
            elif self.candidate_strategy == "adaptive_local":
                sampled, candidate_stats = self._adaptive_candidate_subset(int(M))
            else:
                sampled = self._cand.candidate_subset(pool_size=int(M))
                candidate_stats = {
                    "strategy": "failure_lcb_reselection_uniform",
                    "candidate_count": int(sampled.size),
                    "elite": int(self._cand.elite_arms.size),
                    "local": 0,
                    "global": int(sampled.size),
                }
            keep = ~np.isin(sampled, excluded)
            candidates = sampled[keep]
            if candidate_features is not None:
                candidate_features = candidate_features[keep]
            if candidates.size == 0:
                candidates = self._sample_global_excluding(
                    excluded,
                    min(int(M), self.K - int(excluded.size)),
                )
                candidate_features = self._features_for_arms(candidates)
            candidate_stats = dict(candidate_stats)
            candidate_stats["candidate_count"] = int(candidates.size)
            candidate_stats["excluded"] = int(excluded.size)

        if candidates.size == 0:
            raise RuntimeError("No setup arm remains after temporary exclusions")

        runtime_score, runtime_mean, runtime_uncert = self._score_subset(
            x,
            arms=candidates,
            action_features=candidate_features,
            alpha=self._effective_alpha(),
        )
        shared_uncertainty = (
            runtime_uncert if self.failure_A_inv is self.A_inv else None
        )
        failure_upper, failure_mean, failure_uncert = self._failure_subset(
            x,
            arms=candidates,
            action_features=candidate_features,
            precomputed_uncertainty=shared_uncertainty,
        )
        order = np.lexsort((runtime_score, failure_upper))
        best_location = int(order[0])
        best_failure = float(failure_upper[best_location])
        best_runtime = float(runtime_score[best_location])
        tied = np.flatnonzero(
            np.isclose(failure_upper, best_failure, atol=1.0e-12, rtol=0.0)
            & np.isclose(runtime_score, best_runtime, atol=1.0e-12, rtol=0.0)
        )
        if tied.size > 1:
            best_location = int(self.rng.choice(tied))
        arm = int(candidates[best_location])

        self._last_phi = (
            self._phi_from_feature(x, candidate_features[best_location])
            if candidate_features is not None
            else self._phi(x, arm)
        )
        self._last_arm = arm
        self.history.append(
            SharedLinUCBv4Step(
                t=self.t + 1,
                arm_index=arm,
                loss=float("nan"),
                pred_mean=float(runtime_mean[best_location]),
                pred_uncert=float(runtime_uncert[best_location]),
                pred_failure_mean=float(failure_mean[best_location]),
                pred_failure_uncert=float(failure_uncert[best_location]),
            )
        )
        self.candidate_stats_history.append(candidate_stats)
        return dict(self.actions[arm]), {
            "arm_index": arm,
            "pred_mean": float(runtime_mean[best_location]),
            "pred_uncert": float(runtime_uncert[best_location]),
            "pred_failure_mean": float(failure_mean[best_location]),
            "pred_failure_uncert": float(failure_uncert[best_location]),
            "failure_ucb": float(failure_upper[best_location]),
            "runtime_lcb": float(runtime_score[best_location]),
        }

    @staticmethod
    def _rank_one_inverse_update(inverse: np.ndarray, phi: np.ndarray) -> None:
        projected = inverse @ phi
        denominator = 1.0 + float(phi @ projected)
        if denominator <= 0.0 or not np.isfinite(denominator):
            rebuilt = np.linalg.inv(inverse)
            rebuilt += np.outer(phi, phi)
            inverse[:] = np.linalg.inv(rebuilt)
            return
        inverse -= np.outer(projected, projected) / denominator

    def _after_runtime_precision_update(self, phi: np.ndarray) -> None:
        """Hook for learners that maintain another precision factorization."""

        del phi

    def begin_recovery_transaction(self) -> None:
        """Snapshot learnable state after selection but before a failed update.

        RNG state is intentionally omitted: attempted selections remain consumed
        even when a double failure rolls the learner back.
        """

        if self._recovery_transaction is not None:
            raise RuntimeError("A LinUCB recovery transaction is already active")
        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("Recovery transaction requires a pending action")
        candidate = self._cand
        self._recovery_transaction = {
            "A_inv": self.A_inv.copy(),
            "failure_A_inv": (
                None
                if self.failure_A_inv is self.A_inv
                else self.failure_A_inv.copy()
            ),
            "b": self.b.copy(),
            "failure_b": self.failure_b.copy(),
            "failure_observation_count": int(self.failure_observation_count),
            "t": int(self.t),
            "history_length": max(0, len(self.history) - 1),
            "candidate_history_length": max(
                0, len(self.candidate_stats_history) - 1
            ),
            "elite_arms": candidate._elite_arms.copy(),
            "candidate_statistics": candidate.export_statistics(),
        }

    def observe_pending_failure(
        self,
        *,
        failure_label: float,
    ) -> SharedLinUCBv4DeferredObservation:
        """Update coverage/failure risk now and defer the suffix runtime label."""

        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("observe_pending_failure() called before selection")
        label = float(failure_label)
        if not np.isfinite(label) or not (0.0 <= label <= 1.0):
            raise ValueError("failure_label must be finite and in [0, 1]")

        phi = self._last_phi.copy()
        arm = int(self._last_arm)
        history_index = len(self.history) - 1
        provisional_loss = float(phi @ (self.A_inv @ self.b))

        self._rank_one_inverse_update(self.A_inv, phi)
        self._after_runtime_precision_update(phi)
        if self.failure_A_inv is not self.A_inv:
            self._rank_one_inverse_update(self.failure_A_inv, phi)
        # The provisional label preserves the runtime mean exactly while the
        # failure head can immediately influence same-context reselection.
        self.b += provisional_loss * phi
        self.failure_b += label * phi
        self.failure_observation_count += 1
        self.t += 1

        last = self.history[history_index]
        self.history[history_index] = SharedLinUCBv4Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=last.loss,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
            failure_label=label,
            pred_failure_mean=last.pred_failure_mean,
            pred_failure_uncert=last.pred_failure_uncert,
        )
        self._last_phi = None
        self._last_arm = None
        return SharedLinUCBv4DeferredObservation(
            arm_index=arm,
            phi=phi,
            provisional_loss=provisional_loss,
            history_index=history_index,
        )

    def commit_deferred_observation(
        self,
        observation: SharedLinUCBv4DeferredObservation,
        *,
        loss: float,
    ) -> None:
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")
        self.b += (y - float(observation.provisional_loss)) * observation.phi
        self._cand.observe(int(observation.arm_index), y)
        index = int(observation.history_index)
        last = self.history[index]
        self.history[index] = SharedLinUCBv4Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
            failure_label=last.failure_label,
            pred_failure_mean=last.pred_failure_mean,
            pred_failure_uncert=last.pred_failure_uncert,
        )

    def commit_recovery_transaction(self) -> None:
        self._recovery_transaction = None

    def rollback_recovery_transaction(self) -> None:
        state = self._recovery_transaction
        if state is None:
            self.cancel_pending()
            return
        self.A_inv[:] = np.asarray(state["A_inv"], dtype=float)
        if self.failure_A_inv is not self.A_inv:
            self.failure_A_inv[:] = np.asarray(
                state["failure_A_inv"], dtype=float
            )
        self.b[:] = np.asarray(state["b"], dtype=float)
        self.failure_b[:] = np.asarray(state["failure_b"], dtype=float)
        self.failure_observation_count = int(state["failure_observation_count"])
        self.t = int(state["t"])
        del self.history[int(state["history_length"]) :]
        del self.candidate_stats_history[
            int(state["candidate_history_length"]) :
        ]
        candidate = self._cand
        candidate._elite_arms = np.asarray(state["elite_arms"], dtype=int).copy()
        candidate.restore_statistics(state["candidate_statistics"])
        self._last_phi = None
        self._last_arm = None
        self._recovery_transaction = None

    def update(self, loss: float, *, failure_label: float = 0.0) -> None:
        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")

        phi = self._last_phi
        arm = int(self._last_arm)
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")
        label = float(failure_label)
        if not np.isfinite(label) or not (0.0 <= label <= 1.0):
            raise ValueError("failure_label must be finite and in [0, 1]")

        self._cand.observe(arm, y)
        self._rank_one_inverse_update(self.A_inv, phi)
        self._after_runtime_precision_update(phi)
        if self.failure_A_inv is not self.A_inv:
            self._rank_one_inverse_update(self.failure_A_inv, phi)
        self.b = self.b + y * phi
        self.failure_b = self.failure_b + label * phi
        self.failure_observation_count += 1

        last = self.history[-1]
        self.history[-1] = SharedLinUCBv4Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
            failure_label=label,
            pred_failure_mean=last.pred_failure_mean,
            pred_failure_uncert=last.pred_failure_uncert,
        )

        self.t += 1
        self._last_phi = None
        self._last_arm = None

    def cancel_pending(self) -> None:
        """Discard a selected arm when its solve outcome was never observed."""
        if self._last_phi is None or self._last_arm is None:
            return
        if self.history and not np.isfinite(float(self.history[-1].loss)):
            self.history.pop()
        if len(self.candidate_stats_history) > len(self.history):
            self.candidate_stats_history.pop()
        self._last_phi = None
        self._last_arm = None
