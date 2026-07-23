from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

try:
    from scipy.linalg.blas import dger as _blas_dger
except ImportError:  # pragma: no cover - NumPy fallback keeps SciPy optional.
    _blas_dger = None

from SolvePhase.algorithms.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    build_action_basis,
    joint_action_features,
)


@dataclass(frozen=True)
class BootstrapSarsaSpec:
    members: int = 5
    episode_inclusion_probability: float = 0.8
    uncertainty_beta: float = 1.0


@dataclass(frozen=True)
class StagewiseLsviLcbSpec:
    horizon: int = 50
    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    refit_interval_episodes: int = 1


@dataclass(frozen=True)
class RecursiveMonteCarloLcbSpec:
    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    episode_half_life: float = 500.0


@dataclass(frozen=True)
class RecursiveLstdqLcbSpec:
    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    inverse_denominator_floor: float = 1.0e-10
    lcb_lower_bound_sec: float | None = 0.0


@dataclass(frozen=True)
class RecursiveLstdqV2LcbSpec(RecursiveLstdqLcbSpec):
    """Coverage-calibrated confidence controls for recursive LSTDQ."""

    coverage_ridge: float = 1.0
    residual_scale_window: int = 2048
    residual_scale_min_samples: int = 32


@dataclass(frozen=True)
class RecursiveBlstdqSpec:
    """Bayesian LSTDQ posterior controls for RBLSPI-style exploration."""

    prior_precision: float = 1.0e4
    noise_precision: float = 1.0e6
    gram_ridge: float = 1.0e-6


@dataclass(frozen=True)
class StructuredModelBasedSpec:
    """Shared RLS controls for physical cycle-cost/progress prediction."""

    ridge: float = 1.0
    minimum_samples: int = 32
    scale_window: int = 2048
    cost_floor_fraction: float = 0.1
    progress_floor_fraction: float = 0.1
    numerical_floor: float = 1.0e-12


@dataclass(frozen=True)
class HierarchicalLsviLcbSpec:
    """Stagewise LSVI with a shared cross-stage ridge prior."""

    horizon: int = 50
    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    refit_interval_episodes: int = 100
    refit_sweeps: int = 3
    residual_shrinkage_samples: float = 32.0


_RECURSIVE_MC_CHECKPOINT_VERSION = 2
_RECURSIVE_LSTDQ_CHECKPOINT_VERSION = 2
_RECURSIVE_LSTDQ_V2_CHECKPOINT_VERSION = 1
_RECURSIVE_BLSTDQ_CHECKPOINT_VERSION = 1
_STRUCTURED_MODEL_BASED_CHECKPOINT_VERSION = 1
_HIERARCHICAL_LSVI_CHECKPOINT_VERSION = 1


def _json_dataclass(value: Any) -> Dict[str, Any]:
    return json.loads(json.dumps(asdict(value)))


def _greedy_cost_index(
    values: np.ndarray,
    *,
    allowed_indices: np.ndarray,
    anchor_index: int,
    secondary_values: np.ndarray | None = None,
) -> int:
    allowed = np.asarray(allowed_indices, dtype=int)
    if allowed.size == 0:
        raise ValueError("At least one action must be allowed")
    best_value = float(np.min(np.asarray(values, dtype=float)[allowed]))
    tied = allowed[
        np.isclose(values[allowed], best_value, atol=1.0e-12, rtol=0.0)
    ]
    if secondary_values is not None and tied.size > 1:
        secondary = np.asarray(secondary_values, dtype=float)
        best_secondary = float(np.min(secondary[tied]))
        tied = tied[
            np.isclose(
                secondary[tied],
                best_secondary,
                atol=1.0e-12,
                rtol=0.0,
            )
        ]
    if np.any(tied == int(anchor_index)):
        return int(anchor_index)
    return int(tied[0])


def _sandwich_quadratic(
    projected_features: np.ndarray,
    moment_covariance: np.ndarray,
) -> np.ndarray:
    """Evaluate one sandwich-covariance quadratic form per action."""

    projected = np.asarray(projected_features, dtype=float)
    covariance = np.asarray(moment_covariance, dtype=float)
    return np.sum((projected @ covariance) * projected, axis=1)


class _RollingFloatWindow:
    """Fixed-size numeric rolling window with allocation-free appends."""

    def __init__(self, maxlen: int) -> None:
        if int(maxlen) <= 0:
            raise ValueError("rolling window length must be positive")
        self.maxlen = int(maxlen)
        self._values = np.empty(self.maxlen, dtype=float)
        self._size = 0
        self._cursor = 0

    def __len__(self) -> int:
        return int(self._size)

    def __iter__(self):
        if self._size < self.maxlen:
            yield from self._values[: self._size]
            return
        yield from self._values[self._cursor :]
        yield from self._values[: self._cursor]

    def append(self, value: float) -> None:
        self._values[self._cursor] = float(value)
        self._cursor = (self._cursor + 1) % self.maxlen
        self._size = min(self._size + 1, self.maxlen)

    def extend(self, values: Iterable[float]) -> None:
        for value in values:
            self.append(float(value))

    def clear(self) -> None:
        self._size = 0
        self._cursor = 0

    def as_array(self, *, chronological: bool) -> np.ndarray:
        """Expose current values; order is optional for symmetric statistics."""

        if not chronological or self._size < self.maxlen:
            return self._values[: self._size]
        if self._cursor == 0:
            return self._values
        return np.concatenate(
            (self._values[self._cursor :], self._values[: self._cursor])
        )


def _partition_median(values: np.ndarray) -> float:
    """Compute NumPy's finite-sample median with one partial ordering."""

    size = int(values.size)
    if size == 0:
        raise ValueError("median requires at least one value")
    middle = size // 2
    if size % 2:
        return float(np.partition(values, middle)[middle])
    partitioned = np.partition(values, (middle - 1, middle))
    return float((partitioned[middle - 1] + partitioned[middle]) / 2.0)


def _mad_scale(values: Sequence[float], *, floor: float) -> float:
    """Return a Gaussian-consistent MAD scale with a physical lower floor."""

    samples = np.asarray(values, dtype=float)
    if samples.size == 0:
        return float(floor)
    center = _partition_median(samples)
    mad = _partition_median(np.abs(samples - center))
    return max(1.4826 * mad, float(floor))


def _rank_one_inverse_update(
    inverse: np.ndarray,
    feature: np.ndarray,
    *,
    denominator_floor: float,
) -> bool:
    """Apply a symmetric Sherman-Morrison update in place.

    Returns ``False`` if the denominator is invalid so the caller can rebuild
    from its explicitly maintained design matrix.
    """

    projected = inverse @ feature
    denominator = 1.0 + float(feature @ projected)
    if not np.isfinite(denominator) or denominator <= float(denominator_floor):
        return False
    if _blas_dger is not None and inverse.flags.f_contiguous:
        updated = _blas_dger(
            -1.0 / denominator,
            projected,
            projected,
            a=inverse,
            overwrite_a=1,
        )
        if not np.shares_memory(updated, inverse):
            inverse[:] = updated
    else:
        inverse -= np.outer(projected, projected) / denominator
    return True


def _rank_one_accumulate(
    matrix: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> None:
    """Accumulate ``left @ right.T`` without changing the update equation."""

    if _blas_dger is not None and matrix.flags.f_contiguous:
        updated = _blas_dger(
            1.0,
            left,
            right,
            a=matrix,
            overwrite_a=1,
        )
        if not np.shares_memory(updated, matrix):
            matrix[:] = updated
    else:
        matrix += np.outer(left, right)


def _rank_one_inverse_accumulate(
    inverse: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    denominator: float,
) -> None:
    """Apply the nonsymmetric LSTD Sherman--Morrison correction in place."""

    if _blas_dger is not None and inverse.flags.f_contiguous:
        updated = _blas_dger(
            -1.0 / float(denominator),
            left,
            right,
            a=inverse,
            overwrite_a=1,
        )
        if not np.shares_memory(updated, inverse):
            inverse[:] = updated
    else:
        inverse -= np.outer(left, right) / float(denominator)


class _SharedActionLcbController:
    """Common action representation and behavior policy for linear LCB control."""

    prefer_mean_on_score_ties = False

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        seed: int,
    ) -> None:
        self.config = config
        self.feature_dim = int(feature_dim)
        self.weights = np.asarray(config.weights, dtype=float)
        self.action_basis = build_action_basis(
            self.weights,
            mode=str(config.action_basis_mode),
            rbf_sigma=float(config.action_rbf_sigma),
            rbf_centers=tuple(config.action_basis_centers),
        )
        self.joint_dim = int(self.action_basis.shape[1] * self.feature_dim)
        self.rng = np.random.default_rng(int(seed))
        self.anchor_index = int(
            np.argmin(np.abs(self.weights - float(config.anchor_weight)))
        )
        self._all_action_indices = np.arange(self.weights.size, dtype=int)
        self._initial_environment_weight = float(config.anchor_weight)
        self._default_first_action_pending = bool(config.force_default_first_action)
        self.forced_initial_action_count = 0
        self.steps = 0
        self.episodes = 0

    @property
    def epsilon(self) -> float:
        decay = max(float(self.config.epsilon_decay_steps), 1.0)
        fraction = math.exp(-float(self.steps) / decay)
        return float(
            self.config.epsilon_final
            + (self.config.epsilon_start - self.config.epsilon_final) * fraction
        )

    @property
    def initial_environment_weight(self) -> float:
        return float(self._initial_environment_weight)

    def environment_weight(self, action_index: int, last_weight: float) -> float:
        del last_weight
        return float(self.weights[int(action_index)])

    def state_action_features(self, features: np.ndarray) -> np.ndarray:
        return joint_action_features(self.action_basis, features)

    def state_action_feature(
        self,
        features: np.ndarray,
        action_index: int,
        *,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the joint feature vector for one selected action only."""

        if out is None:
            return joint_action_features(
                self.action_basis,
                features,
                action_indices=int(action_index),
            )[0]
        state = np.asarray(features, dtype=float)
        target = np.asarray(out, dtype=float)
        if state.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected {self.feature_dim} state features, got {state.shape}"
            )
        if target.shape != (self.joint_dim,):
            raise ValueError(
                f"Expected {self.joint_dim} output entries, got {target.shape}"
            )
        index = int(action_index)
        if not 0 <= index < self.action_basis.shape[0]:
            raise IndexError("action index is outside the action basis")
        np.multiply(
            self.action_basis[index, :, None],
            state[None, :],
            out=target.reshape(self.action_basis.shape[1], self.feature_dim),
        )
        return target

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError

    def q_values(self, features: np.ndarray, *, cycle: int | None = None) -> np.ndarray:
        return self._values(features, cycle=cycle)[0]

    def select_action(
        self,
        features: np.ndarray,
        *,
        explore: bool,
        epsilon: float | None = None,
        cycle: int | None = None,
        include_metadata: bool = True,
    ) -> tuple[int, float, Dict[str, Any]]:
        means, uncertainty, scores = self._values(features, cycle=cycle)
        greedy_index = _greedy_cost_index(
            scores,
            allowed_indices=self._all_action_indices,
            anchor_index=self.anchor_index,
            secondary_values=(
                means if self.prefer_mean_on_score_ties else None
            ),
        )
        effective_epsilon = (
            (self.epsilon if epsilon is None else float(epsilon)) if explore else 0.0
        )
        forced = bool(explore and self._default_first_action_pending)
        if forced:
            action_index = int(
                np.argmin(
                    np.abs(self.weights - float(self.initial_environment_weight))
                )
            )
            if not np.isclose(
                self.weights[action_index],
                self.initial_environment_weight,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError("The action space cannot represent the solver default")
            self._default_first_action_pending = False
            self.forced_initial_action_count += 1
            explored = False
            effective_epsilon = 0.0
        else:
            explored = bool(
                effective_epsilon > 0.0
                and float(self.rng.random()) < effective_epsilon
            )
            action_index = int(
                self.rng.choice(self._all_action_indices)
                if explored
                else greedy_index
            )
        metadata: Dict[str, Any] = {}
        if include_metadata:
            metadata = {
                "epsilon": float(effective_epsilon),
                "greedy_index": int(greedy_index),
                "explored": bool(explored),
                "allowed_indices": self._all_action_indices.tolist(),
                "forced_default_first_action": forced,
                "uncertainty": uncertainty.tolist(),
                "q_values": means.tolist(),
                "selection_scores": scores.tolist(),
            }
        return action_index, float(self.weights[action_index]), metadata

    def _load_common(self, payload: Any, *, restore_counters: bool) -> None:
        if restore_counters:
            self.steps = int(payload["steps"].item())
            self.episodes = int(payload["episodes"].item())
        self._default_first_action_pending = bool(
            self.config.force_default_first_action
            and payload["default_first_action_pending"].item()
        )
        self.forced_initial_action_count = int(
            payload["forced_initial_action_count"].item()
        )
        self.rng.bit_generator.state = json.loads(str(payload["rng_state"].item()))

    def _common_checkpoint_fields(self) -> Dict[str, np.ndarray]:
        return {
            "steps": np.asarray(self.steps, dtype=np.int64),
            "episodes": np.asarray(self.episodes, dtype=np.int64),
            "default_first_action_pending": np.asarray(
                self._default_first_action_pending, dtype=np.bool_
            ),
            "forced_initial_action_count": np.asarray(
                self.forced_initial_action_count, dtype=np.int64
            ),
            "config": np.asarray(json.dumps(asdict(self.config))),
            "rng_state": np.asarray(json.dumps(self.rng.bit_generator.state)),
        }

    def _snapshot_common_learning_state(self) -> Dict[str, Any]:
        return {
            "steps": int(self.steps),
            "episodes": int(self.episodes),
            "initial_environment_weight": float(self._initial_environment_weight),
            "default_first_action_pending": bool(self._default_first_action_pending),
            "forced_initial_action_count": int(self.forced_initial_action_count),
        }

    def _restore_common_learning_state(self, state: Dict[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.episodes = int(state["episodes"])
        self._initial_environment_weight = float(
            state["initial_environment_weight"]
        )
        self._default_first_action_pending = bool(
            state["default_first_action_pending"]
        )
        self.forced_initial_action_count = int(
            state["forced_initial_action_count"]
        )


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
        self.last_postfit_td_error = 0.0

    @property
    def requires_next_action(self) -> bool:
        return True

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        self.trace.fill(0.0)
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)

    def _resync_theta(self) -> None:
        """Restore the exact normal-equation solution at an episode boundary."""

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
        projected_trace = self.a_inverse @ self.trace
        projected_difference = difference @ self.a_inverse
        denominator = 1.0 + float(difference @ projected_trace)
        _rank_one_accumulate(self.a_matrix, self.trace, difference)
        self.b += float(cost) * self.trace
        if (
            not np.isfinite(denominator)
            or abs(denominator) < float(self.spec.inverse_denominator_floor)
        ):
            self.a_inverse = np.linalg.pinv(self.a_matrix)
            self.inverse_rebuild_count += 1
            self._resync_theta()
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
                self._resync_theta()
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
            self._resync_theta()
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
            if version != _RECURSIVE_LSTDQ_CHECKPOINT_VERSION:
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
            "stored_transition_count": 0,
        }


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
        self._coverage_basis_dim = int(self.action_basis.shape[1])
        self._coverage_state_lift = np.zeros(
            (self._coverage_basis_dim, self.joint_dim),
            dtype=float,
        )
        self._coverage_state_lift_blocks = self._coverage_state_lift.reshape(
            self._coverage_basis_dim,
            self._coverage_basis_dim,
            self.feature_dim,
        )
        self._coverage_basis_indices = np.arange(
            self._coverage_basis_dim,
            dtype=int,
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
        state = np.asarray(features, dtype=float)
        if state.shape != (self.feature_dim,):
            raise ValueError(
                f"Expected {self.feature_dim} state features, got {state.shape}"
            )

        # Every joint feature is z_a = b_a (x) state.  Contracting the
        # 288-dimensional quadratic through the 9-dimensional action basis is
        # algebraically identical to forming all 41 joint feature vectors, but
        # avoids the 41 x 288 by 288 dense product on every solver cycle.
        self._coverage_state_lift.fill(0.0)
        self._coverage_state_lift_blocks[
            self._coverage_basis_indices,
            self._coverage_basis_indices,
            :,
        ] = state
        reduced_coverage = (
            self._coverage_state_lift @ self.coverage_inverse
        ) @ self._coverage_state_lift.T
        state_parameters = self.theta.reshape(
            self._coverage_basis_dim,
            self.feature_dim,
        ) @ state
        means = self.action_basis @ state_parameters
        quadratic = np.sum(
            (self.action_basis @ reduced_coverage) * self.action_basis,
            axis=1,
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
            if version != _RECURSIVE_LSTDQ_V2_CHECKPOINT_VERSION:
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
            "coverage_inverse_rebuild_count": int(
                self.coverage_inverse_rebuild_count
            ),
            "stored_transition_count": 0,
        }


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


class BootstrapLcbSarsaController:
    """Bootstrap ensemble of shared-feature true-online SARSA learners."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: BootstrapSarsaSpec,
        seed: int,
        initial_parameters: np.ndarray | None = None,
    ) -> None:
        if config.td_algorithm != "true_online_sarsa":
            raise ValueError("bootstrap SARSA requires true_online_sarsa")
        if int(spec.members) < 2:
            raise ValueError("bootstrap SARSA requires at least two members")
        if not 0.0 < float(spec.episode_inclusion_probability) <= 1.0:
            raise ValueError("episode inclusion probability must be in (0, 1]")
        if float(spec.uncertainty_beta) < 0.0:
            raise ValueError("uncertainty_beta must be non-negative")

        self.config = config
        self.spec = spec
        self.feature_dim = int(feature_dim)
        self.weights = np.asarray(config.weights, dtype=float)
        self.rng = np.random.default_rng(int(seed))
        member_config = replace(config, force_default_first_action=False)
        seed_sequence = np.random.SeedSequence(int(seed)).spawn(int(spec.members))
        self.members = [
            ExpectedSarsaLambda(
                feature_dim=self.feature_dim,
                config=member_config,
                seed=int(child.generate_state(1, dtype=np.uint32)[0]),
                initial_parameters=initial_parameters,
            )
            for child in seed_sequence
        ]
        first = self.members[0]
        self.action_basis = first.action_basis
        self.anchor_index = int(first.anchor_index)
        self._all_action_indices = np.arange(self.weights.size, dtype=int)
        self._initial_environment_weight = float(config.anchor_weight)
        self._default_first_action_pending = bool(config.force_default_first_action)
        self.forced_initial_action_count = 0
        self.steps = 0
        self.episodes = 0
        self._active_members = np.ones(len(self.members), dtype=bool)

    @property
    def epsilon(self) -> float:
        decay = max(float(self.config.epsilon_decay_steps), 1.0)
        fraction = math.exp(-float(self.steps) / decay)
        return float(
            self.config.epsilon_final
            + (self.config.epsilon_start - self.config.epsilon_final) * fraction
        )

    @property
    def requires_next_action(self) -> bool:
        return True

    @property
    def initial_environment_weight(self) -> float:
        return float(self._initial_environment_weight)

    def environment_weight(self, action_index: int, last_weight: float) -> float:
        del last_weight
        return float(self.weights[int(action_index)])

    def allowed_action_indices(self, cycle: int | None) -> np.ndarray:
        del cycle
        return self._all_action_indices

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)
        for member in self.members:
            member.start_episode(
                initial_environment_weight=self._initial_environment_weight,
            )
        self._active_members = self.rng.random(len(self.members)) < float(
            self.spec.episode_inclusion_probability
        )
        if not np.any(self._active_members):
            self._active_members[int(self.rng.integers(len(self.members)))] = True

    def member_q_values(self, features: np.ndarray) -> np.ndarray:
        return np.vstack([member.q_values(features) for member in self.members])

    def q_values(self, features: np.ndarray) -> np.ndarray:
        return np.mean(self.member_q_values(features), axis=0)

    def state_action_features(self, features: np.ndarray) -> np.ndarray:
        return self.members[0].state_action_features(features)

    def select_action(
        self,
        features: np.ndarray,
        *,
        explore: bool,
        epsilon: float | None = None,
        cycle: int | None = None,
        include_metadata: bool = True,
    ) -> tuple[int, float, Dict[str, Any]]:
        del cycle
        member_values = self.member_q_values(features)
        means = np.mean(member_values, axis=0)
        uncertainty = np.std(member_values, axis=0, ddof=1)
        scores = means - float(self.spec.uncertainty_beta) * uncertainty
        greedy_index = _greedy_cost_index(
            scores,
            allowed_indices=self._all_action_indices,
            anchor_index=self.anchor_index,
        )
        effective_epsilon = (
            (self.epsilon if epsilon is None else float(epsilon)) if explore else 0.0
        )
        forced = bool(explore and self._default_first_action_pending)
        if forced:
            action_index = int(
                np.argmin(
                    np.abs(self.weights - float(self.initial_environment_weight))
                )
            )
            if not np.isclose(
                self.weights[action_index],
                self.initial_environment_weight,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError("The action space cannot represent the solver default")
            self._default_first_action_pending = False
            self.forced_initial_action_count += 1
            explored = False
            effective_epsilon = 0.0
        else:
            explored = bool(
                effective_epsilon > 0.0
                and float(self.rng.random()) < effective_epsilon
            )
            action_index = int(
                self.rng.choice(self._all_action_indices)
                if explored
                else greedy_index
            )
        metadata: Dict[str, Any] = {}
        if include_metadata:
            metadata = {
                "epsilon": float(effective_epsilon),
                "greedy_index": int(greedy_index),
                "explored": bool(explored),
                "allowed_indices": self._all_action_indices.tolist(),
                "forced_default_first_action": forced,
                "uncertainty": uncertainty.tolist(),
                "q_values": means.tolist(),
                "selection_scores": scores.tolist(),
                "ensemble_active_members": self._active_members.astype(int).tolist(),
            }
        return action_index, float(self.weights[action_index]), metadata

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
        del next_cycle
        errors = [
            self.members[index].update(
                features=features,
                action_index=action_index,
                cost=cost,
                next_features=next_features,
                terminal=terminal,
                next_action_index=next_action_index,
                native_cycle_cost=native_cycle_cost,
                residual_ratio=residual_ratio,
            )
            for index in np.flatnonzero(self._active_members)
        ]
        self.steps += 1
        return float(np.mean(errors))

    def finish_episode(self, *, learned: bool) -> None:
        for index, member in enumerate(self.members):
            member.finish_episode(
                learned=bool(learned and self._active_members[index]),
            )
        if learned:
            self.episodes += 1

    def snapshot_learning_state(self) -> Dict[str, Any]:
        return {
            "members": [member.snapshot_learning_state() for member in self.members],
            "steps": int(self.steps),
            "episodes": int(self.episodes),
            "initial_environment_weight": float(self._initial_environment_weight),
            "default_first_action_pending": bool(self._default_first_action_pending),
            "forced_initial_action_count": int(self.forced_initial_action_count),
            "active_members": self._active_members.copy(),
        }

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        for member, member_state in zip(self.members, state["members"]):
            member.restore_learning_state(member_state)
        self.steps = int(state["steps"])
        self.episodes = int(state["episodes"])
        self._initial_environment_weight = float(
            state["initial_environment_weight"]
        )
        self._default_first_action_pending = bool(
            state["default_first_action_pending"]
        )
        self.forced_initial_action_count = int(
            state["forced_initial_action_count"]
        )
        self._active_members[:] = np.asarray(state["active_members"], dtype=bool)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=np.stack([member.theta for member in self.members]),
            td_counts=np.stack([member.td_counts for member in self.members]),
            member_steps=np.asarray([member.steps for member in self.members], dtype=np.int64),
            member_episodes=np.asarray(
                [member.episodes for member in self.members], dtype=np.int64
            ),
            steps=np.asarray(self.steps, dtype=np.int64),
            episodes=np.asarray(self.episodes, dtype=np.int64),
            default_first_action_pending=np.asarray(
                self._default_first_action_pending, dtype=np.bool_
            ),
            forced_initial_action_count=np.asarray(
                self.forced_initial_action_count, dtype=np.int64
            ),
            config=np.asarray(json.dumps(asdict(self.config))),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            rng_state=np.asarray(json.dumps(self.rng.bit_generator.state)),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            if json.loads(str(payload["config"].item())) != _json_dataclass(self.config):
                raise ValueError("Checkpoint bootstrap SARSA config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(self.spec):
                raise ValueError("Checkpoint bootstrap SARSA spec does not match")
            theta = np.asarray(payload["theta"], dtype=float)
            counts = np.asarray(payload["td_counts"], dtype=float)
            expected_shape = (len(self.members), *self.members[0].theta.shape)
            if theta.shape != expected_shape or counts.shape != expected_shape:
                raise ValueError("Checkpoint bootstrap member shape does not match")
            for index, member in enumerate(self.members):
                member.theta[:] = theta[index]
                if restore_counters:
                    member.td_counts[:] = counts[index]
                    member.steps = int(payload["member_steps"][index])
                    member.episodes = int(payload["member_episodes"][index])
            if restore_counters:
                self.steps = int(payload["steps"].item())
                self.episodes = int(payload["episodes"].item())
            self._default_first_action_pending = bool(
                self.config.force_default_first_action
                and payload["default_first_action_pending"].item()
            )
            self.forced_initial_action_count = int(
                payload["forced_initial_action_count"].item()
            )
            self.rng.bit_generator.state = json.loads(str(payload["rng_state"].item()))
        for member in self.members:
            member.start_episode()
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "members": int(len(self.members)),
            "episode_inclusion_probability": float(
                self.spec.episode_inclusion_probability
            ),
            "uncertainty_beta": float(self.spec.uncertainty_beta),
            "member_steps": [int(member.steps) for member in self.members],
            "member_episodes": [int(member.episodes) for member in self.members],
        }


class StagewiseLsviLcbController:
    """Finite-horizon least-squares value iteration with cost optimism."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: StagewiseLsviLcbSpec,
        seed: int,
    ) -> None:
        if int(spec.horizon) <= 0:
            raise ValueError("LSVI horizon must be positive")
        if float(spec.ridge) <= 0.0:
            raise ValueError("LSVI ridge must be positive")
        if float(spec.uncertainty_beta) < 0.0:
            raise ValueError("LSVI uncertainty beta must be non-negative")
        if float(spec.residual_floor_sec) <= 0.0:
            raise ValueError("LSVI residual floor must be positive")
        if float(spec.q_max_sec) <= 0.0:
            raise ValueError("LSVI Q cap must be positive")
        if int(spec.refit_interval_episodes) <= 0:
            raise ValueError("LSVI refit interval must be positive")

        self.config = config
        self.spec = spec
        self.feature_dim = int(feature_dim)
        self.weights = np.asarray(config.weights, dtype=float)
        self.action_basis = build_action_basis(
            self.weights,
            mode=str(config.action_basis_mode),
            rbf_sigma=float(config.action_rbf_sigma),
            rbf_centers=tuple(config.action_basis_centers),
        )
        self.joint_dim = int(self.action_basis.shape[1] * self.feature_dim)
        self.rng = np.random.default_rng(int(seed))
        self.anchor_index = int(
            np.argmin(np.abs(self.weights - float(config.anchor_weight)))
        )
        self._all_action_indices = np.arange(self.weights.size, dtype=int)
        self._initial_environment_weight = float(config.anchor_weight)
        self._default_first_action_pending = bool(config.force_default_first_action)
        self.forced_initial_action_count = 0
        self.steps = 0
        self.episodes = 0

        horizon = int(spec.horizon)
        ridge_inverse = 1.0 / float(spec.ridge)
        self.design_inverse = np.repeat(
            (ridge_inverse * np.eye(self.joint_dim, dtype=float))[None, :, :],
            horizon,
            axis=0,
        )
        self.policy_design_inverse = self.design_inverse.copy()
        self.theta = np.zeros((horizon, self.joint_dim), dtype=float)
        self.residual_scales = np.full(
            horizon, float(spec.residual_floor_sec), dtype=float
        )
        self.sample_counts = np.zeros(horizon, dtype=np.int64)
        self._features: list[list[np.ndarray]] = [[] for _ in range(horizon)]
        self._costs: list[list[float]] = [[] for _ in range(horizon)]
        self._next_features: list[list[np.ndarray]] = [[] for _ in range(horizon)]
        self._terminals: list[list[bool]] = [[] for _ in range(horizon)]
        self._pending: list[tuple[int, np.ndarray, float, np.ndarray, bool]] = []
        self.refit_count = 0

    @property
    def epsilon(self) -> float:
        decay = max(float(self.config.epsilon_decay_steps), 1.0)
        fraction = math.exp(-float(self.steps) / decay)
        return float(
            self.config.epsilon_final
            + (self.config.epsilon_start - self.config.epsilon_final) * fraction
        )

    @property
    def requires_next_action(self) -> bool:
        return False

    @property
    def initial_environment_weight(self) -> float:
        return float(self._initial_environment_weight)

    def environment_weight(self, action_index: int, last_weight: float) -> float:
        del last_weight
        return float(self.weights[int(action_index)])

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        self._pending.clear()
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)

    def state_action_features(self, features: np.ndarray) -> np.ndarray:
        return joint_action_features(self.action_basis, features)

    def _stage(self, cycle: int | None) -> int:
        return int(np.clip(0 if cycle is None else cycle, 0, int(self.spec.horizon) - 1))

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stage = self._stage(cycle)
        joint = self.state_action_features(features)
        means = joint @ self.theta[stage]
        inverse = self.policy_design_inverse[stage]
        quadratic = np.einsum("ai,ij,aj->a", joint, inverse, joint, optimize=True)
        uncertainty = float(self.residual_scales[stage]) * np.sqrt(
            np.maximum(quadratic, 0.0)
        )
        scores = means - float(self.spec.uncertainty_beta) * uncertainty
        return means, uncertainty, scores

    def q_values(self, features: np.ndarray, *, cycle: int | None = None) -> np.ndarray:
        return self._values(features, cycle=cycle)[0]

    def _optimistic_value(self, features: np.ndarray, *, cycle: int) -> float:
        _means, _uncertainty, scores = self._values(features, cycle=cycle)
        return float(np.clip(np.min(scores), 0.0, float(self.spec.q_max_sec)))

    def _optimistic_values(
        self,
        features: np.ndarray,
        *,
        cycle: int,
    ) -> np.ndarray:
        states = np.asarray(features, dtype=float)
        if states.ndim != 2 or states.shape[1] != self.feature_dim:
            raise ValueError("batched LSVI states do not match feature_dim")
        stage = self._stage(cycle)
        joint = np.einsum(
            "ab,nf->nabf",
            self.action_basis,
            states,
            optimize=True,
        ).reshape(states.shape[0], self.weights.size, self.joint_dim)
        means = joint @ self.theta[stage]
        projected = joint @ self.policy_design_inverse[stage]
        quadratic = np.sum(projected * joint, axis=2)
        uncertainty = float(self.residual_scales[stage]) * np.sqrt(
            np.maximum(quadratic, 0.0)
        )
        scores = means - float(self.spec.uncertainty_beta) * uncertainty
        return np.clip(
            np.min(scores, axis=1),
            0.0,
            float(self.spec.q_max_sec),
        )

    def select_action(
        self,
        features: np.ndarray,
        *,
        explore: bool,
        epsilon: float | None = None,
        cycle: int | None = None,
        include_metadata: bool = True,
    ) -> tuple[int, float, Dict[str, Any]]:
        _means, uncertainty, scores = self._values(features, cycle=cycle)
        greedy_index = _greedy_cost_index(
            scores,
            allowed_indices=self._all_action_indices,
            anchor_index=self.anchor_index,
        )
        effective_epsilon = (
            (self.epsilon if epsilon is None else float(epsilon)) if explore else 0.0
        )
        forced = bool(explore and self._default_first_action_pending)
        if forced:
            action_index = int(
                np.argmin(
                    np.abs(self.weights - float(self.initial_environment_weight))
                )
            )
            if not np.isclose(
                self.weights[action_index],
                self.initial_environment_weight,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError("The action space cannot represent the solver default")
            self._default_first_action_pending = False
            self.forced_initial_action_count += 1
            explored = False
            effective_epsilon = 0.0
        else:
            explored = bool(
                effective_epsilon > 0.0
                and float(self.rng.random()) < effective_epsilon
            )
            action_index = int(
                self.rng.choice(self._all_action_indices)
                if explored
                else greedy_index
            )
        metadata: Dict[str, Any] = {}
        if include_metadata:
            metadata = {
                "epsilon": float(effective_epsilon),
                "greedy_index": int(greedy_index),
                "explored": bool(explored),
                "allowed_indices": self._all_action_indices.tolist(),
                "forced_default_first_action": forced,
                "uncertainty": uncertainty.tolist(),
                "uncertainty_td_scale": float(
                    self.residual_scales[self._stage(cycle)]
                ),
                "q_values": _means.tolist(),
                "selection_scores": scores.tolist(),
            }
        return action_index, float(self.weights[action_index]), metadata

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
        del next_action_index, native_cycle_cost, residual_ratio
        stage = self._stage(None if next_cycle is None else int(next_cycle) - 1)
        joint = self.state_action_features(features)[int(action_index)]
        current = float(joint @ self.theta[stage])
        next_value = 0.0 if terminal else self._optimistic_value(
            next_features,
            cycle=stage + 1,
        )
        target = float(cost + self.config.gamma * next_value)
        self._pending.append(
            (
                stage,
                joint.copy(),
                float(cost),
                np.asarray(next_features, dtype=float).copy(),
                bool(terminal),
            )
        )
        self.steps += 1
        return float(target - current)

    def _commit_pending(self) -> None:
        for stage, joint, cost, next_features, terminal in self._pending:
            inverse = self.design_inverse[stage]
            projected = inverse @ joint
            denominator = 1.0 + float(joint @ projected)
            inverse -= np.outer(projected, projected) / max(denominator, 1.0e-15)
            self._features[stage].append(joint)
            self._costs[stage].append(float(cost))
            self._next_features[stage].append(next_features)
            self._terminals[stage].append(bool(terminal))
            self.sample_counts[stage] += 1

    def _fit_backward(self) -> None:
        for stage in range(int(self.spec.horizon) - 1, -1, -1):
            if not self._features[stage]:
                self.theta[stage].fill(0.0)
                self.residual_scales[stage] = float(self.spec.residual_floor_sec)
                continue
            design = np.vstack(self._features[stage])
            targets = np.asarray(self._costs[stage], dtype=float)
            if stage + 1 < int(self.spec.horizon):
                nonterminal = np.flatnonzero(
                    np.logical_not(np.asarray(self._terminals[stage], dtype=bool))
                )
                if nonterminal.size:
                    next_states = np.vstack(
                        [self._next_features[stage][index] for index in nonterminal]
                    )
                    targets[nonterminal] += float(
                        self.config.gamma
                    ) * self._optimistic_values(
                        next_states,
                        cycle=stage + 1,
                    )
            self.theta[stage] = self.design_inverse[stage] @ (design.T @ targets)
            residuals = targets - design @ self.theta[stage]
            rms = float(np.sqrt(np.mean(np.square(residuals))))
            self.residual_scales[stage] = max(
                rms,
                float(self.spec.residual_floor_sec),
            )

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            self._commit_pending()
            self.episodes += 1
            if self.episodes % int(self.spec.refit_interval_episodes) == 0:
                self.policy_design_inverse[:] = self.design_inverse
                self._fit_backward()
                self.refit_count += 1
        self._pending.clear()

    def snapshot_learning_state(self) -> Dict[str, Any]:
        # Stagewise updates remain pending until finish_episode, so rollback only
        # needs the counters and pending buffer. Historical batches are unchanged.
        return {
            "steps": int(self.steps),
            "episodes": int(self.episodes),
            "refit_count": int(self.refit_count),
            "initial_environment_weight": float(self._initial_environment_weight),
            "default_first_action_pending": bool(self._default_first_action_pending),
            "forced_initial_action_count": int(self.forced_initial_action_count),
        }

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self.steps = int(state["steps"])
        self.episodes = int(state["episodes"])
        self.refit_count = int(state["refit_count"])
        self._initial_environment_weight = float(
            state["initial_environment_weight"]
        )
        self._default_first_action_pending = bool(
            state["default_first_action_pending"]
        )
        self.forced_initial_action_count = int(
            state["forced_initial_action_count"]
        )
        self._pending.clear()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stage_indices: list[int] = []
        feature_rows: list[np.ndarray] = []
        costs: list[float] = []
        next_rows: list[np.ndarray] = []
        terminals: list[bool] = []
        for stage in range(int(self.spec.horizon)):
            for index, feature in enumerate(self._features[stage]):
                stage_indices.append(stage)
                feature_rows.append(feature)
                costs.append(self._costs[stage][index])
                next_rows.append(self._next_features[stage][index])
                terminals.append(self._terminals[stage][index])
        np.savez_compressed(
            path,
            theta=self.theta,
            design_inverse=self.design_inverse,
            policy_design_inverse=self.policy_design_inverse,
            residual_scales=self.residual_scales,
            sample_counts=self.sample_counts,
            stage_indices=np.asarray(stage_indices, dtype=np.int64),
            features=(
                np.vstack(feature_rows)
                if feature_rows
                else np.empty((0, self.joint_dim), dtype=float)
            ),
            costs=np.asarray(costs, dtype=float),
            next_features=(
                np.vstack(next_rows)
                if next_rows
                else np.empty((0, self.feature_dim), dtype=float)
            ),
            terminals=np.asarray(terminals, dtype=np.bool_),
            steps=np.asarray(self.steps, dtype=np.int64),
            episodes=np.asarray(self.episodes, dtype=np.int64),
            refit_count=np.asarray(self.refit_count, dtype=np.int64),
            default_first_action_pending=np.asarray(
                self._default_first_action_pending, dtype=np.bool_
            ),
            forced_initial_action_count=np.asarray(
                self.forced_initial_action_count, dtype=np.int64
            ),
            config=np.asarray(json.dumps(asdict(self.config))),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            rng_state=np.asarray(json.dumps(self.rng.bit_generator.state)),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            if json.loads(str(payload["config"].item())) != _json_dataclass(self.config):
                raise ValueError("Checkpoint LSVI config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(self.spec):
                raise ValueError("Checkpoint LSVI spec does not match")
            theta = np.asarray(payload["theta"], dtype=float)
            inverse = np.asarray(payload["design_inverse"], dtype=float)
            if theta.shape != self.theta.shape or inverse.shape != self.design_inverse.shape:
                raise ValueError("Checkpoint LSVI parameter shape does not match")
            self.theta[:] = theta
            self.design_inverse[:] = inverse
            if "policy_design_inverse" in payload.files:
                self.policy_design_inverse[:] = np.asarray(
                    payload["policy_design_inverse"], dtype=float
                )
            else:
                self.policy_design_inverse[:] = inverse
            self.residual_scales[:] = np.asarray(
                payload["residual_scales"], dtype=float
            )
            self.sample_counts[:] = np.asarray(payload["sample_counts"], dtype=np.int64)
            self._features = [[] for _ in range(int(self.spec.horizon))]
            self._costs = [[] for _ in range(int(self.spec.horizon))]
            self._next_features = [[] for _ in range(int(self.spec.horizon))]
            self._terminals = [[] for _ in range(int(self.spec.horizon))]
            stages = np.asarray(payload["stage_indices"], dtype=np.int64)
            features = np.asarray(payload["features"], dtype=float)
            costs = np.asarray(payload["costs"], dtype=float)
            next_features = np.asarray(payload["next_features"], dtype=float)
            terminals = np.asarray(payload["terminals"], dtype=bool)
            if not (
                stages.size
                == features.shape[0]
                == costs.size
                == next_features.shape[0]
                == terminals.size
            ):
                raise ValueError("Checkpoint LSVI trajectory arrays do not align")
            for index, stage in enumerate(stages):
                self._features[int(stage)].append(features[index].copy())
                self._costs[int(stage)].append(float(costs[index]))
                self._next_features[int(stage)].append(next_features[index].copy())
                self._terminals[int(stage)].append(bool(terminals[index]))
            if restore_counters:
                self.steps = int(payload["steps"].item())
                self.episodes = int(payload["episodes"].item())
                self.refit_count = int(
                    payload["refit_count"].item()
                    if "refit_count" in payload.files
                    else self.episodes
                )
            self._default_first_action_pending = bool(
                self.config.force_default_first_action
                and payload["default_first_action_pending"].item()
            )
            self.forced_initial_action_count = int(
                payload["forced_initial_action_count"].item()
            )
            self.rng.bit_generator.state = json.loads(str(payload["rng_state"].item()))
        self._pending.clear()
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "horizon": int(self.spec.horizon),
            "joint_feature_dim": int(self.joint_dim),
            "sample_counts": self.sample_counts.tolist(),
            "residual_scales_sec": self.residual_scales.tolist(),
            "uncertainty_beta": float(self.spec.uncertainty_beta),
            "refit_interval_episodes": int(self.spec.refit_interval_episodes),
            "refit_count": int(self.refit_count),
            "stored_transition_count": int(np.sum(self.sample_counts)),
        }


class HierarchicalLsviLcbController(_SharedActionLcbController):
    """Stagewise LSVI regularized toward a shared cross-stage value model."""

    prefer_mean_on_score_ties = True

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        spec: HierarchicalLsviLcbSpec,
        seed: int,
    ) -> None:
        if int(spec.horizon) <= 0:
            raise ValueError("hierarchical LSVI horizon must be positive")
        if float(spec.ridge) <= 0.0:
            raise ValueError("hierarchical LSVI ridge must be positive")
        if float(spec.uncertainty_beta) < 0.0:
            raise ValueError("hierarchical LSVI beta must be non-negative")
        if float(spec.residual_floor_sec) <= 0.0:
            raise ValueError("hierarchical LSVI residual floor must be positive")
        if int(spec.refit_interval_episodes) <= 0:
            raise ValueError("hierarchical LSVI refit interval must be positive")
        if int(spec.refit_sweeps) <= 0:
            raise ValueError("hierarchical LSVI refit sweeps must be positive")
        if float(spec.residual_shrinkage_samples) <= 0.0:
            raise ValueError("hierarchical LSVI shrinkage must be positive")
        super().__init__(feature_dim=feature_dim, config=config, seed=seed)
        self.spec = spec
        horizon = int(spec.horizon)
        ridge_inverse = 1.0 / float(spec.ridge)
        initial_inverse = ridge_inverse * np.eye(self.joint_dim, dtype=float)
        self.stage_design_inverse = np.repeat(
            initial_inverse[None, :, :], horizon, axis=0
        )
        self.policy_stage_design_inverse = self.stage_design_inverse.copy()
        self.shared_design_inverse = initial_inverse.copy()
        self.policy_shared_design_inverse = initial_inverse.copy()
        self.theta = np.zeros((horizon, self.joint_dim), dtype=float)
        self.shared_theta = np.zeros(self.joint_dim, dtype=float)
        self.residual_scales = np.full(
            horizon, float(spec.residual_floor_sec), dtype=float
        )
        self.global_residual_scale = float(spec.residual_floor_sec)
        self.sample_counts = np.zeros(horizon, dtype=np.int64)
        self._features: list[list[np.ndarray]] = [[] for _ in range(horizon)]
        self._costs: list[list[float]] = [[] for _ in range(horizon)]
        self._next_features: list[list[np.ndarray]] = [[] for _ in range(horizon)]
        self._terminals: list[list[bool]] = [[] for _ in range(horizon)]
        self._pending: list[tuple[int, np.ndarray, float, np.ndarray, bool]] = []
        self.refit_count = 0
        self.observed_return_max_sec = 0.0
        self.inverse_rebuild_count = 0

    @property
    def requires_next_action(self) -> bool:
        return False

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        self._pending.clear()
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)

    def _stage(self, cycle: int | None) -> int:
        return int(
            np.clip(0 if cycle is None else cycle, 0, int(self.spec.horizon) - 1)
        )

    def _value_components(
        self,
        joint: np.ndarray,
        *,
        stage: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        means = joint @ self.theta[stage]
        stage_projected = joint @ self.policy_stage_design_inverse[stage]
        shared_projected = joint @ self.policy_shared_design_inverse
        stage_quadratic = np.sum(stage_projected * joint, axis=-1)
        shared_quadratic = np.sum(shared_projected * joint, axis=-1)
        variance = (
            float(self.residual_scales[stage]) ** 2
            * np.maximum(stage_quadratic, 0.0)
            + float(self.global_residual_scale) ** 2
            * np.maximum(shared_quadratic, 0.0)
        )
        uncertainty = np.sqrt(np.maximum(variance, 0.0))
        scores = np.maximum(
            means - float(self.spec.uncertainty_beta) * uncertainty,
            0.0,
        )
        return means, uncertainty, scores

    def _values(
        self,
        features: np.ndarray,
        *,
        cycle: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        stage = self._stage(cycle)
        return self._value_components(
            self.state_action_features(features), stage=stage
        )

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
                    "uncertainty_td_scale": float(
                        self.residual_scales[self._stage(cycle)]
                    ),
                    "global_uncertainty_td_scale": float(
                        self.global_residual_scale
                    ),
                    "dynamic_value_cap_sec": float(
                        self.observed_return_max_sec
                    ),
                }
            )
        return action, weight, metadata

    def _optimistic_values(
        self,
        features: np.ndarray,
        *,
        cycle: int,
    ) -> np.ndarray:
        states = np.asarray(features, dtype=float)
        if states.ndim != 2 or states.shape[1] != self.feature_dim:
            raise ValueError("hierarchical LSVI states do not match feature_dim")
        values = np.empty(states.shape[0], dtype=float)
        stage = self._stage(cycle)
        # Keep all-history refits memory bounded at late stages.  A full 4K
        # stage-0 tensor would otherwise exceed several hundred MB.
        for start in range(0, states.shape[0], 128):
            stop = min(start + 128, states.shape[0])
            joint = np.einsum(
                "ab,nf->nabf",
                self.action_basis,
                states[start:stop],
                optimize=True,
            ).reshape(stop - start, self.weights.size, self.joint_dim)
            _means, _uncertainty, scores = self._value_components(
                joint, stage=stage
            )
            values[start:stop] = np.minimum(
                np.min(scores, axis=1), float(self.observed_return_max_sec)
            )
        return values

    def _optimistic_value(self, features: np.ndarray, *, cycle: int) -> float:
        return float(
            self._optimistic_values(
                np.asarray(features, dtype=float)[None, :], cycle=cycle
            )[0]
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
        del next_action_index, native_cycle_cost, residual_ratio
        stage = self._stage(None if next_cycle is None else int(next_cycle) - 1)
        joint = self.state_action_feature(features, int(action_index))
        current = float(joint @ self.theta[stage])
        next_value = 0.0 if terminal else self._optimistic_value(
            next_features, cycle=stage + 1
        )
        target = float(cost + float(self.config.gamma) * next_value)
        self._pending.append(
            (
                stage,
                joint.copy(),
                float(cost),
                np.asarray(next_features, dtype=float).copy(),
                bool(terminal),
            )
        )
        self.steps += 1
        return float(target - current)

    def _rebuild_stage_inverse(self, stage: int) -> None:
        design = float(self.spec.ridge) * np.eye(self.joint_dim, dtype=float)
        if self._features[stage]:
            rows = np.vstack(self._features[stage])
            design += rows.T @ rows
        self.stage_design_inverse[stage] = np.linalg.pinv(design)
        self.inverse_rebuild_count += 1

    def _rebuild_shared_inverse(self) -> None:
        design = float(self.spec.ridge) * np.eye(self.joint_dim, dtype=float)
        rows = [feature for stage in self._features for feature in stage]
        if rows:
            stacked = np.vstack(rows)
            design += stacked.T @ stacked
        self.shared_design_inverse[:] = np.linalg.pinv(design)
        self.inverse_rebuild_count += 1

    def _commit_pending(self) -> None:
        for stage, joint, cost, next_features, terminal in self._pending:
            self._features[stage].append(joint)
            self._costs[stage].append(float(cost))
            self._next_features[stage].append(next_features)
            self._terminals[stage].append(bool(terminal))
            self.sample_counts[stage] += 1
            if not _rank_one_inverse_update(
                self.stage_design_inverse[stage],
                joint,
                denominator_floor=1.0e-15,
            ):
                self._rebuild_stage_inverse(stage)
            if not _rank_one_inverse_update(
                self.shared_design_inverse,
                joint,
                denominator_floor=1.0e-15,
            ):
                self._rebuild_shared_inverse()

    def _stage_targets(self, stage: int) -> tuple[np.ndarray, np.ndarray]:
        design = np.vstack(self._features[stage])
        targets = np.asarray(self._costs[stage], dtype=float).copy()
        if stage + 1 < int(self.spec.horizon):
            nonterminal = np.flatnonzero(
                np.logical_not(np.asarray(self._terminals[stage], dtype=bool))
            )
            if nonterminal.size:
                next_states = np.vstack(
                    [self._next_features[stage][index] for index in nonterminal]
                )
                targets[nonterminal] += float(
                    self.config.gamma
                ) * self._optimistic_values(next_states, cycle=stage + 1)
        return design, targets

    def _fit_hierarchy(self) -> None:
        self.policy_stage_design_inverse[:] = self.stage_design_inverse
        self.policy_shared_design_inverse[:] = self.shared_design_inverse
        final_targets: list[tuple[np.ndarray, np.ndarray] | None] = [
            None for _ in range(int(self.spec.horizon))
        ]
        ridge = float(self.spec.ridge)
        for _sweep in range(int(self.spec.refit_sweeps)):
            shared_b = np.zeros(self.joint_dim, dtype=float)
            for stage in range(int(self.spec.horizon) - 1, -1, -1):
                if not self._features[stage]:
                    self.theta[stage] = self.shared_theta
                    final_targets[stage] = None
                    continue
                design, targets = self._stage_targets(stage)
                self.theta[stage] = self.stage_design_inverse[stage] @ (
                    design.T @ targets + ridge * self.shared_theta
                )
                shared_b += design.T @ targets
                final_targets[stage] = (design, targets)
            self.shared_theta[:] = self.shared_design_inverse @ shared_b

        squared_residual_sums = np.zeros(int(self.spec.horizon), dtype=float)
        all_residuals: list[np.ndarray] = []
        for stage, fitted in enumerate(final_targets):
            if fitted is None:
                continue
            design, targets = fitted
            residuals = targets - design @ self.theta[stage]
            squared_residual_sums[stage] = float(np.sum(np.square(residuals)))
            all_residuals.append(residuals)
        if all_residuals:
            combined = np.concatenate(all_residuals)
            self.global_residual_scale = max(
                float(np.sqrt(np.mean(np.square(combined)))),
                float(self.spec.residual_floor_sec),
            )
        else:
            self.global_residual_scale = float(self.spec.residual_floor_sec)
        prior_count = float(self.spec.residual_shrinkage_samples)
        global_variance = float(self.global_residual_scale) ** 2
        for stage in range(int(self.spec.horizon)):
            count = float(self.sample_counts[stage])
            shrunk_variance = (
                squared_residual_sums[stage] + prior_count * global_variance
            ) / (count + prior_count)
            self.residual_scales[stage] = max(
                math.sqrt(max(shrunk_variance, 0.0)),
                float(self.spec.residual_floor_sec),
            )

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            return_to_go = 0.0
            for _stage, _joint, cost, _next_features, _terminal in reversed(
                self._pending
            ):
                return_to_go = float(cost) + float(self.config.gamma) * return_to_go
                self.observed_return_max_sec = max(
                    self.observed_return_max_sec, return_to_go
                )
            self._commit_pending()
            self.episodes += 1
            if self.episodes % int(self.spec.refit_interval_episodes) == 0:
                self._fit_hierarchy()
                self.refit_count += 1
        self._pending.clear()

    def snapshot_learning_state(self) -> Dict[str, Any]:
        state = self._snapshot_common_learning_state()
        state.update(
            {
                "refit_count": int(self.refit_count),
                "observed_return_max_sec": float(self.observed_return_max_sec),
                "inverse_rebuild_count": int(self.inverse_rebuild_count),
            }
        )
        return state

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self._restore_common_learning_state(state)
        self.refit_count = int(state["refit_count"])
        self.observed_return_max_sec = float(state["observed_return_max_sec"])
        self.inverse_rebuild_count = int(state["inverse_rebuild_count"])
        self._pending.clear()

    def _flat_history(self) -> tuple[list[int], list[np.ndarray], list[float], list[np.ndarray], list[bool]]:
        stages: list[int] = []
        features: list[np.ndarray] = []
        costs: list[float] = []
        next_features: list[np.ndarray] = []
        terminals: list[bool] = []
        for stage in range(int(self.spec.horizon)):
            for index, feature in enumerate(self._features[stage]):
                stages.append(stage)
                features.append(feature)
                costs.append(self._costs[stage][index])
                next_features.append(self._next_features[stage][index])
                terminals.append(self._terminals[stage][index])
        return stages, features, costs, next_features, terminals

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        stages, features, costs, next_features, terminals = self._flat_history()
        np.savez_compressed(
            path,
            theta=self.theta,
            shared_theta=self.shared_theta,
            stage_design_inverse=self.stage_design_inverse,
            policy_stage_design_inverse=self.policy_stage_design_inverse,
            shared_design_inverse=self.shared_design_inverse,
            policy_shared_design_inverse=self.policy_shared_design_inverse,
            residual_scales=self.residual_scales,
            global_residual_scale=np.asarray(
                self.global_residual_scale, dtype=float
            ),
            sample_counts=self.sample_counts,
            stage_indices=np.asarray(stages, dtype=np.int64),
            features=(
                np.vstack(features)
                if features
                else np.empty((0, self.joint_dim), dtype=float)
            ),
            costs=np.asarray(costs, dtype=float),
            next_features=(
                np.vstack(next_features)
                if next_features
                else np.empty((0, self.feature_dim), dtype=float)
            ),
            terminals=np.asarray(terminals, dtype=np.bool_),
            refit_count=np.asarray(self.refit_count, dtype=np.int64),
            observed_return_max_sec=np.asarray(
                self.observed_return_max_sec, dtype=float
            ),
            inverse_rebuild_count=np.asarray(
                self.inverse_rebuild_count, dtype=np.int64
            ),
            checkpoint_version=np.asarray(
                _HIERARCHICAL_LSVI_CHECKPOINT_VERSION, dtype=np.int64
            ),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["checkpoint_version"].item()) != int(
                _HIERARCHICAL_LSVI_CHECKPOINT_VERSION
            ):
                raise ValueError("hierarchical LSVI checkpoint version does not match")
            if json.loads(str(payload["config"].item())) != _json_dataclass(
                self.config
            ):
                raise ValueError("Checkpoint hierarchical LSVI config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(
                self.spec
            ):
                raise ValueError("Checkpoint hierarchical LSVI spec does not match")
            for name in (
                "theta",
                "shared_theta",
                "stage_design_inverse",
                "policy_stage_design_inverse",
                "shared_design_inverse",
                "policy_shared_design_inverse",
                "residual_scales",
                "sample_counts",
            ):
                getattr(self, name)[:] = np.asarray(payload[name], dtype=float)
            self.sample_counts[:] = np.asarray(
                payload["sample_counts"], dtype=np.int64
            )
            self.global_residual_scale = float(
                payload["global_residual_scale"].item()
            )
            self.refit_count = int(payload["refit_count"].item())
            self.observed_return_max_sec = float(
                payload["observed_return_max_sec"].item()
            )
            self.inverse_rebuild_count = int(
                payload["inverse_rebuild_count"].item()
            )
            self._features = [[] for _ in range(int(self.spec.horizon))]
            self._costs = [[] for _ in range(int(self.spec.horizon))]
            self._next_features = [[] for _ in range(int(self.spec.horizon))]
            self._terminals = [[] for _ in range(int(self.spec.horizon))]
            stages = np.asarray(payload["stage_indices"], dtype=np.int64)
            features = np.asarray(payload["features"], dtype=float)
            costs = np.asarray(payload["costs"], dtype=float)
            next_features = np.asarray(payload["next_features"], dtype=float)
            terminals = np.asarray(payload["terminals"], dtype=bool)
            if not (
                stages.size
                == features.shape[0]
                == costs.size
                == next_features.shape[0]
                == terminals.size
            ):
                raise ValueError("Checkpoint hierarchical LSVI history does not align")
            for index, stage in enumerate(stages):
                self._features[int(stage)].append(features[index].copy())
                self._costs[int(stage)].append(float(costs[index]))
                self._next_features[int(stage)].append(next_features[index].copy())
                self._terminals[int(stage)].append(bool(terminals[index]))
            self._load_common(payload, restore_counters=restore_counters)
        self._pending.clear()
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "horizon": int(self.spec.horizon),
            "joint_feature_dim": int(self.joint_dim),
            "sample_counts": self.sample_counts.tolist(),
            "residual_scales_sec": self.residual_scales.tolist(),
            "global_residual_scale_sec": float(self.global_residual_scale),
            "uncertainty_beta": float(self.spec.uncertainty_beta),
            "dynamic_value_cap_sec": float(self.observed_return_max_sec),
            "refit_interval_episodes": int(self.spec.refit_interval_episodes),
            "refit_sweeps": int(self.spec.refit_sweeps),
            "refit_count": int(self.refit_count),
            "inverse_rebuild_count": int(self.inverse_rebuild_count),
            "stored_transition_count": int(np.sum(self.sample_counts)),
        }
