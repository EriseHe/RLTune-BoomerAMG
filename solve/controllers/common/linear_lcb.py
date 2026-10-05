from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any, Dict, Iterable, Sequence

import numpy as np

try:
    from scipy.linalg.blas import dger as _blas_dger
except ImportError:  # pragma: no cover - NumPy fallback keeps SciPy optional.
    _blas_dger = None

from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig

from .action_space import build_action_basis, joint_action_features


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
