"""Online TD(lambda) solve controllers and transactional episode execution."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

from hypre.bindings import (
    AMGNativeError,
    AttemptOutcome,
    AttemptStatus,
    RecoveryOutcome,
    SolveStatus,
    augment_setup_params,
    create_env,
    execute_attempt,
)
from hypre.bindings.recovery import (
    InvalidObservationError, validate_failure_penalty, validate_runtime_cost,
)
from solve.controllers.common.action_space import (
    build_action_basis,
    joint_action_features,
)
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.core.outcomes import classify_rl_failure

from .config import ExpectedSarsaLambdaConfig


class OnlineFixedWeightIncumbent:
    """Balanced black-box calibration using one observed solve per instance."""

    def __init__(
        self,
        *,
        weights: Sequence[float],
        initial_weight: float,
        seed: int,
    ) -> None:
        if not weights:
            raise ValueError("OnlineFixedWeightIncumbent requires at least one weight")
        self.weights = np.asarray(tuple(float(weight) for weight in weights), dtype=float)
        self.rng = np.random.default_rng(int(seed))
        self.counts = np.zeros(self.weights.size, dtype=np.int64)
        self.mean_costs = np.zeros(self.weights.size, dtype=float)
        self.cost_m2 = np.zeros(self.weights.size, dtype=float)
        self.initial_index = int(np.argmin(np.abs(self.weights - float(initial_weight))))
        if not np.isclose(
            self.weights[self.initial_index],
            float(initial_weight),
            atol=1.0e-12,
            rtol=0.0,
        ):
            raise ValueError("initial_weight must be present in weights")

    def select_action(self) -> tuple[int, float]:
        least_visited = int(np.min(self.counts))
        candidates = np.flatnonzero(self.counts == least_visited)
        action_index = int(self.rng.choice(candidates))
        return action_index, float(self.weights[action_index])

    def update(self, action_index: int, cost: float) -> None:
        action = int(action_index)
        observed_cost = float(cost)
        if action < 0 or action >= self.weights.size:
            raise IndexError("action_index is outside the calibration action space")
        if not np.isfinite(observed_cost):
            raise ValueError("calibration cost must be finite")
        self.counts[action] += 1
        delta = observed_cost - self.mean_costs[action]
        self.mean_costs[action] += delta / float(self.counts[action])
        self.cost_m2[action] += delta * (observed_cost - self.mean_costs[action])

    @property
    def incumbent_index(self) -> int:
        visited = np.flatnonzero(self.counts > 0)
        if visited.size == 0:
            return int(self.initial_index)
        visited_costs = self.mean_costs[visited]
        best_cost = float(np.min(visited_costs))
        candidates = visited[np.isclose(visited_costs, best_cost, atol=1.0e-15, rtol=0.0)]
        distances = np.abs(self.weights[candidates] - self.weights[self.initial_index])
        return int(candidates[int(np.argmin(distances))])

    @property
    def incumbent_weight(self) -> float:
        return float(self.weights[self.incumbent_index])

    def summary(self) -> Dict[str, Any]:
        standard_errors = np.full(self.weights.size, np.nan, dtype=float)
        enough_samples = self.counts > 1
        variances = np.zeros(self.weights.size, dtype=float)
        variances[enough_samples] = self.cost_m2[enough_samples] / (
            self.counts[enough_samples] - 1
        )
        standard_errors[enough_samples] = np.sqrt(
            variances[enough_samples] / self.counts[enough_samples]
        )
        return {
            "weights": self.weights.tolist(),
            "counts": self.counts.tolist(),
            "mean_cost_sec": [
                float(cost) if count > 0 else float("nan")
                for cost, count in zip(self.mean_costs, self.counts)
            ],
            "standard_error_sec": standard_errors.tolist(),
            "initial_weight": float(self.weights[self.initial_index]),
            "incumbent_weight": self.incumbent_weight,
        }


class ExpectedSarsaLambda:
    """Persistent online cost-control learner with per-episode eligibility traces."""

    def __init__(
        self,
        *,
        feature_dim: int,
        config: ExpectedSarsaLambdaConfig,
        seed: int,
        initial_parameters: np.ndarray | None = None,
    ) -> None:
        if not config.weights:
            raise ValueError("ExpectedSarsaLambda requires at least one action")
        self.config = config
        self.weights = np.asarray(config.weights, dtype=float)
        self.feature_dim = int(feature_dim)
        self.rng = np.random.default_rng(int(seed))
        self.action_basis = build_action_basis(
            self.weights,
            mode=str(config.action_basis_mode),
            rbf_sigma=float(config.action_rbf_sigma),
            rbf_centers=tuple(config.action_basis_centers),
        )
        # Preserve the old public name for diagnosis code and legacy tests.
        self.action_kernel = self.action_basis
        self.basis_dim = int(self.action_basis.shape[1])
        self.theta = np.zeros((self.basis_dim, self.feature_dim), dtype=float)
        if initial_parameters is None:
            self.theta[:, 0] = float(config.initial_q_sec)
        else:
            initial_parameters = np.asarray(initial_parameters, dtype=float)
            if initial_parameters.shape != (self.feature_dim,):
                raise ValueError("initial_parameters must match feature_dim")
            self.theta[:] = initial_parameters
        self.eligibility = np.zeros_like(self.theta)
        self.td_counts = np.zeros_like(self.theta)
        self.monte_carlo_counts = np.zeros_like(self.theta)
        self.steps = 0
        self.episodes = 0
        self.q_old = 0.0
        self._initial_environment_weight = float(config.anchor_weight)
        self._default_first_action_pending = bool(
            config.force_default_first_action
        )
        self.forced_initial_action_count = 0
        self.anchor_index = int(np.argmin(np.abs(self.weights - float(config.anchor_weight))))
        if not np.isclose(self.weights[self.anchor_index], float(config.anchor_weight), atol=1.0e-12, rtol=0.0):
            raise ValueError("anchor_weight must be present in weights")
        if self.config.exploration_mode not in {"uniform", "least_visited"}:
            raise ValueError(
                "exploration_mode must be either 'uniform' or 'least_visited'"
            )
        if (
            self.config.exploration_mode == "least_visited"
            and self.basis_dim != self.weights.size
        ):
            raise ValueError(
                "least_visited exploration is only defined for independent action heads"
            )
        if self.config.td_algorithm not in {
            "expected_sarsa",
            "sarsa",
            "true_online_sarsa",
        }:
            raise ValueError(
                "td_algorithm must be 'expected_sarsa', 'sarsa', or "
                "'true_online_sarsa'"
            )
        if (
            self.config.td_algorithm == "true_online_sarsa"
            and self.config.td_decay_power > 0.0
        ):
            raise ValueError(
                "true_online_sarsa currently requires a constant scalar step size"
            )
        self._identity_action_kernel = bool(
            self.action_basis.shape == (self.weights.size, self.weights.size)
            and np.array_equal(self.action_basis, np.eye(self.weights.size))
        )
        self._all_action_indices = np.arange(self.weights.size, dtype=int)
        self._anchor_action_indices = np.asarray([self.anchor_index], dtype=int)
        self._empty_selection: Dict[str, Any] = {}

    @property
    def epsilon(self) -> float:
        decay = max(float(self.config.epsilon_decay_steps), 1.0)
        fraction = math.exp(-float(self.steps) / decay)
        return float(
            self.config.epsilon_final
            + (self.config.epsilon_start - self.config.epsilon_final) * fraction
        )

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        self.eligibility.fill(0.0)
        self.q_old = 0.0
        if initial_environment_weight is not None:
            self._initial_environment_weight = float(initial_environment_weight)

    @property
    def requires_next_action(self) -> bool:
        return self.config.td_algorithm in {"sarsa", "true_online_sarsa"}

    @property
    def initial_environment_weight(self) -> float:
        """Return the physical relaxation weight used before the first action."""

        return float(
            getattr(self, "_initial_environment_weight", self.config.anchor_weight)
        )

    def environment_weight(self, action_index: int, last_weight: float) -> float:
        """Map a controller action to the physical relaxation weight."""

        del last_weight
        return float(self.weights[int(action_index)])

    def q_values(self, features: np.ndarray) -> np.ndarray:
        basis_values = self.theta @ np.asarray(features, dtype=float)
        if self._identity_action_kernel:
            return basis_values
        return self.action_basis @ basis_values

    def state_action_features(self, features: np.ndarray) -> np.ndarray:
        """Expose the exact joint feature map used by the linear Q estimate."""

        return joint_action_features(self.action_basis, features)

    def _action_gradient(self, action_index: int, features: np.ndarray) -> np.ndarray:
        return np.outer(
            self.action_basis[int(action_index)],
            np.asarray(features, dtype=float),
        )

    def allowed_action_indices(self, cycle: int | None) -> np.ndarray:
        if (
            cycle is None
            or self.config.adaptive_cycles is None
            or int(cycle) < int(self.config.adaptive_cycles)
        ):
            return self._all_action_indices
        return self._anchor_action_indices

    def _greedy_index(
        self,
        q_values: np.ndarray,
        *,
        allowed_indices: np.ndarray | None = None,
    ) -> int:
        allowed = (
            np.arange(self.weights.size, dtype=int)
            if allowed_indices is None
            else np.asarray(allowed_indices, dtype=int)
        )
        if allowed.size == 0:
            raise ValueError("At least one action must be allowed")
        allowed_values = q_values[allowed]
        best_index = int(allowed[int(np.argmin(allowed_values))])
        best = float(q_values[best_index])
        if np.any(allowed == self.anchor_index) and np.isclose(
            q_values[self.anchor_index],
            best,
            atol=1.0e-12,
            rtol=0.0,
        ):
            return int(self.anchor_index)
        return best_index

    def policy_probabilities(
        self,
        features: np.ndarray,
        *,
        epsilon: float | None = None,
        cycle: int | None = None,
    ) -> np.ndarray:
        eps = self.epsilon if epsilon is None else float(epsilon)
        eps = float(np.clip(eps, 0.0, 1.0))
        return self._probabilities_from_q(
            self.q_values(features),
            epsilon=eps,
            allowed_indices=self.allowed_action_indices(cycle),
        )

    def _probabilities_from_q(
        self,
        q_values: np.ndarray,
        *,
        epsilon: float,
        allowed_indices: np.ndarray | None = None,
    ) -> np.ndarray:
        eps = float(np.clip(epsilon, 0.0, 1.0))
        allowed = (
            np.arange(self.weights.size, dtype=int)
            if allowed_indices is None
            else np.asarray(allowed_indices, dtype=int)
        )
        if allowed.size == 0:
            raise ValueError("At least one action must be allowed")
        probabilities = np.zeros(self.weights.size, dtype=float)
        probabilities[allowed] = eps / float(allowed.size)
        probabilities[self._greedy_index(q_values, allowed_indices=allowed)] += 1.0 - eps
        return probabilities

    def select_action(
        self,
        features: np.ndarray,
        *,
        explore: bool,
        epsilon: float | None = None,
        cycle: int | None = None,
        include_metadata: bool = True,
    ) -> tuple[int, float, Dict[str, Any]]:
        effective_epsilon = (self.epsilon if epsilon is None else float(epsilon)) if explore else 0.0
        q_values = self.q_values(features)
        allowed_indices = self.allowed_action_indices(cycle)
        greedy_index = self._greedy_index(q_values, allowed_indices=allowed_indices)
        if explore and self._default_first_action_pending:
            action_index = int(
                np.argmin(
                    np.abs(self.weights - float(self.initial_environment_weight))
                )
            )
            if not np.isclose(
                self.weights[action_index],
                float(self.initial_environment_weight),
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "The action space cannot represent the prepared solver default"
                )
            if not np.any(allowed_indices == action_index):
                raise ValueError(
                    "The prepared solver default is masked on the first decision"
                )
            self._default_first_action_pending = False
            self.forced_initial_action_count += 1
            metadata = self._empty_selection
            if include_metadata:
                metadata = {
                    "epsilon": 0.0,
                    "greedy_index": int(greedy_index),
                    "explored": False,
                    "allowed_indices": allowed_indices.tolist(),
                    "forced_default_first_action": True,
                    "q_values": q_values.tolist(),
                    "selection_scores": q_values.tolist(),
                }
            return action_index, float(self.weights[action_index]), metadata
        explored = bool(
            effective_epsilon > 0.0 and float(self.rng.random()) < effective_epsilon
        )
        if explored:
            exploration_indices = allowed_indices
            if self.config.exploration_mode == "least_visited":
                active = np.abs(np.asarray(features, dtype=float)) > 1.0e-15
                if np.any(active):
                    visit_counts = np.maximum(
                        self.td_counts,
                        self.monte_carlo_counts,
                    )
                    visit_scores = (
                        visit_counts[:, active]
                        @ np.abs(np.asarray(features, dtype=float)[active])
                    )
                    allowed_scores = visit_scores[allowed_indices]
                    least_visited = float(np.min(allowed_scores))
                    exploration_indices = allowed_indices[
                        np.isclose(allowed_scores, least_visited, atol=1.0e-12, rtol=0.0)
                    ]
            action_index = int(self.rng.choice(exploration_indices))
        else:
            action_index = int(greedy_index)
        metadata = self._empty_selection
        if include_metadata:
            metadata = {
                "epsilon": float(effective_epsilon),
                "greedy_index": int(greedy_index),
                "explored": bool(explored),
                "allowed_indices": allowed_indices.tolist(),
                "forced_default_first_action": False,
                "q_values": q_values.tolist(),
                "selection_scores": q_values.tolist(),
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
        del native_cycle_cost, residual_ratio
        if self.config.td_algorithm == "true_online_sarsa":
            return self._true_online_sarsa_update(
                features=features,
                action_index=action_index,
                cost=cost,
                next_features=next_features,
                next_action_index=next_action_index,
                terminal=terminal,
            )
        if self.config.td_algorithm == "sarsa":
            return self._sarsa_update(
                features=features,
                action_index=action_index,
                cost=cost,
                next_features=next_features,
                next_action_index=next_action_index,
                terminal=terminal,
            )
        features = np.asarray(features, dtype=float)
        current_q = float(self.q_values(features)[int(action_index)])
        if terminal:
            expected_next_q = 0.0
        else:
            next_q_values = self.q_values(next_features)
            next_probabilities = self._probabilities_from_q(
                next_q_values,
                epsilon=self.epsilon,
                allowed_indices=self.allowed_action_indices(next_cycle),
            )
            expected_next_q = float(next_probabilities @ next_q_values)
        td_error = float(cost + self.config.gamma * expected_next_q - current_q)

        self.eligibility *= float(self.config.gamma * self.config.trace_lambda)
        action_gradient = self._action_gradient(int(action_index), features)
        self.eligibility += action_gradient
        if self.config.l2 > 0.0:
            self.theta *= max(0.0, 1.0 - float(self.config.alpha * self.config.l2))
        self.td_counts += np.abs(action_gradient)
        step_sizes = np.full_like(self.theta, float(self.config.alpha))
        if self.config.td_decay_power > 0.0:
            step_sizes /= np.maximum(self.td_counts, 1.0) ** float(
                self.config.td_decay_power
            )
        self.theta += float(td_error) * step_sizes * self.eligibility
        self.steps += 1
        return td_error

    def _sarsa_update(
        self,
        *,
        features: np.ndarray,
        action_index: int,
        cost: float,
        next_features: np.ndarray,
        next_action_index: int | None,
        terminal: bool,
    ) -> float:
        if not terminal and next_action_index is None:
            raise ValueError("sarsa requires the next behavior action before updating")
        features = np.asarray(features, dtype=float)
        current_q = float(self.q_values(features)[int(action_index)])
        next_q = 0.0 if terminal else float(
            self.q_values(next_features)[int(next_action_index)]
        )
        td_error = float(cost + self.config.gamma * next_q - current_q)

        self.eligibility *= float(self.config.gamma * self.config.trace_lambda)
        action_gradient = self._action_gradient(int(action_index), features)
        self.eligibility += action_gradient
        if self.config.l2 > 0.0:
            self.theta *= max(0.0, 1.0 - float(self.config.alpha * self.config.l2))
        self.td_counts += np.abs(action_gradient)
        step_sizes = np.full_like(self.theta, float(self.config.alpha))
        if self.config.td_decay_power > 0.0:
            step_sizes /= np.maximum(self.td_counts, 1.0) ** float(
                self.config.td_decay_power
            )
        self.theta += float(td_error) * step_sizes * self.eligibility
        self.steps += 1
        return td_error

    def _true_online_sarsa_update(
        self,
        *,
        features: np.ndarray,
        action_index: int,
        cost: float,
        next_features: np.ndarray,
        next_action_index: int | None,
        terminal: bool,
    ) -> float:
        if not terminal and next_action_index is None:
            raise ValueError(
                "true_online_sarsa requires the next behavior action before updating"
            )
        action_features = self._action_gradient(int(action_index), features)
        if terminal:
            next_action_features = np.zeros_like(action_features)
        else:
            next_action_features = self._action_gradient(
                int(next_action_index),
                next_features,
            )
        current_q = float(np.sum(self.theta * action_features))
        next_q = float(np.sum(self.theta * next_action_features))
        td_error = float(cost + self.config.gamma * next_q - current_q)
        alpha = float(self.config.alpha)
        gamma_lambda = float(self.config.gamma * self.config.trace_lambda)
        trace_dot = float(np.sum(self.eligibility * action_features))
        self.eligibility *= gamma_lambda
        self.eligibility += action_features
        self.eligibility -= (
            alpha * gamma_lambda * trace_dot * action_features
        )
        self.theta += (
            alpha
            * (td_error + current_q - self.q_old)
            * self.eligibility
            - alpha * (current_q - self.q_old) * action_features
        )
        self.td_counts += np.abs(action_features)
        self.q_old = next_q
        self.steps += 1
        return td_error

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            self.episodes += 1
        self.eligibility.fill(0.0)
        self.q_old = 0.0

    def snapshot_learning_state(self) -> Dict[str, Any]:
        """Capture rollback state without rewinding the behavior-policy RNG."""

        return {
            "theta": self.theta.copy(),
            "eligibility": self.eligibility.copy(),
            "td_counts": self.td_counts.copy(),
            "monte_carlo_counts": self.monte_carlo_counts.copy(),
            "steps": int(self.steps),
            "episodes": int(self.episodes),
            "q_old": float(self.q_old),
            "initial_environment_weight": float(self._initial_environment_weight),
            "default_first_action_pending": bool(self._default_first_action_pending),
            "forced_initial_action_count": int(self.forced_initial_action_count),
        }

    def restore_learning_state(self, state: Dict[str, Any]) -> None:
        self.theta[:] = np.asarray(state["theta"], dtype=float)
        self.eligibility[:] = np.asarray(state["eligibility"], dtype=float)
        self.td_counts[:] = np.asarray(state["td_counts"], dtype=float)
        self.monte_carlo_counts[:] = np.asarray(
            state["monte_carlo_counts"], dtype=float
        )
        self.steps = int(state["steps"])
        self.episodes = int(state["episodes"])
        self.q_old = float(state["q_old"])
        self._initial_environment_weight = float(
            state["initial_environment_weight"]
        )
        self._default_first_action_pending = bool(
            state["default_first_action_pending"]
        )
        self.forced_initial_action_count = int(
            state["forced_initial_action_count"]
        )

    def monte_carlo_update(
        self,
        transitions: Sequence[tuple[np.ndarray, int, float]],
        *,
        baseline_cost_sec: float = 0.0,
    ) -> list[float]:
        if self.config.monte_carlo_alpha <= 0.0:
            return []
        returns = 0.0
        baseline = float(baseline_cost_sec)
        errors = []
        for features, action_index, cost in reversed(transitions):
            returns = float(cost + self.config.gamma * returns)
            action = int(action_index)
            feature_values = np.asarray(features, dtype=float)
            prediction = float(self.q_values(feature_values)[action])
            # A paired reference solve is an action-independent control variate.
            # Subtracting it lowers cross-instance variance without changing the
            # action ordering of the undiscounted finite-horizon objective.
            target = float(returns - baseline)
            error = float(target - prediction)
            action_gradient = self._action_gradient(action, feature_values)
            active = np.abs(action_gradient) > 1.0e-15
            self.monte_carlo_counts += np.abs(action_gradient)
            if self.config.monte_carlo_decay_power > 0.0:
                rates = np.zeros_like(self.theta)
                rates[active] = float(self.config.monte_carlo_alpha) / np.power(
                    self.monte_carlo_counts[active],
                    float(self.config.monte_carlo_decay_power),
                )
                self.theta += rates * error * action_gradient
            else:
                self.theta += float(self.config.monte_carlo_alpha) * error * action_gradient
            errors.append(error)
        return errors

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            steps=np.asarray(self.steps, dtype=np.int64),
            episodes=np.asarray(self.episodes, dtype=np.int64),
            td_counts=self.td_counts,
            monte_carlo_counts=self.monte_carlo_counts,
            default_first_action_pending=np.asarray(
                self._default_first_action_pending,
                dtype=np.bool_,
            ),
            forced_initial_action_count=np.asarray(
                self.forced_initial_action_count,
                dtype=np.int64,
            ),
            config=np.asarray(json.dumps(asdict(self.config))),
            **self._extra_checkpoint_arrays(),
        )

    def _extra_checkpoint_arrays(self) -> Dict[str, np.ndarray]:
        return {}

    def _load_extra_checkpoint_arrays(self, payload: Any) -> None:
        del payload

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            theta = np.asarray(payload["theta"], dtype=float)
            if theta.shape != self.theta.shape:
                raise ValueError(
                    f"Checkpoint theta shape {theta.shape} does not match {self.theta.shape}"
                )
            saved_config = json.loads(str(payload["config"].item()))
            saved_episodes = int(payload["episodes"].item())
            saved_weights = np.asarray(saved_config["weights"], dtype=float)
            if saved_weights.shape != self.weights.shape or not np.allclose(
                saved_weights,
                self.weights,
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError("Checkpoint action weights do not match the controller")
            saved_rbf_sigma = float(saved_config.get("action_rbf_sigma", 0.0))
            if not np.isclose(
                saved_rbf_sigma,
                float(self.config.action_rbf_sigma),
                atol=1.0e-12,
                rtol=0.0,
            ):
                raise ValueError("Checkpoint action RBF width does not match the controller")
            saved_basis_mode = str(saved_config.get("action_basis_mode", "legacy"))
            if saved_basis_mode != str(self.config.action_basis_mode):
                raise ValueError("Checkpoint action basis mode does not match the controller")
            saved_centers = np.asarray(
                saved_config.get("action_basis_centers", ()),
                dtype=float,
            )
            configured_centers = np.asarray(
                self.config.action_basis_centers,
                dtype=float,
            )
            if (
                saved_centers.shape != configured_centers.shape
                or not np.allclose(
                    saved_centers,
                    configured_centers,
                    atol=1.0e-12,
                    rtol=0.0,
                )
            ):
                raise ValueError("Checkpoint action basis centers do not match the controller")
            self.theta[:] = theta
            if restore_counters:
                self.steps = int(payload["steps"].item())
                self.episodes = saved_episodes
                if "td_counts" in payload:
                    counts = np.asarray(payload["td_counts"], dtype=float)
                    if counts.shape != self.td_counts.shape:
                        raise ValueError("Checkpoint TD counts do not match the controller")
                    self.td_counts[:] = counts
                if "monte_carlo_counts" in payload:
                    counts = np.asarray(payload["monte_carlo_counts"], dtype=float)
                    if counts.shape != self.monte_carlo_counts.shape:
                        raise ValueError("Checkpoint Monte Carlo counts do not match the controller")
                    self.monte_carlo_counts[:] = counts
            if "default_first_action_pending" in payload:
                saved_pending = bool(
                    payload["default_first_action_pending"].item()
                )
            else:
                saved_pending = saved_episodes == 0
            self._default_first_action_pending = bool(
                self.config.force_default_first_action and saved_pending
            )
            if "forced_initial_action_count" in payload:
                self.forced_initial_action_count = int(
                    payload["forced_initial_action_count"].item()
                )
            else:
                self.forced_initial_action_count = int(
                    self.config.force_default_first_action
                    and not self._default_first_action_pending
                )
            self._load_extra_checkpoint_arrays(payload)
        self.eligibility.fill(0.0)
        self.q_old = 0.0
        return {
            "path": str(path),
            "restored_steps": int(self.steps),
            "restored_episodes": int(self.episodes),
            "default_first_action_pending": bool(
                self._default_first_action_pending
            ),
            "forced_initial_action_count": int(
                self.forced_initial_action_count
            ),
            "saved_config": saved_config,
        }


def run_td_episode(
    *,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    problem_context: Sequence[float] | None = None,
    solve_tol: float,
    solve_max_cycles: int,
    learn: bool,
    explore: bool,
    epsilon: float | None = None,
    defer_monte_carlo_update: bool = False,
    record_action_metadata: bool = False,
    initial_environment_weight_override: float | None = None,
    fallback_attempt: Callable[[], Mapping[str, Any] | AttemptOutcome] | None = None,
    failure_penalty_sec: float | None = None,
    audit_hierarchy: bool = False,
) -> Dict[str, Any]:
    failure_penalty_sec = validate_failure_penalty(failure_penalty_sec)
    retain_failed_episodes = failure_penalty_sec is not None
    function_started = time.perf_counter()
    decision_runtime = 0.0
    feature_runtime = 0.0
    update_runtime = 0.0
    lifecycle_runtime = 0.0
    solve_runtime = 0.0
    action_counts: Dict[str, int] = {}
    cycle_actions: list[float] = []
    cycle_action_values: list[float] = []
    cycle_residuals: list[float] = []
    cycle_residual_ratios: list[float] = []
    cycle_times: list[float] = []
    cycle_explored: list[bool] = []
    cycle_greedy_indices: list[int] = []
    cycle_epsilons: list[float] = []
    cycle_selected_uncertainties: list[float] = []
    cycle_selected_q_values: list[float] = []
    cycle_uncertainty_td_scales: list[float] = []
    cycle_selection_scores: list[float] = []
    cycle_forced_default_actions: list[bool] = []
    td_errors: list[float] = []
    postfit_td_errors: list[float] = []
    episode_transitions: list[tuple[np.ndarray, int, float]] = []
    residual = float("inf")
    iterations = 0
    initial_environment_weight = float(controller.initial_environment_weight)
    last_weight = initial_environment_weight
    selection: Dict[str, Any] = {}
    pending_action: tuple[int, float, Dict[str, Any]] | None = None
    recovery: RecoveryOutcome | None = None
    episode_started = False
    learning_snapshot = None
    prep = None
    hierarchy_audit = {}
    fast_anchor_tail = bool(
        controller.config.adaptive_cycles is not None
        and int(controller.config.adaptive_cycles) > 0
        and controller.config.alpha == 0.0
        and controller.config.monte_carlo_alpha > 0.0
    )
    try:
        if learn and hasattr(controller, "snapshot_learning_state"):
            lifecycle_started = time.perf_counter()
            try:
                learning_snapshot = controller.snapshot_learning_state()
            finally:
                lifecycle_runtime += time.perf_counter() - lifecycle_started
        with create_env(**dict(mkw)) as env:
            prep = env.prepare_rl(params=augment_setup_params(dict(params)))
            if audit_hierarchy:
                audit_started = time.perf_counter()
                hierarchy_audit = {"hierarchy_fingerprint": env.hierarchy_fingerprint(),
                                   "initial_cycle": env.cycle}
                hierarchy_audit["hierarchy_audit_runtime"] = time.perf_counter() - audit_started
            if retain_failed_episodes:
                validate_runtime_cost(prep.setup_runtime_sec)
            initial_residual = float(prep.initial_residual_norm)
            initial_environment_weight = (
                float(prep.initial_relax_weight)
                if initial_environment_weight_override is None
                else float(initial_environment_weight_override)
            )
            residual = initial_residual
            previous_residual = initial_residual
            last_weight = initial_environment_weight
            last_cycle_time = 0.0
            lifecycle_started = time.perf_counter()
            try:
                controller.start_episode(
                    initial_environment_weight=initial_environment_weight,
                )
            finally:
                lifecycle_runtime += time.perf_counter() - lifecycle_started
            episode_started = True
            feature_started = time.perf_counter()
            features = encoder.encode(
                mkw=mkw,
                setup_params=params,
                problem_context=problem_context,
                initial_residual=initial_residual,
                residual=residual,
                previous_residual=previous_residual,
                cycle=0,
                last_weight=last_weight,
                last_cycle_time=last_cycle_time,
            )
            feature_runtime += float(time.perf_counter() - feature_started)
            if retain_failed_episodes and not np.all(np.isfinite(features)):
                raise InvalidObservationError("Nonfinite initial controller features")

            for cycle in range(int(solve_max_cycles)):
                adaptive_decision = bool(
                    not fast_anchor_tail
                    or cycle < int(controller.config.adaptive_cycles)
                )
                if adaptive_decision:
                    if pending_action is None:
                        decision_started = time.perf_counter()
                        action_index, action_value, selection = controller.select_action(
                            features,
                            explore=explore,
                            epsilon=epsilon,
                            cycle=cycle,
                            include_metadata=record_action_metadata,
                        )
                        decision_runtime += float(
                            time.perf_counter() - decision_started
                        )
                    else:
                        action_index, action_value, selection = pending_action
                        pending_action = None
                    weight = controller.environment_weight(
                        int(action_index),
                        float(last_weight),
                    )
                else:
                    action_index = int(controller.anchor_index)
                    action_value = float(controller.weights[action_index])
                    weight = controller.environment_weight(
                        int(action_index),
                        float(last_weight),
                    )
                try:
                    residual_new, cycle_time = env.step_rl(
                        relax_weight=float(weight),
                        sweeps_down=1,
                        sweeps_up=1,
                        tol=float(solve_tol),
                        max_cycles=int(solve_max_cycles),
                    )
                    if retain_failed_episodes:
                        validate_runtime_cost(cycle_time)
                except InvalidObservationError:
                    raise
                except Exception as step_exc:
                    if retain_failed_episodes and not isinstance(step_exc, AMGNativeError):
                        raise
                    failed_step = AttemptOutcome.from_exception(
                        step_exc,
                        elapsed_sec=0.0,
                    )
                    failed_step_runtime = float(failed_step.solve_runtime_sec)
                    if retain_failed_episodes:
                        validate_runtime_cost(failed_step_runtime)
                    solve_runtime += failed_step_runtime
                    iterations = cycle + 1
                    primary_payload = {
                        "runtime": float(prep.setup_runtime_sec + solve_runtime),
                        "setup_runtime": float(prep.setup_runtime_sec),
                        "native_solve_runtime": float(solve_runtime),
                        "solve_runtime": float(solve_runtime),
                        "infer_runtime": float(
                            feature_runtime + decision_runtime + update_runtime + lifecycle_runtime
                        ),
                        "failed": True,
                        "attempt_status": failed_step.status.value,
                        "failure_reason": failed_step.failure_reason,
                        "failure_stage": "solve",
                        "residual_norm": float(residual),
                        "iterations": int(iterations),
                    }
                    recovery = RecoveryOutcome(
                        primary=AttemptOutcome.from_mapping(primary_payload),
                        fallback=(
                            None
                            if fallback_attempt is None
                            else execute_attempt(fallback_attempt, require_valid_observation=retain_failed_episodes)
                        ),
                    )
                    learning_allowed = bool(retain_failed_episodes or not recovery.unrecovered_failure)
                    transition_cost = float(failed_step_runtime)
                    if recovery.fallback is not None and learning_allowed:
                        transition_cost += float(
                            recovery.fallback.end_to_end_runtime_sec
                        )
                    if retain_failed_episodes and recovery.unrecovered_failure:
                        transition_cost += float(failure_penalty_sec)
                    if learning_allowed:
                        episode_transitions.append(
                            (features.copy(), int(action_index), float(transition_cost))
                        )
                        if learn:
                            update_started = time.perf_counter()
                            td_error = controller.update(
                                features=features,
                                action_index=action_index,
                                cost=transition_cost,
                                next_features=features,
                                terminal=True,
                                next_cycle=iterations,
                                next_action_index=None,
                                native_cycle_cost=None,
                                residual_ratio=None,
                            )
                            update_runtime += float(
                                time.perf_counter() - update_started
                            )
                            td_errors.append(float(td_error))
                            if hasattr(controller, "last_postfit_td_error"):
                                postfit_td_errors.append(
                                    float(controller.last_postfit_td_error)
                                )
                    action_counts[str(action_index)] = int(
                        action_counts.get(str(action_index), 0) + 1
                    )
                    cycle_actions.append(float(weight))
                    cycle_action_values.append(float(action_value))
                    cycle_residuals.append(float(residual))
                    cycle_times.append(float(failed_step_runtime))
                    break
                cycle_time = float(cycle_time)
                solve_runtime += cycle_time
                iterations = cycle + 1
                native_status = env.last_step.status
                nonfinite_result = not math.isfinite(float(residual_new))
                terminated = native_status is SolveStatus.CONVERGED and not nonfinite_result
                truncated = bool(
                    nonfinite_result or native_status is SolveStatus.MAX_CYCLES
                    or ((not terminated) and iterations >= int(solve_max_cycles))
                )
                need_next_features = bool(
                    not (terminated or truncated)
                    and (
                        not fast_anchor_tail
                        or iterations < int(controller.config.adaptive_cycles)
                    )
                )
                next_features = None
                if need_next_features:
                    feature_started = time.perf_counter()
                    next_features = encoder.encode(
                        mkw=mkw,
                        setup_params=params,
                        problem_context=problem_context,
                        initial_residual=initial_residual,
                        residual=float(residual_new),
                        previous_residual=float(residual),
                        cycle=iterations,
                        last_weight=float(weight),
                        last_cycle_time=cycle_time,
                    )
                    feature_runtime += float(time.perf_counter() - feature_started)
                    if retain_failed_episodes and not np.all(np.isfinite(next_features)):
                        raise InvalidObservationError("Nonfinite successor controller features")

                transition_cost = float(cycle_time)
                residual_ratio = (
                    float(residual_new) / float(residual)
                    if np.isfinite(float(residual_new))
                    and np.isfinite(float(residual))
                    and float(residual) > 0.0
                    else None
                )
                if truncated:
                    primary = AttemptOutcome.from_mapping(
                        {
                            "runtime": float(prep.setup_runtime_sec + solve_runtime),
                            "setup_runtime": float(prep.setup_runtime_sec),
                            "solve_runtime": float(solve_runtime),
                            "infer_runtime": float(
                                feature_runtime + decision_runtime + update_runtime + lifecycle_runtime
                            ),
                            "failed": True,
                            "failure_reason": (
                                "nonfinite_residual" if nonfinite_result
                                else "max_cycles_reached_without_convergence"
                            ),
                            "failure_stage": "solve",
                            "residual_norm": float(residual_new),
                            "iterations": int(iterations),
                        }
                    )
                    recovery = RecoveryOutcome(
                        primary=primary,
                        fallback=(
                            None
                            if fallback_attempt is None
                            else execute_attempt(fallback_attempt, require_valid_observation=retain_failed_episodes)
                        ),
                    )
                    if recovery.fallback is not None and (retain_failed_episodes or recovery.recovered):
                        transition_cost += float(
                            recovery.fallback.end_to_end_runtime_sec
                        )
                    if retain_failed_episodes and recovery.unrecovered_failure:
                        transition_cost += float(failure_penalty_sec)
                next_action_index = None
                if (
                    learn
                    and controller.requires_next_action
                    and not (terminated or truncated)
                ):
                    assert next_features is not None
                    decision_started = time.perf_counter()
                    pending_action = controller.select_action(
                        next_features,
                        explore=explore,
                        epsilon=epsilon,
                        cycle=iterations,
                        include_metadata=record_action_metadata,
                    )
                    decision_runtime += float(
                        time.perf_counter() - decision_started
                    )
                    next_action_index = int(pending_action[0])
                learning_allowed = bool(
                    retain_failed_episodes or recovery is None or not recovery.unrecovered_failure
                )
                if learning_allowed:
                    if adaptive_decision or not episode_transitions:
                        episode_transitions.append((features.copy(), int(action_index), float(transition_cost)))
                    else:
                        tail_features, tail_action, accumulated_cost = episode_transitions[-1]
                        episode_transitions[-1] = (
                            tail_features,
                            tail_action,
                            float(accumulated_cost + transition_cost),
                        )
                if learn and learning_allowed and fast_anchor_tail:
                    controller.steps += 1
                elif learn and learning_allowed:
                    assert next_features is not None or terminated or truncated
                    update_started = time.perf_counter()
                    td_error = controller.update(
                        features=features,
                        action_index=action_index,
                        cost=transition_cost,
                        next_features=(features if next_features is None else next_features),
                        terminal=bool(terminated or truncated),
                        next_cycle=iterations,
                        next_action_index=next_action_index,
                        native_cycle_cost=cycle_time,
                        residual_ratio=residual_ratio,
                    )
                    update_runtime += float(time.perf_counter() - update_started)
                    td_errors.append(float(td_error))
                    if hasattr(controller, "last_postfit_td_error"):
                        postfit_td_errors.append(
                            float(controller.last_postfit_td_error)
                        )

                action_counts[str(action_index)] = int(action_counts.get(str(action_index), 0) + 1)
                cycle_actions.append(float(weight))
                cycle_action_values.append(float(action_value))
                cycle_residuals.append(float(residual_new))
                if residual_ratio is not None:
                    cycle_residual_ratios.append(float(residual_ratio))
                cycle_times.append(float(cycle_time))
                if record_action_metadata:
                    cycle_explored.append(bool(selection["explored"]))
                    cycle_greedy_indices.append(int(selection["greedy_index"]))
                    cycle_epsilons.append(float(selection["epsilon"]))
                    uncertainty = selection.get("uncertainty")
                    predicted_q = selection.get("q_values")
                    scores = selection.get("selection_scores")
                    cycle_selected_uncertainties.append(
                        float(uncertainty[action_index])
                        if uncertainty is not None
                        else 0.0
                    )
                    cycle_uncertainty_td_scales.append(
                        float(selection.get("uncertainty_td_scale", 0.0))
                    )
                    cycle_selected_q_values.append(
                        float(predicted_q[action_index])
                        if predicted_q is not None
                        else float("nan")
                    )
                    cycle_selection_scores.append(
                        float(scores[action_index])
                        if scores is not None
                        else float("nan")
                    )
                    cycle_forced_default_actions.append(
                        bool(selection.get("forced_default_first_action", False))
                    )
                if next_features is not None:
                    features = next_features
                previous_residual = float(residual)
                residual = float(residual_new)
                last_weight = float(weight)
                last_cycle_time = cycle_time
                if terminated or truncated:
                    break
            if (
                learn
                and (retain_failed_episodes or recovery is None or not recovery.unrecovered_failure)
                and controller.config.monte_carlo_alpha > 0.0
                and not defer_monte_carlo_update
            ):
                update_started = time.perf_counter()
                td_errors.extend(controller.monte_carlo_update(episode_transitions))
                update_runtime += float(time.perf_counter() - update_started)
            if recovery is not None and recovery.unrecovered_failure and not retain_failed_episodes:
                if learning_snapshot is not None:
                    lifecycle_started = time.perf_counter()
                    try:
                        controller.restore_learning_state(learning_snapshot)
                    finally:
                        lifecycle_runtime += time.perf_counter() - lifecycle_started
            else:
                update_started = time.perf_counter()
                try:
                    controller.finish_episode(learned=learn)
                finally:
                    update_runtime += float(time.perf_counter() - update_started)

        failure_reason = classify_rl_failure(
            residual_norm=float(residual),
            iterations=int(iterations),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
        )
        outcome = {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "native_solve_runtime": float(solve_runtime),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": float(feature_runtime + decision_runtime + update_runtime + lifecycle_runtime),
            "feature_runtime": float(feature_runtime),
            "decision_runtime": float(decision_runtime),
            "update_runtime": float(update_runtime),
            "lifecycle_runtime": float(lifecycle_runtime),
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "residual_norm": float(residual),
            "iterations": int(iterations),
            "initial_environment_weight": initial_environment_weight,
            "final_w": float(last_weight),
            "action_counts": action_counts,
            "cycle_actions": cycle_actions,
            "cycle_action_values": cycle_action_values,
            "cycle_residuals": cycle_residuals,
            "cycle_residual_ratios": cycle_residual_ratios,
            "cycle_times": cycle_times,
            "cycle_explored": cycle_explored,
            "cycle_greedy_indices": cycle_greedy_indices,
            "cycle_epsilons": cycle_epsilons,
            "cycle_selected_uncertainties": cycle_selected_uncertainties,
            "cycle_selected_q_values": cycle_selected_q_values,
            "cycle_uncertainty_td_scales": cycle_uncertainty_td_scales,
            "cycle_selection_scores": cycle_selection_scores,
            "cycle_forced_default_actions": cycle_forced_default_actions,
            "mean_abs_td_error": float(np.mean(np.abs(td_errors))) if td_errors else 0.0,
            "td_errors": td_errors,
            "postfit_td_errors": postfit_td_errors,
            "epsilon": float(controller.epsilon),
            "selection": selection,
            "controller_update_committed": bool(
                learn and (retain_failed_episodes or recovery is None or not recovery.unrecovered_failure)
            ),
        }
        if recovery is not None:
            original_primary = recovery.primary
            recovery = RecoveryOutcome(
                primary=AttemptOutcome(
                    status=original_primary.status,
                    setup_runtime_sec=float(outcome["setup_runtime"]),
                    solve_runtime_sec=float(outcome["native_solve_runtime"]),
                    controller_runtime_sec=float(outcome["infer_runtime"]),
                    failure_reason=original_primary.failure_reason,
                    residual_norm=float(outcome["residual_norm"]),
                    cycles=int(outcome["iterations"]),
                    result=dict(outcome),
                ),
                fallback=recovery.fallback,
            )
            recovered_outcome = recovery.to_result()
            recovered_outcome.update(
                {
                    key: value
                    for key, value in outcome.items()
                    if key not in recovered_outcome
                }
            )
            recovered_outcome["recovery_protocol_applied"] = True
            recovered_outcome["controller_update_committed"] = bool(
                learn and (retain_failed_episodes or not recovery.unrecovered_failure)
            )
            outcome = recovered_outcome
        else:
            outcome.update(
                {
                    "recovery_protocol_applied": False,
                    "primary_status": "success" if not failure_reason else "nonconvergence",
                    "fallback_used": False,
                    "recovered": False,
                    "unrecovered_failure": bool(failure_reason),
                }
            )
        if defer_monte_carlo_update:
            outcome["_monte_carlo_transitions"] = episode_transitions
        outcome.update(hierarchy_audit)
        outcome["failure_feedback_mode"] = "budgeted_penalty" if retain_failed_episodes else "rollback_unrecovered"
        outcome["failure_penalty_sec"] = (
            float(failure_penalty_sec or 0.0) if outcome.get("unrecovered_failure", False) else 0.0
        )
        return outcome
    except Exception as exc:
        if isinstance(exc, InvalidObservationError) or (retain_failed_episodes and not isinstance(exc, AMGNativeError)):
            # Broken measurements and estimator/programming errors are not
            # numerical solver failures. Roll back any partial updates and
            # propagate so the experiment cannot keep training on them.
            if learning_snapshot is not None:
                controller.restore_learning_state(learning_snapshot)
            raise
        controller_runtime = float(feature_runtime + decision_runtime + update_runtime + lifecycle_runtime)
        native_failure = AttemptOutcome.from_exception(
            exc,
            elapsed_sec=float(time.perf_counter() - function_started),
        )
        if episode_started and native_failure.status is AttemptStatus.SETUP_FAILURE:
            native_failure = AttemptOutcome(
                status=AttemptStatus.SOLVE_FAILURE,
                setup_runtime_sec=native_failure.setup_runtime_sec,
                solve_runtime_sec=native_failure.solve_runtime_sec,
                controller_runtime_sec=native_failure.controller_runtime_sec,
                failure_reason=native_failure.failure_reason,
                residual_norm=native_failure.residual_norm,
                cycles=native_failure.cycles,
                result=dict(native_failure.result),
            )
        setup_runtime = float(
            (0.0 if prep is None else prep.setup_runtime_sec)
            + native_failure.setup_runtime_sec
        )
        solve_runtime += float(native_failure.solve_runtime_sec)
        primary = AttemptOutcome.from_mapping({
            "runtime": float(setup_runtime + solve_runtime),
            "setup_runtime": setup_runtime,
            "native_solve_runtime": float(solve_runtime),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": controller_runtime,
            "failed": True,
            "attempt_status": native_failure.status.value,
            "failure_reason": native_failure.failure_reason,
            "failure_stage": native_failure.failure_stage,
            "residual_norm": float(residual),
            "iterations": int(iterations),
        })
        setup_construction_failure = bool(
            native_failure.status is AttemptStatus.SETUP_FAILURE
            and not episode_started
        )
        recovery = RecoveryOutcome(
            primary=primary,
            fallback=(
                None
                if setup_construction_failure or fallback_attempt is None
                else execute_attempt(fallback_attempt, require_valid_observation=retain_failed_episodes)
            ),
        )
        if learning_snapshot is not None:
            lifecycle_started = time.perf_counter()
            try:
                controller.restore_learning_state(learning_snapshot)
            finally:
                lifecycle_runtime += time.perf_counter() - lifecycle_started
        elif episode_started:
            update_started = time.perf_counter()
            try:
                controller.finish_episode(learned=False)
            finally:
                update_runtime += time.perf_counter() - update_started
        # Rollback/finalization is completed after the failed attempt and must
        # be charged once, without contaminating the native cycle targets.
        recovery = RecoveryOutcome(
            primary=replace(
                recovery.primary,
                controller_runtime_sec=float(
                    feature_runtime + decision_runtime + update_runtime + lifecycle_runtime
                ),
            ),
            fallback=recovery.fallback,
        )
        outcome = recovery.to_result()
        outcome.update({
            "feature_runtime": float(feature_runtime),
            "decision_runtime": float(decision_runtime),
            "update_runtime": float(update_runtime),
            "lifecycle_runtime": float(lifecycle_runtime),
            "residual_norm": float(residual),
            "iterations": int(iterations),
            "initial_environment_weight": initial_environment_weight,
            "final_w": float("nan"),
            "action_counts": action_counts,
            "cycle_actions": cycle_actions,
            "cycle_action_values": cycle_action_values,
            "cycle_residuals": cycle_residuals,
            "cycle_residual_ratios": cycle_residual_ratios,
            "cycle_times": cycle_times,
            "cycle_explored": cycle_explored,
            "cycle_greedy_indices": cycle_greedy_indices,
            "cycle_epsilons": cycle_epsilons,
            "cycle_selected_uncertainties": cycle_selected_uncertainties,
            "cycle_selected_q_values": cycle_selected_q_values,
            "cycle_uncertainty_td_scales": cycle_uncertainty_td_scales,
            "cycle_selection_scores": cycle_selection_scores,
            "cycle_forced_default_actions": cycle_forced_default_actions,
            "mean_abs_td_error": float(np.mean(np.abs(td_errors))) if td_errors else 0.0,
            "td_errors": td_errors,
            "postfit_td_errors": postfit_td_errors,
            "epsilon": float(controller.epsilon),
            "selection": {},
            "recovery_protocol_applied": not setup_construction_failure,
            "controller_update_committed": False,
            **hierarchy_audit,
        })
        return outcome
