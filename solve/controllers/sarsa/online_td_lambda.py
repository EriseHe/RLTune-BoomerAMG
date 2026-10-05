"""Online TD(lambda) controller updates and checkpoint persistence."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from solve.controllers.common.action_space import (
    build_action_basis,
    joint_action_features,
)
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.core.episode import run_td_episode

from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig


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
