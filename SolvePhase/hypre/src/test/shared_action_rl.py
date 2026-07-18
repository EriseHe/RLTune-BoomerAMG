from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from online_td_lambda import (
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


@dataclass(frozen=True)
class RecursiveLstdqLcbSpec:
    ridge: float = 1.0
    uncertainty_beta: float = 2.0
    residual_floor_sec: float = 1.0e-3
    q_max_sec: float = 0.1
    inverse_denominator_floor: float = 1.0e-10


def _json_dataclass(value: Any) -> Dict[str, Any]:
    return json.loads(json.dumps(asdict(value)))


def _greedy_cost_index(
    values: np.ndarray,
    *,
    allowed_indices: np.ndarray,
    anchor_index: int,
) -> int:
    allowed = np.asarray(allowed_indices, dtype=int)
    if allowed.size == 0:
        raise ValueError("At least one action must be allowed")
    best_value = float(np.min(np.asarray(values, dtype=float)[allowed]))
    tied = allowed[
        np.isclose(values[allowed], best_value, atol=1.0e-12, rtol=0.0)
    ]
    if np.any(tied == int(anchor_index)):
        return int(anchor_index)
    return int(tied[0])


class _SharedActionLcbController:
    """Common action representation and behavior policy for linear LCB control."""

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


class RecursiveMonteCarloLcbController(_SharedActionLcbController):
    """Shared-action recursive least-squares control on full episodic returns."""

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
        super().__init__(feature_dim=feature_dim, config=config, seed=seed)
        self.spec = spec
        self.design_inverse = np.eye(self.joint_dim, dtype=float) / float(spec.ridge)
        self.b = np.zeros(self.joint_dim, dtype=float)
        self.theta = np.zeros(self.joint_dim, dtype=float)
        self.return_residual_sum_squares = 0.0
        self.return_residual_count = 0
        self.sample_count = 0
        self._pending: list[tuple[np.ndarray, float]] = []

    @property
    def requires_next_action(self) -> bool:
        return False

    @property
    def residual_scale(self) -> float:
        if self.return_residual_count <= 0:
            return float(self.spec.residual_floor_sec)
        return max(
            math.sqrt(
                self.return_residual_sum_squares / float(self.return_residual_count)
            ),
            float(self.spec.residual_floor_sec),
        )

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
        quadratic = np.sum(projected * joint, axis=1)
        uncertainty = float(self.residual_scale) * np.sqrt(
            np.maximum(quadratic, 0.0)
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
    ) -> float:
        del next_features, next_cycle, next_action_index
        joint = self.state_action_features(features)[int(action_index)].copy()
        self._pending.append((joint, float(cost)))
        self.steps += 1
        return float(cost - joint @ self.theta) if terminal else 0.0

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            return_to_go = 0.0
            for joint, cost in reversed(self._pending):
                return_to_go = float(cost) + float(self.config.gamma) * return_to_go
                prediction = float(joint @ self.theta)
                innovation = return_to_go - prediction
                projected = self.design_inverse @ joint
                denominator = 1.0 + float(joint @ projected)
                if denominator <= 1.0e-12 or not np.isfinite(denominator):
                    raise FloatingPointError("recursive MC inverse update became singular")
                self.design_inverse -= np.outer(projected, projected) / denominator
                self.b += return_to_go * joint
                self.theta = self.design_inverse @ self.b
                self.return_residual_sum_squares += innovation * innovation
                self.return_residual_count += 1
                self.sample_count += 1
            self.episodes += 1
        self._pending.clear()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            theta=self.theta,
            design_inverse=self.design_inverse,
            b=self.b,
            return_residual_sum_squares=np.asarray(
                self.return_residual_sum_squares, dtype=float
            ),
            return_residual_count=np.asarray(
                self.return_residual_count, dtype=np.int64
            ),
            sample_count=np.asarray(self.sample_count, dtype=np.int64),
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
            if json.loads(str(payload["config"].item())) != _json_dataclass(self.config):
                raise ValueError("Checkpoint recursive MC config does not match")
            if json.loads(str(payload["spec"].item())) != _json_dataclass(self.spec):
                raise ValueError("Checkpoint recursive MC spec does not match")
            self.theta[:] = np.asarray(payload["theta"], dtype=float)
            self.design_inverse[:] = np.asarray(
                payload["design_inverse"], dtype=float
            )
            self.b[:] = np.asarray(payload["b"], dtype=float)
            self.return_residual_sum_squares = float(
                payload["return_residual_sum_squares"].item()
            )
            self.return_residual_count = int(
                payload["return_residual_count"].item()
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
            "uncertainty_beta": float(self.spec.uncertainty_beta),
            "stored_transition_count": 0,
        }


class RecursiveLstdqLcbController(_SharedActionLcbController):
    """Recursive on-policy LSTDQ(lambda) with sandwich LCB uncertainty."""

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
        self.sample_count = 0
        self.inverse_rebuild_count = 0

    @property
    def requires_next_action(self) -> bool:
        return True

    def start_episode(self, *, initial_environment_weight: float | None = None) -> None:
        self.trace.fill(0.0)
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
        projected = joint @ self.a_inverse
        quadratic = np.einsum(
            "ai,ij,aj->a",
            projected,
            self.moment_covariance,
            projected,
            optimize=True,
        )
        uncertainty = np.sqrt(np.maximum(quadratic, 0.0))
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
    ) -> float:
        del next_cycle
        joint = self.state_action_features(features)[int(action_index)]
        if terminal:
            next_joint = np.zeros(self.joint_dim, dtype=float)
        else:
            if next_action_index is None:
                raise ValueError("recursive LSTDQ requires the executed next action")
            next_joint = self.state_action_features(next_features)[int(next_action_index)]
        difference = joint - float(self.config.gamma) * next_joint
        td_error = float(cost + self.config.gamma * (next_joint @ self.theta) - joint @ self.theta)
        self.trace = (
            float(self.config.gamma) * float(self.config.trace_lambda) * self.trace
            + joint
        )
        moment = self.trace * td_error
        self.moment_covariance += np.outer(moment, moment)

        projected_trace = self.a_inverse @ self.trace
        projected_difference = difference @ self.a_inverse
        denominator = 1.0 + float(difference @ projected_trace)
        self.a_matrix += np.outer(self.trace, difference)
        self.b += float(cost) * self.trace
        if (
            not np.isfinite(denominator)
            or abs(denominator) < float(self.spec.inverse_denominator_floor)
        ):
            self.a_inverse = np.linalg.pinv(self.a_matrix)
            self.inverse_rebuild_count += 1
        else:
            self.a_inverse -= (
                np.outer(projected_trace, projected_difference) / denominator
            )
        self.theta = self.a_inverse @ self.b
        if not np.all(np.isfinite(self.theta)):
            raise FloatingPointError("recursive LSTDQ parameters became non-finite")
        self.steps += 1
        self.sample_count += 1
        return td_error

    def finish_episode(self, *, learned: bool) -> None:
        if learned:
            self.episodes += 1
        self.trace.fill(0.0)

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
            spec=np.asarray(json.dumps(asdict(self.spec))),
            **self._common_checkpoint_fields(),
        )

    def load(self, path: Path, *, restore_counters: bool = True) -> Dict[str, Any]:
        with np.load(path, allow_pickle=False) as payload:
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
    ) -> float:
        del next_action_index
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
