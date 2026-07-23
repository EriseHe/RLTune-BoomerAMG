from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common import _greedy_cost_index, _json_dataclass
from solve.controllers.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
)


@dataclass(frozen=True)
class BootstrapSarsaSpec:
    members: int = 5
    episode_inclusion_probability: float = 0.8
    uncertainty_beta: float = 1.0


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
