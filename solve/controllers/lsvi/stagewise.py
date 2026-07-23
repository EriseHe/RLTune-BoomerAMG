from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common.action_space import (
    build_action_basis,
    joint_action_features,
)
from solve.controllers.common import _greedy_cost_index, _json_dataclass
from solve.controllers.sarsa.config import ExpectedSarsaLambdaConfig

from .config import StagewiseLsviLcbSpec


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
