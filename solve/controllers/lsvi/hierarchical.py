from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np

from solve.controllers.common import (
    _SharedActionLcbController,
    _json_dataclass,
    _rank_one_inverse_update,
)
from solve.controllers.sarsa import ExpectedSarsaLambdaConfig


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

_HIERARCHICAL_LSVI_CHECKPOINT_VERSION = 1


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
