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

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._amg_action_features import (
    GenericActionFeatureEncoder,
    ParameterSpaceSpec,
    action_from_ordered_values,
    action_key_from_parameter_space_spec,
)
from ._candidate_subset import CandidateSelector


@dataclass(frozen=True)
class SharedLinUCBv4Step:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


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
        initial_guess: Optional[Sequence[Any] | Mapping[str, Any]] = None,
        initial_guess_rounds: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")
        if l2_reg <= 0.0:
            raise ValueError("l2_reg must be > 0")
        if candidate_strategy not in {"uniform", "adaptive_local"}:
            raise ValueError("candidate_strategy must be 'uniform' or 'adaptive_local'")
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

        self.actions: List[Dict[str, Any]] = [dict(a) for a in actions]
        self.parameter_spec = parameter_spec
        self._encoder = GenericActionFeatureEncoder(parameter_spec)

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

        self.candidate_strategy = str(candidate_strategy)
        self.candidate_pool_size_burnin = (
            int(candidate_pool_size_burnin) if candidate_pool_size_burnin is not None else None
        )
        self.candidate_pool_burnin_rounds = int(candidate_pool_burnin_rounds)
        self.alpha_decay_burnin_rounds = int(alpha_decay_burnin_rounds)
        self.local_neighbor_radius = int(local_neighbor_radius)
        self.candidate_local_fraction = float(candidate_local_fraction)
        self.candidate_elite_fraction = float(candidate_elite_fraction)

        self.rng = np.random.default_rng(seed)
        self._cand = CandidateSelector(
            self.K,
            candidate_pool_size=candidate_pool_size,
            always_include_arms=always_include_arms,
            elite_cache_size=int(elite_cache_size),
            elite_rank_metric=str(elite_rank_metric),
            rng=self.rng,
        )

        self.A_inv = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.b = np.zeros(self.d_phi, dtype=float)

        self._g_actions = self._encoder.encode_actions(self.actions)

        self._param_by_name = {param.name: param for param in self.parameter_spec.parameters}
        self._param_value_to_index = {
            param.name: {value: idx for idx, value in enumerate(param.values)}
            for param in self.parameter_spec.parameters
        }
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
            for idx, action in enumerate(self.actions):
                action_key = action_key_from_parameter_space_spec(action, self.parameter_spec)
                if action_key == initial_key:
                    self._initial_guess_arm = int(idx)
                    break
            if self._initial_guess_arm is None:
                raise ValueError("initial_guess does not match any action in the provided action set")
        self._action_key_to_arm = None
        if self.candidate_strategy == "adaptive_local":
            self._action_key_to_arm = {
                action_key_from_parameter_space_spec(action, self.parameter_spec): int(i)
                for i, action in enumerate(self.actions)
            }

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[SharedLinUCBv4Step] = []
        self.candidate_stats_history: List[Dict[str, int | str]] = []

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _phi(self, x: np.ndarray, arm: int) -> np.ndarray:
        g = self._g_actions[int(arm)]
        phi = np.empty(self.d_phi, dtype=float)
        phi[: self.d_x] = x
        offset = self.d_x
        phi[offset : offset + self.g_dim] = g
        offset += self.g_dim
        for idx in self.context_interaction_indices:
            phi[offset : offset + self.g_dim] = float(x[idx]) * g
            offset += self.g_dim
        return phi

    def _score_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        G = self._g_actions if arms is None else self._g_actions[np.asarray(arms, dtype=int)]

        theta = self.A_inv @ self.b
        theta_x = theta[: self.d_x]
        offset = self.d_x
        theta_g = theta[offset : offset + self.g_dim]
        offset += self.g_dim

        base = float(theta_x @ x) + np.einsum("ij,j->i", G, theta_g, optimize=False)
        mean = np.asarray(base, dtype=float)
        for idx in self.context_interaction_indices:
            theta_block = theta[offset : offset + self.g_dim]
            offset += self.g_dim
            mean = mean + float(x[idx]) * np.einsum("ij,j->i", G, theta_block, optimize=False)

        c = np.concatenate(
            [x, np.zeros((1 + len(self.context_interaction_indices)) * self.g_dim, dtype=float)],
            axis=0,
        )
        I = np.eye(self.g_dim, dtype=float)
        P_blocks = [np.zeros((self.d_x, self.g_dim), dtype=float), I]
        for idx in self.context_interaction_indices:
            P_blocks.append(float(x[idx]) * I)
        P = np.vstack(P_blocks)

        Ainv = self.A_inv
        Ac = Ainv @ c
        q0 = float(c @ Ac)
        AP = Ainv @ P
        u = P.T @ Ac
        M = P.T @ AP

        quad = q0 + 2.0 * np.einsum("ij,j->i", G, u, optimize=False) + np.einsum("ij,jk,ik->i", G, M, G, optimize=False)
        uncert = np.sqrt(np.maximum(0.0, quad))
        score = mean - float(alpha) * uncert
        return score, mean, uncert

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

    def _local_neighbor_arms(self, center_arms: np.ndarray) -> np.ndarray:
        if self._action_key_to_arm is None or self.local_neighbor_radius <= 0:
            return np.zeros(0, dtype=int)

        neighbors = set()
        for arm in np.asarray(center_arms, dtype=int):
            action = dict(self.actions[int(arm)])
            for param in self.parameter_spec.parameters:
                if not self._is_param_active(param, action):
                    continue
                current_value = action.get(param.name, param.default)
                if param.kind == "categorical":
                    # Categorical knobs do not define a numeric neighborhood. Changes
                    # in categorical signatures are left to the global-random slice.
                    continue

                value_to_index = self._param_value_to_index[param.name]
                idx = value_to_index.get(current_value)
                if idx is None:
                    continue
                for delta in range(1, self.local_neighbor_radius + 1):
                    for sign in (-1, 1):
                        j = int(idx) + sign * int(delta)
                        if not (0 <= j < len(param.values)):
                            continue
                        candidate = dict(action)
                        candidate[param.name] = param.values[j]
                        key = action_key_from_parameter_space_spec(candidate, self.parameter_spec)
                        neighbor = self._action_key_to_arm.get(key)
                        if neighbor is not None and int(neighbor) != int(arm):
                            neighbors.add(int(neighbor))
        if not neighbors:
            return np.zeros(0, dtype=int)
        return np.asarray(sorted(neighbors), dtype=int)

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
        if M is None or int(M) >= self.K:
            score, mean, uncert = self._score_subset(x, arms=None, alpha=alpha_eff)
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
            if self.candidate_strategy == "adaptive_local":
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

            score, mean, uncert = self._score_subset(x, arms=cand, alpha=alpha_eff)
            best_score = float(np.min(score))
            best_loc = np.flatnonzero(score <= best_score + 1e-12)
            loc = int(self.rng.choice(best_loc))
            arm = int(cand[loc])
            best_mean = float(mean[loc])
            best_unc = float(uncert[loc])
            self.candidate_stats_history.append(cand_stats)

        self._last_phi = self._phi(x, arm)
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

    def update(self, loss: float) -> None:
        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")

        phi = self._last_phi
        arm = int(self._last_arm)
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")

        self._cand.observe(arm, y)

        u = self.A_inv @ phi
        denom = 1.0 + float(phi @ u)
        if denom <= 0.0 or not np.isfinite(denom):
            A = np.linalg.inv(self.A_inv)
            A = A + np.outer(phi, phi)
            self.A_inv = np.linalg.inv(A)
        else:
            self.A_inv = self.A_inv - np.outer(u, u) / denom

        self.b = self.b + y * phi

        last = self.history[-1]
        self.history[-1] = SharedLinUCBv4Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
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
