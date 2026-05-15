from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._amg_action_features import ParameterSpaceSpec
from ._candidate_subset import Tune7CandidateMixer, Tune7LatticeCatalog
from ._online_regression import OnlineRidgeRegressor
from ._tune7_online_features import Tune7FeatureBuilder


@dataclass(frozen=True)
class SharedLinTSv4Step:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class SharedLinTS_AMG_v4:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        parameter_spec: ParameterSpaceSpec,
        l2_reg: float = 1.0,
        nu: float = 0.5,
        context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
        candidate_pool_size: int = 512,
        always_include_arms: Optional[Sequence[int]] = None,
        elite_cache_size: int = 128,
        candidate_elite_size: int = 128,
        feature_builder: Optional[Tune7FeatureBuilder] = None,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")
        if l2_reg <= 0.0:
            raise ValueError("l2_reg must be > 0")
        if not np.isfinite(float(nu)) or float(nu) <= 0.0:
            raise ValueError("nu must be finite and > 0")

        self.actions = tuple(feature_builder.actions) if feature_builder is not None else tuple(actions)
        self.parameter_spec = parameter_spec
        self.feature_builder = feature_builder or Tune7FeatureBuilder(
            self.actions,
            self.parameter_spec,
            tuple(int(v) for v in context_interaction_indices),
        )
        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.g_dim = int(self.feature_builder.g_dim)
        self.context_interaction_indices = tuple(int(v) for v in context_interaction_indices)
        for idx in self.context_interaction_indices:
            if not (0 <= idx < self.d_x):
                raise ValueError("context_interaction_indices must be within [0, context_dim)")
        self.d_phi = self.d_x + (1 + len(self.context_interaction_indices)) * self.g_dim

        self.nu = float(nu)
        self.rng = np.random.default_rng(seed)
        self.regressor = OnlineRidgeRegressor(dim=self.d_phi, l2_reg=float(l2_reg))
        if int(candidate_elite_size) < 0 or int(candidate_elite_size) > int(candidate_pool_size):
            raise ValueError("candidate_elite_size must be within [0, candidate_pool_size]")
        self._selector = Tune7CandidateMixer(
            self.K,
            catalog=Tune7LatticeCatalog(
                regime_ids=self.feature_builder.regime_indices,
                lattice_indices=self.feature_builder.lattice_indices,
                grid_shape=self.feature_builder.grid_shape,
            ),
            candidate_pool_size=int(candidate_pool_size),
            always_include_arms=always_include_arms,
            elite_cache_size=int(elite_cache_size),
            rng=self.rng,
            local_size=0,
            elite_size=int(candidate_elite_size),
            random_size=int(candidate_pool_size) - int(candidate_elite_size),
            random_mode="uniform",
        )

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: list[SharedLinTSv4Step] = []
        self.candidate_stats_history: list[Dict[str, int | str]] = []

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _arm_from_params(self, params: Mapping[str, Any]) -> int:
        return int(self.feature_builder.arm_for_params(params))

    def _phi(self, x: np.ndarray, arm: int) -> np.ndarray:
        return np.asarray(
            self.feature_builder.phi_v4_matrix(x, np.asarray([int(arm)], dtype=int))[0],
            dtype=float,
        )

    def _phi_matrix(self, x: np.ndarray, arms: np.ndarray) -> np.ndarray:
        return self.feature_builder.phi_v4_matrix(x, np.asarray(arms, dtype=int).reshape(-1))

    def _record_prediction(self, *, arm: int, pred_mean: float, pred_uncert: float) -> Dict[str, Any]:
        self.history.append(
            SharedLinTSv4Step(
                t=self.t + 1,
                arm_index=int(arm),
                loss=float("nan"),
                pred_mean=float(pred_mean),
                pred_uncert=float(pred_uncert),
            )
        )
        return dict(self.actions[int(arm)])

    def predict_from_candidate_arms(
        self,
        context: Iterable[float],
        candidate_arms: Sequence[int],
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        x = self._validate_x(np.asarray(list(context), dtype=float))
        cand = np.asarray(candidate_arms, dtype=int).reshape(-1)
        if cand.size == 0:
            raise ValueError("candidate_arms must be non-empty")

        Phi = self._phi_matrix(x, cand)
        theta = self.regressor.sample_theta(scale=self.nu, rng=self.rng)
        sampled_loss = Phi @ theta
        loc = int(np.argmin(sampled_loss))
        arm = int(cand[loc])
        phi = Phi[loc]
        pred_mean = float(self.regressor.predict(phi))
        pred_uncert = float(self.regressor.uncertainty(phi)[0])

        self._last_phi = np.asarray(phi, dtype=float)
        self._last_arm = int(arm)
        params = self._record_prediction(arm=arm, pred_mean=pred_mean, pred_uncert=pred_uncert)
        return params, {"pred_mean": pred_mean, "pred_uncert": pred_uncert}

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        candidate_arms, stats = self._selector.candidate_subset(return_stats=True)
        self.candidate_stats_history.append(dict(stats))
        params, _info = self.predict_from_candidate_arms(context, candidate_arms)
        return params

    def update(self, loss: float, *, bounded_loss: Optional[float] = None) -> None:
        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")
        raw_loss = float(loss)
        if not np.isfinite(raw_loss):
            raise ValueError("loss must be finite")
        target = float(raw_loss if bounded_loss is None else bounded_loss)
        if not np.isfinite(target):
            raise ValueError("bounded_loss must be finite")

        self.regressor.observe(self._last_phi, target)
        self._selector.observe(self._last_arm, raw_loss)
        last = self.history[-1]
        self.history[-1] = SharedLinTSv4Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=raw_loss,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )
        self.t += 1
        self._last_phi = None
        self._last_arm = None

    def observe(
        self,
        *,
        context: Iterable[float],
        params: Mapping[str, Any],
        loss: float,
        bounded_loss: Optional[float] = None,
    ) -> None:
        x = self._validate_x(np.asarray(list(context), dtype=float))
        arm = self._arm_from_params(params)
        self._last_phi = self._phi(x, arm)
        self._last_arm = int(arm)
        self.history.append(
            SharedLinTSv4Step(
                t=self.t + 1,
                arm_index=int(arm),
                loss=float("nan"),
                pred_mean=float("nan"),
                pred_uncert=float("nan"),
            )
        )
        self.update(float(loss), bounded_loss=bounded_loss)
