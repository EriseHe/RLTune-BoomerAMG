from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._amg_action_features import ParameterSpaceSpec
from ._candidate_subset import Tune7CandidateMixer, Tune7LatticeCatalog
from ._online_regression import OnlineRidgeRegressor
from ._tune7_online_features import Tune7FeatureBuilder


@dataclass(frozen=True)
class NystromKernelUCBStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class NystromKernelUCB_AMG_v4:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        parameter_spec: ParameterSpaceSpec,
        landmark_x: np.ndarray,
        landmark_z: np.ndarray,
        landmark_regime: np.ndarray,
        sigma_x: float,
        sigma_z: float,
        l2_reg: float = 1e-2,
        beta0: float = 1.0,
        context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
        candidate_pool_size: int = 384,
        always_include_arms: Optional[Sequence[int]] = None,
        elite_cache_size: int = 128,
        candidate_local_size: int = 192,
        candidate_elite_size: int = 64,
        local_seed_count: int = 4,
        feature_builder: Optional[Tune7FeatureBuilder] = None,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")

        self.actions = tuple(feature_builder.actions) if feature_builder is not None else tuple(actions)
        self.parameter_spec = parameter_spec
        self.feature_builder = feature_builder or Tune7FeatureBuilder(
            self.actions,
            self.parameter_spec,
            tuple(int(v) for v in context_interaction_indices),
        )
        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.beta0 = float(beta0)
        self.sigma_x = max(float(sigma_x), 1e-6)
        self.sigma_z = max(float(sigma_z), 1e-6)
        self.rng = np.random.default_rng(seed)
        if int(candidate_local_size) < 0 or int(candidate_elite_size) < 0:
            raise ValueError("candidate_local_size and candidate_elite_size must be >= 0")
        if int(candidate_local_size) + int(candidate_elite_size) > int(candidate_pool_size):
            raise ValueError("local + elite candidate counts must be <= candidate_pool_size")
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
            local_size=int(candidate_local_size),
            elite_size=int(candidate_elite_size),
            random_size=int(candidate_pool_size) - int(candidate_local_size) - int(candidate_elite_size),
            random_mode="regime_stratified",
            local_seed_count=int(local_seed_count),
        )

        self.landmark_x = np.asarray(landmark_x, dtype=float)
        self.landmark_z = np.asarray(landmark_z, dtype=float)
        self.landmark_regime = np.asarray(landmark_regime, dtype=int).reshape(-1)
        if self.landmark_x.ndim != 2 or self.landmark_x.shape[0] == 0:
            raise ValueError("landmark_x must have shape (m, d_x)")
        if self.landmark_x.shape[1] != self.d_x:
            raise ValueError("landmark_x has incompatible context dimension")
        if self.landmark_z.ndim != 2 or self.landmark_z.shape[0] != self.landmark_x.shape[0]:
            raise ValueError("landmark_z must have shape (m, z_dim)")
        if self.landmark_z.shape[1] != self.feature_builder.z_dim:
            raise ValueError("landmark_z has incompatible z dimension")
        if self.landmark_regime.size != self.landmark_x.shape[0]:
            raise ValueError("landmark_regime must have length m")

        W = self._kernel_matrix(
            x_left=self.landmark_x,
            z_left=self.landmark_z,
            regime_left=self.landmark_regime,
            x_right=self.landmark_x,
            z_right=self.landmark_z,
            regime_right=self.landmark_regime,
        )
        eigvals, eigvecs = np.linalg.eigh(0.5 * (W + W.T))
        eigvals = np.maximum(eigvals, 1e-8)
        self.W_inv_sqrt = (eigvecs * (1.0 / np.sqrt(eigvals))) @ eigvecs.T

        self.regressor = OnlineRidgeRegressor(dim=int(self.landmark_x.shape[0]), l2_reg=float(l2_reg))
        self.t = 0
        self._last_feature: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: list[NystromKernelUCBStep] = []
        self.candidate_stats_history: list[Dict[str, int | str]] = []

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _kernel_matrix(
        self,
        *,
        x_left: np.ndarray,
        z_left: np.ndarray,
        regime_left: np.ndarray,
        x_right: np.ndarray,
        z_right: np.ndarray,
        regime_right: np.ndarray,
    ) -> np.ndarray:
        x_diff = x_left[:, None, :] - x_right[None, :, :]
        z_diff = z_left[:, None, :] - z_right[None, :, :]
        k_x = np.exp(-np.sum(x_diff * x_diff, axis=2) / (2.0 * self.sigma_x * self.sigma_x))
        k_z = np.exp(-np.sum(z_diff * z_diff, axis=2) / (2.0 * self.sigma_z * self.sigma_z))
        k_g = (regime_left[:, None] == regime_right[None, :]).astype(float)
        return k_x * k_z * k_g

    def _feature_matrix(self, x: np.ndarray, arms: np.ndarray) -> np.ndarray:
        arm_idx = np.asarray(arms, dtype=int).reshape(-1)
        x_left = np.repeat(x.reshape(1, -1), arm_idx.size, axis=0)
        z_left = self.feature_builder.z_matrix(arm_idx)
        regime_left = self.feature_builder.regime_indices[arm_idx]
        K = self._kernel_matrix(
            x_left=x_left,
            z_left=z_left,
            regime_left=regime_left,
            x_right=self.landmark_x,
            z_right=self.landmark_z,
            regime_right=self.landmark_regime,
        )
        return K @ self.W_inv_sqrt

    def _record_prediction(self, *, arm: int, pred_mean: float, pred_uncert: float) -> Dict[str, Any]:
        self.history.append(
            NystromKernelUCBStep(
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

        Phi = self._feature_matrix(x, cand)
        pred = np.asarray(self.regressor.predict(Phi), dtype=float).reshape(-1)
        unc = np.asarray(self.regressor.uncertainty(Phi), dtype=float).reshape(-1)
        score = pred - float(self.beta0) * unc
        loc = int(np.argmin(score))
        arm = int(cand[loc])
        self._last_feature = np.asarray(Phi[loc], dtype=float)
        self._last_arm = int(arm)
        params = self._record_prediction(arm=arm, pred_mean=float(pred[loc]), pred_uncert=float(unc[loc]))
        return params, {"pred_mean": float(pred[loc]), "pred_uncert": float(unc[loc])}

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        candidate_arms, stats = self._selector.candidate_subset(return_stats=True)
        self.candidate_stats_history.append(dict(stats))
        params, _info = self.predict_from_candidate_arms(context, candidate_arms)
        return params

    def update(self, loss: float, *, bounded_loss: Optional[float] = None) -> None:
        if self._last_feature is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")
        raw_loss = float(loss)
        if not np.isfinite(raw_loss):
            raise ValueError("loss must be finite")
        target = float(raw_loss if bounded_loss is None else bounded_loss)
        if not np.isfinite(target):
            raise ValueError("bounded_loss must be finite")
        self.regressor.observe(self._last_feature, target)
        self._selector.observe(self._last_arm, raw_loss)
        last = self.history[-1]
        self.history[-1] = NystromKernelUCBStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=raw_loss,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )
        self.t += 1
        self._last_feature = None
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
        arm = self.feature_builder.arm_for_params(params)
        feature = np.asarray(self._feature_matrix(x, np.asarray([arm], dtype=int))[0], dtype=float)
        self._last_feature = feature
        self._last_arm = int(arm)
        self.history.append(
            NystromKernelUCBStep(
                t=self.t + 1,
                arm_index=int(arm),
                loss=float("nan"),
                pred_mean=float("nan"),
                pred_uncert=float("nan"),
            )
        )
        self.update(float(loss), bounded_loss=bounded_loss)
