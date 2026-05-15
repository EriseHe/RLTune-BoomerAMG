from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._amg_action_features import ParameterSpaceSpec
from ._candidate_subset import Tune7CandidateMixer, Tune7LatticeCatalog
from ._online_regression import OnlineRidgeRegressor, sample_discrete, squarecb_gamma, squarecb_probabilities
from ._tune7_online_features import Tune7FeatureBuilder


@dataclass(frozen=True)
class HierarchicalSquareCBStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class HierarchicalSquareCB_AMG_v4:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        parameter_spec: ParameterSpaceSpec,
        top_l2_reg: float = 0.3,
        top_gamma_scale: float = 3.0,
        second_l2_reg: float = 1.0,
        second_gamma_scale: float = 5.0,
        gamma_exponent: float = 0.5,
        min_regime_support: int = 8,
        context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
        second_candidate_pool_size: int = 512,
        always_include_arms: Optional[Sequence[int]] = None,
        elite_cache_size: int = 128,
        second_local_size: int = 256,
        second_elite_size: int = 128,
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
        self.top_gamma_scale = float(top_gamma_scale)
        self.second_gamma_scale = float(second_gamma_scale)
        self.gamma_exponent = float(gamma_exponent)
        self.min_regime_support = int(min_regime_support)
        self.rng = np.random.default_rng(seed)
        if int(second_local_size) < 0 or int(second_elite_size) < 0:
            raise ValueError("second_local_size and second_elite_size must be >= 0")
        if int(second_local_size) + int(second_elite_size) > int(second_candidate_pool_size):
            raise ValueError("second local + elite sizes must be <= second_candidate_pool_size")
        self._catalog = Tune7LatticeCatalog(
            regime_ids=self.feature_builder.regime_indices,
            lattice_indices=self.feature_builder.lattice_indices,
            grid_shape=self.feature_builder.grid_shape,
        )
        default_arm = None
        if always_include_arms:
            for arm in always_include_arms:
                ai = int(arm)
                if 0 <= ai < self.K:
                    default_arm = ai
                    break
        self._default_arm = default_arm
        self._default_regime = (
            int(self.feature_builder.regime_indices[int(default_arm)])
            if default_arm is not None
            else None
        )
        self.second_candidate_pool_size = int(second_candidate_pool_size)
        self.second_local_size = int(second_local_size)
        self.second_elite_size = int(second_elite_size)
        self.second_random_size = int(second_candidate_pool_size) - int(second_local_size) - int(second_elite_size)
        self.second_elite_cache_size = int(elite_cache_size)
        self.local_seed_count = int(local_seed_count)

        top_probe = self.feature_builder.hierarchical_regime_matrix(
            np.zeros(self.d_x, dtype=float),
            np.asarray([0], dtype=int),
        )
        second_probe = self.feature_builder.hierarchical_numeric_matrix(
            np.zeros(self.d_x, dtype=float),
            np.asarray([0], dtype=int),
        )
        self.top_regressor = OnlineRidgeRegressor(dim=int(top_probe.shape[1]), l2_reg=float(top_l2_reg))
        self.second_dim = int(second_probe.shape[1])
        self.second_l2_reg = float(second_l2_reg)
        self.regime_regressors: Dict[int, OnlineRidgeRegressor] = {}
        self.regime_selectors: Dict[int, Tune7CandidateMixer] = {}
        self.regime_counts = np.zeros(self.feature_builder.regime_dim, dtype=int)

        self.t = 0
        self._last_arm: Optional[int] = None
        self._last_regime: Optional[int] = None
        self._last_top_feature: Optional[np.ndarray] = None
        self._last_second_feature: Optional[np.ndarray] = None
        self.history: list[HierarchicalSquareCBStep] = []
        self.candidate_stats_history: list[Dict[str, int | str]] = []

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _regressor_for_regime(self, regime: int) -> OnlineRidgeRegressor:
        reg = self.regime_regressors.get(int(regime))
        if reg is None:
            reg = OnlineRidgeRegressor(dim=self.second_dim, l2_reg=self.second_l2_reg)
            self.regime_regressors[int(regime)] = reg
        return reg

    def _selector_for_regime(self, regime: int) -> Tune7CandidateMixer:
        selector = self.regime_selectors.get(int(regime))
        if selector is not None:
            return selector
        always_include = None
        if self._default_regime is not None and int(regime) == int(self._default_regime) and self._default_arm is not None:
            always_include = [int(self._default_arm)]
        selector = Tune7CandidateMixer(
            self.K,
            catalog=self._catalog,
            candidate_pool_size=int(self.second_candidate_pool_size),
            always_include_arms=always_include,
            elite_cache_size=int(self.second_elite_cache_size),
            rng=self.rng,
            local_size=int(self.second_local_size),
            elite_size=int(self.second_elite_size),
            random_size=int(self.second_random_size),
            random_mode="uniform",
            allowed_arms=self._catalog.regime_arms[int(regime)],
            local_seed_count=int(self.local_seed_count),
        )
        self.regime_selectors[int(regime)] = selector
        return selector

    def _record_prediction(self, *, arm: int, pred_mean: float, pred_uncert: float) -> Dict[str, Any]:
        self.history.append(
            HierarchicalSquareCBStep(
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

        present_regimes = np.unique(self.feature_builder.regime_indices[cand])
        top_phi = self.feature_builder.hierarchical_regime_matrix(x, present_regimes)
        top_pred = np.asarray(self.top_regressor.predict(top_phi), dtype=float).reshape(-1)
        top_gamma = squarecb_gamma(
            round_index=self.t,
            gamma_scale=self.top_gamma_scale,
            gamma_exponent=self.gamma_exponent,
        )
        top_probs = squarecb_probabilities(top_pred, gamma=top_gamma)
        regime_loc = sample_discrete(top_probs, rng=self.rng)
        regime = int(present_regimes[regime_loc])

        regime_arms = cand[self.feature_builder.regime_indices[cand] == regime]
        if regime_arms.size == 0:
            raise RuntimeError("selected regime is not present in candidate_arms")

        pred_mean = float("nan")
        pred_uncert = float("nan")
        if int(self.regime_counts[regime]) < self.min_regime_support:
            arm = int(self.rng.choice(regime_arms))
            second_feature = np.asarray(
                self.feature_builder.hierarchical_numeric_matrix(x, np.asarray([arm], dtype=int))[0],
                dtype=float,
            )
        else:
            second_reg = self._regressor_for_regime(regime)
            second_phi = self.feature_builder.hierarchical_numeric_matrix(x, regime_arms)
            second_pred = np.asarray(second_reg.predict(second_phi), dtype=float).reshape(-1)
            second_gamma = squarecb_gamma(
                round_index=self.t,
                gamma_scale=self.second_gamma_scale,
                gamma_exponent=self.gamma_exponent,
            )
            second_probs = squarecb_probabilities(second_pred, gamma=second_gamma)
            arm_loc = sample_discrete(second_probs, rng=self.rng)
            arm = int(regime_arms[arm_loc])
            second_feature = np.asarray(second_phi[arm_loc], dtype=float)
            pred_mean = float(second_pred[arm_loc])
            pred_uncert = float(second_reg.uncertainty(second_feature)[0])

        self._last_arm = int(arm)
        self._last_regime = int(regime)
        self._last_top_feature = np.asarray(top_phi[regime_loc], dtype=float)
        self._last_second_feature = np.asarray(second_feature, dtype=float)
        params = self._record_prediction(arm=arm, pred_mean=pred_mean, pred_uncert=pred_uncert)
        return params, {"pred_mean": pred_mean, "pred_uncert": pred_uncert}

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))
        present_regimes = np.arange(self.feature_builder.regime_dim, dtype=int)
        top_phi = self.feature_builder.hierarchical_regime_matrix(x, present_regimes)
        top_pred = np.asarray(self.top_regressor.predict(top_phi), dtype=float).reshape(-1)
        top_gamma = squarecb_gamma(
            round_index=self.t,
            gamma_scale=self.top_gamma_scale,
            gamma_exponent=self.gamma_exponent,
        )
        top_probs = squarecb_probabilities(top_pred, gamma=top_gamma)
        regime_loc = sample_discrete(top_probs, rng=self.rng)
        regime = int(present_regimes[regime_loc])

        selector = self._selector_for_regime(regime)
        candidate_arms, stats = selector.candidate_subset(return_stats=True)
        self.candidate_stats_history.append(
            {
                "strategy": "hierarchical",
                "candidate_count": int(stats["candidate_count"]),
                "local": int(stats["local"]),
                "elite": int(stats["elite"]),
                "random": int(stats["random"]),
                "regime": int(regime),
            }
        )
        params, _info = self.predict_from_candidate_arms(context, candidate_arms)
        return params

    def update(self, loss: float, *, bounded_loss: Optional[float] = None) -> None:
        if (
            self._last_arm is None
            or self._last_regime is None
            or self._last_top_feature is None
            or self._last_second_feature is None
        ):
            raise RuntimeError("update() called before predict()")
        raw_loss = float(loss)
        if not np.isfinite(raw_loss):
            raise ValueError("loss must be finite")
        target = float(raw_loss if bounded_loss is None else bounded_loss)
        if not np.isfinite(target):
            raise ValueError("bounded_loss must be finite")

        regime = int(self._last_regime)
        self.top_regressor.observe(self._last_top_feature, target)
        self._regressor_for_regime(regime).observe(self._last_second_feature, target)
        self._selector_for_regime(regime).observe(self._last_arm, raw_loss)
        self.regime_counts[regime] += 1

        last = self.history[-1]
        self.history[-1] = HierarchicalSquareCBStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=raw_loss,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )
        self.t += 1
        self._last_arm = None
        self._last_regime = None
        self._last_top_feature = None
        self._last_second_feature = None

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
        regime = int(self.feature_builder.regime_indices[arm])
        self._last_arm = int(arm)
        self._last_regime = int(regime)
        self._last_top_feature = np.asarray(
            self.feature_builder.hierarchical_regime_matrix(x, np.asarray([regime], dtype=int))[0],
            dtype=float,
        )
        self._last_second_feature = np.asarray(
            self.feature_builder.hierarchical_numeric_matrix(x, np.asarray([arm], dtype=int))[0],
            dtype=float,
        )
        self.history.append(
            HierarchicalSquareCBStep(
                t=self.t + 1,
                arm_index=int(arm),
                loss=float("nan"),
                pred_mean=float("nan"),
                pred_uncert=float("nan"),
            )
        )
        self.update(float(loss), bounded_loss=bounded_loss)
