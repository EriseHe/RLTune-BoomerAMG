from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._amg_action_features import ParameterSpaceSpec
from ._candidate_subset import Tune7CandidateMixer, Tune7LatticeCatalog
from ._online_regression import OnlineRidgeRegressor
from ._tune7_online_features import Tune7FeatureBuilder


@dataclass(frozen=True)
class HierarchicalLinUCBv5Step:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class HierarchicalLinUCB_AMG_v5:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        parameter_spec: ParameterSpaceSpec,
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
        candidate_pool_size: int = 512,
        always_include_arms: Optional[Sequence[int]] = None,
        elite_cache_size: int = 128,
        local_size: int = 320,
        elite_size: int = 160,
        local_seed_count: int = 4,
        feature_builder: Optional[Tune7FeatureBuilder] = None,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")
        if not np.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise ValueError("alpha must be finite and > 0")
        if not np.isfinite(float(l2_reg)) or float(l2_reg) <= 0.0:
            raise ValueError("l2_reg must be finite and > 0")
        if int(local_size) < 0 or int(elite_size) < 0:
            raise ValueError("local_size and elite_size must be >= 0")
        if int(local_size) + int(elite_size) > int(candidate_pool_size):
            raise ValueError("local + elite sizes must be <= candidate_pool_size")

        self.actions = tuple(feature_builder.actions) if feature_builder is not None else tuple(actions)
        self.parameter_spec = parameter_spec
        self.feature_builder = feature_builder or Tune7FeatureBuilder(
            self.actions,
            self.parameter_spec,
            tuple(int(v) for v in context_interaction_indices),
        )
        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        self.rng = np.random.default_rng(seed)

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

        self.candidate_pool_size = int(candidate_pool_size)
        self.local_size = int(local_size)
        self.elite_size = int(elite_size)
        self.random_size = int(candidate_pool_size) - int(local_size) - int(elite_size)
        self.elite_cache_size = int(elite_cache_size)
        self.local_seed_count = int(local_seed_count)

        probe = self.feature_builder.hierarchical_numeric_matrix(
            np.zeros(self.d_x, dtype=float),
            np.asarray([0], dtype=int),
        )
        self.regime_dim = int(self.feature_builder.regime_dim)
        self.second_dim = int(probe.shape[1])
        self.regime_regressors: Dict[int, OnlineRidgeRegressor] = {}
        self.regime_selectors: Dict[int, Tune7CandidateMixer] = {}
        self.regime_counts = np.zeros(self.regime_dim, dtype=int)

        self.t = 0
        self._last_feature: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self._last_regime: Optional[int] = None
        self.history: list[HierarchicalLinUCBv5Step] = []
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
            reg = OnlineRidgeRegressor(dim=self.second_dim, l2_reg=self.l2_reg)
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
            candidate_pool_size=int(self.candidate_pool_size),
            always_include_arms=always_include,
            elite_cache_size=int(self.elite_cache_size),
            rng=self.rng,
            local_size=int(self.local_size),
            elite_size=int(self.elite_size),
            random_size=int(self.random_size),
            random_mode="uniform",
            allowed_arms=self._catalog.regime_arms[int(regime)],
            local_seed_count=int(self.local_seed_count),
        )
        self.regime_selectors[int(regime)] = selector
        return selector

    def _record_prediction(self, *, arm: int, pred_mean: float, pred_uncert: float) -> Dict[str, Any]:
        self.history.append(
            HierarchicalLinUCBv5Step(
                t=self.t + 1,
                arm_index=int(arm),
                loss=float("nan"),
                pred_mean=float(pred_mean),
                pred_uncert=float(pred_uncert),
            )
        )
        return dict(self.actions[int(arm)])

    def _score_regime_candidates(
        self,
        *,
        x: np.ndarray,
        regime: int,
        candidate_arms: np.ndarray,
    ) -> Dict[str, Any]:
        cand = np.asarray(candidate_arms, dtype=int).reshape(-1)
        if cand.size == 0:
            raise ValueError("candidate_arms must be non-empty for a regime")
        reg = self._regressor_for_regime(int(regime))
        Phi = self.feature_builder.hierarchical_numeric_matrix(x, cand)
        mean = np.asarray(reg.predict(Phi), dtype=float).reshape(-1)
        uncert = np.asarray(reg.uncertainty(Phi), dtype=float).reshape(-1)
        score = mean - float(self.alpha) * uncert
        loc = int(np.argmin(score))
        return {
            "regime": int(regime),
            "arm": int(cand[loc]),
            "feature": np.asarray(Phi[loc], dtype=float),
            "pred_mean": float(mean[loc]),
            "pred_uncert": float(uncert[loc]),
            "lcb": float(score[loc]),
        }

    def predict_from_candidate_arms(
        self,
        context: Iterable[float],
        candidate_arms: Sequence[int],
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:
        x = self._validate_x(np.asarray(list(context), dtype=float))
        cand = np.asarray(candidate_arms, dtype=int).reshape(-1)
        if cand.size == 0:
            raise ValueError("candidate_arms must be non-empty")

        best: Optional[Dict[str, Any]] = None
        present_regimes = np.unique(self.feature_builder.regime_indices[cand])
        for regime in np.asarray(present_regimes, dtype=int):
            regime_arms = cand[self.feature_builder.regime_indices[cand] == int(regime)]
            if regime_arms.size == 0:
                continue
            scored = self._score_regime_candidates(x=x, regime=int(regime), candidate_arms=regime_arms)
            if best is None or float(scored["lcb"]) < float(best["lcb"]):
                best = scored

        if best is None:
            raise RuntimeError("no valid regime candidates were available")

        self._last_feature = np.asarray(best["feature"], dtype=float)
        self._last_arm = int(best["arm"])
        self._last_regime = int(best["regime"])
        params = self._record_prediction(
            arm=int(best["arm"]),
            pred_mean=float(best["pred_mean"]),
            pred_uncert=float(best["pred_uncert"]),
        )
        return params, {"pred_mean": float(best["pred_mean"]), "pred_uncert": float(best["pred_uncert"])}

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        best: Optional[Dict[str, Any]] = None
        best_stats: Optional[Dict[str, int | str]] = None
        regimes_scored = 0
        for regime in range(self.regime_dim):
            selector = self._selector_for_regime(regime)
            candidate_arms, stats = selector.candidate_subset(return_stats=True)
            scored = self._score_regime_candidates(x=x, regime=int(regime), candidate_arms=np.asarray(candidate_arms, dtype=int))
            regimes_scored += 1
            if best is None or float(scored["lcb"]) < float(best["lcb"]):
                best = scored
                best_stats = dict(stats)

        if best is None or best_stats is None:
            raise RuntimeError("failed to score any regimes")

        self.candidate_stats_history.append(
            {
                "strategy": "hierarchical_regime_value_lincb",
                "regimes_scored": int(regimes_scored),
                "candidate_count": int(best_stats["candidate_count"]),
                "local": int(best_stats["local"]),
                "elite": int(best_stats["elite"]),
                "random": int(best_stats["random"]),
                "regime": int(best["regime"]),
            }
        )
        self._last_feature = np.asarray(best["feature"], dtype=float)
        self._last_arm = int(best["arm"])
        self._last_regime = int(best["regime"])
        return self._record_prediction(
            arm=int(best["arm"]),
            pred_mean=float(best["pred_mean"]),
            pred_uncert=float(best["pred_uncert"]),
        )

    def update(self, loss: float, *, bounded_loss: Optional[float] = None) -> None:
        if self._last_feature is None or self._last_arm is None or self._last_regime is None:
            raise RuntimeError("update() called before predict()")
        raw_loss = float(loss)
        if not np.isfinite(raw_loss):
            raise ValueError("loss must be finite")
        target = float(raw_loss if bounded_loss is None else bounded_loss)
        if not np.isfinite(target):
            raise ValueError("bounded_loss must be finite")

        regime = int(self._last_regime)
        self._regressor_for_regime(regime).observe(self._last_feature, target)
        self._selector_for_regime(regime).observe(self._last_arm, raw_loss)
        self.regime_counts[regime] += 1

        last = self.history[-1]
        self.history[-1] = HierarchicalLinUCBv5Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=raw_loss,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )
        self.t += 1
        self._last_feature = None
        self._last_arm = None
        self._last_regime = None

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
        self._last_feature = np.asarray(
            self.feature_builder.hierarchical_numeric_matrix(x, np.asarray([arm], dtype=int))[0],
            dtype=float,
        )
        self._last_arm = int(arm)
        self._last_regime = int(regime)
        self.history.append(
            HierarchicalLinUCBv5Step(
                t=self.t + 1,
                arm_index=int(arm),
                loss=float("nan"),
                pred_mean=float("nan"),
                pred_uncert=float("nan"),
            )
        )
        self.update(float(loss), bounded_loss=bounded_loss)
