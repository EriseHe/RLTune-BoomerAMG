from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._amg_action_features import ParameterSpaceSpec
from ._candidate_subset import Tune7CandidateMixer, Tune7LatticeCatalog
from ._online_regression import OnlineRidgeRegressor, sample_discrete, squarecb_gamma, squarecb_probabilities
from ._tune7_online_features import Tune7FeatureBuilder


@dataclass(frozen=True)
class SquareCBv4Step:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class SquareCB_AMG_v4:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        parameter_spec: ParameterSpaceSpec,
        oracle_kind: str,
        l2_reg: float = 1.0,
        gamma_scale: float = 10.0,
        gamma_exponent: float = 0.5,
        context_interaction_indices: Sequence[int] = (1, 2, 3, 4),
        candidate_pool_size: int = 512,
        always_include_arms: Optional[Sequence[int]] = None,
        elite_cache_size: int = 128,
        candidate_local_size: int = 0,
        candidate_elite_size: int = 128,
        candidate_random_mode: str = "uniform",
        local_seed_count: int = 4,
        feature_builder: Optional[Tune7FeatureBuilder] = None,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")
        oracle = str(oracle_kind).strip().lower()
        if oracle not in {"linear", "quadratic"}:
            raise ValueError("oracle_kind must be 'linear' or 'quadratic'")

        self.actions = tuple(feature_builder.actions) if feature_builder is not None else tuple(actions)
        self.parameter_spec = parameter_spec
        self.feature_builder = feature_builder or Tune7FeatureBuilder(
            self.actions,
            self.parameter_spec,
            tuple(int(v) for v in context_interaction_indices),
        )
        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.oracle_kind = oracle
        self.gamma_scale = float(gamma_scale)
        self.gamma_exponent = float(gamma_exponent)
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
            random_mode=str(candidate_random_mode),
            local_seed_count=int(local_seed_count),
        )

        probe = self._feature_matrix(np.zeros(self.d_x, dtype=float), np.asarray([0], dtype=int))
        self.regressor = OnlineRidgeRegressor(dim=int(probe.shape[1]), l2_reg=float(l2_reg))

        self.t = 0
        self._last_feature: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: list[SquareCBv4Step] = []
        self.candidate_stats_history: list[Dict[str, int | str]] = []

    def _feature_matrix(self, x: np.ndarray, arms: np.ndarray) -> np.ndarray:
        if self.oracle_kind == "linear":
            return self.feature_builder.phi_v4_matrix(x, arms)
        return self.feature_builder.quadratic_oracle_matrix(x, arms)

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _record_prediction(self, *, arm: int, pred_mean: float, pred_uncert: float) -> Dict[str, Any]:
        self.history.append(
            SquareCBv4Step(
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
        gamma = squarecb_gamma(
            round_index=self.t,
            gamma_scale=self.gamma_scale,
            gamma_exponent=self.gamma_exponent,
        )
        probs = squarecb_probabilities(pred, gamma=gamma)
        loc = sample_discrete(probs, rng=self.rng)
        arm = int(cand[loc])
        feature = np.asarray(Phi[loc], dtype=float)
        pred_mean = float(pred[loc])
        pred_uncert = float(self.regressor.uncertainty(feature)[0])

        self._last_feature = feature
        self._last_arm = int(arm)
        params = self._record_prediction(arm=arm, pred_mean=pred_mean, pred_uncert=pred_uncert)
        return params, {"pred_mean": pred_mean, "pred_uncert": pred_uncert}

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
        self.history[-1] = SquareCBv4Step(
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
            SquareCBv4Step(
                t=self.t + 1,
                arm_index=int(arm),
                loss=float("nan"),
                pred_mean=float("nan"),
                pred_uncert=float("nan"),
            )
        )
        self.update(float(loss), bounded_loss=bounded_loss)
