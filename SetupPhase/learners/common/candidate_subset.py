"""
Shared candidate-subset + elite-cache utilities for large action sets.

This logic is duplicated in older learners (e.g. SharedLinUCB_AMG_v2). New
learners should use this helper to keep behavior consistent across models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def resolve_tune7_candidate_strategy(
    *,
    tune7_variant: str,
    configured_strategy: str = "",
) -> str:
    """Return the setup-phase default candidate strategy for Tune7."""
    strategy = str(configured_strategy).strip().lower()
    if strategy:
        if strategy not in {"uniform", "adaptive_local"}:
            raise ValueError(
                "candidate strategy must be 'uniform' or 'adaptive_local'"
            )
        return strategy

    variant = str(tune7_variant).strip().lower()
    if variant == "categorical":
        return "uniform"
    if variant == "agg_conditional":
        return "adaptive_local"
    raise ValueError(f"Unsupported Tune7 variant: {tune7_variant!r}")


class CandidateSelector:
    def __init__(
        self,
        K: int,
        *,
        candidate_pool_size: Optional[int],
        always_include_arms: Optional[Sequence[int]],
        elite_cache_size: int,
        elite_rank_metric: str = "best_loss",
        sparse_statistics: bool = False,
        rng: np.random.Generator,
    ) -> None:
        if K <= 0:
            raise ValueError("K must be positive")
        if elite_cache_size < 0:
            raise ValueError("elite_cache_size must be >= 0")
        if elite_rank_metric not in {"best_loss", "mean_loss"}:
            raise ValueError("elite_rank_metric must be 'best_loss' or 'mean_loss'")
        self.K = int(K)
        self.rng = rng
        self.candidate_pool_size = int(candidate_pool_size) if candidate_pool_size is not None else None
        self.elite_rank_metric = str(elite_rank_metric)
        self.sparse_statistics = bool(sparse_statistics)

        self._always_include_arms = self._validate_always_include_arms(always_include_arms)
        self.elite_cache_size = int(elite_cache_size)
        self._elite_arms = np.zeros(0, dtype=int)
        use_dense = self.elite_cache_size > 0 and not self.sparse_statistics
        self._arm_best_loss = np.full(self.K, np.inf, dtype=float) if use_dense else None
        self._arm_loss_sum = np.zeros(self.K, dtype=float) if use_dense else None
        self._arm_obs_count = np.zeros(self.K, dtype=int) if use_dense else None
        self._sparse_arm_stats: Dict[int, List[float | int]] | None = (
            {} if self.elite_cache_size > 0 and self.sparse_statistics else None
        )

    def clone_for_independent_updates(
        self,
        *,
        rng: np.random.Generator,
    ) -> "CandidateSelector":
        clone = CandidateSelector(
            self.K,
            candidate_pool_size=self.candidate_pool_size,
            always_include_arms=self._always_include_arms,
            elite_cache_size=self.elite_cache_size,
            elite_rank_metric=self.elite_rank_metric,
            sparse_statistics=self.sparse_statistics,
            rng=rng,
        )
        clone._elite_arms = self._elite_arms.copy()
        if self._arm_best_loss is not None:
            clone._arm_best_loss = self._arm_best_loss.copy()
            clone._arm_loss_sum = self._arm_loss_sum.copy()
            clone._arm_obs_count = self._arm_obs_count.copy()
        if self._sparse_arm_stats is not None:
            clone._sparse_arm_stats = {
                int(arm): list(values)
                for arm, values in self._sparse_arm_stats.items()
            }
        return clone

    def _validate_always_include_arms(self, always_include_arms: Optional[Sequence[int]]) -> np.ndarray:
        if always_include_arms is None:
            return np.zeros(0, dtype=int)
        out: List[int] = []
        seen = set()
        for a in always_include_arms:
            ai = int(a)
            if not (0 <= ai < self.K):
                raise ValueError("always_include_arms contains out-of-range index")
            if ai not in seen:
                out.append(ai)
                seen.add(ai)
        return np.asarray(out, dtype=int)

    def _merged_include_arms(self) -> np.ndarray:
        """
        Merge always-include and elite arms, preserving priority:
        always-include first, then elite.
        """
        if self._always_include_arms.size == 0 and self._elite_arms.size == 0:
            return np.zeros(0, dtype=int)
        out: List[int] = []
        seen = set()
        for a in self._always_include_arms:
            ai = int(a)
            if ai not in seen:
                out.append(ai)
                seen.add(ai)
        for a in self._elite_arms:
            ai = int(a)
            if ai not in seen:
                out.append(ai)
                seen.add(ai)
        return np.asarray(out, dtype=int)

    @property
    def always_include_arms(self) -> np.ndarray:
        return np.asarray(self._always_include_arms, dtype=int)

    @property
    def elite_arms(self) -> np.ndarray:
        return np.asarray(self._elite_arms, dtype=int)

    @property
    def arm_mean_loss(self) -> np.ndarray:
        if self._sparse_arm_stats is not None:
            raise RuntimeError(
                "arm_mean_loss would materialize the full sparse action space"
            )
        if self._arm_loss_sum is None or self._arm_obs_count is None:
            return np.full(self.K, np.inf, dtype=float)
        out = np.full(self.K, np.inf, dtype=float)
        mask = self._arm_obs_count > 0
        out[mask] = self._arm_loss_sum[mask] / self._arm_obs_count[mask]
        return out

    def candidate_subset(self, *, pool_size: Optional[int] = None) -> np.ndarray:
        """
        Return a candidate subset of arms to score when candidate_pool_size is set.
        Always includes the configured always-include arms and the current elite cache.
        """
        M = int(pool_size) if pool_size is not None else (
            int(self.candidate_pool_size) if self.candidate_pool_size is not None else self.K
        )
        if M >= self.K:
            return np.arange(self.K, dtype=int)

        include = self._merged_include_arms()
        if include.size > M:
            include = include[:M]

        cand = self.rng.choice(self.K, size=M, replace=False)
        if include.size:
            cand = np.unique(np.concatenate([cand, include]))
            if cand.size > M:
                remaining = np.setdiff1d(cand, include, assume_unique=False)
                need = M - int(include.size)
                if need <= 0:
                    cand = include
                else:
                    if remaining.size > need:
                        remaining = self.rng.choice(remaining, size=need, replace=False)
                    cand = np.concatenate([include, remaining])
        return np.asarray(cand, dtype=int)

    def observe(self, arm: int, loss: float) -> None:
        """
        Update elite cache statistics using the realized loss from `arm`.
        """
        if self._arm_best_loss is None and self._sparse_arm_stats is None:
            return
        a = int(arm)
        y = float(loss)
        if not (0 <= a < self.K):
            raise ValueError("arm out of range")
        if not np.isfinite(y):
            return

        if self._sparse_arm_stats is not None:
            stats = self._sparse_arm_stats.setdefault(a, [np.inf, 0.0, 0])
            stats[0] = min(float(stats[0]), y)
            stats[1] = float(stats[1]) + y
            stats[2] = int(stats[2]) + 1
            finite = np.fromiter(self._sparse_arm_stats, dtype=int)
            if self.elite_rank_metric == "mean_loss":
                vals = np.asarray(
                    [
                        float(self._sparse_arm_stats[int(arm)][1])
                        / int(self._sparse_arm_stats[int(arm)][2])
                        for arm in finite
                    ],
                    dtype=float,
                )
            else:
                vals = np.asarray(
                    [
                        float(self._sparse_arm_stats[int(arm)][0])
                        for arm in finite
                    ],
                    dtype=float,
                )
        else:
            self._arm_best_loss[a] = min(float(self._arm_best_loss[a]), y)
            self._arm_loss_sum[a] += y
            self._arm_obs_count[a] += 1
            if self.elite_rank_metric == "mean_loss":
                finite = np.flatnonzero(self._arm_obs_count > 0)
                vals = (
                    self._arm_loss_sum[finite] / self._arm_obs_count[finite]
                )
            else:
                finite = np.flatnonzero(np.isfinite(self._arm_best_loss))
                vals = self._arm_best_loss[finite]
        if finite.size == 0:
            return

        k = min(int(self.elite_cache_size), int(finite.size))
        if k <= 0:
            return

        top_loc = np.argpartition(vals, kth=k - 1)[:k]
        elite = finite[top_loc]
        elite_vals = vals[top_loc]
        elite = elite[np.argsort(elite_vals)]
        self._elite_arms = elite.astype(int)

    def export_statistics(self) -> Dict[str, Any]:
        """Serialize observed-arm statistics without forcing dense K-sized arrays."""

        if self._sparse_arm_stats is not None:
            arms = np.asarray(sorted(self._sparse_arm_stats), dtype=np.int64)
            best = np.asarray(
                [self._sparse_arm_stats[int(arm)][0] for arm in arms],
                dtype=float,
            )
            loss_sum = np.asarray(
                [self._sparse_arm_stats[int(arm)][1] for arm in arms],
                dtype=float,
            )
            count = np.asarray(
                [self._sparse_arm_stats[int(arm)][2] for arm in arms],
                dtype=np.int64,
            )
            return {
                "sparse": True,
                "arms": arms,
                "best_loss": best,
                "loss_sum": loss_sum,
                "obs_count": count,
            }
        return {
            "sparse": False,
            "arms": np.empty(0, dtype=np.int64),
            "best_loss": (
                self._arm_best_loss.copy()
                if self._arm_best_loss is not None
                else np.empty(0, dtype=float)
            ),
            "loss_sum": (
                self._arm_loss_sum.copy()
                if self._arm_loss_sum is not None
                else np.empty(0, dtype=float)
            ),
            "obs_count": (
                self._arm_obs_count.copy()
                if self._arm_obs_count is not None
                else np.empty(0, dtype=np.int64)
            ),
        }

    def restore_statistics(self, state: Mapping[str, Any]) -> None:
        sparse = bool(state["sparse"])
        if sparse != bool(self._sparse_arm_stats is not None):
            raise ValueError("candidate-statistics storage mode does not match")
        best = np.asarray(state["best_loss"], dtype=float)
        loss_sum = np.asarray(state["loss_sum"], dtype=float)
        count = np.asarray(state["obs_count"], dtype=np.int64)
        if sparse:
            arms = np.asarray(state["arms"], dtype=np.int64)
            if not (arms.shape == best.shape == loss_sum.shape == count.shape):
                raise ValueError("sparse candidate-statistics shapes do not match")
            if np.any(arms < 0) or np.any(arms >= self.K):
                raise ValueError("sparse candidate statistics contain invalid arms")
            self._sparse_arm_stats = {
                int(arm): [float(b), float(total), int(n)]
                for arm, b, total, n in zip(arms, best, loss_sum, count)
            }
            return
        for current, saved, name in (
            (self._arm_best_loss, best, "best_loss"),
            (self._arm_loss_sum, loss_sum, "loss_sum"),
            (self._arm_obs_count, count, "obs_count"),
        ):
            if current is None:
                if saved.size:
                    raise ValueError(f"unexpected dense candidate statistic {name}")
            elif saved.shape != current.shape:
                raise ValueError(f"dense candidate statistic {name} shape differs")
            else:
                current[:] = saved
