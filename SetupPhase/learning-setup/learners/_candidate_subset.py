"""
Shared candidate-subset + elite-cache utilities for large action sets.

This logic is duplicated in older learners (e.g. SharedLinUCB_AMG_v2). New
learners should use this helper to keep behavior consistent across models.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class CandidateSelector:
    def __init__(
        self,
        K: int,
        *,
        candidate_pool_size: Optional[int],
        always_include_arms: Optional[Sequence[int]],
        elite_cache_size: int,
        elite_rank_metric: str = "best_loss",
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

        self._always_include_arms = self._validate_always_include_arms(always_include_arms)
        self.elite_cache_size = int(elite_cache_size)
        self._elite_arms = np.zeros(0, dtype=int)
        self._arm_best_loss = np.full(self.K, np.inf, dtype=float) if self.elite_cache_size > 0 else None
        self._arm_loss_sum = np.zeros(self.K, dtype=float) if self.elite_cache_size > 0 else None
        self._arm_obs_count = np.zeros(self.K, dtype=int) if self.elite_cache_size > 0 else None

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
        if self._arm_best_loss is None:
            return
        a = int(arm)
        y = float(loss)
        if not (0 <= a < self.K):
            raise ValueError("arm out of range")
        if not np.isfinite(y):
            return

        self._arm_best_loss[a] = min(float(self._arm_best_loss[a]), y)
        self._arm_loss_sum[a] += y
        self._arm_obs_count[a] += 1

        if self.elite_rank_metric == "mean_loss":
            finite = np.flatnonzero(self._arm_obs_count > 0)
        else:
            finite = np.flatnonzero(np.isfinite(self._arm_best_loss))
        if finite.size == 0:
            return

        k = min(int(self.elite_cache_size), int(finite.size))
        if k <= 0:
            return

        if self.elite_rank_metric == "mean_loss":
            vals = self.arm_mean_loss[finite]
        else:
            vals = self._arm_best_loss[finite]
        top_loc = np.argpartition(vals, kth=k - 1)[:k]
        elite = finite[top_loc]
        elite_vals = vals[top_loc]
        elite = elite[np.argsort(elite_vals)]
        self._elite_arms = elite.astype(int)

