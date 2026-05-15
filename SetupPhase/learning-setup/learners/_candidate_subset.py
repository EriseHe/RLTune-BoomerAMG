"""
Shared candidate-subset + elite-cache utilities for large action sets.

This logic is duplicated in older learners (e.g. SharedLinUCB_AMG_v2). New
learners should use this helper to keep behavior consistent across models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

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


class Tune7LatticeCatalog:
    """
    Regime-aware tune7 lattice utilities.

    Numeric-local neighborhoods are defined by offsets in the 5D lattice:
      (strong_threshold, max_row_sum, trunc_factor, P_max_elmts, agg_num_levels)

    No brute-force pairwise distance matrices are built.
    """

    def __init__(
        self,
        *,
        regime_ids: Sequence[int],
        lattice_indices: np.ndarray,
        grid_shape: Sequence[int],
        neighbor_cache_limit: int = 512,
    ) -> None:
        regime_arr = np.asarray(regime_ids, dtype=int).reshape(-1)
        lattice_arr = np.asarray(lattice_indices, dtype=int)
        shape = tuple(int(v) for v in grid_shape)
        if lattice_arr.ndim != 2 or lattice_arr.shape[0] != regime_arr.size:
            raise ValueError("lattice_indices must have shape (K, d)")
        if lattice_arr.shape[1] != len(shape):
            raise ValueError("grid_shape rank must match lattice index dimension")
        if any(v <= 0 for v in shape):
            raise ValueError("grid_shape entries must be positive")
        for dim, size in enumerate(shape):
            coords = lattice_arr[:, dim]
            if np.any(coords < 0) or np.any(coords >= int(size)):
                raise ValueError("lattice indices must be within grid_shape bounds")

        self.K = int(regime_arr.size)
        self.regime_ids = regime_arr
        self.lattice_indices = lattice_arr
        self.grid_shape = shape
        self.dims = int(len(shape))
        self.neighbor_cache_limit = int(max(1, neighbor_cache_limit))
        self.max_l1_radius = int(sum(int(v) - 1 for v in shape))

        self.regime_arms: Dict[int, np.ndarray] = {}
        for regime in np.unique(self.regime_ids):
            self.regime_arms[int(regime)] = np.flatnonzero(self.regime_ids == int(regime)).astype(int)

        self.regime_count = int(max(self.regime_arms.keys(), default=-1) + 1)
        self.lookup = np.full((self.regime_count, *self.grid_shape), -1, dtype=np.int32)
        for arm in range(self.K):
            idx = tuple(int(v) for v in self.lattice_indices[arm])
            self.lookup[(int(self.regime_ids[arm]), *idx)] = int(arm)

        self._offset_shell_cache: Dict[int, Tuple[Tuple[int, ...], ...]] = {}
        self._neighbor_cache: Dict[int, np.ndarray] = {}

    def _offset_shell(self, radius: int) -> Tuple[Tuple[int, ...], ...]:
        r = int(radius)
        if r <= 0:
            return ()
        cached = self._offset_shell_cache.get(r)
        if cached is not None:
            return cached

        out: List[Tuple[int, ...]] = []
        prefix = [0] * self.dims

        def rec(dim: int, remaining: int) -> None:
            if dim == self.dims:
                if remaining == 0 and any(v != 0 for v in prefix):
                    out.append(tuple(int(v) for v in prefix))
                return
            max_abs = min(int(self.grid_shape[dim]) - 1, int(remaining))
            for absval in range(max_abs + 1):
                if absval == 0:
                    prefix[dim] = 0
                    rec(dim + 1, remaining)
                    continue
                prefix[dim] = int(absval)
                rec(dim + 1, remaining - absval)
                prefix[dim] = -int(absval)
                rec(dim + 1, remaining - absval)
            prefix[dim] = 0

        rec(0, r)
        out.sort(key=lambda off: (sum(v * v for v in off), off))
        shell = tuple(out)
        self._offset_shell_cache[r] = shell
        return shell

    def _cached_neighbors(self, seed_arm: int) -> np.ndarray:
        seed = int(seed_arm)
        cached = self._neighbor_cache.get(seed)
        if cached is not None:
            return cached

        regime = int(self.regime_ids[seed])
        base_idx = self.lattice_indices[seed]
        seen = {seed}
        out: List[int] = []

        for radius in range(1, self.max_l1_radius + 1):
            for offset in self._offset_shell(radius):
                idx = [int(base_idx[dim]) + int(offset[dim]) for dim in range(self.dims)]
                if any(v < 0 or v >= int(self.grid_shape[dim]) for dim, v in enumerate(idx)):
                    continue
                candidate = int(self.lookup[(regime, *idx)])
                if candidate < 0 or candidate in seen:
                    continue
                out.append(candidate)
                seen.add(candidate)
                if len(out) >= self.neighbor_cache_limit:
                    neighbors = np.asarray(out, dtype=int)
                    self._neighbor_cache[seed] = neighbors
                    return neighbors

        neighbors = np.asarray(out, dtype=int)
        self._neighbor_cache[seed] = neighbors
        return neighbors

    def local_from_seeds(
        self,
        *,
        seed_arms: Sequence[int],
        need: int,
        excluded: Optional[Sequence[int]] = None,
        allowed_arms: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        if int(need) <= 0:
            return np.zeros(0, dtype=int)
        allowed_set = None if allowed_arms is None else {int(v) for v in np.asarray(allowed_arms, dtype=int)}
        seen = set() if excluded is None else {int(v) for v in np.asarray(excluded, dtype=int)}
        out: List[int] = []
        for seed in np.asarray(seed_arms, dtype=int).reshape(-1):
            for candidate in self._cached_neighbors(int(seed)):
                arm = int(candidate)
                if arm in seen:
                    continue
                if allowed_set is not None and arm not in allowed_set:
                    continue
                out.append(arm)
                seen.add(arm)
                if len(out) >= int(need):
                    return np.asarray(out, dtype=int)
        return np.asarray(out, dtype=int)

    def sample_uniform(
        self,
        *,
        rng: np.random.Generator,
        need: int,
        excluded: Optional[Sequence[int]] = None,
        allowed_arms: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        if int(need) <= 0:
            return np.zeros(0, dtype=int)
        excluded_set = set() if excluded is None else {int(v) for v in np.asarray(excluded, dtype=int)}
        out: List[int] = []
        seen = set(excluded_set)
        if allowed_arms is None:
            while len(out) < int(need) and len(seen) < self.K:
                draw = int(rng.integers(0, self.K))
                if draw in seen:
                    continue
                out.append(draw)
                seen.add(draw)
            return np.asarray(out, dtype=int)

        allowed_arr = np.asarray(allowed_arms, dtype=int).reshape(-1)
        if allowed_arr.size == 0:
            return np.zeros(0, dtype=int)
        if int(need) >= int(allowed_arr.size) - len(excluded_set):
            return np.asarray([int(v) for v in allowed_arr if int(v) not in excluded_set], dtype=int)
        while len(out) < int(need) and len(seen) < int(allowed_arr.size) + len(excluded_set):
            draw_idx = int(rng.integers(0, allowed_arr.size))
            draw = int(allowed_arr[draw_idx])
            if draw in seen:
                continue
            out.append(draw)
            seen.add(draw)
        return np.asarray(out, dtype=int)

    def sample_regime_stratified(
        self,
        *,
        rng: np.random.Generator,
        need: int,
        excluded: Optional[Sequence[int]] = None,
        allowed_arms: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        if int(need) <= 0:
            return np.zeros(0, dtype=int)
        excluded_set = set() if excluded is None else {int(v) for v in np.asarray(excluded, dtype=int)}
        available_by_regime: Dict[int, np.ndarray] = {}
        if allowed_arms is None:
            for regime, pool in self.regime_arms.items():
                available_by_regime[int(regime)] = np.asarray(pool, dtype=int)
        else:
            allowed_arr = np.asarray(allowed_arms, dtype=int).reshape(-1)
            if allowed_arr.size == 0:
                return np.zeros(0, dtype=int)
            for regime in sorted(self.regime_arms):
                pool = allowed_arr[self.regime_ids[allowed_arr] == int(regime)]
                if pool.size:
                    available_by_regime[int(regime)] = np.asarray(pool, dtype=int)
        if not available_by_regime:
            return np.zeros(0, dtype=int)

        active_regimes = sorted(available_by_regime)
        base = int(need) // len(active_regimes)
        rem = int(need) % len(active_regimes)
        regime_order = list(rng.permutation(active_regimes))

        out: List[int] = []
        seen = set(excluded_set)
        for idx, regime in enumerate(regime_order):
            pool = available_by_regime[int(regime)]
            quota = base + (1 if idx < rem else 0)
            if quota <= 0:
                continue
            pool_seen = set(seen)
            picks: List[int] = []
            excluded_in_regime = sum(
                1
                for arm in pool_seen
                if 0 <= int(arm) < self.K and int(self.regime_ids[int(arm)]) == int(regime)
            )
            available_count = int(pool.size) - int(excluded_in_regime)
            if available_count <= 0:
                continue
            if quota >= available_count:
                picks = [int(arm) for arm in pool if int(arm) not in pool_seen]
            else:
                while len(picks) < int(quota) and len(pool_seen) < int(pool.size) + len(seen):
                    draw = int(pool[int(rng.integers(0, pool.size))])
                    if draw in pool_seen:
                        continue
                    picks.append(draw)
                    pool_seen.add(draw)
            for arm in picks:
                if arm in seen:
                    continue
                out.append(int(arm))
                seen.add(int(arm))
                if len(out) >= int(need):
                    return np.asarray(out, dtype=int)

        rem_need = int(need) - len(out)
        if rem_need > 0:
            filler = self.sample_uniform(
                rng=rng,
                need=rem_need,
                excluded=np.asarray(list(seen), dtype=int),
                allowed_arms=allowed_arms,
            )
            out.extend(int(v) for v in np.asarray(filler, dtype=int))
        return np.asarray(out[: int(need)], dtype=int)


class Tune7CandidateMixer:
    """
    Learner-local candidate mixer for tune7 methods.

    Candidate pools are assembled from:
      local + elite + random
    using a full-K elite cache but algorithm-specific sampling policy.
    """

    def __init__(
        self,
        K: int,
        *,
        catalog: Tune7LatticeCatalog,
        candidate_pool_size: int,
        always_include_arms: Optional[Sequence[int]],
        elite_cache_size: int,
        rng: np.random.Generator,
        local_size: int = 0,
        elite_size: int = 0,
        random_size: int = 0,
        random_mode: str = "uniform",
        allowed_arms: Optional[Sequence[int]] = None,
        local_seed_count: int = 4,
        include_incumbent: bool = True,
        elite_rank_metric: str = "best_loss",
    ) -> None:
        if int(candidate_pool_size) <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if int(local_size) < 0 or int(elite_size) < 0 or int(random_size) < 0:
            raise ValueError("local_size/elite_size/random_size must be >= 0")
        if int(local_size) + int(elite_size) + int(random_size) != int(candidate_pool_size):
            raise ValueError("local_size + elite_size + random_size must equal candidate_pool_size")
        if random_mode not in {"uniform", "regime_stratified"}:
            raise ValueError("random_mode must be 'uniform' or 'regime_stratified'")

        self.K = int(K)
        self.catalog = catalog
        self.candidate_pool_size = int(candidate_pool_size)
        self.local_size = int(local_size)
        self.elite_size = int(elite_size)
        self.random_size = int(random_size)
        self.random_mode = str(random_mode)
        self.local_seed_count = int(max(1, local_seed_count))
        self.include_incumbent = bool(include_incumbent)
        self.rng = rng
        self.allowed_arms = (
            np.arange(self.K, dtype=int)
            if allowed_arms is None
            else np.asarray(sorted({int(v) for v in np.asarray(allowed_arms, dtype=int)}), dtype=int)
        )
        if self.allowed_arms.size == 0:
            raise ValueError("allowed_arms must be non-empty")
        self._allowed_set = {int(v) for v in self.allowed_arms}
        allowed_regimes = np.unique(self.catalog.regime_ids[self.allowed_arms])
        self._single_regime = int(allowed_regimes[0]) if allowed_regimes.size == 1 else None
        include = None
        if always_include_arms is not None:
            include = [int(v) for v in always_include_arms if int(v) in self._allowed_set]
        self._cand = CandidateSelector(
            self.K,
            candidate_pool_size=int(candidate_pool_size),
            always_include_arms=include,
            elite_cache_size=int(elite_cache_size),
            elite_rank_metric=str(elite_rank_metric),
            rng=self.rng,
        )

    @property
    def always_include_arms(self) -> np.ndarray:
        return self._cand.always_include_arms

    @property
    def elite_arms(self) -> np.ndarray:
        elite = np.asarray(self._cand.elite_arms, dtype=int)
        if elite.size == 0:
            return elite
        mask = np.asarray([int(v) in self._allowed_set for v in elite], dtype=bool)
        return elite[mask]

    def observe(self, arm: int, loss: float) -> None:
        if int(arm) in self._allowed_set:
            self._cand.observe(int(arm), float(loss))

    def incumbent_arm(self) -> int:
        elite = self.elite_arms
        if elite.size:
            return int(elite[0])
        always_include = np.asarray(self.always_include_arms, dtype=int)
        if always_include.size:
            return int(always_include[0])
        return int(self.allowed_arms[0])

    def _append_unique(self, out: List[int], seen: set[int], values: Sequence[int]) -> None:
        for raw in np.asarray(values, dtype=int).reshape(-1):
            arm = int(raw)
            if arm not in self._allowed_set or arm in seen:
                continue
            out.append(arm)
            seen.add(arm)

    def candidate_subset(
        self,
        *,
        incumbent_arm: Optional[int] = None,
        local_seed_arms: Optional[Sequence[int]] = None,
        return_stats: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, int | str]]:
        if self.allowed_arms.size <= int(self.candidate_pool_size):
            candidate = np.asarray(self.allowed_arms, dtype=int)
            stats = {
                "strategy": self.random_mode,
                "candidate_count": int(candidate.size),
                "local": 0,
                "elite": min(int(candidate.size), int(self.elite_arms.size)),
                "random": int(max(0, candidate.size - min(int(candidate.size), int(self.elite_arms.size)))),
            }
            return (candidate, stats) if return_stats else candidate

        selected: List[int] = []
        seen: set[int] = set()
        self._append_unique(selected, seen, self.always_include_arms)
        if self.include_incumbent:
            base_incumbent = int(self.incumbent_arm() if incumbent_arm is None else incumbent_arm)
            self._append_unique(selected, seen, [base_incumbent])

        local_actual = np.zeros(0, dtype=int)
        elite_actual = np.zeros(0, dtype=int)
        random_actual = np.zeros(0, dtype=int)

        if self.local_size > 0:
            elite = self.elite_arms
            if local_seed_arms is None:
                seed_arms = np.asarray(
                    selected + [int(v) for v in elite[: max(0, min(self.local_seed_count, elite.size))]],
                    dtype=int,
                )
            else:
                seed_arms = np.asarray(local_seed_arms, dtype=int).reshape(-1)
            local_actual = self.catalog.local_from_seeds(
                seed_arms=seed_arms,
                need=int(self.local_size),
                excluded=np.asarray(selected, dtype=int),
                allowed_arms=None if self._single_regime is not None else self.allowed_arms,
            )
            self._append_unique(selected, seen, local_actual)

        if self.elite_size > 0:
            elite_pool = np.setdiff1d(self.elite_arms, np.asarray(selected, dtype=int), assume_unique=False)
            elite_actual = np.asarray(elite_pool[: int(self.elite_size)], dtype=int)
            self._append_unique(selected, seen, elite_actual)

        if self.random_size > 0:
            if self.random_mode == "regime_stratified":
                random_actual = self.catalog.sample_regime_stratified(
                    rng=self.rng,
                    need=int(self.random_size),
                    excluded=np.asarray(selected, dtype=int),
                    allowed_arms=self.allowed_arms,
                )
            else:
                random_actual = self.catalog.sample_uniform(
                    rng=self.rng,
                    need=int(self.random_size),
                    excluded=np.asarray(selected, dtype=int),
                    allowed_arms=self.allowed_arms,
                )
            self._append_unique(selected, seen, random_actual)

        candidate = np.asarray(selected, dtype=int)
        if candidate.size < int(self.candidate_pool_size):
            filler = self.catalog.sample_uniform(
                rng=self.rng,
                need=int(self.candidate_pool_size) - int(candidate.size),
                excluded=candidate,
                allowed_arms=self.allowed_arms,
            )
            self._append_unique(selected, seen, filler)
            candidate = np.asarray(selected, dtype=int)

        if candidate.size > int(self.candidate_pool_size):
            candidate = candidate[: int(self.candidate_pool_size)]

        stats = {
            "strategy": self.random_mode,
            "candidate_count": int(candidate.size),
            "local": int(local_actual.size),
            "elite": int(elite_actual.size),
            "random": int(random_actual.size),
        }
        return (candidate, stats) if return_stats else candidate


class StratifiedRegimeCandidateSelector(CandidateSelector):
    """
    Shared tune7 candidate selector with explicit categorical-regime diversity.

    Candidate pools are built as:
      local_size + elite_size + global_size

    where the global slice is stratified across the provided regime ids and the
    local slice only uses numeric-space neighborhoods inside each regime.
    """

    def __init__(
        self,
        K: int,
        *,
        candidate_pool_size: int,
        always_include_arms: Optional[Sequence[int]],
        elite_cache_size: int,
        rng: np.random.Generator,
        regime_ids: Sequence[int],
        numeric_coords: np.ndarray,
        local_size: int = 192,
        elite_size: int = 128,
        global_size: int = 192,
        elite_rank_metric: str = "best_loss",
    ) -> None:
        super().__init__(
            K,
            candidate_pool_size=int(candidate_pool_size),
            always_include_arms=always_include_arms,
            elite_cache_size=elite_cache_size,
            elite_rank_metric=elite_rank_metric,
            rng=rng,
        )
        if int(candidate_pool_size) <= 0:
            raise ValueError("candidate_pool_size must be positive")
        if int(local_size) < 0 or int(elite_size) < 0 or int(global_size) < 0:
            raise ValueError("local_size/elite_size/global_size must be >= 0")
        if int(local_size) + int(elite_size) + int(global_size) != int(candidate_pool_size):
            raise ValueError("local_size + elite_size + global_size must equal candidate_pool_size")

        regime_arr = np.asarray(regime_ids, dtype=int).reshape(-1)
        if regime_arr.size != int(K):
            raise ValueError("regime_ids must have length K")
        coord_arr = np.asarray(numeric_coords, dtype=float)
        if coord_arr.ndim != 2 or coord_arr.shape[0] != int(K):
            raise ValueError("numeric_coords must have shape (K, d)")
        if not np.all(np.isfinite(coord_arr)):
            raise ValueError("numeric_coords must be finite")

        self.local_size = int(local_size)
        self.elite_size = int(elite_size)
        self.global_size = int(global_size)
        self.regime_ids = regime_arr
        self.numeric_coords = coord_arr

        self._regime_to_arms: Dict[int, np.ndarray] = {}
        for regime in np.unique(self.regime_ids):
            self._regime_to_arms[int(regime)] = np.flatnonzero(self.regime_ids == int(regime)).astype(int)

    def incumbent_arm(self) -> int:
        elite = self.elite_arms
        if elite.size:
            return int(elite[0])
        always_include = self.always_include_arms
        if always_include.size:
            return int(always_include[0])
        return 0

    def _append_unique(self, out: List[int], seen: set[int], values: Sequence[int]) -> None:
        for raw in values:
            arm = int(raw)
            if arm not in seen:
                out.append(arm)
                seen.add(arm)

    def _local_candidates(
        self,
        *,
        seed_arms: np.ndarray,
        excluded: np.ndarray,
        need: int,
    ) -> np.ndarray:
        if need <= 0 or seed_arms.size == 0:
            return np.zeros(0, dtype=int)
        excluded_set = {int(v) for v in np.asarray(excluded, dtype=int)}
        scored: List[Tuple[float, int]] = []
        for arm in np.asarray(seed_arms, dtype=int):
            regime = int(self.regime_ids[int(arm)])
            pool = self._regime_to_arms.get(regime, np.zeros(0, dtype=int))
            if pool.size == 0:
                continue
            diffs = self.numeric_coords[pool] - self.numeric_coords[int(arm)]
            dists = np.linalg.norm(diffs, axis=1)
            for candidate, dist in zip(pool, dists):
                ai = int(candidate)
                if ai == int(arm) or ai in excluded_set:
                    continue
                scored.append((float(dist), ai))
        if not scored:
            return np.zeros(0, dtype=int)
        scored.sort(key=lambda item: (item[0], item[1]))
        out: List[int] = []
        seen = set(excluded_set)
        for _dist, arm in scored:
            if arm in seen:
                continue
            out.append(int(arm))
            seen.add(int(arm))
            if len(out) >= int(need):
                break
        return np.asarray(out, dtype=int)

    def _elite_candidates(self, *, excluded: np.ndarray, need: int) -> np.ndarray:
        if need <= 0:
            return np.zeros(0, dtype=int)
        elite = np.setdiff1d(self.elite_arms, np.asarray(excluded, dtype=int), assume_unique=False)
        if elite.size <= int(need):
            return np.asarray(elite, dtype=int)
        return np.asarray(elite[: int(need)], dtype=int)

    def _stratified_global_candidates(self, *, excluded: np.ndarray, need: int) -> np.ndarray:
        if need <= 0:
            return np.zeros(0, dtype=int)
        excluded_set = {int(v) for v in np.asarray(excluded, dtype=int)}
        regimes = sorted(self._regime_to_arms)
        if not regimes:
            return np.zeros(0, dtype=int)

        available_by_regime: Dict[int, np.ndarray] = {}
        for regime in regimes:
            pool = [int(v) for v in self._regime_to_arms[regime] if int(v) not in excluded_set]
            if pool:
                available_by_regime[regime] = np.asarray(pool, dtype=int)
        if not available_by_regime:
            return np.zeros(0, dtype=int)

        active_regimes = sorted(available_by_regime)
        base = int(need) // len(active_regimes)
        rem = int(need) % len(active_regimes)
        regime_order = list(self.rng.permutation(active_regimes))

        out: List[int] = []
        seen = set(excluded_set)
        for idx, regime in enumerate(regime_order):
            pool = available_by_regime[regime]
            quota = base + (1 if idx < rem else 0)
            if quota <= 0:
                continue
            if pool.size <= quota:
                picks = pool
            else:
                picks = np.asarray(self.rng.choice(pool, size=quota, replace=False), dtype=int)
            self._append_unique(out, seen, picks)

        if len(out) >= int(need):
            return np.asarray(out[: int(need)], dtype=int)

        leftovers = []
        for regime in active_regimes:
            pool = available_by_regime[regime]
            for arm in pool:
                ai = int(arm)
                if ai not in seen:
                    leftovers.append(ai)
        if leftovers:
            leftovers_arr = np.asarray(leftovers, dtype=int)
            if leftovers_arr.size > int(need) - len(out):
                leftovers_arr = np.asarray(
                    self.rng.choice(leftovers_arr, size=int(need) - len(out), replace=False),
                    dtype=int,
                )
            self._append_unique(out, seen, leftovers_arr)
        return np.asarray(out[: int(need)], dtype=int)

    def candidate_subset(
        self,
        *,
        incumbent_arm: Optional[int] = None,
        local_seed_count: int = 4,
        return_stats: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, int]]:
        if self.candidate_pool_size is None or int(self.candidate_pool_size) >= self.K:
            candidate = np.arange(self.K, dtype=int)
            stats = {"candidate_count": int(candidate.size), "local": 0, "elite": 0, "global": int(candidate.size)}
            return (candidate, stats) if return_stats else candidate

        selected: List[int] = []
        seen: set[int] = set()

        always_include = self.always_include_arms
        base_incumbent = int(self.incumbent_arm() if incumbent_arm is None else incumbent_arm)
        self._append_unique(selected, seen, always_include)
        self._append_unique(selected, seen, [base_incumbent])

        elite = self.elite_arms
        seed_arms = np.asarray(
            selected + [int(v) for v in elite[: max(0, min(int(local_seed_count), elite.size))]],
            dtype=int,
        )

        local_pick = self._local_candidates(
            seed_arms=seed_arms,
            excluded=np.asarray(selected, dtype=int),
            need=max(0, self.local_size),
        )
        self._append_unique(selected, seen, local_pick)

        elite_pick = self._elite_candidates(
            excluded=np.asarray(selected, dtype=int),
            need=max(0, self.elite_size),
        )
        self._append_unique(selected, seen, elite_pick)

        global_pick = self._stratified_global_candidates(
            excluded=np.asarray(selected, dtype=int),
            need=max(0, self.global_size),
        )
        self._append_unique(selected, seen, global_pick)

        candidate = np.asarray(selected, dtype=int)
        if candidate.size > int(self.candidate_pool_size):
            candidate = candidate[: int(self.candidate_pool_size)]

        if candidate.size < int(self.candidate_pool_size):
            filler = np.setdiff1d(np.arange(self.K, dtype=int), candidate, assume_unique=False)
            if filler.size:
                need = int(self.candidate_pool_size) - int(candidate.size)
                if filler.size > need:
                    filler = np.asarray(self.rng.choice(filler, size=need, replace=False), dtype=int)
                candidate = np.concatenate([candidate, filler], axis=0)

        stats = {
            "candidate_count": int(candidate.size),
            "local": int(local_pick.size),
            "elite": int(elite_pick.size),
            "global": int(global_pick.size),
        }
        return (np.asarray(candidate, dtype=int), stats) if return_stats else np.asarray(candidate, dtype=int)
