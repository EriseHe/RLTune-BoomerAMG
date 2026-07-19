"""
Shared (joint) LinUCB for BoomerAMG setup-phase tuning (variant).

This file is intentionally separate from `learners/SharedLinUCB_AMG.py` so we
can compare improvements without modifying the original implementation.

Feature definition (paper-ready)
--------------------------------
We tune continuous BoomerAMG knobs:

    a = (th, mxrs, tr)

where:
- th   = strong_threshold
- mxrs = max_row_sum
- tr   = trunc_factor

We center action features around a reference configuration a0:

    a_tilde = a - a0

In experiments we set a0 to the BoomerAMG defaults used in the scripts
(pass `action_center=DEFAULT_PARAMS`). If `action_center` is omitted, a0 is
set to the coordinate-wise mean of the provided finite action set.

Let a_tilde = (th~, mxrs~, tr~). Define:

    g(a_tilde) = [ th~,
                   mxrs~,
                   tr~,
                   th~^2,
                   mxrs~^2,
                   tr~^2,
                   th~*mxrs~,
                   th~*tr~,
                   mxrs~*tr~ ]^T   in R^9.

Context x is taken directly from `utils.stencil27_laplace.stencil_27_laplace`:

    x = [1, s1, s2, s3, c_diag]^T.

We do not normalize or clip s1,s2; they are used as-is.

The shared feature map is the concatenation:

    phi(x,a) = [ x ; g(a_tilde) ; s1*g(a_tilde) ; s2*g(a_tilde) ]   in R^(5+27)=R^32.

Improvements in this variant
----------------------------
1) Center action features around a reference configuration (default-ish):
   build g(a) from (a - a0) instead of a, to reduce early-round corner-seeking.
2) Optional exploration decay: alpha_t = alpha / sqrt(t+1).
3) Optional candidate subsampling: score only M randomly sampled arms per
   round (useful when the action set is huge).

Exploration schedule (paper-ready)
----------------------------------
If `alpha_decay=True`:

    alpha_t = alpha / sqrt(t+1)

where t starts at 0 internally (so the first round uses alpha_0 = alpha).
No explicit cap is applied; the formula ensures alpha_t <= alpha.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ._amg_action_features import (
    ACTION_FEATURE_DIM,
    action_center_from_actions,
    action_param_vector,
    poly2_features,
)


@dataclass(frozen=True)
class SharedLinUCBStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class SharedLinUCB_AMG_v2:
    _G_DIM = ACTION_FEATURE_DIM

    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        alpha: float = 1.0,
        alpha_decay: bool = False,
        l2_reg: float = 1.0,
        s1_index: int = 1,
        s2_index: int = 2,
        action_center: Optional[Dict[str, Any]] = None,
        candidate_pool_size: Optional[int] = None,
        always_include_arms: Optional[Sequence[int]] = None,
        elite_cache_size: int = 0,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")
        if l2_reg <= 0.0:
            raise ValueError("l2_reg must be > 0")

        self.actions: List[Dict[str, Any]] = [dict(a) for a in actions]
        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.g_dim = int(self._G_DIM)
        self.d_phi = self.d_x + 3 * self.g_dim

        self.alpha = float(alpha)
        self.alpha_decay = bool(alpha_decay)
        self.l2_reg = float(l2_reg)
        self.s1_index = int(s1_index)
        self.s2_index = int(s2_index)
        if not (0 <= self.s1_index < self.d_x and 0 <= self.s2_index < self.d_x):
            raise ValueError("s1_index/s2_index must be within [0, context_dim)")

        self.rng = np.random.default_rng(seed)
        self.candidate_pool_size = int(candidate_pool_size) if candidate_pool_size is not None else None

        self._always_include_arms = self._validate_always_include_arms(always_include_arms)
        self.elite_cache_size = int(elite_cache_size)
        if self.elite_cache_size < 0:
            raise ValueError("elite_cache_size must be >= 0")
        self._elite_arms = np.zeros(0, dtype=int)
        self._arm_best_loss = np.full(self.K, np.inf, dtype=float) if self.elite_cache_size > 0 else None

        self._a_center = self._compute_action_center(action_center)

        self.A_inv = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.b = np.zeros(self.d_phi, dtype=float)

        self._g_actions = np.zeros((self.K, self.g_dim), dtype=float)
        for i, a in enumerate(self.actions):
            self._g_actions[i] = self._g_from_action(a)

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[SharedLinUCBStep] = []

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

    def _candidate_subset(self) -> np.ndarray:
        """
        Return a candidate subset of arms to score when candidate_pool_size is set.
        Always includes the configured always-include arms and the current elite cache.
        """
        M = int(self.candidate_pool_size) if self.candidate_pool_size is not None else self.K
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

    def _compute_action_center(self, action_center: Optional[Dict[str, Any]]) -> np.ndarray:
        return action_center_from_actions(self.actions, action_center)

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _g_from_action(self, params: Dict[str, Any]) -> np.ndarray:
        a = action_param_vector(params, err_prefix="SharedLinUCB_AMG_v2")
        a = a - self._a_center
        return poly2_features(a)

    def _phi(self, x: np.ndarray, arm: int) -> np.ndarray:
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        g = self._g_actions[arm]

        phi = np.empty(self.d_phi, dtype=float)
        phi[: self.d_x] = x
        o = self.d_x
        phi[o : o + self.g_dim] = g
        o += self.g_dim
        phi[o : o + self.g_dim] = s1 * g
        o += self.g_dim
        phi[o : o + self.g_dim] = s2 * g
        return phi

    def _score_all(self, x: np.ndarray, *, alpha: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized (score, mean, uncert) for all arms.

        This mirrors the fast implementation in `SharedLinUCB_AMG` and keeps
        overhead small when scoring the full action set.
        """
        return self._score_subset(x, arms=None, alpha=alpha)

    def _score_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized (score, mean, uncert) for either:
        - all arms  (arms is None)
        - a subset  (arms is an array of arm indices)
        """
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        G = self._g_actions if arms is None else self._g_actions[np.asarray(arms, dtype=int)]  # (K', g_dim)

        theta = self.A_inv @ self.b
        theta_x = theta[: self.d_x]
        theta_g = theta[self.d_x : self.d_x + self.g_dim]
        theta_s1 = theta[self.d_x + self.g_dim : self.d_x + 2 * self.g_dim]
        theta_s2 = theta[self.d_x + 2 * self.g_dim : self.d_x + 3 * self.g_dim]

        base = float(theta_x @ x)
        mean = (
            base
            + np.einsum("ij,j->i", G, theta_g, optimize=False)
            + s1 * np.einsum("ij,j->i", G, theta_s1, optimize=False)
            + s2 * np.einsum("ij,j->i", G, theta_s2, optimize=False)
        )

        c = np.concatenate([x, np.zeros(3 * self.g_dim, dtype=float)], axis=0)
        I = np.eye(self.g_dim, dtype=float)
        P = np.vstack(
            [
                np.zeros((self.d_x, self.g_dim), dtype=float),
                I,
                s1 * I,
                s2 * I,
            ]
        )

        Ac = self.A_inv @ c
        q0 = float(c @ Ac)
        AP = self.A_inv @ P
        u = P.T @ Ac
        M = P.T @ AP

        quad = q0 + 2.0 * np.einsum("ij,j->i", G, u, optimize=False) + np.einsum("ij,jk,ik->i", G, M, G, optimize=False)
        uncert = np.sqrt(np.maximum(0.0, quad))
        score = mean - float(alpha) * uncert
        return score, mean, uncert

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        alpha_eff = float(self.alpha) / np.sqrt(self.t + 1.0) if self.alpha_decay else float(self.alpha)
        if self.candidate_pool_size is None or self.candidate_pool_size >= self.K:
            score, mean, uncert = self._score_all(x, alpha=alpha_eff)
            best_score = float(np.min(score))
            best_arms = np.flatnonzero(score <= best_score + 1e-12)
            arm = int(self.rng.choice(best_arms)) if best_arms.size > 1 else int(best_arms[0])
            best_mean = float(mean[arm])
            best_unc = float(uncert[arm])
        else:
            cand = self._candidate_subset()
            score, mean, uncert = self._score_subset(x, arms=cand, alpha=alpha_eff)
            best_score = float(np.min(score))
            best_loc = np.flatnonzero(score <= best_score + 1e-12)
            loc = int(self.rng.choice(best_loc)) if best_loc.size > 1 else int(best_loc[0])
            arm = int(cand[loc])
            best_mean = float(mean[loc])
            best_unc = float(uncert[loc])

        self._last_phi = self._phi(x, arm)
        self._last_arm = arm

        self.history.append(
            SharedLinUCBStep(
                t=self.t + 1,
                arm_index=arm,
                loss=float("nan"),
                pred_mean=best_mean,
                pred_uncert=best_unc,
            )
        )

        return dict(self.actions[arm])

    def update(self, loss: float) -> None:
        if self._last_phi is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")

        phi = self._last_phi
        arm = int(self._last_arm)
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")

        if self._arm_best_loss is not None:
            self._arm_best_loss[arm] = min(float(self._arm_best_loss[arm]), float(y))

        u = self.A_inv @ phi
        denom = 1.0 + float(phi @ u)
        if denom <= 0.0 or not np.isfinite(denom):
            A = np.linalg.inv(self.A_inv)
            A = A + np.outer(phi, phi)
            self.A_inv = np.linalg.inv(A)
        else:
            self.A_inv = self.A_inv - np.outer(u, u) / denom

        self.b = self.b + y * phi

        if self._arm_best_loss is not None:
            finite = np.flatnonzero(np.isfinite(self._arm_best_loss))
            if finite.size:
                k = min(int(self.elite_cache_size), int(finite.size))
                if k > 0:
                    vals = self._arm_best_loss[finite]
                    top_loc = np.argpartition(vals, kth=k - 1)[:k]
                    elite = finite[top_loc]
                    elite = elite[np.argsort(self._arm_best_loss[elite])]
                    self._elite_arms = elite.astype(int)

        last = self.history[-1]
        self.history[-1] = SharedLinUCBStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_phi = None
        self._last_arm = None
