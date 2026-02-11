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

Context x is taken directly from `utils.problem_amg.stencil_27_laplace`:

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


@dataclass(frozen=True)
class SharedLinUCBStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class SharedLinUCB_AMG_v2:
    _G_DIM = 9

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

    def _compute_action_center(self, action_center: Optional[Dict[str, Any]]) -> np.ndarray:
        if action_center is not None:
            return np.array(
                [
                    float(action_center["strong_threshold"]),
                    float(action_center["max_row_sum"]),
                    float(action_center["trunc_factor"]),
                ],
                dtype=float,
            )

        th = np.array([float(a["strong_threshold"]) for a in self.actions], dtype=float)
        mxrs = np.array([float(a["max_row_sum"]) for a in self.actions], dtype=float)
        tr = np.array([float(a["trunc_factor"]) for a in self.actions], dtype=float)
        return np.array([float(np.mean(th)), float(np.mean(mxrs)), float(np.mean(tr))], dtype=float)

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d_x:
            raise ValueError(f"context has dim {x.size}, expected {self.d_x}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _g_from_action(self, params: Dict[str, Any]) -> np.ndarray:
        try:
            th = float(params["strong_threshold"])
            mxrs = float(params["max_row_sum"])
            tr = float(params["trunc_factor"])
        except KeyError as e:
            raise KeyError(f"SharedLinUCB_AMG_v2 action missing required key: {e}") from e

        if not (np.isfinite(th) and np.isfinite(mxrs) and np.isfinite(tr)):
            raise ValueError("action parameters must be finite")

        th -= float(self._a_center[0])
        mxrs -= float(self._a_center[1])
        tr -= float(self._a_center[2])

        return np.array(
            [
                th,
                mxrs,
                tr,
                th * th,
                mxrs * mxrs,
                tr * tr,
                th * mxrs,
                th * tr,
                mxrs * tr,
            ],
            dtype=float,
        )

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
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        G = self._g_actions  # (K, g_dim)

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
            theta = self.A_inv @ self.b
            cand = self.rng.choice(self.K, size=self.candidate_pool_size, replace=False)

            best_score = float("inf")
            best_mean = float("inf")
            best_unc = float("inf")
            best_arms: List[int] = []
            for a in cand:
                phi = self._phi(x, int(a))
                mean_a = float(theta @ phi)
                quad_a = float(phi @ (self.A_inv @ phi))
                unc_a = float(np.sqrt(max(0.0, quad_a)))
                score_a = mean_a - alpha_eff * unc_a
                if score_a < best_score - 1e-12:
                    best_score = float(score_a)
                    best_mean = float(mean_a)
                    best_unc = float(unc_a)
                    best_arms = [int(a)]
                elif abs(score_a - best_score) <= 1e-12:
                    best_arms.append(int(a))

            arm = int(self.rng.choice(best_arms)) if len(best_arms) > 1 else int(best_arms[0])

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
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")

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
