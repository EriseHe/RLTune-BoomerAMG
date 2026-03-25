"""
Shared (joint) LinUCB for BoomerAMG setup-phase tuning (continuous knobs).

This is a *contextual* bandit: each round observes a context vector x_t
(features describing the current linear-system instance) and selects a
multi-parameter BoomerAMG setup configuration from a finite action set.

Unlike the existing disjoint LinUCB (per-arm models), this learner uses a
single shared linear model:

    E[loss_t | x_t, a] ≈ phi(x_t, a)^T theta

so samples gathered from one (nearby) action can generalize to others.

Feature map (as requested)
--------------------------
Let a = (th, mxrs, tr) and define:

    g(a) = [th, mxrs, tr, th^2, mxrs^2, tr^2, th*mxrs, th*tr, mxrs*tr]^T

and for context x = [1, s1, s2, ...], define:

    phi(x, a) = [ x ; g(a) ; s1*g(a) ; s2*g(a) ].

We use LinUCB-style lower confidence bounds (LCB) to *minimize* loss:

    choose a_t = argmin_a ( phi^T theta_hat - alpha * sqrt(phi^T A^{-1} phi) ).
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


class SharedLinUCB_AMG:
    """
    Shared (joint) LinUCB for minimizing a scalar loss.

    Parameters
    ----------
    actions
        List of BoomerAMG parameter dicts. Each entry is passed to
        solver.solve(params=...).
        For this learner, each action must contain:
          - strong_threshold
          - max_row_sum
          - trunc_factor
    context_dim
        Dimension of the context feature vector x_t.
    alpha
        Exploration strength (higher => more exploration).
    l2_reg
        Ridge regularization parameter (lambda). Starts with A = lambda * I.
    s1_index, s2_index
        Indices inside the context vector corresponding to s1 and s2.
        Defaults match `utils.stencil27_laplace.stencil_27_laplace` context:
          x = [1, s1, s2, s3, c_diag]
    seed
        RNG seed used only for tie-breaking.
    """

    _G_DIM = 9

    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        s1_index: int = 1,
        s2_index: int = 2,
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
        self.l2_reg = float(l2_reg)
        self.s1_index = int(s1_index)
        self.s2_index = int(s2_index)
        if not (0 <= self.s1_index < self.d_x and 0 <= self.s2_index < self.d_x):
            raise ValueError("s1_index/s2_index must be within [0, context_dim)")

        self.rng = np.random.default_rng(seed)

        # Shared ridge inverse (Sherman-Morrison updates).
        self.A_inv = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.b = np.zeros(self.d_phi, dtype=float)

        # Precompute g(a) for all actions (depends only on action dict).
        self._g_actions = np.zeros((self.K, self.g_dim), dtype=float)
        for i, a in enumerate(self.actions):
            self._g_actions[i] = self._g_from_action(a)

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[SharedLinUCBStep] = []

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
            raise KeyError(f"SharedLinUCB_AMG action missing required key: {e}") from e

        if not (np.isfinite(th) and np.isfinite(mxrs) and np.isfinite(tr)):
            raise ValueError("action parameters must be finite")

        g = np.array(
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
        return g

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

    def _arm_stats(self, x: np.ndarray, arm: int) -> Tuple[float, float]:
        phi = self._phi(x, arm)
        theta = self.A_inv @ self.b
        mean = float(theta @ phi)
        quad = float(phi @ (self.A_inv @ phi))
        uncert = float(np.sqrt(max(0.0, quad)))
        return mean, uncert

    def _score_all(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized (score, mean, uncert) for all arms.

        Uses the structure of phi(x,a) = [x; g; s1*g; s2*g] to compute
        phi^T A^{-1} phi in O(K*g_dim^2) per round rather than O(K*d_phi^2).
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
        # Avoid BLAS for these small contractions; OpenBLAS can be very slow here.
        mean = (
            base
            + np.einsum("ij,j->i", G, theta_g, optimize=False)
            + s1 * np.einsum("ij,j->i", G, theta_s1, optimize=False)
            + s2 * np.einsum("ij,j->i", G, theta_s2, optimize=False)
        )

        # Build c and P such that phi = c + P g.
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
        u = P.T @ Ac  # (g_dim,)
        M = P.T @ AP  # (g_dim, g_dim)

        quad = q0 + 2.0 * np.einsum("ij,j->i", G, u, optimize=False) + np.einsum("ij,jk,ik->i", G, M, G, optimize=False)
        uncert = np.sqrt(np.maximum(0.0, quad))
        score = mean - self.alpha * uncert
        return score, mean, uncert

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        score, mean, uncert = self._score_all(x)
        best_score = float(np.min(score))
        best_arms = np.flatnonzero(score <= best_score + 1e-12)
        arm = int(self.rng.choice(best_arms)) if best_arms.size > 1 else int(best_arms[0])
        best_mean = float(mean[arm])
        best_unc = float(uncert[arm])

        # Cache for update()
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
        arm = self._last_arm
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")

        # Sherman-Morrison update for shared A_inv.
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
            arm_index=arm,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1

        # Clear cache to prevent accidental double-update.
        self._last_phi = None
        self._last_arm = None
