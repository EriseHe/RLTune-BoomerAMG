"""
Disjoint LinUCB for BoomerAMG setup-phase tuning.

This is a *contextual* bandit: each round observes a context vector x_t
(features describing the current linear-system instance) and selects a
multi-parameter BoomerAMG setup configuration from a finite action set.

We model the loss (e.g., Work Units) for each arm a as:

    E[loss_t | x_t, a] ≈ x_t^T theta_a

and use LinUCB-style lower confidence bounds (LCB) to *minimize* loss:

    choose a_t = argmin_a  (x_t^T theta_hat_a - alpha * sqrt(x_t^T A_a^{-1} x_t))

where A_a is the ridge-regularized design matrix for arm a.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LinUCBStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class LinUCB_AMG:
    """
    Disjoint LinUCB (per-arm linear model) for minimizing a scalar loss.

    Parameters
    ----------
    actions
        List of BoomerAMG parameter dicts. Each entry is passed to
        solver.solve(params=...).
    context_dim
        Dimension of the context feature vector x_t.
    alpha
        Exploration strength (higher => more exploration).
    l2_reg
        Ridge regularization parameter (lambda). Each arm starts with
        A = lambda * I.
    seed
        RNG seed used only for tie-breaking.
    """

    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if not actions:
            raise ValueError("actions must be non-empty")

        self.actions: List[Dict[str, Any]] = [dict(a) for a in actions]
        self.K = len(self.actions)
        self.d = int(context_dim)

        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        if self.l2_reg <= 0.0:
            raise ValueError("l2_reg must be > 0")

        self.rng = np.random.default_rng(seed)

        # Per-arm ridge matrices in inverse form (Sherman-Morrison updates).
        # A_a = lambda I + sum x x^T, so A_a^{-1} starts at (1/lambda) I.
        eye = np.eye(self.d, dtype=float) / self.l2_reg
        self.A_inv = np.repeat(eye[None, :, :], self.K, axis=0)
        self.b = np.zeros((self.K, self.d), dtype=float)

        self.t = 0
        self._last_x: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[LinUCBStep] = []

    def _validate_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.d:
            raise ValueError(f"context has dim {x.size}, expected {self.d}")
        if not np.all(np.isfinite(x)):
            raise ValueError("context contains non-finite values")
        return x

    def _arm_stats(self, arm: int, x: np.ndarray) -> Tuple[float, float]:
        """
        Returns (pred_mean, pred_uncert) for loss model of a given arm.
        """
        A_inv = self.A_inv[arm]
        theta = A_inv @ self.b[arm]
        mean = float(theta @ x)
        # x^T A^{-1} x is always >= 0 (numerically may go slightly negative).
        quad = float(x @ (A_inv @ x))
        uncert = float(np.sqrt(max(0.0, quad)))
        return mean, uncert

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        """
        Choose an arm for the given context and return its parameter dict.
        """
        x = self._validate_x(np.asarray(list(context), dtype=float))

        # Vectorized scoring across all arms.
        # u_a = A_a^{-1} x, mean_a = b_a^T u_a, uncert_a = sqrt(x^T u_a)
        u = self.A_inv @ x
        mean = np.sum(self.b * u, axis=1)
        quad = np.sum(u * x, axis=1)
        uncert = np.sqrt(np.maximum(0.0, quad))

        score = mean - self.alpha * uncert  # LCB for loss minimization
        best_score = float(np.min(score))
        best_arms = np.flatnonzero(score <= best_score + 1e-12)
        arm = int(self.rng.choice(best_arms)) if best_arms.size > 1 else int(best_arms[0])
        best_mean = float(mean[arm])
        best_unc = float(uncert[arm])

        # Cache for update()
        self._last_x = x
        self._last_arm = arm

        # Record prediction stats for diagnostics (loss filled in update()).
        self.history.append(
            LinUCBStep(
                t=self.t + 1,
                arm_index=arm,
                loss=float("nan"),
                pred_mean=best_mean,
                pred_uncert=best_unc,
            )
        )

        return dict(self.actions[arm])

    def update(self, loss: float) -> None:
        """
        Update the chosen arm with the observed scalar loss.
        """
        if self._last_x is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")

        x = self._last_x
        arm = self._last_arm
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")

        # Sherman-Morrison: (A + x x^T)^{-1}
        A_inv = self.A_inv[arm]
        u = A_inv @ x
        denom = 1.0 + float(x @ u)
        if denom <= 0.0 or not np.isfinite(denom):
            # Extremely unlikely; fall back to explicit inverse update.
            A = np.linalg.inv(A_inv)
            A = A + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(A)
        else:
            self.A_inv[arm] = A_inv - np.outer(u, u) / denom

        self.b[arm] = self.b[arm] + y * x

        # Patch the last history entry with realized loss.
        last = self.history[-1]
        self.history[-1] = LinUCBStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1

        # Clear cache to prevent accidental double-update.
        self._last_x = None
        self._last_arm = None
