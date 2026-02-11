"""
LinUCB for BoomerAMG setup-phase tuning (variant).

This is the same algorithm as `learners/LinUCB_AMG.py`, but with one small
extension:
- optional exploration decay: alpha_t = alpha / sqrt(t+1)

This file exists so experiments can compare against the original
implementation without modifying it.

Exploration schedule (paper-ready)
----------------------------------
If `alpha_decay=True`:

    alpha_t = alpha / sqrt(t+1)

where t starts at 0 internally (so the first round uses alpha_0 = alpha).
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


class LinUCB_AMG_v2:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        alpha: float = 1.0,
        alpha_decay: bool = False,
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
        self.alpha_decay = bool(alpha_decay)
        self.l2_reg = float(l2_reg)
        if self.l2_reg <= 0.0:
            raise ValueError("l2_reg must be > 0")

        self.rng = np.random.default_rng(seed)

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

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        u = self.A_inv @ x
        mean = np.sum(self.b * u, axis=1)
        quad = np.sum(u * x, axis=1)
        uncert = np.sqrt(np.maximum(0.0, quad))

        alpha_eff = float(self.alpha) / np.sqrt(self.t + 1.0) if self.alpha_decay else float(self.alpha)
        score = mean - alpha_eff * uncert

        best_score = float(np.min(score))
        best_arms = np.flatnonzero(score <= best_score + 1e-12)
        arm = int(self.rng.choice(best_arms)) if best_arms.size > 1 else int(best_arms[0])

        self._last_x = x
        self._last_arm = arm

        self.history.append(
            LinUCBStep(
                t=self.t + 1,
                arm_index=arm,
                loss=float("nan"),
                pred_mean=float(mean[arm]),
                pred_uncert=float(uncert[arm]),
            )
        )

        return dict(self.actions[arm])

    def update(self, loss: float) -> None:
        if self._last_x is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")

        x = self._last_x
        arm = self._last_arm
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")

        A_inv = self.A_inv[arm]
        u = A_inv @ x
        denom = 1.0 + float(x @ u)
        if denom <= 0.0 or not np.isfinite(denom):
            A = np.linalg.inv(A_inv)
            A = A + np.outer(x, x)
            self.A_inv[arm] = np.linalg.inv(A)
        else:
            self.A_inv[arm] = A_inv - np.outer(u, u) / denom

        self.b[arm] = self.b[arm] + y * x

        last = self.history[-1]
        self.history[-1] = LinUCBStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_x = None
        self._last_arm = None
