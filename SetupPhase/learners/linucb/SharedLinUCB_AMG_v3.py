"""
Shared (joint) LinUCB for BoomerAMG setup-phase tuning (v3).

Changes vs v2:
- Scale action coordinates before building quadratic action features.
- Use full context-action interactions: s1,s2,s3,c_diag.

Feature definition (paper-ready)
--------------------------------
We tune:

    a = (th, mxrs, tr)

Center at a0 and scale by s:

    a_hat = (a - a0) / s

Let a_hat = (th~, mxrs~, tr~). Define g(a_hat) in R^9:

    g = [ th~, mxrs~, tr~, th~^2, mxrs~^2, tr~^2, th~*mxrs~, th~*tr~, mxrs~*tr~ ]^T

Context x in R^5 from `utils.stencil27_laplace.stencil_27_laplace`:

    x = [1, s1, s2, s3, c_diag]^T

Shared feature map:

    phi(x,a) = [ x ; g ; s1*g ; s2*g ; s3*g ; c_diag*g ] in R^(5+5*9)=R^50
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..common.action_features import (
    ACTION_FEATURE_DEFAULT_SCALES,
    ACTION_FEATURE_DIM,
    action_center_from_actions,
    action_param_vector,
    normalize_action_scales,
    poly2_features,
)
from ..common.candidate_subset import CandidateSelector


@dataclass(frozen=True)
class SharedLinUCBv3Step:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class SharedLinUCB_AMG_v3:
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
        s3_index: int = 3,
        cdiag_index: int = 4,
        action_center: Optional[Dict[str, Any]] = None,
        action_scales: Sequence[float] = ACTION_FEATURE_DEFAULT_SCALES,
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
        # x + (g + 4 context interactions) => 5 g-blocks
        self.d_phi = self.d_x + 5 * self.g_dim

        self.alpha = float(alpha)
        self.alpha_decay = bool(alpha_decay)
        self.l2_reg = float(l2_reg)

        self.s1_index = int(s1_index)
        self.s2_index = int(s2_index)
        self.s3_index = int(s3_index)
        self.cdiag_index = int(cdiag_index)
        for idx_name, idx in [
            ("s1_index", self.s1_index),
            ("s2_index", self.s2_index),
            ("s3_index", self.s3_index),
            ("cdiag_index", self.cdiag_index),
        ]:
            if not (0 <= int(idx) < self.d_x):
                raise ValueError(f"{idx_name} must be within [0, context_dim)")

        self._a_scales = normalize_action_scales(action_scales)

        self.rng = np.random.default_rng(seed)
        self._cand = CandidateSelector(
            self.K,
            candidate_pool_size=candidate_pool_size,
            always_include_arms=always_include_arms,
            elite_cache_size=int(elite_cache_size),
            rng=self.rng,
        )

        self._a_center = self._compute_action_center(action_center)

        self.A = np.eye(self.d_phi, dtype=float) * self.l2_reg
        self.A_inv = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.b = np.zeros(self.d_phi, dtype=float)

        self._g_actions = np.zeros((self.K, self.g_dim), dtype=float)
        for i, a in enumerate(self.actions):
            self._g_actions[i] = self._g_from_action(a)

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[SharedLinUCBv3Step] = []

    def _rebuild_inverse(self) -> None:
        A = 0.5 * (self.A + self.A.T)
        eye = np.eye(self.d_phi, dtype=float)
        scale = max(1.0, float(np.trace(A)) / max(1, self.d_phi))
        jitter = 1e-12 * scale
        for _ in range(8):
            try:
                L = np.linalg.cholesky(A + jitter * eye)
                y = np.linalg.solve(L, eye)
                self.A_inv = np.linalg.solve(L.T, y)
                self.A = A + jitter * eye
                return
            except np.linalg.LinAlgError:
                jitter *= 10.0
        self.A_inv = np.linalg.pinv(A, rcond=1e-12)
        self.A = A

    def _ensure_numeric_state(self) -> None:
        if not np.all(np.isfinite(self.A)) or not np.all(np.isfinite(self.A_inv)) or not np.all(np.isfinite(self.b)):
            self.A = np.nan_to_num(self.A, nan=0.0, posinf=1e12, neginf=-1e12)
            self.b = np.nan_to_num(self.b, nan=0.0, posinf=1e12, neginf=-1e12)
            if not np.any(self.A):
                self.A = np.eye(self.d_phi, dtype=float) * self.l2_reg
            self._rebuild_inverse()

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
        a = action_param_vector(params, err_prefix="SharedLinUCB_AMG_v3")
        a = (a - self._a_center) / self._a_scales
        return poly2_features(a)

    def _phi(self, x: np.ndarray, arm: int) -> np.ndarray:
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        s3 = float(x[self.s3_index])
        cd = float(x[self.cdiag_index])
        g = self._g_actions[int(arm)]

        phi = np.empty(self.d_phi, dtype=float)
        phi[: self.d_x] = x
        o = self.d_x
        phi[o : o + self.g_dim] = g
        o += self.g_dim
        phi[o : o + self.g_dim] = s1 * g
        o += self.g_dim
        phi[o : o + self.g_dim] = s2 * g
        o += self.g_dim
        phi[o : o + self.g_dim] = s3 * g
        o += self.g_dim
        phi[o : o + self.g_dim] = cd * g
        return phi

    def _score_subset(
        self,
        x: np.ndarray,
        *,
        arms: Optional[np.ndarray],
        alpha: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._ensure_numeric_state()
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        s3 = float(x[self.s3_index])
        cd = float(x[self.cdiag_index])
        G = self._g_actions if arms is None else self._g_actions[np.asarray(arms, dtype=int)]  # (K', g_dim)

        theta = np.linalg.solve(self.A, self.b)
        theta_x = theta[: self.d_x]
        off = self.d_x
        theta_g = theta[off : off + self.g_dim]
        off += self.g_dim
        theta_s1 = theta[off : off + self.g_dim]
        off += self.g_dim
        theta_s2 = theta[off : off + self.g_dim]
        off += self.g_dim
        theta_s3 = theta[off : off + self.g_dim]
        off += self.g_dim
        theta_cd = theta[off : off + self.g_dim]

        base = float(theta_x @ x)
        mean = (
            base
            + np.einsum("ij,j->i", G, theta_g, optimize=False)
            + s1 * np.einsum("ij,j->i", G, theta_s1, optimize=False)
            + s2 * np.einsum("ij,j->i", G, theta_s2, optimize=False)
            + s3 * np.einsum("ij,j->i", G, theta_s3, optimize=False)
            + cd * np.einsum("ij,j->i", G, theta_cd, optimize=False)
        )

        # Structured uncertainty: phi = c + P g, where g varies per arm.
        c = np.concatenate([x, np.zeros(5 * self.g_dim, dtype=float)], axis=0)
        I = np.eye(self.g_dim, dtype=float)
        P = np.vstack(
            [
                np.zeros((self.d_x, self.g_dim), dtype=float),
                I,
                s1 * I,
                s2 * I,
                s3 * I,
                cd * I,
            ]
        )

        Ac = np.linalg.solve(self.A, c)
        q0 = float(c @ Ac)
        AP = np.linalg.solve(self.A, P)
        u = P.T @ Ac
        M = P.T @ AP

        quad = q0 + 2.0 * np.einsum("ij,j->i", G, u, optimize=False) + np.einsum("ij,jk,ik->i", G, M, G, optimize=False)
        uncert = np.sqrt(np.maximum(0.0, quad))
        score = mean - float(alpha) * uncert
        return score, mean, uncert

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        alpha_eff = float(self.alpha) / np.sqrt(self.t + 1.0) if self.alpha_decay else float(self.alpha)
        M = self._cand.candidate_pool_size
        if M is None or int(M) >= self.K:
            score, mean, uncert = self._score_subset(x, arms=None, alpha=alpha_eff)
            best_score = float(np.min(score))
            best_arms = np.flatnonzero(score <= best_score + 1e-12)
            arm = int(np.min(best_arms))
            best_mean = float(mean[arm])
            best_unc = float(uncert[arm])
        else:
            cand = self._cand.candidate_subset()
            score, mean, uncert = self._score_subset(x, arms=cand, alpha=alpha_eff)
            best_score = float(np.min(score))
            best_loc = np.flatnonzero(score <= best_score + 1e-12)
            loc = int(best_loc[np.argmin(cand[best_loc])])
            arm = int(cand[loc])
            best_mean = float(mean[loc])
            best_unc = float(uncert[loc])

        self._last_phi = self._phi(x, arm)
        self._last_arm = arm

        self.history.append(
            SharedLinUCBv3Step(
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

        self._cand.observe(arm, y)

        self.A = self.A + np.outer(phi, phi)
        self.b = self.b + y * phi
        self._rebuild_inverse()

        last = self.history[-1]
        self.history[-1] = SharedLinUCBv3Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_phi = None
        self._last_arm = None
