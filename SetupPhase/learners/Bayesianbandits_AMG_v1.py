"""
Bayesianbandits-style shared linear Thompson Sampling for BoomerAMG tuning.

This learner follows the same public API as the other setup-phase learners:
`predict(context) -> action_dict` and `update(loss)`.

Model:
  y_t = phi(x_t, a_t)^T theta + eps_t,    eps_t ~ N(0, sigma^2)

Posterior:
  (theta, sigma^2) use a Normal-Inverse-Gamma conjugate posterior and
  Thompson sampling for arm selection, inspired by the `bayesianbandits`
  package's linear contextual examples.

Feature map (same as SharedLinUCB_AMG_v3 / SharedLinTS_AMG):
  phi(x,a) = [ x ; g ; s1*g ; s2*g ; s3*g ; c_diag*g ] in R^50
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ._amg_action_features import (
    ACTION_FEATURE_DEFAULT_SCALES,
    ACTION_FEATURE_DIM,
    action_center_from_actions,
    action_param_vector,
    normalize_action_scales,
    poly2_features,
)
from ._candidate_subset import CandidateSelector


@dataclass(frozen=True)
class BayesianbanditsStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


def _robust_cholesky(A: np.ndarray) -> np.ndarray:
    """
    Compute a Cholesky factor with jitter/eigendecomposition fallback.
    Returns L such that A ~= L L^T.
    """
    A = 0.5 * (A + A.T)
    n = A.shape[0]
    I = np.eye(n, dtype=float)
    jitter = 0.0
    for exp in range(6):
        try:
            return np.linalg.cholesky(A + jitter * I)
        except np.linalg.LinAlgError:
            jitter = 10.0 ** (-12 + 2 * exp)

    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T


class Bayesianbandits_AMG_v1:
    _G_DIM = ACTION_FEATURE_DIM

    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        # Keep alpha in signature for compatibility with existing factories.
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        prior_a: float = 2.0,
        prior_b: float = 1.0,
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
        if not np.isfinite(l2_reg) or float(l2_reg) <= 0.0:
            raise ValueError("l2_reg must be finite and > 0")
        if not np.isfinite(alpha) or float(alpha) <= 0.0:
            raise ValueError("alpha must be finite and > 0")
        if not np.isfinite(prior_a) or float(prior_a) <= 0.0:
            raise ValueError("prior_a must be finite and > 0")
        if not np.isfinite(prior_b) or float(prior_b) <= 0.0:
            raise ValueError("prior_b must be finite and > 0")

        self.actions: List[Dict[str, Any]] = [dict(a) for a in actions]
        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.g_dim = int(self._G_DIM)
        self.d_phi = self.d_x + 5 * self.g_dim

        # `alpha` acts as posterior sample scale (temperature-like).
        self.alpha = float(alpha)
        self.l2_reg = float(l2_reg)
        self.prior_a = float(prior_a)
        self.prior_b = float(prior_b)

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
        self._g_actions = np.zeros((self.K, self.g_dim), dtype=float)
        for i, a in enumerate(self.actions):
            self._g_actions[i] = self._g_from_action(a)

        # Normal-Inverse-Gamma posterior state.
        self._A0 = self.l2_reg * np.eye(self.d_phi, dtype=float)
        self._m0 = np.zeros(self.d_phi, dtype=float)
        self._A0_m0 = self._A0 @ self._m0
        self._m0_quad = float(self._m0 @ self._A0_m0)

        self.A = self._A0.copy()
        self.A_inv = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.rhs = self._A0_m0.copy()
        self.y_sq_sum = 0.0
        self.n_obs = 0

        self._theta_mean = self._m0.copy()
        self._a_post = self.prior_a
        self._b_post = self.prior_b

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[BayesianbanditsStep] = []

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
        a = action_param_vector(params, err_prefix="Bayesianbandits_AMG_v1")
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

    def _eval_theta_dot_phi_subset(self, x: np.ndarray, *, arms: np.ndarray, theta: np.ndarray) -> np.ndarray:
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        s3 = float(x[self.s3_index])
        cd = float(x[self.cdiag_index])
        G = self._g_actions[np.asarray(arms, dtype=int)]

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
        vals = (
            base
            + np.einsum("ij,j->i", G, theta_g, optimize=False)
            + s1 * np.einsum("ij,j->i", G, theta_s1, optimize=False)
            + s2 * np.einsum("ij,j->i", G, theta_s2, optimize=False)
            + s3 * np.einsum("ij,j->i", G, theta_s3, optimize=False)
            + cd * np.einsum("ij,j->i", G, theta_cd, optimize=False)
        )
        return vals

    def _refresh_posterior(self) -> None:
        self._theta_mean = self.A_inv @ self.rhs
        self._a_post = self.prior_a + 0.5 * float(self.n_obs)

        quad = float(self._theta_mean @ (self.A @ self._theta_mean))
        b_post = self.prior_b + 0.5 * (float(self.y_sq_sum) + self._m0_quad - quad)
        if (not np.isfinite(b_post)) or b_post <= 1e-12:
            b_post = 1e-12
        self._b_post = float(b_post)

    def _sample_theta(self) -> np.ndarray:
        # sigma^2 ~ InvGamma(a,b), sampled via precision tau ~ Gamma(a, rate=b)
        tau = self.rng.gamma(shape=self._a_post, scale=1.0 / self._b_post)
        sigma2 = 1.0 / float(tau)

        L = _robust_cholesky(self.A_inv)
        z = self.rng.standard_normal(self.d_phi)
        return self._theta_mean + float(self.alpha) * np.sqrt(sigma2) * (L @ z)

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))
        theta = self._sample_theta()

        cand = self._cand.candidate_subset()
        vals = self._eval_theta_dot_phi_subset(x, arms=cand, theta=theta)
        best_val = float(np.min(vals))
        best_loc = np.flatnonzero(vals <= best_val + 1e-12)
        loc = int(best_loc[np.argmin(cand[best_loc])])
        arm = int(cand[loc])

        phi = self._phi(x, arm)
        pred_mean = float(self._theta_mean @ phi)
        quad = float(phi @ (self.A_inv @ phi))
        mean_var_scale = self._b_post / max(self._a_post - 1.0, 1e-12)
        pred_unc = float(np.sqrt(max(0.0, mean_var_scale * quad)))

        self._last_phi = phi
        self._last_arm = arm
        self.history.append(
            BayesianbanditsStep(
                t=self.t + 1,
                arm_index=arm,
                loss=float("nan"),
                pred_mean=pred_mean,
                pred_uncert=pred_unc,
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

        u = self.A_inv @ phi
        denom = 1.0 + float(phi @ u)
        if denom <= 0.0 or not np.isfinite(denom):
            self.A_inv = np.linalg.inv(self.A)
        else:
            self.A_inv = self.A_inv - np.outer(u, u) / denom

        self.rhs = self.rhs + y * phi
        self.y_sq_sum += y * y
        self.n_obs += 1
        self._refresh_posterior()

        last = self.history[-1]
        self.history[-1] = BayesianbanditsStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_phi = None
        self._last_arm = None
