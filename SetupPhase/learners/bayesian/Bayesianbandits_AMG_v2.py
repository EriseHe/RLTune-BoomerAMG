"""
Bayesianbandits-style shared linear Thompson Sampling for BoomerAMG tuning (v2).

v2 applies three practical improvements over v1:
1) Reduced late-stage exploration via decayed sampling scale alpha_t.
2) Stabilized posterior noise via clipped/tempered sigma^2 sampling.
3) Stronger prior toward a known good action region using pseudo-observations.

Public API is unchanged: `predict(context) -> action_dict`, `update(loss)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

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
class BayesianbanditsV2Step:
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


class Bayesianbandits_AMG_v2:
    _G_DIM = ACTION_FEATURE_DIM

    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        # Keep alpha in signature for compatibility with existing factories.
        alpha: float = 1.0,
        alpha_decay: bool = True,
        alpha_decay_power: float = 0.5,
        alpha_min: float = 0.20,
        l2_reg: float = 1.0,
        prior_a: float = 2.5,
        prior_b: float = 1.0,
        sigma2_floor: float = 1e-8,
        sigma2_clip_mult: float = 6.0,
        sigma2_mean_mix: float = 0.25,
        prior_action: Optional[Dict[str, Any]] = None,
        prior_action_strength: float = 6.0,
        prior_action_target_loss: float = 0.04,
        prior_context: Optional[Sequence[float]] = None,
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
        if not np.isfinite(alpha) or float(alpha) <= 0.0:
            raise ValueError("alpha must be finite and > 0")
        if not np.isfinite(alpha_decay_power) or float(alpha_decay_power) < 0.0:
            raise ValueError("alpha_decay_power must be finite and >= 0")
        if not np.isfinite(alpha_min) or float(alpha_min) <= 0.0:
            raise ValueError("alpha_min must be finite and > 0")
        if not np.isfinite(l2_reg) or float(l2_reg) <= 0.0:
            raise ValueError("l2_reg must be finite and > 0")
        if not np.isfinite(prior_a) or float(prior_a) <= 0.0:
            raise ValueError("prior_a must be finite and > 0")
        if not np.isfinite(prior_b) or float(prior_b) <= 0.0:
            raise ValueError("prior_b must be finite and > 0")
        if not np.isfinite(sigma2_floor) or float(sigma2_floor) <= 0.0:
            raise ValueError("sigma2_floor must be finite and > 0")
        if not np.isfinite(sigma2_clip_mult) or float(sigma2_clip_mult) <= 0.0:
            raise ValueError("sigma2_clip_mult must be finite and > 0")
        if not np.isfinite(sigma2_mean_mix) or not (0.0 <= float(sigma2_mean_mix) <= 1.0):
            raise ValueError("sigma2_mean_mix must be in [0, 1]")
        if not np.isfinite(prior_action_strength) or float(prior_action_strength) < 0.0:
            raise ValueError("prior_action_strength must be finite and >= 0")
        if not np.isfinite(prior_action_target_loss):
            raise ValueError("prior_action_target_loss must be finite")

        self.actions: List[Dict[str, Any]] = [dict(a) for a in actions]
        self.K = len(self.actions)
        self.d_x = int(context_dim)
        self.g_dim = int(self._G_DIM)
        self.d_phi = self.d_x + 5 * self.g_dim

        self.alpha = float(alpha)
        self.alpha_decay = bool(alpha_decay)
        self.alpha_decay_power = float(alpha_decay_power)
        self.alpha_min = float(alpha_min)

        self.l2_reg = float(l2_reg)
        self.prior_a = float(prior_a)
        self.prior_b = float(prior_b)

        self.sigma2_floor = float(sigma2_floor)
        self.sigma2_clip_mult = float(sigma2_clip_mult)
        self.sigma2_mean_mix = float(sigma2_mean_mix)

        self.prior_action_strength = float(prior_action_strength)
        self.prior_action_target_loss = float(prior_action_target_loss)
        self.prior_action = dict(
            prior_action
            if prior_action is not None
            else {
                "strong_threshold": 0.95,
                "max_row_sum": 0.95,
                "trunc_factor": 0.0,
            }
        )

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

        # Prior context for pseudo-observations (intercept-only by default).
        if prior_context is None:
            x_ref = np.zeros(self.d_x, dtype=float)
            if self.d_x > 0:
                x_ref[0] = 1.0
            self._x_prior = x_ref
        else:
            self._x_prior = self._validate_x(np.asarray(list(prior_context), dtype=float))

        # Normal-Inverse-Gamma prior with optional action-targeted pseudo-observations.
        self._A0 = self.l2_reg * np.eye(self.d_phi, dtype=float)
        self._A0_m0 = np.zeros(self.d_phi, dtype=float)
        if self.prior_action_strength > 0.0:
            g_prior = self._g_from_action(self.prior_action)
            phi_prior = self._phi_from_g(self._x_prior, g_prior)
            w = self.prior_action_strength
            self._A0 = self._A0 + w * np.outer(phi_prior, phi_prior)
            self._A0_m0 = self._A0_m0 + (w * self.prior_action_target_loss) * phi_prior

        self._m0 = np.linalg.solve(self._A0, self._A0_m0)
        self._m0_quad = float(self._m0 @ self._A0_m0)

        self.A = self._A0.copy()
        self.A_inv = np.linalg.inv(self._A0)
        self.rhs = self._A0_m0.copy()
        self.y_sq_sum = 0.0
        self.n_obs = 0

        self._theta_mean = self._m0.copy()
        self._a_post = self.prior_a
        self._b_post = self.prior_b
        self._refresh_posterior()

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[BayesianbanditsV2Step] = []

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
        a = action_param_vector(params, err_prefix="Bayesianbandits_AMG_v2")
        a = (a - self._a_center) / self._a_scales
        return poly2_features(a)

    def _phi_from_g(self, x: np.ndarray, g: np.ndarray) -> np.ndarray:
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        s3 = float(x[self.s3_index])
        cd = float(x[self.cdiag_index])

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

    def _phi(self, x: np.ndarray, arm: int) -> np.ndarray:
        return self._phi_from_g(x, self._g_actions[int(arm)])

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

    def _posterior_sigma2_mean(self) -> float:
        return float(self._b_post / max(self._a_post - 1.0, 1e-12))

    def _effective_alpha(self) -> float:
        if not self.alpha_decay:
            return self.alpha
        out = self.alpha / ((self.t + 1.0) ** self.alpha_decay_power)
        return float(max(self.alpha_min, out))

    def _refresh_posterior(self) -> None:
        self._theta_mean = self.A_inv @ self.rhs
        self._a_post = self.prior_a + 0.5 * float(self.n_obs)

        quad = float(self._theta_mean @ (self.A @ self._theta_mean))
        b_post = self.prior_b + 0.5 * (float(self.y_sq_sum) + self._m0_quad - quad)
        if (not np.isfinite(b_post)) or b_post <= 1e-12:
            b_post = 1e-12
        self._b_post = float(b_post)

    def _sample_theta(self) -> np.ndarray:
        # sigma^2 ~ InvGamma(a,b), sampled via precision tau ~ Gamma(a, rate=b).
        tau = self.rng.gamma(shape=self._a_post, scale=1.0 / self._b_post)
        if (not np.isfinite(tau)) or tau <= 0.0:
            tau = 1.0 / max(self._posterior_sigma2_mean(), self.sigma2_floor)

        sigma2 = 1.0 / float(tau)
        sigma2_mean = self._posterior_sigma2_mean()
        sigma2_cap = max(self.sigma2_floor, self.sigma2_clip_mult * sigma2_mean)
        sigma2 = float(np.clip(sigma2, self.sigma2_floor, sigma2_cap))
        sigma2 = (1.0 - self.sigma2_mean_mix) * sigma2 + self.sigma2_mean_mix * sigma2_mean

        alpha_eff = self._effective_alpha()
        L = _robust_cholesky(self.A_inv)
        z = self.rng.standard_normal(self.d_phi)
        return self._theta_mean + float(alpha_eff) * np.sqrt(max(self.sigma2_floor, sigma2)) * (L @ z)

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
        pred_unc = float(np.sqrt(max(0.0, self._posterior_sigma2_mean() * quad)))

        self._last_phi = phi
        self._last_arm = arm
        self.history.append(
            BayesianbanditsV2Step(
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
        self.history[-1] = BayesianbanditsV2Step(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_phi = None
        self._last_arm = None
