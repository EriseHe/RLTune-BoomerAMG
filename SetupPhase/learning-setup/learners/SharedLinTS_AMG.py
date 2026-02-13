"""
Shared Linear Thompson Sampling (LinTS) for BoomerAMG setup-phase tuning.

Uses the same shared feature map as SharedLinUCB_AMG_v3:

  phi(x,a) = [ x ; g ; s1*g ; s2*g ; s3*g ; c_diag*g ]  in R^50

Action selection samples a parameter vector from the posterior and chooses
the arm minimizing sampled loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ._candidate_subset import CandidateSelector


@dataclass(frozen=True)
class SharedLinTSStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


def _robust_cholesky(A: np.ndarray) -> np.ndarray:
    """
    Compute a Cholesky factor of a symmetric PSD-ish matrix with jitter fallback.
    Returns L such that A ≈ L L^T.
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

    # Eigen fallback: clamp eigenvalues to keep PSD.
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, 1e-12)
    return (V * np.sqrt(w)) @ V.T


class SharedLinTS_AMG:
    _G_DIM = 9

    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        # Keep alpha in signature for compatibility with the factory harness.
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        sigma: float = 0.1,
        s1_index: int = 1,
        s2_index: int = 2,
        s3_index: int = 3,
        cdiag_index: int = 4,
        action_center: Optional[Dict[str, Any]] = None,
        action_scales: Sequence[float] = (0.25, 0.1, 0.2),
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
        self.d_phi = self.d_x + 5 * self.g_dim  # x + (g + 4 interactions)

        self.l2_reg = float(l2_reg)
        self.sigma = float(sigma)
        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError("sigma must be finite and > 0")

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

        scales = np.asarray(list(action_scales), dtype=float).reshape(-1)
        if scales.size != 3:
            raise ValueError("action_scales must have length 3")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("action_scales must be finite and > 0")
        self._a_scales = scales

        self.rng = np.random.default_rng(seed)
        self._cand = CandidateSelector(
            self.K,
            candidate_pool_size=candidate_pool_size,
            always_include_arms=always_include_arms,
            elite_cache_size=int(elite_cache_size),
            rng=self.rng,
        )

        self._a_center = self._compute_action_center(action_center)

        self.A_inv = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.b = np.zeros(self.d_phi, dtype=float)

        self._g_actions = np.zeros((self.K, self.g_dim), dtype=float)
        for i, a in enumerate(self.actions):
            self._g_actions[i] = self._g_from_action(a)

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[SharedLinTSStep] = []

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
            raise KeyError(f"SharedLinTS_AMG action missing required key: {e}") from e

        if not (np.isfinite(th) and np.isfinite(mxrs) and np.isfinite(tr)):
            raise ValueError("action parameters must be finite")

        th = (th - float(self._a_center[0])) / float(self._a_scales[0])
        mxrs = (mxrs - float(self._a_center[1])) / float(self._a_scales[1])
        tr = (tr - float(self._a_center[2])) / float(self._a_scales[2])

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

    def _quad_one(self, x: np.ndarray, arm: int) -> float:
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        s3 = float(x[self.s3_index])
        cd = float(x[self.cdiag_index])
        g = self._g_actions[int(arm)]

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

        Ainv = self.A_inv
        Ac = Ainv @ c
        q0 = float(c @ Ac)
        u = P.T @ Ac
        M = P.T @ (Ainv @ P)
        quad = q0 + 2.0 * float(g @ u) + float(g @ (M @ g))
        return quad

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        mu = self.A_inv @ self.b
        L = _robust_cholesky(self.A_inv)
        z = self.rng.standard_normal(self.d_phi)
        theta = mu + float(self.sigma) * (L @ z)

        cand = self._cand.candidate_subset()
        vals = self._eval_theta_dot_phi_subset(x, arms=cand, theta=theta)
        best_val = float(np.min(vals))
        best_loc = np.flatnonzero(vals <= best_val + 1e-12)
        loc = int(best_loc[np.argmin(cand[best_loc])])
        arm = int(cand[loc])

        phi = self._phi(x, arm)
        pred_mean = float(mu @ phi)
        quad = float(self._quad_one(x, arm))
        pred_unc = float(np.sqrt(max(0.0, quad)))

        self._last_phi = phi
        self._last_arm = arm
        self.history.append(
            SharedLinTSStep(
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
        self.history[-1] = SharedLinTSStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_phi = None
        self._last_arm = None

