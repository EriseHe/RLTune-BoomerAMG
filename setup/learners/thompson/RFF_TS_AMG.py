"""
Kernel Thompson Sampling via Random Fourier Features (RFF-TS) for BoomerAMG.

We build a compact input vector:

  u = [s1, s2, s3, c_diag / c_scale, th~, mxrs~, tr~]  in R^7

where (th~,mxrs~,tr~) are scaled, centered action coordinates.

We approximate an RBF kernel with random Fourier features z(u) in R^D:

  z(u)_i = sqrt(2/D) * cos(omega_i^T u + b_i)

and then run linear Thompson Sampling on z(u).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..common.candidate_subset import CandidateSelector
from .SharedLinTS_AMG import _robust_cholesky


@dataclass(frozen=True)
class RFFTSStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class RFF_TS_AMG:
    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        # Keep alpha in signature for compatibility with the factory harness.
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        sigma: float = 0.1,
        rff_dim: int = 128,
        lengthscale: float = 1.0,
        cdiag_scale: float = 2.5,
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
        if not np.isfinite(sigma) or float(sigma) <= 0.0:
            raise ValueError("sigma must be finite and > 0")
        if int(rff_dim) <= 0:
            raise ValueError("rff_dim must be positive")
        if not np.isfinite(lengthscale) or float(lengthscale) <= 0.0:
            raise ValueError("lengthscale must be finite and > 0")
        if not np.isfinite(cdiag_scale) or float(cdiag_scale) <= 0.0:
            raise ValueError("cdiag_scale must be finite and > 0")

        self.actions: List[Dict[str, Any]] = [dict(a) for a in actions]
        self.K = len(self.actions)
        self.d_x = int(context_dim)

        self.l2_reg = float(l2_reg)
        self.sigma = float(sigma)

        self.rff_dim = int(rff_dim)
        self.lengthscale = float(lengthscale)
        self.cdiag_scale = float(cdiag_scale)

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

        # Precompute scaled action coordinates (th~,mxrs~,tr~) for all arms.
        self._a_scaled = np.zeros((self.K, 3), dtype=float)
        for i, a in enumerate(self.actions):
            self._a_scaled[i] = self._scaled_action(a)

        # RFF parameters: omega ~ N(0, ell^{-2} I), b ~ Unif(0,2pi)
        d_u = 7
        omega_scale = 1.0 / self.lengthscale
        self._omega = self.rng.normal(loc=0.0, scale=omega_scale, size=(self.rff_dim, d_u)).astype(float)
        self._b = self.rng.uniform(0.0, 2.0 * np.pi, size=(self.rff_dim,)).astype(float)
        self._z_scale = float(np.sqrt(2.0 / float(self.rff_dim)))

        # Linear TS in RFF space
        self.A_inv = np.eye(self.rff_dim, dtype=float) / self.l2_reg
        self.b_lin = np.zeros(self.rff_dim, dtype=float)

        self.t = 0
        self._last_z: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[RFFTSStep] = []

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

    def _scaled_action(self, params: Dict[str, Any]) -> np.ndarray:
        th = float(params["strong_threshold"])
        mxrs = float(params["max_row_sum"])
        tr = float(params["trunc_factor"])
        th = (th - float(self._a_center[0])) / float(self._a_scales[0])
        mxrs = (mxrs - float(self._a_center[1])) / float(self._a_scales[1])
        tr = (tr - float(self._a_center[2])) / float(self._a_scales[2])
        return np.array([th, mxrs, tr], dtype=float)

    def _u_matrix(self, x: np.ndarray, arms: np.ndarray) -> np.ndarray:
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        s3 = float(x[self.s3_index])
        cd = float(x[self.cdiag_index]) / float(self.cdiag_scale)

        A = self._a_scaled[np.asarray(arms, dtype=int)]  # (K',3)
        U = np.empty((A.shape[0], 7), dtype=float)
        U[:, 0] = s1
        U[:, 1] = s2
        U[:, 2] = s3
        U[:, 3] = cd
        U[:, 4:7] = A
        return U

    def _z_matrix(self, U: np.ndarray) -> np.ndarray:
        # U: (K',7), omega: (D,7) => proj: (K',D)
        proj = U @ self._omega.T
        proj = proj + self._b  # broadcast
        return self._z_scale * np.cos(proj)

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        mu = self.A_inv @ self.b_lin
        L = _robust_cholesky(self.A_inv)
        z0 = self.rng.standard_normal(self.rff_dim)
        theta = mu + float(self.sigma) * (L @ z0)

        cand = self._cand.candidate_subset()
        U = self._u_matrix(x, cand)
        Z = self._z_matrix(U)  # (K', D)
        vals = Z @ theta
        best_val = float(np.min(vals))
        best_loc = np.flatnonzero(vals <= best_val + 1e-12)
        loc = int(best_loc[np.argmin(cand[best_loc])])
        arm = int(cand[loc])

        z_sel = Z[loc].astype(float, copy=True)
        pred_mean = float(mu @ z_sel)
        quad = float(z_sel @ (self.A_inv @ z_sel))
        pred_unc = float(np.sqrt(max(0.0, quad)))

        self._last_z = z_sel
        self._last_arm = arm
        self.history.append(
            RFFTSStep(
                t=self.t + 1,
                arm_index=arm,
                loss=float("nan"),
                pred_mean=pred_mean,
                pred_uncert=pred_unc,
            )
        )
        return dict(self.actions[arm])

    def update(self, loss: float) -> None:
        if self._last_z is None or self._last_arm is None:
            raise RuntimeError("update() called before predict()")

        z = self._last_z
        arm = int(self._last_arm)
        y = float(loss)
        if not np.isfinite(y):
            raise ValueError("loss must be finite")

        self._cand.observe(arm, y)

        u = self.A_inv @ z
        denom = 1.0 + float(z @ u)
        if denom <= 0.0 or not np.isfinite(denom):
            A = np.linalg.inv(self.A_inv)
            A = A + np.outer(z, z)
            self.A_inv = np.linalg.inv(A)
        else:
            self.A_inv = self.A_inv - np.outer(u, u) / denom

        self.b_lin = self.b_lin + y * z

        last = self.history[-1]
        self.history[-1] = RFFTSStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_z = None
        self._last_arm = None
