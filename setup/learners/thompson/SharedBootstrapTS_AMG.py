"""
Bootstrap Thompson Sampling (online Poisson bootstrap) for BoomerAMG tuning.

Maintains an ensemble of linear models over the v3 shared feature map:

  phi(x,a) = [ x ; g ; s1*g ; s2*g ; s3*g ; c_diag*g ]  in R^50

Action selection:
- pick a random head J
- choose arm minimizing mu_J^T phi(x,a) over candidate set

Update:
- for each head j, sample w_j ~ Poisson(1) and apply a weighted rank-1 update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..common.candidate_subset import CandidateSelector


@dataclass(frozen=True)
class SharedBootstrapTSStep:
    t: int
    arm_index: int
    loss: float
    pred_mean: float
    pred_uncert: float


class SharedBootstrapTS_AMG:
    _G_DIM = 9

    def __init__(
        self,
        actions: Sequence[Dict[str, Any]],
        context_dim: int,
        *,
        # Keep alpha in signature for compatibility with the factory harness.
        alpha: float = 1.0,
        l2_reg: float = 1.0,
        heads: int = 10,
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
        self.d_phi = self.d_x + 5 * self.g_dim

        self.l2_reg = float(l2_reg)

        self.heads = int(heads)
        if self.heads <= 0:
            raise ValueError("heads must be positive")

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

        eye = np.eye(self.d_phi, dtype=float) / self.l2_reg
        self.A_inv = np.repeat(eye[None, :, :], self.heads, axis=0)  # (B, d, d)
        self.b = np.zeros((self.heads, self.d_phi), dtype=float)  # (B, d)

        self._g_actions = np.zeros((self.K, self.g_dim), dtype=float)
        for i, a in enumerate(self.actions):
            self._g_actions[i] = self._g_from_action(a)

        self.t = 0
        self._last_phi: Optional[np.ndarray] = None
        self._last_arm: Optional[int] = None
        self.history: List[SharedBootstrapTSStep] = []

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
            raise KeyError(f"SharedBootstrapTS_AMG action missing required key: {e}") from e

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

    def _eval_mu_dot_phi_subset(self, x: np.ndarray, *, arms: np.ndarray, mu: np.ndarray) -> np.ndarray:
        s1 = float(x[self.s1_index])
        s2 = float(x[self.s2_index])
        s3 = float(x[self.s3_index])
        cd = float(x[self.cdiag_index])
        G = self._g_actions[np.asarray(arms, dtype=int)]

        mu_x = mu[: self.d_x]
        off = self.d_x
        mu_g = mu[off : off + self.g_dim]
        off += self.g_dim
        mu_s1 = mu[off : off + self.g_dim]
        off += self.g_dim
        mu_s2 = mu[off : off + self.g_dim]
        off += self.g_dim
        mu_s3 = mu[off : off + self.g_dim]
        off += self.g_dim
        mu_cd = mu[off : off + self.g_dim]

        base = float(mu_x @ x)
        vals = (
            base
            + np.einsum("ij,j->i", G, mu_g, optimize=False)
            + s1 * np.einsum("ij,j->i", G, mu_s1, optimize=False)
            + s2 * np.einsum("ij,j->i", G, mu_s2, optimize=False)
            + s3 * np.einsum("ij,j->i", G, mu_s3, optimize=False)
            + cd * np.einsum("ij,j->i", G, mu_cd, optimize=False)
        )
        return vals

    def _quad_one(self, x: np.ndarray, arm: int, *, A_inv: np.ndarray) -> float:
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

        Ac = A_inv @ c
        q0 = float(c @ Ac)
        u = P.T @ Ac
        M = P.T @ (A_inv @ P)
        quad = q0 + 2.0 * float(g @ u) + float(g @ (M @ g))
        return quad

    def predict(self, context: Iterable[float]) -> Dict[str, Any]:
        x = self._validate_x(np.asarray(list(context), dtype=float))

        j = int(self.rng.integers(0, self.heads))
        mu_j = self.A_inv[j] @ self.b[j]

        cand = self._cand.candidate_subset()
        vals = self._eval_mu_dot_phi_subset(x, arms=cand, mu=mu_j)
        best_val = float(np.min(vals))
        best_loc = np.flatnonzero(vals <= best_val + 1e-12)
        loc = int(best_loc[np.argmin(cand[best_loc])])
        arm = int(cand[loc])

        phi = self._phi(x, arm)
        pred_mean = float(mu_j @ phi)
        quad = float(self._quad_one(x, arm, A_inv=self.A_inv[j]))
        pred_unc = float(np.sqrt(max(0.0, quad)))

        self._last_phi = phi
        self._last_arm = arm
        self.history.append(
            SharedBootstrapTSStep(
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

        # Online Poisson bootstrap update (weighted rank-1).
        w = self.rng.poisson(1.0, size=self.heads)
        for j in range(self.heads):
            wj = float(w[j])
            if wj <= 0.0:
                continue
            phi_w = np.sqrt(wj) * phi

            A_inv = self.A_inv[j]
            u = A_inv @ phi_w
            denom = 1.0 + float(phi_w @ u)
            if denom <= 0.0 or not np.isfinite(denom):
                A = np.linalg.inv(A_inv)
                A = A + np.outer(phi_w, phi_w)
                self.A_inv[j] = np.linalg.inv(A)
            else:
                self.A_inv[j] = A_inv - np.outer(u, u) / denom

            self.b[j] = self.b[j] + (wj * y) * phi

        last = self.history[-1]
        self.history[-1] = SharedBootstrapTSStep(
            t=last.t,
            arm_index=last.arm_index,
            loss=y,
            pred_mean=last.pred_mean,
            pred_uncert=last.pred_uncert,
        )

        self.t += 1
        self._last_phi = None
        self._last_arm = None
