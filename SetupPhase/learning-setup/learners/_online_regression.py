from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def robust_cholesky(matrix: np.ndarray) -> np.ndarray:
    """
    Compute a Cholesky-like factor for a symmetric PSD-ish matrix with jitter.
    """
    A = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
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


@dataclass
class OnlineRidgeRegressor:
    dim: int
    l2_reg: float

    def __post_init__(self) -> None:
        if int(self.dim) <= 0:
            raise ValueError("dim must be positive")
        if float(self.l2_reg) <= 0.0:
            raise ValueError("l2_reg must be > 0")
        self.dim = int(self.dim)
        self.l2_reg = float(self.l2_reg)
        self.A_inv = np.eye(self.dim, dtype=float) / self.l2_reg
        self.b = np.zeros(self.dim, dtype=float)

    @property
    def mean(self) -> np.ndarray:
        return self.A_inv @ self.b

    def predict(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            return np.asarray(float(self.mean @ X), dtype=float)
        return X @ self.mean

    def sample_theta(self, *, scale: float, rng: np.random.Generator) -> np.ndarray:
        L = robust_cholesky(self.A_inv)
        z = rng.standard_normal(self.dim)
        return self.mean + float(scale) * (L @ z)

    def uncertainty(self, features: np.ndarray) -> np.ndarray:
        X = np.asarray(features, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        AX = X @ self.A_inv
        quad = np.einsum("ij,ij->i", AX, X, optimize=False)
        return np.sqrt(np.maximum(0.0, quad))

    def observe(self, feature: np.ndarray, target: float) -> None:
        phi = np.asarray(feature, dtype=float).reshape(-1)
        if phi.size != self.dim:
            raise ValueError(f"feature dim {phi.size} does not match regressor dim {self.dim}")
        y = float(target)
        if not np.isfinite(y):
            raise ValueError("target must be finite")

        u = self.A_inv @ phi
        denom = 1.0 + float(phi @ u)
        if denom <= 0.0 or not np.isfinite(denom):
            A = np.linalg.inv(self.A_inv)
            A = A + np.outer(phi, phi)
            self.A_inv = np.linalg.inv(A)
        else:
            self.A_inv = self.A_inv - np.outer(u, u) / denom
        self.b = self.b + y * phi


def squarecb_gamma(*, round_index: int, gamma_scale: float, gamma_exponent: float) -> float:
    t = max(0, int(round_index))
    gamma = float(gamma_scale) * float((t + 1) ** float(gamma_exponent))
    return max(gamma, 1e-9)


def squarecb_probabilities(pred_losses: np.ndarray, *, gamma: float) -> np.ndarray:
    """
    SquareCB sampling rule with the standard mu = K choice:
      p(a) = 1 / (K + gamma * (loss_hat[a] - loss_hat[best])) for non-best arms
      p(best) = 1 - sum_{a != best} p(a)
    """
    losses = np.asarray(pred_losses, dtype=float).reshape(-1)
    if losses.size == 0:
        raise ValueError("pred_losses must be non-empty")
    if not np.all(np.isfinite(losses)):
        raise ValueError("pred_losses must be finite")

    K = int(losses.size)
    best = int(np.argmin(losses))
    best_loss = float(losses[best])
    probs = np.zeros(K, dtype=float)
    if K == 1:
        probs[0] = 1.0
        return probs

    mu = float(K)
    total_other = 0.0
    for idx, loss_hat in enumerate(losses):
        if idx == best:
            continue
        gap = max(0.0, float(loss_hat) - best_loss)
        p = 1.0 / max(mu + float(gamma) * gap, 1e-12)
        probs[idx] = float(p)
        total_other += float(p)

    probs[best] = max(0.0, 1.0 - total_other)
    mass = float(np.sum(probs))
    if not np.isfinite(mass) or mass <= 0.0:
        return np.full(K, 1.0 / K, dtype=float)
    return probs / mass


def sample_discrete(probs: np.ndarray, *, rng: np.random.Generator) -> int:
    p = np.asarray(probs, dtype=float).reshape(-1)
    if p.size == 0:
        raise ValueError("probs must be non-empty")
    mass = float(np.sum(p))
    if not np.isfinite(mass) or mass <= 0.0:
        p = np.full(p.size, 1.0 / p.size, dtype=float)
    else:
        p = p / mass
    return int(rng.choice(np.arange(p.size, dtype=int), p=p))
