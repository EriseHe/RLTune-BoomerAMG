from __future__ import annotations

from typing import Sequence

import numpy as np


def build_action_basis(
    weights: Sequence[float],
    *,
    mode: str,
    rbf_sigma: float,
    rbf_centers: Sequence[float] = (),
) -> np.ndarray:
    """Build a fixed action basis shared by linear value estimators."""

    action_values = np.asarray(tuple(float(weight) for weight in weights), dtype=float)
    if action_values.ndim != 1 or action_values.size == 0:
        raise ValueError("weights must contain at least one finite action")
    if not np.all(np.isfinite(action_values)):
        raise ValueError("weights must be finite")
    basis_mode = str(mode).strip().lower()
    sigma = float(rbf_sigma)
    if sigma < 0.0:
        raise ValueError("action_rbf_sigma must be non-negative")

    if basis_mode == "legacy":
        if rbf_centers:
            raise ValueError("legacy action basis does not accept explicit centers")
        if sigma <= 0.0:
            return np.eye(action_values.size, dtype=float)
        centers = action_values
    elif basis_mode == "compact_rbf":
        centers = np.asarray(tuple(float(value) for value in rbf_centers), dtype=float)
        if sigma <= 0.0:
            raise ValueError("compact_rbf action basis requires a positive RBF width")
        if centers.ndim != 1 or centers.size == 0 or not np.all(np.isfinite(centers)):
            raise ValueError("compact_rbf action basis requires finite centers")
    else:
        raise ValueError("action_basis_mode must be 'legacy' or 'compact_rbf'")

    distances = action_values[:, None] - centers[None, :]
    basis = np.exp(-0.5 * np.square(distances / sigma))
    basis[np.abs(distances) > 2.5 * sigma] = 0.0
    row_sums = np.sum(basis, axis=1, keepdims=True)
    if np.any(row_sums <= 1.0e-15):
        raise ValueError("action RBF centers do not cover every action")
    return basis / row_sums


def joint_action_features(
    action_basis: np.ndarray,
    features: np.ndarray,
    *,
    action_indices: int | Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    """Return flattened joint features for all or selected discrete actions."""

    basis = np.asarray(action_basis, dtype=float)
    state = np.asarray(features, dtype=float)
    if basis.ndim != 2 or state.ndim != 1:
        raise ValueError("action basis must be 2D and state features must be 1D")
    if action_indices is not None:
        indices = np.asarray(action_indices, dtype=int)
        if indices.ndim == 0:
            indices = indices.reshape(1)
        if indices.ndim != 1:
            raise ValueError("action_indices must be a scalar or one-dimensional")
        if np.any(indices < 0) or np.any(indices >= basis.shape[0]):
            raise IndexError("action index is outside the action basis")
        basis = basis[indices]
        return (basis[:, :, None] * state[None, None, :]).reshape(
            basis.shape[0],
            basis.shape[1] * state.size,
        )
    return np.einsum("ab,f->abf", basis, state, optimize=True).reshape(
        basis.shape[0],
        basis.shape[1] * state.size,
    )


__all__ = ["build_action_basis", "joint_action_features"]
