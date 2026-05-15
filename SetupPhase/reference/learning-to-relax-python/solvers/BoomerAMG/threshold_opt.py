"""
Computes a heuristic optimal strong_threshold for BoomerAMG.

This is analogous to omega_opt.py for SOR, which computes the asymptotically
optimal omega from the spectral radius of the Jacobi iteration matrix.

For BoomerAMG, there is no closed-form optimal, but heuristics exist:
- 2D problems: strong_threshold ~ 0.25
- 3D problems: strong_threshold ~ 0.5-0.6
- Anisotropic problems may need different values

The heuristic here estimates problem dimensionality from the matrix structure
(average row density as a proxy for stencil size).
"""

import numpy as np
import scipy.sparse as sp


def threshold_opt(A, problem_dim: int = None) -> float:
    """
    Computes a heuristic optimal strong_threshold for BoomerAMG.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix
    problem_dim : int, optional
        Problem dimensionality (2 or 3). If None, estimated from matrix structure.

    Returns
    -------
    opt : float
        Heuristic optimal strong_threshold value

    Notes
    -----
    The strong_threshold parameter controls which off-diagonal connections
    are considered "strong" in the coarsening phase of AMG. The optimal
    value depends on the underlying problem:

    - For 2D problems (5-point stencil): ~0.25 works well
    - For 3D problems (7-point stencil): ~0.5-0.6 works better
    - For anisotropic or stretched grids: may need tuning

    Unlike SOR's omega which has a closed-form optimal based on the spectral
    radius, strong_threshold is more empirical. This function provides a
    reasonable starting point based on matrix structure.

    The heuristic uses average row density (non-zeros per row) as a proxy:
    - ~5 nnz/row suggests 2D 5-point stencil -> 0.25
    - ~7 nnz/row suggests 3D 7-point stencil -> 0.5
    - ~9+ nnz/row suggests 3D 27-point or higher-order -> 0.6
    """
    if problem_dim is not None:
        if problem_dim == 2:
            return 0.25
        elif problem_dim == 3:
            return 0.5
        else:
            # Default for unknown dimension
            return 0.25

    # Convert to sparse if needed
    if not sp.issparse(A):
        A = sp.csr_matrix(A)
    else:
        A = sp.csr_matrix(A)

    n = A.shape[0]
    nnz = A.nnz

    # Average non-zeros per row (proxy for stencil size)
    avg_nnz_per_row = nnz / n

    # Heuristic thresholds based on typical stencil sizes:
    # - 2D 5-point stencil: ~5 nnz/row
    # - 3D 7-point stencil: ~7 nnz/row
    # - 3D 27-point stencil: ~27 nnz/row
    # - Higher-order FEM: variable

    if avg_nnz_per_row < 6:
        # Likely 2D problem or very sparse
        return 0.25
    elif avg_nnz_per_row < 10:
        # Likely 3D 7-point stencil
        return 0.5
    elif avg_nnz_per_row < 20:
        # Likely 3D with more connectivity
        return 0.6
    else:
        # Dense connectivity, be more conservative
        return 0.7


def estimate_problem_dimension(A) -> int:
    """
    Estimate the problem dimensionality from matrix structure.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix

    Returns
    -------
    dim : int
        Estimated dimensionality (2 or 3)
    """
    if not sp.issparse(A):
        A = sp.csr_matrix(A)

    n = A.shape[0]
    avg_nnz = A.nnz / n

    if avg_nnz < 6:
        return 2
    else:
        return 3
