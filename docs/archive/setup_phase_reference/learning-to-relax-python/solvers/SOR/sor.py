"""
Computes the number of iterations required for SOR to solve a linear system 
to a given tolerance, returning the approximate solution obtained as a second 
output; caps the number of iterations at 10000.
Direct translation of sor.m
"""

import numpy as np
from scipy.linalg import solve_triangular


def sor(A, b, x, omega, tol):
    """
    Computes the number of iterations required for SOR to solve Ax = b.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix
    b : ndarray
        Right-hand side vector
    x : ndarray
        Initial guess
    omega : float
        Relaxation parameter (should be in (0, 2))
    tol : float
        Relative tolerance for convergence
    
    Returns
    -------
    k : int
        Number of iterations required
    out : ndarray
        Approximate solution
    """
    # Convert to dense if sparse
    if hasattr(A, 'toarray'):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)
    
    n = A_dense.shape[0]
    x = x.copy().astype(float)
    
    # D = diag(diag(A))
    D = np.diag(np.diag(A_dense))
    
    # M = D/omega + tril(A, -1) (strictly lower triangular part of A)
    L = np.tril(A_dense, -1)
    M = D / omega + L
    
    # Initial residual
    r = b - A_dense @ x
    norm0 = np.linalg.norm(r)
    
    for k in range(1, 10001):
        # x = x + M \ r (M is lower triangular, so use forward substitution)
        x = x + solve_triangular(M, r, lower=True)
        r = b - A_dense @ x
        if np.linalg.norm(r) / norm0 < tol:
            return k, x
    
    return 10000, x
