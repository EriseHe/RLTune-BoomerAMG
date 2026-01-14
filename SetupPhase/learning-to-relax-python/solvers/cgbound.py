"""
Computes a bound on the number of iterations required for SSOR-preconditioned 
CG to solve a linear system to a specified tolerance; the bound is derived 
from the condition number analysis of Axelsson (1994, Theorem 7.17).
Direct translation of cgbound.m
"""

import numpy as np
from scipy.sparse.linalg import eigs as sparse_eigs


def cgbound(A, omega, epsilon):
    """
    Computes a bound on SSOR-preconditioned CG iterations.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix
    omega : float or ndarray
        Relaxation parameter(s)
    epsilon : float
        Tolerance
    
    Returns
    -------
    bound : float or ndarray
        Upper bound on iteration count
    """
    # Convert to dense if sparse
    if hasattr(A, 'toarray'):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)
    
    D = np.diag(np.diag(A_dense))
    L = np.tril(A_dense, -1)
    invA = np.linalg.inv(A_dense)
    
    # kappa = eigs(A, 1) * eigs(invA, 1)
    # Note: eigs returns largest magnitude eigenvalue
    eig_A = np.max(np.abs(np.linalg.eigvals(A_dense)))
    eig_invA = np.max(np.abs(np.linalg.eigvals(invA)))
    kappa = eig_A * eig_invA
    
    # alpha = real(eigs(D*invA, 1))
    alpha = np.real(np.max(np.linalg.eigvals(D @ invA)))
    
    # beta = max(real(eig(full((L*inv(D)*L'-.25*D)*invA))))
    invD = np.linalg.inv(D)
    inner = (L @ invD @ L.T - 0.25 * D) @ invA
    beta = np.max(np.real(np.linalg.eigvals(inner)))
    
    # Handle scalar or array omega
    omega = np.atleast_1d(omega)
    tmo = 2.0 - omega  # two minus omega
    tmooo = tmo / omega  # (2-omega)/omega
    ic = tmo / (1.0 + 0.25 * tmo * tmooo * alpha + beta * omega)
    
    # bound = 1 + log(...) / -log(1 - 2/(1+sqrt(1/ic)))
    numerator = np.log(np.sqrt(kappa) / epsilon + np.sqrt(kappa / epsilon**2 - 1))
    denominator = -np.log(1.0 - 2.0 / (1.0 + np.sqrt(1.0 / ic)))
    bound = 1.0 + numerator / denominator
    
    # Return scalar if input was scalar
    if bound.size == 1:
        return float(bound)
    return bound
