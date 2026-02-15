"""
Computes the spectral radius of the Jacobi iteration matrix.
Direct translation of rho_jacobi.m
"""

import numpy as np


def rho_jacobi(A):
    """
    Computes the spectral radius of the Jacobi iteration matrix.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix (must be square)
    
    Returns
    -------
    beta : float
        Spectral radius of I - D^{-1} A where D = diag(A)
    """
    n = A.shape[0]
    
    # Convert to dense if sparse
    if hasattr(A, 'toarray'):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)
    
    # D = diag(diag(A))
    D_diag = np.diag(A_dense)
    D_inv = np.diag(1.0 / D_diag)
    
    # Jacobi iteration matrix: I - D^{-1} A
    M = np.eye(n) - D_inv @ A_dense
    
    # Spectral radius = max |eigenvalue|
    eigenvalues = np.linalg.eigvals(M)
    beta = np.max(np.abs(eigenvalues))
    
    return beta
