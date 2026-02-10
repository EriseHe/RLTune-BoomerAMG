"""
Computes a bound on the energy norm of the SOR iteration matrix.
(Hackbusch 2016, Corollary 3.45)
Direct translation of energy_norm.m
"""

import numpy as np


def energy_norm(A, omega):
    """
    Computes a bound on the energy norm of the SOR iteration matrix.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix
    omega : float
        Relaxation parameter
    
    Returns
    -------
    bound : float
        Upper bound on the energy norm
    """
    # Convert to dense if sparse
    if hasattr(A, 'toarray'):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)
    
    Omega = (2.0 - omega) / (2.0 * omega)
    
    # invD = inv(diag(diag(A)))
    D_diag = np.diag(A_dense)
    invD = np.diag(1.0 / D_diag)
    
    # L = tril(A, -1)
    L = np.tril(A_dense, -1)
    
    # gamma = 1 - max(abs(eig(invD*(L+L'))))
    gamma = 1.0 - np.max(np.abs(np.linalg.eigvals(invD @ (L + L.T))))
    
    # bound = sqrt(1 - 2*Omega*gamma / (Omega^2 + gamma/omega + max(abs(eig(invD*L*invD*L'))) - 0.25))
    inner_matrix = invD @ L @ invD @ L.T
    max_eig = np.max(np.abs(np.linalg.eigvals(inner_matrix)))
    
    denominator = Omega**2 + gamma / omega + max_eig - 0.25
    bound = np.sqrt(1.0 - 2.0 * Omega * gamma / denominator)
    
    return bound
