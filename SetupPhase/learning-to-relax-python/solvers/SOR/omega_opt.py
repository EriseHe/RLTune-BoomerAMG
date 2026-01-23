"""
Computes the asymptotically optimal omega for SOR.
Direct translation of omega_opt.m
"""

from .rho_jacobi import rho_jacobi
import numpy as np


def omega_opt(A):
    """
    Computes the asymptotically optimal omega for SOR.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix
    
    Returns
    -------
    opt : float
        Optimal relaxation parameter omega
    """
    beta = rho_jacobi(A)
    opt = 1.0 + (beta / (1.0 + np.sqrt(1.0 - beta**2)))**2
    return opt
