"""
Returns a grid of evenly spaced omegas plus the optimal omega for SOR.
Direct translation of omega_grid.m
"""

import numpy as np
from .omega_opt import omega_opt


def omega_grid(A, omega_min, omega_max, step):
    """
    Returns a grid of evenly spaced omegas plus the optimal omega for SOR.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix
    omega_min : float
        Minimum omega value
    omega_max : float
        Maximum omega value
    step : float
        Step size between grid points
    
    Returns
    -------
    grid : ndarray
        Array of omega values including the optimal omega
    """
    opt = omega_opt(A)
    before = int(np.floor((opt - omega_min) / step)) + 1
    after = int(np.floor((omega_max - opt) / step)) + 1
    
    grid = np.zeros(before + 1 + after)
    omega = omega_min
    
    # Fill values before optimal
    for i in range(before):
        grid[i] = omega
        omega = omega + step
    
    # Insert optimal omega
    grid[before] = opt
    
    # Fill values after optimal
    for i in range(before + 1, before + after + 1):
        grid[i] = omega
        omega = omega + step
    
    return grid
