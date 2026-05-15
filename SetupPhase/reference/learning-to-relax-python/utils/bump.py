"""
Computes a bump function at the given point(s) with the specified center and radius.
Direct translation of bump.m
"""

import numpy as np


def bump(x, center, radius):
    """
    Computes a bump function.
    
    Parameters
    ----------
    x : ndarray
        Points at which to evaluate (n x d array for n points in d dimensions)
    center : array_like
        Center of the bump function
    radius : float
        Radius of the bump function
    
    Returns
    -------
    out : ndarray
        Bump function values at each point
    """
    x = np.atleast_2d(x)
    center = np.asarray(center)
    
    # Compute squared distance from center, normalized by radius^2
    # sum((x - center).^2 / radius^2, 2) sums along dimension 2 (columns in MATLAB)
    diff = x - center
    r2 = np.sum(diff**2 / radius**2, axis=1)
    
    # Clip to avoid numerical issues: min(1, r2)
    r2_clipped = np.minimum(1.0, r2)
    
    # exp(1 - 1 / (1 - r2_clipped))
    # When r2_clipped = 1, the denominator is 0, but we clipped, so this happens at boundary
    out = np.exp(1.0 - 1.0 / (1.0 - r2_clipped))
    
    return out
