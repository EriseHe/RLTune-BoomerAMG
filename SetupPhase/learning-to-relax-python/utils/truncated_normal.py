"""
Samples a radially-truncated standard Gaussian using rejection sampling.
Direct translation of truncated_normal.m
"""

import numpy as np


def truncated_normal(n):
    """
    Samples a radially-truncated standard Gaussian.
    
    The sample is rejected if its norm exceeds sqrt(n).
    
    Parameters
    ----------
    n : int
        Dimension of the sample
    
    Returns
    -------
    sample : ndarray
        n-dimensional sample from truncated Gaussian
    """
    while True:
        sample = np.random.randn(n)
        if np.linalg.norm(sample) <= np.sqrt(n):
            return sample
