"""
Returns a grid of evenly spaced strong_threshold values plus the heuristic optimal.

This is analogous to omega_grid.py for SOR, which creates a parameter grid
for the Tsallis-INF bandit algorithm to explore.
"""

import numpy as np
from .threshold_opt import threshold_opt


def threshold_grid(A, theta_min: float, theta_max: float, step: float) -> np.ndarray:
    """
    Returns a grid of strong_threshold values including the heuristic optimal.

    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix (used to compute heuristic optimal)
    theta_min : float
        Minimum threshold value (must be > 0)
    theta_max : float
        Maximum threshold value (must be < 1)
    step : float
        Step size between grid points

    Returns
    -------
    grid : ndarray
        Array of strong_threshold values including the heuristic optimal

    Notes
    -----
    The grid is constructed similarly to omega_grid for SOR:
    - Evenly spaced values from theta_min to theta_max
    - The heuristic optimal value is inserted at its correct position

    This grid is used by the Tsallis-INF bandit algorithm to explore
    the parameter space for strong_threshold.

    Example
    -------
    >>> grid = threshold_grid(A, 0.1, 0.9, 0.1)
    >>> # Returns something like: [0.1, 0.2, 0.25, 0.3, 0.4, ...]
    >>> # where 0.25 is the heuristic optimal inserted in order
    """
    # Get heuristic optimal
    opt = threshold_opt(A)

    # Clamp optimal to valid range
    opt = max(theta_min, min(theta_max, opt))

    # Count grid points before and after optimal
    before = int(np.floor((opt - theta_min) / step)) + 1
    after = int(np.floor((theta_max - opt) / step)) + 1

    # Build grid
    grid = np.zeros(before + 1 + after)
    theta = theta_min

    # Fill values before optimal
    for i in range(before):
        grid[i] = theta
        theta = theta + step

    # Insert optimal
    grid[before] = opt

    # Fill values after optimal
    theta = opt + step
    for i in range(before + 1, before + after + 1):
        grid[i] = theta
        theta = theta + step

    # Ensure all values are in valid range and unique
    grid = np.clip(grid, theta_min, theta_max)
    grid = np.unique(grid)

    return grid


def uniform_threshold_grid(theta_min: float, theta_max: float, 
                           num_points: int) -> np.ndarray:
    """
    Returns a uniformly spaced grid of strong_threshold values.

    Parameters
    ----------
    theta_min : float
        Minimum threshold value
    theta_max : float
        Maximum threshold value
    num_points : int
        Number of grid points

    Returns
    -------
    grid : ndarray
        Array of uniformly spaced strong_threshold values

    Notes
    -----
    This is a simpler alternative to threshold_grid that doesn't
    require the matrix A. Useful when you just want a fixed grid
    for the bandit to explore.

    Example
    -------
    >>> grid = uniform_threshold_grid(0.1, 0.9, 9)
    >>> # Returns: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    """
    return np.linspace(theta_min, theta_max, num_points)
