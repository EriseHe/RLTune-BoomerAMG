"""
Helper functions for creating test matrices (equivalent to MATLAB's delsq/numgrid).
"""

import numpy as np
from scipy.sparse import diags, kron, eye, lil_matrix


def numgrid(region, n):
    """
    Generate a numbering for a grid (simplified version of MATLAB's numgrid).
    
    Parameters
    ----------
    region : str
        'S' for square, 'L' for L-shaped
    n : int
        Grid size
    
    Returns
    -------
    G : ndarray
        n x n array with numbering (0 for exterior points)
    """
    if region == 'S':
        # Square region: interior points only
        G = np.zeros((n, n), dtype=int)
        count = 1
        for j in range(1, n - 1):
            for i in range(1, n - 1):
                G[i, j] = count
                count += 1
        return G
    elif region == 'L':
        # L-shaped region (upper-left quadrant removed)
        G = np.zeros((n, n), dtype=int)
        count = 1
        half = n // 2
        for j in range(1, n - 1):
            for i in range(1, n - 1):
                # Exclude upper-left quadrant
                if not (i < half and j >= half):
                    G[i, j] = count
                    count += 1
        return G
    else:
        raise ValueError(f"Unknown region: {region}")


def delsq(G):
    """
    Generate the 5-point discrete Laplacian (negative).
    
    Parameters
    ----------
    G : ndarray
        Grid numbering from numgrid
    
    Returns
    -------
    A : sparse matrix
        Sparse Laplacian matrix
    """
    m, n = G.shape
    
    # Count interior points
    num_pts = int(np.max(G))
    
    if num_pts == 0:
        return lil_matrix((0, 0))
    
    # Build sparse matrix
    A = lil_matrix((num_pts, num_pts))
    
    for i in range(m):
        for j in range(n):
            idx = G[i, j]
            if idx > 0:
                idx -= 1  # Convert to 0-indexed
                
                # Diagonal: 4
                A[idx, idx] = 4.0
                
                # Neighbors: -1 each
                # Up
                if i > 0 and G[i - 1, j] > 0:
                    A[idx, G[i - 1, j] - 1] = -1.0
                # Down
                if i < m - 1 and G[i + 1, j] > 0:
                    A[idx, G[i + 1, j] - 1] = -1.0
                # Left
                if j > 0 and G[i, j - 1] > 0:
                    A[idx, G[i, j - 1] - 1] = -1.0
                # Right
                if j < n - 1 and G[i, j + 1] > 0:
                    A[idx, G[i, j + 1] - 1] = -1.0
    
    return A.tocsr()
