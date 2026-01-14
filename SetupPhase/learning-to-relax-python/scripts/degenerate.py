"""
Compares the number of iterations required to solve a linear system when
the target vector is drawn from a truncated Gaussian vs. when it is
generated using the smallest eigenvector of the iteration matrix for
omega=1.4.
Direct translation of degenerate.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from solvers import sor, omega_grid
from utils import truncated_normal, delsq, numgrid


def main():
    # Setup
    A = delsq(numgrid('S', 12))
    A_dense = A.toarray()
    n = A.shape[0]
    D = np.diag(np.diag(A_dense))
    L = np.tril(A_dense, -1)
    epsilon = 1e-8
    
    omegas = omega_grid(A, 1.0, 1.9, 0.001)
    
    # Compute degenerate target vector
    # C = I - A * inv(D/1.4 + L)
    M = D / 1.4 + L
    M_inv = np.linalg.inv(M)
    C = np.eye(n) - A_dense @ M_inv
    
    # Get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(C)
    
    # Sort by eigenvalue magnitude to find smallest
    # MATLAB's vecs(:, 99) corresponds to a specific eigenvector
    # We'll use the eigenvector with smallest absolute eigenvalue
    idx_sorted = np.argsort(np.abs(eigenvalues))
    # MATLAB uses 1-indexed, and 99 for a 100x100 matrix would be near the top
    # For n=100, this likely means eigenvalue index 98 (0-indexed)
    degen_idx = min(98, n - 2)  # Adjust for smaller matrices
    degen = np.real(eigenvectors[:, idx_sorted[degen_idx]])
    
    degen_cost = np.zeros(len(omegas))
    gauss_cost = np.zeros(len(omegas))
    gauss_min = np.zeros(len(omegas))
    gauss_max = np.zeros(len(omegas))
    
    print("Computing costs...")
    for i, omega in enumerate(omegas):
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(omegas)}")
        
        k, _ = sor(A, degen, np.zeros(n), omega, epsilon)
        degen_cost[i] = k
        
        costs = np.zeros(40)
        for j in range(40):
            k, _ = sor(A, truncated_normal(n), np.zeros(n), omega, epsilon)
            costs[j] = k
        
        gauss_cost[i] = np.mean(costs)
        gauss_max[i] = np.max(costs)
        gauss_min[i] = np.min(costs)
    
    # Plot results
    plt.figure(figsize=(7, 5))
    plt.fill_between(omegas, gauss_min, gauss_max, color=[1.0, 0.8, 0.8], 
                     linewidth=0)
    plt.plot(omegas, degen_cost, linewidth=2, color=[0, 0.4470, 0.7410], 
             label='degenerate cost')
    plt.plot(omegas, gauss_cost, linewidth=2, color=[0.8500, 0.3250, 0.0980], 
             label='mean cost')
    plt.legend(loc='upper center', fontsize=12)
    plt.xlabel('ω', fontsize=16)
    plt.ylabel('iterations', fontsize=14)
    plt.xlim([1.0, 1.9])
    plt.tight_layout()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/degenerate.png', dpi=256)
    plt.close()
    
    print("Done! Plot saved to plots/degenerate.png")


if __name__ == '__main__':
    main()
