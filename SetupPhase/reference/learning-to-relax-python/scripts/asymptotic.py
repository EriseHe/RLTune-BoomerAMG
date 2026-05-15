"""
Plots several quantities associated with the near-asymptotic bound on the
number of SOR iterations required to converge.
Direct translation of asymptotic.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye

from solvers import sor, omega_grid, rho_jacobi, energy_norm
from utils import truncated_normal, delsq, numgrid

########################################################
def progress_bar(current, total, prefix="", every=None):
    width = 30
    if total <= 0:
        return
    if every is None:
        every = max(1, total // 100)  # ~100 updates
    if current not in (1, total) and current % every != 0:
        return
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    msg = f"\r{prefix} [{bar}] {current}/{total}"
    print(msg, end="", flush=True)
    if current == total:
        print()
########################################################

def main():
    # Setup
    A = delsq(numgrid('S', 12))
    A_dense = A.toarray()
    n = A.shape[0]
    D = np.diag(np.diag(A_dense))
    L = np.tril(A_dense, -1)
    b = truncated_normal(n)
    epsilon = 1e-8
    
    # Generate omega grid
    omegas = omega_grid(A, 1.0, 1.99, 0.01)
    actual = np.zeros(len(omegas))
    radius = np.zeros(len(omegas))
    errs = np.zeros(len(omegas))
    energy = np.zeros(len(omegas))
    
    print("Computing SOR iterations and spectral quantities...")
    for i, omega in enumerate(omegas):
        progress_bar(i + 1, len(omegas), prefix="  omegas")
        
        k, _ = sor(A, b, np.zeros(n), omega, epsilon)
        
        # C = I - A * inv(D/omega + L)
        M = D / omega + L
        M_inv = np.linalg.inv(M)
        C = np.eye(n) - A_dense @ M_inv
        
        # Spectral radius
        eigvals = np.linalg.eigvals(C)
        radius[i] = np.max(np.abs(eigvals))
        
        # Asymptocity error
        if k > 1:
            C_power = np.linalg.matrix_power(C, k - 1)
            errs[i] = np.linalg.norm(C_power, 2) ** (1.0 / (k - 1)) - radius[i]
        else:
            errs[i] = 0.0
        
        actual[i] = k
        energy[i] = energy_norm(A, omega)
    
    # Compute tau
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tau_values = errs / (1.0 - radius)
        tau_values = np.where(np.isfinite(tau_values), tau_values, 0.0)
    tau = np.max(tau_values)
    
    # Condition number estimate
    kappa = np.linalg.cond(A_dense)
    
    # Plot 1: Comparison of different cost estimates
    print("Generating plots...")
    plt.figure(1, figsize=(7, 5))
    plt.semilogy(omegas, actual, linewidth=2, label='actual cost')
    plt.semilogy(omegas, np.log(epsilon) / np.log(radius), linewidth=2, 
                 color=[0.4660, 0.6740, 0.1880], label='asymptotic estimate')
    plt.semilogy(omegas, np.log(epsilon / (2 * np.sqrt(kappa))) / np.log(energy), 
                 linewidth=2, color=[0.9290, 0.6940, 0.1250], label='energy bound')
    plt.semilogy(omegas, np.log(epsilon) / np.log(radius + tau * (1 - radius)), 
                 linewidth=2, color=[0.8500, 0.3250, 0.0980], label='near-asymptotic bound')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('ω', fontsize=16)
    plt.ylabel('iterations', fontsize=14)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/bound_comparison.png', dpi=256)
    plt.close()
    
    # Plot 2: Asymptocity
    plt.figure(2, figsize=(7, 5))
    plt.plot(omegas, errs, linewidth=2, label='||C_ω^k||_2^{1/k} - ρ(C_ω)')
    plt.plot(omegas, tau * (1 - radius), linewidth=2, label='τ (1 - ρ(C_ω))')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('ω', fontsize=16)
    plt.tight_layout()
    plt.savefig('plots/asymptocity.png', dpi=256)
    plt.close()
    
    # Compute tau and beta at different offsets
    print("Computing tau and beta at different offsets...")
    cs = np.linspace(-0.15, 0.45, 97)
    taus = np.zeros(len(cs))
    betas = np.zeros(len(cs))
    
    for i, c in enumerate(cs):
        progress_bar(i + 1, len(cs), prefix="  offsets")
        
        Ac = A + c * speye(n, format="csr")
        Ac_dense = Ac if isinstance(Ac, np.ndarray) else Ac.toarray()
        omegas_c = omega_grid(A, 1.0, 1.9, 0.01)
        Dc = np.diag(np.diag(Ac_dense))
        Lc = np.tril(Ac_dense, -1)
        
        radius_c = np.zeros(len(omegas_c))
        errs_c = np.zeros(len(omegas_c))
        
        for j, omega in enumerate(omegas_c):
            k, _ = sor(Ac, b, np.zeros(n), omega, epsilon)
            M = Dc / omega + Lc
            M_inv = np.linalg.inv(M)
            C = np.eye(n) - Ac_dense @ M_inv
            eigvals = np.linalg.eigvals(C)
            radius_c[j] = np.max(np.abs(eigvals))
            
            if k > 1:
                C_power = np.linalg.matrix_power(C, k - 1)
                errs_c[j] = np.linalg.norm(C_power, 2) ** (1.0 / (k - 1)) - radius_c[j]
            else:
                errs_c[j] = 0.0
        
        betas[i] = rho_jacobi(Ac)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_vals = errs_c / (1.0 - radius_c)
            tau_vals = np.where(np.isfinite(tau_vals), tau_vals, 0.0)
        taus[i] = np.max(tau_vals)
    
    # Plot 3: tau and beta values
    plt.figure(3, figsize=(7, 5))
    plt.plot(cs, betas, linewidth=2, label='β')
    plt.plot(cs, 4 / np.exp(2) * (1 - 1 / np.exp(2)) * np.ones(len(cs)), 
             linewidth=2, linestyle='--', label='4(1-1/e²)/e²')
    plt.plot(cs, taus, linewidth=2, label='τ')
    plt.plot(cs, np.ones(len(cs)) / np.exp(2), linewidth=2, linestyle='--', label='1/e²')
    plt.legend(loc='upper left', fontsize=12)
    plt.xlabel('c', fontsize=16)
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('plots/tau_beta.png', dpi=256)
    plt.close()
    
    print("Done! Plots saved to plots/")


if __name__ == '__main__':
    main()
