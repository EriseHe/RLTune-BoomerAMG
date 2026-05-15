"""
Computes and plots the number of iterations required for SOR to converge,
as well as the near-asymptotic bound for comparison, averaged across forty
different randomly sampled linear systems, where the randomness determines 
a scalar offset of the diagonal.
Direct translation of comparators.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye

from solvers import sor, omega_grid, rho_jacobi
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
    epsilon = 1e-8
    trials = 40
    omegas = omega_grid(A, 1.0, 1.99, 0.01)
    
    os.makedirs('plots', exist_ok=True)
    
    # Low-variance offset distribution
    print("Running low-variance offset distribution...")
    cs = -0.15 + 0.6 * np.random.beta(2.0, 6.0, trials)
    taus = np.zeros(trials)
    betas = np.zeros(trials)
    actual = np.zeros(len(omegas))
    dynamic_actual = 0.0
    predicted = np.zeros(len(omegas))
    dynamic_predicted = 0.0
    
    for i, c in enumerate(cs):
        print(f"  Trial {i + 1}/{trials}")
        Ac = A + c * speye(n, format="csr")
        Ac_dense = Ac if isinstance(Ac, np.ndarray) else Ac.toarray()
        b = truncated_normal(n)
        Dc = np.diag(np.diag(Ac_dense))
        Lc = np.tril(Ac_dense, -1)
        
        radius = np.zeros(len(omegas))
        errs = np.zeros(len(omegas))
        current_best = np.inf
        
        for j, omega in enumerate(omegas):
            k, _ = sor(Ac, b, np.zeros(n), omega, epsilon)
            
            M = Dc / omega + Lc
            M_inv = np.linalg.inv(M)
            C = np.eye(n) - Ac_dense @ M_inv
            eigvals = np.linalg.eigvals(C)
            radius[j] = np.max(np.abs(eigvals))
            
            if k > 1:
                C_power = np.linalg.matrix_power(C, k - 1)
                errs[j] = np.linalg.norm(C_power, 2) ** (1.0 / (k - 1)) - radius[j]
            else:
                errs[j] = 0.0
            
            actual[j] += k
            if k < current_best:
                current_best = k
            progress_bar(j + 1, len(omegas), prefix=f"    trial {i + 1}/{trials}")
        
        dynamic_actual += current_best
        
        D_inv = np.diag(1.0 / np.diag(Dc))
        betas[i] = np.max(np.abs(np.linalg.eigvals(np.eye(n) - D_inv @ Ac_dense)))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_vals = errs / (1.0 - radius)
            tau_vals = np.where(np.isfinite(tau_vals), tau_vals, 0.0)
        taus[i] = np.max(tau_vals)
        
        current_predicted = 1.0 + np.log(epsilon) / np.log(radius + taus[i] * (1.0 - radius))
        dynamic_predicted += np.min(current_predicted)
        predicted += current_predicted
    
    # Plot low-variance results
    plt.figure(1, figsize=(7, 5))
    plt.semilogy(omegas, 35 * np.ones(len(omegas)), color='white')
    plt.semilogy(omegas, actual / len(cs), linewidth=2, color=[0, 0.4470, 0.7410], label='actual cost')
    plt.semilogy(omegas, dynamic_actual * np.ones(len(omegas)) / trials, linewidth=2, 
                 linestyle='--', color=[0, 0.4470, 0.7410], label='(instance-optimal)')
    plt.semilogy(omegas, 35 * np.ones(len(omegas)), color='white')
    plt.semilogy(omegas, predicted / len(cs), linewidth=2, color=[0.8500, 0.3250, 0.0980], 
                 label='near-asymptotic bound')
    plt.semilogy(omegas, dynamic_predicted * np.ones(len(omegas)) / trials, linewidth=2, 
                 linestyle='--', color=[0.8500, 0.3250, 0.0980], label='(instance-optimal)')
    plt.legend(loc='upper center', fontsize=10)
    plt.xlabel('ω', fontsize=16)
    plt.ylabel('iterations', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/low_variance.png', dpi=256)
    plt.close()
    
    # High-variance offset distribution
    print("Running high-variance offset distribution...")
    cs = -0.15 + 0.6 * np.random.beta(0.5, 1.5, trials)
    taus = np.zeros(trials)
    betas = np.zeros(trials)
    actual = np.zeros(len(omegas))
    dynamic_actual = 0.0
    predicted = np.zeros(len(omegas))
    dynamic_predicted = 0.0
    
    for i, c in enumerate(cs):
        print(f"  Trial {i + 1}/{trials}")
        Ac = A + c * speye(n, format="csr")
        Ac_dense = Ac if isinstance(Ac, np.ndarray) else Ac.toarray()
        b = truncated_normal(n)
        Dc = np.diag(np.diag(Ac_dense))
        Lc = np.tril(Ac_dense, -1)
        
        radius = np.zeros(len(omegas))
        errs = np.zeros(len(omegas))
        current_best = np.inf
        
        for j, omega in enumerate(omegas):
            k, _ = sor(Ac, b, np.zeros(n), omega, epsilon)
            
            M = Dc / omega + Lc
            M_inv = np.linalg.inv(M)
            C = np.eye(n) - Ac_dense @ M_inv
            eigvals = np.linalg.eigvals(C)
            radius[j] = np.max(np.abs(eigvals))
            
            if k > 1:
                C_power = np.linalg.matrix_power(C, k - 1)
                errs[j] = np.linalg.norm(C_power, 2) ** (1.0 / (k - 1)) - radius[j]
            else:
                errs[j] = 0.0
            
            actual[j] += k
            if k < current_best:
                current_best = k
            progress_bar(j + 1, len(omegas), prefix=f"    trial {i + 1}/{trials}")
        
        dynamic_actual += current_best
        
        D_inv = np.diag(1.0 / np.diag(Dc))
        betas[i] = np.max(np.abs(np.linalg.eigvals(np.eye(n) - D_inv @ Ac_dense)))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            tau_vals = errs / (1.0 - radius)
            tau_vals = np.where(np.isfinite(tau_vals), tau_vals, 0.0)
        taus[i] = np.max(tau_vals)
        
        current_predicted = 1.0 + np.log(epsilon) / np.log(radius + taus[i] * (1.0 - radius))
        dynamic_predicted += np.min(current_predicted)
        predicted += current_predicted
    
    # Plot high-variance results
    plt.figure(2, figsize=(7, 5))
    plt.semilogy(omegas, 35 * np.ones(len(omegas)), color='white')
    plt.semilogy(omegas, actual / len(cs), linewidth=2, color=[0, 0.4470, 0.7410], label='actual cost')
    plt.semilogy(omegas, dynamic_actual * np.ones(len(omegas)) / trials, linewidth=2, 
                 linestyle='--', color=[0, 0.4470, 0.7410], label='(instance-optimal)')
    plt.semilogy(omegas, 35 * np.ones(len(omegas)), color='white')
    plt.semilogy(omegas, predicted / len(cs), linewidth=2, color=[0.8500, 0.3250, 0.0980], 
                 label='near-asymptotic bound')
    plt.semilogy(omegas, dynamic_predicted * np.ones(len(omegas)) / trials, linewidth=2, 
                 linestyle='--', color=[0.8500, 0.3250, 0.0980], label='(instance-optimal)')
    plt.legend(loc='upper center', fontsize=10)
    plt.xlabel('ω', fontsize=16)
    plt.ylabel('iterations', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/high_variance.png', dpi=256)
    plt.close()
    
    print("Done! Plots saved to plots/")


if __name__ == '__main__':
    main()
