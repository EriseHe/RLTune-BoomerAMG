"""
Compares the performance of Tsallis-INF and several fixed choices of 
strong_threshold on a sequence of linear systems using BoomerAMG.

The loss function is Work Units (WU = iterations * cum_nnz_AP), which
properly accounts for the varying cost of different AMG hierarchies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye

from learners import TsallisINF_AMG
from solvers.BoomerAMG import boomeramg
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
    # Setup - use a larger grid for BoomerAMG
    A = delsq(numgrid('S', 20))  # Larger problem for meaningful AMG hierarchy
    n = A.shape[0]
    epsilon = 1e-8
    T = 500  # Number of instances (reduced for faster testing)
    trials = 2  # Number of trials as requested
    
    # Strong threshold grid (analogous to omega for SOR)
    thresholds = np.array([0.1, 0.25, 0.4, 0.5, 0.7])
    
    # Grid for Tsallis-INF
    threshold_grid = np.linspace(0.1, 0.9, 9)
    
    print(f"Problem size: {n}x{n}, nnz(A)={A.nnz}")
    print(f"Running {T} instances over {trials} trials")
    print(f"Fixed thresholds: {thresholds}")
    print(f"Bandit grid: {threshold_grid}")
    print()
    
    threshold_costs = np.zeros((T, trials, len(thresholds)))
    tinf_costs = np.zeros((T, trials))
    
    # High-variance offset distribution
    print("=" * 60)
    print("Running high-variance offset distribution (BoomerAMG)...")
    print("=" * 60)
    
    for trial in range(trials):
        print(f"\n  Trial {trial + 1}/{trials}")
        tinf = TsallisINF_AMG(threshold_grid, T)
        
        for t in range(T):
            # Random diagonal shift (same distribution as SOR experiments)
            c = -0.15 + 0.6 * np.random.beta(0.5, 1.5)
            At = A + c * speye(n, format="csr")
            bt = truncated_normal(n)
            
            # Tsallis-INF prediction and solve
            theta_pred = tinf.predict()
            k, cum_nnz_AP, _ = boomeramg(At, bt, np.zeros(n), theta_pred, epsilon)
            WU = k * cum_nnz_AP
            tinf_costs[t, trial] = WU
            tinf.update(k, cum_nnz_AP)
            
            # Fixed threshold solves
            for i, theta in enumerate(thresholds):
                k, cum_nnz_AP, _ = boomeramg(At, bt, np.zeros(n), theta, epsilon)
                threshold_costs[t, trial, i] = k * cum_nnz_AP
            
            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")
    
    # Plot high-variance results
    plt.figure(1, figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(thresholds)))
    
    for i in range(len(thresholds)):
        label = f'θ={thresholds[i]:.2f}'
        if thresholds[i] == 0.25:
            label += ' (2D default)'
        plt.plot(np.mean(np.cumsum(threshold_costs[:, :, i], axis=0), axis=1), 
                 T - np.arange(1, T + 1), linewidth=2, linestyle='--', 
                 color=colors[i])
    
    plt.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=3, color='black')
    
    legend_labels = [f'θ={t:.2f}' + (' (2D default)' if t == 0.25 else '') 
                     for t in thresholds]
    legend_labels.append('Tsallis-INF (BoomerAMG)')
    plt.legend(legend_labels, fontsize=11, loc='upper right')
    
    plt.xlabel('Total Work Units (WU)', fontsize=14)
    plt.ylabel('Instances Remaining', fontsize=14)
    plt.title('BoomerAMG: High-Variance Offset Distribution', fontsize=14)
    plt.tight_layout()
    
    os.makedirs('plots/BoomerAMG', exist_ok=True)
    plt.savefig('plots/BoomerAMG/learning_high_variance.png', dpi=256)
    plt.close()
    print("\n  Saved: plots/BoomerAMG/learning_high_variance.png")
    
    # Low-variance offset distribution
    print("\n" + "=" * 60)
    print("Running low-variance offset distribution (BoomerAMG)...")
    print("=" * 60)
    
    for trial in range(trials):
        print(f"\n  Trial {trial + 1}/{trials}")
        tinf = TsallisINF_AMG(threshold_grid, T)
        
        for t in range(T):
            # Random diagonal shift (lower variance)
            c = -0.15 + 0.6 * np.random.beta(2.0, 6.0)
            At = A + c * speye(n, format="csr")
            bt = truncated_normal(n)
            
            # Tsallis-INF prediction and solve
            theta_pred = tinf.predict()
            k, cum_nnz_AP, _ = boomeramg(At, bt, np.zeros(n), theta_pred, epsilon)
            WU = k * cum_nnz_AP
            tinf_costs[t, trial] = WU
            tinf.update(k, cum_nnz_AP)
            
            # Fixed threshold solves
            for i, theta in enumerate(thresholds):
                k, cum_nnz_AP, _ = boomeramg(At, bt, np.zeros(n), theta, epsilon)
                threshold_costs[t, trial, i] = k * cum_nnz_AP
            
            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")
    
    # Plot low-variance results
    plt.figure(2, figsize=(8, 6))
    
    for i in range(len(thresholds)):
        label = f'θ={thresholds[i]:.2f}'
        if thresholds[i] == 0.25:
            label += ' (2D default)'
        plt.plot(np.mean(np.cumsum(threshold_costs[:, :, i], axis=0), axis=1), 
                 T - np.arange(1, T + 1), linewidth=2, linestyle='--',
                 color=colors[i])
    
    plt.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=3, color='black')
    
    plt.legend(legend_labels, fontsize=11, loc='upper right')
    
    plt.xlabel('Total Work Units (WU)', fontsize=14)
    plt.ylabel('Instances Remaining', fontsize=14)
    plt.title('BoomerAMG: Low-Variance Offset Distribution', fontsize=14)
    plt.tight_layout()
    
    plt.savefig('plots/BoomerAMG/learning_low_variance.png', dpi=256)
    plt.close()
    print("\n  Saved: plots/BoomerAMG/learning_low_variance.png")
    
    print("\n" + "=" * 60)
    print("Done! Plots saved to plots/BoomerAMG/")
    print("=" * 60)


if __name__ == '__main__':
    main()
