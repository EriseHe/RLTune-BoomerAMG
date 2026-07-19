"""
Compares the performance of Tsallis-INF and several fixed choices of 
omega on a sequence of 5000 i.i.d. linear systems; all results are 
averages over 40 trials.
Direct translation of learning.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye

from learners import TsallisINF
from solvers import sor
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
    n = A.shape[0]
    epsilon = 1e-8
    T = 5000
    trials = 5
    omegas = np.linspace(1.0, 1.8, 5)
    
    omega_costs = np.zeros((T, trials, len(omegas)))
    tinf_costs = np.zeros((T, trials))
    
    # High-variance offset distribution
    print("Running high-variance offset distribution...")
    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}")
        tinf = TsallisINF(np.linspace(1.0, 1.95, 20), T)
        
        for t in range(T):
            c = -0.15 + 0.6 * np.random.beta(0.5, 1.5)
            At = A + c * speye(n, format="csr")
            bt = truncated_normal(n)
            
            k, _ = sor(At, bt, np.zeros(n), tinf.predict(), epsilon)
            tinf_costs[t, trial] = k
            tinf.update(k)
            
            for i, omega in enumerate(omegas):
                k, _ = sor(At, bt, np.zeros(n), omega, epsilon)
                omega_costs[t, trial, i] = k
            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")
    
    # Plot high-variance results
    plt.figure(1, figsize=(7, 5))
    for i in range(len(omegas)):
        plt.plot(np.mean(np.cumsum(omega_costs[:, :, i], axis=0), axis=1), 
                 T - np.arange(1, T + 1), linewidth=2, linestyle='--')
    plt.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2, color='black')
    plt.legend([f'ω={omegas[0]:.1f}', f'ω={omegas[1]:.1f}', f'ω={omegas[2]:.1f}', 
                f'ω={omegas[3]:.1f}', f'ω={omegas[4]:.1f} (≈ω*)', 'Tsallis-INF'], 
               fontsize=12)
    plt.xlabel('total iterations', fontsize=14)
    plt.ylabel('instances remaining', fontsize=14)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/learning_high_variance.png', dpi=256)
    plt.close()
    
    # Low-variance offset distribution
    print("Running low-variance offset distribution...")
    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}")
        tinf = TsallisINF(np.linspace(1.0, 1.95, 20), T)
        
        for t in range(T):
            c = -0.15 + 0.6 * np.random.beta(2.0, 6.0)
            At = A + c * speye(n, format="csr")
            bt = truncated_normal(n)
            
            k, _ = sor(At, bt, np.zeros(n), tinf.predict(), epsilon)
            tinf_costs[t, trial] = k
            tinf.update(k)
            
            for i, omega in enumerate(omegas):
                k, _ = sor(At, bt, np.zeros(n), omega, epsilon)
                omega_costs[t, trial, i] = k
            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")
    
    # Plot low-variance results
    plt.figure(2, figsize=(7, 5))
    for i in range(len(omegas)):
        plt.plot(np.mean(np.cumsum(omega_costs[:, :, i], axis=0), axis=1), 
                 T - np.arange(1, T + 1), linewidth=2, linestyle='--')
    plt.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2, color='black')
    plt.legend([f'ω={omegas[0]:.1f}', f'ω={omegas[1]:.1f}', f'ω={omegas[2]:.1f}', 
                f'ω={omegas[3]:.1f} (≈ω*)', f'ω={omegas[4]:.1f}', 'Tsallis-INF'], 
               fontsize=12)
    plt.xlabel('total iterations', fontsize=14)
    plt.ylabel('instances remaining', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/learning_low_variance.png', dpi=256)
    plt.close()
    
    print("Done! Plots saved to plots/")


if __name__ == '__main__':
    main()
