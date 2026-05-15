"""
Compares the performance of different learning algorithms---including 
contextual bandit algorithms using diagonal offsets as context---on a 
sequence of 5000 i.i.d. linear systems; all results are averages over 40 
trials.
Direct translation of contextual.m
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye

from learners import TsallisINF, TsallisINFCB, ChebCB
from solvers import sor, omega_opt
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
    trials = 40
    
    tinf_costs = np.zeros((T, trials))
    cheb_costs = np.zeros((T, trials))
    tinfcb_costs = np.zeros((T, trials))
    opt_costs = np.zeros((T, trials))
    omega_costs = np.zeros((T, trials))
    
    # High-variance offset distribution
    print("Running high-variance offset distribution...")
    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}")
        tinf = TsallisINF(np.linspace(1.0, 1.95, 20), T)
        cheb = ChebCB(np.linspace(1.0, 1.95, 20), T, 6, -0.15, 0.65)
        tinfcb = TsallisINFCB(np.linspace(1.0, 1.95, 20), 5, -0.15, 0.65)
        
        for t in range(T):
            c = -0.15 + 0.6 * np.random.beta(0.5, 1.5)
            At = A + c * speye(n, format="csr")
            bt = truncated_normal(n)
            
            k, _ = sor(At, bt, np.zeros(n), tinf.predict(), epsilon)
            tinf_costs[t, trial] = k
            tinf.update(k)
            
            k, _ = sor(At, bt, np.zeros(n), cheb.predict(c), epsilon)
            cheb_costs[t, trial] = k
            cheb.update(k)
            
            k, _ = sor(At, bt, np.zeros(n), tinfcb.predict(c), epsilon)
            tinfcb_costs[t, trial] = k
            tinfcb.update(k)
            
            k, _ = sor(At, bt, np.zeros(n), omega_opt(At), epsilon)
            opt_costs[t, trial] = k
            
            k, _ = sor(At, bt, np.zeros(n), 1.8, epsilon)
            omega_costs[t, trial] = k
            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")
    
    # Plot high-variance results
    plt.figure(1, figsize=(7, 5))
    plt.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(omega_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(tinfcb_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(cheb_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(opt_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.legend(['Tsallis-INF', 'ω=1.8 (≈ω*)', 'Tsallis-INF-CB', 'ChebCB', 'Instance-Optimal'], 
               fontsize=12)
    plt.xlabel('total iterations', fontsize=14)
    plt.ylabel('instances remaining', fontsize=14)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/contextual_high_variance.png', dpi=256)
    plt.close()
    
    # Low-variance offset distribution
    print("Running low-variance offset distribution...")
    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}")
        tinf = TsallisINF(np.linspace(1.0, 1.95, 20), T)
        cheb = ChebCB(np.linspace(1.0, 1.95, 20), T, 6, -0.15, 0.65)
        tinfcb = TsallisINFCB(np.linspace(1.0, 1.95, 20), 5, -0.15, 0.65)
        
        for t in range(T):
            c = -0.15 + 0.6 * np.random.beta(2.0, 6.0)
            At = A + c * speye(n, format="csr")
            bt = truncated_normal(n)
            
            k, _ = sor(At, bt, np.zeros(n), tinf.predict(), epsilon)
            tinf_costs[t, trial] = k
            tinf.update(k)
            
            k, _ = sor(At, bt, np.zeros(n), cheb.predict(c), epsilon)
            cheb_costs[t, trial] = k
            cheb.update(k)
            
            k, _ = sor(At, bt, np.zeros(n), tinfcb.predict(c), epsilon)
            tinfcb_costs[t, trial] = k
            tinfcb.update(k)
            
            k, _ = sor(At, bt, np.zeros(n), omega_opt(At), epsilon)
            opt_costs[t, trial] = k
            
            k, _ = sor(At, bt, np.zeros(n), 1.6, epsilon)
            omega_costs[t, trial] = k
            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")
    
    # Plot low-variance results
    plt.figure(2, figsize=(7, 5))
    plt.plot(np.mean(np.cumsum(tinf_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(omega_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(tinfcb_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(cheb_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.plot(np.mean(np.cumsum(opt_costs, axis=0), axis=1), 
             T - np.arange(1, T + 1), linewidth=2)
    plt.legend(['Tsallis-INF', 'ω=1.6 (≈ω*)', 'Tsallis-INF-CB', 'ChebCB', 'Instance-Optimal'], 
               fontsize=12)
    plt.xlabel('total iterations', fontsize=14)
    plt.ylabel('instances remaining', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/contextual_low_variance.png', dpi=256)
    plt.close()
    
    print("Done! Plots saved to plots/")


if __name__ == '__main__':
    main()
