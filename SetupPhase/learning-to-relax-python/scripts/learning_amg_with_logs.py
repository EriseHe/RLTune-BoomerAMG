"""
BoomerAMG Learning Experiment with Detailed Logging

Compares the performance of Tsallis-INF bandit and fixed strong_threshold 
choices, logging every decision made by the bandit.

Output:
- plots/BoomerAMG/learning_high_variance.png
- plots/BoomerAMG/learning_low_variance.png  
- plots/BoomerAMG/experiment_log_high_variance.csv
- plots/BoomerAMG/experiment_log_low_variance.csv
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye
import csv
from datetime import datetime

from learners import TsallisINF_AMG
from solvers.BoomerAMG import boomeramg
from utils import truncated_normal, delsq, numgrid

########################################################
def progress_bar(current, total, prefix="", every=None):
    width = 30
    if total <= 0:
        return
    if every is None:
        every = max(1, total // 100)
    if current not in (1, total) and current % every != 0:
        return
    filled = int(width * current / total)
    bar = "=" * filled + "-" * (width - filled)
    msg = f"\r{prefix} [{bar}] {current}/{total}"
    print(msg, end="", flush=True)
    if current == total:
        print()
########################################################


def run_experiment(A, n, T, trials, thresholds, threshold_grid, epsilon, 
                   beta_params, experiment_name, output_dir):
    """
    Run experiment with logging.
    
    Parameters
    ----------
    beta_params : tuple
        (alpha, beta) for the Beta distribution of diagonal shifts
    experiment_name : str
        Name for this experiment (e.g., "high_variance")
    """
    
    threshold_costs = np.zeros((T, trials, len(thresholds)))
    tinf_costs = np.zeros((T, trials))
    
    # Detailed log storage
    all_logs = []
    
    print(f"\n{'='*60}")
    print(f"Running {experiment_name} experiment (BoomerAMG)...")
    print(f"{'='*60}")
    print(f"Problem size: {n}x{n}, nnz(A)={A.nnz}")
    print(f"Instances: {T}, Trials: {trials}")
    print(f"Beta distribution params: alpha={beta_params[0]}, beta={beta_params[1]}")
    print(f"Threshold grid: {threshold_grid}")
    print(f"Loss transform: 1 + log(1 + WU/40)")
    print()
    
    for trial in range(trials):
        print(f"\n  Trial {trial + 1}/{trials}")
        tinf = TsallisINF_AMG(threshold_grid, T)
        
        for t in range(T):
            # Random diagonal shift
            c = -0.15 + 0.6 * np.random.beta(beta_params[0], beta_params[1])
            At = A + c * speye(n, format="csr")
            bt = truncated_normal(n)
            
            # Tsallis-INF prediction and solve
            theta_pred = tinf.predict()
            k_tinf, cum_nnz_AP_tinf, _ = boomeramg(At, bt, np.zeros(n), theta_pred, epsilon)
            WU_tinf = k_tinf * cum_nnz_AP_tinf
            tinf_costs[t, trial] = WU_tinf
            
            # Faithful log transform: loss = 1 + log(1 + WU/a)
            # No cap, no saturation - preserves hard instance signal
            # a = 40 is a "knee" near easy-instance scale
            a = 40.0
            loss_tinf = 1.0 + math.log1p(WU_tinf / a)
            tinf.update(loss_tinf)
            
            # Fixed threshold solves
            fixed_results = []
            for i, theta in enumerate(thresholds):
                k, cum_nnz_AP, _ = boomeramg(At, bt, np.zeros(n), theta, epsilon)
                WU = k * cum_nnz_AP
                threshold_costs[t, trial, i] = WU
                fixed_results.append({
                    'theta': theta,
                    'iterations': k,
                    'cum_nnz_AP': cum_nnz_AP,
                    'WU': WU
                })
            
            # Find best fixed threshold for this instance
            best_fixed_idx = np.argmin([r['WU'] for r in fixed_results])
            best_fixed = fixed_results[best_fixed_idx]
            
            # Log entry
            log_entry = {
                'trial': trial + 1,
                'instance': t + 1,
                'diagonal_shift': c,
                'bandit_theta': theta_pred,
                'bandit_iterations': k_tinf,
                'bandit_cum_nnz_AP': cum_nnz_AP_tinf,
                'bandit_WU': WU_tinf,
                'best_fixed_theta': best_fixed['theta'],
                'best_fixed_iterations': best_fixed['iterations'],
                'best_fixed_cum_nnz_AP': best_fixed['cum_nnz_AP'],
                'best_fixed_WU': best_fixed['WU'],
                'regret': WU_tinf - best_fixed['WU']
            }
            
            # Add all fixed threshold results
            for i, r in enumerate(fixed_results):
                log_entry[f'theta_{r["theta"]:.2f}_WU'] = r['WU']
            
            all_logs.append(log_entry)
            
            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")
    
    # Save log to CSV
    log_file = os.path.join(output_dir, f'experiment_log_{experiment_name}.csv')
    if all_logs:
        fieldnames = list(all_logs[0].keys())
        with open(log_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_logs)
        print(f"\n  Saved log: {log_file}")
    
    # Generate plot
    plt.figure(figsize=(8, 6))
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
    
    variance_label = "High" if beta_params[0] < 1 else "Low"
    plt.title(f'BoomerAMG: {variance_label}-Variance Offset Distribution', fontsize=14)
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, f'learning_{experiment_name}.png')
    plt.savefig(plot_file, dpi=256)
    plt.close()
    print(f"  Saved plot: {plot_file}")
    
    # Print summary statistics
    print(f"\n  Summary for {experiment_name}:")
    print(f"    Total instances: {T * trials}")
    
    # Average costs
    avg_tinf = np.mean(tinf_costs)
    print(f"    Avg Tsallis-INF WU: {avg_tinf:.2f}")
    for i, theta in enumerate(thresholds):
        avg = np.mean(threshold_costs[:, :, i])
        print(f"    Avg θ={theta:.2f} WU: {avg:.2f}")
    
    # Cumulative regret
    best_per_instance = np.min(threshold_costs, axis=2)
    cumulative_regret = np.sum(tinf_costs - best_per_instance)
    print(f"    Cumulative regret: {cumulative_regret:.2f}")
    
    return threshold_costs, tinf_costs, all_logs


def main():
    print("="*60)
    print("BoomerAMG Learning Experiment with Detailed Logging")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    # Setup
    A = delsq(numgrid('S', 20))
    n = A.shape[0]
    epsilon = 1e-8
    T = 5000
    trials = 1
    
    thresholds = np.array([0.1, 0.25, 0.4, 0.5, 0.7])
    threshold_grid = np.linspace(0.1, 0.9, 9)
    
    output_dir = 'plots/BoomerAMG'
    os.makedirs(output_dir, exist_ok=True)
    
    # High-variance experiment (Beta(0.5, 1.5))
    run_experiment(A, n, T, trials, thresholds, threshold_grid, epsilon,
                   beta_params=(0.5, 1.5), 
                   experiment_name='high_variance',
                   output_dir=output_dir)
    
    # Low-variance experiment (Beta(2.0, 6.0))
    run_experiment(A, n, T, trials, thresholds, threshold_grid, epsilon,
                   beta_params=(2.0, 6.0),
                   experiment_name='low_variance', 
                   output_dir=output_dir)
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nOutput files in {output_dir}/:")
    print("  - learning_high_variance.png")
    print("  - learning_low_variance.png")
    print("  - experiment_log_high_variance.csv")
    print("  - experiment_log_low_variance.csv")


if __name__ == '__main__':
    main()
