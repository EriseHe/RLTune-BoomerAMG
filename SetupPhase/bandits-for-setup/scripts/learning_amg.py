"""
Compares Tsallis-INF bandit vs fixed strong_threshold values on BoomerAMG.

Matrix randomization matches the solve-phase RL environment
(SolvePhase/hypre/src/test/amg_gym_env.py with randomize_A=True, randomize_b=True):
  - Default here uses 60x60x1 grid, 27pt stencil (fast 2D-like test)
  - Each round: random off-diagonal multipliers s1,s2,s3 ~ U(0.5, 2.0),
    random diagonal shift c ~ U(0.0, 5.0), random RHS seed
  - Different matrix every round

Loss = work units = iterations * complexity (cumulative nnz ratio).

Direct AMG analog of learning.py (which does Tsallis-INF for SOR omega).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt

from learners import TsallisINF_AMG
from solver import solve

# ---- Base 27pt coefficients (same as solve-phase) ---------------------------

A1_BASE = -4.0
A2_BASE = -0.15
A3_BASE = -0.0125

NX, NY, NZ = 60, 60, 1
STENCIL = 27

# ---- Matrix sampler (matches amg_gym_env.py randomize_A/randomize_b) --------

def sample_matrix_kwargs(rng):
    """
    Sample random matrix coefficients matching the solve-phase environment.
    Returns kwargs dict to pass to solve().
    """
    # Random off-diagonal multipliers
    s1 = float(rng.uniform(0.5, 2.0))
    s2 = float(rng.uniform(0.5, 2.0))
    s3 = float(rng.uniform(0.5, 2.0))

    a1 = A1_BASE * s1
    a2 = A2_BASE * s2
    a3 = A3_BASE * s3

    # Random diagonal shift for SPD/M-matrix enforcement
    c_diag = float(rng.uniform(0.0, 5.0))
    # Off-diagonal neighbor counts depend on dimension:
    # 3D 27pt: faces=6, edges=12, corners=8
    # 2D (nz=1): faces=4, edges=4, corners=0
    if NZ == 1:
        a0 = -(4.0 * a1 + 4.0 * a2) + c_diag
    else:
        a0 = -(6.0 * a1 + 12.0 * a2 + 8.0 * a3) + c_diag

    # Random RHS seed
    rhs_seed = int(rng.integers(0, 2**63 - 1))

    return dict(
        nx=NX, ny=NY, nz=NZ,
        stencil=STENCIL,
        rhs_type=1,
        rhs_seed=rhs_seed,
        k=1.0, c=0.0,
        a0=a0, a1=a1, a2=a2, a3=a3,
    )

# ---- Progress bar -----------------------------------------------------------

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
    print(f"\r{prefix} [{bar}] {current}/{total}", end="", flush=True)
    if current == total:
        print()


# ---- Main -------------------------------------------------------------------

def main():
    T = 1000          # rounds (each round = one full BoomerAMG solve)
    trials = 1        # independent trials for averaging
    master_seed = 42  # for reproducibility

    # Fixed baselines: single comparator
    thresholds_fixed = [0.1, 0.25, 0.4, 0.5, 0.7]

    # Bandit grid: finer discretization for Tsallis-INF
    bandit_grid = np.linspace(0.05, 0.95, 19)

    # Storage
    fixed_costs = np.zeros((T, trials, len(thresholds_fixed)))
    bandit_costs = np.zeros((T, trials))

    print(f"Matrix: {NX}x{NY}x{NZ}, {STENCIL}pt stencil (randomized each round)")
    print(f"Rounds T={T}, trials={trials}")
    print(f"Fixed baselines: {thresholds_fixed}")
    print(f"Bandit grid: {len(bandit_grid)} arms in [{bandit_grid[0]:.2f}, {bandit_grid[-1]:.2f}]")
    print()

    for trial in range(trials):
        print(f"Trial {trial + 1}/{trials}")

        # Each trial gets its own RNG for matrix sampling (reproducible)
        rng = np.random.default_rng(master_seed + trial)
        bandit = TsallisINF_AMG(bandit_grid, T)

        for t in range(T):
            # Sample a new random matrix (same distribution as solve-phase)
            mkw = sample_matrix_kwargs(rng)

            # --- Bandit arm ---
            threshold = bandit.predict()
            result = solve(params={"strong_threshold": threshold}, **mkw)
            wu = result.work_units
            bandit_costs[t, trial] = wu
            bandit.update(wu)

            # --- Fixed baselines (same matrix instance) ---
            for i, th in enumerate(thresholds_fixed):
                result_fixed = solve(params={"strong_threshold": th}, **mkw)
                fixed_costs[t, trial, i] = result_fixed.work_units

            progress_bar(t + 1, T, prefix=f"  trial {trial + 1}")

        print(f"  Bandit fallbacks: {bandit.fallback_count}")

    # ---- Plot: cumulative cost curves ---------------------------------------
    plt.figure(figsize=(8, 5))

    for i, th in enumerate(thresholds_fixed):
        cum = np.mean(np.cumsum(fixed_costs[:, :, i], axis=0), axis=1)
        plt.plot(cum, T - np.arange(1, T + 1),
                 linewidth=2, linestyle='--', label=f'threshold={th}')

    cum_bandit = np.mean(np.cumsum(bandit_costs, axis=0), axis=1)
    plt.plot(cum_bandit, T - np.arange(1, T + 1),
             linewidth=2, color='black', label='Tsallis-INF')

    plt.xlabel('cumulative work units', fontsize=14)
    plt.ylabel('instances remaining', fontsize=14)
    plt.title('BoomerAMG Setup: Tsallis-INF vs Fixed strong_threshold', fontsize=13)
    plt.legend(fontsize=10)
    plt.tight_layout()

    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/learning_amg.png', dpi=256)
    plt.close()

    # ---- Summary stats ------------------------------------------------------
    print()
    print("=== Summary (mean total WU over all rounds) ===")
    for i, th in enumerate(thresholds_fixed):
        total = np.mean(np.sum(fixed_costs[:, :, i], axis=0))
        print(f"  threshold={th:<5}  total WU = {total:.1f}")
    total_bandit = np.mean(np.sum(bandit_costs, axis=0))
    print(f"  Tsallis-INF       total WU = {total_bandit:.1f}")
    print()
    print("Plot saved to plots/learning_amg.png")


if __name__ == '__main__':
    main()
