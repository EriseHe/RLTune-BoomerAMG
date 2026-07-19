"""
Compares Tsallis-INF bandit vs fixed strong_threshold values on BoomerAMG.

Matrix randomization matches the solve-phase RL environment
(SolvePhase/core/amg_gym_env.py with randomize_A=True, randomize_b=True):
  - 60x60x1 grid, 27pt stencil
  - Each round: random off-diagonal multipliers s1,s2,s3 ~ U(0.5, 2.0),
    random diagonal shift c ~ U(0.0, 5.0), random RHS seed
  - Different matrix every round

Loss = work units = iterations * complexity (cumulative nnz ratio).

Direct AMG analog of learning.py (which does Tsallis-INF for SOR omega).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import _project_paths  # noqa: E402,F401

import contextlib
import numpy as np
import matplotlib.pyplot as plt

from learners import TsallisINF_AMG
from hypre.bindings import solve
from problems.amg import NX, NY, NZ, STENCIL, stencil_27_laplace
from utils.paths import SETUP_RESULTS_ROOT

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


@contextlib.contextmanager
def suppress_solver_output(enabled=True):
    """
    Suppress native (C-level) stdout/stderr noise from HYPRE matrix builders.
    Keeps Python-side progress bar on one updating line.
    """
    if not enabled:
        yield
        return

    with open(os.devnull, "w") as devnull:
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)

# ---- Main -------------------------------------------------------------------

def main():
    T = 1000          # rounds (each round = one full BoomerAMG solve)
    trials = 1        # independent trials for averaging
    master_seed = 42  # for reproducibility
    quiet_solver_output = True

    # Fixed baselines: single comparator
    thresholds_fixed = [0.25]

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
            mkw, _, _ = stencil_27_laplace(rng=rng)

            with suppress_solver_output(quiet_solver_output):
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

    output_dir = SETUP_RESULTS_ROOT / "learning_amg"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "learning_amg.png"
    plt.savefig(plot_path, dpi=256)
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
    print(f"Plot saved to {plot_path}")


if __name__ == '__main__':
    main()
