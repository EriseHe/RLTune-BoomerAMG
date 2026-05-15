"""
Direct learning.py-style adaptation for BoomerAMG head-to-head:
- TsallisINF (SOR translation)
- TsallisINF_AMG (AMG numerical handling)

High-variance only, single trial by default.
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib

_mpl_cache = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import eye as speye

from experiments.run_AMG_bandit import _ensure_hypre_env, _ensure_mpi_env
from learners import TsallisINF
from learners.TsallisINF_AMG import TsallisINF_AMG
from solvers.BoomerAMG import boomeramg
from utils import delsq, numgrid, truncated_normal


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


def _loss_from_wu(wu, mode, knee):
    if mode == "raw":
        return 1.0 + wu
    if mode == "linear":
        return 1.0 + wu / knee
    if mode == "log":
        return 1.0 + np.log1p(wu / knee)
    raise ValueError(f"Unknown loss mode: {mode}")


def main():
    _ensure_hypre_env(None)
    _ensure_mpi_env()

    A = delsq(numgrid("S", 20))
    n = A.shape[0]
    epsilon = 1e-8
    T = 5000
    trials = 1

    np.random.seed(20260208)
    baselines = np.array([0.1, 0.25, 0.4, 0.5, 0.7])
    bandit_grid = np.linspace(0.1, 0.9, 9)
    loss_mode = "raw"
    knee = 40.0

    baseline_costs = np.zeros((T, trials, len(baselines)))
    sor_bandit_costs = np.zeros((T, trials))
    amg_bandit_costs = np.zeros((T, trials))

    I = speye(n, format="csr")

    print("Running high-variance offset distribution (BoomerAMG head-to-head)...")
    for trial in range(trials):
        print(f"  Trial {trial + 1}/{trials}")
        sor_bandit = TsallisINF(bandit_grid, T)
        amg_bandit = TsallisINF_AMG(bandit_grid, T)

        # Keep bandit sampling independent but deterministic.
        state_sor = np.random.RandomState(7).get_state()
        state_amg = np.random.RandomState(7).get_state()

        for t in range(T):
            c = -0.15 + 0.6 * np.random.beta(0.5, 1.5)
            At = A + c * I
            bt = truncated_normal(n)
            x0 = np.zeros(n)

            for i, theta in enumerate(baselines):
                k, comp, _ = boomeramg(At, bt, x0, float(theta), epsilon)
                baseline_costs[t, trial, i] = float(k) * float(comp)

            np.random.set_state(state_sor)
            theta_sor = float(sor_bandit.predict())
            k_sor, comp_sor, _ = boomeramg(At, bt, x0, theta_sor, epsilon)
            wu_sor = float(k_sor) * float(comp_sor)
            sor_bandit_costs[t, trial] = wu_sor
            sor_bandit.update(_loss_from_wu(wu_sor, loss_mode, knee))
            state_sor = np.random.get_state()

            np.random.set_state(state_amg)
            theta_amg = float(amg_bandit.predict())
            k_amg, comp_amg, _ = boomeramg(At, bt, x0, theta_amg, epsilon)
            wu_amg = float(k_amg) * float(comp_amg)
            amg_bandit_costs[t, trial] = wu_amg
            amg_bandit.update(_loss_from_wu(wu_amg, loss_mode, knee))
            state_amg = np.random.get_state()

            progress_bar(t + 1, T, prefix=f"    trial {trial + 1}/{trials}")

    out_dir = Path("plots/BoomerAMG")
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for i in range(len(baselines)):
        label = f"θ={baselines[i]:.2f}" + (" (2D default)" if baselines[i] == 0.25 else "")
        plt.plot(
            np.mean(np.cumsum(baseline_costs[:, :, i], axis=0), axis=1),
            T - np.arange(1, T + 1),
            linewidth=2,
            linestyle="--",
            label=label,
        )

    plt.plot(
        np.mean(np.cumsum(sor_bandit_costs, axis=0), axis=1),
        T - np.arange(1, T + 1),
        linewidth=2.5,
        color="black",
        label="TsallisINF_SOR (raw)",
    )

    plt.plot(
        np.mean(np.cumsum(amg_bandit_costs, axis=0), axis=1),
        T - np.arange(1, T + 1),
        linewidth=2.5,
        color="magenta",
        label="TsallisINF_AMG (raw)",
    )

    plt.xlabel("Total Work Units (WU)", fontsize=14)
    plt.ylabel("Instances Remaining", fontsize=14)
    plt.title("BoomerAMG high variance: TsallisINF_SOR vs TsallisINF_AMG", fontsize=14)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()

    out_path = out_dir / "head2head_sor_vs_amg_highvar_raw_T5000_trial1_S20.png"
    plt.savefig(out_path, dpi=220)
    plt.close()

    avg_sor = float(np.mean(sor_bandit_costs))
    avg_amg = float(np.mean(amg_bandit_costs))
    print(f"\nSaved plot: {out_path.resolve()}")
    print(f"Avg WU TsallisINF_SOR: {avg_sor:.6f}")
    print(f"Avg WU TsallisINF_AMG: {avg_amg:.6f}")


if __name__ == "__main__":
    main()
