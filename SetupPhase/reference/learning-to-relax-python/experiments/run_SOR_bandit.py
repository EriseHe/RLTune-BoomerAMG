"""
SOR Tsallis-INF learning experiment (faithful to the original MATLAB script) + diagnostics.

This script is a faithful adaptation of:
  SetupPhase/learning-to-relax-original/scripts/learning.m

It compares Tsallis-INF to fixed ω baselines on a sequence of T i.i.d. systems
and (optionally) averages over multiple trials.

Instance generation matches the paper/script:
  A = delsq(numgrid('S', S))
  c ~ Beta(·,·) and A_t = A + (-0.15 + 0.6*c) I
  b_t ~ truncated Gaussian (radially truncated)
  x0 = 0
  cost feedback = SOR iterations

Outputs (default directory: `plots/SOR/`):
Faithful learning plots (same shape/axes as MATLAB):
- learning_high_variance.png
- learning_low_variance.png

Additional diagnostics (not in the original MATLAB script, but useful):
- omega_trace_high_variance.png
- omega_trace_low_variance.png
- rolling_regret_high_variance.png
- rolling_regret_low_variance.png

Notes:
- This uses the repo's `solvers/SOR/sor.py` implementation (dense + solve_triangular).
  That implementation converts sparse A to dense internally; we pass dense matrices
  directly to avoid repeated conversions.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

# Reduce matplotlib cache warnings in restricted environments
_mpl_cache = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

# matplotlib backend must be set before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular


def _py_root() -> Path:
    # .../SetupPhase/learning-to-relax-python
    return Path(__file__).resolve().parents[1]


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or x.size == 0:
        return x
    w = int(window)
    w = min(w, int(x.size))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = (c[w:] - c[:-w]) / float(w)
    pad_left = w // 2
    pad_right = int(x.size) - int(out.size) - pad_left
    return np.pad(out, (pad_left, pad_right), mode="edge")


def _progress(t: int, T: int, start_time: float, tag: str, every: int) -> None:
    if every > 0 and ((t + 1) % every == 0 or (t + 1) == T):
        mins = (time.time() - start_time) / 60.0
        print(f"{tag}: {t+1}/{T} ({mins:.1f} min)")


def _parse_grid(spec: str) -> np.ndarray:
    gmin, gmax, gcount = spec.split(":")
    return np.linspace(float(gmin), float(gmax), int(gcount))


def _parse_list(spec: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in spec.split(",") if x.strip()], dtype=np.float64)


def _run_one_variance(
    *,
    A_dense: np.ndarray,
    A_csr: sp.csr_matrix,
    L_csr: sp.csr_matrix,
    diagA: np.ndarray,
    T: int,
    trials: int,
    epsilon: float,
    baselines: np.ndarray,
    omega_grid: np.ndarray,
    beta_a: float,
    beta_b: float,
    tag: str,
    seed: int | None,
    progress_every: int,
    fast: bool,
) -> Dict[str, object]:
    from learners import TsallisINF
    from solvers import sor as sor_dense
    from utils import truncated_normal

    n = int(A_dense.shape[0])
    x0 = np.zeros(n, dtype=np.float64)

    def sor_sparse_shifted(b: np.ndarray, omega: float, c_shift: float) -> int:
        """
        Sparse SOR implementation equivalent to solvers/SOR/sor.py, but:
        - uses sparse matvec (A_csr @ x) instead of dense
        - applies diagonal shift c_shift without forming A_t explicitly
        """
        omega = float(omega)
        c_shift = float(c_shift)

        # M = D/omega + L where D is diagonal of (A + cI) and L is strictly lower of A.
        D_diag = diagA + c_shift
        M = L_csr + sp.diags(D_diag / omega, offsets=0, format="csr")

        x = x0.copy()
        r = b - (A_csr @ x + c_shift * x)
        norm0 = np.linalg.norm(r)
        if norm0 == 0:
            return 1

        for k in range(1, 10001):
            # x = x + M \ r
            dx = spsolve_triangular(M, r, lower=True)
            x = x + dx
            r = b - (A_csr @ x + c_shift * x)
            if np.linalg.norm(r) / norm0 < epsilon:
                return k
        return 10000

    baseline_k = np.zeros((T, trials, len(baselines)), dtype=np.float64)
    tinf_k = np.zeros((T, trials), dtype=np.float64)
    tinf_omega = np.zeros((T, trials), dtype=np.float64)

    start = time.time()
    for trial in range(trials):
        # Faithful to MATLAB: each trial is an independent random run.
        if seed is not None:
            np.random.seed(int(seed) + trial)

        bandit = TsallisINF(omega_grid, T)

        for t in range(T):
            c = -0.15 + 0.6 * np.random.beta(beta_a, beta_b)

            b = truncated_normal(n)

            # Bandit play
            omega = float(bandit.predict())
            if fast:
                k = sor_sparse_shifted(b, omega, c)
            else:
                # Build At_dense = A_dense + c*I without allocating an identity.
                At_dense = A_dense.copy()
                At_dense.flat[:: n + 1] += c
                k, _ = sor_dense(At_dense, b, x0, omega, epsilon)
            tinf_k[t, trial] = float(k)
            tinf_omega[t, trial] = omega
            bandit.update(k)

            # Fixed baselines for the same instance
            for i, omega_fixed in enumerate(baselines):
                if fast:
                    k = sor_sparse_shifted(b, float(omega_fixed), c)
                else:
                    k, _ = sor_dense(At_dense, b, x0, float(omega_fixed), epsilon)
                baseline_k[t, trial, i] = float(k)

            if trial == 0:
                _progress(t, T, start, tag, every=progress_every)

    avg_baselines = baseline_k.mean(axis=(0, 1))
    best_idx = int(np.argmin(avg_baselines))

    return {
        "tag": tag,
        "beta": (beta_a, beta_b),
        "baselines": baselines,
        "baseline_k": baseline_k,
        "best_idx": best_idx,
        "best_omega": float(baselines[best_idx]),
        "best_avg": float(avg_baselines[best_idx]),
        "tinf_k": tinf_k,
        "tinf_omega": tinf_omega,
        "tinf_avg": float(tinf_k.mean()),
    }


def _plot_learning_faithful(
    *,
    result: Dict[str, object],
    out_path: Path,
    legend_labels: Tuple[str, str, str, str, str],
    omega_star_idx: int,
) -> None:
    baselines = np.asarray(result["baselines"], dtype=np.float64)  # type: ignore[arg-type]
    baseline_k = np.asarray(result["baseline_k"], dtype=np.float64)  # type: ignore[arg-type]
    tinf_k = np.asarray(result["tinf_k"], dtype=np.float64)  # type: ignore[arg-type]

    T = int(tinf_k.shape[0])
    y = T - np.arange(1, T + 1)

    plt.figure(figsize=(7, 5))
    ax = plt.gca()

    # MATLAB: plot(mean(cumsum(costs),2), T-(1:T))
    colors = plt.cm.tab10(np.linspace(0, 1, len(baselines)))
    for i, omega in enumerate(baselines):
        x = baseline_k[:, :, i].cumsum(axis=0).mean(axis=1)
        label = legend_labels[i]
        ax.plot(x, y, linewidth=2, linestyle="--", color=colors[i], label=label)

    x_tinf = tinf_k.cumsum(axis=0).mean(axis=1)
    ax.plot(x_tinf, y, linewidth=2, color="black", label="Tsallis-INF")

    ax.set_xlabel("total iterations", fontsize=20)
    ax.set_ylabel("instances remaining", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=14, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=256)
    plt.close()


def _plot_omega_trace(*, results_by_tag: Dict[str, Dict[str, object]], out_path: Path, window: int) -> None:
    tags = list(results_by_tag.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(8 * len(tags), 4), sharey=True)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        r = results_by_tag[tag]
        omega = np.asarray(r["tinf_omega"], dtype=np.float64).mean(axis=1)  # type: ignore[arg-type]
        best_omega = float(r["best_omega"])  # type: ignore[arg-type]

        T = int(omega.size)
        t = np.arange(1, T + 1)
        ax.plot(t, _moving_average(omega, window), linewidth=2, color="black", label="Tsallis-INF (MA)")
        ax.axhline(best_omega, linestyle="--", color="tab:blue", alpha=0.6, linewidth=2, label="best fixed baseline")

        ax.set_title(f"{tag.replace('_', ' ')} ω(t) (MA{window})", fontsize=13)
        ax.set_xlabel("t", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10, loc="upper right")

    axes[0].set_ylabel("ω", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def _plot_rolling_regret(*, results_by_tag: Dict[str, Dict[str, object]], out_path: Path, window: int) -> None:
    tags = list(results_by_tag.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(8 * len(tags), 4), sharey=True)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        r = results_by_tag[tag]
        baseline_k = np.asarray(r["baseline_k"], dtype=np.float64)  # type: ignore[arg-type]
        best_idx = int(r["best_idx"])  # type: ignore[arg-type]
        best_series = baseline_k[:, :, best_idx]

        tinf_k = np.asarray(r["tinf_k"], dtype=np.float64)  # type: ignore[arg-type]
        regret = (tinf_k - best_series).mean(axis=1)

        T = int(regret.size)
        t = np.arange(1, T + 1)
        ax.plot(t, _moving_average(regret, window), linewidth=2, color="black")
        ax.axhline(0.0, linestyle="--", color="tab:blue", alpha=0.6, linewidth=2)

        ax.set_title(f"{tag.replace('_', ' ')} regret vs best fixed (MA{window})", fontsize=13)
        ax.set_xlabel("t", fontsize=12)
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Iterations regret", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def run_compare(
    *,
    T: int = 5000,
    S: int = 12,
    trials: int = 40,
    epsilon: float = 1e-8,
    variance: str = "both",
    omega_grid: str = "1.0:1.95:20",
    baselines: str = "1.0,1.2,1.4,1.6,1.8",
    out_dir: str | None = None,
    seed: int | None = 0,
    ma_window: int = 25,
    progress_every: int = 100,
    fast: bool = False,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Path]]:
    if T <= 0:
        raise ValueError("T must be positive")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if variance not in {"high", "low", "both"}:
        raise ValueError("variance must be one of: high, low, both")
    if ma_window <= 0:
        raise ValueError("ma_window must be positive")

    py_root = _py_root()
    sys.path.insert(0, str(py_root))

    from utils import delsq, numgrid

    A = delsq(numgrid("S", S))
    A_dense = A.toarray().astype(np.float64, copy=False)
    A_csr = sp.csr_matrix(A, dtype=np.float64)
    L_csr = sp.tril(A_csr, k=-1, format="csr")
    diagA = A_csr.diagonal().astype(np.float64, copy=True)

    baseline_omegas = _parse_list(baselines)
    bandit_grid = _parse_grid(omega_grid)

    output_dir = Path(out_dir) if out_dir else (py_root / "plots" / "SOR")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_by_tag: Dict[str, Dict[str, object]] = {}
    if variance in {"high", "both"}:
        results_by_tag["high_variance"] = _run_one_variance(
            A_dense=A_dense,
            A_csr=A_csr,
            L_csr=L_csr,
            diagA=diagA,
            T=T,
            trials=trials,
            epsilon=epsilon,
            baselines=baseline_omegas,
            omega_grid=bandit_grid,
            beta_a=0.5,
            beta_b=1.5,
            tag="high_variance",
            seed=seed,
            progress_every=progress_every,
            fast=fast,
        )
    if variance in {"low", "both"}:
        results_by_tag["low_variance"] = _run_one_variance(
            A_dense=A_dense,
            A_csr=A_csr,
            L_csr=L_csr,
            diagA=diagA,
            T=T,
            trials=trials,
            epsilon=epsilon,
            baselines=baseline_omegas,
            omega_grid=bandit_grid,
            beta_a=2.0,
            beta_b=6.0,
            tag="low_variance",
            seed=(seed + 10_000) if seed is not None else None,
            progress_every=progress_every,
            fast=fast,
        )

    plot_paths: Dict[str, Path] = {}

    # Faithful learning plots: two separate figures like MATLAB.
    if "high_variance" in results_by_tag:
        out_path = output_dir / "learning_high_variance.png"
        _plot_learning_faithful(
            result=results_by_tag["high_variance"],
            out_path=out_path,
            legend_labels=(r"ω=1.0", r"ω=1.2", r"ω=1.4", r"ω=1.6", r"ω=1.8 (≈ω*)"),
            omega_star_idx=4,
        )
        plot_paths["learning_high_variance"] = out_path

        omega_trace_path = output_dir / "omega_trace_high_variance.png"
        rolling_regret_path = output_dir / "rolling_regret_high_variance.png"
        _plot_omega_trace(results_by_tag={"high_variance": results_by_tag["high_variance"]}, out_path=omega_trace_path, window=int(ma_window))
        _plot_rolling_regret(results_by_tag={"high_variance": results_by_tag["high_variance"]}, out_path=rolling_regret_path, window=int(ma_window))
        plot_paths["omega_trace_high_variance"] = omega_trace_path
        plot_paths["rolling_regret_high_variance"] = rolling_regret_path

    if "low_variance" in results_by_tag:
        out_path = output_dir / "learning_low_variance.png"
        _plot_learning_faithful(
            result=results_by_tag["low_variance"],
            out_path=out_path,
            legend_labels=(r"ω=1.0", r"ω=1.2", r"ω=1.4", r"ω=1.6 (≈ω*)", r"ω=1.8"),
            omega_star_idx=3,
        )
        plot_paths["learning_low_variance"] = out_path

        omega_trace_path = output_dir / "omega_trace_low_variance.png"
        rolling_regret_path = output_dir / "rolling_regret_low_variance.png"
        _plot_omega_trace(results_by_tag={"low_variance": results_by_tag["low_variance"]}, out_path=omega_trace_path, window=int(ma_window))
        _plot_rolling_regret(results_by_tag={"low_variance": results_by_tag["low_variance"]}, out_path=rolling_regret_path, window=int(ma_window))
        plot_paths["omega_trace_low_variance"] = omega_trace_path
        plot_paths["rolling_regret_low_variance"] = rolling_regret_path

    return results_by_tag, plot_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=5000, help="Number of instances")
    parser.add_argument("--S", type=int, default=12, help="numgrid('S', S) size (matrix size grows with S)")
    parser.add_argument("--trials", type=int, default=40, help="Number of trials to average (MATLAB uses 40)")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="SOR tolerance")
    parser.add_argument("--seed", type=int, default=0, help="Seed for env + bandit RNG (set -1 for random)")
    parser.add_argument("--variance", choices=["high", "low", "both"], default="both")
    parser.add_argument("--omega-grid", type=str, default="1.0:1.95:20", help="Bandit grid as 'min:max:count'")
    parser.add_argument("--baselines", type=str, default="1.0,1.2,1.4,1.6,1.8", help="Baseline ω list")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--ma-window", type=int, default=25, help="Moving-average window for diagnostics plots")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N instances (0 disables)")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a sparse SOR implementation (much faster; same algorithm, slight numeric differences possible)",
    )
    args = parser.parse_args(argv)

    seed = None if args.seed < 0 else int(args.seed)

    results_by_tag, plot_paths = run_compare(
        T=args.T,
        S=args.S,
        trials=args.trials,
        epsilon=args.epsilon,
        variance=args.variance,
        omega_grid=args.omega_grid,
        baselines=args.baselines,
        out_dir=args.out_dir,
        seed=seed,
        ma_window=args.ma_window,
        progress_every=args.progress_every,
        fast=args.fast,
    )

    print("\n=== Summary (avg iterations) ===")
    for tag, r in results_by_tag.items():
        print(f"{tag}:")
        print(f"  best fixed ω={float(r['best_omega']):.2f} avg iters {float(r['best_avg']):.2f}")
        print(f"  Tsallis-INF     avg iters {float(r['tinf_avg']):.2f}")

    print("\nSaved plots:")
    for name, p in plot_paths.items():
        print(f"  {name}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
