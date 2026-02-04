"""
Setup-phase bandit experiment: compare bandit loss transforms for BoomerAMG.

Reproduces the A/B test:
- Same exact sequence of linear system instances (At, bt)
- Same baseline θ comparisons
- Two Tsallis-INF bandits that differ only in the update loss:

  (1) log loss:     loss = 1 + log(1 + WU / knee)
  (2) linear loss:  loss = 1 + (WU / knee)

where WU = iterations * cum_nnz_AP.

Outputs (default directory: `plots/BoomerAMG/`):
- loss_compare_log_T{T}_trial1_S{S}.png
- loss_compare_linear_T{T}_trial1_S{S}.png
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

import numpy as np

# Reduce matplotlib cache warnings in restricted environments
_mpl_cache = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

# matplotlib backend must be set before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _py_root() -> Path:
    # .../SetupPhase/learning-to-relax-python
    return Path(__file__).resolve().parents[1]


def _default_hypre_lib() -> Path:
    return _repo_root() / "SolvePhase" / "hypre" / "src" / "lib" / "libHYPRE.dylib"


def _ensure_hypre_env(hypre_lib: str | None) -> None:
    if hypre_lib:
        os.environ["HYPRE_LIBHYPRE"] = hypre_lib
        return
    os.environ.setdefault("HYPRE_LIBHYPRE", str(_default_hypre_lib()))


def _ensure_mpi_env() -> None:
    """
    Ensure MPI is initialized and HYPRE_MPI_COMM_WORLD is set.

    The repo’s bundled libHYPRE is MPI-enabled on macOS; boomeramg.py reads
    HYPRE_MPI_COMM_WORLD to obtain a communicator handle.
    """
    if os.environ.get("HYPRE_MPI_COMM_WORLD"):
        return

    try:
        from mpi4py import MPI  # type: ignore

        if not MPI.Is_initialized():
            MPI.Init()
        os.environ["HYPRE_MPI_COMM_WORLD"] = str(MPI.COMM_WORLD.handle)
    except Exception as exc:
        raise RuntimeError(
            "MPI is required for the bundled libHYPRE.dylib but mpi4py is not available.\n"
            "Install mpi4py or set HYPRE_MPI_COMM_WORLD manually."
        ) from exc


def _truncated_normal_rng(rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Same rejection sampler as utils/truncated_normal.py, but driven by a local RNG
    so we can keep the instance stream identical across loss variants.
    """
    while True:
        sample = rng.standard_normal(n)
        if np.linalg.norm(sample) <= np.sqrt(n):
            return sample


def _progress(t: int, T: int, start_time: float, tag: str, every: int = 100) -> None:
    if (t + 1) % every == 0 or (t + 1) == T:
        mins = (time.time() - start_time) / 60.0
        print(f"{tag}: {t+1}/{T} ({mins:.1f} min)")


def _parse_grid(spec: str) -> np.ndarray:
    gmin, gmax, gcount = spec.split(":")
    return np.linspace(float(gmin), float(gmax), int(gcount))


def _parse_list(spec: str) -> np.ndarray:
    return np.array([float(x.strip()) for x in spec.split(",") if x.strip()], dtype=np.float64)


def _run_one_variance(
    *,
    A,
    I,
    n: int,
    T: int,
    epsilon: float,
    baselines: np.ndarray,
    bandit_grid: np.ndarray,
    beta_a: float,
    beta_b: float,
    tag: str,
    seed_env: int,
    seed_bandit: int,
    knee: float,
) -> Dict[str, object]:
    from learners import TsallisINF_AMG
    from solvers.BoomerAMG import boomeramg

    rng_env = np.random.default_rng(seed_env)

    baseline_wu = np.zeros((T, len(baselines)), dtype=np.float64)
    wu_log = np.zeros(T, dtype=np.float64)
    wu_linear = np.zeros(T, dtype=np.float64)

    tinf_log = TsallisINF_AMG(bandit_grid, T)
    tinf_linear = TsallisINF_AMG(bandit_grid, T)

    # TsallisINF_AMG uses NumPy global RNG for sampling.
    # Give both bandits the same starting RNG state but then keep independent streams.
    np.random.seed(seed_bandit)
    base_state = np.random.get_state()
    state_log = base_state
    state_linear = base_state

    start = time.time()
    for t in range(T):
        # Environment instance (independent of bandit randomness):
        c = -0.15 + 0.6 * rng_env.beta(beta_a, beta_b)
        At = A + c * I
        bt = _truncated_normal_rng(rng_env, n)
        x0 = np.zeros(n)

        # Baselines (computed once; reused for both loss-mode comparisons)
        for i, base_theta in enumerate(baselines):
            k, comp, _ = boomeramg(At, bt, x0, float(base_theta), epsilon)
            baseline_wu[t, i] = float(k) * float(comp)

        # Bandit (log-loss): loss = 1 + log1p(WU / knee)
        np.random.set_state(state_log)
        theta = float(tinf_log.predict())
        k, comp, _ = boomeramg(At, bt, x0, theta, epsilon)
        WU = float(k) * float(comp)
        wu_log[t] = WU
        tinf_log.update(1.0 + float(np.log1p(WU / knee)))
        state_log = np.random.get_state()

        # Bandit (linear-loss): loss = 1 + (WU / knee)
        np.random.set_state(state_linear)
        theta = float(tinf_linear.predict())
        k, comp, _ = boomeramg(At, bt, x0, theta, epsilon)
        WU = float(k) * float(comp)
        wu_linear[t] = WU
        tinf_linear.update(1.0 + float(WU / knee))
        state_linear = np.random.get_state()

        _progress(t, T, start, tag)

    avg_baselines = baseline_wu.mean(axis=0)
    best_idx = int(np.argmin(avg_baselines))

    return {
        "tag": tag,
        "beta": (beta_a, beta_b),
        "baselines": baselines,
        "baseline_wu": baseline_wu,
        "best_theta": float(baselines[best_idx]),
        "best_avg": float(avg_baselines[best_idx]),
        "wu_log": wu_log,
        "wu_linear": wu_linear,
        "avg_log": float(wu_log.mean()),
        "avg_linear": float(wu_linear.mean()),
    }


def _plot_mode(*, results_by_tag: Dict[str, Dict[str, object]], mode: str, out_path: Path) -> None:
    tags = list(results_by_tag.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(8 * len(tags), 6), sharey=True)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        r = results_by_tag[tag]
        T = int(len(r["wu_log"]))  # type: ignore[arg-type]
        y = T - np.arange(1, T + 1)

        baselines = r["baselines"]  # type: ignore[assignment]
        baseline_wu = r["baseline_wu"]  # type: ignore[assignment]

        colors = plt.cm.tab10(np.linspace(0, 1, len(baselines)))
        for i, th in enumerate(baselines):
            ax.plot(
                np.cumsum(baseline_wu[:, i]),
                y,
                linestyle="--",
                linewidth=2,
                color=colors[i],
                label=f"θ={float(th):.2f}",
            )

        if mode == "log":
            ax.plot(
                np.cumsum(r["wu_log"]),
                y,
                color="black",
                linewidth=3,
                label="Tsallis-INF (log loss)",
            )  # type: ignore[arg-type]
            title = f"{tag.replace('_', ' ')} (log loss)"
        else:
            ax.plot(
                np.cumsum(r["wu_linear"]),
                y,
                color="black",
                linewidth=3,
                label="Tsallis-INF (linear loss)",
            )  # type: ignore[arg-type]
            title = f"{tag.replace('_', ' ')} (linear loss)"

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Total Work Units (WU)", fontsize=13)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10, loc="upper right")

    axes[0].set_ylabel("Instances Remaining", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=5000)
    parser.add_argument("--S", type=int, default=20, help="numgrid('S', S) size (matrix size grows with S)")
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--knee", type=float, default=40.0, help="knee constant used in both loss variants")
    parser.add_argument("--seed-bandit", type=int, default=0)
    parser.add_argument("--seed-env-high", type=int, default=0)
    parser.add_argument("--seed-env-low", type=int, default=1)
    parser.add_argument("--hypre-lib", type=str, default=None, help="Path to libHYPRE (overrides HYPRE_LIBHYPRE)")
    parser.add_argument(
        "--variance",
        choices=["high", "low", "both"],
        default="both",
        help="Which offset distribution(s) to run",
    )
    parser.add_argument("--theta-grid", type=str, default="0.1:0.9:9", help="Bandit grid as 'min:max:count'")
    parser.add_argument("--baselines", type=str, default="0.1,0.25,0.4,0.5,0.7", help="Baseline θ list")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args(argv)

    if args.T <= 0:
        raise SystemExit("--T must be positive")
    if args.knee <= 0:
        raise SystemExit("--knee must be positive")

    _ensure_hypre_env(args.hypre_lib)
    _ensure_mpi_env()

    py_root = _py_root()
    sys.path.insert(0, str(py_root))

    from utils import delsq, numgrid

    A = delsq(numgrid("S", args.S))
    n = int(A.shape[0])
    I = speye(n, format="csr")

    baselines = _parse_list(args.baselines)
    bandit_grid = _parse_grid(args.theta_grid)

    out_dir = Path(args.out_dir) if args.out_dir else (py_root / "plots" / "BoomerAMG")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Problem: numgrid('S', {args.S}) -> n={n}, nnz(A)={A.nnz}")
    print("T=%d, trials=1" % args.T)
    print(f"Baselines: {baselines}")
    print(f"Bandit grid: {bandit_grid}")
    print(f"knee={args.knee}")
    print(f"HYPRE_LIBHYPRE={os.environ.get('HYPRE_LIBHYPRE')}")
    print(f"HYPRE_MPI_COMM_WORLD={os.environ.get('HYPRE_MPI_COMM_WORLD')}")

    results: Dict[str, Dict[str, object]] = {}
    if args.variance in {"high", "both"}:
        results["high_variance"] = _run_one_variance(
            A=A,
            I=I,
            n=n,
            T=args.T,
            epsilon=args.epsilon,
            baselines=baselines,
            bandit_grid=bandit_grid,
            beta_a=0.5,
            beta_b=1.5,
            tag="high_variance",
            seed_env=args.seed_env_high,
            seed_bandit=args.seed_bandit,
            knee=args.knee,
        )
    if args.variance in {"low", "both"}:
        results["low_variance"] = _run_one_variance(
            A=A,
            I=I,
            n=n,
            T=args.T,
            epsilon=args.epsilon,
            baselines=baselines,
            bandit_grid=bandit_grid,
            beta_a=2.0,
            beta_b=6.0,
            tag="low_variance",
            seed_env=args.seed_env_low,
            seed_bandit=args.seed_bandit,
            knee=args.knee,
        )

    # Summary
    print("\n=== Summary (avg WU) ===")
    for tag, r in results.items():
        best_theta = float(r["best_theta"])  # type: ignore[arg-type]
        best_avg = float(r["best_avg"])  # type: ignore[arg-type]
        avg_log = float(r["avg_log"])  # type: ignore[arg-type]
        avg_linear = float(r["avg_linear"])  # type: ignore[arg-type]
        winner = "log" if avg_log < avg_linear else "linear"
        delta = abs(avg_log - avg_linear)
        pct = (delta / min(avg_log, avg_linear)) * 100.0
        print(f"{tag}:")
        print(f"  best fixed θ={best_theta:.2f} avg WU {best_avg:.2f}")
        print(f"  bandit log-loss     avg WU {avg_log:.2f}")
        print(f"  bandit linear-loss  avg WU {avg_linear:.2f}")
        print(f"  winner: {winner} (Δ {delta:.2f} WU, {pct:.2f}%)")

    # Plots
    suffix = f"T{args.T}_trial1_S{args.S}"
    log_path = out_dir / f"loss_compare_log_{suffix}.png"
    lin_path = out_dir / f"loss_compare_linear_{suffix}.png"
    _plot_mode(results_by_tag=results, mode="log", out_path=log_path)
    _plot_mode(results_by_tag=results, mode="linear", out_path=lin_path)

    print("\nSaved plots:")
    print(log_path)
    print(lin_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
