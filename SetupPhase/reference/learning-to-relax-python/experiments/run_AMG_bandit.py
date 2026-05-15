"""
Setup-phase bandit experiment: compare bandit loss transforms for BoomerAMG.

Reproduces the A/B test:
- Same exact sequence of linear system instances (At, bt)
- Same baseline θ comparisons
- Multiple Tsallis-INF bandits that differ only in the update loss:

  Example:
    (1) log loss:     loss = 1 + log(1 + WU / knee)
    (2) linear loss:  loss = 1 + (WU / knee)

where WU = iterations * cum_nnz_AP.

Outputs (default directory: `plots/BoomerAMG/`):
- loss_compare_<loss_name>_T{T}_trial1_S{S}.png
- loss_compare_<loss_name>_T{T}_trial1_S{S}_high.png  (if running only high variance)
- loss_compare_<loss_name>_T{T}_trial1_S{S}_low.png   (if running only low variance)

Optional:
- `--log` writes per-instance CSVs: loss_compare_<tag>_T{T}_trial1_S{S}.csv
- `--diagnostics` writes learning diagnostics plots:
    - theta_trace_T{T}_trial1_S{S}.png
    - rolling_regret_T{T}_trial1_S{S}.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Tuple

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


LossFn = Callable[[float, float], float]


# ============================================================================
# Loss functions (EDIT ME)
# ----------------------------------------------------------------------------
# Define your loss directly here. Each function maps (WU, knee) -> loss scalar
# passed to TsallisINF_AMG.update(loss).
#
# Notes:
# - WU is still the metric we report/plot (WU = iterations * cum_nnz_AP).
# - The bandit only sees the returned `loss`.
# ============================================================================
def loss_log(wu: float, knee: float) -> float:
    return 1.0 + float(np.log1p(wu / knee))


def loss_linear(wu: float, knee: float) -> float:
    return 1.0 + float(wu / knee)


def loss_raw(wu: float, knee: float) -> float:
    return 1.0 + float(wu)


LOSS_FUNCTIONS: Dict[str, LossFn] = {
    "log": loss_log,
    "linear": loss_linear,
    "raw": loss_raw,
}


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


def _progress(t: int, T: int, start_time: float, tag: str, every: int) -> None:
    if every > 0 and ((t + 1) % every == 0 or (t + 1) == T):
        mins = (time.time() - start_time) / 60.0
        print(f"{tag}: {t+1}/{T} ({mins:.1f} min)")


def _parse_grid(spec: str) -> np.ndarray:
    spec = spec.strip()
    if ":" in spec:
        gmin, gmax, gcount = spec.split(":")
        return np.linspace(float(gmin), float(gmax), int(gcount))
    # Allow an explicit list as well (e.g., "0.1,0.25,0.4,0.5,0.7")
    return np.array([float(x.strip()) for x in spec.split(",") if x.strip()], dtype=np.float64)


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
    knee: float,
    loss_functions: Mapping[str, LossFn],
    rng_env: np.random.Generator,
    bandit_base_state,
    progress_every: int,
) -> Dict[str, object]:
    from learners import TsallisINF_AMG
    from solvers.BoomerAMG import boomeramg

    baseline_wu = np.zeros((T, len(baselines)), dtype=np.float64)
    baseline_k = np.zeros((T, len(baselines)), dtype=np.float64)
    baseline_comp = np.zeros((T, len(baselines)), dtype=np.float64)
    diag_shift = np.zeros(T, dtype=np.float64)

    loss_names = list(loss_functions.keys())
    bandits = {name: TsallisINF_AMG(bandit_grid, T) for name in loss_names}
    states = {name: bandit_base_state for name in loss_names}

    bandit_theta = {name: np.zeros(T, dtype=np.float64) for name in loss_names}
    bandit_k = {name: np.zeros(T, dtype=np.float64) for name in loss_names}
    bandit_comp = {name: np.zeros(T, dtype=np.float64) for name in loss_names}
    bandit_wu = {name: np.zeros(T, dtype=np.float64) for name in loss_names}
    bandit_loss = {name: np.zeros(T, dtype=np.float64) for name in loss_names}

    start = time.time()
    for t in range(T):
        # Environment instance (independent of bandit randomness):
        c = -0.15 + 0.6 * rng_env.beta(beta_a, beta_b)
        diag_shift[t] = c
        At = A + c * I
        bt = _truncated_normal_rng(rng_env, n)
        x0 = np.zeros(n)

        # Baselines (computed once; reused for both loss-mode comparisons)
        for i, base_theta in enumerate(baselines):
            k, comp, _ = boomeramg(At, bt, x0, float(base_theta), epsilon)
            baseline_k[t, i] = float(k)
            baseline_comp[t, i] = float(comp)
            baseline_wu[t, i] = float(k) * float(comp)

        # Bandits (only difference is the loss function used in update()).
        for name in loss_names:
            np.random.set_state(states[name])
            theta = float(bandits[name].predict())
            k, comp, _ = boomeramg(At, bt, x0, theta, epsilon)
            WU = float(k) * float(comp)
            loss = float(loss_functions[name](WU, knee))

            bandit_theta[name][t] = theta
            bandit_k[name][t] = float(k)
            bandit_comp[name][t] = float(comp)
            bandit_wu[name][t] = WU
            bandit_loss[name][t] = loss

            bandits[name].update(loss)
            states[name] = np.random.get_state()

        _progress(t, T, start, tag, every=progress_every)

    avg_baselines = baseline_wu.mean(axis=0)
    best_idx = int(np.argmin(avg_baselines))

    avg_by_loss = {name: float(bandit_wu[name].mean()) for name in loss_names}

    return {
        "tag": tag,
        "beta": (beta_a, beta_b),
        "baselines": baselines,
        "baseline_wu": baseline_wu,
        "baseline_k": baseline_k,
        "baseline_comp": baseline_comp,
        "diag_shift": diag_shift,
        "best_theta": float(baselines[best_idx]),
        "best_avg": float(avg_baselines[best_idx]),
        "best_idx": best_idx,
        "loss_names": loss_names,
        "bandit_theta": bandit_theta,
        "bandit_k": bandit_k,
        "bandit_comp": bandit_comp,
        "bandit_wu": bandit_wu,
        "bandit_loss": bandit_loss,
        "avg_by_loss": avg_by_loss,
    }


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


def _plot_theta_trace(*, results_by_tag: Dict[str, Dict[str, object]], out_path: Path, window: int) -> None:
    tags = list(results_by_tag.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(8 * len(tags), 4), sharey=True)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        r = results_by_tag[tag]
        loss_names = list(r["loss_names"])  # type: ignore[assignment]
        bandit_theta = r["bandit_theta"]  # type: ignore[assignment]
        T = int(len(r["diag_shift"]))  # type: ignore[arg-type]
        t = np.arange(1, T + 1)

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(loss_names))))
        for i, name in enumerate(loss_names):
            theta = np.asarray(bandit_theta[name], dtype=np.float64)
            ax.plot(t, _moving_average(theta, window), linewidth=2, color=colors[i], label=name)

        best_theta = float(r["best_theta"])  # type: ignore[arg-type]
        ax.axhline(best_theta, linestyle="--", color="black", alpha=0.35, linewidth=1.5, label="best fixed baseline")

        ax.set_title(f"{tag.replace('_', ' ')} θ(t) (MA{window})", fontsize=13)
        ax.set_xlabel("t", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10, loc="upper right")

    axes[0].set_ylabel("θ", fontsize=12)
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
        baseline_wu = np.asarray(r["baseline_wu"], dtype=np.float64)  # type: ignore[arg-type]
        best_idx = int(r["best_idx"])  # type: ignore[arg-type]
        best_series = baseline_wu[:, best_idx]

        loss_names = list(r["loss_names"])  # type: ignore[assignment]
        bandit_wu = r["bandit_wu"]  # type: ignore[assignment]

        T = int(best_series.size)
        t = np.arange(1, T + 1)

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(loss_names))))
        for i, name in enumerate(loss_names):
            wu = np.asarray(bandit_wu[name], dtype=np.float64)
            regret = wu - best_series
            ax.plot(t, _moving_average(regret, window), linewidth=2, color=colors[i], label=name)

        ax.axhline(0.0, linestyle="--", color="black", alpha=0.35, linewidth=1.5)
        ax.set_title(f"{tag.replace('_', ' ')} regret vs best fixed (MA{window})", fontsize=13)
        ax.set_xlabel("t", fontsize=12)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10, loc="upper right")

    axes[0].set_ylabel("WU regret", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def _plot_loss(*, results_by_tag: Dict[str, Dict[str, object]], loss_name: str, out_path: Path) -> None:
    tags = list(results_by_tag.keys())
    fig, axes = plt.subplots(1, len(tags), figsize=(8 * len(tags), 6), sharey=True)
    if len(tags) == 1:
        axes = [axes]

    for ax, tag in zip(axes, tags):
        r = results_by_tag[tag]
        baselines = r["baselines"]  # type: ignore[assignment]
        baseline_wu = r["baseline_wu"]  # type: ignore[assignment]
        T = int(baseline_wu.shape[0])
        y = T - np.arange(1, T + 1)

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

        bandit_wu = r["bandit_wu"][loss_name]  # type: ignore[index]
        ax.plot(
            np.cumsum(bandit_wu),
            y,
            color="black",
            linewidth=3,
            label=f"Tsallis-INF ({loss_name})",
        )
        title = f"{tag.replace('_', ' ')} ({loss_name})"

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Total Work Units (WU)", fontsize=13)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=10, loc="upper right")

    axes[0].set_ylabel("Instances Remaining", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=256)
    plt.close(fig)


def _write_log_csv(*, result: Dict[str, object], out_path: Path, baselines: np.ndarray, loss_names: Iterable[str]) -> None:
    diag_shift = result["diag_shift"]  # type: ignore[assignment]
    baseline_wu = result["baseline_wu"]  # type: ignore[assignment]
    baseline_k = result["baseline_k"]  # type: ignore[assignment]
    baseline_comp = result["baseline_comp"]  # type: ignore[assignment]

    bandit_theta = result["bandit_theta"]  # type: ignore[assignment]
    bandit_k = result["bandit_k"]  # type: ignore[assignment]
    bandit_comp = result["bandit_comp"]  # type: ignore[assignment]
    bandit_wu = result["bandit_wu"]  # type: ignore[assignment]
    bandit_loss = result["bandit_loss"]  # type: ignore[assignment]

    fieldnames = ["instance", "diagonal_shift"]
    for th in baselines:
        th_key = f"{float(th):.2f}"
        fieldnames += [
            f"baseline_theta_{th_key}_iterations",
            f"baseline_theta_{th_key}_cum_nnz_AP",
            f"baseline_theta_{th_key}_WU",
        ]
    for name in loss_names:
        fieldnames += [
            f"bandit_{name}_theta",
            f"bandit_{name}_iterations",
            f"bandit_{name}_cum_nnz_AP",
            f"bandit_{name}_WU",
            f"bandit_{name}_loss",
        ]

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        T = int(len(diag_shift))
        for t in range(T):
            row: Dict[str, float] = {"instance": float(t + 1), "diagonal_shift": float(diag_shift[t])}
            for i, th in enumerate(baselines):
                th_key = f"{float(th):.2f}"
                row[f"baseline_theta_{th_key}_iterations"] = float(baseline_k[t, i])
                row[f"baseline_theta_{th_key}_cum_nnz_AP"] = float(baseline_comp[t, i])
                row[f"baseline_theta_{th_key}_WU"] = float(baseline_wu[t, i])
            for name in loss_names:
                row[f"bandit_{name}_theta"] = float(bandit_theta[name][t])
                row[f"bandit_{name}_iterations"] = float(bandit_k[name][t])
                row[f"bandit_{name}_cum_nnz_AP"] = float(bandit_comp[name][t])
                row[f"bandit_{name}_WU"] = float(bandit_wu[name][t])
                row[f"bandit_{name}_loss"] = float(bandit_loss[name][t])
            writer.writerow(row)


def run_compare(
    *,
    T: int = 5000,
    S: int = 20,
    epsilon: float = 1e-8,
    knee: float = 40.0,
    variance: str = "both",
    theta_grid: str = "0.1:0.9:9",
    baselines: str = "0.1,0.25,0.4,0.5,0.7",
    out_dir: str | None = None,
    seed: int | None = 0,
    losses: Mapping[str, LossFn] | None = None,
    write_log: bool = False,
    diagnostics: bool = False,
    ma_window: int = 25,
    progress_every: int = 100,
    hypre_lib: str | None = None,
) -> Tuple[Dict[str, Dict[str, object]], Dict[str, Path], Dict[str, Path]]:
    """
    High-level interface: run the setup-phase BoomerAMG bandit experiment.

    Returns:
      - results_by_tag: per-variance results
      - plot_paths_by_loss: plot path for each loss name (plus diagnostics keys if enabled)
      - log_paths_by_tag: csv path for each variance tag (if write_log=True)
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if knee <= 0:
        raise ValueError("knee must be positive")
    if variance not in {"high", "low", "both"}:
        raise ValueError("variance must be one of: high, low, both")

    _ensure_hypre_env(hypre_lib)
    _ensure_mpi_env()

    py_root = _py_root()
    sys.path.insert(0, str(py_root))

    from utils import delsq, numgrid

    A = delsq(numgrid("S", S))
    n = int(A.shape[0])
    I = speye(n, format="csr")

    baseline_thetas = _parse_list(baselines)
    bandit_grid = _parse_grid(theta_grid)

    output_dir = Path(out_dir) if out_dir else (py_root / "plots" / "BoomerAMG")
    output_dir.mkdir(parents=True, exist_ok=True)

    active_losses = dict(losses) if losses is not None else dict(LOSS_FUNCTIONS)
    if not active_losses:
        raise ValueError("No losses selected")

    # Optional: set a single seed for both environment + bandit sampling.
    rng_high = np.random.default_rng(seed if seed is not None else None)
    rng_low = np.random.default_rng((seed + 1) if seed is not None else None)
    if seed is not None:
        np.random.seed(seed)
    base_state = np.random.get_state()

    results_by_tag: Dict[str, Dict[str, object]] = {}
    if variance in {"high", "both"}:
        results_by_tag["high_variance"] = _run_one_variance(
            A=A,
            I=I,
            n=n,
            T=T,
            epsilon=epsilon,
            baselines=baseline_thetas,
            bandit_grid=bandit_grid,
            beta_a=0.5,
            beta_b=1.5,
            tag="high_variance",
            knee=knee,
            loss_functions=active_losses,
            rng_env=rng_high,
            bandit_base_state=base_state,
            progress_every=progress_every,
        )
    if variance in {"low", "both"}:
        results_by_tag["low_variance"] = _run_one_variance(
            A=A,
            I=I,
            n=n,
            T=T,
            epsilon=epsilon,
            baselines=baseline_thetas,
            bandit_grid=bandit_grid,
            beta_a=2.0,
            beta_b=6.0,
            tag="low_variance",
            knee=knee,
            loss_functions=active_losses,
            rng_env=rng_low,
            bandit_base_state=base_state,
            progress_every=progress_every,
        )

    # Plots (one file per loss; each file contains subplots for the chosen variances)
    suffix = f"T{T}_trial1_S{S}"
    if variance == "high":
        suffix += "_high"
    elif variance == "low":
        suffix += "_low"

    plot_paths_by_loss: Dict[str, Path] = {}
    for name in active_losses.keys():
        plot_path = output_dir / f"loss_compare_{name}_{suffix}.png"
        _plot_loss(results_by_tag=results_by_tag, loss_name=name, out_path=plot_path)
        plot_paths_by_loss[name] = plot_path

    if diagnostics:
        if ma_window <= 0:
            raise ValueError("ma_window must be positive")
        theta_trace_path = output_dir / f"theta_trace_{suffix}.png"
        rolling_regret_path = output_dir / f"rolling_regret_{suffix}.png"
        _plot_theta_trace(results_by_tag=results_by_tag, out_path=theta_trace_path, window=int(ma_window))
        _plot_rolling_regret(results_by_tag=results_by_tag, out_path=rolling_regret_path, window=int(ma_window))
        plot_paths_by_loss["theta_trace"] = theta_trace_path
        plot_paths_by_loss["rolling_regret"] = rolling_regret_path

    # Optional logs (one CSV per variance distribution)
    log_paths_by_tag: Dict[str, Path] = {}
    if write_log:
        for tag, r in results_by_tag.items():
            log_path = output_dir / f"loss_compare_{tag}_{suffix}.csv"
            _write_log_csv(result=r, out_path=log_path, baselines=baseline_thetas, loss_names=active_losses.keys())
            log_paths_by_tag[tag] = log_path

    return results_by_tag, plot_paths_by_loss, log_paths_by_tag


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=5000, help="Number of instances")
    parser.add_argument("--S", type=int, default=20, help="numgrid('S', S) size (matrix size grows with S)")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="Solver tolerance")
    parser.add_argument("--knee", type=float, default=40.0, help="Scale parameter used by loss functions")
    parser.add_argument("--seed", type=int, default=0, help="Seed for env + bandit RNG (set -1 for random)")
    parser.add_argument(
        "--variance",
        choices=["high", "low", "both"],
        default="both",
        help="Which offset distribution(s) to run",
    )
    parser.add_argument(
        "--theta-grid",
        type=str,
        default="0.1:0.9:9",
        help="Bandit grid as 'min:max:count' or comma list (e.g. '0.1,0.25,0.4')",
    )
    parser.add_argument("--baselines", type=str, default="0.1,0.25,0.4,0.5,0.7", help="Baseline θ list")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for plots/logs")
    parser.add_argument("--log", action="store_true", help="Write per-instance CSV logs")
    parser.add_argument("--diagnostics", action="store_true", help="Write diagnostics plots (theta trace + rolling regret)")
    parser.add_argument("--ma-window", type=int, default=25, help="Moving-average window for diagnostics plots")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N instances (0 disables)")
    parser.add_argument("--hypre-lib", type=str, default=None, help="Path to libHYPRE (overrides HYPRE_LIBHYPRE)")
    args = parser.parse_args(argv)

    seed = None if args.seed < 0 else int(args.seed)

    results_by_tag, plot_paths_by_loss, log_paths_by_tag = run_compare(
        T=args.T,
        S=args.S,
        epsilon=args.epsilon,
        knee=args.knee,
        variance=args.variance,
        theta_grid=args.theta_grid,
        baselines=args.baselines,
        out_dir=args.out_dir,
        seed=seed,
        write_log=args.log,
        diagnostics=args.diagnostics,
        ma_window=args.ma_window,
        progress_every=args.progress_every,
        hypre_lib=args.hypre_lib,
    )

    # Summary
    print("\n=== Summary (avg WU) ===")
    for tag, r in results_by_tag.items():
        best_theta = float(r["best_theta"])  # type: ignore[arg-type]
        best_avg = float(r["best_avg"])  # type: ignore[arg-type]
        avg_by_loss = r["avg_by_loss"]  # type: ignore[assignment]
        print(f"{tag}:")
        print(f"  best fixed θ={best_theta:.2f} avg WU {best_avg:.2f}")
        for name, avg in avg_by_loss.items():
            print(f"  bandit {name:<10s} avg WU {float(avg):.2f}")

    print("\nSaved plots:")
    for name, p in plot_paths_by_loss.items():
        print(f"  {name}: {p}")
    if log_paths_by_tag:
        print("\nSaved logs:")
        for tag, p in log_paths_by_tag.items():
            print(f"  {tag}: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
