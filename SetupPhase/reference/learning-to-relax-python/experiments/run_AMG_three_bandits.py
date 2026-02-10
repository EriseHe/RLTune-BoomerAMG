"""
BoomerAMG head-to-head on the run_AMG-style matrix stream (A + c I):
- TsallisINF_AMG
- LinUCB (contextual, 1-D context from diagonal shift c)

This script keeps both bandits on the exact same instance sequence.
It does not modify existing experiment files.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np

_mpl_cache = Path(tempfile.gettempdir()) / "matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.run_AMG_bandit import _ensure_hypre_env, _ensure_mpi_env
from learners.TsallisINF_AMG import TsallisINF_AMG
from solvers.BoomerAMG import boomeramg
from utils import delsq, numgrid, truncated_normal


def _load_linucb_class():
    repo_root = Path(__file__).resolve().parents[3]
    linucb_path = repo_root / "SetupPhase" / "bandits-for-setup" / "learners" / "LinUCB_AMG.py"
    if not linucb_path.exists():
        raise FileNotFoundError(f"LinUCB learner not found: {linucb_path}")
    spec = importlib.util.spec_from_file_location("linucb_amg_external", str(linucb_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load LinUCB module from {linucb_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LinUCB_AMG


def _progress_bar(current: int, total: int, prefix: str = "", every: int | None = None) -> None:
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


def _loss_raw(wu: float) -> float:
    return 1.0 + float(wu)


def _sample_instances(*, A, I, n: int, T: int, beta_a: float, beta_b: float, seed: int):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(T):
        c = -0.15 + 0.6 * float(rng.beta(beta_a, beta_b))
        At = A + c * I
        bt = truncated_normal(n)
        out.append((At, bt, float(c)))
    return out


def _solve_wu(At, bt: np.ndarray, theta: float, epsilon: float) -> float:
    k, comp, _ = boomeramg(At, bt, np.zeros_like(bt), float(theta), epsilon)
    return float(k) * float(comp)


def run(
    *,
    T: int,
    S: int,
    epsilon: float,
    seed: int,
    variance: str,
    out_dir: Path,
    alpha: float,
    l2: float,
) -> Path:
    _ensure_hypre_env(None)
    _ensure_mpi_env()

    if variance == "high":
        beta_a, beta_b = 0.5, 1.5
    elif variance == "low":
        beta_a, beta_b = 2.0, 6.0
    else:
        raise ValueError("variance must be 'high' or 'low'")

    A = delsq(numgrid("S", S))
    n = A.shape[0]
    I = speye(n, format="csr")
    print(f"Matrix from run_AMG style: delsq(numgrid('S',{S})) -> n={n}")
    print(f"Variance mode: {variance} (Beta({beta_a}, {beta_b}))")

    baselines = np.array([0.1, 0.25, 0.4, 0.5, 0.7], dtype=np.float64)
    bandit_grid = np.linspace(0.1, 0.9, 9)

    instances = _sample_instances(A=A, I=I, n=n, T=T, beta_a=beta_a, beta_b=beta_b, seed=seed)

    ts_amg = TsallisINF_AMG(bandit_grid, T)

    LinUCB_AMG = _load_linucb_class()
    lin_actions = [{"strong_threshold": float(v)} for v in bandit_grid]
    linucb = LinUCB_AMG(lin_actions, context_dim=2, alpha=float(alpha), l2_reg=float(l2), seed=seed + 123)

    baseline_wu = np.zeros((T, len(baselines)), dtype=np.float64)
    amg_wu = np.zeros(T, dtype=np.float64)
    lin_wu = np.zeros(T, dtype=np.float64)

    state_amg = np.random.RandomState(seed + 7).get_state()

    for t, (At, bt, c) in enumerate(instances):
        for i, th in enumerate(baselines):
            baseline_wu[t, i] = _solve_wu(At, bt, float(th), epsilon)

        np.random.set_state(state_amg)
        th_amg = float(ts_amg.predict())
        wu_amg = _solve_wu(At, bt, th_amg, epsilon)
        amg_wu[t] = wu_amg
        ts_amg.update(_loss_raw(wu_amg))
        state_amg = np.random.get_state()

        x_ctx = np.array([1.0, c], dtype=np.float64)
        act = linucb.predict(x_ctx)
        th_lin = float(act["strong_threshold"])
        wu_lin = _solve_wu(At, bt, th_lin, epsilon)
        lin_wu[t] = wu_lin
        linucb.update(_loss_raw(wu_lin))

        _progress_bar(t + 1, T, prefix="  run")

    out_dir.mkdir(parents=True, exist_ok=True)
    avg_base = baseline_wu.mean(axis=0)
    best_idx = int(np.argmin(avg_base))
    y = T - np.arange(1, T + 1)

    # Plot 1: TsallisINF_AMG vs fixed baselines.
    plt.figure(figsize=(10, 6))
    for i, th in enumerate(baselines):
        is_default = abs(th - 0.25) < 1e-12
        is_best = i == best_idx
        suffix = ""
        if is_default:
            suffix += " (default)"
        if is_best:
            suffix += " (best fixed)"
        label = f"θ={th:.2f}{suffix}"
        plt.plot(np.cumsum(baseline_wu[:, i]), y, linewidth=2, linestyle="--", label=label)
    plt.plot(np.cumsum(amg_wu), y, linewidth=2.8, color="black", label="TsallisINF_AMG (raw)")
    plt.xlabel("Total Work Units (WU)", fontsize=14)
    plt.ylabel("Instances Remaining", fontsize=14)
    plt.title(f"BoomerAMG ({variance} variance): TsallisINF_AMG vs fixed baselines, S={S}, T={T}", fontsize=13)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    out_path_amg = out_dir / f"head2head_tsallis_amg_{variance}_T{T}_S{S}_seed{seed}.png"
    plt.savefig(out_path_amg, dpi=220)
    plt.close()

    # Plot 2: LinUCB vs fixed baselines.
    plt.figure(figsize=(10, 6))
    for i, th in enumerate(baselines):
        is_default = abs(th - 0.25) < 1e-12
        is_best = i == best_idx
        suffix = ""
        if is_default:
            suffix += " (default)"
        if is_best:
            suffix += " (best fixed)"
        label = f"θ={th:.2f}{suffix}"
        plt.plot(np.cumsum(baseline_wu[:, i]), y, linewidth=2, linestyle="--", label=label)
    plt.plot(np.cumsum(lin_wu), y, linewidth=2.8, color="black", label="LinUCB (raw, x=[1,c])")
    plt.xlabel("Total Work Units (WU)", fontsize=14)
    plt.ylabel("Instances Remaining", fontsize=14)
    plt.title(f"BoomerAMG ({variance} variance): LinUCB vs fixed baselines, S={S}, T={T}", fontsize=13)
    plt.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    out_path_lin = out_dir / f"head2head_linucb_{variance}_T{T}_S{S}_seed{seed}.png"
    plt.savefig(out_path_lin, dpi=220)
    plt.close()

    print("\n=== Summary ===")
    print(f"best fixed baseline θ={baselines[best_idx]:.2f}, avg WU={avg_base[best_idx]:.6f}")
    print(f"TsallisINF_AMG avg WU: {float(np.mean(amg_wu)):.6f}")
    print(f"LinUCB avg WU: {float(np.mean(lin_wu)):.6f}")
    print(f"Saved plot (TsallisINF_AMG): {out_path_amg.resolve()}")
    print(f"Saved plot (LinUCB): {out_path_lin.resolve()}")
    return out_path_amg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--S", type=int, default=20)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=20260209)
    parser.add_argument("--variance", type=str, default="high", choices=["high", "low"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("plots/BoomerAMG"))
    args = parser.parse_args()

    run(
        T=int(args.T),
        S=int(args.S),
        epsilon=float(args.epsilon),
        seed=int(args.seed),
        variance=str(args.variance),
        out_dir=Path(args.out_dir),
        alpha=float(args.alpha),
        l2=float(args.l2),
    )


if __name__ == "__main__":
    main()
