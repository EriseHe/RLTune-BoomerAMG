"""LinUCB AMG setup experiment wrapper over the shared core interface."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.setup_amg import (
    build_actions_th_coarsen_interp,
    build_actions_th_mxrs_tr,
    moving_average,
    run_amg_setup_experiment,
    save_logs,
    summarize_totals,
)
from utils.problem_amg import NX, NY, NZ, STENCIL, stencil_27_laplace
from learners.LinUCB_AMG import LinUCB_AMG


# ============================================================================
# LinUCB Policy Wrapper
# ============================================================================

class _LinUCBTrialPolicy:
    """Wrapper to adapt LinUCB_AMG to the experiment interface."""
    
    def __init__(self, actions: Sequence[Dict[str, Any]], context_dim: int, alpha: float, l2_reg: float, seed: int):
        self.model = LinUCB_AMG(actions, context_dim=context_dim, alpha=alpha, l2_reg=l2_reg, seed=seed)

    def select(self, context: np.ndarray, **_):
        params = self.model.predict(context)
        step = self.model.history[-1]
        info = {"pred_mean": step.pred_mean, "pred_uncert": step.pred_uncert}
        return params, info

    def update(self, loss: float, **_):
        self.model.update(loss)


class LinUCBBanditFactory:
    """Factory for creating LinUCB policies per trial."""
    
    def __init__(self, alpha: float, l2_reg: float):
        self.alpha = alpha
        self.l2_reg = l2_reg

    def new_trial(self, parameter_space: Dict[str, Any], seed: int, **_):
        return _LinUCBTrialPolicy(
            actions=parameter_space["actions"],
            context_dim=parameter_space.get("context_dim", 5),
            alpha=self.alpha,
            l2_reg=self.l2_reg,
            seed=seed,
        )


# ============================================================================
# Loss Functions
# ============================================================================

def runtime_loss(outcome: Dict[str, float], **_) -> float:
    return outcome["runtime"]


def wu_loss(outcome: Dict[str, float], **_) -> float:
    return outcome["wu"]


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_learning_curve(result: Dict[str, Any], metric: str, output_path: Path, loss_type: str) -> None:
    """Plot cumulative metric vs instances remaining."""
    base_arr = result[f"baseline_{metric}"]
    bandit_arr = result[f"bandit_{metric}"]
    
    xlabel = "cumulative work units (WU)" if metric == "wu" else "cumulative runtime (seconds)"
    title = f"BoomerAMG Setup: LinUCB (3-param) vs Fixed Baselines [loss={loss_type}]"
    if metric == "runtime":
        title = f"BoomerAMG Setup: runtime view [loss={loss_type}]"
    
    plt.figure(figsize=(9, 5))
    
    # Plot baselines
    for i, name in enumerate(result["baseline_names"]):
        curve = np.mean(np.cumsum(base_arr[:, :, i], axis=0), axis=1)
        plt.plot(curve, result["T"] - np.arange(1, result["T"] + 1), 
                linewidth=2, linestyle="--", label=name)
    
    # Plot bandit
    bandit_curve = np.mean(np.cumsum(bandit_arr, axis=0), axis=1)
    plt.plot(bandit_curve, result["T"] - np.arange(1, result["T"] + 1), 
            linewidth=2.5, color="black", label="LinUCB (3-param)")
    
    plt.xlabel(xlabel, fontsize=13)
    plt.ylabel("instances remaining", fontsize=13)
    plt.title(title, fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=256)
    plt.close()


def plot_parameter_trace(logs: List[Dict[str, Any]], trace_params: List[str], 
                        ma_window: int, output_path: Path) -> None:
    """Plot chosen parameters over time with moving average."""
    if not logs:
        return
    
    t_axis = np.arange(1, len(logs) + 1)
    
    fig, axes = plt.subplots(len(trace_params), 1, 
                            figsize=(10, 2.2 * len(trace_params)), 
                            sharex=True)
    if len(trace_params) == 1:
        axes = [axes]
    
    for i, pname in enumerate(trace_params):
        values = np.array([r.get(f"chosen_{pname}", np.nan) for r in logs])
        axes[i].plot(t_axis, moving_average(values, ma_window), linewidth=1.5)
        axes[i].set_ylabel(pname, fontsize=10)
    
    axes[0].set_title(f"LinUCB chosen parameters (MA{ma_window})", fontsize=12)
    axes[-1].set_xlabel("t", fontsize=11)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=256)
    plt.close(fig)


def plot_regret(result: Dict[str, Any], metric: str, ma_window: int, output_path: Path) -> None:
    """Plot rolling regret vs best baseline."""
    bandit_arr = result[f"bandit_{metric}"]
    baseline_arr = result[f"baseline_{metric}"]
    
    # Compute regret: bandit - best baseline
    regret = bandit_arr - np.min(baseline_arr, axis=2)
    regret_ma = moving_average(np.mean(regret, axis=1), ma_window)
    
    ylabel = "runtime regret vs best baseline (sec)" if metric == "runtime" else "WU regret vs best baseline"
    
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(1, result["T"] + 1), regret_ma, linewidth=1.8)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("t", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f"Rolling regret (MA{ma_window})", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=256)
    plt.close()


# ============================================================================
# Main Experiment
# ============================================================================

def main(argv: List[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--ma", type=int, default=25, help="moving-average window")
    ap.add_argument("--fail-penalty", type=float, default=1e6)
    ap.add_argument("--tune", type=str, default="th_coarsen_interp",
                   choices=["th_coarsen_interp", "th_mxrs_tr"])
    ap.add_argument("--loss", type=str, default="wu", choices=["wu", "runtime"])
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args(argv)

    # ========================================================================
    # Configure experiment based on tune mode
    # ========================================================================
    thresholds = [0.10, 0.25, 0.40, 0.50, 0.70]
    
    if args.tune == "th_mxrs_tr":
        actions = build_actions_th_mxrs_tr(
            thresholds, [0.10, 0.50, 0.90], [0.00, 0.10, 0.20],
            fixed_params={"coarsen_type": 10, "interp_type": 6},
        )
        baselines = [
            ("default-ish", {"strong_threshold": 0.25, "max_row_sum": 0.90, "trunc_factor": 0.00, "coarsen_type": 10, "interp_type": 6}),
            ("th=0.10", {"strong_threshold": 0.10, "max_row_sum": 0.90, "trunc_factor": 0.00, "coarsen_type": 10, "interp_type": 6}),
            ("th=0.40", {"strong_threshold": 0.40, "max_row_sum": 0.90, "trunc_factor": 0.00, "coarsen_type": 10, "interp_type": 6}),
            ("mxrs=0.50", {"strong_threshold": 0.25, "max_row_sum": 0.50, "trunc_factor": 0.00, "coarsen_type": 10, "interp_type": 6}),
            ("tr=0.10", {"strong_threshold": 0.25, "max_row_sum": 0.90, "trunc_factor": 0.10, "coarsen_type": 10, "interp_type": 6}),
            ("mxrs=0.50,tr=0.10", {"strong_threshold": 0.25, "max_row_sum": 0.50, "trunc_factor": 0.10, "coarsen_type": 10, "interp_type": 6}),
        ]
        trace_params = ["strong_threshold", "max_row_sum", "trunc_factor"]
    else:
        actions = build_actions_th_coarsen_interp(thresholds, [10, 8, 6], [6, 8, 0])
        baselines = [
            ("default-ish", {"strong_threshold": 0.25, "coarsen_type": 10, "interp_type": 6}),
            ("th=0.10", {"strong_threshold": 0.10, "coarsen_type": 10, "interp_type": 6}),
            ("th=0.40", {"strong_threshold": 0.40, "coarsen_type": 10, "interp_type": 6}),
            ("coarsen=8", {"strong_threshold": 0.25, "coarsen_type": 8, "interp_type": 6}),
            ("interp=8", {"strong_threshold": 0.25, "coarsen_type": 10, "interp_type": 8}),
            ("coarsen=8,interp=8", {"strong_threshold": 0.25, "coarsen_type": 8, "interp_type": 8}),
        ]
        trace_params = ["strong_threshold", "coarsen_type", "interp_type"]

    parameter_space = {"actions": actions, "context_dim": 5}
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).parent.parent / "plots" / "LinUCB_AMG"
    loss_fn = runtime_loss if args.loss == "runtime" else wu_loss

    # ========================================================================
    # Print experiment info
    # ========================================================================
    print(f"Matrix: {NX}x{NY}x{NZ}, {STENCIL}pt stencil (randomized each round)")
    print(f"T={args.T}, trials={args.trials}, seed={args.seed}")
    print(f"LinUCB: alpha={args.alpha}, l2={args.l2}, actions={len(actions)}")
    print(f"Bandit update loss: {args.loss}")
    print(f"Tune set: {args.tune}")
    print()

    # ========================================================================
    # Run experiment
    # ========================================================================
    result = run_amg_setup_experiment(
        stencil_27_laplace,
        parameter_space,
        LinUCBBanditFactory(args.alpha, args.l2),
        loss_fn,
        T=args.T,
        trials=args.trials,
        seed=args.seed,
        baselines=baselines,
        fail_penalty=args.fail_penalty,
    )

    # ========================================================================
    # Save results
    # ========================================================================
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_logs(result["logs"], out_dir / "log_linucb_amg.csv")
    
    plot_learning_curve(result, "wu", out_dir / "learning_linucb_amg.png", args.loss)
    plot_learning_curve(result, "runtime", out_dir / "learning_linucb_amg_runtime.png", args.loss)
    plot_parameter_trace(result["logs"], trace_params, args.ma, out_dir / "trace_linucb_amg.png")
    plot_regret(result, args.loss, args.ma, out_dir / "rolling_regret_linucb_amg.png")

    # ========================================================================
    # Print summary
    # ========================================================================
    wu_totals = summarize_totals(result, "wu")
    rt_totals = summarize_totals(result, "runtime")
    
    print("=== Summary ===")
    print("Mean total WU over all rounds:")
    for name in result["baseline_names"]:
        print(f"  {name:<20} total WU = {wu_totals[name]:.1f}")
    print(f"  {'LinUCB (3-param)':<20} total WU = {wu_totals['bandit']:.1f}")
    
    print("Mean total runtime over all rounds:")
    for name in result["baseline_names"]:
        print(f"  {name:<20} total sec = {rt_totals[name]:.3f}")
    print(f"  {'LinUCB (3-param)':<20} total sec = {rt_totals['bandit']:.3f}")
    
    print()
    print(f"Plots saved under: {out_dir}")


if __name__ == "__main__":
    main()
