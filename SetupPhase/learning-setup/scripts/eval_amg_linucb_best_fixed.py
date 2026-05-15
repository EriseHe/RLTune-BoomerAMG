"""
Offline evaluator: compute best fixed config in the LinUCB action set.

Reads a run directory produced by run_amg_linucb_T5000_default.py, reconstructs
each instance deterministically from (s1,s2,s3,c_diag,rhs_seed), evaluates all
action configs, and finds the best fixed config in hindsight.

Outputs (into the same run_dir):
  - best_fixed_summary.json
  - learning_with_best_fixed.png
  - rolling_regret_vs_best_fixed_ma{MA}.png
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMPI_MCA_btl", "self,sm")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solver import solve
from utils.problem_amg import build_matrix_kwargs
from utils.setup_amg import build_actions_th_coarsen_interp, moving_average, progress_bar


def safe_wu(params: Dict[str, Any], mkw: Dict[str, Any], *, fail_penalty: float) -> float:
    try:
        return float(solve(params=params, **mkw).work_units)
    except Exception:
        return float(fail_penalty)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ma", type=int, default=25)
    ap.add_argument("--fail-penalty", type=float, default=1e6)
    ap.add_argument("--rtol", type=float, default=1e-9, help="default recompute relative tolerance")
    ap.add_argument("--atol", type=float, default=1e-6, help="default recompute absolute tolerance")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    csv_path = run_dir / "log_linucb_amg.csv"
    cfg_path = run_dir / "run_config.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing log: {csv_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    cfg = json.loads(cfg_path.read_text())
    T = int(cfg["T"])
    fixed_params = dict(cfg["fixed_params"])
    default_params = dict(cfg["default_params"])
    tuned = cfg["tuned"]
    thresholds = [float(x) for x in tuned["strong_threshold"]]
    coarsen_types = [int(x) for x in tuned["coarsen_type"]]
    interp_types = [int(x) for x in tuned["interp_type"]]

    actions = build_actions_th_coarsen_interp(thresholds, coarsen_types, interp_types, fixed_params=fixed_params)
    K = len(actions)

    # Load log rows
    rows: List[Dict[str, Any]] = []
    with open(csv_path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    if len(rows) != T:
        raise RuntimeError(f"Log has {len(rows)} rows but config T={T}")

    print(f"Evaluating best-fixed over K={K} actions on T={T} instances")
    print(f"run_dir: {run_dir}")
    print()

    wu_lin = np.zeros(T, dtype=float)
    wu_def_logged = np.zeros(T, dtype=float)
    wu_def_re = np.zeros(T, dtype=float)
    wu_actions = np.zeros((T, K), dtype=float)

    # Evaluate all actions per instance
    for t, row in enumerate(rows):
        s1 = float(row["s1"])
        s2 = float(row["s2"])
        s3 = float(row["s3"])
        c_diag = float(row["c_diag"])
        rhs_seed = int(row["rhs_seed"])

        mkw = build_matrix_kwargs(s1=s1, s2=s2, s3=s3, c_diag=c_diag, rhs_seed=rhs_seed, nx=60, ny=60, nz=1, stencil=27)

        wu_lin[t] = float(row["wu_linucb"])
        wu_def_logged[t] = float(row["wu_default"])

        wu_def_re[t] = safe_wu(default_params, mkw, fail_penalty=float(args.fail_penalty))

        for k, params in enumerate(actions):
            wu_actions[t, k] = safe_wu(params, mkw, fail_penalty=float(args.fail_penalty))

        progress_bar(t + 1, T, prefix="  eval")

    # Reconstructability check: default recompute should match logged default.
    diff = np.abs(wu_def_re - wu_def_logged)
    ok = np.allclose(wu_def_re, wu_def_logged, rtol=float(args.rtol), atol=float(args.atol))
    print()
    print("=== Default recompute check ===")
    print(f"  allclose: {ok} (rtol={args.rtol}, atol={args.atol})")
    print(f"  max abs diff: {float(np.max(diff)):.6g}")
    print(f"  mean abs diff: {float(np.mean(diff)):.6g}")
    if not ok:
        print("  WARNING: default WU mismatch; instance reconstruction may be inconsistent.")

    totals = np.sum(wu_actions, axis=0)
    best_k = int(np.argmin(totals))
    best_params = actions[best_k]
    best_wu = wu_actions[:, best_k]

    # Plots
    ma = int(args.ma)
    t_axis = np.arange(1, T + 1)
    y = T - np.arange(1, T + 1)

    plt.figure(figsize=(9, 5))
    plt.plot(np.cumsum(wu_def_logged), y, linewidth=2.0, linestyle="--", label="default (table)")
    plt.plot(np.cumsum(best_wu), y, linewidth=2.0, linestyle="--", label="best fixed (action set)")
    plt.plot(np.cumsum(wu_lin), y, linewidth=2.5, color="black", label="LinUCB (3-param)")
    plt.xlabel("cumulative work units (WU)", fontsize=13)
    plt.ylabel("instances remaining", fontsize=13)
    plt.title("BoomerAMG Setup: LinUCB vs Default vs Best Fixed", fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    learning_path = run_dir / "learning_with_best_fixed.png"
    plt.savefig(learning_path, dpi=256)
    plt.close()

    regret_best = moving_average(wu_lin - best_wu, ma)
    plt.figure(figsize=(10, 4))
    plt.plot(t_axis, regret_best, linewidth=1.8)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    plt.xlabel("t", fontsize=12)
    plt.ylabel("WU regret (LinUCB - best fixed)", fontsize=12)
    plt.title(f"Rolling regret vs best fixed (MA{ma})", fontsize=12)
    plt.tight_layout()
    regret_path = run_dir / f"rolling_regret_vs_best_fixed_ma{ma}.png"
    plt.savefig(regret_path, dpi=256)
    plt.close()

    summary = {
        "T": T,
        "K": K,
        "best_fixed_index": best_k,
        "best_fixed_params": best_params,
        "total_wu_best_fixed": float(np.sum(best_wu)),
        "total_wu_default": float(np.sum(wu_def_logged)),
        "total_wu_linucb": float(np.sum(wu_lin)),
        "mean_wu_best_fixed": float(np.mean(best_wu)),
        "mean_wu_default": float(np.mean(wu_def_logged)),
        "mean_wu_linucb": float(np.mean(wu_lin)),
        "default_recompute_allclose": bool(ok),
        "default_recompute_max_abs_diff": float(np.max(diff)),
        "learning_plot": str(learning_path),
        "regret_plot": str(regret_path),
    }

    (run_dir / "best_fixed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))

    print()
    print("=== Best-fixed summary ===")
    print(f"  best fixed params: {best_params}")
    print(f"  best fixed total WU: {summary['total_wu_best_fixed']:.1f}")
    print(f"  default    total WU: {summary['total_wu_default']:.1f}")
    print(f"  LinUCB     total WU: {summary['total_wu_linucb']:.1f}")
    print(f"  outputs: {run_dir}")


if __name__ == "__main__":
    main()
