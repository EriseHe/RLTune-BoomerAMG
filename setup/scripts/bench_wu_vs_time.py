"""
Benchmark how well wall time correlates with Work Units (WU).

WU is defined as:
  WU = iterations * complexity

This script samples random instances and random configs from the action set,
times the solve, and fits a simple affine model:
  time_ms ≈ a * WU + b
reporting R^2 and Pearson correlation.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("OMPI_MCA_btl", "self,sm")

try:
    from setup.scripts import _project_paths  # noqa: E402,F401
except ModuleNotFoundError:
    import _project_paths  # type: ignore[no-redef]  # noqa: E402,F401

from hypre.bindings import solve
from problems.amg import stencil_27_laplace
from setup.utils.paths import SETUP_RESULTS_ROOT
from setup.utils.setup_amg import build_actions_th_coarsen_interp


def r2_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = y.astype(float, copy=False)
    yhat = yhat.astype(float, copy=False)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=800, help="number of (instance, config) solves to sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="output dir (default: results/setup/timing/WU_vs_time_seedX)")
    args = ap.parse_args()

    actions = build_actions_th_coarsen_interp(
        [0.10, 0.25, 0.40, 0.50, 0.70],
        [10, 8, 6],
        [6, 8, 0],
        fixed_params={"trunc_factor": 0.0, "P_max_elmts": 4, "agg_num_levels": 0, "agg_interp_type": 4},
    )
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out) if args.out else (SETUP_RESULTS_ROOT / "timing" / f"WU_vs_time_seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "wu_time_samples.csv"
    plot_path = out_dir / "wu_vs_time_scatter.png"

    WU = np.zeros(args.N, dtype=float)
    ms = np.zeros(args.N, dtype=float)

    # Warmup
    mkw0, _, _ = stencil_27_laplace(rng=rng)
    _ = solve(params=actions[0], **mkw0)

    with open(csv_path, "w", newline="") as f:
        fieldnames = ["i", "wu", "time_ms", "strong_threshold", "coarsen_type", "interp_type", "s1", "s2", "s3", "c_diag"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(args.N):
            mkw, _, meta = stencil_27_laplace(rng=rng)
            params = actions[int(rng.integers(0, len(actions)))]

            t0 = time.perf_counter_ns()
            r = solve(params=params, **mkw)
            t1 = time.perf_counter_ns()

            wu = float(r.work_units)
            dt_ms = (t1 - t0) / 1e6

            WU[i] = wu
            ms[i] = dt_ms
            w.writerow(
                {
                    "i": i,
                    "wu": wu,
                    "time_ms": dt_ms,
                    "strong_threshold": float(params["strong_threshold"]),
                    "coarsen_type": int(params["coarsen_type"]),
                    "interp_type": int(params["interp_type"]),
                    "s1": float(meta["s1"]),
                    "s2": float(meta["s2"]),
                    "s3": float(meta["s3"]),
                    "c_diag": float(meta["c_diag"]),
                }
            )

    # Fit affine model time_ms = a*WU + b
    A = np.vstack([WU, np.ones_like(WU)]).T
    a, b = np.linalg.lstsq(A, ms, rcond=None)[0]
    pred = a * WU + b
    r2 = r2_score(ms, pred)
    corr = float(np.corrcoef(WU, ms)[0, 1])

    # Plot scatter + fit
    plt.figure(figsize=(7, 5))
    plt.scatter(WU, ms, s=10, alpha=0.35, label="samples")
    xs = np.linspace(float(np.min(WU)), float(np.max(WU)), 200)
    plt.plot(xs, a * xs + b, color="black", linewidth=2, label="affine fit")
    plt.xlabel("work units (WU)", fontsize=12)
    plt.ylabel("wall time (ms)", fontsize=12)
    plt.title(f"WU vs time (N={args.N})  corr={corr:.3f}, R^2={r2:.3f}", fontsize=11)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=256)
    plt.close()

    summary = {
        "N": int(args.N),
        "seed": int(args.seed),
        "fit_time_ms": {"a_per_WU": float(a), "b_ms": float(b)},
        "pearson_corr": corr,
        "r2": float(r2),
        "csv": str(csv_path),
        "plot": str(plot_path),
    }
    (out_dir / "summary.json").write_text(str(summary))

    print("=== WU vs time benchmark ===")
    print(f"N={args.N}, seed={args.seed}")
    print(f"time_ms ≈ {a:.6f} * WU + {b:.6f}")
    print(f"corr(WU,time) = {corr:.4f}")
    print(f"R^2 = {r2:.4f}")
    print(f"outputs: {out_dir}")


if __name__ == "__main__":
    main()
