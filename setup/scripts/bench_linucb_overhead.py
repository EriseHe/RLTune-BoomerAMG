"""
Benchmark the overhead of bandit decision-making (predict+update) vs solve time.

This measures:
  - average time per bandit predict+update step (no solver calls)
  - average time for one BoomerAMG solve on the same instance distribution

Note: this is *Python-side* overhead only. The AMG solve is dominated by C/HYPRE.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

# Reduce noisy OpenMPI TCP binding warnings on macOS environments.
os.environ.setdefault("OMPI_MCA_btl", "self,sm")

try:
    from setup.scripts import _project_paths  # noqa: E402,F401
except ModuleNotFoundError:
    import _project_paths  # type: ignore[no-redef]  # noqa: E402,F401

from setup.learners import LinUCB_AMG, TsallisINF_AMG
from hypre.bindings import solve
from problems.amg import stencil_27_laplace
from setup.utils.setup_amg import build_actions_th_coarsen_interp


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=200_000, help="bandit steps for overhead benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--solve-samples", type=int, default=200, help="number of solves to time")
    args = ap.parse_args()

    actions = build_actions_th_coarsen_interp(
        [0.10, 0.25, 0.40, 0.50, 0.70],
        [10, 8, 6],
        [6, 8, 0],
        fixed_params={"trunc_factor": 0.0, "P_max_elmts": 4, "agg_num_levels": 0, "agg_interp_type": 4},
    )
    d = 5

    rng = np.random.default_rng(args.seed)
    xs = rng.normal(size=(args.T, d)).astype(float)
    xs[:, 0] = 1.0  # bias
    # Non-negative, heavy-ish losses to mimic WU scale (doesn't matter for timing).
    ys = rng.lognormal(mean=3.0, sigma=0.6, size=(args.T,)).astype(float)

    bandit = LinUCB_AMG(actions, context_dim=d, alpha=float(args.alpha), l2_reg=float(args.l2), seed=args.seed + 1)

    # Warmup (jit caches, numpy alloc paths, etc.)
    for i in range(2000):
        _ = bandit.predict(xs[i])
        bandit.update(float(ys[i]))

    t0 = time.perf_counter_ns()
    for i in range(args.T):
        _ = bandit.predict(xs[i])
        bandit.update(float(ys[i]))
    t1 = time.perf_counter_ns()

    elapsed_s = (t1 - t0) / 1e9
    per_step_us = (elapsed_s / float(args.T)) * 1e6

    # ---- Tsallis-INF (strong_threshold-only) predict+update -----------------
    np.random.seed(args.seed + 2)
    grid = np.array([p["strong_threshold"] for p in actions], dtype=float)
    tsallis = TsallisINF_AMG(grid, int(args.T))
    # Warmup
    for i in range(2000):
        _ = tsallis.predict()
        tsallis.update(1.0 + float(ys[i]))
    t0 = time.perf_counter_ns()
    for i in range(args.T):
        _ = tsallis.predict()
        tsallis.update(1.0 + float(ys[i]))
    t1 = time.perf_counter_ns()
    ts_elapsed_s = (t1 - t0) / 1e9
    ts_per_step_us = (ts_elapsed_s / float(args.T)) * 1e6

    # Time real solves (default params on random instances).
    default_params = {"trunc_factor": 0.0, "P_max_elmts": 4, "agg_num_levels": 0, "agg_interp_type": 4,
                      "strong_threshold": 0.25, "coarsen_type": 10, "interp_type": 6}
    solve_rng = np.random.default_rng(args.seed + 10_000)
    # Warmup solve
    mkw, _, _ = stencil_27_laplace(rng=solve_rng)
    _ = solve(params=default_params, **mkw)
    t0 = time.perf_counter_ns()
    wus = []
    for _i in range(int(args.solve_samples)):
        mkw, _, _ = stencil_27_laplace(rng=solve_rng)
        r = solve(params=default_params, **mkw)
        wus.append(float(r.work_units))
    t1 = time.perf_counter_ns()
    solve_elapsed_s = (t1 - t0) / 1e9
    per_solve_ms = (solve_elapsed_s / float(args.solve_samples)) * 1e3

    wu_mean = float(np.mean(wus)) if wus else float("nan")

    print("=== Bandit overhead benchmark ===")
    print(f"Steps: {args.T}")
    print(f"LinUCB predict+update:     {per_step_us:.3f} us/step")
    print(f"Tsallis-INF predict+update:{ts_per_step_us:.3f} us/step")
    print()
    print("=== Solve timing (default params) ===")
    print(f"Solve samples: {args.solve_samples}")
    print(f"Avg solve wall time: {per_solve_ms:.3f} ms/solve")
    print(f"Avg WU (default): {wu_mean:.3f}")
    print()
    ratio = (per_step_us / 1e3) / per_solve_ms  # (us->ms)/ms
    ts_ratio = (ts_per_step_us / 1e3) / per_solve_ms
    print("=== Relative overhead ===")
    print(f"LinUCB decision / one solve:  {ratio*100.0:.4f}%")
    print(f"Tsallis-INF decision / solve: {ts_ratio*100.0:.4f}%")


if __name__ == "__main__":
    main()
