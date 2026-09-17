# 01 — Numerical recovery

## September 16: RL timing coverage

RL preparation now includes its initial residual computation. Each controlled
step includes parameter updates, the AMG cycle, external residual monitoring
and stopping check; errors retain completed work. The existing controller,
bandit, recovery and reporting paths charge the returned costs once.
68 relevant tests passed; see the [timing report](../../../results/paper_final/01_numerics/timing/REPORT.md)
for the frozen before/after replay and the exact timer boundaries. Historical
02/03 trajectories and checkpoints retain their original measurements.

## September 16: BoomerAMG stopping-status alignment

The prepared RL wrapper and Python outcome classifier now follow the linked
BoomerAMG rule at the cycle limit: with positive tolerance, reaching the limit
reports `MAX_CYCLES` even when that final cycle achieves the residual target.
Before the limit, convergence requires strict `< tol`. HYPRE itself is unchanged.
60 relevant tests and two exact historical cycle-50 replays passed; see the
[alignment report](../../../results/paper_final/01_numerics/stopping_alignment/REPORT.md).
Old experiment records retain their original semantics. Timing coverage is a
separate correction documented above.

## LSTDQ inverse recovery

The shared LSTDQ mean update now tracks whether the cached matrix is a true
inverse. A singular or truncated-pseudoinverse prefix cannot reenter the
Sherman–Morrison update until a true inverse is restored. Rebuilds use the
accumulated estimating matrix; both inverse residuals are checked. A completed
episode cannot commit while its inverse remains invalid.

The denominator guard also covers magnitudes above `1/sqrt(machine epsilon)`:
exiting a nearly singular prefix can otherwise subtract huge inverse entries
and lose accuracy. Normal updates keep the existing quadratic-cost path.
Dense inverse audits are confined to rebuilds and legacy checkpoint loading.

The validity flag participates in snapshots, rollback, checkpoints and
summaries. New checkpoint versions are v1: 3, v2: 2, v3: 2. Compatible prior
versions load through an inverse audit. These are checkpoint format versions;
the paper algorithm remains **recursive LSTDQ v3**.

## Verification

65 targeted tests passed: new singular/near-singular/truncated-prefix tests
across all three controller families, coherent episode trace-energy identities,
normal recursive versus batch solves, checkpoint/rollback, recovered terminal
charges, schedule accounting, and the existing formal-suite contract tests.
The scalar regression finishes with `theta = 19/28`.

Log: `results/paper_final/01_numerics/tests.log`.

```bash
PYTHONPATH=experiments/diagnostics/solve_control:experiments/joint/solve_control \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/opt/anaconda3/envs/rl/bin/python -m unittest \
  experiments.paper_final.test_01_numerics \
  experiments.paper_final.test_02_protocol \
  solve.tests.test_recursive_lstdq_v3 \
  solve.tests.test_shared_action_rl \
  solve.tests.test_recovery_protocol \
  experiments.diagnostics.solve_control.test_paper_final -v
```

This fixes specified exceptional paths. It does not make every floating-point
prefix well conditioned, turn the uncertainty statistic into a confidence
bound, or establish policy convergence. File checkpoint loading starts a new
episode; in-memory snapshots retain the trace for within-episode rollback.
