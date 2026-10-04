# Historical Module 05: frozen policy comparison (now Module 06 evidence)

## Current corrected comparison — Run 05, September 29, 2026

[Run 05](RUN05.md) retimes all five current policies on the same six frozen
Joint-trained checkpoints and 100 diffusion 60³ inputs, with three repetitions.
Its only periodic baseline is **(2.85,1.10)**; historical (1,3) and (2.6,1)
are omitted. All 8460 distinct solves completed without recovery or final
failure. The unfinished first attempt was discarded before a clean restart.

Native RL reductions are 41.82% against W1, 11.72% against the stream-wide
fixed reference, 9.58% against the per-instance fixed reference, and 0.63%
against Periodic. Including controller cost gives a 3.65% RL cost increase
against Periodic. All checkpoints, hierarchy choices and fixed selections
are retained; all current timestamps come from the new session.

The [Run 05 report and archive](RUN05.md) define the current corrected
matched-hierarchy evidence. [Run 04](COMPLETED.md) and the earlier sections
below remain historical records, with their original data and identifiers.

## Original evaluation protocol

The 2026-09-27 user-approved scope is 60³ diffusion only, all six existing
Module 04 final checkpoints, 100 shared fresh test problems and 100 separate
shared development problems. Both setup-only and Joint frozen selectors
provide matched setup configurations. No controller or setup learner trains.

`run_05_policy.py` records the complete protocol, generated inputs, frozen
setup choices, checkpoint copies, environment checks and source hashes before
native evaluation. Smoke inputs use a separate seed offset and are not
formal results. Numerical tolerances, 50-cycle cap, native binaries and the
single-thread environment match the completed Module 04 runs.

The main fixed comparator is **Best fixed — test hindsight**: evaluate every
weight in `1.00, 1.05, ..., 3.00` on every test problem, then select one weight
per training checkpoint and hierarchy source by cumulative native solve plus
recovery setup time. Only weights with successful complete procedures on
every prescribed trial qualify. This is a best-observed finite-grid diagnostic,
not a continuous optimum or an out-of-sample tuning claim. Per-problem best
fixed is a separate, stronger hindsight diagnostic. Timing repetitions are
averaged within each problem before aggregation; no fastest-repeat selection.

Development cases select one deployable fixed weight, periodic schedule and
prefix-tail schedule before test execution. Native HYPRE and Chebyshev
smoothers are additional practical references, using full setup-plus-solve
costs for the primary comparison so spectral preparation remains charged.
All attempts, recovery costs and final failures remain in the raw records.
Programming errors stop the experiment rather than becoming solver failures.

The final analysis presents no-overhead costs first, retains overhead-inclusive
costs, and reports every training replica. Shared matrices and hierarchy
sources must not be treated as independent training replications. These
comparisons assess frozen-policy utility; they do not prove feedback or RL
is necessary.

Run from the repository root with the unchanged Module 04 Python environment:

```sh
python -m experiments.paper_final.run_05_policy prepare
python -m experiments.paper_final.run_05_policy preflight
python -m experiments.paper_final.run_05_policy run --workers 3
python -m experiments.paper_final.run_05_policy watch
```

The supervisor runs at most three independent single-thread workers, prevents
system sleep during computation, and resumes without repeating completed
trials. It does not claim CPU affinity or isolated-core timing. Each worker
randomizes serial policy order within a problem/hierarchy block. Closing the
progress viewer does not stop the separately launched supervisor.

Results are under
`results/paper_final/05_policy/20260927_diffusion60_6seeds_100cases/`.
`status.json` and `supervisor.log` show progress; `raw/` contains all trials;
`analysis/` is written after all phases finish. The full protocol and exact
selection rules are in `protocol.json`.

Publication figures are generated separately from the completed logs by
`python -m experiments.paper_final.plot_05_policy`. See [FIGURES.md](FIGURES.md)
for the figure menu, plotting-code layout, export formats and scientific
conventions. The plotting entry point does not execute native solves.

## Second timing run

[Run 02](RUN02.md) repeats the same six checkpoints, 100 inputs and saved Joint
setups, with three fresh repetitions of every problem. It retimes six selected
policies, including both `(2.5,1)` and `(1,3)` schedules, without repeating the
weight search or any training. Its main heatmap contains global fixed,
per-instance fixed, Periodic `(1,3)`, and frozen RL.

Outputs are separate in
`results/paper_final/05_policy/20260927_run02_diffusion60_joint_6seeds_100cases/`.
The repeat runner creates the analysis and figures automatically and records
comparisons against Run 01. Its saved `watch_run02.command` opens live progress.

The [best-seed presentation revision](RUN02_BEST_SEED_FIGURES.md) uses the
checkpoint with the largest cumulative no-overhead RL reduction versus `w=1`
for main trajectory examples, labels the selection, and retains all six seeds
in aggregate figures. Its gallery is `analysis/paper_figures_best_seed/`.
