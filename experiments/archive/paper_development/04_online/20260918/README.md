# Module 04 — September 18 online comparison

This revision follows the completed September 17 diffusion runs at commit
`b2d37c99`. It fixes the omitted controller lifecycle timing and runs one
user-requested seed on both PDE families and all three grid sizes. The launch
records the committed Git revision, source/native hashes, configuration hashes,
and input hashes. The earlier results remain in their original directories.

## Differences from the September 17 experiments

| Item | September 17 | This suite | Purpose |
|---|---|---|---|
| Controller timing | Feature construction, action selection, updates | Those components plus rollback snapshot, episode initialization and rollback | Charge recurring lifecycle work once |
| Setup cost feedback | Included the original controller timer | Also includes the newly measured lifecycle cost | Keep observed setup cost consistent with the charged controller overhead |
| Solve TD target | Native cycle cost and applicable terminal recovery | Same target | Keep learning overhead separate from physical cycle costs |
| Additional stopwatch | Component sums and whole-process elapsed time | Also one wall stopwatch per complete method call | Expose matrix construction and wrapper/binding work outside component timers |
| Problem-generator base seed | 59700150 | 56700120 | User requested the earlier pre-run diagnosis tuple |
| PDE families | Diffusion only | Diffusion and diffusion–advection | Complete both families at the requested seed |
| V-cycle cap | 50 | Diffusion 50; advection 100 | Give slow advection solves more headroom |
| Advection construction | Forward DifConv available but not run in this batch | Same forward DifConv construction, `atype=0` | Preserve the HYPRE-style benchmark and earlier development comparison |
| Unrecovered failures | Stop later groups after a failed group | Record and continue for advection; diffusion retains the stop rule | User explicitly accepts advection failures as part of a bounded-budget stress test |

Adding lifecycle cost changes LinUCB's observations and can change later setup
choices. The seed also differs from September 17. These are separate online
training runs, not an isolated timing-correction ablation. Advection's larger
cycle budget also changes the encoder's cycle/max_cycles feature and failure
feedback. All learners start from scratch; historical checkpoints are not used
to initialize this suite.

The native timer and HYPRE library are unchanged by this revision. Native-cost
figures retain the same definition. For a new trajectory, subtracting
`lifecycle_runtime` from the component sum reconstructs the older timing scope,
but does not reconstruct its counterfactual learning trajectory.

## Fixed protocol

- 5000 paired problems per method and group, including the first 1000.
- Default, LinUCB v4 setup-only, and LinUCB–recursive LSTDQ v3.
- Setup learns from problem 1; solve RL starts on problem 1001.
- Independent mutable setup learners; seeded randomized method order per
  problem; one native solve at a time.
- Relative residual tolerance 1e-6; 18/18/9 smoother profile; standalone AMG.
- Diffusion context 4D and advection context 7D, including the intercept.
- Existing coefficient ranges, candidate generation, action grid, exploration,
  state encoding and bounded recovery policy are retained.
- Sequential order: diffusion 40, advection 40, diffusion 60, advection 60,
  diffusion 80, advection 80. Each group contains all three methods.

The shared seed tuple is base **56700120**, bandit base **56760120**, controller
base **56766120**, and method-order **56772120**. Generator seeds are
56700120 + 6000*j for j=0,...,7, with 625 cases each; shuffle seed is 56748120.
Existing controller-factory offsets remain unchanged. The same tuple is used
across sizes and families; these six groups are not six independent replicates.
This seed previously appeared in development/pre-run diagnosis and is labelled
as such, rather than being claimed as an untouched holdout.

Our DifConv coefficients follow HYPRE's `BuildParDifConv` in
`hypre/source/src/test/ij.c`, whose test-driver default is `atype=0`. The
random coefficient stream and online-learning protocol are project choices.
Positive-velocity forward differencing can yield non-monotone difficult
systems. A 100-cycle budget does not guarantee their convergence. A preflight
replay of six historical failures converged one case in 62 cycles, while five
still had growing residuals at 100. These selected cases are not a failure-rate
estimate. There is no change to upwind discretization or removal of difficult
inputs in this suite.

## Timing and interpretation

The main recorded component cost remains:

    setup + native solve + controller + bandit overhead.

`controller = feature + decision + update + lifecycle`. Lifecycle is a subset,
not another term to add to the main total. Failed attempts and fallback native
costs already belong to setup/solve. The same costs are not added twice.

`method_wall_runtime` measures the whole per-method call, including matrix/RHS
construction, solver object lifecycle, binding/wrapper work and learner work.
It excludes outer trajectory I/O, checkpointing and plotting. It is recorded
for all methods and is reporting-only; neither learner uses this extra wall
stopwatch as its objective. The main component metric still excludes upfront
learner/AOT preparation and the other established exclusions. Consequently,
neither is a complete application-payback measure including all preparation.

Timing schema 2 records the scope in each result's protocol. Audits require
finite, nonnegative timing, consistent controller sums, wall time covering the
component sum, paired input hashes and consistent final-attempt residuals.
Successful solves must reach residual < 1e-6 before their family-specific cap.
Null/nonfinite final residuals are retained only for failed outcomes.

Advection groups retain unrecovered outcomes and proceed to the next group.
Such costs describe budgeted attempted solves, not successful completion of
every problem. Report reductions together with success/failure counts; do not
delete failures, switch seeds, or label faster failure as successful speedup.
Protocol errors, source changes, invalid accounting and unexpected process
errors still stop the suite. The launcher never overwrites existing runs.

## Run and artifacts

```bash
experiments/paper_final/04_online/20260918/run.command
```

Results: `results/paper_final/04_online/20260918/`. Batch and per-group logs are
under `logs/`; preflight checks are under `preflight/`. The normal figures,
checkpoints, raw trajectories and Module 1/2 summaries are generated for every
completed group. Additional lifecycle and method-wall totals appear in the
aggregate timing audit and per-run JSON summaries.

Preflight includes 140 relevant unit/integration tests and short 8³ native
recording checks for both families. The advection recording smoke uses mild
coefficients solely to exercise accounting; it is separate from the difficult
case replay and from the full benchmark.

The previous diffusion batch took 5 h 43 min. A provisional budget for this
six-group suite is 12–16 h on the same Mac, with substantial variation possible
from learned setup paths and sustained machine load. Actual subprocess times
are recorded per group. Keep AC power and use the sequential launcher.
