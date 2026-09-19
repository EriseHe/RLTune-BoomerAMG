# Module 04 — retained failure feedback

This four-group development comparison runs diffusion and diffusion–advection
at 40³ and 60³. Each group contains 5000 paired inputs for Default, LinUCB v4,
and LinUCB–LSTDQ v3. All learners start fresh; setup learns from problem 1 and
RL starts on problem 1001. Sequential order is diffusion 40, advection 40,
diffusion 60, advection 60. Only these four groups are prescribed.

The seed tuple and inputs match September 18: base 56700120, bandit 56760120,
controller 56766120, method order 56772120. Diffusion keeps cap 50 and advection
keeps cap 100. The forward discretization, 18/18/9 profile, tolerance 1e-6,
action grids, candidates and learner settings are unchanged. This reused seed
is development evidence, not an untouched final-paper replicate.

## Change from the previous experiments

Valid final failures, including failed fallback, now retain setup and TD
updates. Setup trains on finite expenditure plus a fixed final-failure penalty;
the TD terminal target includes the last cycle, uncharged fallback work and
the same penalty once, with zero successor feature. Ordinary setup choices use
this penalized cost head. The auxiliary failure head still measures primary
attempt failure and is used only for construction reselection.

Successful completion receives no penalty. Real runtime, including failed
work, is reported separately from the penalized objective. No speed reduction
includes an artificial penalty. Retaining failure observations can change
later setups and controller actions, so the effect on speed and reliability
must be measured rather than assumed.

Selected arm IDs are captured before rollback. Invalid measured times and
non-solver execution errors abort; they do not become valid failure samples.
Failed finite LSTDQ episodes preserve the existing complete-episode algebra;
this does not establish a policy-quality or confidence-coverage guarantee.
The new wrapper checks also add some execution work, which the method-wall
timer records. This is a comparison of recorded implementations, not an exact
counterfactual isolation of only one target update on the same wall times.

## Prespecified penalty calibration

The user selected a Default-budget rule before any new learning outcomes.
For each group, eight independently seeded Default inputs estimate setup and
per-cycle native costs. One unrecorded warmup solve precedes them. With cap H:

    Lambda = ceil(1000 * 2 * (median(setup) + H * median(solve/cycles))) / 1000.

The factor two represents a primary budget plus one restarted fallback budget.
This is a representative finite-work scale, not an upper bound, a reliability
constraint, or an estimate of time to eventual convergence. It is frozen for
all methods and all 5000 inputs in that group.

| Group | Cap | Fixed Lambda (seconds) |
|---|---:|---:|
| Diffusion 40³ | 50 | 0.436 |
| Diffusion–advection 40³ | 100 | 0.684 |
| Diffusion 60³ | 50 | 1.468 |
| Diffusion–advection 60³ | 100 | 2.378 |

Calibration uses generator seeds 74000123 + 6000*j, j=0,...,7, and shuffle
74048123. None of its inputs overlap the benchmark stream. Raw measurements,
source hashes, power state and the calibration hash are recorded under
`results/paper_final/04_online/20260919_failure_feedback/preflight/` and in
`suite.json`. The calibration script is
`experiments/paper_final/calibrate_04_failure_penalty.py`. Do not recalibrate
these frozen coefficients in response to outcomes from this comparison.

## Execution and artifacts

```bash
experiments/paper_final/04_online/20260919_failure_feedback/run.command
```

The launcher requires a committed working tree, uses one native process at a
time, sets single-thread BLAS/OpenMP, and enables macOS idle-sleep prevention.
Keep the laptop open and connected to AC. Invalid timing aborts the run; partial
runs are preserved and never silently overwritten or resumed.

Results, normal plots, trajectories and final checkpoints go to
`results/paper_final/04_online/20260919_failure_feedback/`, with batch and
per-group logs in `logs/`. The aggregate report appears in
`analysis/modules_1_2.md` as groups finish. Historical results remain separate.

The four matched prior groups took about 3 h 38 min in total. Reserve roughly
4–5 h for this rerun; different learned paths and sustained machine load can
change that estimate. The launch commit and source/config/native hashes are
recorded automatically before learning starts.
