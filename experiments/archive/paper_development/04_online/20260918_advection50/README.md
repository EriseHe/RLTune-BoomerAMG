# Module 04 — diffusion–advection, 50-cycle rerun

This separate batch follows commit `6576ed54`. It reruns grids 40³, 60³ and
80³ sequentially with the user-requested **50-cycle cap**, replacing the
previous advection cap of 100. All earlier results remain in their original
directories, including the interrupted 80³ run.

Each group has 5000 paired problems and three methods: Default, LinUCB v4,
and LinUCB–LSTDQ v3. RL starts on problem 1001. The input hashes, forward
DifConv stencil, coefficient distributions, 7D context, action grid, recovery
policy and timing schema 2 match the preceding suite. The shared seed tuple
remains base **56700120**, bandit **56760120**, controller **56766120** and
method order **56772120**. This is the development/diagnosis seed, not a new
replicate. All learners start fresh. The smaller budget changes failure
feedback and the cycle/max_cycles feature, so this is a newly trained run.

## Changes and purpose

- A suite may explicitly select one PDE family; older manifests retain their
  original two-family default. This batch includes advection only.
- The new suite enables an immediate consistency check after each method call:
  charged component time must fit inside the independent method wall timer
  (1 microsecond tolerance, matching the existing final audit). On violation,
  save the offending observation in `timing_anomaly.json` and stop the batch.
  A learner may already have consumed that observation; a fresh run is required.
  This catches the clock mismatch seen during the earlier laptop sleep. It
  cannot detect every kind of interruption. Valid measurements and learning
  objectives keep the previous definitions.
- The Terminal launcher uses `caffeinate -is` to inhibit idle/system sleep
  while on AC power. Keep the laptop plugged in with its lid open; forced sleep
  can still interrupt the experiment.

Unrecovered numerical failures remain recorded and do not stop later grids.
Timings therefore describe budgeted attempted solves; report failure counts
alongside time reductions. Timing/protocol/process errors stop the batch.

## Run and outputs

```bash
experiments/paper_final/04_online/20260918_advection50/run.command
```

Results go to `results/paper_final/04_online/20260918_advection50/`, with batch
and group logs in `logs/`. The existing pipeline generates figures, raw
trajectories, checkpoints and aggregate reports. The launch requires committed
source and records its Git commit, source/binary hashes and input/config hashes.
Existing run directories are never overwritten.

Preflight passed 53 relevant tests, including the immediate-abort check and
old/new suite validation. A separate 8³ recording smoke passed all 36
method-problem timing checks, including 10 controlled episodes with the guard
enabled. Its mild inputs and early activation are for functional validation
only; they do not change the formal benchmark.
