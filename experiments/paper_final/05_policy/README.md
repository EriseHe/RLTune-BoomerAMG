# 05 — Controlled RL comparisons

This development study is authorized after the completed Module 04 results.
The rollback point is `checkpoint/20260920-before-timing-rl-study`, commit
`007bd195b9aa976f6b640bd0de861669e3dcde7f`. New code lives on
`experiment/timing-rl-isolation`. The original branch and experiment outputs
are retained. The pre-change runtime binary and its checksum are also saved
locally in `results/paper_final/01_numerics/pre_timing_study/`.

## Protocol

Read [protocol.json](protocol.json) before interpreting the results. Keep 60³
diffusion–advection, cap 50, tolerance 1e-6, original failure rollback, controller
features and action space. No penalty or RNG-stream redesign is introduced.

1. **Calibration:** 16 fresh inputs, Default setup and fixed weight 1, three
   identical-work measurements per input. Estimate one heuristic timing scale
   independently of the adaptive branches. Each near-tie logical execution is
   charged the full calibration cost, even though the artifact is reused.
2. **Frozen evaluation B:** four final setup snapshots (old/new × setup-only/
   Joint), old/new frozen LSTDQ controllers, fixed weights and periodic schedules.
   Thirty-two fresh inputs select one fixed weight and one schedule per source.
   Another 128 inputs evaluate those choices, weight 1 and both controllers;
   the first 16 have three timing repetitions. Selection and evaluation inputs
   are disjoint. Freeze LSTDQ learning and random exploration while retaining
   its learned LCB scoring rule. Timing repeats can still change decisions
   through measured cycle-time features and are not independent training runs.
3. **Online comparison A:** the frozen **new setup-only** source supplies all
   5000 fresh hierarchies. Compare weight 1, the selected fixed weight, the
   selected schedule and initially untrained LSTDQ. The RL branch shares the
   physically executed weight-1 prefix through input 1000, then learns from its
   own trajectory. This first trial has one training execution; it does not
   estimate training-repeat uncertainty or select an activation boundary.
4. **Timing stability:** run the separate [09 protocol](../09_algorithms/README.md)
   sequentially. Three execution repetitions compare the raw rule, stable
   numerical ties, and the same stable rule with a calibrated near-tie band.

The frozen setup sources use `recommend`, their final effective alpha, and a
common, fixed 2048-arm bank containing the default and the union of their saved
elite sets, filled using a prespecified random seed. Thus this is an explicit
snapshot deployment rule; it does not replay the original adaptive candidate
search. No solve outcomes update that bank or any setup snapshot. All mutable
source-checkpoint arrays and RNG states are compared before and after the study.

Every compared solve uses a fresh environment. A read-only CPU fingerprint
checks all level A/P/R CSR values and topology, RHS and initial iterate before
the first cycle; initial cycle counters must be zero. The FNV-1a digest is a
noncryptographic equality diagnostic. Native execution is single-rank CPU.
No prepared solver is reused across policies. Fixed/schedule controls use the
same step interface as RL; recovery uses the unchanged default fallback.

## Accounting and interpretation

- Direct solve-control cost includes native solve, policy dispatch, features,
  inference, updates, controller lifecycle and branch-specific fallback work.
- Paired logical totals add **the same** primary setup cost (the mean of the
  paired physical setup measurements) and source-selection cost to every policy.
  Actual measured physical totals are retained separately. Shared prefix records
  identify their physical source and are not additional solver executions.
- All failed attempts remain in the totals. Final failures are reported alongside
  runtime; these are attempted-stream costs, not all-success completion costs.
- Hierarchy hashing, matrix assembly, uninstrumented wrapper work, telemetry,
  checkpoint/file IO and the common four-input warmup are harness work outside
  the algorithm metric. Hash time and enclosing method-wall time are retained.
- The externally supplied frozen source is not free end-to-end Joint training.
  Its historical training expense is excluded from this conditional comparison.
- Payback is only the final nonnegative savings stretch within the observed
  horizon. No independent-training confidence interval is inferred from inputs
  within one trajectory or from frozen timing repetitions.

## Run and outputs

Use the `rl` environment, commit the source first, then run:

```bash
./experiments/paper_final/05_policy/run.command
```

The launcher records the commit and native checksum, waits for AC power, checks
that the source did not change, and launches with `caffeinate`. The runner uses
one global native-timing lock, checks disk space and power, refuses existing
output directories, and never overwrites the historical inputs. It writes
phase reports and status while running; a failed audit stops the remaining queue.

- `results/paper_final/05_policy/controlled/status.json`: current progress.
- `frozen/`: held-out cross evaluation, baseline selection and frozen-state audit.
- `online/`: common-hierarchy learning curve, cost comparisons and checkpoints.
- `results/paper_final/09_algorithms/timing/`: three execution repetitions.
- `results/paper_final/05_policy/controlled_launch/`: launch, power and exit logs.

The source checkpoints are local artifacts from the two previously supplied
ZIP packages; cloning this repository alone does not download them.

Validation only: `python -m experiments.paper_final.run_05_controlled`.
A separate 8³ integration check uses `--smoke --output-root <unused-directory>`;
its measurements are excluded from experiment conclusions.

To inspect or resume the saved pre-study code without deleting the experiment
branch, create a new branch at the rollback tag and rebuild the native interface.
The original local binary backup is available when an exact binary restore is
needed. Do not overwrite either completed run's model checkpoints.
