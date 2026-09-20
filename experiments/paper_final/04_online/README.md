# PAPER_FINAL: Modules 1 and 2

The active [original-policy cap comparison](20260919_cap_baseline/README.md)
restores the September 18 failure-learning rollback rule and varies only the
cap (50, 100, 200, 500) on the same 60³ diffusion–advection stream. Cap 50
is an additional development run requested after 100/200/500 completed. The
[penalized-feedback trial](20260919_cap_comparison/README.md) was stopped and
withdrawn from the active plan; its recorded results remain development history.

The September 17 diffusion runs are complete. The separately versioned
[September 18 suite](20260918/README.md) charges controller lifecycle work,
records per-method wall time, reuses the user-requested diagnosis seed, and
runs both families with a 100-cycle advection budget. The protocol below
documents the original September 17 suite.

Prepared after local code checkpoint `ce609c1`, with subsequent recovery,
stopping, timing and final-residual recording fixes. Source and native-library
hashes are captured at launch. The manuscript is outside this code change.

September 17 selection: **diffusion, replicate 1 only**, grids 40³ → 60³ → 80³.
One launch runs those three groups sequentially. Within each group, the three
methods process each paired problem in a seeded random order before advancing
to the next problem. There is no concurrent native solve. The full 18-group
manifest remains available; other groups are not included in this launch.

Launch checks: 74 relevant recovery/native/runner/reporting tests passed and
all 18 revised config/stream/context validations passed. Scheduling tests mock
formal experiment dispatch; a separate 8³ development check verifies recording.

## Fixed study design

The suite contains 18 groups: two PDE families × grids 40³, 60³, 80³ × three
prespecified replicates. Each group contains these three independent methods:

| Paper label | Setup | Solve |
|---|---|---|
| Default | Reference setup | Reference relaxation, w = 1 |
| LinUCB | Online setup learning | Reference relaxation, w = 1 |
| LinUCB–LSTDQ | Online setup learning | Reference relaxation on problems 1–1000; LSTDQ on problems 1001–5000 |

Every method processes the same 5000 matrices and RHS inputs in a group. Setup
learning starts on problem 1 and continues throughout. The first 1000 problems
are included in the reported 5000; there is no additional uncharged training
prefix. The JSON field `stream.warmup_cases` is therefore zero. The solve
activation boundary is 1000, which the existing runner interprets as enabling
RL on the following problem, 1001.

Both learned methods have independent mutable LinUCB state but matched initial
RNGs and candidate-sampling seeds. They receive their own realized costs, so
their chosen setup paths may diverge. The LinUCB–LSTDQ versus LinUCB comparison
measures the complete adaptive frameworks. It does not isolate the effect of
relaxation on an identical hierarchy. Module 3's matched-hierarchy experiment
and best-fixed-weight sweep are deferred.

### Relationship to stage 05

Stage 05 uses the final setup and controller snapshots from stage 04 for
independent matched-hierarchy evaluation. The finer fixed-weight grid, exact
endpoint schedules, development-selected alternating prefixes/tails, and actual
advection setup sources belong there; they are not prerequisite experiments
for this online suite.

The later 02/03 review also recommends a representative complete online
LinUCB-plus-schedule comparison. That is a separate online extension: its setup
learner continues adapting to schedule costs, whereas stage 05 freezes the
learners. It can be scheduled separately after independent development selection
of the schedule, with prespecified paired inputs and timing rules. It has not
been added to these 18 three-method configurations and does not have to precede
them. Remaining numerical/recording P0 issues must still be resolved before
formal execution; see the [preflight report](../../../results/paper_final/04_online/preflight/20260917/REPORT.md).

All methods use the 18/18/9 down/up/coarse smoother profile, tolerance 10⁻⁶,
50-cycle cap, and the existing bounded-recovery protocol. LSTDQ uses the
corrected setup-category and relaxation-weight encoding, weights
`1.00:0.05:3.00` (41 actions), and the same basis, ridge, epsilon schedule and
uncertainty parameters as the completed context/activation study. Setup uses
the same resolution-20 space, categorical choices, structured-512 candidates
and AOT candidate protocol. No optimizer or recovery rule is retuned here.

## Shared problem context

Let d_i = log(c_i)/log(1000) and
v_i = sign(a_i) log(1 + |a_i|)/log(1001).

| PDE family | Context shared by both learners | Dimension including intercept | LSTDQ state / Q feature dimension |
|---|---|---:|---:|
| Scalar anisotropic diffusion | (1, d_x, d_y, d_z) | 4 | 31 / 279 |
| Scalar anisotropic diffusion–advection | (1, d_x, d_y, d_z, v_x, v_y, v_z) | 7 | 34 / 306 |

The diffusion coefficients are sampled on [1, 1000]. The advection family uses
the existing `problems/scalar_anisotropic_diffusion_advection.py` generator,
with each of the three advection coefficients sampled on [1, 1000]. Coefficients
are spatially constant within one problem and vary between problems. Its
existing discretization and RHS construction are retained.

The stream's historical eight-field storage format is retained for old runs.
Both learners explicitly project that storage to the appropriate shared view.
The implementation keys are `diffusion3d` and `canonical_no_c_mean`; these are
configuration keys, not paper method names. `c_mean` is excluded from both
learners. The LSTDQ state adds residual history, cycle information, last
relaxation and cycle time, and setup descriptors. Its leading constant also
serves as the shared intercept; the intercept is not duplicated in its PDE
feature slice. Original 8D and diffusion3d/diffusion4d behavior is retained.

## Seeds and provenance

The three new replicates were chosen before any PAPER_FINAL outcomes, by adding
the following offsets to the completed study's base seed tuple. Unlike the
earlier C/D comparison, these replicates change the problem/RHS stream as well
as algorithm randomness.

| Replicate | Offset | First problem-generator seed | Bandit base seed | Controller base seed | Method-order seed |
|---|---:|---:|---:|---:|---:|
| 1 | 7000093 | 59700150 | 59760150 | 59766150 | 59772150 |
| 2 | 5000077 | 57700134 | 57760134 | 57766134 | 57772134 |
| 3 | 6000083 | 58700140 | 58760140 | 58766140 | 58772140 |

All eight generator seeds and the shuffle seed are shifted by the same offset;
the group sizes remain 8 × 625 = 5000. Method seed offsets are zero. Existing
factory-specific internal RNG offsets are retained. The same replicate tuple
is reused across grid sizes and families; comparisons across those settings
must not be counted as additional independent replicates.

Before formal launch on September 17, replicate 1 was changed consistently
across all six family/grid configurations: the original offset 4000063 reused
inputs already consumed by development work and the timing probe. The new
tuple was fixed without formal outcomes; the new and replaced input/coefficient
sets have zero overlap. Original configs and the hash revision are retained in
`results/paper_final/04_online/preflight/20260917/seed_revision/`.

`experiments/paper_final/04_online/suite.json` fixes all 18 config hashes, input-stream hashes,
and the sequential execution order. Each config is self-contained and uses
the existing `run_joint_experiment.py`. The suite wrapper records the code and
native-library hashes, Git revision, uncommitted tracked patch, new source
files, Python/NumPy/platform information, and thread settings at launch.
Resuming requires the same code/config/library versions.

## Timing and reporting

Module 1's primary metric is total recorded online end-to-end time over all
5000 problems. Each problem's cost is

    setup + native solve + controller overhead + bandit overhead.

Setup/solve costs include unsuccessful attempts and recovery. These are the
existing timer boundaries: one-time learner initialization and AOT candidate
generation, matrix/RHS assembly, trajectory I/O, checkpointing and plotting are
outside the per-problem metric. Whole-subprocess elapsed time is recorded
separately, but covers all three branches together and cannot be attributed to
individual methods. Claims about total application payback must account for
the excluded preparation costs separately.

The September 16 [native timing correction](../../../results/paper_final/01_numerics/timing/REPORT.md)
includes RL parameter updates, external residual monitoring and stopping work.
The initial residual is charged once to RL preparation; full-solve computes it
inside its solve timer. Interpret the setup/solve component split accordingly.
Native timers also exclude initial-vector reset, solver creation/destruction,
initial setup parameter application outside the prepare helper and binding
overhead. The recorded online metric is not whole-process elapsed time.
The linked HYPRE library and full-solve timer are unchanged.

The analysis retains every prescribed replicate and normalizes by its paired
Default. Time reduction is `100 × (1 − method/default)`; the speedup factor is
`default/method`. It reports per-seed results and the equally weighted mean and
sample SD of the three seed-level reductions. These are descriptive summaries,
not significance claims from treating 5000 adaptive observations as independent.
Any per-problem bootstrap intervals in the existing diagnostic plot bundle are
not the paper's cross-seed uncertainty measure.

Module 2 reuses both families' 60³ logs for the all-5000 and last-1000 windows.
It reports setup, solve, controller and bandit time, first-attempt failures,
unrecovered failures, and recovery costs. Fallback time is a subset of the
setup/solve totals and is never added twice. The same windows are available for
all grids. No separate native run is needed for Module 2.

Every new recovery record includes `completed_residual_norm`,
`completed_cycles` and `completed_status` for the final attempt, plus the
fallback residual/cycles when fallback is used. Legacy `residual_norm` still
describes the primary attempt. The formal audit requires the new fields,
checks them against the final attempt, and verifies that a successful solve
has residual strictly below tolerance before the cycle cap. Setup snapshots
are saved in `final_bandit_states/`; final solve models are in `checkpoints/`.

The existing per-run plot generator uses the paper labels Default, LinUCB and
LinUCB–LSTDQ for this roster. The shared settings show LinUCB (4D) or LinUCB (7D)
once; the dimension includes the intercept. Context comparisons use the same
dimension convention, with algorithm names rather than feature-removal names.
The aggregate paper tables are written to
`analysis/modules_1_2.md`, with auditable numbers in `modules_1_2.json`.

## Commands

From the repository root, validate without native experiments:

```bash
/opt/anaconda3/envs/rl/bin/python -m experiments.paper_final.run_04_online --validate-only
```

Run the selected diffusion replicate, all three grids sequentially:

```bash
experiments/paper_final/04_online/run.command
```

This invokes `--run --family diffusion --seed 1`. Its batch log, process ID,
start/finish timestamps and exit status are under `results/paper_final/04_online/logs/`.
The per-group logs are in the same folder. A stopped/incomplete group requires
inspection; the launcher never silently overwrites it.

The command for the full 18-group suite remains available for a later run:

```bash
caffeinate -i /opt/anaconda3/envs/rl/bin/python -u -m experiments.paper_final.run_04_online --run
```

`--output-root /absolute/path/PAPER_FINAL` can put all new results on another
disk. The default is `results/paper_final/04_online`. Native subprocesses have
single-thread BLAS/OpenMP settings and cleared ambient smoother/cycle overrides;
the JSON profile supplies 18/18/9. Keep the machine on AC power and avoid running
other timing workloads concurrently.

Default invocation is validation only. The wrapper uses one process at a time,
locks the output root, and keeps a log per group. On successful completion it
audits inputs, activation, time identities and reported totals before writing
the group's completion marker. Repeating the command verifies completed groups
and continues with the next unused group. It will not overwrite or automatically
restart an incomplete group, mix source versions, or silently continue after
unrecovered failures. Inspect partial runs before deciding how to resume them.

Allow at least about 6 GiB free for the complete suite. This is a planning
allowance based on previous local artifacts, not an upper bound: the launcher
checks 256 MiB per remaining group plus a 1 GiB reserve before each run. It
does not remove any existing files. September 17 preflight found about 10.5 GiB
free; the launch rechecks the available space for its selected groups.

Rebuild the completed Module 1/2 report without PDE solves:

```bash
/opt/anaconda3/envs/rl/bin/python -m experiments.paper_final.analyze_04_online
```

`--allow-partial` produces a clearly marked interim report. The suite writes
interim reports automatically after each completed group. Completing the three
selected groups still produces a partial report relative to the full 18-group
plan. There is no score-based seed selection or early stopping.

## Historical failure-feedback trial after the September 19 review

This opt-in trial has been withdrawn from active paper runs. The active suite
explicitly uses `rollback_unrecovered` for both learners; it does not retain
unrecovered observations or apply a penalty. The options below describe the
historical trial and remain available for reproducing it at its recorded
settings. No feedback-mode or penalty-size comparison is currently scheduled.

The joint runner now supports an explicit finite-budget objective
`L_H = C_H + Lambda * final_failure`. `C_H` is measured online E2E under the
existing component scope, including failed attempts, recovery and recurring
controller/bandit overhead. The penalty has units of seconds but is not a
measurement or an estimate of uncapped completion time.

Enable this protocol with a top-level `failure_feedback` object whose `mode`
is `budgeted_penalty` and whose `penalty_sec` is an explicit finite,
nonnegative number. Omitting the object, or selecting `rollback_unrecovered`
without a penalty, reproduces the historical failure-learning rule. No penalty
value has been prescribed for a new formal suite. Zero is allowed as an
explicit bounded-expenditure objective; it does not prioritize reliability.
Choose and record Lambda before the controlled comparison, and keep the cap
fixed. The new protocol currently supports composable online setup learning
from problem 1, including the delayed-RL prefix. Frozen setup replay, separate
setup warmup and PPO are rejected rather than silently using mixed protocols.

Valid final failures now update both setup targets: the cost model receives
the observed suffix cost plus Lambda, and the auxiliary failure head retains
the primary-attempt failure label. Both models therefore use the same accepted
observations and can continue sharing their precision matrix. Ordinary setup
selection uses the penalty-adjusted cost LCB. The auxiliary primary-failure
head still only controls reselection after construction failure; it does not
estimate the final failure probability of the complete recovery procedure.

Setup training retains the existing estimate of update overhead from the
previous problem. `bandit_learning_cost` records that actual training label;
it need not equal the retrospectively measured `penalized_cost`. The solve TD
target remains native cycle cost, plus uncharged recovery cost and Lambda on
a final failure. Controller overhead remains outside that TD target and is
included in the reported component cost and setup feedback. Earlier cycles
are not charged again at the terminal transition.

Finite failed solve episodes terminate with a zero successor feature and
retain their updates and episode covariance contribution. The complete-episode
LSTDQ coercivity identity requires the executed feature/trace sequence and
zero terminal feature, not successful numerical convergence. This statement
does not supply a confidence-coverage, Q-accuracy or learning-convergence
guarantee. Native breakdown can use the preceding finite decision features;
NaN residuals are not encoded into a successor state.

The new protocol rejects nonfinite or negative cost labels before the affected
learning update. Non-solver execution errors are propagated rather than turned
into failed solver observations. Controller arithmetic/feature errors restore
the episode snapshot and stop the run. Native-versus-Python clock assertions
were removed after a 0.252 ms clock discrepancy stopped the cap-200 run; those
environment-dependent comparisons do not belong in the learning algorithms.
Validation and wrapper work are covered by `method_wall_runtime`; the declared
component timing scope is otherwise unchanged. An interrupted or corrupted run
must not be resumed as an uncontaminated training trajectory.

Per-problem `failure_penalty_sec` is the applied penalty (zero on success).
`penalized_cost` is measured E2E plus that penalty. Both appear separately in
summaries; every runtime reduction continues to use measured time only.
The resolved protocol records the coefficient, target scopes and mode.

The selected setup arm is now captured before any update or rollback and
carried as `selected_arm_index`. Online and warmup logs use this immutable ID.
Historical raw files remain unchanged: where their top-level ID is stale,
use `outcome.primary_attempts[-1].arm_index` after checking the attempt's
parameters. The review's supplied stale-arm table already identifies the
affected historical records.

Tests cover retained final failures, recovered failures without a penalty,
penalty-free success, invalid measurements, estimator errors, batch/recursive
LSTDQ agreement, and correct arm IDs after rollback. Functional 8-cubed native
checks are stored under
`results/paper_final/04_online/preflight/20260919_failure_feedback/`.
Their 0.5-second penalty and short caps are test fixtures, not formal choices.
