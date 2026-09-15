# PAPER_FINAL: Modules 1 and 2

Prepared after local code checkpoint `ce609c1`. The new preparation is not
committed. No formal PAPER_FINAL experiment was launched during preparation.
The professor's Overleaf template and manuscript are outside this code change.

Preparation checks passed: 183 solve-control tests, 14 problem tests, and all
18 config/stream/context validations. Native experiment dispatch was mocked in
the suite scheduling/resume tests; the formal experiments remain unstarted.

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
| 1 | 4000063 | 56700120 | 56760120 | 56766120 | 56772120 |
| 2 | 5000077 | 57700134 | 57760134 | 57766134 | 57772134 |
| 3 | 6000083 | 58700140 | 58760140 | 58766140 | 58772140 |

All eight generator seeds and the shuffle seed are shifted by the same offset;
the group sizes remain 8 × 625 = 5000. Method seed offsets are zero. Existing
factory-specific internal RNG offsets are retained. The same replicate tuple
is reused across grid sizes and families; comparisons across those settings
must not be counted as additional independent replicates.

`configs/PAPER_FINAL/suite.json` fixes all 18 config hashes, input-stream hashes,
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

The existing per-run plot generator uses the paper labels Default, LinUCB and
LinUCB–LSTDQ for this roster. Historical context/activation plots keep their
distinct labels. The aggregate paper tables are written to
`analysis/modules_1_2.md`, with auditable numbers in `modules_1_2.json`.

## Commands

From the repository root, validate without native experiments:

```bash
/opt/anaconda3/envs/rl/bin/python experiments/joint/solve_control/run_paper_final.py --validate-only
```

Only when ready to start, run all 18 groups sequentially:

```bash
caffeinate -i /opt/anaconda3/envs/rl/bin/python -u experiments/joint/solve_control/run_paper_final.py --run
```

`--output-root /absolute/path/PAPER_FINAL` can put all new results on another
disk. The default is `results/joint/PAPER_FINAL`. Native subprocesses have
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
does not remove any existing files. Disk space was below 1 GiB during preparation,
so it must be freed before launch or an output root on another disk selected.

Rebuild the completed Module 1/2 report without PDE solves:

```bash
/opt/anaconda3/envs/rl/bin/python experiments/joint/solve_control/analyze_paper_final.py
```

`--allow-partial` produces a clearly marked interim report. The suite writes
interim reports automatically after each completed group and the final report
after all 18 groups. There is no score-based seed selection or early stopping.
