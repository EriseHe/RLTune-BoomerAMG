# Same-seed activation-time sensitivity sweep

Status: proposed design, prepared on 2026-09-15 after the composite-activation
pilot. This document does not launch an experiment. The current shared-prefix
runner supports one fixed/dynamic pair; a shared reference trunk with several
fork times requires a targeted extension and validation before execution.

## Question

On the same prespecified 60³ diffusion stream, can a later RL start reduce
the complete 5000-problem online cost? Measure the sensitivity of the full
joint learner to start time, including the resulting setup learning path.

The preceding pilot started dynamic RL on problem 898 and used 50.273 seconds
more than fixed start 1001. Dynamic saved 5.130 seconds on problems 898–1000,
then lost 55.003 seconds on problems 1001–5000. This motivates a timing sweep;
it does not establish that later starts will improve monotonically.

## Six prespecified methods

Here tau counts completed reference problems; RL begins on tau + 1.
Setup learns from problem 1 and continues after activation.

| tau | Reference problems | First RL problem | RL problems |
|---|---|---:|---:|
| 750 | 1–750 | 751 | 4250 |
| 1000 | 1–1000 | 1001 | 4000 |
| 1250 | 1–1250 | 1251 | 3750 |
| 1500 | 1–1500 | 1501 | 3500 |
| 2000 | 1–2000 | 2001 | 3000 |
| infinity | 1–5000 | never | 0 |

Infinity is an online LinUCB setup learner with reference solve throughout,
not a fixed-default-setup baseline. Do not force activation on the last problem.

Keep the grid, 5000 matrix/RHS inputs, seed tuple, setup and solve parameters
from `configs/paper_test_n60_composite_activation_seed1.json`. Its input hash is
`e79ce1e088712599abd69a0b6b48cfbea54fbcf6c684b057353bd64af38ec823`.
All finite-tau controllers have the same initial seed and untrained state.
Exploration and controller learning start when that branch is enabled; do not
age the controller schedule during reference problems.

Run tau=1000 again in this batch. The previous fixed/dynamic results remain
historical context, rather than a replacement for the contemporaneous baseline.
Actual timing feedback means identical seed values alone do not reproduce an
identical setup training path across separate native runs.

## Shared reference trunk and forks

Execute one continuously learning reference-solve trunk, also serving as the
infinity method. At each finite tau, after completing that problem, copy its
complete setup decision state into that method's independent branch. Activate
the branch's untrained LSTDQ controller on the following problem.

Examples:

- Problems 1–750 are measured once and belong to all six logical methods.
- Problems 751–1000 execute the tau=750 branch and the reference trunk; the
  latter supplies the still-unstarted methods' prefix.
- At problem 1000, fork tau=1000 from the current reference trunk, not from
  the already RL-controlled tau=750 branch.
- Continue similarly at 1250, 1500 and 2000. After 2000, execute all six
  methods independently on each shared input.

Reuse the existing mutable-state save/load and fork validation helpers.
Preserve regression/failure arrays, candidate statistics, history used by
structured-candidate anchors, RNG state, AOT cursor, previous-update timing
estimate and counters. Verify the newly enabled controller is untrained at
each fork. Share immutable AOT candidate data, with independent mutable cursors.

Native executions remain sequential. Randomize the order of physical active
branches and the reference trunk on each problem with a fixed method-order
seed. Mark inherited rows with their physical source. Count each shared-prefix
cost once per logical method; report actual physical executions separately.
One-time fork/preparation work is outside online cost and reported separately.

There are 30000 logical method-problems and exactly

    5000 + (5000-750) + (5000-1000) + (5000-1250)
         + (5000-1500) + (5000-2000) = 23500

physical method-problems before additional recovery attempts. This is about
2.58 times the preceding pilot's 9103 physical method-problems. A rough initial
wall-time allowance is 1.5–2 hours, to be revised after measuring progress;
reference solve and later learning states may have different costs.

## Prespecified evaluation

Primary: cumulative online end-to-end cost over all 5000 problems, including
reference prefixes, setup, solve, controller, bandit and bounded recovery.
Report all six values and differences relative to the new tau=1000 baseline.
Also compare finite starts with infinity to quantify the realized benefit of
enabling RL in this stream.

Secondary: cumulative cost curves; common problem blocks and last 1000;
setup/solve/overhead and recovery decomposition; first-primary and unrecovered
failures; actual controller steps; cycle/action summaries; setup-choice changes.
The common evaluation horizon intentionally leaves fewer RL training problems
for later starts. Do not replace the primary metric with equal post-start
training lengths or select a favorable suffix after seeing the results.

Interpret the six-point curve as sensitivity on this particular seed. Report
the complete curve, including infinity, and do not call the winning grid point
a generally optimal start time. Repeated runs or other prespecified seeds would
be needed to separate systematic start-time effects from learning-path and
timing variability. The sweep does not by itself identify the mechanism behind
the previous dynamic/fixed difference.

No p0/delta/onset-prior changes, additional PDE families, or paper changes are
part of this proposed sweep.

## Required implementation checks

- Forks occur after tau; RL first acts on tau+1; infinity never acts with RL.
- Pending methods inherit exactly the same reference rows and complete state.
- Fork targets have independent mutable learner/controller state and AOT cursors.
- All six methods have 5000 ordered records with identical matrix/RHS inputs.
- Shared costs, physical execution counts and online components reconcile.
- Existing fixed/dynamic shared-prefix behavior retains its tested semantics.
- A small mocked multi-fork test and native smoke precede the 60³ run.
