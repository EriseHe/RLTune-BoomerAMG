# Module 05: prepare solve-specific frozen checkpoints

Updated September 28, 2026. This module trains the setup selectors needed by
Module 06. Module 04 retains its accepted protocol and results.

**Complete:** the five-branch run finished September 28 at 06:10 New York
time after 54.81 minutes. All 21,000 physical method–problem executions
completed with zero unrecovered failures. Final audits and all five setup
checkpoint roundtrips passed; the LSTDQ controller is also saved and verified.
See the [training report](../../../results/paper_final/05_online_policies/20260928_shared_prefix/training/TRAINING_REPORT.md)
and [frozen artifact manifest](../../../results/paper_final/05_online_policies/20260928_shared_prefix/training/frozen_artifacts.json).

The [full-5000 and final-1000 figures](../../../results/paper_final/05_online_policies/20260928_shared_prefix/training/analysis/paper_figures/module05_runtime_comparisons.pdf)
reuse Module 04's native/end-to-end stacked-bar renderer. The
[cost report](../../../results/paper_final/05_online_policies/20260928_shared_prefix/training/analysis/paper_figures/REPORT.md)
explains why RL is slightly cheaper cumulatively while Periodic is already
cheaper in the final thousand. All five branches, shared-prefix accounting,
controller/setup-learning overhead and recovery costs remain included.
The [combined Module 05/06 archive](../../../results/paper_final/releases/module05_module06_20260928.zip)
contains the current measurements, checkpoints, figures and reproduction sources.

The original four-branch training run was stopped and deleted at the user's
request. The fresh configuration adds **fixed 1.60** as a fifth branch and
starts again from the beginning. The
fresh five-branch run launched September 28, 2026 at 09:15 UTC. The
[launch status](../../../results/paper_final/05_online_policies/20260928_shared_prefix/launch.json)
records completion or failure; the
[training progress](../../../results/paper_final/05_online_policies/20260928_shared_prefix/training/progress.json)
updates every 100 inputs. One diffusion 60³ training replicate is locked.
Default-only calibration selected **w_dev = 1.40** from 4100 trials. The
added **1.60** is the pooled historical learned-hierarchy fixed-grid minimum.
Ten integration tests and the revised native functional check passed:
32 method–problem executions, 63 native attempts including recovery, and
zero unrecovered failures. All five checkpoint roundtrips passed.

[Preparation audit](../../../results/paper_final/05_online_policies/20260928_shared_prefix/READY.json)
· [Calibration report](../../../results/paper_final/05_online_policies/20260928_shared_prefix/default_calibration/REPORT.md)
· [Historical 1.60 selection](../../../results/paper_final/05_online_policies/20260928_shared_prefix/historical_fixed_reference.json)
· [Setup-only hierarchy 1.60 selection](../../../results/paper_final/05_online_policies/20260928_shared_prefix/setup_only_fixed_reference.json)
· [Locked training configuration](training.json)
· [Launch script](run.command)

## Training contract

Execute problems 1–1000 once with LinUCB setup learning and W1 solves.
Clone the complete resulting setup state into five independently mutable
branches, including regression and failure models, candidate statistics,
candidate cursor, random state, selection history, and the previous-update
cost estimate. Each branch then processes the same next 4000 inputs, learns
from its own costs, and can choose different hierarchies.

| Branch | Problems 1001–5000 | Frozen output at problem 5000 |
|---|---|---|
| W1 | Setup learning with w=1 | W1-adapted setup selector |
| Fixed 1.40 | Setup learning with Default-development-selected w_dev=1.40 | 1.40-adapted setup selector |
| Fixed 1.60 | Setup learning with historical learned-hierarchy reference w=1.60 | 1.60-adapted setup selector |
| Periodic | Setup learning with (2.85,1.10) | Periodic-adapted setup selector |
| RL | Setup learning alongside the existing online LSTDQ | RL-adapted setup selector and LSTDQ controller |

This is 1000 + 5×4000 = **21,000 actual method–problem executions** per
replicate, before reselection/recovery. Each branch has 5000 logical training
exposures. Prefix rows are copied into each branch log for alignment; the
physical preparation cost counts the prefix once.

Keep the accepted Module 04 search space, LinUCB settings, diffusion context,
LSTDQ features/hyperparameters, tolerance 1e-6, cap 50, one-pre/one-post sweep
profile, failure rollback, and Default/W1 fallback. Periodic applies the
same weight to a complete V-cycle and resets high first on every primary
attempt. LSTDQ remains untrained during the common prefix.

Native execution is serial, with one MPI rank and one library thread.
After branching, permute method order independently within each problem.
Use one fixed training stream; multiple training seeds and other grids
require a later protocol decision. Equal training opportunities do not imply
that any branch has found its optimal setup policy.

## Default-only constant calibration

Per the user's choice, select w_dev only on **Default setup hierarchies**,
using 100 separate diffusion 60³ development inputs and all 41 weights in
{1,1.05,…,3}. Execute one timing trial per input and weight, in randomized
within-input order: 4100 primary trials, plus recovery. No W1-adapted or
RL-adapted hierarchy data select this coefficient.

Among constants completing every development input, minimize mean accounted
continuation time: native solve, policy dispatch, and all recovery setup,
solve and dispatch. Initial hierarchy construction is the same configuration
for every weight and is excluded from this selection criterion; its measured
cost remains in the logs. Ties select the smaller weight. Freeze the chosen
coefficient before training; do not select it again using training or test
results. The selected value is **1.40**, with 54.391 ms mean inclusive continuation
time and no recovery on the 100 calibration inputs. The label is
**Default-setup development-selected fixed weight**.
It is not an optimum for all subsequently learned hierarchies.

Calibration inputs use seed 92805101. The training stream uses eight groups
of 625, starting with seed 92815001 and increments of 6000, followed by the
fixed shuffle seed 92863001. Calibration, training, and functional-check
inputs are checked for overlap. Module 06 will use fresh evaluation inputs.

## Additional historical fixed reference

The user's requested additional branch retains **1.60**, independently of
the Default-only calibration. Reconstructing all 29,520 retained fixed-grid
records from the accepted six-checkpoint Joint study, including refreshed
seed 4, selects 1.60 by pooled native continuation time. Prescribed timing
repetitions are averaged within each case before summing; every eligible
weight must complete every case. The mean over 600 seed–case combinations
is 79.818 ms for 1.60 and 82.621 ms for 1.65. The individual seed minima are
1.55 for seed 1 and 1.60 for seeds 2–6.

Label this **historical learned-hierarchy fixed reference**, not a
Default-setup or universally optimal coefficient. The source was a hindsight
scan on the earlier study's test inputs; those results now motivate this
prespecified reference. Module 06 must use fresh inputs, including separation
from that earlier scan. Both fixed branches receive their own independent
setup learner after the same shared prefix.

The retained setup-only-hierarchy scan was also independently reconstructed
from all 29,520 fixed-weight trials at the user's request. Its pooled minimum
is likewise **1.60**: 80.229 ms versus 82.311 ms for 1.65. Individual minima
are again 1.55 for seed 1 and 1.60 for seeds 2–6. Consequently the single
1.60 branch represents the historical fixed choice from both hierarchy
sources; there is no second distinct coefficient to add. Its own setup
learner still starts from the common W1 prefix and adapts afresh.

## Artifacts and acceptance checks

Save the shared-prefix checkpoint and a fork audit for each continuation.
At problem 5000, save all five setup statistics checkpoints, supplementary
candidate/selection history, the LSTDQ checkpoint, configuration, stream and
source hashes, and both fixed-weight selection records. Verify checkpoint
roundtrips and numerical/recovery/timing identities.

Training costs, first-1000/active-4000/final-1000 windows, failure counts and
setup trajectories are diagnostics. **Training victory or payback is not an
acceptance condition.** Do not treat problems in one adaptive stream as
independent training replicates.

The local Module 05 adapter reuses the shared online loop, prefix cloning,
setup update/recovery, fixed solves and LSTDQ execution. It extends the
allowed roster to include delayed fixed/periodic continuations without
changing Module 04's configuration parser or numerical code.

## Commands

Use the accepted Python environment from the repository root:

```sh
/Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python -m experiments.paper_final.calibrate_05_default_weight --run
/Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python -m experiments.paper_final.run_05_checkpoint_training --prepare
/Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python -m experiments.paper_final.run_05_checkpoint_training --check
```

The passed functional check uses 2 shared + 6 continuation problems on a separate
60³ stream; its checkpoints are explicitly ineligible for Module 06. It checks the same code paths; unit tests separately verify the
actual 1000/1001 activation boundary. The full command is explicit:

```sh
caffeinate -i /Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python -m experiments.paper_final.run_05_checkpoint_training --run
```

Full training refuses to start without a passed functional audit and refuses
to overwrite an existing run. New outputs use
`results/paper_final/05_online_policies/20260928_shared_prefix/`.

## Module 06 handoff

The primary fresh-input evaluation pairs each frozen setup selector with its
own solve policy. Include setup selection and construction, solve execution,
controller computation, and recovery; neither learner updates on test data.

A small crossed comparison of periodic/RL solves on periodic-adapted and
RL-adapted hierarchy sources addresses compatibility. It is complementary
to the complete frozen-method comparison and does not require a four-by-four
study. See [Module 06](../06_policy/README.md).

[Original paper plan](../../../docs/theory/paper_completion_plan_20260915.md)
· [Period-two mathematical basis](../../../docs/theory/period_two_weighted_minimax_20260928.md)

The review's claimed commit ee1f2ab was not present in this checkout. These
local planning edits implement the user-approved design; no such commit is
claimed as local provenance.
