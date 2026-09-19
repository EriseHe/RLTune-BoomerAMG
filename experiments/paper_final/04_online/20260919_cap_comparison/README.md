# Module 04 — 60³ diffusion–advection cap comparison

Cap 100 completed at commit `f24e11f`. The incomplete cap-200 run was deleted
at the user's request after a cross-clock assertion stopped it. The remaining
caps now start fresh using `remaining.json` and `run_remaining.command`, with
results under `results/paper_final/04_online/20260919_cap_comparison/restart/`.
The completed reference and its original source manifest stay at the parent
location. The new launch has its own source manifest; no completed run is
relabeled with the newer code version.

The cleanup removes per-setup, per-cycle and per-method comparisons between
native MPI time and Python time, along with their environment-variable switch.
It also removes the duplicate Python setup/cycle stopwatches. Native C/HYPRE
timers, cost scopes, failure target, algorithms and input configurations are
unchanged. Finite nonnegative costs and valid features remain required for
learning. Python still measures learning overhead; the complete method-call
stopwatch remains reporting-only in experiment code. Reporting does not reject
an otherwise finite record because the two independent clocks disagree.

This is one post-fix development comparison before choosing the cap for a final
multi-grid suite. It runs the complete 5000-problem stream for each setting:

| Cap | Role |
|---:|---|
| 100 | Reference |
| 200 | Main untested alternative |
| 500 | High-budget experiment; no presumed superiority |

Every setting contains Default, LinUCB v4 setup-only and LinUCB–LSTDQ v3.
Each method processes all 5000 paired inputs, including the first 1000 setup
learning problems. RL begins on problem 1001. All learners start fresh for
each cap. Groups run sequentially in the order 100, 200, 500; within a group,
the three methods run in a seeded randomized order on each problem.

## What is held fixed

The base seed is 56700120, bandit seed 56760120, controller seed 56766120 and
method-order seed 56772120. All three settings have the identical matrix/RHS
stream hash, `05abc1324ecade29d4215a4f7047a80621411502be0a396022dd39fbb973412f`.
The seed was already used for development and is not an untouched test seed.
Three caps are three settings, not three independent training replicates.

Use the existing forward DifConv discretization, coefficient ranges [1,1000],
18/18/9 smoother profile, tolerance 1e-6, 7D setup context including intercept,
34D solve state and 306D Q features. Action/candidate grids, exploration and
learner hyperparameters are unchanged. The cap applies to the primary solve
and the existing restarted fallback. Bounded setup-construction reselection
retains its existing rule.

Changing H changes both the available work and the controller's cycle/H state
coordinate. Each controller is therefore trained under its own declared H;
no checkpoint or learned prefix is reinterpreted under a different horizon.
The comparison measures complete online frameworks at each budget.

## Exact failure target and provenance

The corrected implementation was committed and pushed as
`d22753b7dbed533b6af4fa7f346d6d081fcc57e8`. Its predecessor `e0aa897` used the
historical final-failure rollback rule. The launch records the exact newer
commit that contains this cap-comparison configuration and runner changes.

All caps use the same explicit objective:

    L_H = C_H + 2.378 seconds * final_failure.

C_H is measured finite expenditure, including failed attempts, fallback and
declared recurring controller/bandit overhead. Final failure means the complete
primary/recovery procedure did not converge. The penalty is neither elapsed
time nor an estimate of eventual completion time; runtime and failures remain
separate reporting quantities. The common coefficient makes the objective
comparable across caps instead of changing the penalty together with H.

The value 2.378 seconds was fixed before any new advection learning outcome,
using independent Default measurements at 60³ and cap 100. The rule was twice
the sum of median setup cost and 100 times median native cost per cycle,
rounded up to milliseconds. The coefficient and rule are preserved in commit
`d22753b`; the canceled suite's raw calibration and run results were deleted
at the user's request. This comparison carries forward the committed
coefficient without retuning it against cap outcomes.

Setup cost observations include the finite cost plus the final penalty. The
auxiliary risk head retains its primary-attempt failure meaning and is used
for construction reselection; ordinary selection uses the penalized cost head.
The cost/risk heads use the same accepted observations and precision updates.
Setup still estimates update overhead using the prior update's measured cost.

LSTDQ retains valid failed episodes and terminates with a zero next feature.
Its terminal cost includes the last cycle, uncharged recovery and the final
penalty once. Its TD target remains native work rather than controller
overhead. Finite failure observations retain the episode's covariance update.
Invalid measurements, nonfinite encoded features and non-solver execution
errors stop the run and do not become valid numerical-failure samples.

Retained failure signals remove the earlier cost of discarding those valid
episodes. This does not establish that cap 500 improves learning, reliability
or cumulative runtime. The outcome is to be measured.

## Evaluation and execution

Prelaunch verification passed 110 unit/integration tests and a native replay
of one known difficult 60³ input at all three horizons. All three finite failed
episodes retained their controller update and a valid LSTDQ inverse; the
cap-500 replay ended in numerical failure before its full budget. These are
functional checks, excluded from the 5000-problem comparison. Their records
are in `preflight/horizon_verification.json` under the result directory.

Report all-5000 and last-1000 measured E2E and native time, recovery cost,
primary/final failures, and penalized cost for all caps. Compare absolute cost
and failure counts with cap 100, as well as each method's reduction against
its own paired Default. Keep all prescribed outcomes; no outcome-based cap
stopping or seed changes. Timing/estimator integrity failures still abort.
The report labels every cap separately and never pools caps as replicates.

```bash
experiments/paper_final/04_online/20260919_cap_comparison/run_remaining.command
```

The launcher requires committed source and waits for AC power before starting.
Keep the laptop plugged in and open. It enables idle-sleep prevention and
single-thread execution. Once started it continues independently of Codex.
Source/config/native hashes and the exact Git revision are recorded at launch.

The original full-comparison results and normal plots/checkpoints are under
`results/paper_final/04_online/20260919_cap_comparison/cap_100`, `cap_200`, and
`cap_500`; logs stay in the parent's `logs/` folder. Fresh 200/500 outputs and
their logs use the `restart/` subdirectory described above. Per-cap results and the
aggregate report are generated automatically. Earlier historical suites are
not changed. Budget roughly 5–8 hours provisionally; cap 500 is unmeasured and
learned trajectories can move that estimate substantially.
