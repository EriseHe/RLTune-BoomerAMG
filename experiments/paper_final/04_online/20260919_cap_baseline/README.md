# Module 04 — original-policy cap comparison

This suite restores the learning/recovery policy of the September 18 60³
diffusion–advection reference at commit `6576ed5`. The retained-failure and
failure-penalty trial is withdrawn from the active paper experiment plan.
The only experimental variable in this suite is the V-cycle cap: 100, 200, 500.

## Fixed protocol

- 60³ diffusion–advection, original forward DifConv stencil and coefficient
  ranges; relative residual tolerance 1e-6; 18/18/9 smoother profile.
- 5000 paired problems per method and cap. Default, LinUCB v4 setup-only,
  and LinUCB–LSTDQ v3. Setup learns from problem 1; RL starts on problem 1001.
- Same original seed tuple: base 56700120, bandit 56760120, controller
  56766120 and method order 56772120. The 5000 matrix/RHS inputs have hash
  `05abc1324ecade29d4215a4f7047a80621411502be0a396022dd39fbb973412f`.
- Same 7D setup context including intercept, 34D solve state, 306D Q features,
  action/candidate grids, exploration schedules, ridge and uncertainty settings.
- Independent fresh learners for each cap. No checkpoint reuse and no shared
  learned prefix. Sequential order is 100, 200, 500; within each group the
  methods run serially in seeded random order on each paired problem.

## Restored failure rule

Every config explicitly selects `failure_feedback.mode = rollback_unrecovered`.
There is no penalty coefficient. This restores the original behavior:

1. Successful primary: retain the usual update.
2. Failed primary followed by successful fallback: retain learning from the
   complete successful procedure, including the failed work and recovery cost.
3. Final failure after recovery: restore LinUCB's learning/coverage/failure-head
   state and LSTDQ's episode learning state. No final-failure training sample
   or penalty is retained. Selection randomness follows the original rules.

All attempted work and final failures still appear in physical runtime and
failure reports. Learning rollback never removes their recorded expenditure.
Setting a penalty to zero would retain failed observations and is therefore
not this protocol. This suite does not compare feedback modes or penalty sizes.

The previously corrected arm IDs, recovery accounting, LSTDQ inverse handling
and measured overhead scopes remain. Environment-dependent cross-clock guards
remain removed. Native C/HYPRE timers and libraries match the reference.
Changing the cap also changes the existing cycle/H feature; each controller is
trained fresh with its declared horizon. Runtime feedback can produce different
learning trajectories across executions even with identical random seeds.

## Launch and provenance

```bash
experiments/paper_final/04_online/20260919_cap_baseline/run.command
```

The launcher requires committed code, uses single-thread execution, and records
the exact source/config/native hashes. This machine's AC/sleep handling stays
in the experiment launcher. Keep the laptop plugged in and open.

New outputs are separate from all previous runs:
`results/paper_final/04_online/20260919_cap_baseline/cap_100`, `cap_200`, `cap_500`.
Logs are in the parent's `logs/` folder. Normal plots, trajectories, checkpoints
and per-cap reports are generated automatically. The previous penalty trial
is preserved as development history; its cap-500 continuation was stopped at
the user's request, with the last progress report at 4000/5000.

Cap 100 is the new same-code reference. This is one previously exposed
development seed; the three caps are not independent replications. Report all
5000 problems and the last 1000, measured time and failures, with every cap
listed separately. No cap is chosen based on an incomplete curve.
