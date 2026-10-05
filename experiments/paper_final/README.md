# Paper experiments

The current paper uses the accepted Module 04 online comparison and corrected
matched-hierarchy Run 05. Start with the [reproduction index](../../docs/reproduction.md)
for commands, exact artifact requirements, and timing scope.

## Current evidence

| Study | Protocol / reading entry | Execution |
|---|---|---|
| Module 04 online autotuning | [September 20 frozen design](04_online/20260920_formal/README.md), followed by two captured three-seed batches | `python -m experiments.paper_final.run_04_online --suite SUITE_JSON ...` |
| Matched-hierarchy Run 05 | [Corrected Run 05](05_policy/RUN05.md), six frozen checkpoints, 100 diffusion 60³ inputs, three repetitions, prescribed periodic pair `(2.85, 1.10)` | `python -m experiments.paper_final.run_05_policy_minimax ...` |

Module 04 compares Default, LinUCB, and LinUCB–LSTDQ over 5000 problems, with
RL active from problem 1001. Setup contexts have 4 diffusion or 7
diffusion–advection entries including the intercept. LinUCB v4 and recursive
LSTDQ v3 are implementation versions, rather than additional method names.

The September 20 suite contains six PDE/grid groups on one seed. The later
six-seed batches have separate captured suites and input hashes. Run 05 retains
the same checkpoints and cases examined in earlier matched-hierarchy work,
including the refreshed seed-4 checkpoint. Historical identifiers are retained
so commands and source snapshots remain traceable.

The separate solve-specific Module 05 checkpoint study and Module 06 frozen
complete-method study were removed from the current paper. Their protocols and
completed artifacts remain research history.

## Development and historical studies

| Directory | Purpose |
|---|---|
| [01_numerics](01_numerics/README.md) | Inverse recovery, stopping semantics, rollback, timing verification |
| [02_diagnostics](02_diagnostics/README.md) | Earlier fixed/schedule checks and functional diagnostics |
| [03_activation](03_activation/README.md) | Development studies of the RL activation boundary |
| [04_online](04_online/README.md) | Earlier online protocols; current frozen suite linked above |
| [05_policy](05_policy/README.md) | Matched-hierarchy policy history, including corrected Run 05 |
| [05_online_policies](05_online_policies/README.md) | Separate solve-specific checkpoint preparation, outside current paper scope |
| [06_policy](06_policy/README.md) | Separate frozen complete-method evaluation, outside current paper scope |

The withdrawn [penalized-feedback trial](04_online/20260919_cap_comparison/README.md)
is development history. Accepted online runs use `rollback_unrecovered`; this
cleanup does not change their recovery or learning objective.

The [September 29 progress record](../../docs/archive/paper_final_progress_20260929.md)
preserves the previous detailed chronology, numbers, and historical links.
The [paper completion plan](../../docs/theory/paper_completion_plan_20260915.md)
and [theory index](../../docs/theory/README.md) retain the mathematical background.
