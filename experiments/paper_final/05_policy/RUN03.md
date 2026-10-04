# Run 03 — repeat-trained seed 4 Joint checkpoint

This diagnostic follows the user's request to repeat Module 04 diffusion 60³
for seeds 4, 5, and 6. All three runs repeat the original 5,000 inputs and
initial random seeds with fresh learners. Original results are retained.
The Module 04 repeat's `comparison.json` reports all-5,000 and last-1,000
no-overhead reductions for both versions. `preferred_results.json` identifies
better repeats by full-5,000 reduction and explicitly records outcome selection.
It does not overwrite the original prespecified evidence.

The user explicitly selected **new seed 4 setup + RL** for this Module 05
follow-up. Seeds 1, 2, 3, 5, and 6 use their original checkpoints, including
the original setup choices for the 100 common test problems. The new Module 04
seeds 5 and 6 are timing-sensitivity diagnostics and are not substituted here.
Seed 4's new Joint checkpoints are used in this follow-up regardless of whether
the Module 04 repeat outperforms the original.

The evaluation keeps the same 100 test problems, numerical environment,
recovery, candidate schedule, and six policy roles as Run 02. Seed 4's setup
selector is frozen: its candidate cursor is reset for evaluation, the 100 old
development contexts consume the same initial rows, and the following rows
select the test hierarchies without learning from outcomes.

Because seed 4's hierarchy choices can change, its fixed baselines are selected
again from all 41 weights (1.00 through 3.00, step 0.05), on all 100 cases.
The same preselected 10 cases from Run 01 receive two additional repetitions.
This is 4,920 scan evaluations. Global and per-instance choices minimize native
continuation cost averaged over their prescribed repetitions, restricting
choices to successful complete procedures. These are best-observed grid
diagnostics. All attempts and recovery remain charged.

After freezing these choices, all six policy roles on all six checkpoints
receive **three fresh timing repetitions per case**. This independently retimes
the selected constants; no fastest repetition is picked. Other seeds keep
their original fixed choices. The prior development-selected (2.5,1) schedule
and periodic (1,3) are retained; neither is claimed newly optimal for seed 4.
There is no (3,1), prefix-tail, or setup-only-hierarchy evaluation.

The Module 04 supervisor automatically launches this follow-up after all
three training repeats pass their completion audits. It prevents sleep
through both stages. At most three single-thread native workers run at once.

Results are under
`results/paper_final/05_policy/20260927_run03_new_seed4_joint_6seeds_100cases/`.
Code is `run_05_policy_refresh.py`, `analyze_05_policy_refresh.py`, and
`plot_05_policy_refresh.py`. The summary compares policy roles against Run 02,
explicitly marking seed 4's changed checkpoint/hierarchies. Its figures choose
the best observed seed by cumulative no-overhead reduction versus matched
weight 1, label that selection, and retain all six checkpoints in aggregates.

To regenerate the final exports with legible seed legends and a tighter RL
atlas cycle axis, run `python -m experiments.paper_final.polish_05_policy_refresh_figures
--output results/paper_final/05_policy/20260927_run03_new_seed4_joint_6seeds_100cases`.
This separate presentation revision preserves all hash-frozen experiment
sources and numerical results. Its source is included in the figure pack,
and its changes are recorded in the figure provenance.
