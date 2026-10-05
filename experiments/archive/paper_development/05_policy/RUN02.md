# Module 05 — Run 02

This is a fresh timing replication of selected policies, using the six existing
RL checkpoints, the same 100 test inputs, and the exact saved **Joint** setup
choices from `20260927_diffusion60_6seeds_100cases` (Run 01). No learner trains.

Run 02 uses **three repetitions of every problem**, up to three independent
single-thread workers, randomized case and policy order, and the same native
build, Python environment, tolerance, cap, residual monitoring and recovery
protocol as Run 01. The supervisor prevents sleep until computation and figures
finish. The existing source/checkpoint hashes and native preflight are checked.

## Compared policies

1. Default `w=1`.
2. The global fixed weight selected by Run 01's 41-weight test-hindsight scan.
3. The per-instance fixed weight selected by that same Run 01 scan.
4. Development-tuned periodic `(2.5,1)`.
5. Theory-motivated periodic `(1,3)`.
6. Frozen RL, with unchanged LCB rule and exploration disabled.

The `(3,1)` ordering is excluded at the user's request before launch.

There is no new weight/schedule search. All selected policies are timed anew.
If global and per-instance fixed choices coincide for a problem, one native
execution contributes to both labels. Retimed fixed selections are not claimed
to be newly optimal under Run 02 timings. Recovery and final failures stay in
the accounting; unsuccessful outcomes are never used to reselect a policy.

Main action/residual comparison heatmaps show exactly four methods: global
fixed, per-instance fixed, **Periodic (1,3)**, and RL. They omit default and the
tuned (2.5,1) schedule. The (1,3) ordering is chosen before Run 02 from Run 01,
where it beat (3,1) on every seed. The two retained periodic pairs have their own
comparison heatmap. Other figures
include `w=1`. Problem rank is horizontal and AMG cycles increase upward.
The baseline-difficulty ordering is identical to Run 01.

## Files and operation

Code is `run_05_policy_repeat.py`, `analyze_05_policy_repeat.py`, and
`plot_05_policy_repeat.py` in `experiments/paper_final/`.

Default outputs are kept separately in
`results/paper_final/05_policy/20260927_run02_diffusion60_joint_6seeds_100cases/`.

```sh
python -m unittest experiments.paper_final.test_05_policy_repeat
python -m experiments.paper_final.run_05_policy_repeat prepare
python -m experiments.paper_final.run_05_policy_repeat preflight
python -m experiments.paper_final.run_05_policy_repeat run --workers 3
python -m experiments.paper_final.run_05_policy_repeat watch
```

The supervisor automatically audits full coverage, analyzes all repetitions,
compares Run 01 against Run 02, and creates PDF/SVG/600-dpi PNG figures with an
offline gallery. Completion requires the figures as well as the native trials.
Resume with the saved worker count and unchanged prepared inputs. Closing the
read-only progress viewer does not stop the background supervisor.
