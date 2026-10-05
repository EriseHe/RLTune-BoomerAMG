# Module 04 — final six-group run

User-approved protocol on September 20: one seed, both PDE families, all three
grid sizes, and cap 50 throughout. All six groups start fresh using the current
original-policy implementation. Previous results remain separate.

## Frozen design

- Order: diffusion 40³, advection 40³, diffusion 60³, advection 60³,
  diffusion 80³, advection 80³. Groups and native solves run serially.
- 5000 paired problems per group and method: Default, LinUCB v4 setup-only,
  and LinUCB–LSTDQ v3. Method order is randomized per problem with a fixed seed.
- Setup learns from problem 1; RL starts at problem 1001. No imported
  checkpoints, shared learned prefixes, or additional training problems.
- Both families use cap 50, relative residual tolerance 1e-6, and the 18/18/9
  smoother profile. Preserve the forward DifConv discretization for advection.
- Setup contexts include the intercept: diffusion 4D, advection 7D. All action
  grids, candidate generation, exploration and LSTDQ settings are unchanged.
- `failure_feedback.mode = rollback_unrecovered`. Primary failures followed
  by successful fallback remain learned. Final unrecovered failures roll back
  both learners; there is no failure penalty. All failed work, fallback cost
  and measured overhead remain in the reported time.

## Remembered seed and provenance

| RNG | Seed |
|---|---:|
| Base | 56700120 |
| Bandit | 56760120 |
| Controller | 56766120 |
| Method order | 56772120 |

The eight input seeds are `56700120 + 6000*k`, for k = 0,...,7; the stream
shuffle seed is 56748120. Configs and `suite.json` preserve the six established
matrix/RHS stream hashes. The six groups reuse the same seed tuple and are
different problem settings, not six independent seeds.

The user explicitly chose this seed after the completed cap 50/100/200/500
development comparison. These are final-paper runs on an already examined
seed, not an untouched holdout or an additional independent replicate of the
development results. Report that provenance when describing the experiments.

The learning, solver, native-library and timing implementations are unchanged
from the completed original-policy cap comparison. Launch records the actual
commit and all source/config/native hashes; no source changes are made while
the suite runs. There are no machine-specific cross-clock stop checks.

## Launch and outputs

```bash
experiments/paper_final/04_online/20260920_formal/run.command
```

The existing suite runner executes the six groups, audits each completed run,
and updates the combined report. The local launcher uses one compute thread,
requires AC power and keeps the machine awake. It runs independently of Codex.

Results: `results/paper_final/04_online/20260920_formal/`.
Each named group contains its plots, trajectories, learner snapshots and
reports; all launch and group logs are under `logs/`. The combined report is
`analysis/module04.md`, updated after each group completes.

Primary reporting includes overhead and all 5000 attempted problems, alongside
the last 1000 and final failure counts. Also report setup + native solve without
learning overhead. The time metric excludes one-time initialization, AOT
preparation, matrix/RHS assembly and output I/O; subprocess elapsed time is
recorded separately. When failures remain, this is attempted-solve cost under
the fixed budget, not the cost of successfully solving every input.

Previous matching runs total approximately 10.6 hours across the six groups.
Allow roughly 11–13 hours for this launch; realized learning paths and machine
conditions can change that estimate.
