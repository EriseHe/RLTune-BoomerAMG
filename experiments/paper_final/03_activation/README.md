# 03 — RL activation time

## Current follow-up — s2, early starts

The user requested one new development seed with **only activation after 250, 500 and 750**. Keep **80³ diffusion–advection, 5000 inputs per branch**, LinUCB v4 with a 7D context including intercept, LSTDQ v3, and all previous setup/solve/recovery/timing settings. This does not launch stage 04.

| Branch | Fixed-weight prefix | First RL input |
|---|---|---:|
| `start_250` | 1–250 | 251 |
| `start_500` | 1–500 | 501 |
| `start_750` | 1–750 | 751 |

The latest-starting branch supplies the common setup-learning prefix. Fork its full mutable setup state at 250 and 500, then let each branch learn independently. The 750 branch activates its own initially untrained controller on input 751. There is no additional setup-only or later-start branch. Each logical trajectory includes its prefix cost once; this requires **14,250 physical primary executions**, plus recovery attempts.

- Configuration: [early_starts.json](early_starts.json).
- New seed tuple: base 92317018; bandit 92377018; controller 92383018; method order 92389018. The previously prepared, unused s2 stream is retained with its original hash.
- Stream SHA256: `e613d87b3a006ccf4e41d7de54d16de23ed8a4944eb14df2ebcf11264531826d`.
- The three branches differ only in activation boundary. All settings outside seeds and branch roster match the completed s1 run.
- Primary comparison: all-5000 recorded cumulative cost among the three starts. Also report last-1000 cost, failures and cost components. This is a fresh-seed development follow-up, not a formal robustness study.

### Run and read

```bash
./experiments/paper_final/03_activation/run.command
```

The standalone launcher uses the `rl` environment, one thread, and `caffeinate`. Existing results are never overwritten. The supervisor automatically performs the completion audit and generates the usual figure pack. The command above refuses a second launch once the output folder exists.

[All activation runs](../../../results/paper_final/03_activation/REPORT.md) · [s1 report](../../../results/paper_final/03_activation/advection_80_s1/REPORT.md) · [s2 report/status](../../../results/paper_final/03_activation/advection_80_s2/REPORT.md)

Each run has its own `REPORT.md`, `figures/`, `logs/`, `audits/` and `provenance/`. Raw `trajectories/` and `checkpoints/` remain directly under the run folder. A hidden launch-staging folder exists only during preparation, then its contents move into the run automatically after the native runner's overwrite check.


## Figures

The completed s1 run has the usual ten-figure joint plot pack in
[`results/paper_final/03_activation/advection_80_s1/figures/`](../../../results/paper_final/03_activation/advection_80_s1/figures/README.md).
Regenerate it from existing records with the standard plot-only entry point:

```bash
MPLBACKEND=Agg python experiments/joint/solve_control/generate_joint_experiment_plots.py \
  --result-dir results/paper_final/03_activation/advection_80_s1
```

These remain single-seed development figures. Recorded costs include failed
attempts; read the accompanying failure plot and report. Weight trends average
available records in each trailing window, so a missing action trace does not
hide subsequent recorded weights.

## Completed first run — s1

**Authorized to run before completing stage 02.** The user requested stage 03
first so its current-version checkpoints can supply the remaining diagnostics.

Use only **80³ diffusion–advection**, 5000 problems per logical method, with
one development replicate, explicitly confirmed by the user. All six methods learn setup online from
problem 1 through problem 5000 using LinUCB v4.

| Short name | Solve policy |
|---|---|
| `setup_only` | Fixed weight 1 throughout; no solve RL |
| `start_750` | Weight 1 for problems 1–750; LSTDQ v3 starts on 751 |
| `start_1000` | Weight 1 for problems 1–1000; LSTDQ v3 starts on 1001 |
| `start_1250` | Weight 1 for problems 1–1250; LSTDQ v3 starts on 1251 |
| `start_1500` | Weight 1 for problems 1–1500; LSTDQ v3 starts on 1501 |
| `start_2000` | Weight 1 for problems 1–2000; LSTDQ v3 starts on 2001 |

`setup_only` answers whether adding RL pays off. The five RL branches answer
when starting it changes cumulative cost. Fully default AMG belongs to stage
04's main comparison; calling the setup-learning baseline `default` would be
ambiguous.

Use one physical setup-only reference path. At each specified boundary, clone
its full mutable setup state, candidate cursor/history and RNG into the RL
branch. Its controller starts from the common untrained initialization. Once
forked, each branch learns from its own costs and may choose a different setup.
This estimates the effect on the **whole joint learning path**; it does not
isolate RL under a permanently identical hierarchy.

Each logical curve includes its shared prefix cost once. The nested execution
requires `5000 + 4250 + 4000 + 3750 + 3500 + 3000 = 23500` physical primary
method-problem executions in this diagnostic; recovery attempts are additional.
The first run used only `advection_80_s1`. The original full six-branch s2/s3 configurations remain unexecuted; the new s2 follow-up has its own three-branch configuration above.

Primary: all-5000 cumulative recorded cost versus `setup_only`, including failed attempts.
Secondary: paired difference versus `start_1000`, cost components, failures,
cycles, training steps, hierarchy divergence and time-feature saturation.
Report this single development trajectory; do not infer seed robustness. A short post-activation segment cannot establish total
payback for a late start.

The originally prepared three seed tuples and hashes remain in `suite.json` as
provenance. The user reduced the requested scope to replicate 1 before its first
RL activation, for development diagnosis and checkpoint generation. This is not
the formal paper experiment and its outcomes are not automatically paper results.
The existing fork engine implements these six paths. Sixteen targeted tests
passed, followed by a 12-input native smoke run with five forks and 42 physical
primary executions. The smoke run passed the same prefix, recovery and cost
audit used for the full study, with no unrecovered failures. Its 8³ grid and
shortened boundaries are integration checks, excluded from performance claims.

The original suite can still be checked with `python -m experiments.paper_final.run_03_activation` (validation only). Its immutable six-path configurations and hashes remain historical protocol records. The current launcher targets the new three-path follow-up described above.

The completed run's logs, audits and source/config snapshots are in `advection_80_s1/logs/`, `audits/` and `provenance/`. Raw trajectories and checkpoint paths were retained so stage 02 can still locate its preselected model. Each RL branch saves checkpoints every 1000 stream inputs and at the end.

The requested single seed completed on September 16 at 04:15 New York time,
with all six 5000-input paths and five final checkpoints independently audited.
Stage 02 subsequently used replicate 1's `start_1000_final.npz`, selected before
seeing stage 03 outcomes, and completed all 336 remaining comparisons. Neither
stage 02 outcomes nor a favorable partial trajectory selected stage 03 seeds,
activation times or stopping points. Source snapshots and numerical residuals
are retained with the results. Stage 04 remains unlaunched.
