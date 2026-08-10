# Paper experiment: scalar diffusion, physics-linear context, and solve RL

This directory is the complete result bundle for the 31 July 2026 paper run on
a strictly paired stream of 5,000 scalar anisotropic-diffusion problems at
`60^3`. It compares the native BoomerAMG default, LinUCB v5, the compact
physics-linear LinUCB context, and Recursive LSTDQ v3 solve control.

The headline result is strong: the aligned physics-linear setup + solve-RL
pipeline reduced mean end-to-end runtime from **374.909 ms to 222.395 ms** over
all 5,000 cases, a paired improvement of **40.68%** (95% bootstrap CI
**[39.83%, 41.50%]**). In the final 1,000 cases it reached **190.570 ms**, a
**49.46%** improvement (**[48.93%, 50.00%]**) over the native default.

## Experiment question

Does replacing the canonical PDE context with compact, physically meaningful
features help both online BoomerAMG setup selection and per-cycle relaxation
control?

The five paired branches were:

1. **Default**: default setup + default solve.
2. **v5 + default**: canonical-8D LinUCB v5 setup + default solve.
3. **Physics + default**: physics-linear-7D LinUCB setup + default solve.
4. **v5 + RL**: canonical-8D LinUCB v5 setup + Recursive LSTDQ v3 with the
   canonical solve context.
5. **Physics + RL**: physics-linear-7D LinUCB setup + Recursive LSTDQ v3 with
   the aligned physics-linear solve context.

The setup learner starts online learning at case 1. Both solve controllers are
deliberately disabled for cases 1--1,000, use the native default solve during
that prefix, and activate at case 1,001. Each therefore receives exactly 4,000
online controller updates. No solve controller learns from zero.

## Locked protocol

| Item | Value |
|---|---|
| PDE | `scalar_anisotropic_diffusion` |
| Grid | `60 x 60 x 60` |
| Diffusion range | `c_i in [1, 1000]` |
| Advection | `(0, 0, 0)` |
| Stream | 5,000 shared, deterministically shuffled cases |
| Stream SHA-256 | `07258e0e51e1b9ecc73cdac1344d25050496b6e150da5de1359f0234a2afc722` |
| Solve stopping rule | relative residual tolerance `1e-6` |
| Maximum AMG cycles | 50 |
| Setup space | recommended Tune-7 |
| Coarsen types | `[0, 2, 3, 6, 8, 10]` |
| Interpolation types | `[0, 2, 3, 6, 8, 17]` |
| Aggressive interpolation types | `[4]` |
| Setup candidates | deterministic structured-512 AOT schedule; at most 3 selections/case |
| Solve actions | `w = 1.00, 1.05, ..., 3.00` |
| Solve action basis | RBF centers `1.00, 1.25, ..., 3.00`; sigma `0.2` |
| LSTDQ | ridge `1`, beta `2`, lambda `0.8`, residual floor `1e-3 s` |
| Exploration | epsilon `0.30 -> 0.03` over 20,000 controller steps |
| Method execution | randomized within every paired case |
| Recovery | one measured default fallback; its cost is included in feedback and runtime |

All seeds and the exact high-level specification are stored in
[`experiment_config.json`](experiment_config.json). The exact deterministic
stream is recorded in [`stream_manifest.json`](stream_manifest.json).

### Contexts

The canonical v5 PDE context is

```text
[bias, c_x, c_y, c_z, c_mean, a_x, a_y, a_z]
```

The physics-linear context is

```text
[bias, diffusion_log_mean, diffusion_log_contrast_xy,
 diffusion_log_contrast_xyz, cell_Peclet_x, cell_Peclet_y, cell_Peclet_z]
```

For this diffusion-only experiment, the three advection or Peclet entries are
zero. The names remain 8D and 7D because those same interfaces also support the
diffusion-advection problem without changing learner code. Setup and solve use
the same per-instance PDE representation within each aligned branch. The
result metadata records solve state dimensions of 35 (canonical) and 34
(physics-linear), and joint state-action dimensions of 315 and 306.

## Results

All values below are per-case means in milliseconds. End-to-end time includes
setup, native solve, controller decision/update time, and setup-bandit overhead.
The confidence intervals use the paired case stream and are relative to
default setup + default solve.

### All 5,000 cases

| Method | Setup | Native solve | Controller | Bandit | End-to-end | Cycles | E2E improvement (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 259.857 | 115.052 | 0.000 | 0.000 | 374.909 | 12.58 | baseline |
| v5 + default | 96.667 | 170.108 | 0.000 | 7.946 | 274.721 | 19.92 | 26.72% [25.97, 27.46] |
| Physics + default | 98.167 | 169.873 | 0.000 | 7.664 | 275.704 | 19.93 | 26.46% [25.70, 27.16] |
| v5 + RL | 96.959 | 132.661 | 4.973 | 7.883 | 242.475 | 16.71 | 35.32% [34.47, 36.07] |
| **Physics + RL** | **98.553** | **112.170** | **3.991** | **7.680** | **222.395** | **14.05** | **40.68% [39.83, 41.50]** |

### Final 1,000 cases (cases 4,001--5,000)

| Method | Setup | Native solve | Controller | Bandit | End-to-end | Cycles | E2E improvement (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 260.527 | 116.517 | 0.000 | 0.000 | 377.044 | 12.58 | baseline |
| v5 + default | 90.266 | 162.083 | 0.000 | 7.969 | 260.318 | 18.97 | 30.96% [29.95, 31.82] |
| Physics + default | 91.518 | 160.355 | 0.000 | 7.669 | 259.542 | 18.88 | 31.16% [30.42, 31.88] |
| v5 + RL | 87.642 | 113.417 | 6.467 | 7.942 | 215.468 | 14.79 | 42.85% [42.19, 43.53] |
| **Physics + RL** | **90.329** | **87.583** | **5.038** | **7.620** | **190.570** | **11.53** | **49.46% [48.93, 50.00]** |

The final-1,000 physics + RL native solve is **24.83% faster** than default
(95% CI **[23.54%, 26.13%]**), even before counting its much cheaper setup.

## Interpretation

- **Setup learning supplies the first large gain.** Both learned-setup +
  default-solve branches are about 26.5% faster end-to-end over all 5K cases,
  despite taking more solve cycles than the native default, because setup falls
  from roughly 260 ms to 97--98 ms.
- **The two setup contexts alone are statistically tied over all 5K.** Physics
  + default differs from v5 + default by -0.36% end-to-end, with a paired 95%
  CI of [-1.12%, 0.36%]. Thus the large final result is not explained by a
  standalone setup-context advantage.
- **The aligned physics pipeline learns a better solve policy.** Physics + RL
  is 8.28% faster end-to-end than v5 + RL over all 5K (paired 95% CI
  [7.32%, 9.24%]) and 11.56% faster in the final 1K ([10.87%, 12.32%]). Its
  final-1K solve uses 11.53 cycles on average versus 14.79 for v5 + RL.
- **The improvement strengthens with online experience.** Physics + RL moves
  from 8.78% faster than default during the controller-disabled first 1K to
  49.46% faster in the final 1K. The first-1K controller runtime and decision
  counts are both exactly zero, confirming the activation boundary.
- **The comparison is reliable under the configured recovery protocol.** All
  five branches finish all 5,000 external cases with zero unrecovered failures.
  Learned branches invoke 109, 118, 143, and 121 measured default fallbacks,
  respectively; those costs are already included in every reported mean.

This is persuasive evidence for the paper, but it is still one persistent
online stream on one machine. The paired bootstrap intervals quantify
case-to-case uncertainty on this stream; they are not a substitute for an
independent multi-seed or multi-machine replication.

## Reproduce the run

First follow the repository [build requirements](../../../README.md). The
command below checks out the commit that added this frozen experiment bundle,
creates the declared environment, builds HYPRE, and starts a fresh run:

```bash
git checkout "$(git log --diff-filter=A -1 --format=%H -- \
  results/joint/paper_n60_scalar_diffusion_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6_20260731/README.md)"
conda env create -f environment.yml
conda activate rl
make -C hypre
PYTHON_BIN="$(command -v python)" \
  ./results/joint/paper_n60_scalar_diffusion_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6_20260731/reproduce.sh
```

`reproduce.sh` writes to a sibling directory ending in `_reproduction` and the
runner refuses to overwrite an existing non-empty result directory. Override
`OUTPUT_DIR`, `PYTHON_BIN`, or `REPO_ROOT` through environment variables when
needed.

The original run used Python 3.10.20, NumPy 2.2.6, SciPy 1.15.3, mpi4py 4.1.2,
and macOS 26.5 on an 8-core Apple M2 MacBook Air with 8 GB RAM. It ran from
12:04 to 14:07 EDT (about 2 h 3 min). Wall-clock values will naturally vary
with hardware, load, and thermal state; the stream hash and algorithmic
trajectory are the stronger reproducibility checks.

## Regenerate analysis and figures

Plotting is separate from numerical execution. After the reproduction finishes,
regenerate its figures with:

```bash
python experiments/joint/solve_control/generate_joint_experiment_plots.py \
  --result-dir results/joint/paper_n60_scalar_diffusion_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6_20260731_reproduction \
  --rolling-window 100
```

Important analysis outputs are:

- [`screen_report.md`](screen_report.md): readable all-5K and first/last-window
  statistics with paired intervals and controller diagnostics.
- [`summary_5000.csv`](summary_5000.csv): compact all-5K per-method summary.
- [`figures/last_1000_runtime_breakdown.png`](figures/last_1000_runtime_breakdown.png):
  primary paper-facing runtime decomposition.
- [`figures/learned_per_cycle_action_trajectories.png`](figures/learned_per_cycle_action_trajectories.png):
  learned relaxation-weight heatmaps.

The repository keeps only the frozen config, stream manifest, compact reports,
reproduction script, and two primary figures. A reproduced run writes the full
trajectories, candidate schedules, checkpoints, mutable final states, and
derived analysis locally; those large artifacts are intentionally ignored by
Git.
