# Composable Joint Setup/Solve Experiments

`run_joint_experiment.py` is the current entry point for paired online
BoomerAMG experiments. It reads one JSON file, validates the problem stream and
method roster, and delegates every branch to the shared execution and recovery
engine.

Complete the environment and HYPRE build in the repository
[`README.md`](../../../README.md) before running an experiment.

The final paper study is documented in [`PAPER_FINAL.md`](PAPER_FINAL.md).
Its 18 frozen configurations cover Modules 1 and 2. The suite entry point
`run_paper_final.py` defaults to validation; native runs require `--run`.

## Run from a JSON configuration

Validate a configuration without launching HYPRE:

```bash
python -u experiments/joint/solve_control/run_joint_experiment.py \
  --config experiments/joint/solve_control/configs/n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6.json \
  --validate-only
```

Run the same configuration into a fresh directory:

```bash
python -u experiments/joint/solve_control/run_joint_experiment.py \
  --config experiments/joint/solve_control/configs/n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6.json \
  --output-dir results/joint/hybrid_reproduction
```

The runner refuses to overwrite a non-empty output directory.

## Paper test: context dimension and RL activation

[`paper_test_n60_context_activation.json`](configs/paper_test_n60_context_activation.json)
uses the encoder-corrected 60³ seed-C experiment as its reference. Run from the
repository root in the existing `rl` environment:

```bash
python -u experiments/joint/solve_control/run_joint_experiment.py \
  --config experiments/joint/solve_control/configs/paper_test_n60_context_activation.json
```

Append `--validate-only` to check the config and the deterministic stream
without running any PDE solves. Default output is
`results/joint/paper_test_n60_context_activation`; use `--output-dir` for a
fresh subsequent run.

| Branch | Shared setup/solve context | RL starts |
| --- | --- | --- |
| `default_setup_default_solve` | Reference solver | Never |
| `context8d_fixed` | Original 8-field encoding, unchanged | Problem 1001 |
| `context3d_fixed` | Three log diffusion coefficients + intercept | Problem 1001 |
| `context4d_fixed` | Three log coefficients, their mean + intercept | Problem 1001 |
| `context3d_dynamic` | Same as the 3D fixed-start branch | After the sequential gate crosses |
| `context4d_dynamic` | Same as the 4D fixed-start branch | After the sequential gate crosses |

There is no setup-only bandit branch. Here 3D/4D count descriptors **excluding**
the intercept, so their stored setup context sizes are 4/5. The historical 8D
name includes its intercept and three zero-advection slots. Both compact views
reject nonzero advection. The solve state retains all residual, cycle, previous
weight/time, and setup features; its state/joint feature sizes are 31/279 for
3D, 32/288 for 4D, and the original 35/315 for 8D.

The frozen reference is
`paper_final_n60_canonical8d_lstdq_v3_seed_stability_5k_encoderfix.json`, branch
`linucb_canonical8d_lstdq_v3_seed_c`. The new 8D branch changes only its ID.
Problem, stream, setup, solve and global seed config sections are copied
unchanged. All five learned branches use seed offset `2000029`, giving setup
seed `54760086` and controller seed `54768104` (including the existing v3
controller offset `2018`). Each branch owns its mutable learner state and
candidate cursor. The matrix/RHS stream SHA-256 must remain
`07258e0e51e1b9ecc73cdac1344d25050496b6e150da5de1359f0234a2afc722`.
All 5000 problems count, including each branch's initial default-solve prefix.
The original run/config/checkpoints are not overwritten. Identical seeds do
not guarantee identical wall-clock measurements or time-driven learning paths;
the six-branch execution order also differs from the original four-branch run.

The dynamic rule implements the horizon-free onset mixture in
[`dynamic_rl_start_and_theorem_framework.md`](../../../docs/theory/dynamic_rl_start_and_theorem_framework.md),
equations (1)--(4), with `p_bad=0.05`, `p_good=0.01`, `delta=0.05` and onset
weights `1/[t(t+1)]`. These are prospectively fixed design settings from the
note's illustration, not empirically calibrated thresholds. The error budget
is per branch, not a simultaneous 5% guarantee across the two dynamic branches.
Each branch monitors its own first-attempt failures, including construction
failures and failures later recovered by retry/fallback. It updates once after
each problem, in log space, and enables RL on the **next** problem after crossing.
There is no mandatory 1000-case prefix, forced deadline or post-activation
monitoring. If evidence never crosses, that branch remains setup-only throughout
the run. The certificate concerns persistent pre-activation unreliability under
the stated conditional-risk model, not guaranteed reliability of the RL policy.

Activation is announced in the terminal when it occurs. Per-problem evidence
and the controller-enabled flag are in each dynamic trajectory's
`rl_activation`; `rl_activation.json` tracks the latest gate states and
`result.json` includes final activation summaries. `activation_runtime` is
recorded separately and included in the existing controller-overhead component
and measured total, including during the default-solve prefix. Comparisons
against both Default and `context8d_fixed` are available in each result window.
The existing plotter uses distinct 3D/4D/start labels and the observed activation
boundaries. Its rolling display setting does not enter the activation rule.

## Configuration contract

The high-level JSON fixes:

- the PDE, grid, deterministic shared stream, and all seeds;
- the setup action space, candidate schedule, and setup context;
- the solve tolerance, cycle limit, actions, controller, and solve context;
- the independent setup-plus-solve branches and solve activation case;
- reporting and plot-generation settings.

Each method owns independent mutable learner state. Methods see the same PDE
instances, and their execution order is randomized within each paired case.
Setup and solve contexts are configured separately, so aligned and hybrid
context experiments use the same runner.

## Reproduce a frozen paper run

Every completed run receives an `experiment_config.json` and `reproduce.sh`.
For example:

```bash
OUTPUT_DIR=results/joint/hybrid_reproduction \
  ./results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/reproduce.sh
```

Reference bundles:

- [primary 60^3 scalar-diffusion comparison](../../../results/joint/paper_n60_scalar_diffusion_v5_vs_physics_linear_default_vs_lstdq_v3_staged1000_5k_tol1e6_20260731/README.md)
- [latest 60^3 diffusion-advection hybrid comparison](../../../results/joint/paper_n60_diffusion_advection_v5_setup_physics_solve_hybrid_staged1000_5k_tol1e6_20260801/README.md)

## Outputs and plotting

Full local runs write trajectories, checkpoints, mutable final states, resolved
protocol metadata, compact summaries, and a reproduction script. Raw generated
artifacts are ignored by Git; curated paper bundles retain only the frozen
config, stream manifest, compact reports, and primary figures.

Numerical execution and plotting are separate. Regenerate plots after a run:

```bash
python -u experiments/joint/solve_control/generate_joint_experiment_plots.py \
  --result-dir results/joint/hybrid_reproduction \
  --rolling-window 100
```

## Recovery protocol

A learned branch may make up to three measured setup attempts. A setup failure,
solve failure, non-finite result, or max-cycle nonconvergence then triggers one
measured default setup/default solve fallback. All consumed work is included in
runtime and learner feedback; pending updates are rolled back only if fallback
also fails.

## Main modules

- `run_joint_experiment.py`: config parsing, validation, and top-level run.
- `composable_joint_4k.py`: method construction and setup/solve composition.
- `joint_4k_execution.py`: paired execution, updates, recovery, and artifacts.
- `generate_joint_experiment_plots.py`: plot-only entry point.
