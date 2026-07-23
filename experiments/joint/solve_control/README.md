# Active Joint Setup/Solve Code

This directory was reduced to the modules that still participate in the
retained Exp44 workflow.

## Active modules

- `run_mature_bandit_rl_pipeline.py`
  - top-level training pipeline
- `train_ppo_frozen_bandit_step.py`
  - PPO/LSTM training loop over frozen mature-bandit traces
- `setup_aware_compare_common.py`
  - runner, setup-aware solve logic, trace builders, and shared evaluation helpers
- `frozen_bandit_step_env.py`
  - Gymnasium adapter used to train PPO over frozen traces
- `solve/core/amg_gym_env.py`
  - action decoding shared by training and direct forward evaluation
- `setup_action_space.py`
  - setup parameter specifications and observation encoding; no native runtime
- `joint_online_common.py`
  - instance streams, reporting, and recovery invariants
- `joint_4k_runner.py`
  - canonical stream/protocol orchestrator; legacy study modes remain here
- `composable_joint_4k.py`
  - composable method validation, named setup-space/AOT branch assembly,
    typed solve-controller bundles, and frozen PPO resolution
- `joint_4k_execution.py`
  - mode-neutral online comparison, checkpoint, retry/fallback, and reporting
    loop
- `run_joint_online_sarsa_4k.py`
  - compatibility-only facade for the historical runner name
- `solve/controllers/ppo/`
  - PPO policy definitions

## Active absolute `w` control

The retained Exp44 PPO predicts an absolute physical relaxation weight:

```text
w_t = clip(1.5 + 0.5 * a_t, 1.0, 2.0)
```

The decoder still supports the older residual `+/-0.02` action mode for
archived experiments, but it is not the active Exp44 configuration.

Key locations for the active path:

- `solve/core/amg_gym_env.py`
  - `decode_policy_action(...)`
- `frozen_bandit_step_env.py`
  - `step(...)` default continuous-action branch
- `experiments/joint/exp44/exp44_common.sh`
  - `ACTION_MODE=continuous_absolute`

## Bandit + RL boundary

The mature bandit still chooses setup parameters first. RL only controls solve.

Key location:

- `setup_aware_compare_common.py`
  - `solve_setup_aware_rl_case(...)`
  - `augment_setup_params(...)`
  - `fixed_trace(...)`
  - `eval_runner(...)`

That function:

1. receives bandit-selected setup params
2. calls `prepare_rl(params=...)`
3. lets the RL policy control solve-phase `w`

## Runtime and failure recovery

All active setup, solve, and joint workflows call the same
`hypre.bindings` package and `hypre/interfaces/libamg_runtime` native library.
The numerical HYPRE fork under `hypre/source/` remains unmodified.

Each external instance permits at most four native attempts:

1. up to three learned setup attempts in the same context, including the first;
2. learned reselection occurs only after setup construction failure;
3. after a solve failure, or after all three setup attempts fail, one default
   setup + default solve fallback.

There are no unbounded retry loops or artificial failure penalties. Failed
setup arms are excluded only within the current instance. LinUCB receives the
measured suffix cost for every learned attempt and a context-conditioned
failure label. If the default fallback also fails, the LinUCB transaction and
solve-controller learning state are rolled back while consumed RNG state is
retained.

Solve-controller cost is raw measured time. Residual potential shaping is a
retired diagnostic and is not part of the active protocol.

Recursive LSTDQ-LCB keeps the full measured recovery cost in its LSTD target.
Its confidence estimate uses the unclipped post-fit Bellman residual in the
sandwich covariance. Because remaining runtime is non-negative, policy LCBs are
bounded below by zero. If that support bound ties multiple actions, the lower
predicted mean breaks the tie before the anchor action is considered.

Recursive MC-LCB uses episode-level exponentially weighted RLS. Its sandwich
covariance clusters all returns from one solve before updating, so correlated
cycle returns are not treated as independent observations.

## 60^3 fresh 4K reference run

The post-fit covariance and bounded recovery protocol were exercised in the
following joint-from-scratch reference run:

- matrix grid: `60^3`
- setup discretization: `20`
- stream partition: `0` warmup + `4000` joint-online instances
- solve action grid: `1.00:0.05:3.00`
- maximum solve cycles: `50`
- Recursive MC-LCB episode half-life: `500`
- stream seeds: `40800039,40806039,...,40842039`
- shuffle seed: `40848039`
- stream SHA-256: `156e6fdbbed6733d98e2c5f7e550e5d230c45217d0833ac64567563b5434459b`
- PPO checkpoint:
  `results/joint/exp44/run_logs/exp44_absolute_lstm_train2000_instances_seed39396939_20260719/training/model_best.zip`
- result directory:
  `results/joint/online_linear_lcb_v2/run_logs/joint_online_recursive_lcb_ppo_joint4k_n60_w1to3_step005_recovery_v2_20260720/`

The six branches are default setup + solve, online LinUCB + default solve,
online LinUCB + fixed `w=1.6`, online LinUCB + absolute PPO, online LinUCB +
Recursive MC-LCB, and online LinUCB + Recursive LSTDQ-LCB. Every LinUCB branch
starts from the same empty mutable state and then updates independently on the
same external instance stream.

```bash
/opt/anaconda3/envs/rl/bin/python -u \
  experiments/joint/solve_control/joint_4k_runner.py \
  --output-dir results/joint/online_linear_lcb_v2/run_logs/joint_online_recursive_lcb_ppo_joint4k_n60_w1to3_step005_recovery_v2_20260720 \
  --study-mode recursive_lcb_ppo \
  --ppo-model results/joint/exp44/run_logs/exp44_absolute_lstm_train2000_instances_seed39396939_20260719/training/model_best.zip \
  --seed 40800039 --bandit-seed 40860039 \
  --controller-seed 40866039 --method-order-seed 40872039 \
  --train-cases 4000 --warmup-cases 0 --online-cases 4000 \
  --train-seed-groups 40800039,40806039,40812039,40818039,40824039,40830039,40836039,40842039 \
  --train-shuffle-seeds 40848039 --train-cases-per-seed 500 \
  --train-group-take 4000 --matrix-grid-n 60 \
  --setup-param-resolution 20 --max-cycles 50 \
  --shared-action-profile 1to3_step0p05 \
  --recursive-mc-episode-half-life 500 \
  --include-default-setup-baseline --progress-every 100
```

## Five-controller 60^3 screen

`--study-mode solve_controller_screen` is the locked successor screening mode.
It contains exactly five independent joint-online branches and excludes the
already-screened default, PPO, and Recursive MC controllers:

1. Online LinUCB + fixed `w=1.6`;
2. Online LinUCB + current post-fit-sandwich Recursive LSTDQ (v1);
3. Online LinUCB + rolling-MAD coverage-scaled Recursive LSTDQ (v2);
4. Online LinUCB + shared-RLS structured residual model-based control;
5. Online LinUCB + hierarchical, recalibrated LSVI-LCB.

Every branch starts from an empty, independently mutable LinUCB state. The
runner enforces the canonical 4K stream hash, `60^3`, setup resolution `20`,
the `1.00:0.05:3.00` solve-action grid, and the exact method roster. It writes
the complete trajectories, per-1K checkpoints, final controller and LinUCB
states, configuration, stream manifest, a compact screening report, and a
reproduction script into a fresh output directory. LSTDQ v2 and LSVI beta must
be frozen from the disjoint calibration before this mode is used for the final
screen.

The completed canonical screen is stored at
`results/joint/online_linear_lcb_v3/run_logs/solve_controller_screen_joint4k_n60_20260721/`.
LSTDQ v2 achieved `290.001 ms/case` end to end versus `300.183 ms/case` for
fixed `w=1.6`, an improvement of `3.39%` with paired-bootstrap 95% interval
`[2.42%, 4.28%]`. Its MAD-scaled coverage quantity is an exploration scale,
not a calibrated confidence interval; see the run's `FINAL_REPORT.md` for the
component audit and limitations.

## Composable high-level runner

New joint experiments should use `run_joint_experiment.py`. It reads one JSON
configuration and delegates to the same active joint-online engine used by the
locked studies; it does not maintain a second solve/recovery loop. The config
selects the problem and grid, instance counts and seeds, setup discretization,
solve action/RBF discretization, controller parameters, and an explicit list of
independent setup x solve branches.

Supported setup choices are `default`, `linucb`, and `lints`. A named setup
space can select either `uniform512` or `structured512` candidate sampling.
The structured oracle keeps the same 512-arm scoring budget and combines 64
default/incumbent/elite anchors, 192 branch-balanced global arms, up to 192
cached coordinate neighbors, and 64 persistent Sobol numeric exploration
arms. Both modes use AOT arm IDs and the same factorized RAM feature cache;
the final 512-row feature block is reconstructed once per decision.

Supported solve choices include `default`, `fixed`, PPO, Recursive MC, LSTDQ
v1/v2, recursive BLSTDQ/RBLSPI, stagewise LSVI, structured model-based
control, and recalibrated LSVI. RBLSPI maintains fixed-size `A`, `C`, and `b`
statistics for the BLSTD posterior, samples one coherent Q-function per solve,
and stores no accumulated transition history. Each online learner is
instantiated independently even when two branches use the same algorithm.

Validate the resolved stream and method roster without launching HYPRE:

```bash
/opt/anaconda3/envs/rl/bin/python -u \
  experiments/joint/solve_control/run_joint_experiment.py \
  --config experiments/joint/solve_control/configs/n40_lstdq_v2_factorial_joint4k.json \
  --validate-only
```

Remove `--validate-only` to run and automatically generate the standard plots.
Every result directory receives the resolved high-level config, low-level
protocol, stream manifest, trajectories, checkpoints, reports, and a
`reproduce.sh` script. `--output-dir` safely overrides the configured output
path without editing the JSON.

## Numbered workflow

### Training-side workflow

1. Read the mature bandit branch from the saved pickle, or rebuild it.
2. Regenerate diffusion-convection instances from the training seeds.
3. For each training case, call `branch.policy.select(...)` to choose setup
   params.
4. Store the resulting `(mkw, params)` pairs as the frozen training trace.
5. Repeat the same process for internal validation seeds.
6. Train PPO/LSTM on the frozen training trace.
7. Select the checkpoint with the configured internal objective.

### Held-out evaluation workflow

1. Regenerate diffusion-convection instances from each held-out seed.
2. Build the same-trace continuation used by the archived model protocol.
3. For each case, run the selected method on top of the mature setup:
   - default
   - bandit
   - fixed `w=1.6`
   - PPO absolute control
4. Save one JSON per seed.
5. Aggregate all seeds into one summary JSON.

## Legacy material

Older experiment drivers, sweeps, teacher/BC code, and obsolete docs were moved
to:

- `experiments/archive/legacy_exp44_pre_cleanup/`
- `experiments/archive/legacy_online_gym/`
- `experiments/archive/legacy_failure_protocol/`

They are not part of the retained Exp44 workflow.
