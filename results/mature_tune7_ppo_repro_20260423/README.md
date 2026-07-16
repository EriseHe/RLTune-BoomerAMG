# Exp44 Solve-Phase RL Workspace

This directory now keeps only the current full-range RL path and the minimum
scripts needed to retrain and evaluate it.

## High-level idea

Exp44 is the current clean formulation:

- setup is chosen by a **mature saved bandit**
- RL only controls the **solve phase**
- the allowed relax-weight family is still the full range `w in [1, 2]`
- the policy is **not** trained with teacher labels, sweeps, or rule warmstarts

The key design choice is a **residual action** instead of an absolute action:

```text
w_{t+1} = clip(w_t + 0.02 * a_t, 1.0, 2.0)
```

where:

- the initial anchor is the midpoint `w_0 = 1.5`
- `a_t` is the policy output in `[-1, 1]`
- the policy is an LSTM and sees setup + cycle/action context

This keeps the method full-range at the family level, while making the control
problem much easier to optimize than "predict an absolute `w` from scratch".

## Python environment

The active scripts assume a Python environment that already has:

- `numpy`
- `scipy`
- `matplotlib`
- `torch`
- `gymnasium`
- `stable-baselines3`
- `sb3-contrib`
- `mpi4py`

A reproducible conda setup is kept at the repository root:

```bash
conda env create -f environment.yml
conda activate rl
```

To update an existing `rl` environment from the same file:

```bash
conda env update -n rl -f environment.yml --prune
```

The equivalent manual setup is:

```bash
conda create -n rl python=3.10 -y
conda activate rl
pip install numpy scipy matplotlib torch gymnasium stable-baselines3 sb3-contrib mpi4py
```

Notes:

- `mpi4py` is needed because the local HYPRE build is MPI-enabled.
- The native HYPRE/MPI libraries are **not** installed by the `pip` command
  above; they are expected to already exist under this repository's local HYPRE
  build tree:
  - `SolvePhase/hypre/src/hypre/include`
  - `SolvePhase/hypre/src/hypre/lib`
- Ensure `mpicc` and `mpicxx` are available in `PATH`. For a Homebrew MPI
  installation, discover its prefix instead of assuming a CPU-specific path:

```bash
export PATH="$(brew --prefix open-mpi)/bin:$PATH"
```

## Structure

### Scripts

- `exp44_common.sh`
  - shared defaults for train/eval/retrain workflows
- `train_exp44_midpoint_lstm.sh`
  - trains a new Exp44-style model
- `eval_exp44_model.sh`
  - evaluates one model on held-out forward continuation
- `run_exp44_retrain_and_eval.sh`
  - convenience wrapper: train new model, re-test old model, test new model

### Core evaluation helpers

- `evaluate_saved_model_live_forward.sh`
- `eval_forward_continuation_dual.py`
- `aggregate_forward_continuation_from_run.py`

## Native-library prerequisites

The retained Exp44 path needs two local shared libraries:

1. solve-phase library:
   - `SolvePhase/hypre/src/test/libamg_env.dylib`
2. setup-phase library:
   - `SetupPhase/learning-setup/solver/libamg_setup_solver.dylib`

If you start from a fresh checkout and these files do not exist, build them
before running training or evaluation.

### Build `libamg_env.dylib`

```bash
cd /path/to/RLTune-BoomerAMG/SolvePhase/hypre/src/test
make -B libamg_env.dylib
```

### Build `libamg_setup_solver.dylib`

```bash
cd /path/to/RLTune-BoomerAMG/SetupPhase/learning-setup/solver
bash build_libamg_setup_solver.sh
```

That wrapper uses repository-relative paths and builds the setup solver with
`@loader_path`-based HYPRE lookup, so the result does not depend on your local
absolute checkout path.

### Important retained artifacts under `run_logs/`

- `mature40_tune7_bandit_state_case2.pkl`
  - saved mature setup-bandit state
- `exp44_midpoint_lstm_seedmeanstd_vsbandit_20260512T1`
  - baseline retained Exp44 training run
- `exp44_midpoint_lstm_seedmeanstd_vsbandit_20260515T1`
  - later retrain run, if present
- `exp44_midpoint_lstm_eval1000_5seed_rerun_20260513T1`
  - canonical 5-seed rerun
- `exp44_midpoint_lstm_eval1000_10seed_full4_20260513T1`
  - canonical 10-seed full table (`default`, `bandit`, `fixed`, `ppo`)
- `exp44_midpoint_lstm_seedmeanstd_vsbandit_20260515T1_old_model_eval`
  - old-model comparison run, if present
- `experiment_search_20260510.md`
  - chronological experiment log

## Training workflow

```bash
conda activate rl
cd /path/to/RLTune-BoomerAMG

RUN_TAG=exp44_midpoint_lstm_seedmeanstd_vsbandit_20260515T1 \
bash results/mature_tune7_ppo_repro_20260423/train_exp44_midpoint_lstm.sh
```

The training script does two things:

1. If `run_logs/mature40_tune7_bandit_state_case2.pkl` exists, it reuses it.
2. Otherwise it rebuilds the mature bandit warmup state and saves it there.

### End-to-end training flow

1. Load `mature40_tune7_bandit_state_case2.pkl`, or warm up the setup bandit and
   save it if the file is missing.
2. Recover the mature setup-bandit branch from the pickle payload
   `{"branch": branch}`.
3. For each training seed, regenerate diffusion-convection cases with
   `generate_difconv_instances(...)`.
4. For each regenerated case, call `branch.policy.select(...)` to choose setup
   params.
5. Save `(mkw, params)` pairs into a frozen training trace.
6. Repeat the same trace-building process for the internal validation seeds.
7. Train PPO/LSTM on the frozen training trace.
8. Score the checkpoint on internal validation traces using
   `seed_mean_minus_std_vs_bandit`.
9. Save the retained checkpoint and its training summary under `run_logs/`.

Default Exp44 training config:

- train seeds: `39396939,39402939,39408939`
- eval seeds: `39414939,39420939,39426939`
- `ACTION_MODE=continuous_residual`
- `MODEL_TYPE=lstm`
- `OBS_MODE=cycle_action_setup`
- `W_CENTER=1.5`
- `W_SCALE=0.02`
- `W_GLOBAL_MIN=1.0`
- `W_GLOBAL_MAX=2.0`
- `TOTAL_TIMESTEPS=2500`
- `CHECKPOINT_OBJECTIVE=seed_mean_minus_std_vs_bandit`

## Evaluation workflow

Evaluate an existing model:

```bash
conda activate rl
cd /path/to/RLTune-BoomerAMG

MODEL_PATH=$PWD/results/mature_tune7_ppo_repro_20260423/run_logs/exp44_midpoint_lstm_seedmeanstd_vsbandit_20260512T1/model_ckpt_2500.zip \
RUN_ID=exp44_eval_example \
bash results/mature_tune7_ppo_repro_20260423/eval_exp44_model.sh
```

Default held-out evaluation:

- methods: `default,bandit,fixed,ppo`
- seeds: `39393939,39394939,39400939,39406939,39412939`
- window: `1500:2500` (`eval_1000`)
- no VecNormalize file is needed for the retained Exp44 path

### End-to-end held-out evaluation flow

1. Read the trained PPO checkpoint passed through `MODEL_PATH`.
2. For each held-out seed, regenerate diffusion-convection cases with the same
   archived protocol.
3. Rebuild one same-trace continuation with `T=2500`.
4. For each case in the evaluation window, reuse the mature setup-bandit logic
   to obtain the setup params for the `bandit`, `fixed`, and `ppo` comparisons.
5. Evaluate:
   - `default`: default setup + default solve
   - `bandit`: mature bandit setup + default solve
   - `fixed`: mature bandit setup + fixed `w=1.6`
   - `ppo`: mature bandit setup + learned RL solve control
6. Write one per-seed JSON file: `forward_dual_seed<seed>.json`.
7. Aggregate all per-seed JSON files into one
   `forward_continuation_summary.json`.

## Code path: high level to low level

### 1. Train/eval entrypoints

- `results/mature_tune7_ppo_repro_20260423/train_exp44_midpoint_lstm.sh`
- `results/mature_tune7_ppo_repro_20260423/eval_exp44_model.sh`
- `results/mature_tune7_ppo_repro_20260423/run_exp44_retrain_and_eval.sh`

### 2. Main training pipeline

- `SolvePhase/hypre/src/test/run_mature_bandit_rl_pipeline.py`
- `SolvePhase/hypre/src/test/README_ACTIVE.md`

Important responsibilities:

- load or build the mature bandit state
- build frozen traces from that mature setup policy
- train PPO/LSTM solve-phase policy
- evaluate checkpoints on internal validation traces

### 3. Where bandit state is loaded/saved

- `run_mature_bandit_rl_pipeline.py`
  - `_warmup_bandit()`

That function either:

- loads `{"branch": branch}` from a pickle, or
- warms up the setup bandit and saves the resulting branch

The practical meaning is:

1. load mature bandit branch from pickle
2. regenerate cases from seeds
3. run `branch.policy.select(...)` per case
4. record the resulting setup trace

### 4. Where bandit and RL are combined

- `SolvePhase/hypre/src/test/setup_aware_compare_common.py`
  - `solve_setup_aware_rl_case(...)`
  - `augment_setup_params(...)`
  - `fixed_trace(...)`
  - `eval_runner(...)`

This is the main boundary:

1. mature bandit has already chosen setup params
2. `prepare_rl(params=...)` builds that setup
3. RL controls only the solve phase on top of it

### 5. Where residual action is decoded

- `SolvePhase/hypre/src/test/amg_gym_env.py`
  - `decode_policy_action_residual(...)`

This implements:

```text
w_{t+1} = clip(w_t + w_scale * a_t, w_min, w_max)
```

### 6. Where the env applies residual control

- `SolvePhase/hypre/src/test/frozen_bandit_step_env.py`

Relevant pieces:

- reset path sets the initial `last_w`
- step path decodes residual action and updates `last_w`

## Interpretation of the current result

The most stable current claim is:

- PPO clearly improves over the mature bandit baseline
- PPO remains competitive with the strong fixed `w=1.6` baseline
- the method does this without teacher labels or fixed-1.6 initialization

That is the reason Exp44 is the only retained mainline method here.
