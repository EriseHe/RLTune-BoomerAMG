# Exp44 Solve-Phase RL Workspace

This directory now keeps only the current full-range RL path and the minimum
scripts needed to retrain and evaluate it.

## High-level idea

Exp44 is the current clean formulation:

- setup is chosen by a **mature saved bandit**
- RL only controls the **solve phase**
- the allowed relax-weight family is still the full range `w in [1, 2]`
- the policy is **not** trained with teacher labels, sweeps, or rule warmstarts

The active PPO now predicts an **absolute physical relaxation weight**:

```text
w_t = clip(1.5 + 0.5 * a_t, 1.0, 2.0)
```

where:

- `a_t` is the policy output in `[-1, 1]`
- the policy is an LSTM and sees setup + cycle/action context
- the initial observation uses the prepared solver default `w=1.0`
- the actor head is initialized to output that default
- the first comparison action is forced to the prepared default once globally;
  every later cycle and instance is policy controlled

This removes the old per-instance midpoint reset and the residual `+/-0.02`
restriction.

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
  out-of-source installation:
  - `hypre/install/include`
  - `hypre/install/lib`
- Ensure `mpicc` and `mpicxx` are available in `PATH`. For a Homebrew MPI
  installation, discover its prefix instead of assuming a CPU-specific path:

```bash
export PATH="$(brew --prefix open-mpi)/bin:$PATH"
```

## Structure

### Scripts

- `exp44_common.sh`
  - shared defaults for train/eval/retrain workflows
- `train_exp44_absolute_lstm.sh`
  - trains the active absolute-action Exp44 model
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

1. solve-phase library: `hypre/interfaces/libamg_env.dylib`
2. setup-phase library: `hypre/interfaces/libamg_setup_solver.dylib`

If you start from a fresh checkout and these files do not exist, build them
before running training or evaluation.

### Build HYPRE and both interfaces

```bash
cd /path/to/RLTune-BoomerAMG
make -C hypre
```

That wrapper uses repository-relative paths and builds the setup solver with
`@loader_path`-based HYPRE lookup, so the result does not depend on your local
absolute checkout path.

## Output layout

The active workflow writes all Exp44 artifacts under one run root:

```text
results/joint/exp44/
├── shared/
│   └── mature40_tune7_bandit_state_case2.npz
└── run_logs/<run-tag>/
    ├── training/
    │   ├── result.json
    │   ├── model.zip
    │   ├── model_best.zip
    │   ├── validation_table.csv
    │   ├── run.log
    │   └── figures/
    └── evaluation/
        ├── forward_continuation_summary.json
        ├── main_table.csv
        ├── per_seed/
        ├── model_snapshot/
        ├── run.log
        └── figures/
```

The shared mature bandit is not duplicated for every PPO run. Training and
held-out evaluation results are kept together under the same run tag.

## Training workflow

```bash
conda activate rl
cd /path/to/RLTune-BoomerAMG

RUN_TAG=exp44_absolute_default_lstm_canonical_20260718 \
bash experiments/joint/exp44/train_exp44_absolute_lstm.sh
```

The training script does two things:

1. If `results/joint/exp44/shared/mature40_tune7_bandit_state_case2.npz`
   exists, it reuses it.
2. Otherwise it rebuilds the mature bandit warmup state and saves it there.

### End-to-end training flow

1. Load `mature40_tune7_bandit_state_case2.npz`, or warm up the setup bandit and
   save it if the file is missing.
2. Rebuild the canonical immutable action catalog and restore the mature
   LinUCB matrices, candidate statistics, and RNG state from the lightweight
   checkpoint.
3. For each training seed, regenerate diffusion-convection cases with
   `generate_difconv_instances(...)`.
4. For each regenerated case, call `branch.policy.select(...)` to choose setup
   params.
5. Save `(mkw, params)` pairs into a frozen training trace.
6. Repeat the same trace-building process for the internal validation seeds.
7. Train PPO/LSTM on the frozen training trace.
8. Score the checkpoint on internal validation traces using
   `seed_mean_minus_std_vs_bandit`.
9. Save the retained checkpoint, validation table, figures, and training log
   under `<run-tag>/training/`.

Default Exp44 training config:

- `MATRIX_GRID_N=40`
- `SETUP_PARAM_RESOLUTION=20`
- Tune7 categorical setup actions: `2,880,000`
- train seeds: `39396939,39402939,39408939`
- eval seeds: `39414939,39420939,39426939`
- `ACTION_MODE=continuous_absolute`
- `MODEL_TYPE=lstm`
- `OBS_MODE=cycle_action_setup`
- `W_CENTER=1.5`
- `W_SCALE=0.5`
- `W_GLOBAL_MIN=1.0`
- `W_GLOBAL_MAX=2.0`
- `INITIAL_OBSERVATION_WEIGHT=1.0`
- `INITIAL_POLICY_WEIGHT=1.0`
- `TOTAL_TIMESTEPS=2500`
- `CHECKPOINT_OBJECTIVE=seed_mean_minus_std_vs_bandit`

To train on one complete pass over a fixed number of unique problem
instances, set `TRAIN_EPISODES` equal to the generated training-trace size.
For example, `RL_TRAIN_SEEDS=39396939`, `RL_TRAIN_CASES=2000`, and
`TRAIN_EPISODES=2000` stop PPO after exactly 2000 completed solves; the result
also records the resulting number of cycle transitions. The default
`TRAIN_EPISODES=0` retains transition-budget training.

## Evaluation workflow

Evaluate an existing model:

```bash
conda activate rl
cd /path/to/RLTune-BoomerAMG

MODEL_PATH=$PWD/results/joint/exp44/run_logs/exp44_absolute_default_lstm_canonical_20260718/training/model_best.zip \
RUN_ID=exp44_eval_example \
bash experiments/joint/exp44/eval_exp44_model.sh
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
6. Write one per-seed JSON file under `evaluation/per_seed/`.
7. Aggregate all per-seed JSON files into one
   `forward_continuation_summary.json`, then generate `main_table.csv` and the
   runtime/action figures.

The formal evaluator times LinUCB selection, feedback-loss evaluation, and
update separately while constructing the online setup trace. The table reports
their sum as `bandit_overhead_ms_per_instance`; the end-to-end figure stacks
native runtime, PPO controller overhead, and LinUCB overhead.

## Code path: high level to low level

### 1. Train/eval entrypoints

- `experiments/joint/exp44/train_exp44_absolute_lstm.sh`
- `experiments/joint/exp44/eval_exp44_model.sh`
- `experiments/joint/exp44/run_exp44_retrain_and_eval.sh`

### 2. Main training pipeline

- `experiments/joint/solve_control/run_mature_bandit_rl_pipeline.py`
- `experiments/joint/solve_control/README.md`

Important responsibilities:

- load or build the mature bandit state
- build frozen traces from that mature setup policy
- train PPO/LSTM solve-phase policy
- evaluate checkpoints on internal validation traces

### 3. Where bandit state is loaded/saved

- `run_mature_bandit_rl_pipeline.py`
  - `_warmup_bandit()`

That function either:

- rebuilds the canonical branch and restores its lightweight mutable state, or
- warms up the setup bandit and saves that mutable state

The practical meaning is:

1. rebuild the canonical bandit branch and load its mutable state
2. regenerate cases from seeds
3. run `branch.policy.select(...)` per case
4. record the resulting setup trace

### 4. Where bandit and RL are combined

- `experiments/joint/solve_control/setup_aware_compare_common.py`
  - `solve_setup_aware_rl_case(...)`
  - `augment_setup_params(...)`
  - `fixed_trace(...)`
  - `eval_runner(...)`

This is the main boundary:

1. mature bandit has already chosen setup params
2. `prepare_rl(params=...)` builds that setup
3. RL controls only the solve phase on top of it

### 5. Where the absolute action is decoded

- `experiments/joint/solve_control/setup_aware_compare_common.py`
  - `decode_policy_action(...)`

This implements:

```text
w_t = clip(w_center + w_scale * a_t, w_min, w_max)
```

### 6. Where the env applies absolute control

- `experiments/joint/solve_control/frozen_bandit_step_env.py`

Relevant pieces:

- reset path exposes the prepared solver default as the initial observation
- every step applies the newly decoded absolute physical weight

`FrozenBanditStepEnv` is a `gymnasium.Env` only because Stable-Baselines3
expects that API during PPO training. Formal held-out evaluation calls
`model.predict(...)` and the native HYPRE `step_rl(...)` interface directly;
the non-PPO solve controllers do not depend on Gymnasium.

## Interpretation of the current result

The July 16/18 local Exp44 retrains and online comparisons were invalidated by
an environment configuration bug that used the matrix grid size (`40`) as the
setup parameter resolution. Those runs used `23,040,001` setup actions rather
than the canonical `2,880,000` and have been moved to:

```text
results/invalid/setup_grid_coupling_20260718/
```

They are retained for audit only and must not be used for performance claims or
warm starts. The active workflow now keeps matrix size and setup parameter
resolution independent and rejects any Exp44 branch whose action count is not
exactly `2,880,000`.
