# Active Exp44 Code Path

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
  - solve-phase Gym-style environment over frozen traces
- `amg_gym_env.py`
  - action decoding, including residual `w` updates
- `amg_env.py`
- `amg_setup_gym_env.py`
- `custom_policy.py`
- `util.py`

## Residual `w` control

The Exp44 policy does not predict an absolute `w`. It predicts a residual step:

```text
w_{t+1} = clip(w_t + w_scale * a_t, w_min, w_max)
```

Key locations:

- `amg_gym_env.py`
  - `decode_policy_action_residual(...)`
- `frozen_bandit_step_env.py`
  - `step(...)` branch for `action_mode == "continuous_residual"`

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
   - PPO residual control
4. Save one JSON per seed.
5. Aggregate all seeds into one summary JSON.

## Legacy material

Older experiment drivers, sweeps, teacher/BC code, and obsolete docs were moved
to:

- `legacy_exp44_pre_cleanup/`

They are not part of the retained Exp44 workflow.
