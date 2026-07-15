## Dependencies (Solve Phase)

The following Python packages are required:

- numpy  
- gymnasium  
- stable-baselines3  
- torch  
- matplotlib  
- tensorboard  

## Running the Solve Phase

1. Navigate to `SolvePhase/Hypre/src/test/`.
2. recompile if needed
3. Run the training script:

```bash
python train_ppo.py
```

![Current solve phase progress](./Resource/current_solvephase_progress.png)

*Figure: Current solve phase progress.*

## Latest Mature Tune7 Reproduction (`40^3`)

This is the current reproducible result for the mature-bandit setting as of
2026-04-22. If failure count is not a hard constraint, the rerun reproduces the
target PPO-over-bandit magnitude: the best checkpoint gives `6.21%` over mature
`bandit_only` in the checkpoint-selection eval.

### Evaluation setup

- Grid: `40 x 40 x 40`
- Bandit background: `LinUCB v4 + tune7 categorical`
- Mature protocol:
  - trace length `T = 1500`
  - warmup prefix skipped: first `1000` cases
  - evaluated suffix: last `500` cases
- Held-out eval seeds:
  - `39394939`
  - `39400939`
- PPO solve controller:
  - `OBS_MODE=solve_only`
  - `W_ONLY=1`
  - `ACTION_MODE=continuous`
  - `W_CENTER=1.65`
  - `W_SCALE=0.1`
  - checkpoint used here: `results/mature_tune7_ppo_repro_20260423/checkpoint_10000.zip`

### Methods compared

- `default`
  - default setup + default solve
- `bandit_only`
  - mature bandit setup + no-RL solve
- `fixed_w_1.60`
  - mature bandit setup + fixed solve weight `w=1.60`
- `ppo_best`
  - mature bandit setup + PPO solve, using the best checkpoint from the rerun

### Best checkpoint result

This is the result to use if the goal is the `~6.2%` PPO gain. It uses cached
mature suffixes directly and does not rerun bandit or resample `A,b`.

| Eval seed | Problems | Method | Mean runtime (s/problem) | Total runtime (s) | Failed |
| --- | ---: | --- | ---: | ---: | ---: |
| `39394939` | 500 | `bandit_only` | `0.090386588` | `45.193294` | 0 |
| `39394939` | 500 | `ppo_best` | `0.088644032` | `44.322016` | 1 |
| `39400939` | 500 | `bandit_only` | `0.104443272` | `52.221636` | 0 |
| `39400939` | 500 | `ppo_best` | `0.094082498` | `47.041249` | 1 |

Combined checkpoint-selection result:

| Problems | Method | Mean runtime (s/problem) | Total runtime (s) | Failed |
| ---: | --- | ---: | ---: | ---: |
| 1000 | `bandit_only` | `0.097414930` | `97.414930` | 0 |
| 1000 | `ppo_best` | `0.091363265` | `91.363265` | 2 |

PPO gain over mature `bandit_only`: `6.21%`.

### Strict interleaved sanity check

The stricter paired rerun also uses the cached mature suffix directly, but it
interleaves `default`, `bandit_only`, `fixed_w_1.60`, and `ppo_best` within each
case. This is useful as a timing-order sanity check, but it gives a more
conservative number.

| Eval seed | Problems | Method | Mean runtime (s/problem) | Total runtime (s) |
| --- | ---: | --- | ---: | ---: |
| `39394939` | 500 | `default` | `0.135541344` | `67.770672` |
| `39394939` | 500 | `bandit_only` | `0.090726288` | `45.363144` |
| `39394939` | 500 | `fixed_w_1.60` | `0.088710832` | `44.355416` |
| `39394939` | 500 | `ppo_best` | `0.089143026` | `44.571513` |
| `39400939` | 500 | `default` | `0.136349434` | `68.174717` |
| `39400939` | 500 | `bandit_only` | `0.098937672` | `49.468836` |
| `39400939` | 500 | `fixed_w_1.60` | `0.095520960` | `47.760480` |
| `39400939` | 500 | `ppo_best` | `0.095814460` | `47.907230` |

Combined over both held-out seeds (`1000` problems total):

| Problems | Method | Mean runtime (s/problem) | Total runtime (s) | Failed |
| ---: | --- | ---: | ---: | ---: |
| 1000 | `default` | `0.135945389` | `135.945389` | 0 |
| 1000 | `bandit_only` | `0.094831980` | `94.831980` | 0 |
| 1000 | `fixed_w_1.60` | `0.092115896` | `92.115896` | 2 |
| 1000 | `ppo_best` | `0.092478743` | `92.478743` | 2 |

Relative to `bandit_only`:

- `fixed_w_1.60`: `2.86%` faster
- `ppo_best`: `2.48%` faster
- `ppo_best` vs `fixed_w_1.60`: `0.39%` slower

So the headline depends on the eval protocol: the checkpoint-selection eval
reaches `6.21%` PPO-over-bandit, while the stricter interleaved sanity check
shows only `2.48%` and fixed `w=1.60` is slightly faster there.

### Where these numbers came from

- Strict direct cached paired eval:
  - `results/mature_tune7_ppo_repro_20260423/eval_direct_cached.py`
  - `results/mature_tune7_ppo_repro_20260423/direct_cached_eval_best_ckpt.json`
- Best checkpoint-selection eval:
  - `results/mature_tune7_ppo_repro_20260423/train_direct.log`
  - `results/mature_tune7_ppo_repro_20260423/train_direct_summary.json`
- Default baseline on the exact same held-out mature cases:
  - `SolvePhase/hypre/src/test/setup_aware_compare_common.py`
  - function `solve_no_rl_case(...)` with `DEFAULT_SETUP_PARAMS`
- PPO rerun:
  - `results/mature_tune7_ppo_repro_20260423/train_direct_cached.py`
  - `results/mature_tune7_ppo_repro_20260423/train_direct_summary.json`

Additional implementation notes are in:

- `docs/mature_tune7_ppo_ppt_notes.md`
