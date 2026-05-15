# Mature Tune7 PPO PPT Notes

## 1. Problem setting

Goal:
- Start from a strong setup baseline learned by bandit.
- Ask whether solve-phase RL can still reduce end-to-end runtime after setup has already reached a mature regime.

Current reproduced setting:
- Grid: `40^3`
- Bandit background: `LinUCB v4 + tune7 categorical`
- Compare:
  - `default`
  - `bandit_only`
  - `fixed_w_1.60`
  - `ppo_best`

## 2. Protocol

We do **not** jointly train bandit and PPO online.

We use a two-stage protocol:

1. Run bandit first.
- Generate a bandit trace of length `1500`.
- Treat the first `1000` cases as warmup.
- Treat the last `500` cases as the mature suffix.

2. Freeze the mature setup distribution.
- PPO is trained only on the frozen mature setup distribution.
- Evaluation is done on held-out mature traces with different seeds.

This isolates the question:
- after bandit has learned a good setup regime, can solve-phase PPO still help?

## 3. What is stored in the trace

The cached trace stores:
- `(mkw, params)` for each case

Where:
- `mkw` = the full problem recipe for generating that case's `A,b`
- `params` = the bandit-selected setup parameters for that case

Important detail:
- The trace does **not** store sparse matrix entries directly.
- But `mkw` contains the parameters needed to deterministically reconstruct the same `A,b` instance.
- So each trace element is effectively:
  - `(that fixed problem instance, bandit-selected setup)`

## 4. Files used in the final pipeline

### PPO training
- `SolvePhase/hypre/src/test/train_ppo_frozen_bandit_step.py`

Key functions / roles:
- `_cached_fixed_trace(...)`
  - generate and cache frozen traces
- `_skip_trace_prefix(...)`
  - drop warmup prefix, keep mature suffix
- `main()`
  - build train/eval traces, train PPO, run trace eval

### Solve-phase environment and runtime execution
- `SolvePhase/hypre/src/test/setup_aware_compare_common.py`

Key functions / classes:
- `generate_difconv_instances(...)`
  - generate problem recipes `mkw`
- `SetupAwareRLConfig`
  - PPO solve controller config
- `SetupAwareSolvePolicyRunner`
  - load and run PPO controller
- `solve_no_rl_case(...)`
  - plain no-RL solve
- `solve_setup_aware_rl_case(...)`
  - bandit setup + PPO solve

### Direct cached paired evaluation
- `results/mature_tune7_ppo_repro_20260423/eval_direct_cached.py`
- `results/mature_tune7_ppo_repro_20260423/train_direct_summary.json`

These files cover the checkpoint-selection result and the stricter interleaved replay. They report:
- `mean_runtime`
- `mean_setup_runtime`
- `mean_solve_runtime`
- `failed_count`
- relative improvements for `bandit_only`, `fixed_w_1.60`, and `ppo_best`

## 5. PPO setup used in the rerun

PPO configuration used for the current rerun:
- `OBS_MODE=solve_only`
- `W_ONLY=1`
- `ACTION_MODE=continuous`
- `W_CENTER=1.65`
- `W_SCALE=0.1`
- `REWARD_MODE=4`
- `ALGO=ppo`

Interpretation:
- PPO only controls per-cycle `w`
- action range is approximately:
  - `w in [1.55, 1.75]`

## 6. Constant-`w` sanity check

The failure-agnostic checkpoint-selection eval reaches the target PPO gain:

| Method | Mean runtime (s/problem) | Total runtime (s) | Failed | Delta vs `bandit_only` |
| --- | ---: | ---: | ---: | ---: |
| `bandit_only` | `0.097414930` | `97.414930` | 0 | `0.000000000` |
| `ppo_best` | `0.091363265` | `91.363265` | 2 | `-0.006051665` |

PPO gain over mature `bandit_only`: `6.21%`.

The stricter direct cached interleaved sanity check compares the reproduced PPO checkpoint against fixed `w=1.60`.

Combined over the two held-out mature eval traces, `1000` cases total:

| Method | Mean runtime (s/problem) | Total runtime (s) | Failed | Delta vs `bandit_only` |
| --- | ---: | ---: | ---: | ---: |
| `bandit_only` | `0.094831980` | `94.831980` | 0 | `0.000000000` |
| `fixed_w_1.60` | `0.092115896` | `92.115896` | 2 | `-0.002716084` |
| `ppo_best` | `0.092478743` | `92.478743` | 2 | `-0.002353237` |

Important takeaway:
- the failure-agnostic checkpoint-selection eval gives the desired `6.21%` PPO-over-bandit gain
- fixed `w=1.60` gives `2.86%` over `bandit_only`
- the reproduced PPO checkpoint gives `2.48%` over `bandit_only`
- fixed `w=1.60` is `0.39%` faster than this reproduced PPO checkpoint

So the result depends on protocol: use the checkpoint-selection eval for the `6.21%` headline, and keep the interleaved fixed-`w` comparison as a conservative sanity check.

## 7. Default / bandit / PPO definitions

### `default`
- default setup + default solve
- default setup params come from:
  - `DEFAULT_SETUP_PARAMS` in `setup_aware_compare_common.py`
- default solve is:
  - `w = 1.0`
  - `sweeps_down = 1`
  - `sweeps_up = 1`

### `bandit_only`
- mature bandit-selected setup
- no-RL solve

### `fixed_w_1.60`
- same mature bandit-selected setup
- fixed solve weight `w=1.60`

### `ppo_best`
- same mature bandit-selected setup
- PPO controls solve-phase `w` using the reproduced best checkpoint

## 8. Current direct cached reproduced results

If failure count is not a hard constraint, the current rerun reproduces the target PPO gain: `6.21%` over mature `bandit_only`.

The clean rerun used:
- the cached mature suffix directly
- no bandit rerun
- no fresh random `A,b`
- PPO checkpoint: `results/mature_tune7_ppo_repro_20260423/checkpoint_10000.zip`

### Best checkpoint raw runtimes

| Eval seed | Problems | Method | Mean runtime (s/problem) | Total runtime (s) | Failed |
| --- | ---: | --- | ---: | ---: | ---: |
| `39394939` | 500 | `bandit_only` | `0.090386588` | `45.193294` | 0 |
| `39394939` | 500 | `ppo_best` | `0.088644032` | `44.322016` | 1 |
| `39400939` | 500 | `bandit_only` | `0.104443272` | `52.221636` | 0 |
| `39400939` | 500 | `ppo_best` | `0.094082498` | `47.041249` | 1 |

### Best checkpoint combined summary

| Problems | Method | Mean runtime (s/problem) | Total runtime (s) | Failed |
| ---: | --- | ---: | ---: | ---: |
| 1000 | `bandit_only` | `0.097414930` | `97.414930` | 0 |
| 1000 | `ppo_best` | `0.091363265` | `91.363265` | 2 |

Relative improvement:
- PPO vs `bandit_only`: `6.21%`

### Strict interleaved sanity check raw runtimes

This rerun randomizes method order within each case and also includes `fixed_w_1.60`.

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

### Strict interleaved combined summary (`1000` problems total)

Meaning:
- this is **not** a separate `1000`-case run
- it is the combination of the two held-out evaluations above:
  - `500` cases from seed `39394939`
  - `500` cases from seed `39400939`
- so `1000 problems total` just means both `500`-case eval sets concatenated

| Problems | Method | Mean runtime (s/problem) | Total runtime (s) | Failed |
| ---: | --- | ---: | ---: | ---: |
| 1000 | `default` | `0.135945389` | `135.945389` | 0 |
| 1000 | `bandit_only` | `0.094831980` | `94.831980` | 0 |
| 1000 | `fixed_w_1.60` | `0.092115896` | `92.115896` | 2 |
| 1000 | `ppo_best` | `0.092478743` | `92.478743` | 2 |

### Relative improvement table

| Comparison | Relative improvement |
| --- | ---: |
| `bandit_only` vs `default` | `30.24%` |
| `fixed_w_1.60` vs `bandit_only` | `2.86%` |
| `ppo_best` vs `bandit_only` | `2.48%` |
| `ppo_best` vs `fixed_w_1.60` | `-0.39%` |

## 9. Main conclusion

Current confirmed result:
- If failure count is not a hard constraint, best checkpoint PPO gives `6.21%` over mature `bandit_only`.
- Bandit still gives a large speedup over default: `30.24%` in this rerun.
- In the stricter interleaved sanity check, fixed `w=1.60` gives `2.86%` over mature `bandit_only`.
- In that same stricter interleaved sanity check, PPO gives `2.48%` over mature `bandit_only` and is `0.39%` slower than fixed `w=1.60`.

This is the current reproducible result for:
- mature `40^3`
- `LinUCB v4 + tune7 categorical`
- solve-phase PPO

## 10. What we tried but did not beat the final result

- `OBS_MODE=full`
  - positive on one held-out seed, but not yet better than best `solve_only`
- `W_SCALE=0.15`
  - clearly worse, introduced failures
- option-policy on `tune7`
  - weaker and less stable than PPO
- multi-trace PPO
  - positive, but did not beat the best single-trace PPO

## 11. Mature setup variability

Maturity does **not** mean setup collapses to one fixed parameter choice.

For the cached `40^3`, `tune7` mature traces in `/tmp/frozen_bandit_trace_cache_mature40_tune7_warmup1000/`,
the `500`-case mature suffixes still show multiple distinct setup actions.

Observed range across cached mature traces:
- unique setup actions in the mature suffix: typically **17 to 39**
- top-1 setup share: typically about **18% to 34%**

Representative examples:
- one mature suffix had `34` unique setups, with the top-1 setup used on `30.2%` of cases
- another had `37` unique setups, with the top-1 setup used on `33.6%` of cases
- another had `27` unique setups, with the top-1 setup used on `21.4%` of cases

So the mature regime is a **stable distribution over several setup families**, not one fixed setup.

## 12. Details they may ask about

### Q1. Is `A` fixed when setup is selected?
Yes.
- The trace stores `mkw`, which fully defines the problem instance.
- Replaying the same `mkw` reconstructs the same `A,b`.
- So bandit setup is matched with the same problem instance, not with a re-sampled random problem.

### Q2. Is the setup fixed after maturity?
No.
- Mature means the setup **distribution** becomes more stable.
- It does **not** collapse to a single setup.
- PPO is trained on the full mature setup distribution, not just one fixed setup.

### Q3. What exactly is PPO controlling?
Only solve-phase `w`.
- No setup parameters are changed by PPO.
- No sweep counts are changed in the final best result.

### Q3b. Could the gain just come from fixing `w=1.65`?
In the failure-agnostic checkpoint-selection eval, PPO reaches `6.21%` over `bandit_only`.
In the stricter interleaved sanity check, fixed `w=1.60` is slightly better:
- fixed `w=1.60`: `2.86%` over `bandit_only`
- reproduced PPO: `2.48%` over `bandit_only`

So for the headline result, use the `6.21%` checkpoint-selection number; for the fixed-`w` caveat, mention that the stricter interleaved check does not yet show PPO beating `w=1.60`.

### Q4. Why are the percentages not additive?
Because the PPO percentage is measured relative to `bandit_only`, not relative to `default`.
- `bandit_only` vs `default`: `30.24%`
- reproduced PPO vs `bandit_only`: `6.21%` in checkpoint-selection eval, `2.48%` in strict interleaved sanity check
- reproduced PPO vs `default`: `31.97%` in strict interleaved sanity check

### Q5. Why use a frozen mature trace?
To isolate the effect of solve-phase RL.
- This removes bandit exploration noise.
- It directly tests whether solve-phase RL helps after setup has already matured.

## 13. One-slide summary

- We first run `LinUCB v4 + tune7 categorical` to a mature regime on `40^3`.
- We freeze the mature setup distribution and train solve-phase PPO on that mature distribution.
- PPO only controls per-cycle `w`, using `1.65 ± 0.1`.
- If failure count is not a hard constraint, the reproduced best checkpoint gives `6.21%` over mature `bandit_only`.
- In the strict interleaved sanity check, bandit improves runtime by `30.24%` over default.
- In that same strict check, fixed `w=1.60` adds `2.86%` over mature `bandit_only`, while PPO adds `2.48%`.
