# Script Map (AMG Setup Bandits)

## Core runner

The shared setup-only loop is
`utils.setup_amg.run_amg_setup_experiment(...)`. New entry points should reuse
it instead of copying the solve and logging loop.

## Main experiment entrypoints
- `learning_amg.py`
  - Tsallis-INF AMG setup experiment.
- `learning_amg_linucb.py`
  - LinUCB AMG setup experiment.
- `run_setup_bandit_comparison.py`
  - Current multi-method setup-only comparison.

## Compatibility wrapper
- `run_amg_linucb_T5000_default.py`
  - Historical entrypoint name.
  - Thin wrapper that delegates to `learning_amg_linucb.py` with defaults.

## Stream/timing utilities
- `generate_amg_instance_stream.py`
- `run_amg_policy_on_stream.py`
- `run_amg_stream_timing_suite.py`
- `plot_amg_stream_timing.py`

## Bench utilities
- `bench_linucb_overhead.py`
- `bench_wu_vs_time.py`

## Best-fixed evaluator
- `eval_amg_linucb_best_fixed.py`

Historical numbered scripts were moved to
`experiments/archive/setup_phase_legacy/`. Cross-phase `test_9.py` and
`test_10.py` were moved to `experiments/joint/legacy_setup_solve/`.
