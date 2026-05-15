# Script Map (AMG Setup Bandits)

## Core runner
- `amg_setup_runner.py`
  - Shared core loop:
    `run_amg_setup_experiment(problem_set, parameter_space, bandit, loss, ...)`
  - All new experiment scripts should call this.

## Main experiment entrypoints
- `learning_amg.py`
  - Tsallis-INF AMG setup experiment.
- `learning_amg_linucb.py`
  - LinUCB AMG setup experiment.

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
