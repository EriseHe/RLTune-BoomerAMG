# AMG Setup Bandit Scripts

## Core runner

`run_setup_bandit_comparison.py` is the active setup-only experiment. It uses
`hypre.bindings`, allows up to three learned setup attempts after
setup-construction failures, and performs at most one default fallback.

## Active entrypoint

- `run_setup_bandit_comparison.py`: current LinUCB setup comparison.

## Bench utilities
- `bench_linucb_overhead.py`
- `bench_wu_vs_time.py`

Historical numbered scripts were moved to
`experiments/archive/setup_phase_legacy/`. Cross-phase `test_9.py` and
`test_10.py` were moved to `experiments/archive/legacy_setup_solve/`.
Scripts that depended on artificial failure penalties are under
`experiments/archive/legacy_failure_protocol/setup_phase/`.
