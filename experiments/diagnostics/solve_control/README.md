# Disposable diagnostics

This directory contains development-only diagnostics, ablations, smoke tests,
and unit tests created while stabilizing the online solve controller. The
formal tuning and five-method 2K pipeline do not import this package, so the
entire directory can be removed without affecting the main algorithm.

Run diagnostics from the repository root; their path bootstrap loads the
production setup and solve modules:

```bash
python -m unittest discover -s experiments/diagnostics/solve_control -p 'test_*.py'
python experiments/diagnostics/solve_control/run_frozen_setup_td_lambda.py --help
```

The controlled shared-action screen compares fixed `w=1.6`, the retained
independent-head SARSA baseline, shared-action SARSA, bootstrap-LCB SARSA, and
stagewise LSVI-LCB on identical frozen `(instance, setup)` streams:

```bash
python experiments/diagnostics/solve_control/run_shared_action_rl_study.py \
  --output-dir results/online_td_lambda_v1/run_logs/diagnosis/shared_action_screen \
  --smoke

python experiments/diagnostics/solve_control/run_shared_action_rl_study.py \
  --output-dir results/online_td_lambda_v1/run_logs/diagnosis/shared_action_screen

python experiments/diagnostics/solve_control/plot_shared_action_rl_study.py \
  --result results/online_td_lambda_v1/run_logs/diagnosis/shared_action_screen/result.json
```

The full run uses 1000 training cases, three controller seeds, and three
independent 250-case frozen validation streams. Potential shaping is disabled
by construction so the screen isolates action sharing and value estimation.

Generated development runs live under
`results/online_td_lambda_v1/run_logs/diagnosis/`. Formal 2K outputs and the
shared 1000-instance warmup artifact remain outside that directory.
