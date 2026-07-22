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

Diagnostics that depend on the former artificial failure-penalty protocol are
kept under `experiments/archive/legacy_failure_protocol/` and are not part of
the default discovery command.

Generated development runs live under
`results/diagnostics/`. Formal outputs remain outside that directory.

`diagnose_lstdq_recovery_cost.py` replays the locked 60-cubed stream with a
fixed setup sequence and one injected failed primary attempt. It isolates the
effect of measured fallback cost on Recursive LSTDQ-LCB without allowing a
controller decision to change later LinUCB setups. The corresponding formal
short joint run uses `--study-mode recursive_lstdq_lcb`.

## Solve-controller calibration

`run_solve_controller_calibration.py` performs the locked, disjoint 600-case
calibration for the five-controller screen. It first learns one Online LinUCB
setup trace with fixed `w=1.6`, then replays that exact `(instance, setup)`
trace for LSTDQ v2 betas `1,2,4`, recalibrated LSVI betas `1,2,4`, and the
parameter-free structured model-based controller. Candidate order is shuffled
per instance, while each candidate owns independent solve-controller state.

```bash
/opt/anaconda3/envs/rl/bin/python -u \
  experiments/diagnostics/solve_control/run_solve_controller_calibration.py \
  --output-dir results/diagnostics/solve_controller_calibration_dev600 \
  --progress-every 100
```

The runner locks the three stream seeds and all shuffle/bandit/controller/order
seeds in the experiment protocol. `selected_config.json` is the only parameter
handoff to the independent joint smoke and canonical 4K screen.
