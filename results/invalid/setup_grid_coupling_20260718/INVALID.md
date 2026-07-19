# Invalid setup-grid coupling results

These artifacts are retained only for audit and must not be used as Exp44
results, baselines, checkpoints, or warm-start states.

The affected runners incorrectly used the matrix grid size (`40`) as the
parameter-space resolution for each continuous Tune7 setup parameter. This
produced 23,040,001 setup arms instead of the canonical Exp44 configuration:

- matrix grid: `MATRIX_GRID_N=40`
- setup parameter resolution: `SETUP_PARAM_RESOLUTION=20`
- Tune7 categorical action count: `2,880,001`

The quarantined content includes the contaminated `online_td_lambda_v1`
experiment series and the July 16/18 Exp44 snapshots, PPO models, evaluations,
and derived diagnostics that used the invalid 23M-arm setup policy.

The artifacts were moved here on 2026-07-18 rather than deleted so the error
and its downstream effects remain auditable. They are intentionally outside
all active `run_logs` paths.
