# Legacy failure protocol diagnostics

These diagnostics reproduce experiments that used artificial failure penalties
or the pre-recovery controller semantics. They are retained as historical
evidence and are excluded from the active experiment and test paths.

This archive also contains the retired residual-potential shaping diagnostics
and native interface variants. They are not part of the default build and may
require the historical commit that originally produced their results.

`diagnostics/diagnose_lstdq_recovery_cost.py` retains the experimental fixed
`50 ms` covariance clip solely to reproduce the mechanism audit. Active LSTDQ
uses an unclipped post-fit Bellman residual.

The July 19 `60^3` result below used the retired retry-and-penalty protocol and
must not be combined with new recovery-protocol results:

```text
results/joint/online_linear_lcb_v1/run_logs/
joint_online_recursive_lcb_ppo_joint4k_n60_w1to3_step005_from_scratch_20260719/
```
