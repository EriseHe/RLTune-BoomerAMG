# 60^3 Solve-Controller Screen

This directory contains the locked five-branch joint-online comparison.

- methods: bandit_fixed_w1.6, bandit_recursive_lstdq_lcb, bandit_recursive_lstdq_v2_lcb, bandit_structured_model_based, bandit_recalibrated_lsvi_lcb
- stream SHA-256: `156e6fdbbed6733d98e2c5f7e550e5d230c45217d0833ac64567563b5434459b`
- LSTDQ v2 beta: `4`
- recalibrated LSVI beta: `1`
- action grid: `1.00:0.05:3.00`
- recovery and timing: active bounded-recovery joint protocol

Run `OUTPUT_DIR=/new/path ./reproduce.sh` to reproduce without overwriting
this directory.  `config.json`, `stream_manifest.json`, trajectories, 1K
checkpoints, and final mutable states are written by the runner.

## Final result

Recursive LSTDQ v2-LCB is the screening winner at **290.001 ms/case** end to
end, **3.39% faster** than Online LinUCB + fixed `w=1.6` (paired-bootstrap 95%
CI **[2.42%, 4.28%]**). See [`FINAL_REPORT.md`](FINAL_REPORT.md) for the audited
interpretation and [`screen_report.md`](screen_report.md) for all locked windows.
