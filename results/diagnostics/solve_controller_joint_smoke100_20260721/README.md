# 60^3 Solve-Controller Screen

This directory contains the locked five-branch joint-online comparison.

- methods: bandit_fixed_w1.6, bandit_recursive_lstdq_lcb, bandit_recursive_lstdq_v2_lcb, bandit_structured_model_based, bandit_recalibrated_lsvi_lcb
- stream SHA-256: `a568345aac64d34bd45226df746b3f689de66341b259fd6bfa4d88064ed67be5`
- LSTDQ v2 beta: `4`
- recalibrated LSVI beta: `1`
- action grid: `1.00:0.05:3.00`
- recovery and timing: active bounded-recovery joint protocol

Run `OUTPUT_DIR=/new/path ./reproduce.sh` to reproduce without overwriting
this directory.  `config.json`, `stream_manifest.json`, trajectories, 1K
checkpoints, and final mutable states are written by the runner.
