# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds. Improvement and
paired 95% intervals are relative to Online LinUCB + fixed `w=1.6`.

## all_100

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit_fixed_w1.6 | 444.263 | 392.643 | 836.906 | 0.000 | 8.739 | 845.644 | 47.69 | baseline | 92/92/0 | 1.000 |
| bandit_recursive_lstdq_lcb | 425.694 | 394.474 | 820.168 | 36.068 | 8.762 | 864.998 | 49.55 | -2.29% [-9.88, 4.76] | 97/97/0 | 0.300 |
| bandit_recursive_lstdq_v2_lcb | 416.040 | 381.057 | 797.097 | 44.685 | 8.653 | 850.435 | 47.06 | -0.57% [-9.22, 7.35] | 90/90/0 | 0.320 |
| bandit_structured_model_based | 385.937 | 367.916 | 753.854 | 37.768 | 9.190 | 800.812 | 45.46 | 5.30% [-2.58, 13.07] | 83/83/0 | 0.180 |
| bandit_recalibrated_lsvi_lcb | 431.492 | 396.904 | 828.396 | 64.178 | 8.257 | 900.831 | 48.76 | -6.53% [-15.57, 1.32] | 96/96/0 | 0.390 |

Controller diagnostics:

| method | decisions | mean w | explored | mean uncertainty (ms) | lower-bound saturation | calibration coverage 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| bandit_recursive_lstdq_lcb | 4955 | 1.817 | 0.272 | 71.953 | 0.031 | 0.983/0.993/0.999 |
| bandit_recursive_lstdq_v2_lcb | 4706 | 1.856 | 0.282 | 2.770 | 0.001 | 0.238/0.414/0.623 |
| bandit_structured_model_based | 4546 | 1.705 | 0.275 | n/a | n/a | n/a |
| bandit_recalibrated_lsvi_lcb | 4876 | 1.270 | 0.268 | 4.492 | 1.000 | 0.427/0.889/0.963 |
