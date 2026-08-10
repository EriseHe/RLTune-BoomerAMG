# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds.
Improvement and paired 95% intervals are relative to default setup + default solve.

## all_5000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 625.946 | 292.678 | 918.625 | 0.000 | 0.000 | 918.625 | 13.35 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 258.738 | 440.875 | 699.614 | 0.000 | 8.226 | 707.840 | 20.34 | 22.95% [22.17, 23.73] | 119/119/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 241.617 | 446.644 | 688.261 | 0.000 | 8.149 | 696.410 | 21.64 | 24.19% [23.30, 25.05] | 148/148/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 233.157 | 294.654 | 527.811 | 4.781 | 8.155 | 540.747 | 15.55 | 41.14% [40.33, 41.97] | 133/133/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 272.003 | 300.957 | 572.960 | 4.316 | 8.034 | 585.310 | 14.59 | 36.28% [35.43, 37.14] | 116/116/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 52586 | 1.830 | 0.125 | 5.592 | 0.016 | 0.298/0.511/0.746 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 47045 | 1.825 | 0.134 | 5.917 | 0.014 | 0.351/0.584/0.829 |

## first_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 626.418 | 291.019 | 917.437 | 0.000 | 0.000 | 917.437 | 13.32 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 322.176 | 499.487 | 821.663 | 0.000 | 8.378 | 830.041 | 23.30 | 9.53% [6.40, 12.62] | 113/113/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 315.843 | 520.270 | 836.113 | 0.000 | 8.293 | 844.406 | 25.02 | 7.96% [4.35, 11.23] | 128/128/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 312.758 | 517.211 | 829.970 | 0.000 | 8.372 | 838.342 | 25.19 | 8.62% [5.18, 11.68] | 127/127/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 312.025 | 557.713 | 869.737 | 0.000 | 8.345 | 878.083 | 25.90 | 4.29% [1.13, 7.37] | 108/108/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |

## last_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 632.533 | 296.876 | 929.408 | 0.000 | 0.000 | 929.408 | 13.33 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 238.713 | 426.217 | 664.929 | 0.000 | 8.270 | 673.199 | 19.87 | 27.57% [26.42, 28.64] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 223.911 | 426.486 | 650.397 | 0.000 | 8.204 | 658.601 | 20.52 | 29.14% [28.11, 30.09] | 1/1/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 214.019 | 238.301 | 452.320 | 6.020 | 8.219 | 466.559 | 13.02 | 49.80% [49.20, 50.39] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 261.747 | 232.907 | 494.654 | 5.385 | 8.164 | 508.203 | 11.49 | 45.32% [44.30, 46.30] | 1/1/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 13019 | 1.837 | 0.058 | 3.324 | 0.016 | 0.216/0.409/0.642 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 11488 | 1.816 | 0.065 | 4.252 | 0.011 | 0.316/0.548/0.798 |

## last_500

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 626.899 | 294.195 | 921.094 | 0.000 | 0.000 | 921.094 | 13.31 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 238.247 | 422.571 | 660.817 | 0.000 | 8.210 | 669.028 | 19.78 | 27.37% [25.94, 28.77] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 221.488 | 420.614 | 642.102 | 0.000 | 8.145 | 650.247 | 20.37 | 29.40% [28.21, 30.51] | 0/0/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 212.744 | 235.119 | 447.863 | 5.904 | 8.148 | 461.915 | 12.93 | 49.85% [49.07, 50.60] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 260.287 | 230.685 | 490.971 | 5.363 | 8.276 | 504.611 | 11.41 | 45.22% [43.84, 46.53] | 0/0/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 6466 | 1.836 | 0.052 | 2.757 | 0.015 | 0.210/0.398/0.631 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 5704 | 1.822 | 0.063 | 3.554 | 0.009 | 0.306/0.530/0.782 |
