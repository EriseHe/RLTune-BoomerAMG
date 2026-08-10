# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds.
Improvement and paired 95% intervals are relative to default setup + default solve.

## all_5000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 76.286 | 33.704 | 109.991 | 0.000 | 0.000 | 109.991 | 12.59 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 30.043 | 53.677 | 83.720 | 0.000 | 7.803 | 91.523 | 21.44 | 16.79% [15.91, 17.66] | 197/197/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 29.112 | 52.515 | 81.627 | 0.000 | 7.623 | 89.250 | 21.02 | 18.86% [18.02, 19.66] | 163/163/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 28.420 | 32.451 | 60.870 | 3.492 | 7.864 | 72.227 | 14.32 | 34.33% [33.48, 35.14] | 163/163/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 28.855 | 34.052 | 62.908 | 3.880 | 7.621 | 74.409 | 15.02 | 32.35% [31.51, 33.15] | 153/153/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 46448 | 1.879 | 0.134 | 0.867 | 0.043 | 0.350/0.572/0.794 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 50345 | 1.869 | 0.128 | 0.756 | 0.039 | 0.318/0.535/0.772 |

## first_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 75.678 | 33.277 | 108.955 | 0.000 | 0.000 | 108.955 | 12.57 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 42.737 | 64.658 | 107.394 | 0.000 | 7.889 | 115.283 | 26.33 | -5.81% [-9.42, -2.19] | 188/188/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 40.526 | 61.693 | 102.219 | 0.000 | 7.742 | 109.961 | 24.92 | -0.92% [-4.22, 2.50] | 153/153/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 39.251 | 62.414 | 101.665 | 0.000 | 7.948 | 109.613 | 25.18 | -0.60% [-3.85, 2.55] | 152/152/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 39.598 | 61.063 | 100.661 | 0.000 | 7.658 | 108.319 | 24.78 | 0.58% [-2.81, 3.90] | 149/149/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |

## last_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 75.910 | 34.104 | 110.014 | 0.000 | 0.000 | 110.014 | 12.59 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 26.194 | 51.135 | 77.329 | 0.000 | 7.815 | 85.144 | 20.35 | 22.61% [21.76, 23.43] | 2/2/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 25.991 | 49.792 | 75.783 | 0.000 | 7.635 | 83.419 | 19.90 | 24.17% [23.23, 25.08] | 1/1/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 25.373 | 24.147 | 49.520 | 4.542 | 7.842 | 61.905 | 11.28 | 43.73% [42.96, 44.40] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 26.271 | 25.887 | 52.157 | 4.728 | 7.565 | 64.450 | 11.87 | 41.42% [40.81, 41.98] | 0/0/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 11284 | 1.901 | 0.065 | 0.455 | 0.034 | 0.261/0.463/0.696 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 11873 | 1.887 | 0.061 | 0.427 | 0.035 | 0.228/0.422/0.676 |

## last_500

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 76.195 | 34.422 | 110.617 | 0.000 | 0.000 | 110.617 | 12.59 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 26.365 | 51.783 | 78.148 | 0.000 | 7.971 | 86.119 | 20.31 | 22.15% [21.28, 23.02] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 26.521 | 50.476 | 76.997 | 0.000 | 7.793 | 84.790 | 19.79 | 23.35% [21.96, 24.65] | 1/1/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 25.597 | 24.275 | 49.872 | 4.809 | 7.974 | 62.655 | 11.23 | 43.36% [42.66, 44.03] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 26.618 | 25.977 | 52.595 | 5.151 | 7.708 | 65.454 | 11.75 | 40.83% [40.01, 41.60] | 0/0/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 5613 | 1.903 | 0.062 | 0.410 | 0.028 | 0.253/0.444/0.675 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 5876 | 1.879 | 0.055 | 0.396 | 0.030 | 0.217/0.409/0.670 |
