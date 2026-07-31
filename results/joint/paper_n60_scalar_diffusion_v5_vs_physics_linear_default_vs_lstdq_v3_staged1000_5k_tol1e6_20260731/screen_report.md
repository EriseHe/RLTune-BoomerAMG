# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds.
Improvement and paired 95% intervals are relative to default setup + default solve.

## all_5000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 259.857 | 115.052 | 374.909 | 0.000 | 0.000 | 374.909 | 12.58 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 96.667 | 170.108 | 266.775 | 0.000 | 7.946 | 274.721 | 19.92 | 26.72% [25.97, 27.46] | 109/109/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 98.167 | 169.873 | 268.040 | 0.000 | 7.664 | 275.704 | 19.93 | 26.46% [25.70, 27.16] | 118/118/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 96.959 | 132.661 | 229.619 | 4.973 | 7.883 | 242.475 | 16.71 | 35.32% [34.47, 36.07] | 143/143/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 98.553 | 112.170 | 210.724 | 3.991 | 7.680 | 222.395 | 14.05 | 40.68% [39.83, 41.50] | 121/121/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 59987 | 1.719 | 0.115 | 3.489 | 0.024 | 0.367/0.610/0.844 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 46796 | 1.800 | 0.134 | 2.982 | 0.021 | 0.321/0.540/0.777 |

## first_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 265.729 | 116.594 | 382.324 | 0.000 | 0.000 | 382.324 | 12.56 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 125.205 | 202.127 | 327.332 | 0.000 | 8.267 | 335.599 | 23.11 | 12.22% [9.08, 14.98] | 103/103/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 126.892 | 200.840 | 327.731 | 0.000 | 7.946 | 335.678 | 23.09 | 12.20% [9.24, 15.09] | 108/108/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 136.004 | 204.706 | 340.710 | 0.000 | 8.129 | 348.839 | 23.57 | 8.76% [5.40, 12.02] | 132/132/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 133.830 | 206.869 | 340.699 | 0.000 | 8.051 | 348.750 | 23.47 | 8.78% [5.25, 12.10] | 118/118/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |

## last_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 260.527 | 116.517 | 377.044 | 0.000 | 0.000 | 377.044 | 12.58 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 90.266 | 162.083 | 252.349 | 0.000 | 7.969 | 260.318 | 18.97 | 30.96% [29.95, 31.82] | 1/1/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 91.518 | 160.355 | 251.873 | 0.000 | 7.669 | 259.542 | 18.88 | 31.16% [30.42, 31.88] | 0/0/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 87.642 | 113.417 | 201.059 | 6.467 | 7.942 | 215.468 | 14.79 | 42.85% [42.19, 43.53] | 1/1/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 90.329 | 87.583 | 177.912 | 5.038 | 7.620 | 190.570 | 11.53 | 49.46% [48.93, 50.00] | 0/0/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 14786 | 1.710 | 0.048 | 2.036 | 0.030 | 0.285/0.501/0.760 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 11530 | 1.806 | 0.065 | 1.396 | 0.013 | 0.217/0.390/0.640 |

## last_500

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 255.948 | 115.546 | 371.494 | 0.000 | 0.000 | 371.494 | 12.58 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 88.878 | 159.042 | 247.920 | 0.000 | 7.849 | 255.769 | 18.89 | 31.15% [30.22, 32.13] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 90.360 | 158.336 | 248.696 | 0.000 | 7.570 | 256.266 | 18.82 | 31.02% [30.06, 31.96] | 0/0/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 86.460 | 111.379 | 197.839 | 6.380 | 7.816 | 212.034 | 14.66 | 42.92% [42.08, 43.73] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 89.213 | 85.837 | 175.051 | 4.790 | 7.523 | 187.363 | 11.40 | 49.57% [48.89, 50.24] | 0/0/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 7329 | 1.716 | 0.044 | 1.745 | 0.032 | 0.274/0.481/0.745 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 5700 | 1.816 | 0.062 | 1.001 | 0.013 | 0.206/0.378/0.628 |
