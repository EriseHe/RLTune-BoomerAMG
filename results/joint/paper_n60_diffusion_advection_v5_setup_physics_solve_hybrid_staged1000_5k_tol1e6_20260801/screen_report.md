# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds.
Improvement and paired 95% intervals are relative to default setup + default solve.

## all_5000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 258.450 | 118.089 | 376.539 | 0.000 | 0.000 | 376.539 | 12.99 | baseline | 51/0/51 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 106.179 | 158.338 | 264.516 | 5.505 | 8.193 | 278.215 | 19.69 | 26.11% [25.04, 27.22] | 291/242/49 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 106.074 | 167.362 | 273.437 | 6.261 | 7.832 | 287.529 | 21.44 | 23.64% [22.50, 24.77] | 274/226/48 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 115.209 | 157.358 | 272.567 | 5.164 | 8.123 | 285.853 | 18.50 | 24.08% [22.91, 25.23] | 296/246/50 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 67884 | 1.883 | 0.112 | 6.947 | 0.065 | 0.410/0.643/0.848 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 76027 | 1.865 | 0.104 | 8.488 | 0.064 | 0.431/0.639/0.831 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 62247 | 1.815 | 0.119 | 8.673 | 0.058 | 0.436/0.662/0.872 |

## first_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 256.893 | 118.016 | 374.909 | 0.000 | 0.000 | 374.909 | 12.89 | baseline | 9/0/9 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 160.685 | 266.381 | 427.066 | 0.000 | 8.424 | 435.491 | 30.58 | -16.16% [-19.82, -12.83] | 230/221/9 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 146.334 | 282.149 | 428.483 | 0.000 | 8.176 | 436.659 | 31.15 | -16.47% [-20.26, -12.92] | 195/186/9 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 158.638 | 264.906 | 423.544 | 0.000 | 8.333 | 431.877 | 30.24 | -15.19% [-19.03, -11.59] | 228/219/9 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |

## last_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 257.991 | 117.340 | 375.331 | 0.000 | 0.000 | 375.331 | 12.97 | baseline | 10/0/10 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 95.861 | 121.597 | 217.458 | 6.306 | 8.001 | 231.765 | 15.37 | 38.25% [36.55, 39.89] | 15/5/10 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 93.895 | 135.565 | 229.460 | 8.231 | 7.701 | 245.392 | 19.86 | 34.62% [32.56, 36.45] | 22/12/10 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 101.902 | 128.126 | 230.029 | 6.357 | 8.027 | 244.413 | 15.37 | 34.88% [32.96, 36.66] | 17/7/10 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 15370 | 1.901 | 0.047 | 5.347 | 0.076 | 0.282/0.501/0.744 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 19856 | 1.955 | 0.043 | 6.662 | 0.063 | 0.399/0.598/0.787 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 15373 | 1.817 | 0.051 | 6.571 | 0.066 | 0.344/0.553/0.791 |

## last_500

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 256.974 | 119.667 | 376.641 | 0.000 | 0.000 | 376.641 | 13.15 | baseline | 7/0/7 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 98.079 | 125.794 | 223.873 | 6.439 | 8.060 | 238.372 | 15.48 | 36.71% [33.75, 39.31] | 10/3/7 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 96.032 | 138.733 | 234.765 | 8.739 | 7.753 | 251.257 | 20.28 | 33.29% [29.63, 36.36] | 15/8/7 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 103.729 | 131.192 | 234.921 | 6.359 | 8.137 | 249.418 | 15.20 | 33.78% [30.66, 36.44] | 11/4/7 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 7742 | 1.894 | 0.045 | 5.843 | 0.073 | 0.289/0.498/0.735 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 10141 | 1.970 | 0.040 | 7.365 | 0.062 | 0.414/0.609/0.789 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 7600 | 1.811 | 0.047 | 7.339 | 0.064 | 0.343/0.547/0.780 |
