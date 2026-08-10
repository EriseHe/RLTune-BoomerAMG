# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds.
Improvement and paired 95% intervals are relative to default setup + default solve.

## all_5000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 262.780 | 119.320 | 382.100 | 0.000 | 0.000 | 382.100 | 12.99 | baseline | 51/0/51 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 112.932 | 229.556 | 342.487 | 0.000 | 8.273 | 350.760 | 25.55 | 8.20% [7.16, 9.32] | 342/293/49 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 118.131 | 208.483 | 326.613 | 0.000 | 7.873 | 334.487 | 22.87 | 12.46% [11.47, 13.38] | 225/177/48 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 113.759 | 216.290 | 330.049 | 0.000 | 7.995 | 338.044 | 23.47 | 11.53% [10.38, 12.61] | 312/263/49 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 103.489 | 216.644 | 320.133 | 0.000 | 7.319 | 327.452 | 23.43 | 14.30% [13.37, 15.21] | 156/106/50 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 113.975 | 167.972 | 281.947 | 5.911 | 8.122 | 295.980 | 20.56 | 22.54% [21.40, 23.66] | 320/273/47 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 137.462 | 126.010 | 263.472 | 4.686 | 7.842 | 275.999 | 15.97 | 27.77% [26.76, 28.76] | 225/177/48 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 71072 | 1.740 | 0.109 | 9.260 | 0.067 | 0.494/0.706/0.891 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 55292 | 1.644 | 0.128 | 6.746 | 0.064 | 0.412/0.616/0.831 |

## first_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 256.330 | 116.066 | 372.395 | 0.000 | 0.000 | 372.395 | 12.89 | baseline | 9/0/9 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 159.322 | 270.874 | 430.196 | 0.000 | 8.265 | 438.461 | 31.08 | -17.74% [-21.73, -14.12] | 238/229/9 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 140.932 | 248.070 | 389.002 | 0.000 | 7.832 | 396.834 | 27.91 | -6.56% [-10.54, -3.36] | 159/150/9 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 163.389 | 265.041 | 428.429 | 0.000 | 8.062 | 436.491 | 29.89 | -17.21% [-21.60, -12.94] | 241/232/9 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 124.715 | 237.792 | 362.508 | 0.000 | 7.203 | 369.711 | 26.86 | 0.72% [-2.33, 3.40] | 109/100/9 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 168.162 | 274.070 | 442.232 | 0.000 | 8.231 | 450.464 | 31.73 | -20.96% [-24.68, -17.42] | 253/244/9 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 159.267 | 208.733 | 368.001 | 0.000 | 7.887 | 375.888 | 24.56 | -0.94% [-4.42, 2.49] | 160/151/9 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |

## last_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 265.385 | 121.046 | 386.431 | 0.000 | 0.000 | 386.431 | 12.97 | baseline | 10/0/10 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 100.892 | 209.314 | 310.206 | 0.000 | 8.342 | 318.548 | 22.85 | 17.57% [15.90, 19.12] | 17/8/9 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 115.195 | 193.824 | 309.020 | 0.000 | 7.909 | 316.929 | 20.97 | 17.99% [16.25, 19.55] | 14/5/9 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 100.312 | 200.903 | 301.215 | 0.000 | 8.209 | 309.424 | 21.63 | 19.93% [18.25, 21.60] | 16/7/9 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 98.125 | 207.198 | 305.323 | 0.000 | 7.380 | 312.703 | 22.08 | 19.08% [17.42, 20.61] | 11/1/10 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 99.513 | 139.115 | 238.628 | 7.275 | 8.120 | 254.024 | 17.40 | 34.26% [32.32, 35.92] | 15/5/10 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 131.508 | 106.410 | 237.917 | 5.896 | 7.856 | 251.670 | 13.76 | 34.87% [33.24, 36.38] | 19/9/10 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 17402 | 1.742 | 0.045 | 6.495 | 0.061 | 0.386/0.582/0.801 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 13760 | 1.656 | 0.059 | 5.973 | 0.054 | 0.340/0.530/0.765 |

## last_500

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 270.685 | 126.976 | 397.661 | 0.000 | 0.000 | 397.661 | 13.15 | baseline | 7/0/7 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 106.056 | 213.783 | 319.839 | 0.000 | 8.618 | 328.457 | 22.11 | 17.40% [14.83, 19.79] | 8/2/6 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 116.087 | 203.371 | 319.459 | 0.000 | 8.163 | 327.621 | 21.17 | 17.61% [14.81, 20.13] | 9/3/6 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 105.273 | 210.757 | 316.029 | 0.000 | 8.134 | 324.164 | 21.86 | 18.48% [15.81, 20.90] | 11/5/6 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 102.572 | 216.147 | 318.719 | 0.000 | 7.492 | 326.212 | 21.87 | 17.97% [15.37, 20.30] | 8/1/7 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 104.568 | 146.833 | 251.401 | 7.443 | 8.374 | 267.219 | 17.37 | 32.80% [29.40, 35.77] | 11/4/7 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 135.762 | 113.536 | 249.298 | 6.141 | 8.063 | 263.502 | 13.99 | 33.74% [31.16, 36.00] | 12/5/7 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 8687 | 1.742 | 0.044 | 7.532 | 0.059 | 0.380/0.582/0.797 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 6997 | 1.666 | 0.052 | 7.709 | 0.057 | 0.370/0.560/0.778 |
