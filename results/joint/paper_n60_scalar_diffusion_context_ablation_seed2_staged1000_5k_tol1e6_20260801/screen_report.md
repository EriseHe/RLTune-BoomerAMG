# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds.
Improvement and paired 95% intervals are relative to default setup + default solve.

## all_5000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 258.537 | 114.467 | 373.004 | 0.000 | 0.000 | 373.004 | 12.57 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 97.443 | 172.819 | 270.262 | 0.000 | 7.847 | 278.109 | 20.32 | 25.44% [24.68, 26.16] | 130/130/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 108.711 | 172.975 | 281.685 | 0.000 | 7.626 | 289.311 | 19.03 | 22.44% [21.68, 23.20] | 121/121/0 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 104.339 | 178.509 | 282.849 | 0.000 | 7.702 | 290.551 | 19.91 | 22.11% [21.33, 22.81] | 133/133/0 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 104.099 | 170.279 | 274.378 | 0.000 | 7.057 | 281.435 | 18.68 | 24.55% [23.80, 25.20] | 73/73/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 96.418 | 111.363 | 207.781 | 4.033 | 7.814 | 219.628 | 14.06 | 41.12% [40.27, 41.88] | 130/130/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 97.565 | 111.415 | 208.980 | 4.068 | 7.635 | 220.683 | 14.03 | 40.84% [40.04, 41.63] | 115/115/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 47192 | 1.776 | 0.138 | 1.536 | 0.013 | 0.234/0.418/0.664 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 46972 | 1.795 | 0.138 | 1.051 | 0.005 | 0.209/0.380/0.627 |

## first_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 255.064 | 114.150 | 369.214 | 0.000 | 0.000 | 369.214 | 12.56 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 127.154 | 199.840 | 326.994 | 0.000 | 7.788 | 334.782 | 23.60 | 9.33% [6.05, 12.41] | 127/127/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 135.085 | 201.593 | 336.679 | 0.000 | 7.566 | 344.244 | 22.64 | 6.76% [3.71, 9.67] | 119/119/0 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 128.081 | 214.058 | 342.139 | 0.000 | 7.679 | 349.818 | 24.52 | 5.25% [2.07, 8.46] | 123/123/0 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 120.440 | 187.330 | 307.770 | 0.000 | 6.908 | 314.678 | 20.83 | 14.77% [12.01, 17.28] | 69/69/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 127.420 | 197.339 | 324.758 | 0.000 | 7.840 | 332.598 | 23.12 | 9.92% [6.72, 12.97] | 126/126/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 128.173 | 199.723 | 327.897 | 0.000 | 7.601 | 335.498 | 23.18 | 9.13% [5.85, 12.25] | 114/114/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 0 | nan | 0.000 | 0.000 | 0.000 | 0.000/0.000/0.000 |

## last_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 266.377 | 116.972 | 383.350 | 0.000 | 0.000 | 383.350 | 12.55 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 91.491 | 169.756 | 261.247 | 0.000 | 8.163 | 269.410 | 19.43 | 29.72% [28.92, 30.51] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 102.387 | 167.442 | 269.829 | 0.000 | 7.832 | 277.661 | 18.09 | 27.57% [26.59, 28.58] | 0/0/0 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 98.792 | 172.456 | 271.249 | 0.000 | 7.993 | 279.241 | 18.85 | 27.16% [26.15, 28.21] | 2/2/0 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 100.271 | 167.864 | 268.136 | 0.000 | 7.271 | 275.407 | 18.11 | 28.16% [27.16, 29.11] | 0/0/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 90.794 | 90.779 | 181.574 | 5.129 | 8.086 | 194.789 | 11.62 | 49.19% [48.61, 49.73] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 91.578 | 89.646 | 181.225 | 5.118 | 7.818 | 194.160 | 11.50 | 49.35% [48.82, 49.89] | 0/0/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 11618 | 1.765 | 0.068 | 1.269 | 0.014 | 0.191/0.348/0.573 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 11501 | 1.810 | 0.069 | 0.686 | 0.005 | 0.138/0.265/0.486 |

## last_500

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default_setup_default_solve | 263.870 | 116.154 | 380.023 | 0.000 | 0.000 | 380.023 | 12.51 | baseline | 0/0/0 | 1.000 |
| linucb_v5_canonical8d_recommended_structured512_default_solve | 92.179 | 170.121 | 262.300 | 0.000 | 8.252 | 270.552 | 19.40 | 28.81% [27.49, 30.02] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_default_solve | 102.179 | 167.110 | 269.289 | 0.000 | 7.857 | 277.146 | 18.01 | 27.07% [25.61, 28.41] | 0/0/0 | 0.000 |
| linucb_canonical_no_c_mean7d_recommended_structured512_default_solve | 97.995 | 171.945 | 269.939 | 0.000 | 8.023 | 277.962 | 18.83 | 26.86% [25.45, 28.19] | 1/1/0 | 0.000 |
| linucb_canonical_means_only3d_recommended_structured512_default_solve | 99.985 | 167.716 | 267.701 | 0.000 | 7.279 | 274.981 | 18.05 | 27.64% [26.19, 29.00] | 0/0/0 | 0.000 |
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 90.626 | 90.380 | 181.006 | 5.051 | 8.155 | 194.212 | 11.58 | 48.89% [48.16, 49.68] | 0/0/0 | 0.000 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 91.618 | 90.142 | 181.760 | 5.096 | 7.845 | 194.701 | 11.53 | 48.77% [48.02, 49.54] | 0/0/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean parameter uncertainty (ms) | lower-bound saturation | TD residual within parameter width 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| linucb_v5_canonical8d_recommended_structured512_lstdq_v3_canonical_staged1000_tol1e6 | 5789 | 1.775 | 0.065 | 1.013 | 0.015 | 0.178/0.337/0.559 |
| linucb_physics_linear7d_recommended_structured512_lstdq_v3_physics_linear_staged1000_tol1e6 | 5766 | 1.805 | 0.065 | 0.681 | 0.006 | 0.134/0.259/0.476 |
