# Solve-Controller Screening Report

All runtimes are per-instance means in milliseconds. Improvement and
paired 95% intervals are relative to Online LinUCB + fixed `w=1.6`.

## all_4000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit_fixed_w1.6 | 108.715 | 184.372 | 293.088 | 0.000 | 7.095 | 300.183 | 22.65 | baseline | 129/129/0 | 1.000 |
| bandit_recursive_lstdq_lcb | 112.915 | 165.745 | 278.660 | 14.211 | 6.909 | 299.780 | 20.71 | 0.13% [-0.85, 1.12] | 134/134/0 | 0.007 |
| bandit_recursive_lstdq_v2_lcb | 111.252 | 155.983 | 267.235 | 16.072 | 6.694 | 290.001 | 18.24 | 3.39% [2.42, 4.28] | 122/122/0 | 0.009 |
| bandit_structured_model_based | 113.824 | 165.748 | 279.572 | 15.671 | 6.940 | 302.184 | 19.36 | -0.67% [-1.52, 0.23] | 134/134/0 | 0.009 |
| bandit_recalibrated_lsvi_lcb | 134.218 | 189.859 | 324.077 | 166.598 | 7.416 | 498.091 | 23.07 | -65.93% [-84.17, -50.16] | 426/426/0 | 0.008 |

Controller diagnostics:

| method | decisions | mean w | explored | mean uncertainty (ms) | lower-bound saturation | calibration coverage 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| bandit_recursive_lstdq_lcb | 82854 | 1.811 | 0.094 | 11.434 | 0.083 | 0.502/0.709/0.894 |
| bandit_recursive_lstdq_v2_lcb | 72973 | 1.834 | 0.101 | 0.838 | 0.085 | 0.062/0.117/0.208 |
| bandit_structured_model_based | 77429 | 1.655 | 0.099 | n/a | n/a | n/a |
| bandit_recalibrated_lsvi_lcb | 92277 | 1.777 | 0.088 | 2.441 | 0.132 | 0.266/0.486/0.738 |

## first_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit_fixed_w1.6 | 130.754 | 213.688 | 344.441 | 0.000 | 7.294 | 351.735 | 27.28 | baseline | 123/123/0 | 1.000 |
| bandit_recursive_lstdq_lcb | 142.094 | 195.031 | 337.125 | 15.443 | 7.130 | 359.698 | 23.72 | -2.26% [-4.45, -0.03] | 121/121/0 | 0.028 |
| bandit_recursive_lstdq_v2_lcb | 138.347 | 186.270 | 324.617 | 20.750 | 6.986 | 352.352 | 22.19 | -0.18% [-2.97, 2.46] | 119/119/0 | 0.034 |
| bandit_structured_model_based | 142.726 | 201.865 | 344.591 | 18.460 | 7.294 | 370.345 | 23.56 | -5.29% [-7.99, -2.83] | 125/125/0 | 0.034 |
| bandit_recalibrated_lsvi_lcb | 161.664 | 215.484 | 377.148 | 69.623 | 7.770 | 454.541 | 26.50 | -29.23% [-40.36, -19.38] | 198/198/0 | 0.032 |

Controller diagnostics:

| method | decisions | mean w | explored | mean uncertainty (ms) | lower-bound saturation | calibration coverage 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| bandit_recursive_lstdq_lcb | 23719 | 1.793 | 0.188 | 28.361 | 0.109 | 0.751/0.898/0.980 |
| bandit_recursive_lstdq_v2_lcb | 22188 | 1.826 | 0.189 | 1.859 | 0.097 | 0.127/0.232/0.384 |
| bandit_structured_model_based | 23559 | 1.637 | 0.188 | n/a | n/a | n/a |
| bandit_recalibrated_lsvi_lcb | 26500 | 1.756 | 0.179 | 4.115 | 0.366 | 0.310/0.589/0.850 |

## last_1000

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit_fixed_w1.6 | 99.508 | 164.384 | 263.892 | 0.000 | 6.847 | 270.739 | 19.97 | baseline | 0/0/0 | 1.000 |
| bandit_recursive_lstdq_lcb | 100.485 | 152.926 | 253.411 | 14.356 | 6.609 | 274.375 | 19.34 | -1.34% [-3.69, 0.75] | 4/4/0 | 0.000 |
| bandit_recursive_lstdq_v2_lcb | 99.190 | 141.595 | 240.784 | 13.765 | 6.432 | 260.981 | 16.89 | 3.60% [2.55, 4.50] | 1/1/0 | 0.000 |
| bandit_structured_model_based | 102.131 | 150.533 | 252.664 | 14.154 | 6.454 | 273.272 | 18.02 | -0.94% [-2.04, 0.11] | 4/4/0 | 0.000 |
| bandit_recalibrated_lsvi_lcb | 114.503 | 174.980 | 289.483 | 258.644 | 6.932 | 555.060 | 21.67 | -105.02% [-167.71, -51.24] | 55/55/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean uncertainty (ms) | lower-bound saturation | calibration coverage 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| bandit_recursive_lstdq_lcb | 19338 | 1.854 | 0.036 | 3.645 | 0.053 | 0.333/0.543/0.791 |
| bandit_recursive_lstdq_v2_lcb | 16891 | 1.857 | 0.042 | 0.244 | 0.073 | 0.027/0.053/0.106 |
| bandit_structured_model_based | 18023 | 1.663 | 0.039 | n/a | n/a | n/a |
| bandit_recalibrated_lsvi_lcb | 21666 | 1.788 | 0.036 | 1.518 | 0.043 | 0.223/0.407/0.656 |

## last_500

| method | setup | native solve | native total | controller | bandit | end-to-end | cycles | E2E improvement (95% CI) | primary/recovered/unrecovered | same setup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bandit_fixed_w1.6 | 96.742 | 155.402 | 252.145 | 0.000 | 6.277 | 258.421 | 19.33 | baseline | 0/0/0 | 1.000 |
| bandit_recursive_lstdq_lcb | 97.889 | 154.087 | 251.976 | 15.213 | 6.066 | 273.254 | 19.30 | -5.74% [-9.75, -2.66] | 1/1/0 | 0.000 |
| bandit_recursive_lstdq_v2_lcb | 95.759 | 136.913 | 232.673 | 12.753 | 5.886 | 251.313 | 16.84 | 2.75% [1.93, 3.59] | 0/0/0 | 0.000 |
| bandit_structured_model_based | 98.955 | 145.496 | 244.451 | 13.399 | 5.800 | 263.650 | 17.99 | -2.02% [-3.12, -1.08] | 1/1/0 | 0.000 |
| bandit_recalibrated_lsvi_lcb | 112.954 | 178.705 | 291.658 | 270.102 | 5.994 | 567.754 | 22.37 | -119.70% [-218.74, -40.97] | 30/30/0 | 0.000 |

Controller diagnostics:

| method | decisions | mean w | explored | mean uncertainty (ms) | lower-bound saturation | calibration coverage 1x/2x/4x |
|---|---:|---:|---:|---:|---:|---:|
| bandit_recursive_lstdq_lcb | 9649 | 1.861 | 0.031 | 3.139 | 0.047 | 0.311/0.526/0.779 |
| bandit_recursive_lstdq_v2_lcb | 8418 | 1.872 | 0.040 | 0.229 | 0.078 | 0.026/0.051/0.098 |
| bandit_structured_model_based | 8997 | 1.672 | 0.039 | n/a | n/a | n/a |
| bandit_recalibrated_lsvi_lcb | 11185 | 1.769 | 0.034 | 1.530 | 0.035 | 0.222/0.411/0.664 |
