| Method | Setup ms | Native solve ms | Native total ms | Solve improvement vs reference | Controller ms | Bandit overhead ms | End-to-end ms | Primary failures | Recovered | Unrecovered |
|---|---|---|---|---|---|---|---|---|---|---|
| Default setup
+ default solve | 262.780 | 119.320 | 382.100 | +0.00% | 0.000 | 0.000 | 382.100 | 51 | 0 | 51 |
| LinUCB v5 (canonical 8D)
+ default solve | 112.932 | 229.556 | 342.487 | -92.39% | 0.000 | 8.273 | 350.760 | 342 | 293 | 49 |
| Physics-linear (7D)
+ default solve | 118.131 | 208.483 | 326.613 | -74.73% | 0.000 | 7.873 | 334.487 | 225 | 177 | 48 |
| No-c_mean (7D)
+ default solve | 113.759 | 216.290 | 330.049 | -81.27% | 0.000 | 7.995 | 338.044 | 312 | 263 | 49 |
| Means-only (3D)
+ default solve | 103.489 | 216.644 | 320.133 | -81.57% | 0.000 | 7.319 | 327.452 | 156 | 106 | 50 |
| LinUCB v5 (canonical 8D)
+ LSTDQ v3 | 113.975 | 167.972 | 281.947 | -40.77% | 5.911 | 8.122 | 295.980 | 320 | 273 | 47 |
| Physics-linear (7D)
+ LSTDQ v3 | 137.462 | 126.010 | 263.472 | -5.61% | 4.686 | 7.842 | 275.999 | 225 | 177 | 48 |
