| Method | Setup ms | Native solve ms | Native total ms | Solve improvement vs reference | Controller ms | Bandit overhead ms | End-to-end ms | Primary failures | Recovered | Unrecovered |
|---|---|---|---|---|---|---|---|---|---|---|
| Default setup
+ default solve | 258.537 | 114.467 | 373.004 | +0.00% | 0.000 | 0.000 | 373.004 | 0 | 0 | 0 |
| LinUCB v5 (canonical 8D)
+ default solve | 97.443 | 172.819 | 270.262 | -50.98% | 0.000 | 7.847 | 278.109 | 130 | 130 | 0 |
| Physics-linear (7D)
+ default solve | 108.711 | 172.975 | 281.685 | -51.11% | 0.000 | 7.626 | 289.311 | 121 | 121 | 0 |
| No-c_mean (7D)
+ default solve | 104.339 | 178.509 | 282.849 | -55.95% | 0.000 | 7.702 | 290.551 | 133 | 133 | 0 |
| Means-only (3D)
+ default solve | 104.099 | 170.279 | 274.378 | -48.76% | 0.000 | 7.057 | 281.435 | 73 | 73 | 0 |
| LinUCB v5 (canonical 8D)
+ LSTDQ v3 | 96.418 | 111.363 | 207.781 | +2.71% | 4.033 | 7.814 | 219.628 | 130 | 130 | 0 |
| Physics-linear (7D)
+ LSTDQ v3 | 97.565 | 111.415 | 208.980 | +2.67% | 4.068 | 7.635 | 220.683 | 115 | 115 | 0 |
