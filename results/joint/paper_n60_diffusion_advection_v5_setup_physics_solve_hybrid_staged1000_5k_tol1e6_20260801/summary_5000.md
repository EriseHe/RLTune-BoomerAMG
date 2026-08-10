| Method | Setup ms | Native solve ms | Native total ms | Solve improvement vs reference | Controller ms | Bandit overhead ms | End-to-end ms | Primary failures | Recovered | Unrecovered |
|---|---|---|---|---|---|---|---|---|---|---|
| Default setup
+ default solve | 258.450 | 118.089 | 376.539 | +0.00% | 0.000 | 0.000 | 376.539 | 51 | 0 | 51 |
| LinUCB v5 (canonical 8D)
+ LSTDQ v3 | 106.179 | 158.338 | 264.516 | -34.08% | 5.505 | 8.193 | 278.215 | 291 | 242 | 49 |
| Physics-linear (7D)
+ LSTDQ v3 | 106.074 | 167.362 | 273.437 | -41.72% | 6.261 | 7.832 | 287.529 | 274 | 226 | 48 |
| LinUCB v5 (canonical 8D)
+ LSTDQ v3
solve_context=physics-linear | 115.209 | 157.358 | 272.567 | -33.25% | 5.164 | 8.123 | 285.853 | 296 | 246 | 50 |
