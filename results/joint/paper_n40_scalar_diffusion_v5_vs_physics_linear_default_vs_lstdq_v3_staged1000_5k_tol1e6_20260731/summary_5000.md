| Method | Setup ms | Native solve ms | Native total ms | Solve improvement vs reference | Controller ms | Bandit overhead ms | End-to-end ms | Primary failures | Recovered | Unrecovered |
|---|---|---|---|---|---|---|---|---|---|---|
| Default setup + default solve | 76.286 | 33.704 | 109.991 | +0.00% | 0.000 | 0.000 | 109.991 | 0 | 0 | 0 |
| Online LinUCB v5 + default solve | 30.043 | 53.677 | 83.720 | -59.26% | 0.000 | 7.803 | 91.523 | 197 | 197 | 0 |
| Online LinUCB [context=physics-linear] + default solve | 29.112 | 52.515 | 81.627 | -55.81% | 0.000 | 7.623 | 89.250 | 163 | 163 | 0 |
| Online LinUCB v5 + Recursive LSTDQ v3-LCB; activate=1000; tol=1e-06 | 28.420 | 32.451 | 60.870 | +3.72% | 3.492 | 7.864 | 72.227 | 163 | 163 | 0 |
| Online LinUCB [context=physics-linear] + Recursive LSTDQ v3-LCB; activate=1000; tol=1e-06; context=physics_linear | 28.855 | 34.052 | 62.908 | -1.03% | 3.880 | 7.621 | 74.409 | 153 | 153 | 0 |
