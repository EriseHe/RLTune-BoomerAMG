| Method | Setup ms | Native solve ms | Native total ms | Solve improvement vs reference | Controller ms | Bandit overhead ms | End-to-end ms | Primary failures | Recovered | Unrecovered |
|---|---|---|---|---|---|---|---|---|---|---|
| Default setup + default solve | 259.857 | 115.052 | 374.909 | +0.00% | 0.000 | 0.000 | 374.909 | 0 | 0 | 0 |
| Online LinUCB v5 + default solve | 96.667 | 170.108 | 266.775 | -47.85% | 0.000 | 7.946 | 274.721 | 109 | 109 | 0 |
| Online LinUCB [context=physics-linear] + default solve | 98.167 | 169.873 | 268.040 | -47.65% | 0.000 | 7.664 | 275.704 | 118 | 118 | 0 |
| Online LinUCB v5 + Recursive LSTDQ v3-LCB; activate=1000; tol=1e-06 | 96.959 | 132.661 | 229.619 | -15.30% | 4.973 | 7.883 | 242.475 | 143 | 143 | 0 |
| Online LinUCB [context=physics-linear] + Recursive LSTDQ v3-LCB; activate=1000; tol=1e-06; context=physics_linear | 98.553 | 112.170 | 210.724 | +2.50% | 3.991 | 7.680 | 222.395 | 121 | 121 | 0 |
