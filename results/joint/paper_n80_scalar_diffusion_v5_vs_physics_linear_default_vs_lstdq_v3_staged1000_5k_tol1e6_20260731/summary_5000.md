| Method | Setup ms | Native solve ms | Native total ms | Solve improvement vs reference | Controller ms | Bandit overhead ms | End-to-end ms | Primary failures | Recovered | Unrecovered |
|---|---|---|---|---|---|---|---|---|---|---|
| Default setup + default solve | 625.946 | 292.678 | 918.625 | +0.00% | 0.000 | 0.000 | 918.625 | 0 | 0 | 0 |
| Online LinUCB v5 + default solve | 258.738 | 440.875 | 699.614 | -50.63% | 0.000 | 8.226 | 707.840 | 119 | 119 | 0 |
| Online LinUCB [context=physics-linear] + default solve | 241.617 | 446.644 | 688.261 | -52.61% | 0.000 | 8.149 | 696.410 | 148 | 148 | 0 |
| Online LinUCB v5 + Recursive LSTDQ v3-LCB; activate=1000; tol=1e-06 | 233.157 | 294.654 | 527.811 | -0.68% | 4.781 | 8.155 | 540.747 | 133 | 133 | 0 |
| Online LinUCB [context=physics-linear] + Recursive LSTDQ v3-LCB; activate=1000; tol=1e-06; context=physics_linear | 272.003 | 300.957 | 572.960 | -2.83% | 4.316 | 8.034 | 585.310 | 116 | 116 | 0 |
