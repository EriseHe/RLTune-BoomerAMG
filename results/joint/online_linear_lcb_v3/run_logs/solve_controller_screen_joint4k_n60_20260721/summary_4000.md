| Method | Setup ms | Native solve ms | Native total ms | Solve improvement vs reference | Controller ms | Bandit overhead ms | End-to-end ms | Primary failures | Recovered | Unrecovered |
|---|---|---|---|---|---|---|---|---|---|---|
| Online LinUCB + fixed w=1.6 | 108.715 | 184.372 | 293.088 | +0.00% | 0.000 | 7.095 | 300.183 | 129 | 129 | 0 |
| Online LinUCB + Recursive LSTDQ-LCB | 112.915 | 165.745 | 278.660 | +10.10% | 14.211 | 6.909 | 299.780 | 134 | 134 | 0 |
| Online LinUCB + Recursive LSTDQ v2-LCB | 111.252 | 155.983 | 267.235 | +15.40% | 16.072 | 6.694 | 290.001 | 122 | 122 | 0 |
| Online LinUCB + Structured model-based | 113.824 | 165.748 | 279.572 | +10.10% | 15.671 | 6.940 | 302.184 | 134 | 134 | 0 |
| Online LinUCB + Recalibrated LSVI-LCB | 134.218 | 189.859 | 324.077 | -2.98% | 166.598 | 7.416 | 498.091 | 426 | 426 | 0 |
