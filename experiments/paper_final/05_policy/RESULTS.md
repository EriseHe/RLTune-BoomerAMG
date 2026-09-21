# Completed controlled RL and timing study

Completed: 2026-09-21T10:44:56Z.

All costs include unsuccessful attempts and recovery. Final failures remain unresolved; these totals do not mean all inputs were solved.

## Common-hierarchy online learning

One development execution on 5000 fresh inputs. All methods receive the same frozen setup-only hierarchy source; RL starts on input 1001. Logical totals add identical primary setup and source-selection costs to each policy.

| Policy | Logical total (s) | Final failures |
|---|---:|---:|
| Online LSTDQ | 1345.016 | 36 |
| Selected fixed weight | 1303.828 | 34 |
| Selected schedule | 1281.183 | 38 |
| Fixed weight 1 | 1550.155 | 37 |

RL saves 13.23% versus weight 1, including its learning overhead, but does not beat the selected fixed weight or schedule in cumulative runtime on this trajectory. The selected schedule has more final failures than RL.

## Corrected timing study

Two independently initialized executions of the same development stream, 5000 inputs per variant. Near-tie totals include the full independently measured calibration charge in each logical execution.

| Variant | Execution 1 (s) | Execution 2 (s) | Mean (s) |
|---|---:|---:|---:|
| near_tie | 1531.467 | 1479.055 | 1505.261 |
| raw | 1512.646 | 1424.129 | 1468.387 |
| stable | 1473.327 | 1445.861 | 1459.594 |

Near-tie is slower than raw in both executions (1.24% and 3.86%, including calibration). Stable priority is faster in one execution and slower in the other. Do not infer reliable variance reduction from two executions or select the more favorable execution. These results do not justify replacing the raw rule.

## Frozen evaluation and audit

The frozen evaluation retains all 128 held-out inputs on each of four hierarchy sources. Both old and new controllers and the prespecified fixed/schedule references are available in the machine-readable summary. Frozen timing repetitions are not training replicates.

All 5640 unique source recommendations in the retained attribution phases were independently reproduced. Their 24736 records contain no programming errors or mismatched paired hierarchy fingerprints. Online LSTDQ performed 57285 updates. Every corrected timing controller has nonzero training steps and changed parameters.

The initial timing stage at 75fdbbd passed a seven-value setup context to an eight-value solve API; those runs and their old smoke artifacts were discarded. They are not evidence for any algorithmic comparison. The corrected stage ran at 810d417 and ended with exit code 0.

## Artifacts

- [Machine-readable completed results](results_summary.json)
- [Frozen evaluation](../../../results/paper_final/05_policy/controlled/frozen/REPORT.md)
- [Common-hierarchy online learning](../../../results/paper_final/05_policy/controlled/online/REPORT.md)
- [Corrected timing results](../../../results/paper_final/09_algorithms/timing_corrected/runs/REPORT.md)

Raw datasets remain local and are delivered in the results ZIP. Git stores the code, protocol, validation evidence and compact result summary.
