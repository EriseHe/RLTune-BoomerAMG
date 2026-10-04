# Module 05 - historical Run 04 complete

Superseded for the current matched-hierarchy comparison by [Run 05](RUN05.md).

This historical evaluation uses the prescribed anchored **(2.6,1)** schedule. The earlier interrupted attempt was deleted without archiving; all current Run 04 data comes from the clean restart.

## Accepted scope

Six unchanged Run 03 Joint setup/controller checkpoints, including refreshed seed 4; the same 100 diffusion 60³ cases; three new timing repetitions. Fixed weights and setup tuples are unchanged. The only policy replacement is (2.5,1) to (2.6,1). Periodic (1,3) remains.

## Findings

Mean reductions compare ratios of summed costs, after averaging repetitions within each case. Sample SD is across six checkpoints. Native costs include all recovery work and exclude common primary setup and controller overhead.

| RL compared with | Native mean ± SD (pp) | Controller-inclusive mean ± SD (pp) | Native winning seeds |
|---|---:|---:|---:|
| Default w=1 | 41.67% ± 1.46 | 39.02% ± 1.49 | 6/6 |
| Global best fixed* | 11.60% ± 1.71 | 7.59% ± 1.76 | 6/6 |
| Per-instance best fixed* | 9.41% ± 1.49 | 5.30% ± 1.49 | 6/6 |
| Anchored (2.6, 1) | 3.22% ± 1.26 | -1.16% ± 1.25 | 6/6 |
| Periodic (1, 3) | 11.21% ± 5.01 | 7.19% ± 5.29 | 6/6 |

All 10,260 distinct final executions passed coverage and cost audits; zero unrecovered failures and 15 recovered executions, all charged. The archive also retains and independently verifies the 29,520 fixed-grid records supporting unchanged fixed choices.

RL has a native-cost advantage over (2.6,1) on all six checkpoints. With controller overhead included, RL wins on only one of six checkpoints and has a negative mean reduction. These results do not establish feedback necessity or a general runtime advantage over prescribed scheduling.

## Mathematical scope

The anchored minimax proof gives a*=(3+√5)/2; evaluating all 41 action-grid values selects 2.6. The native full-row ℓ₁-Jacobi smoother matches the normalization. The exact two-grid identity is valid, but full multilevel cycles have a weight-dependent recursive coarse solver. The schedule is therefore a theory-prescribed baseline whose runtime is tested empirically.

## Current deliverables

- [Complete Run 04 ZIP](../../../results/paper_final/05_policy/releases/module05_run04_anchored26_diffusion60_six_seeds_20260927.zip)
- [Results, tables and eight main figures](../../../results/paper_final/05_policy/releases/module05_run04_anchored26_diffusion60_six_seeds_20260927/RUN04_RESULTS.pdf)
- [Reviewed theory note](../../../results/paper_final/05_policy/releases/module05_run04_anchored26_diffusion60_six_seeds_20260927/theory/anchored_schedule_review.pdf)
- [Machine-readable completion](COMPLETION.json)

All 18 current figures and both PDFs were visually inspected. A fresh ZIP extraction passed the standalone standard-library verifier. Run `python3 reproduction/verify_archive.py` inside the extracted archive.

The existing Run 03 archive remains separate. The current archive contains only Run 04 final measurements, retained fixed-selection evidence, and current figures. The shared cases and accepted checkpoints were examined previously; this is not a fresh holdout or independent training replication. Main trajectory examples use the explicitly selected best observed seed 2; all six seeds remain in aggregate and supplementary figures.
