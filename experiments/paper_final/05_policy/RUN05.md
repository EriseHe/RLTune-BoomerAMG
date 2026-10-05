# Module 05 Run 05 — grid-constrained weighted-minimax relaxation

Diffusion 60³; the same six Run 04 checkpoints (refreshed seed 4), 100 inputs and three fresh timing repetitions. The prescribed (2.85,1.10) replaces (2.6,1), and (1,3) is omitted; fixed weights and all hierarchy choices are retained.

Percentages are ratios of summed costs after averaging the three repetitions per case. Native continuation includes primary solves and all recovery setup/solve; common initial setup and controller dispatch are excluded. Inclusive continuation adds actual controller costs.

| RL compared with | Native reduction, mean ± sample SD (pp) | Inclusive reduction, mean ± sample SD (pp) | Native winning seeds |
|---|---:|---:|---:|
| Fixed w=1 | 41.82% ± 1.47 | 39.30% ± 1.49 | 6/6 |
| Stream-wide fixed* | 11.72% ± 1.68 | 7.90% ± 1.72 | 6/6 |
| Per-instance fixed* | 9.58% ± 1.43 | 5.67% ± 1.42 | 6/6 |
| Periodic (2.85, 1.10) | 0.63% ± 0.55 | -3.65% ± 0.63 | 5/6 |

Audit: 8,460 distinct executions, exact planned coverage; 0 unrecovered failures and 0 recovered executions (charged).

The jointly selected pair minimizes a normalized SPD polynomial-smoothing surrogate. It does not minimize the measured multilevel runtime by theorem. Full-cycle scheduling and high-first phase are explicit experimental transfers. The rigorous derivation and implementation audit are in `theory/`.

All checkpoints and test cases were already inspected in prior work; seed 4 was retrained after earlier results. This is a prescribed policy follow-up, not a fresh holdout or six newly trained seeds. Fixed comparators use successful best-observed 41-grid choices followed by retiming. The illustrative best seed is selected after evaluation; all six remain in the tables and figures.

Run 04 is retained separately as historical evidence. All five current policy roles are freshly timed together; current comparisons do not combine measurements from different runs.


## Current files and reproduction

- [Accepted protocol](../reproduction/matched_policy/protocol.json)
- [Accepted compact results](../reproduction/matched_policy/summary.json)
- [Frozen-input bundle and provenance](../reproduction/matched_policy/README.md)

The tracked bundle contains the six accepted frozen controllers, their encoders
and configurations, the exact input and job files, the retained selections, and
compact fixed-grid evidence. It is sufficient for a new matched retiming without
earlier study folders. Original artifact hashes and the accepted environment are
retained; the compact scan representation has its own hashes and derivation
record. It reconstructs every fixed choice from the original recorded times.

```sh
python -m experiments.paper_final.run_05_policy_minimax prepare --output NEW_DIRECTORY
python -m experiments.paper_final.run_05_policy_minimax run --output NEW_DIRECTORY
python -m experiments.paper_final.plot_05_policy_minimax --output NEW_DIRECTORY
python -m experiments.paper_final.package_05_policy_minimax --run NEW_DIRECTORY --output NEW_RELEASE_DIRECTORY
```

Use `--bundle PATH` to select another copy of the verified accepted bundle.
Preparation preserves the accepted choices and execution order, records the
current source and runtime, and compares that runtime with the accepted reference.
New timing measurements are distinct from the reported September measurements.
Resume requires the same prepared inputs, source files, native libraries, Python,
packages, MPI, and thread settings. Existing prepared measurements retain their
original source guards and cannot be resumed with changed execution code.

The packaged release includes final evaluation records, compressed fixed-grid
selection evidence, checkpoint hashes, captured sources, theory, tables and
figures. Its standard-library verifier checks coverage, executed schedules,
reported cost arrays, and independently reconstructed fixed choices. The original
raw fixed-scan hashes are recorded as provenance; compressed derived records are
verified against their own hashes.
