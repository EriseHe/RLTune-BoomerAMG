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

- [Results and eight main figures](../../../../results/paper_final/05_policy/releases/module05_run05_minimax285_diffusion60_six_seeds_20260929/RUN05_RESULTS.pdf)
- [Eight-figure atlas](../../../../results/paper_final/05_policy/20260929_run05_minimax285_joint_6seeds_100cases/analysis/paper_figures_best_seed/main_figure_atlas.pdf)
- [Complete data, checkpoint, figure and source archive](../../../../results/paper_final/05_policy/releases/module05_run05_minimax285_diffusion60_six_seeds_20260929.zip)
- [Independent archive verification](../../../../results/paper_final/05_policy/20260929_run05_minimax285_joint_6seeds_100cases/analysis/archive_verification.json)

The unfinished first attempt was discarded without archiving, as requested.
Every accepted measurement comes from the clean restart. The fresh run took
7.54 minutes including preflight and analysis. Each of the five policy roles
uses the same frozen Joint-trained hierarchy within checkpoint and input.
All six setup/controller checkpoints are unchanged. The main examples select
seed 2 by full-panel native RL reduction against W1; all six remain in the
aggregate results and ten supplementary heatmaps.

The original figure system supplies eight main figures and ten supplements.
The current comparison excludes both historical periodic baselines (1,3)
and (2.6,1). Historical runs remain intact and are not mixed with Run 05 timing.
The independent verifier reconstructs all cost arrays and fixed-grid choices;
a fresh ZIP extraction passed the verifier and all file hashes.

```sh
python -m experiments.paper_final.run_05_policy_minimax prepare --output NEW_DIRECTORY
python -m experiments.paper_final.run_05_policy_minimax run --output NEW_DIRECTORY
python -m experiments.paper_final.plot_05_policy_minimax --output NEW_DIRECTORY
```

The current paper uses only the official Module 04 online comparison and
this accepted Module 05 matched-hierarchy result. The separate solve-specific
checkpoint and frozen complete-method studies are archived development work
and are outside the current submission. Their source is retained in
[the development archive](../../archive/paper_development/README.md).

Historical prepared runs retain strict source hashes. After repository
reorganization, use their captured source snapshot to resume execution; use
the independent archive verifier to check completed measurements. New prepared
runs record the current shared helpers in their source manifests.
