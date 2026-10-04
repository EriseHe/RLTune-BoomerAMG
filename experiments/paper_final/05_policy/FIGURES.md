# Module 05 academic figures

The plotting entry point follows the repository's numbered experiment layout:

- `experiments/paper_final/plot_05_policy.py`: command-line entry, exports, gallery and archive.
- `experiments/paper_final/module05_figures/data.py`: raw-data verification, repeat averaging, paired cases and oracle selection.
- `experiments/paper_final/module05_figures/panels.py`: scientific figure designs.
- `experiments/paper_final/module05_figures/style.py`: manuscript style and PDF/SVG/PNG exports.
- `experiments/paper_final/test_05_figures.py`: checks against misleading statistical or trajectory transformations.

The figure code reads saved JSON and NumPy arrays; it does not import HYPRE,
MPI, or the experiment runner, and does not execute or train a solver.

## Reproduction

From the repository root, using the existing `rl` environment:

```sh
/Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python -m unittest experiments.paper_final.test_05_figures
/Users/erisehe/Library/Science/miniforge3/envs/rl/bin/python -m experiments.paper_final.plot_05_policy
```

Optional arguments are `--result-dir`, `--output`, `--dpi` (default 600) and
`--main-only` (skip most individual-seed supplements). Plotting depends only on
Python, NumPy and Matplotlib; the environment used here has NumPy 2.2.6 and
Matplotlib 3.10.9.

Default output:
`results/paper_final/05_policy/20260927_diffusion60_6seeds_100cases/analysis/paper_figures/`.

- `index.html`: local, offline gallery with main/all-seed filters and export links.
- `main_figure_atlas.pdf`: all main figures, one per page.
- Individual `.pdf`, `.svg`, 600-dpi `.png`, and smaller `_preview.png` files.
- `CAPTIONS.md`: full manuscript-caption drafts and interpretation notes.
- `figure_data.json`: per-case timing data, oracle choices and shared ordering.
- `figure_arrays.npz`: exact plotted first-execution action/residual/time matrices;
  stored axes are `[problem rank, AMG cycle]` and heatmaps display the transpose,
  so problems run horizontally and cycles increase upward. Stored rows follow
  `case_order_0_based`, missing cycles are NaN. Fixed-grid arrays use
  axes `[training seed, original case ID, weight-grid index]`.
- `provenance.json`: input/log hashes, code hashes and validation status.
- `figure_pack.zip`: figures, captions, plotted numerical data and plotting-source copies.

## Choose figures by the question

| Figure | Question | Suggested placement |
|---|---|---|
| 01 Policy savings | Which policies reduce cumulative solve time, and how variable are the six checkpoints? | Main performance result |
| 02 Paired RL advantage | Does RL beat each constant or schedule comparator? | Main comparison or alternative to 01 |
| 03 Matched action heatmaps | How do the actual RL weights differ from constants and prescribed schedules? | Main mechanism figure |
| 04 Matched residual heatmaps | How do those policies contract the residual over the same problems? | Companion to 03 |
| 05 RL seed atlases, two sources | Do different training seeds learn different action patterns? | Supplement |
| 06 Casewise gain heatmaps | Which problems and checkpoints favor each method? | Robustness view |
| 07 Paired-cost scatter | Are individual paired solves faster, and at what cost scale? | Alternative robustness view |
| 08 Fixed-weight scan | What did the complete constant-weight search find? | Main baseline validation or supplement |
| 09 Convergence examples | How do individual residual trajectories relate to the action sequence? | Main numerical illustration |
| 10 Overhead | How much does charging controller work change the result? | Practical-cost result |
| 11 Native references | How do tested native smoothers compare after setup is charged? | Conventional baseline panel |
| 12 Paired-gain ECDF | What is the distribution of per-problem advantages and losses? | Alternative to 06/07 |

The full set also exports matched action/residual heatmaps and convergence
examples for **every seed and both hierarchy sources** under `all_seeds/`.
The default main example uses seed 1 because it is the lowest numbered seed,
not because it wins. Seed 1's prior development use must remain disclosed in
the manuscript. Figures for the other five checkpoints are available without
rerunning an experiment.

## Scientific conventions

1. Main performance figures use native solve time plus recovery setup, excluding
   common primary setup and controller overhead. Figure 10 explicitly changes
   the overhead accounting. Figure 11 includes native setup because smoother
   preparation is not common. All unsuccessful primary attempts remain charged.
2. Cumulative reductions are ratios of summed costs **within each seed**. Seed
   means/SDs are then calculated. Cellwise reduction maps and ECDFs use ratios
   per case; averaging those percentages is not the cumulative estimator.
3. The same 100 test inputs appear under every checkpoint and source. Six hundred
   checkpoint/problem pairs are not six hundred independent matrices. Whiskers
   are sample SDs, not confidence intervals. No bootstrap or equivalence claim
   is added by plotting.
4. Timing repetitions are averaged within a case, so the repeated subset is not
   overweighted. Heatmaps use the actual first execution; averaging discrete
   action sequences or selecting the fastest repetition would be misleading.
5. Trajectory heatmaps follow Module 04's orientation: problem rank horizontally,
   AMG cycle vertically increasing upward (cycles numbered from 1). Problem
   columns share one order based only on mean `w=1` cycle count over all
   checkpoints and both sources, ties by original case ID. Colors and cycle axes
   are identical across corresponding panels. Gray means no executed cycle;
   nothing is zero-padded or continued beyond termination.
6. Convergence examples are chosen at the nearest ranks to 10%, 50% and 90% of
   that baseline-difficulty order, without using RL's win/loss. They illustrate
   trajectories and do not establish population-level superiority.
7. Hindsight fixed policies are visibly marked `*`. They minimize costs over the
   tested 41-weight grid, not over a continuous action space. Both schedules
   were selected on the separate development cases.
8. The scalar action heatmaps show what was executed; a complicated pattern is
   not proof that feedback is necessary. The close prefix–tail comparison and
   the weaker transfer to setup-only hierarchies remain visible.

## Schedule and oracle interpretation

The selected periodic schedule repeats `(2.5, 1)` until convergence in every
seed/source group. The selected prefix–tail schedules alternate `1` and `3`
for the first eight cycles, then hold `1.5` for every remaining cycle. Joint
seeds 1 and 4 start with `3`; the other groups start with `1`. Both schedules
were selected on development inputs and execute without current-solve feedback.

The dashed line in each prefix–tail action/residual panel marks the boundary
after cycle 8, not termination. In the main seed-1 Joint example, the final
yellow stripe is cycle 7, cycle 8 uses weight 1, and cycles 9 onward use 1.5.
The recorded solves finish in 9–14 cycles; gray begins only after convergence.

`Per-problem fixed*` means the best-observed constant weight for each test
problem, selected in hindsight from the 41-weight grid using the prescribed
timing averages. Its weight remains constant throughout that solve. It is a
per-problem fixed-weight oracle diagnostic, not a per-cycle dynamic oracle or
a proven continuous-weight optimum. `Best fixed*` instead chooses one constant
for all 100 test problems within a checkpoint/hierarchy-source group.

Plots use two-column manuscript widths, serif fonts, embedded TrueType text in
PDF, editable SVG text, and consistent method colors. Heatmaps and dense scatter
marks are rasterized where appropriate; other geometry and text are vector.
