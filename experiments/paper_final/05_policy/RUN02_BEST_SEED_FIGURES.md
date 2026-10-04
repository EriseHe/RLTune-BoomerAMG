# Run 02: best-seed publication examples

The user requested the best of the six RL seeds for the main example figures
while Run 02 was running. The criterion was fixed before reading the completed
rerun: maximize cumulative native RL time reduction against matched `w=1`.
This is an outcome-selected illustration, not a representative-seed estimate.

For each checkpoint, average all three timing repetitions within each problem
and policy, sum over all 100 problems, and calculate
`100 * (1 - RL_native_sum / w1_native_sum)`. An exact tie uses the lower seed.
Native cost includes recovery setup and solve work. Common initial setup and
controller overhead are excluded. No problems or failed attempts are dropped.

The main action, residual, and periodic-comparison heatmaps all use this same
selected seed. Actual first-repetition traces are shown; a fastest repetition
is never selected. All six checkpoints remain in aggregate comparisons and
the seed atlas, and individual heatmaps are exported for every seed. The main
four methods and axis conventions are unchanged from the Run 02 protocol.

Run the presentation revision after the audited experiment completes:

```sh
python -m experiments.paper_final.plot_05_policy_repeat_best_seed \
  --output results/paper_final/05_policy/20260927_run02_diffusion60_joint_6seeds_100cases
```

The revised gallery is `analysis/paper_figures_best_seed/index.html`; its
`figure_pack.zip` includes PDF, SVG, 600-dpi PNG, captions, numerical data,
selection rankings, and the exact reproduction source. `provenance.json`
records the plotting revision and hashes. The original hash-frozen runner,
plotter, source manifest, data, and initial gallery remain intact. This
separate plotting entry point preserves the original experiment's audit.

The completed Run 02 selects seed 2, with 44.0251% no-overhead reduction
against matched `w=1`. This selection is recomputed from the saved summary
on every figure generation rather than hard-coded.
