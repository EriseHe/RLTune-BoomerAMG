# Results

Generated experiment outputs live here, separate from executable source code.

- `by_date/`: canonical zero-copy browsing view, grouped by the day each run
  started; regenerate it with
  `python experiments/diagnostics/index_results_by_date.py`.
- `setup/`: setup-only runs.
- `solve/`: solve-only runs.
- `joint/`: setup plus solve experiments.
- `diagnostics/`: smoke tests, profiling, and protocol diagnostics.
- `archive/`: historical outputs retained for reference.
- `invalid/`: quarantined results from invalidated protocols.

Large run logs and checkpoints are ignored by default. Curated summaries may be
tracked when they are needed to reproduce a reported result.

## Curated paper-result bundles

Git tracks only the smallest useful evidence bundle inside `joint/paper_*`:

- `README.md`, `experiment_config.json`, and `stream_manifest.json` identify the
  source runner configuration and the exact shared input stream;
- `summary_*.csv`, `summary_*.md`, and `screen_report.md` preserve the compact
  numerical result and reporting windows;
- `reproduce.sh` records the supported rerun entry point;
- `figures/last_1000_runtime_breakdown.png` is the primary performance figure;
- `figures/learned_per_cycle_action_trajectories.png` preserves the most direct
  controller-behavior visualization.

Raw trajectories, candidate schedules, checkpoints, mutable learner states, and
the remaining generated plots stay in the local result directory but are not
committed. This keeps paper evidence reviewable without turning the repository
into an experiment-artifact store.

The date view uses relative symbolic links rather than moving run directories.
This keeps saved `output_dir` values and reproduction scripts valid, and it is
safe to refresh while an experiment is running.
