# Reproduction index

This index connects the current paper experiments to their protocols, runners,
analysis, and required artifacts. Commands run from the repository root in the
activated `rl` environment. Build the project-owned native interfaces first as
shown in the [README](../README.md).

## Accepted studies and compact evidence

| Evidence | Protocol and compact numerical record | Runner / reporting |
|---|---|---|
| Module 04: online comparison and cost breakdown for 40³/60³/80³ diffusion and diffusion–advection | [September 20 design](../experiments/paper_final/04_online/20260920_formal/README.md), [single-seed suite](../experiments/paper_final/04_online/20260920_formal/suite.json), [six-seed aggregate](../experiments/paper_final/reproduction/online/six_seeds.json) | `run_04_online`, `analyze_04_online`; joint runner generates per-group plots |
| Module 05 / Run 05: matched-hierarchy frozen policies and periodic smoothing comparison | [Run 05 record](../experiments/paper_final/05_policy/RUN05.md), [captured protocol](../experiments/paper_final/reproduction/matched_policy/protocol.json), [weighted-minimax derivation](theory/period_two_weighted_minimax_20260928.md) | `run_05_policy_minimax`, `analyze_05_policy_minimax`, `plot_05_policy_minimax` |

Compact records describe completed measurements. They do not replace raw logs
for recomputing statistics, or frozen checkpoints for native retiming. No public
download location for those large artifacts is configured in this repository.

## Module 04: validate, run, and analyze

Validate the tracked single-seed suite without executing solves:

```bash
conda activate rl
python -m experiments.paper_final.run_04_online \
  --suite experiments/paper_final/04_online/20260920_formal/suite.json \
  --validate-only
```

Run into a new directory, preserving completed results:

```bash
python -m experiments.paper_final.run_04_online \
  --suite experiments/paper_final/04_online/20260920_formal/suite.json \
  --output-root results/paper_final/04_online/reproduction_single_seed \
  --run
python -m experiments.paper_final.analyze_04_online \
  --suite experiments/paper_final/04_online/20260920_formal/suite.json \
  --output-root results/paper_final/04_online/reproduction_single_seed
```

This launches six groups: two families × three grids, each with 5000 inputs
and three methods. It is a substantial native experiment. The runner refuses to
restart an incomplete group automatically. Frozen JSON specifies seeds, stream
hashes, candidate settings, activation, tolerance, and cycle cap.

The later six-seed evidence has its original configurations preserved under
[online reproduction data](../experiments/paper_final/reproduction/online/):

| Global training seeds | Tracked configuration directory | Original output batch under `results/paper_final/04_online/` |
|---|---|---|
| 1–3 | `experiments/paper_final/reproduction/online/seeds_1_to_3/` | `20260925_macmini_module04_3seeds/seed_N/` |
| 4–6 | `experiments/paper_final/reproduction/online/seeds_4_to_6/` | `20260926_macmini_module04_seeds4to6/seed_N/` |

Pass a captured `suite_seed_N.json` through the same `--suite` argument and use
a new output root. The runner's local replicate labels remain 1–3; global labels
distinguish the two batches. The six-seed aggregate covers 36 PDE/grid/seed runs.
Those records came from concurrent single-thread workers; retain that provenance
when comparing timings.

The September 27 diffusion-60 repeat is a separate sensitivity run. Its
`preferred_results.json` does not replace the original six-seed evidence.
Run 05 uses a refreshed seed-4 checkpoint from that follow-up, as its protocol
records.

Completed-run analysis requires the raw per-method trajectories and matching
suite. It writes `analysis/modules_1_2.md` and `modules_1_2.json`. Per-group
stream manifests, summaries, final bandit states, and solve checkpoints remain
under their original run folders. The historical combined aggregate was stored
under `20260926_macmini_module04_overnight/analysis/six_seeds.json`.

## Module 05 / Run 05: artifacts and verification

The accepted corrected run is
`results/paper_final/05_policy/20260929_run05_minimax285_joint_6seeds_100cases/`.
Its local release is
`results/paper_final/05_policy/releases/module05_run05_minimax285_diffusion60_six_seeds_20260929/`
with a neighboring `.zip` archive. These large generated artifacts are not
included in a fresh clone by default.

The full release contains final measurements, retained fixed-grid evidence,
six frozen setup/controller checkpoints, test inputs, selected fixed weights,
captured source, tables, figures, and a standalone verifier. From an extracted
release, verify coverage, hashes, costs, and fixed-grid choices:

```bash
python reproduction/verify_archive.py
```

This verifier uses the standard library and does not run HYPRE. The repository
counterpart accepts the extracted archive directory:

```bash
python -m experiments.paper_final.verify_05_policy_minimax --root EXTRACTED_RELEASE
```

To rebuild figures from an existing completed run with its `analysis/summary.json`
and `trace_arrays.npz`, without fresh solves:

```bash
python -m experiments.archive.paper_development.plot_05_policy_minimax --output COMPLETED_RUN
```

The runner calls the `analyze(output)` helper in `analyze_05_policy_minimax`
automatically after native evaluation. That helper rebuilds summaries from the
completed raw logs; it currently has no separate command-line entry point.

The historical native retiming commands are:

```bash
python -m experiments.archive.paper_development.run_05_policy_minimax prepare --output NEW_RUN
python -m experiments.archive.paper_development.run_05_policy_minimax run --output NEW_RUN
```

Preparation depends on the completed parent
`20260927_run04_anchored26_joint_6seeds_100cases` tree and unchanged captured
execution-source hashes. Refactored source will fail that historical identity
check. Use the captured source snapshot for an exact historical rerun and retain
original manifests. A new experiment on cleaned source must record its own
provenance and must not replace accepted measurements.

## Timing, recovery, and interpretation

- Online native cost includes attempted setup and native solve/recovery work.
  Total online cost additionally includes recurring learner/controller work.
  One-time initialization, AOT preparation, matrix/RHS assembly, and output I/O
  are outside the declared component scope; method wall time is separate.
- The accepted protocol uses tolerance `1e-6`, cap 50 per solve attempt, and the
  18/18/9 smoother profile. Construction failure allows at most three learned
  attempts; solve nonconvergence invokes default recovery directly. There is
  at most one default attempt. All work remains charged; final unrecovered
  cases roll back provisional learner observations.
- Run 05 native continuation excludes common initial setup and controller
  dispatch, but includes recovery setup/solve. Controller-inclusive continuation
  adds controller cost. Average prescribed repetitions within case before
  forming ratios of summed costs.
- Fixed comparators are successful best-observed choices on the 41-weight grid,
  followed by retiming. The six checkpoints share 100 test inputs examined in
  earlier work. The periodic pair comes from a smoothing surrogate; its
  multilevel runtime is evaluated experimentally.

Development studies, older periodic baselines, the withdrawn penalized-feedback
trial, and separate solve-specific checkpoint study retain their original
identifiers. See the [paper experiment index](../experiments/paper_final/README.md)
and [historical progress record](archive/paper_final_progress_20260929.md).
