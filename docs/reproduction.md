# Official data and reproduction

The [registry](../experiments/paper_final/reproduction/manifest.json) selects
exactly Modules 04, 05, and 06. The compact publication bundle uses stable module
names; the full local data lives under `data/official/`.

## Module 04

`data/official/module04/` contains six problem/grid combinations, each with six
seeds and 5,000 paired inputs. Four independent cold methods execute every input:
Default, fixed-setup LSTDQ, setup-only LinUCB, and joint LinUCB–LSTDQ.
There are 180,000 paired inputs and 720,000 method-cases.

`campaign.json` explicitly lists the twelve three-seed components. `summary.json`
and the corresponding CSV files provide the complete six-seed results. Each
component retains its exact inputs, configurations, source identities, cold-start
audits, trajectories, checkpoints, and figure arrays. Full and final-1,000-case
windows are included. Read the campaign entries; do not glob for run directories.

## Module 05

`data/official/module05/` contains the six frozen joint checkpoints from official
Module 04 diffusion 60³. Each checkpoint is evaluated on 100 independent cases.
Fresh scans cover all 41 weights from 1 to 3 in steps of 0.05; global and per-case
choices are frozen before independent evaluation. Five policy roles are retimed
with three repetitions. Identical execution cells are shared by the prescribed
protocol: 29,520 scan trials and 8,535 distinct evaluation trials are recorded.

The study contains `inputs.json`, `checkpoints/`, `selections.json`, `raw/scan/`,
`raw/test/`, `analysis/`, and `paper_figures/`. Scan times are not evaluation times.
All six checkpoints remain in the results.

## Module 06

`data/official/module06/` contains the Type 2 variable-flow 60³ study: six seeds,
5,000 cases each, and four cold branches learning from case 1, with β=2 and 1,024
setup candidates. Aggressive levels are 0–2 with conditional interpolation 4/6/7.
Cycle weights are 1–3; the attempt cap is 50 and tolerance is 10⁻⁶. The recorded
cap-first stopping rule is retained. Failure Model and adjacent recovery are used.

There are 120,000 method-cases. `final_verification.json`, `fairness_audit.json`,
and `phase_breakdown.json` accompany the full and final-1,000-case summaries.

## Timing and recovery

For Modules 04 and 06, native time includes every incurred setup and solve attempt,
retry, and Default fallback. Inclusive time adds learner work. Matrix assembly and
file I/O are outside these solver metrics. Failed cases remain in full totals;
Module 06 also reports comparisons on cases where all methods succeed.

Module 05 compares continuation costs on matched hierarchies. The shared primary
setup is excluded; recovery setup and solve costs are included. Inclusive
continuation time adds controller work. Its cost boundary differs from online totals.

## Verification and portability

```sh
python tools/verify_official_data.py
python tools/verify_official_data.py --full
```

The first command verifies the compact bundle, complete seed/method coverage,
summary arithmetic, original Module 06 configuration, and accepted execution
source. The second also hashes all full local data against `raw_manifest.json`.
Neither command runs native solves or modifies learned state.

Measured records are retained byte for byte. Absolute locations inside captured
metadata describe their execution provenance. `tools/official_data.py` translates
those recorded paths to current official locations using `source_paths.json`.
The resolver rejects archived or unregistered data and never substitutes a nearby
run when a file is missing.

The numerical source files at the repository root match the official source
manifest. `data/official/runtime/` retains the frozen runtime and native binaries
used for the measurements. New runs must use new output directories and record
fresh environment, source, input, and native identities. Saved completed runs
must never be resumed or overwritten.

## Paper assets

The compact bundle contains the current Module 04 and Module 05 SVG/PNG figures.
The paper-generation scripts are under `tools/paper/` and in the local
SISC-Overleaf repository. They read the official registry and verify the saved
records. Module 04's illustrative Default residual replay is retained under
`module04/figure_evidence/`. Module 06 has no paper figure.

To regenerate into a separate paper workspace, provide explicit paths:

```sh
python tools/paper/prepare_module4_current.py --code "$PWD" --paper /path/to/paper
python tools/paper/prepare_module5_current.py --run "$PWD/data/official/module05" --runtime "$PWD/data/official/runtime" --paper /path/to/paper
```

These publishers require full local data and the numerical Python environment.
They generate figures and table material from saved records; they do not rerun
experiments. The compact bundle supports verification and inspection without the
large local raw data.

## Archive

`data/archive/20261014/` holds exploratory and unselected experiments, historical
publication bundles, unused exports, and earlier source/configuration snapshots.
It is excluded from Git and from the official registry. Legacy filesystem aliases
outside the official tree support old provenance references; no official data
folder is a symlink into the archive.
