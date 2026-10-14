# RLTune-BoomerAMG

Reproducibility code for **Online Autotuning of BoomerAMG with Contextual Bandits
and Reinforcement Learning**, by Erise He, Jonathan Wang, and Lance Ding.

Shared context-action LinUCB selects BoomerAMG hierarchy parameters; recursive
LSTDQ selects relaxation weights during the solve.

## Official paper studies

The [official data registry](experiments/paper_final/reproduction/manifest.json)
is the single selection of paper data. Each module is a complete study.

| Module | Study | Local full data |
|---|---|---|
| 04 | Diffusion and advection, 40³/60³/80³, six seeds, 5,000 cases per seed and problem, four methods | `data/official/module04/` |
| 05 | Six diffusion-60³ checkpoints, 100 independent cases, 41 fixed weights, five policies, three evaluation repetitions | `data/official/module05/` |
| 06 | Type 2 variable-flow 60³, six seeds, 5,000 cases per seed, four cold branches, β=2 | `data/official/module06/` |

Module 06 uses aggressive levels 0–2 with conditional aggressive interpolation
4/6/7. The archived expanded-depth configurations are outside the official set.

Compact configurations, summaries, audit records, and current paper figures are
in [experiments/paper_final/reproduction/](experiments/paper_final/reproduction/).
The full local data includes original inputs, trajectories, checkpoints, scans,
retiming records, and the frozen execution runtime. Large local data and the
archive are excluded from Git.

## Verify the data

With Python 3.10 or later, verify the compact publication bundle and its source:

```sh
python tools/verify_official_data.py
```

When full local data is available, also verify every recorded file:

```sh
python tools/verify_official_data.py --full
```

Both commands only read saved files. See the [data and reproduction guide](docs/reproduction.md)
for coverage, timing definitions, and paper-generation paths.

## Install and build

Use Python 3.10, CMake, Make, a C/C++ compiler, and MPI with `mpicc` and `mpicxx`.
From the repository root, install into an activated environment:

```sh
python -m pip install -e '.[artifacts]'
make -C hypre JOBS=4
```

The Python dependencies include NumPy, SciPy, mpi4py, Matplotlib, and mpmath.
The supplied `environment.yml` describes the research environment. Native build
outputs belong in `hypre/build/`, `hypre/install/`, and `hypre/interfaces/`.
The recorded official native libraries are retained in `data/official/runtime/`.

## Organization

Use [the repository layout](docs/repository_layout.md) for source responsibilities.
All historical experiment data, exploratory configurations, old publication
bundles, and unused figure exports are stored locally under `data/archive/`.
Paper readers use the explicit official registry and do not search the archive.
New experiments should write to `results/` and require an explicit selection
before becoming official data.

## Licensing

A license for the project-owned code has not yet been selected. HYPRE retains
its [copyright](hypre/source/COPYRIGHT), [license files](hypre/source/LICENSE-MIT),
and [notices](hypre/source/NOTICE).
