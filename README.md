# RLTune-BoomerAMG

Research code for online BoomerAMG setup tuning with contextual bandits and
per-cycle relaxation control with reinforcement learning.

## Start here

- [Reproduction index](docs/reproduction.md): accepted studies, commands,
  timing definitions, and required data.
- [Repository layout](docs/repository_layout.md): component ownership and
  dependency direction.
- [Paper experiment index](experiments/paper_final/README.md): current studies
  and separately labeled development history.
- [Joint experiment runner](experiments/joint/solve_control/README.md): JSON
  configuration and general experiment execution.

The repository contains experiment source, configurations, and compact numerical
evidence. Large trajectories, frozen checkpoints, and generated figure archives
are separate artifacts. A fresh clone does not include every historical result
folder; the reproduction index identifies what each operation requires.

## Build from a fresh checkout

Install Conda, CMake, Make, a C/C++ compiler, and MPI compiler wrappers
(`mpicc` and `mpicxx`). Confirm that the wrappers reference installed compilers
before building. From the repository root:

```bash
conda env create -f environment.yml
conda activate rl
make -C hypre
```

The environment is named `rl`. Activate it before running Python commands.
HYPRE is built out of source under `hypre/build/` and `hypre/install/`;
project-owned native interfaces are built under `hypre/interfaces/`.
The vendored HYPRE implementation in `hypre/source/` is kept unchanged.

## Validate the accepted online protocol

```bash
python -m experiments.paper_final.run_04_online \
  --suite experiments/paper_final/04_online/20260920_formal/suite.json \
  --validate-only
```

This validates six PDE/grid groups without executing solves. It is the
September 20 single-seed protocol. The later six-seed evidence has separate
captured configurations; see the [reproduction index](docs/reproduction.md).

## Failure protocol

Construction failures permit up to three learned setup attempts total, with
previously failed exact configurations excluded on the same problem. Solve
nonconvergence goes directly to default recovery. There is at most one default
attempt. All attempted work and measured recovery cost remain in reported
runtime. Under the accepted `rollback_unrecovered` protocol, provisional
learning feedback is rolled back if recovery also fails.

## Tests

Run in the activated environment after building the native interfaces:

```bash
python -m pip install -e '.[dev]'
python scripts/check_repository.py
```

Native integration checks require the MPI compiler toolchain in addition to the
Python environment. Timing depends on hardware and system load; preserve inputs,
seeds, stopping rules, and accounting when comparing runs.

The checker compiles project Python source, runs fatal static checks, and runs
each test group in a separate process. CI builds the native interfaces on Linux
before running the same checks. Development diagnostic/archive tests run when
those directories are present; required core/native/paper tests always run.

Install `.[artifacts]` when generating the experiment release PDF with the
Module 05 packager. The [cleanup report](docs/repository_cleanup_20261004.md)
records the reorganization and validation.
