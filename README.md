# RLTune-BoomerAMG

Reproducibility code for **Online Autotuning of BoomerAMG with Contextual Bandits
and Reinforcement Learning**, by Erise He, Jonathan Wang, and Lance Ding.

Code and frozen protocols for the SISC studies of online BoomerAMG autotuning:
shared context-action LinUCB V4 selects hierarchy parameters, and recursive
LSTDQ V3 selects relaxation weights during the solve.

This submission tree contains Module 04 online comparison and the accepted
Module 05 matched-hierarchy Run 05. It includes 42 exact online configurations,
compact accepted results, 20 MB of exact online inputs, and a verified 14 MB
Run 05 frozen-input bundle.
Original raw measurement logs and generated figure releases are not included.
The bundle is sufficient for new matched retiming without earlier result folders.

## Start here

| Task | Where to go |
|---|---|
| Install and check the code | [Install and build](#install-and-build), then [check the checkout](#check-the-checkout) below |
| Run Module 04 online autotuning | [Module 04 reproduction](docs/reproduction.md#module-04-exact-configurations) |
| Run Module 05 matched-hierarchy comparison | [Module 05 reproduction](docs/reproduction.md#module-05-verify-and-retime-the-frozen-bundle) |
| Understand or maintain the code | [Repository layout](docs/repository_layout.md), with setup, solve and native-binding details linked there |
| Inspect the prescribed smoothing rule | [Theory references](docs/theory/README.md) |

The reproduction guide links the detailed protocols, accepted results and input
provenance for each study. Use it as the starting point for experiment commands.

## Install and build

Use Python 3.10, CMake, Make, a C/C++ compiler, and MPI with `mpicc` and
`mpicxx`. From the repository root, install into an activated Python environment:

```sh
python -m pip install -e '.[dev,artifacts]'
make -C hypre JOBS=4
```

Core Python dependencies are NumPy, SciPy, mpi4py, Matplotlib and mpmath. The
`dev` extra provides Ruff; `artifacts` provides ReportLab and pypdf for PDF release
packaging. `python -m pip install -e .` installs the computational dependencies.
Conda users can create the supplied environment with `conda env create -f
environment.yml`, activate `rl`, then install the extras above.

The build writes to `hypre/build/`, `hypre/install/` and `hypre/interfaces/`.
The HYPRE implementation under `hypre/source/` is preserved unchanged; our native
and Python wiring lives in `hypre/interfaces/` and `hypre/bindings/`.

## Check the checkout

```sh
python scripts/check_repository.py --static-only
python scripts/check_repository.py --tests-only
python -m unittest solve.tests.test_amg_runtime_binding
```

The static check runs fatal source checks. The test check covers five groups:
setup, PDE problems, solve control, experiment utilities and paper studies.
Build first for native status/timing tests. The last command runs the native
binding integration tests directly. These checks do not launch the full paper
experiments. The validation record is in the
[organization report](docs/sisc_repository_cleanup.md).

Validate the formal online protocol without solving PDEs:

```sh
python -m experiments.paper_final.run_04_online --validate-only
```

Verify the accepted Run 05 input bundle without native solves:

```sh
python -c 'from experiments.paper_final.common.frozen_inputs import DEFAULT_BUNDLE, verify_bundle; print(verify_bundle(DEFAULT_BUNDLE))'
```

Use the [reproduction guide](docs/reproduction.md) to launch fresh runs. New runs
record current source and environment provenance; their measured times are
separate from the accepted September measurements.

## Development history and licensing

The full development and archive trees remain on
[`online-bandit-rl`](https://github.com/EriseHe/RLTune-BoomerAMG/tree/online-bandit-rl)
and [`cleanup/sisc-repository-20261004`](https://github.com/EriseHe/RLTune-BoomerAMG/tree/cleanup/sisc-repository-20261004).
They are outside this submission tree.

A license for the project-owned code has not yet been selected. HYPRE retains
its [copyright](hypre/source/COPYRIGHT), [license files](hypre/source/LICENSE-MIT)
and [notices](hypre/source/NOTICE).
