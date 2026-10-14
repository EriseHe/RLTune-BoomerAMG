# RLTune-BoomerAMG

Code and data for **Online Autotuning of BoomerAMG with Contextual Bandits and
Reinforcement Learning**, by Erise He, Jonathan Wang, and Lance Ding.

| Module | Study | Data |
|---|---|---|
| 1 | Diffusion and advection at 40³, 60³, and 80³; six seeds; 5,000 cases per seed | [module1](data/module1) |
| 2 | Six frozen checkpoints; 100 evaluation cases; 41 weights; five policies | [module2](data/module2) |
| 3 | Variable-flow 60³; six seeds; 5,000 cases per seed | [module3](data/module3) |

## Reproduce the tables and figures

With Python 3.10 or later:

```sh
python -m pip install -e .
python tools/verify_data.py
python tools/reproduce.py --output output/paper
```

The output contains the paper figures and tables computed from the released
measurements. This does not require a native HYPRE build. Use a new output
folder for each invocation.

## Run the experiments

Build HYPRE and the native interface with CMake, Make, a C/C++ compiler, and MPI:

```sh
make -C hypre JOBS=4
```

[Experiment instructions](docs/experiments.md) give the commands and timing
definitions. Full local trajectories are stored under `data/official/module1`,
`module2`, and `module3` and are excluded from Git. Verify them, when available,
with `python tools/verify_data.py --raw`.

HYPRE retains its [copyright](hypre/source/COPYRIGHT),
[license](hypre/source/LICENSE-MIT), and [notices](hypre/source/NOTICE).
A license for the project-owned code has not yet been selected.
