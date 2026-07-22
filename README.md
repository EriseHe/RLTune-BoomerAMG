# RLTune-BoomerAMG

Research code for online setup tuning and per-cycle solve control in HYPRE
BoomerAMG.

## Repository layout

- `hypre/source/`: unmodified shared HYPRE fork.
- `hypre/interfaces/`: the single project-owned native runtime.
- `hypre/bindings/`: Python binding and shared failure recovery protocol.
- `hypre/build/`, `hypre/install/`: ignored out-of-source build products.
- `problems/`: PDE definitions and deterministic instance streams.
- `SetupPhase/`: contextual-bandit algorithms and setup-only experiments.
- `SolvePhase/`: solve controllers and solve-only tests.
- `experiments/`: workflows that combine setup and solve learning.
- `experiments/archive/`: historical protocols excluded from active runs.
- `docs/`: design and implementation notes.
- `results/`: generated experiment outputs and reproducibility records.

Setup and solve code both use `hypre.bindings` and the same
`hypre/interfaces/libamg_runtime` library. No project interface or experiment
code is built inside the HYPRE source fork.

## Build

Create the Python environment:

```bash
conda env create -f environment.yml
```

Build the unchanged HYPRE fork out of source, install it under `hypre/install/`,
and build the shared runtime:

```bash
make -C hypre
```

## Learning architecture

- Setup phase: Shared LinUCB v4 selects one Tune7 setup action per instance.
- Solve phase: PPO, SARSA, and shared-action linear LCB controllers may select
  a relaxation action at each AMG cycle.
- Joint experiments: each method owns an independently updating LinUCB branch
  after any configured common warmup snapshot.

The active Exp44 protocol and commands are documented in
`experiments/joint/exp44/README.md`.

## Failure protocol

Every external instance starts with one learned setup attempt. A setup
construction failure may trigger same-context reselection, up to three learned
setup attempts in total. A solve failure, non-finite evaluation, max-cycle
nonconvergence, or exhaustion of the learned setup attempts triggers one
default setup + default solve fallback from a zero initial solution.

- A recovered failure is charged its measured primary and fallback time.
- LinUCB commits at most one transaction per external instance; a recovered
  setup-reselection transaction may contain multiple measured observations.
- A solve controller commits at most one episode per external instance.
- If fallback also fails, pending learning updates are rolled back and the
  instance is recorded as unrecovered.
- No retry loops, artificial failure penalties, fake runtimes, or residual
  potential shaping are used by active experiments.

Historical retry, shaping, and old online-Gym workflows are retained only under
`experiments/archive/`.

## Tests

The main active test groups are:

```bash
python -m unittest discover -s SetupPhase/tests -p 'test_*.py' -v
python -m unittest discover -s SolvePhase/tests -p 'test_*.py' -v
python -m unittest discover -s experiments/diagnostics/solve_control -p 'test_*.py' -v
```

Repository ownership and dependency rules are documented in
`docs/repository_layout.md`.
