# Repository Layout

The repository is divided by ownership rather than by experiment history.

```text
problems/     shared PDE definitions and deterministic instance streams
hypre/        unmodified HYPRE source, build/install trees, and native wrappers
SetupPhase/   setup-only learners and entry points
SolvePhase/   solve-only controllers, environments, and entry points
experiments/  workflows that compose setup and solve components
results/      generated data, checkpoints, tables, and figures
docs/         active documentation and archived research notes
```

## Dependency Direction

```text
problems ───────────────┐
hypre/bindings ─────────┼──> SetupPhase
                       ├──> SolvePhase
SetupPhase + SolvePhase ┴──> experiments
experiments ───────────────> results
```

`SetupPhase` and `SolvePhase` must not import each other's experiment scripts.
Code that needs both belongs in `experiments/`. Generated artifacts never
belong beside source files.

The two phase directories share lifecycle folders such as `scripts/` and
`tests/`, but their algorithm-specific internals need not be identical. Setup
owns contextual-bandit learners grouped by family. Solve owns PPO, SARSA, and
LCB algorithm packages plus the solver environment adapters used by them.

## Naming

`SetupPhase` and `SolvePhase` are clear research-area names, although lowercase
Python package names are more conventional. If this repository later becomes
an installable package, the natural package-level names are `rltune.setup` and
`rltune.solve`. That naming-only migration should be kept separate from
behavioral or experiment changes.
