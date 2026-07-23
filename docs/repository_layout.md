# Repository Layout

The repository is divided by ownership rather than by experiment history.

```text
problems/     shared PDE definitions and deterministic instance streams
hypre/        unmodified HYPRE source, build/install trees, and native wrappers
setup/        setup-only learners and entry points
solve/        solve-only controllers, environments, and entry points
experiments/  workflows that compose setup and solve components
results/      generated data, checkpoints, tables, and figures
docs/         active documentation and archived research notes
```

## Dependency Direction

```text
problems ───────────────┐
hypre/bindings ─────────┼──> setup
                       ├──> solve
setup + solve ──────────┴──> experiments
experiments ───────────────> results
```

`setup` and `solve` must not import each other's experiment scripts.
Code that needs both belongs in `experiments/`. Generated artifacts never
belong beside source files.

The two phase directories share lifecycle folders such as `scripts/` and
`tests/`, but their algorithm-specific internals need not be identical. Setup
owns contextual-bandit learners grouped by family. Solve owns PPO, SARSA, and
LCB algorithm packages plus the solver environment adapters used by them.

## Naming

The canonical Python packages are lowercase: `setup` and `solve`. Their public
construction APIs are `setup.registry` and `solve.registry`; algorithm
implementations remain organized under `setup.learners` and
`solve.controllers`. The former `SetupPhase` and `SolvePhase` names are
compatibility namespaces only and contain no implementation.
