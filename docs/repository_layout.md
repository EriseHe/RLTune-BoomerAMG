# Repository layout

The directory structure follows component ownership. Historical experiment
identifiers stay with their recorded protocols and artifacts.

| Directory | Responsibility |
|---|---|
| `problems/` | PDE definitions, coefficient encoders, deterministic instance streams |
| `hypre/source/` | Vendored HYPRE implementation; unchanged by project cleanup |
| `hypre/interfaces/`, `hypre/bindings/` | Project-owned native/Python wiring into HYPRE |
| `setup/` | Setup action space, contextual-bandit learners, construction helpers, tests |
| `solve/` | Solver environments, controller families, episode execution, tests |
| `experiments/` | Composition of setup and solve, study protocols, reporting, diagnostics |
| `results/` | Generated trajectories, checkpoints, figures, and curated numerical evidence |
| `docs/` | Current guides, theory notes, and labeled historical documentation |

## Dependency direction

`setup` and `solve` use shared problem definitions and project-owned HYPRE
wrappers. `experiments` combines those components into complete methods.
Algorithm implementations should not import executable experiment scripts.
Setup and solve experiment entry points should not import one another.
Code that composes both phases belongs under `experiments/`.

`setup.registry` and `solve.registry` contain construction helpers used by
experiments. Implementations live under `setup.learners` and
`solve.controllers`. The `SetupPhase` and `SolvePhase` namespaces provide
historical import compatibility and do not own implementations.

Native changes belong in the project-owned interface layer when needed for
wiring. Build outputs under `hypre/build/` and `hypre/install/` are generated,
rather than source files to commit.

## Experiments and artifacts

Use package-qualified imports and run entry points from the repository root with
`python -m package.module`. Shared study utilities belong in named modules that
entry points and reporting code can both import. Importing a utility should not
start a run or set process/thread configuration.

Numbered `experiments/paper_final/` stage directories hold protocols and source
documentation. Outputs belong under `results/paper_final/<stage>/<run>/`.
Recorded run names and captured manifests are provenance identifiers; keep them
when moving reusable source into clearer modules.

The [reproduction index](reproduction.md) lists accepted evidence separately from
development runs. Large logs/checkpoints are ignored by default. A reference to
a local artifact path is not a promise that a fresh clone includes that artifact.

## Documentation

- [Root README](../README.md): build, entry points, and tests.
- [Reproduction index](reproduction.md): current evidence and artifact requirements.
- [Paper experiment index](../experiments/paper_final/README.md): stage navigation.
- [Theory index](theory/README.md): derivations and deterministic checks.
- `docs/archive/` and `experiments/archive/`: historical/reference material,
  outside the active execution path.
