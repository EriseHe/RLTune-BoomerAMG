# Setup Phase

Setup-only learning and evaluation for BoomerAMG. These experiments choose one
AMG setup configuration per problem instance and use the default or a fixed
solve policy.

## Layout

- `learners/`: contextual bandits grouped into `linucb/`, `bayesian/`,
  `thompson/`, and `tsallis/` families. These modules own the implementations
  and their checkpoint formats.
- `registry.py`: typed learner selection and construction shared by setup-only
  and joint experiments.
- `scripts/`: setup-only experiment and benchmark entry points.
- `utils/`: setup action spaces, output paths, plotting, and experiment helpers.
- `tests/`: setup-only unit and integration tests.

Shared PDE streams live in `problems/`. Historical setup plots live in
`results/archive/setup_phase/legacy_plots/`, and reference implementations live
in `docs/archive/setup_phase_reference/`.

The canonical Python package and command paths are lowercase `setup/`.
`SetupPhase/` remains only as a compatibility layer for historical imports.

## Native Solver

Both setup and solve experiments use the same unmodified fork in
`hypre/source/` and the project wrappers in `hypre/interfaces/`. Build HYPRE
and both wrappers with:

```bash
make -C hypre
```

The shared Python binding lives in `hypre/bindings/`; setup and solve code import
the same module and link to the same native installation.

## Environment

From the repository root:

```bash
conda env create -f environment.yml
```

Setup-specific Python requirements are also listed in `requirements.txt`.
