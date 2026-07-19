# Solve Phase

Solve-phase policy code and solve-only tests belong here. Native project
wrappers live in `hypre/interfaces/`; no project Python or interface code lives
inside the HYPRE fork.

Build the unchanged fork and both shared interfaces with `make -C hypre`.
Solve environments load `hypre/interfaces/libamg_env.dylib` by default.

## Layout

- `core/`: solve environments, policies, and online controllers.
- `scripts/`: solve-only training, evaluation, and smoke entry points.
- `tests/`: solve-only tests.

PDE definitions and deterministic instance streams live in the shared
`problems/` package. Workflows that also use a setup learner belong in
`experiments/`, not in this directory.
