# Solve Phase

Solve-phase policy code and solve-only tests belong here. Native project
wrappers live in `hypre/interfaces/`; no project Python or interface code lives
inside the HYPRE fork.

Build the unchanged fork and the shared runtime with `make -C hypre`.
All setup and solve environments use `hypre/interfaces/libamg_runtime.dylib`
through `hypre.bindings`.

## Layout

- `algorithms/`: PPO, SARSA, and shared-action LCB algorithm packages.
- `core/`: solver environments and algorithm-independent outcome handling.
- `scripts/`: solve-only training, evaluation, and smoke entry points.
- `tests/`: solve-only tests.

PDE definitions and deterministic instance streams live in the shared
`problems/` package. Workflows that also use a setup learner belong in
`experiments/`, not in this directory.
