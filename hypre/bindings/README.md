# HYPRE Python bindings

This directory contains the Python binding shared by setup learners, solve
controllers, and joint experiments. Run the commands below from the repository
root.

## What is here

- `boomeramg.py`
  - Python binding loaded by the setup-phase learner code
- `recovery.py`
  - shared one-primary/one-default-fallback protocol and typed outcomes
- `config.py`
  - shared setup-parameter environment overrides
- `Makefile`
  - delegates to the shared build in `hypre/interfaces/`
- `build_libamg_runtime.sh`
  - convenience wrapper around `make`

## Build from a fresh checkout

Requirements:

1. `mpicc` available in `PATH`
2. the shared HYPRE build created with `make -C hypre`

The canonical build command is run from the repository root:

```bash
make -C hypre
```

On macOS, the convenience wrapper and local Makefile remain available as
alternatives after building HYPRE:

```bash
bash hypre/bindings/build_libamg_runtime.sh
```

Or directly:

```bash
make -C hypre/bindings
```

The resulting library is shared by all setup experiments at:

```text
hypre/interfaces/libamg_runtime.dylib  # macOS
hypre/interfaces/libamg_runtime.so     # Linux
```

It links to the single out-of-source HYPRE installation at `hypre/install/`.
The runtime search path is `@loader_path/../install/lib` on macOS and the literal
`$ORIGIN/../install/lib` on Linux. Keep the installed HYPRE libraries alongside
the interface library in this repository layout.

The binding reports setup errors, solve errors, non-finite results,
convergence, and max-cycle nonconvergence as explicit statuses. Failed native
operations return the measured work completed before failure; no fake runtime
is synthesized by the binding.

## Timing boundaries

The prepared RL path includes its residual monitoring in the native cost:

| Returned field | Measured work |
|---|---|
| Full-solve `setup_runtime_sec` | `HYPRE_BoomerAMGSetup` |
| Full-solve `solve_runtime_sec` | `HYPRE_BoomerAMGSolve`, including its internal residual monitoring |
| RL `PrepareResult.setup_runtime_sec` | Single-cycle solver configuration, `HYPRE_BoomerAMGSetup`, and initial residual computation |
| RL step `runtime_sec` | Per-step parameter updates, one AMG cycle, external residual computation, and stopping check |

The initial residual is charged once: to preparation for RL and to solve for
the full-solve API. Thus the setup/solve component boundaries differ; both
native totals include the monitoring required by their path. A timed operation
that fails still returns the time spent before its failure was detected.

The controller consumes the returned step cost. Recovery and setup-bandit
accounting reuse these returned times; do not add a second residual timer or
the surrounding Python wall time. Controller and bandit overhead remain
separate components of the recorded online cost.

These are scoped solver timers, not whole-process elapsed time. Matrix/RHS
assembly, initial-vector reset, solver creation/destruction, initial setup
parameter application outside the shared prepare helper, binding overhead,
logging and checkpointing remain outside the native timers. Timing regression tests live in
[`solve/tests/test_amg_runtime_binding.py`](../../solve/tests/test_amg_runtime_binding.py)
and [`hypre/interfaces/tests/test_rl_timing.c`](../interfaces/tests/test_rl_timing.c).

## Convergence at the iteration limit

With positive tolerance, the prepared RL step path follows the linked HYPRE
3.0.0 `hypre_BoomerAMGSolve` semantics: success requires relative residual
strictly below `tol` **before** the iteration limit. Reaching `max_iter` produces
`HYPRE_ERROR_CONV` in the full solve even when the final residual is below the
target; the prepared path reports `SolveStatus.MAX_CYCLES` in the same case.
Its caller consequently uses the configured nonconvergence recovery protocol.

For example, with a budget of 50, first reaching the target on cycle 50 reports
`MAX_CYCLES` in both paths. Raising a diagnostic budget to 51 permits a success
on cycle 50; this does not change the experiment's prescribed budget of 50.
The step API's zero-tolerance, fixed-budget diagnostic mode is separate from
positive-tolerance convergence and is unchanged.

Regression coverage compares the actual linked full solve and prepared path
below, at and above the first target-reaching cycle, plus exact tolerance
equality and LSTDQ recovery/rollback. These cases are covered by
[`solve/tests/test_amg_runtime_binding.py`](../../solve/tests/test_amg_runtime_binding.py).

## Clean

```bash
make -C hypre/bindings clean
```

## If loading fails

On macOS, check the dynamic dependencies and loader search path:

```bash
otool -L hypre/interfaces/libamg_runtime.dylib
otool -l hypre/interfaces/libamg_runtime.dylib | rg "LC_RPATH|path"
```

You should see:

- `@rpath/libHYPRE-3.0.0.dylib`
- an `LC_RPATH` entry rooted at `@loader_path/...`

On Linux, inspect the shared-library dependencies with:

```bash
ldd hypre/interfaces/libamg_runtime.so
```
