# Setup-Phase solver wrapper

This directory contains the thin C/Python wrapper used by the setup-phase
bandit code to call BoomerAMG directly.

## What is here

- `amg_setup_solver.c`
  - C wrapper that builds the matrix, applies setup parameters, and calls HYPRE
- `boomeramg.py`
  - Python binding loaded by the setup-phase learner code
- `Makefile`
  - portable repository-relative build for `libamg_setup_solver.dylib`
- `build_libamg_setup_solver.sh`
  - convenience wrapper around `make`

## Build from a fresh checkout

Requirements:

1. `mpicc` available in `PATH`
2. the local HYPRE build present under:
   - `SolvePhase/hypre/src/hypre/include`
   - `SolvePhase/hypre/src/hypre/lib`

From this directory:

```bash
bash build_libamg_setup_solver.sh
```

Or directly:

```bash
make
```

The resulting library is:

```text
libamg_setup_solver.dylib
```

The build uses:

```text
@loader_path/../../../SolvePhase/hypre/src/hypre/lib
```

as its runtime search path for `libHYPRE-3.0.0.dylib`, so the built library
remains portable across machines as long as the repository layout is preserved.

## Clean

```bash
make clean
```

## If loading fails

Check the dynamic dependencies:

```bash
otool -L libamg_setup_solver.dylib
otool -l libamg_setup_solver.dylib | rg "LC_RPATH|path"
```

You should see:

- `@rpath/libHYPRE-3.0.0.dylib`
- an `LC_RPATH` entry rooted at `@loader_path/...`
