# HYPRE Python bindings

This directory contains the Python binding shared by setup learners, solve
controllers, and joint experiments.

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

The local convenience wrapper remains available from this directory:

```bash
bash build_libamg_runtime.sh
```

Or directly:

```bash
make
```

The resulting library is shared by all setup experiments at:

```text
hypre/interfaces/libamg_runtime.dylib
```

It links to the single out-of-source HYPRE installation at `hypre/install/`
using this runtime search path:

```text
@loader_path/../install/lib
```

as its runtime search path for `libHYPRE-3.0.0.dylib`, so the built library
remains portable across machines as long as the repository layout is preserved.

The binding reports setup errors, solve errors, non-finite results,
convergence, and max-cycle nonconvergence as explicit statuses. Failed native
operations return the measured work completed before failure; no fake runtime
is synthesized by the binding.

## Clean

```bash
make clean
```

## If loading fails

Check the dynamic dependencies:

```bash
otool -L ../../hypre/interfaces/libamg_runtime.dylib
otool -l ../../hypre/interfaces/libamg_runtime.dylib | rg "LC_RPATH|path"
```

You should see:

- `@rpath/libHYPRE-3.0.0.dylib`
- an `LC_RPATH` entry rooted at `@loader_path/...`
