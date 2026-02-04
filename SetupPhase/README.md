# SetupPhase

This folder contains:
- A copy of the **Learning to Relax** paper (`Learning to Relax.md`)
- The authors’ original MATLAB repo (`learning-to-relax-original/`)
- A Python translation + BoomerAMG adaptation (`learning-to-relax-python/`)

This README documents the **exact commands** I used to run the experiments you asked for (BoomerAMG + a small SOR run), without modifying your solver/learner implementations.

## Recommended “one clean entrypoint” (BoomerAMG)

Use `SetupPhase/learning-to-relax-python/scripts/run_boomeramg_bandit.py` for BoomerAMG experiments. It supports:
- `--variance high|low|both`
- `--log` to write CSV
- `--force-first-high/--force-first-low` to force a bad first θ

Examples:

```bash
python SetupPhase/learning-to-relax-python/scripts/run_boomeramg_bandit.py --T 5000 --trials 1 --variance both
python SetupPhase/learning-to-relax-python/scripts/run_boomeramg_bandit.py --T 5000 --trials 1 --variance both --log
python SetupPhase/learning-to-relax-python/scripts/run_boomeramg_bandit.py --T 5000 --trials 1 --variance both --force-first-high 0.1 --force-first-low 0.7
```

## Environment notes (important)

### BoomerAMG needs a loadable `libHYPRE`
Your Python BoomerAMG wrapper (`learning-to-relax-python/solvers/BoomerAMG/boomeramg.py`) loads HYPRE via:
- `HYPRE_LIBHYPRE` (recommended): full path to `libHYPRE.*`
- or `HYPRE_DIR` / `HYPRE_PREFIX`

In this repo, an existing macOS dylib is at:
- `SolvePhase/hypre/src/lib/libHYPRE.dylib`

### MPI-enabled HYPRE: pass an initialized `MPI_COMM_WORLD`
The provided `libHYPRE.dylib` is linked against OpenMPI (MPI-enabled). To avoid “MPI_Comm_size called before MPI_INIT”, I ran BoomerAMG experiments by:
1) importing `mpi4py` (which initializes MPI)
2) setting `HYPRE_MPI_COMM_WORLD` to `MPI.COMM_WORLD.handle`

This avoids code changes in `boomeramg.py`.

## What to run (entry points)

In the original Python translation, most experiments are standalone scripts under
`SetupPhase/learning-to-relax-python/scripts/`.

- **BoomerAMG (setup-phase bandit):**
  - Reference script: `SetupPhase/learning-to-relax-python/scripts/learning_amg_with_logs.py`
  - Recommended single entrypoint: `SetupPhase/learning-to-relax-python/scripts/run_boomeramg_bandit.py`
- **SOR (original paper translation):**
  - Learning curve: `SetupPhase/learning-to-relax-python/scripts/learning.py`
  - “Run everything” helper (SOR-only): `SetupPhase/learning-to-relax-python/experiments/run_all.py`

## Commands I ran (BoomerAMG, T=5000, trial=1)

All commands below assume you start at repo root:
`/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG`

### 1) BoomerAMG Tsallis-INF (high + low variance)

```bash
python SetupPhase/learning-to-relax-python/scripts/run_boomeramg_bandit.py --T 5000 --trials 1 --variance both
```

### 2) BoomerAMG with a forced “bad” first action

You asked to force:
- high variance: first θ = 0.10
- low variance: first θ = 0.70

```bash
python SetupPhase/learning-to-relax-python/scripts/run_boomeramg_bandit.py --T 5000 --trials 1 --variance both --force-first-high 0.10 --force-first-low 0.70
```

### Outputs

Plots (and optional CSV logs) are written under:
`SetupPhase/learning-to-relax-python/plots/BoomerAMG/`
