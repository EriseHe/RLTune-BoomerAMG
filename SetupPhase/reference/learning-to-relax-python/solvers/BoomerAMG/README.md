# BoomerAMG Solver Module for Learning to Relax

This module provides a Python interface to HYPRE's BoomerAMG solver, adapted for the "Learning to Relax" bandit-based parameter tuning framework.

## Overview

The original "Learning to Relax" paper uses **SOR (Successive Over-Relaxation)** with the relaxation parameter `omega` tuned via a Tsallis-INF bandit algorithm. The loss function is simply the **iteration count**.

For **BoomerAMG**, we extend this framework with a more faithful cost metric: **Work Units (WU)**.

## Loss Function

### SOR Loss Function (Original)
```
Loss_SOR = iteration_count
```

### BoomerAMG Loss Function (This Implementation)
```
WU = iterations × cum_nnz_AP
```

Where:
- `iterations`: Number of V-cycles to converge
- `cum_nnz_AP`: Cumulative nonzeros ratio of the AMG hierarchy

### Bandit Loss (what Tsallis-INF consumes)

In the Python adaptation here, Tsallis-INF is fed a **log-transformed** loss derived from work units:

```
loss = 1 + log(1 + WU / 40)
```

This is the convention used by `scripts/learning_amg_with_logs.py` and keeps the magnitude of losses
well-scaled across easy vs hard instances.

### The cum_nnz_AP Metric

The `cum_nnz_AP` metric captures the **memory and computational complexity** of the AMG hierarchy:

```
                    L              L-1
                   ___            ___
                   \              \
cum_nnz_AP =       /   nnz(A_ℓ) + /   nnz(P_ℓ)
                   ---            ---
                   ℓ=0            ℓ=0
               ─────────────────────────────────
                         nnz(A_0)
```

Where:
- `L`: Number of levels in the AMG hierarchy
- `A_ℓ`: Operator matrix at level ℓ (A_0 is the original matrix)
- `P_ℓ`: Interpolation operator from level ℓ+1 to level ℓ

**Typical values:**
- 2D problems: cum_nnz_AP ≈ 1.5 - 3.5
- 3D problems: cum_nnz_AP ≈ 2.0 - 5.0

### Why Work Units?

The iteration count alone is **not a fair cost metric** for AMG because:

1. **Different hierarchies have different costs**: A threshold of 0.25 might create a 4-level hierarchy, while 0.7 creates a 6-level hierarchy
2. **Each V-cycle costs differently**: More levels and more nonzeros = more work per cycle
3. **The strong_threshold parameter affects hierarchy construction**: Lower thresholds → more aggressive coarsening → smaller hierarchies → cheaper cycles

Work Units properly account for this by multiplying:
- How many cycles you need (iterations)
- How expensive each cycle is (cum_nnz_AP)

## Tunable Parameter

| Solver | Parameter | Range | Purpose |
|--------|-----------|-------|---------|
| SOR | `omega` | (1.0, 2.0) | Relaxation factor |
| BoomerAMG | `strong_threshold` | (0.0, 1.0) | Coarsening aggressiveness |

### Strong Threshold Effect

- **Lower values (0.1-0.25)**: More connections considered "strong" → More aggressive coarsening → Fewer levels → Smaller hierarchy
- **Higher values (0.5-0.7)**: Fewer strong connections → More conservative coarsening → More levels → Larger hierarchy

**Default heuristics:**
- 2D problems: θ = 0.25
- 3D problems: θ = 0.5 - 0.6

## API

### Solver Function

```python
from solvers.BoomerAMG import boomeramg

# Returns: (iterations, cum_nnz_AP, solution)
k, complexity, x = boomeramg(A, b, x0, strong_threshold=0.25, tol=1e-8)

# Compute work units
WU = k * complexity
```

### Bandit Learner

```python
from learners import TsallisINF_AMG

# Create bandit with threshold grid
grid = np.linspace(0.1, 0.9, 9)
bandit = TsallisINF_AMG(grid, T=1000)

# Predict and update
theta = bandit.predict()
k, cum_nnz_AP, x = boomeramg(A, b, x0, theta, tol)
WU = k * cum_nnz_AP
loss = 1 + np.log1p(WU / 40.0)
bandit.update(loss)
```

## File Structure

```
solvers/BoomerAMG/
├── __init__.py          # Module exports
├── boomeramg.py         # Main solver with HYPRE ctypes bindings
├── hypre_loader.py      # HYPRE DLL loading utilities
├── threshold_opt.py     # Heuristic optimal threshold
├── threshold_grid.py    # Parameter grid generation
└── README.md            # This file

learners/
├── TsallisINF_SOR.py    # Original SOR bandit (loss = iterations)
└── TsallisINF_AMG.py    # BoomerAMG bandit (loss = log-transformed WU)

scripts/
├── learning.py          # SOR experiments
├── learning_amg_with_logs.py  # BoomerAMG reference experiment (plots + CSV)
└── run_boomeramg_bandit.py    # BoomerAMG single entrypoint (CLI)
```

## HYPRE Requirements

This module requires a shared HYPRE library (e.g. `libHYPRE.dylib` on macOS).

In this repo, the bundled `SolvePhase/hypre/src/lib/libHYPRE.dylib` is MPI-enabled, so you must
either:
- set `HYPRE_MPI_COMM_WORLD` to an initialized communicator handle (recommended via `mpi4py`), or
- rebuild HYPRE without MPI for purely sequential use.

Set environment variable:
```bash
export HYPRE_DIR=/path/to/hypre/install
# or
export HYPRE_LIBHYPRE=/path/to/libHYPRE.dll
```

## References

- Original paper: "Learning to Relax: Setting Solver Parameters Across a Sequence of Linear System Instances"
- HYPRE documentation: https://hypre.readthedocs.io/
- BoomerAMG: Parallel algebraic multigrid solver in HYPRE
