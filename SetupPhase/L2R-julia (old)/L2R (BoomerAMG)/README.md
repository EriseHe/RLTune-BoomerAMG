# L2R (Multigrid) - Learning to Relax with AMG

Extension of the Learning to Relax framework to Algebraic Multigrid (AMG).

## Design Principle

**Reuse everything from `LearningToRelax`, only swap the solver.**

| Component | Source | Modified? |
|-----------|--------|-----------|
| `TsallisINF` | `LearningToRelax` | ❌ No - imported directly |
| `truncated_normal` | `LearningToRelax` | ❌ No - imported directly |
| `create_laplacian_2d` | `LearningToRelax` | ❌ No - imported directly |
| `amg_iterations` | **New** | ✅ Yes - replaces `sor` |

## Parameter Mapping

| SOR (Original) | AMG (This Extension) |
|----------------|----------------------|
| ω (relaxation parameter) | `relax_wt` (relaxation weight) |
| Range: [1.0, 2.0) | Range: [0.0, 1.0] |
| Optimal: problem-dependent | Optimal: typically 0.5-1.0 |

## Structure

```
L2R (Multigrid)/
├── src/
│   └── amg_solver.jl       # AMG iteration counter (HYPRE wrapper)
├── scripts/
│   └── learning_amg.jl     # Standard bandit experiment with AMG
└── plots/                  # Output figures
```

## Usage

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Run learning experiment with AMG
include("scripts/learning_amg.jl")
```

## Key Difference from SOR

The cost function changes from:
```julia
# SOR: iterations to reduce residual by tol
cost = sor(A, b, x0, ω, tol)
```

To:
```julia
# AMG: iterations to reduce residual by tol
cost = amg_iterations(A, b, relax_wt, tol)
```

The bandit algorithm (Tsallis-INF) remains **identical** - it only sees:
- A grid of parameter values to choose from
- A cost (iteration count) for each choice

