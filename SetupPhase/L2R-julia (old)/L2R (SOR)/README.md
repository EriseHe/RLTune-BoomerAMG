# Learning to Relax - Julia Implementation

Julia translation of the MATLAB code from ["Learning to Relax: Setting Solver Parameters Across a Sequence of Linear System Instances"](https://arxiv.org/abs/2310.02246) (Khodak et al., ICLR 2024).

## Structure

This repository mirrors the original MATLAB structure exactly:

```
LearningToRelax/
├── src/
│   ├── LearningToRelax.jl      # Main module
│   ├── learners/
│   │   ├── TsallisINF.jl       # learners/TsallisINF.m
│   │   ├── TsallisINFCB.jl     # learners/TsallisINFCB.m
│   │   └── ChebCB.jl           # learners/ChebCB.m
│   ├── solvers/
│   │   ├── sor.jl              # solvers/sor.m
│   │   ├── rho_jacobi.jl       # solvers/rho_jacobi.m
│   │   ├── omega_opt.jl        # solvers/omega_opt.m
│   │   ├── omega_grid.jl       # solvers/omega_grid.m
│   │   └── energy_norm.jl      # solvers/energy_norm.m
│   └── utils/
│       ├── truncated_normal.jl # utils/truncated_normal.m
│       └── laplacian.jl        # delsq(numgrid('S', n))
├── scripts/
│   ├── learning.jl             # scripts/learning.m
│   ├── contextual.jl           # scripts/contextual.m
│   ├── degenerate.jl           # scripts/degenerate.m
│   ├── asymptotic.jl           # scripts/asymptotic.m
│   └── comparators.jl          # scripts/comparators.m
└── plots/                      # Output figures
```

## MATLAB ↔ Julia File Mapping

| MATLAB File | Julia File | Output |
|-------------|-----------|--------|
| `scripts/learning.m` | `scripts/learning.jl` | `learning_high_variance.png`, `learning_low_variance.png` |
| `scripts/contextual.m` | `scripts/contextual.jl` | `contextual_high_variance.png`, `contextual_low_variance.png` |
| `scripts/degenerate.m` | `scripts/degenerate.jl` | `degenerate.png` |
| `scripts/asymptotic.m` | `scripts/asymptotic.jl` | `bound_comparison.png`, `asymptocity.png`, `tau_beta.png` |
| `scripts/comparators.m` | `scripts/comparators.jl` | `low_variance.png`, `high_variance.png` |

## Usage

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()

# Run learning experiment (Figure 5 from paper)
include("scripts/learning.jl")

# Run contextual bandit experiment (Figure 2 from paper)
include("scripts/contextual.jl")

# Run degenerate target vector experiment
include("scripts/degenerate.jl")

# Run asymptotic bounds experiment
include("scripts/asymptotic.jl")

# Run comparators experiment
include("scripts/comparators.jl")
```

## Reference

```bibtex
@inproceedings{khodak2024learning,
  title={Learning to Relax: Setting Solver Parameters Across a Sequence of Linear System Instances},
  author={Khodak, Mikhail and Mackey, Lester and Wei, Yun},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
