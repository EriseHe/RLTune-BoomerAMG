# LearningToRelax.jl
# Julia implementation of "Learning to Relax" (Khodak et al., ICLR 2024)
# Faithful reproduction of the MATLAB code from learning-to-relax

module LearningToRelax

using LinearAlgebra
using SparseArrays
using Random
using Statistics
using StatsBase: sample, Weights

# Solvers (matching solvers/ folder)
include("solvers/sor.jl")
include("solvers/rho_jacobi.jl")
include("solvers/omega_opt.jl")
include("solvers/omega_grid.jl")
include("solvers/energy_norm.jl")

# Utils (matching utils/ folder)
include("utils/truncated_normal.jl")
include("utils/laplacian.jl")

# Learners (matching learners/ folder)
include("learners/TsallisINF.jl")
include("learners/TsallisINFCB.jl")
include("learners/ChebCB.jl")

# Convenience wrapper for sor (x0 defaults to zeros)
sor_iterations(A, b, omega, tol) = sor(A, b, zeros(length(b)), omega, tol)

# Exports
export sor, sor_iterations
export rho_jacobi, omega_opt, omega_grid, energy_norm
export truncated_normal, create_laplacian_2d
export TsallisINF, TsallisINFCB, ChebCB
export predict!, update!

end
