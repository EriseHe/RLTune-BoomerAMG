# Compares the number of iterations required to solve a linear system when
# the target vector is drawn from a truncated Gaussian vs. when it is
# generated using the smallest eigenvector of the iteration matrix for omega=1.4
# Direct translation of degenerate.m

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Statistics
using Plots

include(joinpath(@__DIR__, "..", "src", "LearningToRelax.jl"))
using .LearningToRelax

println("Running degenerate experiment...")

A = create_laplacian_2d(12)
n = size(A, 1)
A_dense = Matrix(A)
D = Diagonal(diag(A_dense))
L_mat = tril(A_dense, -1)
epsilon = 1e-8

omegas = omega_grid(A, 1.0, 1.9, 0.001)

# C = I - A * inv(D/1.4 + L)
M_14 = D / 1.4 + L_mat
C = Matrix(I, n, n) - A_dense * inv(M_14)
F = eigen(C)
vecs = F.vectors
# Get eigenvector corresponding to smallest eigenvalue (index 99 in MATLAB = end-1 here)
degen = real.(vecs[:, end-1])

degen_cost = zeros(length(omegas))
gauss_cost = zeros(length(omegas))
gauss_min = zeros(length(omegas))
gauss_max = zeros(length(omegas))

println("Computing costs for $(length(omegas)) omega values...")
for i in 1:length(omegas)
    if i % 100 == 0
        println("  Progress: $i / $(length(omegas))")
    end
    degen_cost[i] = sor_iterations(A, degen, omegas[i], epsilon)
    costs = zeros(40)
    for j in 1:40
        costs[j] = sor_iterations(A, truncated_normal(n), omegas[i], epsilon)
    end
    gauss_cost[i] = mean(costs)
    gauss_max[i] = maximum(costs)
    gauss_min[i] = minimum(costs)
end

# Plot
plt = plot(size=(700, 500), dpi=150)

# Fill between min and max (shaded region)
plot!(plt, omegas, gauss_cost, ribbon=(gauss_cost .- gauss_min, gauss_max .- gauss_cost),
      fillalpha=0.3, fillcolor=:red, linewidth=0, label="")

# Degenerate cost line
plot!(plt, omegas, degen_cost, linewidth=2, color=RGB(0, 0.447, 0.741), label="Degenerate cost")

# Mean cost line
plot!(plt, omegas, gauss_cost, linewidth=2, color=RGB(0.85, 0.325, 0.098), label="Mean cost (Gaussian)")

xlabel!(plt, "ω")
ylabel!(plt, "Iterations")
xlims!(plt, 1.0, 1.9)

# Save plot
mkpath(joinpath(@__DIR__, "..", "plots"))
savefig(plt, joinpath(@__DIR__, "..", "plots", "degenerate.png"))
println("Saved: plots/degenerate.png")

