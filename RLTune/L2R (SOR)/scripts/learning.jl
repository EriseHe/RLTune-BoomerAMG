# learning.jl
# Faithful Julia translation of scripts/learning.m
#
# Compares the performance of Tsallis-INF and several fixed choices of
# omega on a sequence of 5000 i.i.d. linear systems; all results are
# averages over 40 trials

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using Distributions
using LinearAlgebra
using SparseArrays
using Plots
using ProgressMeter

include(joinpath(@__DIR__, "..", "src", "LearningToRelax.jl"))
using .LearningToRelax

# Include visualization functions (supplementary, not part of MATLAB translation)
include(joinpath(@__DIR__, "visualization.jl"))

# Setup (matching MATLAB lines 9-16)
A = create_laplacian_2d(12)
n = size(A, 1)
epsilon = 1e-8
T = 5000
trials = 40
omegas = collect(range(1.0, 1.8, length=5))  # linspace(1., 1.8, 5)
omega_grid = collect(range(1.0, 1.95, length=20))  # Grid for Tsallis-INF
omega_costs = zeros(T, trials, length(omegas))
tinf_costs = zeros(T, trials)

# Visualization tracking (recorded during trial 1 only - no extra compute)
chosen_omegas_viz = zeros(T)
action_probs_viz = zeros(T, length(omega_grid))
costs_viz = zeros(T)  # costs incurred during trial 1

# Create output directory
mkpath(joinpath(@__DIR__, "..", "plots"))

println("Learning Experiment")
println("=" ^ 40)

# High-variance offset distribution (lines 18-31)
println("\nHigh-variance offset distribution...")
@showprogress for trial in 1:trials
    tinf = TsallisINF(omega_grid, T)
    for t in 1:T
        c = -0.15 + 0.6 * rand(Beta(0.5, 1.5))
        At = A + c * sparse(I, n, n)
        bt = truncated_normal(n)
        
        omega_chosen = LearningToRelax.predict!(tinf)
        tinf_costs[t, trial] = sor(At, bt, zeros(n), omega_chosen, epsilon)
        
        # Track bandit behavior during trial 1 (for visualization, no extra compute)
        if trial == 1
            chosen_omegas_viz[t] = omega_chosen
            costs_viz[t] = tinf_costs[t, trial]  # record cost for this step
            # Compute action probabilities (same calculation as in predict!)
            eta = 2.0 / sqrt(t)
            x = -1.0
            probs = zeros(length(omega_grid))
            for _ in 1:20
                probs .= 4.0 .* (eta .* (tinf.k ./ tinf.scale .- x)) .^ (-2.0)
                x = x - (sum(probs) - 1.0) / (eta * sum(probs .^ 1.5))
            end
            action_probs_viz[t, :] = probs ./ sum(probs)
        end
        
        LearningToRelax.update!(tinf, tinf_costs[t, trial])
        for i in 1:5
            omega_costs[t, trial, i] = sor(At, bt, zeros(n), omegas[i], epsilon)
        end
    end
end

# Plot (lines 33-46)
p1 = plot(xlabel="total iterations", ylabel="instances remaining", legend=:topright)
for i in 1:5
    plot!(p1, vec(mean(cumsum(omega_costs[:, :, i], dims=1), dims=2)), T:-1:1,
          linewidth=2, linestyle=:dash, label="ω=$(omegas[i])")
end
plot!(p1, vec(mean(cumsum(tinf_costs, dims=1), dims=2)), T:-1:1,
      linewidth=2, color=:black, label="Tsallis-INF")
savefig(p1, joinpath(@__DIR__, "..", "plots", "learning_high_variance.png"))
println("Saved learning_high_variance.png")

# Low-variance offset distribution (lines 48-61)
println("\nLow-variance offset distribution...")
@showprogress for trial in 1:trials
    tinf = TsallisINF(omega_grid, T)
    for t in 1:T
        c = -0.15 + 0.6 * rand(Beta(2.0, 6.0))
        At = A + c * sparse(I, n, n)
        bt = truncated_normal(n)
        omega_chosen = LearningToRelax.predict!(tinf)
        tinf_costs[t, trial] = sor(At, bt, zeros(n), omega_chosen, epsilon)
        LearningToRelax.update!(tinf, tinf_costs[t, trial])
        for i in 1:5
            omega_costs[t, trial, i] = sor(At, bt, zeros(n), omegas[i], epsilon)
        end
    end
end

# Plot (lines 63-76)
p2 = plot(xlabel="total iterations", ylabel="instances remaining", legend=:topright)
for i in 1:5
    plot!(p2, vec(mean(cumsum(omega_costs[:, :, i], dims=1), dims=2)), T:-1:1,
          linewidth=2, linestyle=:dash, label="ω=$(omegas[i])")
end
plot!(p2, vec(mean(cumsum(tinf_costs, dims=1), dims=2)), T:-1:1,
      linewidth=2, color=:black, label="Tsallis-INF")
savefig(p2, joinpath(@__DIR__, "..", "plots", "learning_low_variance.png"))
println("Saved learning_low_variance.png")

#=============================================================================
  VISUALIZATION: Bandit behavior plots (from trial 1 data, no extra compute)
  NOTE: This is supplementary analysis, not part of original MATLAB translation
=============================================================================#

generate_bandit_visualizations(
    chosen_omegas_viz,
    action_probs_viz,
    costs_viz,
    omega_grid,
    "ω",
    joinpath(@__DIR__, "..", "plots"),
    "SOR"
)

println("\nDone!")

