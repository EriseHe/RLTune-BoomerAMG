# contextual.jl
# Faithful Julia translation of scripts/contextual.m
#
# Compares the performance of different learning algorithms---including
# contextual bandit algorithms using diagonal offsets as context---on a
# sequence of 5000 i.i.d. linear systems; all results are averages over 40
# trials

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

# Setup (matching MATLAB lines 10-19)
A = create_laplacian_2d(12)
n = size(A, 1)
epsilon = 1e-8
T = 5000
trials = 40
tinf_costs = zeros(T, trials)
cheb_costs = zeros(T, trials)
tinfcb_costs = zeros(T, trials)
opt_costs = zeros(T, trials)
omega_costs = zeros(T, trials)

# Create output directory
mkpath(joinpath(@__DIR__, "..", "plots"))

println("Contextual Bandit Experiment")
println("=" ^ 40)

# High-variance offset distribution (lines 21-40)
println("\nHigh-variance offset distribution...")
@showprogress for trial in 1:trials
    tinf = TsallisINF(collect(range(1.0, 1.95, length=20)), T)
    cheb = ChebCB(collect(range(1.0, 1.95, length=20)), T, 6, -0.15, 0.65)
    tinfcb = TsallisINFCB(collect(range(1.0, 1.95, length=20)), 5, -0.15, 0.65)
    for t in 1:T
        c = -0.15 + 0.6 * rand(Beta(0.5, 1.5))
        At = A + c * sparse(I, n, n)
        bt = truncated_normal(n)
        tinf_costs[t, trial] = sor(At, bt, zeros(n), LearningToRelax.predict!(tinf), epsilon)
        LearningToRelax.update!(tinf, tinf_costs[t, trial])
        cheb_costs[t, trial] = sor(At, bt, zeros(n), LearningToRelax.predict!(cheb, c), epsilon)
        LearningToRelax.update!(cheb, cheb_costs[t, trial])
        tinfcb_costs[t, trial] = sor(At, bt, zeros(n), LearningToRelax.predict!(tinfcb, c), epsilon)
        LearningToRelax.update!(tinfcb, tinfcb_costs[t, trial])
        opt_costs[t, trial] = sor(At, bt, zeros(n), omega_opt(At), epsilon)
        omega_costs[t, trial] = sor(At, bt, zeros(n), 1.8, epsilon)
    end
    println("trial $trial finished")
end

# Plot (lines 42-56)
p1 = plot(xlabel="total iterations", ylabel="instances remaining", legend=:topright)
plot!(p1, vec(mean(cumsum(tinf_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="Tsallis-INF")
plot!(p1, vec(mean(cumsum(omega_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="ω=1.8 (≈ω*)")
plot!(p1, vec(mean(cumsum(tinfcb_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="Tsallis-INF-CB")
plot!(p1, vec(mean(cumsum(cheb_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="ChebCB")
plot!(p1, vec(mean(cumsum(opt_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="Instance-Optimal")
savefig(p1, joinpath(@__DIR__, "..", "plots", "contextual_high_variance.png"))
println("Saved contextual_high_variance.png")

# Low-variance offset distribution (lines 58-77)
println("\nLow-variance offset distribution...")
@showprogress for trial in 1:trials
    tinf = TsallisINF(collect(range(1.0, 1.95, length=20)), T)
    cheb = ChebCB(collect(range(1.0, 1.95, length=20)), T, 6, -0.15, 0.65)
    tinfcb = TsallisINFCB(collect(range(1.0, 1.95, length=20)), 5, -0.15, 0.65)
    for t in 1:T
        c = -0.15 + 0.6 * rand(Beta(2.0, 6.0))
        At = A + c * sparse(I, n, n)
        bt = truncated_normal(n)
        tinf_costs[t, trial] = sor(At, bt, zeros(n), LearningToRelax.predict!(tinf), epsilon)
        LearningToRelax.update!(tinf, tinf_costs[t, trial])
        cheb_costs[t, trial] = sor(At, bt, zeros(n), LearningToRelax.predict!(cheb, c), epsilon)
        LearningToRelax.update!(cheb, cheb_costs[t, trial])
        tinfcb_costs[t, trial] = sor(At, bt, zeros(n), LearningToRelax.predict!(tinfcb, c), epsilon)
        LearningToRelax.update!(tinfcb, tinfcb_costs[t, trial])
        opt_costs[t, trial] = sor(At, bt, zeros(n), omega_opt(At), epsilon)
        omega_costs[t, trial] = sor(At, bt, zeros(n), 1.6, epsilon)  # Note: 1.6 for low-variance
    end
    println("trial $trial finished")
end

# Plot (lines 79-93)
p2 = plot(xlabel="total iterations", ylabel="instances remaining", legend=:topright)
plot!(p2, vec(mean(cumsum(tinf_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="Tsallis-INF")
plot!(p2, vec(mean(cumsum(omega_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="ω=1.6 (≈ω*)")
plot!(p2, vec(mean(cumsum(tinfcb_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="Tsallis-INF-CB")
plot!(p2, vec(mean(cumsum(cheb_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="ChebCB")
plot!(p2, vec(mean(cumsum(opt_costs, dims=1), dims=2)), T:-1:1, linewidth=2, label="Instance-Optimal")
savefig(p2, joinpath(@__DIR__, "..", "plots", "contextual_low_variance.png"))
println("Saved contextual_low_variance.png")

println("\nDone!")

