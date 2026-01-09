# Computes and plots the number of iterations required for SOR to converge,
# as well as the near-asymptotic bound for comparison, averaged across forty
# different randomly sampled linear systems, where the randomness
# determines a scalar offset of the diagonal
# Direct translation of comparators.m

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Statistics
using Distributions
using Plots

include(joinpath(@__DIR__, "..", "src", "LearningToRelax.jl"))
using .LearningToRelax

function run_comparators()
    println("Running comparators experiment...")

    A = create_laplacian_2d(12)
    n = size(A, 1)
    A_dense = Matrix(A)
    epsilon = 1e-8
    trials = 40
    omegas = omega_grid(A, 1.0, 1.99, 0.01)

    # Low-variance offset distribution: Beta(2, 6)
    println("Low-variance offset distribution (Beta(2,6))...")
    cs = -0.15 .+ 0.6 .* rand(Beta(2.0, 6.0), trials)
    taus = zeros(trials)
    betas = zeros(trials)
    actual = zeros(length(omegas))
    dynamic_actual = 0.0
    predicted = zeros(length(omegas))
    dynamic_predicted = 0.0

    for i in 1:length(cs)
        if i % 10 == 0
            println("  Progress: $i / $(length(cs))")
        end
        Ac = A + cs[i] * I
        Ac_dense = Matrix(Ac)
        b = truncated_normal(n)
        Dc = Diagonal(diag(Ac_dense))
        Lc = tril(Ac_dense, -1)
        radius = zeros(length(omegas))
        errs = zeros(length(omegas))
        current_best = Inf
        
        for j in 1:length(omegas)
            k = sor_iterations(Ac, b, omegas[j], epsilon)
            Mc = Dc / omegas[j] + Lc
            C = Matrix(I, n, n) - Ac_dense * inv(Mc)
            radius[j] = maximum(abs.(eigvals(C)))
            errs[j] = norm(C^(k-1))^(1/(k-1)) - radius[j]
            actual[j] += k
            if k < current_best
                current_best = k
            end
        end
        dynamic_actual += current_best
        betas[i] = maximum(abs.(eigvals(Matrix(I, n, n) - inv(Dc) * Ac_dense)))
        taus[i] = maximum(errs ./ (1 .- radius))
        current_predicted = 1 .+ log.(epsilon) ./ log.(radius .+ taus[i] .* (1 .- radius))
        dynamic_predicted += minimum(current_predicted)
        predicted .+= current_predicted
    end

    # Plot low-variance
    plt1 = plot(size=(700, 500), dpi=150, yscale=:log10)
    plot!(plt1, omegas, actual ./ length(cs), linewidth=2, color=RGB(0, 0.447, 0.741), label="actual cost")
    plot!(plt1, omegas, fill(dynamic_actual / trials, length(omegas)), linewidth=2, linestyle=:dash, 
          color=RGB(0, 0.447, 0.741), label="(instance-optimal)")
    plot!(plt1, omegas, predicted ./ length(cs), linewidth=2, color=RGB(0.85, 0.325, 0.098), label="near-asymptotic bound")
    plot!(plt1, omegas, fill(dynamic_predicted / trials, length(omegas)), linewidth=2, linestyle=:dash,
          color=RGB(0.85, 0.325, 0.098), label="(instance-optimal)")
    xlabel!(plt1, "ω")
    ylabel!(plt1, "iterations")

    mkpath(joinpath(@__DIR__, "..", "plots"))
    savefig(plt1, joinpath(@__DIR__, "..", "plots", "low_variance.png"))
    println("Saved: plots/low_variance.png")

    # High-variance offset distribution: Beta(0.5, 1.5)
    println("High-variance offset distribution (Beta(0.5,1.5))...")
    cs = -0.15 .+ 0.6 .* rand(Beta(0.5, 1.5), trials)
    taus = zeros(trials)
    betas = zeros(trials)
    actual = zeros(length(omegas))
    dynamic_actual = 0.0
    predicted = zeros(length(omegas))
    dynamic_predicted = 0.0

    for i in 1:length(cs)
        if i % 10 == 0
            println("  Progress: $i / $(length(cs))")
        end
        Ac = A + cs[i] * I
        Ac_dense = Matrix(Ac)
        b = truncated_normal(n)
        Dc = Diagonal(diag(Ac_dense))
        Lc = tril(Ac_dense, -1)
        radius = zeros(length(omegas))
        errs = zeros(length(omegas))
        current_best = Inf
        
        for j in 1:length(omegas)
            k = sor_iterations(Ac, b, omegas[j], epsilon)
            Mc = Dc / omegas[j] + Lc
            C = Matrix(I, n, n) - Ac_dense * inv(Mc)
            radius[j] = maximum(abs.(eigvals(C)))
            errs[j] = norm(C^(k-1))^(1/(k-1)) - radius[j]
            actual[j] += k
            if k < current_best
                current_best = k
            end
        end
        dynamic_actual += current_best
        betas[i] = maximum(abs.(eigvals(Matrix(I, n, n) - inv(Dc) * Ac_dense)))
        taus[i] = maximum(errs ./ (1 .- radius))
        current_predicted = 1 .+ log.(epsilon) ./ log.(radius .+ taus[i] .* (1 .- radius))
        dynamic_predicted += minimum(current_predicted)
        predicted .+= current_predicted
    end

    # Plot high-variance
    plt2 = plot(size=(700, 500), dpi=150, yscale=:log10)
    plot!(plt2, omegas, actual ./ length(cs), linewidth=2, color=RGB(0, 0.447, 0.741), label="actual cost")
    plot!(plt2, omegas, fill(dynamic_actual / trials, length(omegas)), linewidth=2, linestyle=:dash,
          color=RGB(0, 0.447, 0.741), label="(instance-optimal)")
    plot!(plt2, omegas, predicted ./ length(cs), linewidth=2, color=RGB(0.85, 0.325, 0.098), label="near-asymptotic bound")
    plot!(plt2, omegas, fill(dynamic_predicted / trials, length(omegas)), linewidth=2, linestyle=:dash,
          color=RGB(0.85, 0.325, 0.098), label="(instance-optimal)")
    xlabel!(plt2, "ω")
    ylabel!(plt2, "iterations")

    savefig(plt2, joinpath(@__DIR__, "..", "plots", "high_variance.png"))
    println("Saved: plots/high_variance.png")
end

run_comparators()
