# learning_amg.jl
# Learning to Relax with AMG (Algebraic Multigrid)
#
# This script is analogous to LearningToRelax/scripts/learning.jl
# but swaps SOR with AMG.
#
# Key changes from learning.jl:
#   1. sor(A, b, x0, ω, tol) → amg_work_units(A, b, param, tol)
#   2. Cost: SOR iterations → AMG work units (bounded to [1,2] for Tsallis-INF)
#   3. Tunable param set in amg.jl (default: StrongThreshold ∈ [0.1, 0.9])

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Statistics
using Distributions
using LinearAlgebra
using SparseArrays
using Plots
using ProgressMeter

# Import components from L2R (SOR) project (unchanged learners/utils)
L2R_PATH = joinpath(@__DIR__, "..", "..", "L2R (SOR)")
include(joinpath(L2R_PATH, "src", "LearningToRelax.jl"))
using .LearningToRelax: TsallisINF, predict!, update!
using .LearningToRelax: truncated_normal, create_laplacian_2d

# Import AMG solver (returns work-like units via amg_work_units)
include(joinpath(@__DIR__, "..", "src", "solvers", "amg.jl"))

# Include visualization functions
include(joinpath(@__DIR__, "visualization.jl"))

#=============================================================================
  CONFIGURATION
  
  Changes from learning.jl:
  1. Tunable parameter set in amg.jl (default: StrongThreshold)
  2. Solver: amg_solve (returns time_ms) instead of sor (returns iterations)
  3. Cost normalization: time_ms / ref_time (keeps losses ~O(1) for bandit)
=============================================================================#

# Problem setup (same as learning.jl)
A = create_laplacian_2d(12)
n = size(A, 1)
tol = 1e-8
T = 5000      # Number of instances per trial
trials = 40   # Number of trials
nnz_A0 = nnz(A)  # Base nnz for AP-complexity normalization

# AMG work-unit bounding for Tsallis-INF (loss fed as 1 + min(raw/K_WU, 1))
const NU_PRE = 1
const NU_POST = 1
const MAXITER_AMG = 10000
const K_WU_SAFETY = 5.0  # conservative upper bound for AP-complexity variation
const K_WU = MAXITER_AMG * (NU_PRE + NU_POST) * K_WU_SAFETY

# AMG tunable parameter grid (set based on TUNABLE_PARAM in amg.jl)
# StrongThreshold: typical range 0.1-0.9 (0.25 default, lower=more aggressive coarsening)
# RelaxWt: typical range 0.3-1.0 (only if using RelaxType 0/3/6)
param_values = collect(range(0.1, 0.9, length=5))   # Fixed values for comparison
param_grid = collect(range(0.1, 0.9, length=20))    # Grid for Tsallis-INF
baseline_param = 0.25  # Reference for normalization (default StrongThreshold)
param_label = get_tunable_param_name()  # "θ" for strong_threshold, "w" for relax_wt

# Cost arrays (store work units)
fixed_wu = zeros(T, trials, length(param_values))
tinf_wu = zeros(T, trials)

# Visualization tracking (recorded during trial 1 only - no extra compute)
chosen_params_viz = zeros(T)
action_probs_viz = zeros(T, length(param_grid))
costs_viz = zeros(T)  # costs incurred during trial 1

# Create output directory
mkpath(joinpath(@__DIR__, "..", "plots"))

println("\nLearning to Relax with AMG (work-unit feedback)")
println("=" ^ 40)
println("Tunable parameter: $(TUNABLE_PARAM) ($param_label)")
println("Instances per trial: $T")
println("Number of trials: $trials")
println("Parameter grid: $(param_grid[1]) to $(param_grid[end])")

#=============================================================================
  HIGH-VARIANCE OFFSET DISTRIBUTION
  Same as learning.jl, but using normalized time as cost
=============================================================================#

println("\nHigh-variance offset distribution (Beta(0.5, 1.5))...")
for trial in 1:trials
    tinf = TsallisINF(param_grid, T)
    prog = Progress(T; desc="High-var trial $trial/$trials", dt=1.0)
    
    for t in 1:T
        # Generate random system (same as learning.jl)
        c = -0.15 + 0.6 * rand(Beta(0.5, 1.5))
        At = A + c * sparse(I, n, n)
        bt = truncated_normal(n)
        
        # Bandit predicts → AMG solves → returns work units (WU)
        predicted_param = predict!(tinf)
        wu, num_cycles, apcomp = amg_work_units(
            At, bt, predicted_param, tol;
            nu_pre=NU_PRE, nu_post=NU_POST, base_nnz=nnz_A0
        )
        tinf_wu[t, trial] = wu
        
        # Track bandit behavior during trial 1 (for visualization, no extra compute)
        if trial == 1
            chosen_params_viz[t] = predicted_param
            costs_viz[t] = wu  # record cost for this step
            # Compute action probabilities (same calculation as in predict!)
            eta = 2.0 / sqrt(t)
            x = -1.0
            probs = zeros(length(param_grid))
            for _ in 1:20
                probs .= 4.0 .* (eta .* (tinf.k ./ tinf.scale .- x)) .^ (-2.0)
                x = x - (sum(probs) - 1.0) / (eta * sum(probs .^ 1.5))
            end
            action_probs_viz[t, :] = probs ./ sum(probs)
        end
        
        # Feed bounded loss: 1 + min(raw_wu / K_WU, 1)
        loss_bounded = 1.0 + min(wu / K_WU, 1.0)
        update!(tinf, loss_bounded)
        
        # Run fixed parameter values for comparison (work units)
        for i in 1:length(param_values)
            wu_fixed, _, _ = amg_work_units(
                At, bt, param_values[i], tol;
                nu_pre=NU_PRE, nu_post=NU_POST, base_nnz=nnz_A0
            )
            fixed_wu[t, trial, i] = wu_fixed
        end

        next!(prog)
    end
end

# Plot (using cumulative work units)
p1 = plot(xlabel="cumulative work units (WU)", ylabel="instances remaining", legend=:topright,
          title="AMG: High-variance offset (tuning $param_label)")
for i in 1:length(param_values)
    plot!(p1, vec(mean(cumsum(fixed_wu[:, :, i], dims=1), dims=2)), T:-1:1,
          linewidth=2, linestyle=:dash, label="$param_label=$(round(param_values[i], digits=2))")
end
plot!(p1, vec(mean(cumsum(tinf_wu, dims=1), dims=2)), T:-1:1,
      linewidth=2, color=:black, label="Tsallis-INF")
savefig(p1, joinpath(@__DIR__, "..", "plots", "amg_high_variance.png"))
println("Saved amg_high_variance.png")

#=============================================================================
  LOW-VARIANCE OFFSET DISTRIBUTION
  Same as learning.jl, but using work units (no wall-clock timing)
=============================================================================#

println("\nLow-variance offset distribution (Beta(2.0, 6.0))...")
for trial in 1:trials
    tinf = TsallisINF(param_grid, T)
    prog = Progress(T; desc="Low-var trial $trial/$trials", dt=1.0)
    
    for t in 1:T
        c = -0.15 + 0.6 * rand(Beta(2.0, 6.0))
        At = A + c * sparse(I, n, n)
        bt = truncated_normal(n)
        
        predicted_param = predict!(tinf)
        wu, num_cycles, apcomp = amg_work_units(
            At, bt, predicted_param, tol;
            nu_pre=NU_PRE, nu_post=NU_POST, base_nnz=nnz_A0
        )
        tinf_wu[t, trial] = wu

        loss_bounded = 1.0 + min(wu / K_WU, 1.0)
        update!(tinf, loss_bounded)

        for i in 1:length(param_values)
            wu_fixed, _, _ = amg_work_units(
                At, bt, param_values[i], tol;
                nu_pre=NU_PRE, nu_post=NU_POST, base_nnz=nnz_A0
            )
            fixed_wu[t, trial, i] = wu_fixed
        end

        next!(prog)
    end
end

# Plot
p2 = plot(xlabel="cumulative work units (WU)", ylabel="instances remaining", legend=:topright,
          title="AMG: Low-variance offset (tuning $param_label)")
for i in 1:length(param_values)
    plot!(p2, vec(mean(cumsum(fixed_wu[:, :, i], dims=1), dims=2)), T:-1:1,
          linewidth=2, linestyle=:dash, label="$param_label=$(round(param_values[i], digits=2))")
end
plot!(p2, vec(mean(cumsum(tinf_wu, dims=1), dims=2)), T:-1:1,
      linewidth=2, color=:black, label="Tsallis-INF")
savefig(p2, joinpath(@__DIR__, "..", "plots", "amg_low_variance.png"))
println("Saved amg_low_variance.png")

#=============================================================================
  VISUALIZATION: Bandit behavior plots (from trial 1 data, no extra compute)
=============================================================================#

generate_bandit_visualizations(
    chosen_params_viz,
    action_probs_viz,
    costs_viz,
    param_grid,
    param_label,
    joinpath(@__DIR__, "..", "plots"),
    "AMG"
)

println("\nDone!")
