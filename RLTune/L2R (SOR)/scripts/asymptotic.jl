# Plots several quantities associated with the near-asymptotic bound on the
# number of SOR iterations required to converge
# Direct translation of asymptotic.m

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using LinearAlgebra
using Statistics
using Plots

include(joinpath(@__DIR__, "..", "src", "LearningToRelax.jl"))
using .LearningToRelax

println("Running asymptotic experiment...")

A = create_laplacian_2d(12)
n = size(A, 1)
A_dense = Matrix(A)
D = Diagonal(diag(A_dense))
L_mat = tril(A_dense, -1)
b = truncated_normal(n)
epsilon = 1e-8

# Computes and plots different estimates of SOR iterations
omegas = omega_grid(A, 1.0, 1.99, 0.01)
actual = zeros(length(omegas))
radius = zeros(length(omegas))
errs = zeros(length(omegas))
energy = zeros(length(omegas))

println("Computing bounds for $(length(omegas)) omega values...")
for i in 1:length(omegas)
    k = sor_iterations(A, b, omegas[i], epsilon)
    M = D / omegas[i] + L_mat
    C = Matrix(I, n, n) - A_dense * inv(M)
    radius[i] = maximum(abs.(eigvals(C)))
    errs[i] = norm(C^(k-1))^(1/(k-1)) - radius[i]
    actual[i] = k
    energy[i] = energy_norm(A, omegas[i])
end

tau = maximum(errs ./ (1 .- radius))

# Figure 1: bound_comparison.png
plt1 = plot(size=(700, 500), dpi=150, yscale=:log10, legend=:topleft)
plot!(plt1, omegas, actual, linewidth=2, label="actual cost")
plot!(plt1, omegas, log.(epsilon) ./ log.(radius), linewidth=2, 
      color=RGB(0.466, 0.674, 0.188), label="asymptotic estimate")
plot!(plt1, omegas, log.(epsilon / (2*sqrt(cond(A_dense)))) ./ log.(energy), linewidth=2,
      color=RGB(0.929, 0.694, 0.125), label="energy bound")
plot!(plt1, omegas, log.(epsilon) ./ log.(radius .+ tau .* (1 .- radius)), linewidth=2,
      color=RGB(0.85, 0.325, 0.098), label="near-asymptotic bound")
xlabel!(plt1, "ω")
ylabel!(plt1, "iterations")

mkpath(joinpath(@__DIR__, "..", "plots"))
savefig(plt1, joinpath(@__DIR__, "..", "plots", "bound_comparison.png"))
println("Saved: plots/bound_comparison.png")

# Figure 2: asymptocity.png
plt2 = plot(size=(700, 500), dpi=150)
plot!(plt2, omegas, errs, linewidth=2, label="||Cω^k||^(1/k) - ρ(Cω)")
plot!(plt2, omegas, tau .* (1 .- radius), linewidth=2, label="τ(1 - ρ(Cω))")
xlabel!(plt2, "ω")

savefig(plt2, joinpath(@__DIR__, "..", "plots", "asymptocity.png"))
println("Saved: plots/asymptocity.png")

# Figure 3: tau_beta.png - computes tau and beta at different offsets
println("Computing tau and beta for different offsets...")
cs = range(-0.15, 0.45, length=97)
taus = zeros(length(cs))
betas = zeros(length(cs))

for i in 1:length(cs)
    if i % 20 == 0
        println("  Progress: $i / $(length(cs))")
    end
    Ac = A + cs[i] * I
    Ac_dense = Matrix(Ac)
    omegas_c = omega_grid(A, 1.0, 1.9, 0.01)
    Dc = Diagonal(diag(Ac_dense))
    Lc = tril(Ac_dense, -1)
    radius_c = zeros(length(omegas_c))
    errs_c = zeros(length(omegas_c))
    
    for j in 1:length(omegas_c)
        k = sor_iterations(Ac, b, omegas_c[j], epsilon)
        Mc = Dc / omegas_c[j] + Lc
        C = Matrix(I, n, n) - Ac_dense * inv(Mc)
        radius_c[j] = maximum(abs.(eigvals(C)))
        errs_c[j] = norm(C^(k-1))^(1/(k-1)) - radius_c[j]
    end
    betas[i] = rho_jacobi(Ac)
    taus[i] = maximum(errs_c ./ (1 .- radius_c))
end

plt3 = plot(size=(700, 500), dpi=150)
plot!(plt3, cs, betas, linewidth=2, label="β")
plot!(plt3, cs, fill(4/exp(2)*(1-1/exp(2)), length(cs)), linewidth=2, linestyle=:dash, label="4(1-1/e²)/e²")
plot!(plt3, cs, taus, linewidth=2, label="τ")
plot!(plt3, cs, fill(1/exp(2), length(cs)), linewidth=2, linestyle=:dash, label="1/e²")
xlabel!(plt3, "c")
ylims!(plt3, 0, 1)

savefig(plt3, joinpath(@__DIR__, "..", "plots", "tau_beta.png"))
println("Saved: plots/tau_beta.png")

