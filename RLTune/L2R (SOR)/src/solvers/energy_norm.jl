# Computes a bound on the energy norm of the SOR iteration matrix
# (Hackbusch 2016, Corollary 3.45)
# Direct translation of energy_norm.m

using LinearAlgebra

function energy_norm(A, omega::Float64)
    A_dense = Matrix(A)
    Omega = (2 - omega) / (2 * omega)
    invD = inv(Diagonal(diag(A_dense)))
    L = tril(A_dense, -1)
    gamma = 1.0 - maximum(abs.(eigvals(invD * (L + L'))))
    bound = sqrt(1.0 - 2 * Omega * gamma / (Omega^2 + gamma / omega + maximum(abs.(eigvals(invD * L * invD * L'))) - 0.25))
    return bound
end

