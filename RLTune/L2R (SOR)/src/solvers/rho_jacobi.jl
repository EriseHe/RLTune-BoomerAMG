# rho_jacobi.jl
# Faithful Julia translation of solvers/rho_jacobi.m

"""
    rho_jacobi(A) -> Float64

Computes the spectral radius of the Jacobi iteration matrix.
"""
function rho_jacobi(A)
    A_full = issparse(A) ? Matrix(A) : A
    n = size(A, 1)
    return maximum(abs.(eigvals(I(n) - inv(Diagonal(diag(A_full))) * A_full)))
end

