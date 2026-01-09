# omega_opt.jl
# Faithful Julia translation of solvers/omega_opt.m

"""
    omega_opt(A) -> Float64

Computes the asymptotically optimal omega for SOR.
"""
function omega_opt(A)
    beta = rho_jacobi(A)
    return 1 + (beta / (1 + sqrt(1 - beta^2)))^2
end

