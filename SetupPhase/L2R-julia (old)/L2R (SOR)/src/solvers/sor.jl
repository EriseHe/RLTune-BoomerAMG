# sor.jl
# Faithful Julia translation of solvers/sor.m

"""
    sor(A, b, x, omega, tol) -> Int

Computes the number of iterations required for SOR to solve a linear
system to a given tolerance; caps the number of iterations at 10000.

# Arguments
- `A`: System matrix
- `b`: Right-hand side vector
- `x`: Initial guess
- `omega`: Relaxation parameter
- `tol`: Relative tolerance

# Returns
- Number of iterations to convergence
"""
function sor(A, b, x, omega, tol)
    A_dense = issparse(A) ? Matrix(A) : A
    n = length(b)
    
    D = Diagonal(diag(A_dense))
    M = D / omega + tril(A_dense, -1)
    
    x = copy(x)
    r = b - A_dense * x
    norm0 = norm(r)
    
    for k in 1:10000
        x = x + M \ r
        r = b - A_dense * x
        if norm(r) / norm0 < tol
            return k
        end
    end
    
    return 10000
end
