"""
Computes the number of iterations required for SSOR-preconditioned CG to 
solve a linear system to a given tolerance, returning the approximate 
solution obtained as a second output; caps the number of iterations at 10000.
Direct translation of ssor_pcg.m
"""

import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse import issparse, diags
from scipy.linalg import solve_triangular


def ssor_pcg(A, b, x, omega, tol):
    """
    SSOR-preconditioned CG solver.
    
    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix (symmetric positive definite)
    b : ndarray
        Right-hand side vector
    x : ndarray
        Initial guess
    omega : float
        Relaxation parameter
    tol : float
        Relative tolerance for convergence
    
    Returns
    -------
    k : int
        Number of iterations required
    out : ndarray
        Approximate solution
    
    Notes
    -----
    MATLAB code:
        D = diag(diag(A));
        L = tril(A, -1);
        X = D + omega * L;
        [out, ~, ~, k] = pcg(A, b, tol, 10000, X'*inv(D), X/omega/(2.-omega), x);
    
    The preconditioner M = M1 * M2 where:
        M1 = X' * inv(D)
        M2 = X / omega / (2 - omega)
    
    So M = X' * inv(D) * X / omega / (2 - omega)
    and M \ r = (2-omega) * omega * inv(X) * D * inv(X') * r
    """
    # Convert to dense if sparse for preconditioner construction
    if hasattr(A, 'toarray'):
        A_dense = A.toarray()
    else:
        A_dense = np.asarray(A)
    
    n = A_dense.shape[0]
    
    # D = diag(diag(A))
    D_diag = np.diag(A_dense)
    D = np.diag(D_diag)
    
    # L = tril(A, -1)
    L = np.tril(A_dense, -1)
    
    # X = D + omega * L
    X = D + omega * L
    
    # Preconditioner: M = M1 * M2 = X' * inv(D) * X / omega / (2 - omega)
    # M \ r = (2 - omega) * omega * inv(X) * D * inv(X') * r
    # 
    # Since X = D + omega * L is lower triangular,
    # X' = D + omega * L' is upper triangular.
    # We solve: X' * y1 = r           (upper triangular solve)
    #           y2 = D * y1           (diagonal scaling)
    #           X * y3 = y2           (lower triangular solve)
    #           result = (2 - omega) * omega * y3
    
    def preconditioner(r):
        # X' is upper triangular (D + omega * L.T)
        X_T = D + omega * L.T
        y1 = solve_triangular(X_T, r, lower=False)
        y2 = D @ y1
        y3 = solve_triangular(X, y2, lower=True)
        return (2.0 - omega) * omega * y3
    
    # Create LinearOperator for preconditioner
    from scipy.sparse.linalg import LinearOperator
    M = LinearOperator((n, n), matvec=preconditioner)
    
    # Track iterations
    iteration_count = [0]
    
    def callback(xk):
        iteration_count[0] += 1
    
    # Run PCG
    # Note: scipy.cg uses relative tolerance ||r||/||b||, same as MATLAB pcg
    out, info = cg(A_dense, b, x0=x, tol=tol, maxiter=10000, M=M, callback=callback)
    
    # scipy.cg doesn't directly return iteration count, we use callback
    # But we also need to handle the case where it converges immediately
    k = iteration_count[0]
    if k == 0:
        k = 1  # At least one iteration
    
    return k, out
