"""
BoomerAMG solver for Learning to Relax framework.

Provides a simple interface to HYPRE's BoomerAMG solver that returns
(iterations, cum_nnz_AP, solution) where:
- iterations: number of V-cycles
- cum_nnz_AP: cumulative nnz ratio (sum nnz(A_l) + sum nnz(P_l)) / nnz(A_0)

The work units (WU) for bandit learning is: WU = iterations * cum_nnz_AP

The main tunable parameter is strong_threshold, which controls the
coarsening behavior in AMG (analogous to omega in SOR).
"""

from __future__ import annotations

import ctypes
import os
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp

from .hypre_loader import load_hypre as _load_hypre


# Type definitions for HYPRE ctypes bindings
def _hypre_int_type():
    env = os.environ.get("HYPRE_INT64", "").strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return ctypes.c_longlong
    return ctypes.c_int


HYPRE_Int = _hypre_int_type()
HYPRE_Real = ctypes.c_double

HYPRE_IJMatrix = ctypes.c_void_p
HYPRE_IJVector = ctypes.c_void_p
HYPRE_Solver = ctypes.c_void_p
HYPRE_ParCSRMatrix = ctypes.c_void_p
HYPRE_ParVector = ctypes.c_void_p

# HYPRE object type constant
HYPRE_PARCSR = 5555

# Maximum iterations cap (same as SOR)
MAX_ITERATIONS = 10000


def _mpi_comm_world() -> ctypes.c_void_p:
    """Return an MPI_COMM_WORLD handle for sequential HYPRE."""
    raw = os.environ.get("HYPRE_MPI_COMM_WORLD", "").strip()
    if raw:
        return ctypes.c_void_p(int(raw, 0))
    # For sequential HYPRE (built without MPI), return NULL
    return ctypes.c_void_p(0)


def _bind(lib: ctypes.CDLL, name: str, restype, argtypes):
    """Bind a function from the HYPRE library."""
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


def _has_symbol(lib: ctypes.CDLL, name: str) -> bool:
    """Check if a symbol exists in the library."""
    try:
        getattr(lib, name)
        return True
    except AttributeError:
        return False


def _hypre_initialize(lib: ctypes.CDLL) -> None:
    """Initialize HYPRE."""
    if _has_symbol(lib, "HYPRE_Initialize"):
        fn = _bind(lib, "HYPRE_Initialize", HYPRE_Int, [])
        fn()
    elif _has_symbol(lib, "HYPRE_Init"):
        fn = _bind(lib, "HYPRE_Init", HYPRE_Int, [])
        fn()


def _hypre_finalize(lib: ctypes.CDLL) -> None:
    """Finalize HYPRE."""
    if _has_symbol(lib, "HYPRE_Finalize"):
        fn = _bind(lib, "HYPRE_Finalize", HYPRE_Int, [])
        fn()


def boomeramg(A, b, x, strong_threshold: float, tol: float) -> Tuple[int, float, np.ndarray]:
    """
    Solve Ax=b using BoomerAMG as a standalone solver.

    This function returns (iterations, cum_nnz_AP, solution) where:
    - iterations is the number of V-cycles
    - cum_nnz_AP is the cumulative nonzeros ratio for work unit calculation

    The work units (WU) for bandit learning is: WU = iterations * cum_nnz_AP

    Parameters
    ----------
    A : ndarray or sparse matrix
        The system matrix (should be SPD)
    b : ndarray
        Right-hand side vector
    x : ndarray
        Initial guess (currently ignored, always starts from zero)
    strong_threshold : float
        AMG strong connection threshold (0.0 to 1.0)
        - Lower values: more aggressive coarsening
        - Higher values: more conservative coarsening
        - Typical: 0.25 for 2D, 0.5-0.6 for 3D
    tol : float
        Relative tolerance for convergence

    Returns
    -------
    k : int
        Number of iterations (V-cycles) required (capped at 10000)
    cum_nnz_AP : float
        Cumulative nnz ratio: (sum nnz(A_l) + sum nnz(P_l)) / nnz(A_0)
        This represents the memory/work complexity of the AMG hierarchy
    out : ndarray
        Approximate solution

    Notes
    -----
    The strong_threshold parameter is analogous to omega in SOR - it's the
    main tunable parameter that significantly affects convergence behavior.
    
    The work units formula: WU = iterations * cum_nnz_AP accounts for the
    fact that different strong_threshold values create hierarchies with
    different complexities.
    """
    # Convert to CSR format
    A_csr = sp.csr_matrix(A)
    if A_csr.shape[0] != A_csr.shape[1]:
        raise ValueError(f"A must be square, got {A_csr.shape}")

    n = int(A_csr.shape[0])
    b_vec = np.asarray(b, dtype=np.float64).reshape(-1)
    if b_vec.shape[0] != n:
        raise ValueError(f"b has wrong length: expected {n}, got {b_vec.shape[0]}")

    # Load HYPRE
    lib, _ = _load_hypre()
    _hypre_initialize(lib)

    comm = _mpi_comm_world()
    ilower = HYPRE_Int(0)
    iupper = HYPRE_Int(n - 1)

    # Create HYPRE objects
    ij_matrix = HYPRE_IJMatrix()
    ij_vector_b = HYPRE_IJVector()
    ij_vector_x = HYPRE_IJVector()
    solver = HYPRE_Solver()
    par_matrix = HYPRE_ParCSRMatrix()
    par_vector_b = HYPRE_ParVector()
    par_vector_x = HYPRE_ParVector()
    iters = HYPRE_Int()
    final_res = HYPRE_Real()
    cum_nnz_AP = HYPRE_Real()

    # Bind HYPRE functions
    HYPRE_IJMatrixCreate = _bind(
        lib, "HYPRE_IJMatrixCreate", HYPRE_Int,
        [ctypes.c_void_p, HYPRE_Int, HYPRE_Int, HYPRE_Int, HYPRE_Int, 
         ctypes.POINTER(HYPRE_IJMatrix)]
    )
    HYPRE_IJMatrixSetObjectType = _bind(
        lib, "HYPRE_IJMatrixSetObjectType", HYPRE_Int, 
        [HYPRE_IJMatrix, HYPRE_Int]
    )
    HYPRE_IJMatrixInitialize = _bind(
        lib, "HYPRE_IJMatrixInitialize", HYPRE_Int, 
        [HYPRE_IJMatrix]
    )
    HYPRE_IJMatrixSetValues = _bind(
        lib, "HYPRE_IJMatrixSetValues", HYPRE_Int,
        [HYPRE_IJMatrix, HYPRE_Int, ctypes.POINTER(HYPRE_Int), 
         ctypes.POINTER(HYPRE_Int), ctypes.POINTER(HYPRE_Int), 
         ctypes.POINTER(HYPRE_Real)]
    )
    HYPRE_IJMatrixAssemble = _bind(
        lib, "HYPRE_IJMatrixAssemble", HYPRE_Int, 
        [HYPRE_IJMatrix]
    )
    HYPRE_IJMatrixGetObject = _bind(
        lib, "HYPRE_IJMatrixGetObject", HYPRE_Int,
        [HYPRE_IJMatrix, ctypes.POINTER(HYPRE_ParCSRMatrix)]
    )
    HYPRE_IJMatrixDestroy = _bind(
        lib, "HYPRE_IJMatrixDestroy", HYPRE_Int, 
        [HYPRE_IJMatrix]
    )

    HYPRE_IJVectorCreate = _bind(
        lib, "HYPRE_IJVectorCreate", HYPRE_Int,
        [ctypes.c_void_p, HYPRE_Int, HYPRE_Int, ctypes.POINTER(HYPRE_IJVector)]
    )
    HYPRE_IJVectorSetObjectType = _bind(
        lib, "HYPRE_IJVectorSetObjectType", HYPRE_Int,
        [HYPRE_IJVector, HYPRE_Int]
    )
    HYPRE_IJVectorInitialize = _bind(
        lib, "HYPRE_IJVectorInitialize", HYPRE_Int,
        [HYPRE_IJVector]
    )
    HYPRE_IJVectorSetValues = _bind(
        lib, "HYPRE_IJVectorSetValues", HYPRE_Int,
        [HYPRE_IJVector, HYPRE_Int, ctypes.POINTER(HYPRE_Int), 
         ctypes.POINTER(HYPRE_Real)]
    )
    HYPRE_IJVectorAssemble = _bind(
        lib, "HYPRE_IJVectorAssemble", HYPRE_Int,
        [HYPRE_IJVector]
    )
    HYPRE_IJVectorGetObject = _bind(
        lib, "HYPRE_IJVectorGetObject", HYPRE_Int,
        [HYPRE_IJVector, ctypes.POINTER(HYPRE_ParVector)]
    )
    HYPRE_IJVectorGetValues = _bind(
        lib, "HYPRE_IJVectorGetValues", HYPRE_Int,
        [HYPRE_IJVector, HYPRE_Int, ctypes.POINTER(HYPRE_Int), 
         ctypes.POINTER(HYPRE_Real)]
    )
    HYPRE_IJVectorDestroy = _bind(
        lib, "HYPRE_IJVectorDestroy", HYPRE_Int,
        [HYPRE_IJVector]
    )

    HYPRE_BoomerAMGCreate = _bind(
        lib, "HYPRE_BoomerAMGCreate", HYPRE_Int,
        [ctypes.POINTER(HYPRE_Solver)]
    )
    HYPRE_BoomerAMGDestroy = _bind(
        lib, "HYPRE_BoomerAMGDestroy", HYPRE_Int,
        [HYPRE_Solver]
    )
    HYPRE_BoomerAMGSetMaxIter = _bind(
        lib, "HYPRE_BoomerAMGSetMaxIter", HYPRE_Int,
        [HYPRE_Solver, HYPRE_Int]
    )
    HYPRE_BoomerAMGSetTol = _bind(
        lib, "HYPRE_BoomerAMGSetTol", HYPRE_Int,
        [HYPRE_Solver, HYPRE_Real]
    )
    HYPRE_BoomerAMGSetPrintLevel = _bind(
        lib, "HYPRE_BoomerAMGSetPrintLevel", HYPRE_Int,
        [HYPRE_Solver, HYPRE_Int]
    )
    HYPRE_BoomerAMGSetStrongThreshold = _bind(
        lib, "HYPRE_BoomerAMGSetStrongThreshold", HYPRE_Int,
        [HYPRE_Solver, HYPRE_Real]
    )
    HYPRE_BoomerAMGSetup = _bind(
        lib, "HYPRE_BoomerAMGSetup", HYPRE_Int,
        [HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector]
    )
    HYPRE_BoomerAMGSolve = _bind(
        lib, "HYPRE_BoomerAMGSolve", HYPRE_Int,
        [HYPRE_Solver, HYPRE_ParCSRMatrix, HYPRE_ParVector, HYPRE_ParVector]
    )
    HYPRE_BoomerAMGGetNumIterations = _bind(
        lib, "HYPRE_BoomerAMGGetNumIterations", HYPRE_Int,
        [HYPRE_Solver, ctypes.POINTER(HYPRE_Int)]
    )
    HYPRE_BoomerAMGGetFinalRelativeResidualNorm = _bind(
        lib, "HYPRE_BoomerAMGGetFinalRelativeResidualNorm", HYPRE_Int,
        [HYPRE_Solver, ctypes.POINTER(HYPRE_Real)]
    )
    HYPRE_BoomerAMGGetCumNnzAP = _bind(
        lib, "HYPRE_BoomerAMGGetCumNnzAP", HYPRE_Int,
        [HYPRE_Solver, ctypes.POINTER(HYPRE_Real)]
    )
    HYPRE_BoomerAMGSetCumNnzAP = _bind(
        lib, "HYPRE_BoomerAMGSetCumNnzAP", HYPRE_Int,
        [HYPRE_Solver, HYPRE_Real]
    )

    # Prepare data arrays
    all_indices = (HYPRE_Int * n)(*map(HYPRE_Int, range(n)))
    b_values = (HYPRE_Real * n)(*map(float, b_vec))
    x0_values = (HYPRE_Real * n)(*([0.0] * n))

    try:
        # Create and fill matrix
        HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, 
                             ctypes.byref(ij_matrix))
        HYPRE_IJMatrixSetObjectType(ij_matrix, HYPRE_Int(HYPRE_PARCSR))
        HYPRE_IJMatrixInitialize(ij_matrix)

        # Fill matrix row-by-row
        indptr = A_csr.indptr
        indices = A_csr.indices
        data = A_csr.data.astype(np.float64, copy=False)

        for row_index in range(n):
            start = int(indptr[row_index])
            end = int(indptr[row_index + 1])
            row_cols = indices[start:end]
            row_vals = data[start:end]
            nnz = int(row_cols.shape[0])
            if nnz == 0:
                continue

            ncols_arr = (HYPRE_Int * 1)(HYPRE_Int(nnz))
            rows_arr = (HYPRE_Int * 1)(HYPRE_Int(row_index))
            cols_arr = (HYPRE_Int * nnz)(*map(int, row_cols))
            vals_arr = (HYPRE_Real * nnz)(*map(float, row_vals))

            HYPRE_IJMatrixSetValues(
                ij_matrix, HYPRE_Int(1), ncols_arr, rows_arr, cols_arr, vals_arr
            )

        HYPRE_IJMatrixAssemble(ij_matrix)
        HYPRE_IJMatrixGetObject(ij_matrix, ctypes.byref(par_matrix))

        # Create vectors
        for vec, name in ((ij_vector_b, "b"), (ij_vector_x, "x")):
            HYPRE_IJVectorCreate(comm, ilower, iupper, ctypes.byref(vec))
            HYPRE_IJVectorSetObjectType(vec, HYPRE_Int(HYPRE_PARCSR))
            HYPRE_IJVectorInitialize(vec)

        HYPRE_IJVectorSetValues(ij_vector_b, HYPRE_Int(n), all_indices, b_values)
        HYPRE_IJVectorSetValues(ij_vector_x, HYPRE_Int(n), all_indices, x0_values)

        for vec in (ij_vector_b, ij_vector_x):
            HYPRE_IJVectorAssemble(vec)

        HYPRE_IJVectorGetObject(ij_vector_b, ctypes.byref(par_vector_b))
        HYPRE_IJVectorGetObject(ij_vector_x, ctypes.byref(par_vector_x))

        # Create and configure BoomerAMG solver
        HYPRE_BoomerAMGCreate(ctypes.byref(solver))
        HYPRE_BoomerAMGSetPrintLevel(solver, HYPRE_Int(0))
        HYPRE_BoomerAMGSetMaxIter(solver, HYPRE_Int(MAX_ITERATIONS))
        HYPRE_BoomerAMGSetTol(solver, HYPRE_Real(tol))
        
        # Set the tunable parameter (analogous to omega in SOR)
        HYPRE_BoomerAMGSetStrongThreshold(solver, HYPRE_Real(strong_threshold))
        
        # Enable computation of cumulative nnz(A) + nnz(P) for work units
        # Must be set to a positive value before setup for HYPRE to compute it
        HYPRE_BoomerAMGSetCumNnzAP(solver, HYPRE_Real(1.0))

        # Setup and solve
        HYPRE_BoomerAMGSetup(solver, par_matrix, par_vector_b, par_vector_x)
        HYPRE_BoomerAMGSolve(solver, par_matrix, par_vector_b, par_vector_x)

        # Get iteration count and hierarchy complexity (the feedback for bandit learning)
        HYPRE_BoomerAMGGetNumIterations(solver, ctypes.byref(iters))
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, ctypes.byref(final_res))
        HYPRE_BoomerAMGGetCumNnzAP(solver, ctypes.byref(cum_nnz_AP))

        # Extract solution
        x_out = (HYPRE_Real * n)()
        HYPRE_IJVectorGetValues(ij_vector_x, HYPRE_Int(n), all_indices, x_out)
        out = np.frombuffer(x_out, dtype=np.float64).copy()

        k = int(iters.value)
        # Normalize by nnz(A_0) to get the complexity ratio as per the formula:
        # MemComp = (sum nnz(A_l) + sum nnz(P_l)) / nnz(A_0)
        nnz_A0 = float(A_csr.nnz)
        complexity = float(cum_nnz_AP.value) / nnz_A0 if nnz_A0 > 0 else 1.0
        return k, complexity, out

    finally:
        # Cleanup
        if solver:
            HYPRE_BoomerAMGDestroy(solver)
        if ij_vector_b:
            HYPRE_IJVectorDestroy(ij_vector_b)
        if ij_vector_x:
            HYPRE_IJVectorDestroy(ij_vector_x)
        if ij_matrix:
            HYPRE_IJMatrixDestroy(ij_matrix)
        _hypre_finalize(lib)
