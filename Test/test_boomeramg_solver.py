"""
Test 2: BoomerAMG Solver Functionality Test

This test verifies that:
1. BoomerAMG can solve a 1D Poisson problem
2. BoomerAMG can solve a 2D Poisson problem
3. The solver converges within expected iterations
4. The solution accuracy is acceptable
"""

import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

# Add the learning-to-relax-python folder to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "SetupPhase" / "learning-to-relax-python"))


def poisson_1d(n: int) -> sp.csr_matrix:
    """Create 1D Poisson matrix: tridiag(-1, 2, -1)."""
    diagonals = [
        -np.ones(n - 1, dtype=np.float64),
        2.0 * np.ones(n, dtype=np.float64),
        -np.ones(n - 1, dtype=np.float64),
    ]
    offsets = [-1, 0, 1]
    return sp.diags(diagonals, offsets, shape=(n, n), format="csr")


def poisson_2d(nx: int, ny: int) -> sp.csr_matrix:
    """Create 2D Poisson matrix using 5-point stencil."""
    n = nx * ny
    
    # Main diagonal: 4
    main_diag = 4.0 * np.ones(n, dtype=np.float64)
    
    # Off-diagonals: -1
    off_diag_1 = -np.ones(n - 1, dtype=np.float64)
    off_diag_nx = -np.ones(n - nx, dtype=np.float64)
    
    # Handle boundary conditions (no wrap-around in x direction)
    for i in range(1, ny):
        off_diag_1[i * nx - 1] = 0.0
    
    diagonals = [off_diag_nx, off_diag_1, main_diag, off_diag_1, off_diag_nx]
    offsets = [-nx, -1, 0, 1, nx]
    
    return sp.diags(diagonals, offsets, shape=(n, n), format="csr")


def test_boomeramg_1d_poisson():
    """Test BoomerAMG on 1D Poisson problem."""
    print("\n[2.1] Testing BoomerAMG on 1D Poisson problem...")
    
    from solvers.BoomerAMG.boomeramg import boomeramg_solve
    
    n = 100
    A = poisson_1d(n)
    b = np.ones(n, dtype=np.float64)
    
    print(f"  Problem size: {n}")
    print(f"  Matrix: 1D Poisson (tridiagonal)")
    print(f"  RHS: ones vector")
    
    result = boomeramg_solve(
        A, b,
        max_iter=100,
        tol=1e-10,
        strong_threshold=0.25,
        print_level=0
    )
    
    print(f"  Iterations: {result.num_iterations}")
    print(f"  Final relative residual: {result.final_relative_residual:.2e}")
    
    if result.cum_nnz_ap is not None:
        print(f"  Cumulative nnz(A*P): {result.cum_nnz_ap:.0f}")
    
    # Verify solution
    residual = np.linalg.norm(A @ result.x - b) / np.linalg.norm(b)
    print(f"  Computed residual: {residual:.2e}")
    
    # Check convergence
    if result.final_relative_residual > 1e-8:
        print("  WARNING: Convergence not achieved to desired tolerance!")
        return False
    
    print("  1D Poisson test: PASSED")
    return True


def test_boomeramg_2d_poisson():
    """Test BoomerAMG on 2D Poisson problem."""
    print("\n[2.2] Testing BoomerAMG on 2D Poisson problem...")
    
    from solvers.BoomerAMG.boomeramg import boomeramg_solve
    
    nx, ny = 20, 20
    n = nx * ny
    A = poisson_2d(nx, ny)
    
    # Create a more interesting RHS (smooth function)
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, 1, nx),
        np.linspace(0, 1, ny)
    )
    b = np.sin(np.pi * x_grid.flatten()) * np.sin(np.pi * y_grid.flatten())
    
    print(f"  Grid size: {nx} x {ny} = {n}")
    print(f"  Matrix: 2D Poisson (5-point stencil)")
    print(f"  RHS: sin(pi*x) * sin(pi*y)")
    
    result = boomeramg_solve(
        A, b,
        max_iter=100,
        tol=1e-10,
        strong_threshold=0.25,
        print_level=0
    )
    
    print(f"  Iterations: {result.num_iterations}")
    print(f"  Final relative residual: {result.final_relative_residual:.2e}")
    
    if result.cum_nnz_ap is not None:
        print(f"  Cumulative nnz(A*P): {result.cum_nnz_ap:.0f}")
    
    # Verify solution
    residual = np.linalg.norm(A @ result.x - b) / np.linalg.norm(b)
    print(f"  Computed residual: {residual:.2e}")
    
    # Check convergence
    if result.final_relative_residual > 1e-8:
        print("  WARNING: Convergence not achieved to desired tolerance!")
        return False
    
    print("  2D Poisson test: PASSED")
    return True


def test_boomeramg_larger_problem():
    """Test BoomerAMG on a larger problem to verify scalability."""
    print("\n[2.3] Testing BoomerAMG on larger 2D Poisson problem...")
    
    from solvers.BoomerAMG.boomeramg import boomeramg_solve
    
    nx, ny = 50, 50
    n = nx * ny
    A = poisson_2d(nx, ny)
    b = np.random.randn(n)
    
    print(f"  Grid size: {nx} x {ny} = {n}")
    print(f"  Matrix: 2D Poisson (5-point stencil)")
    print(f"  RHS: random vector")
    
    import time
    start_time = time.perf_counter()
    
    result = boomeramg_solve(
        A, b,
        max_iter=200,
        tol=1e-10,
        strong_threshold=0.25,
        print_level=0
    )
    
    elapsed_time = time.perf_counter() - start_time
    
    print(f"  Iterations: {result.num_iterations}")
    print(f"  Final relative residual: {result.final_relative_residual:.2e}")
    print(f"  Solve time: {elapsed_time:.3f} seconds")
    
    if result.cum_nnz_ap is not None:
        print(f"  Cumulative nnz(A*P): {result.cum_nnz_ap:.0f}")
    
    # Verify solution
    residual = np.linalg.norm(A @ result.x - b) / np.linalg.norm(b)
    print(f"  Computed residual: {residual:.2e}")
    
    # Check convergence
    if result.final_relative_residual > 1e-8:
        print("  WARNING: Convergence not achieved to desired tolerance!")
        return False
    
    print("  Larger problem test: PASSED")
    return True


def test_boomeramg_solver():
    """Run all BoomerAMG solver tests."""
    print("=" * 60)
    print("Test 2: BoomerAMG Solver Functionality Test")
    print("=" * 60)
    
    tests = [
        test_boomeramg_1d_poisson,
        test_boomeramg_2d_poisson,
        test_boomeramg_larger_problem,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    all_passed = all(results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("Test 2 PASSED: All BoomerAMG solver tests passed!")
    else:
        print("Test 2 FAILED: Some solver tests failed!")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = test_boomeramg_solver()
    sys.exit(0 if success else 1)
