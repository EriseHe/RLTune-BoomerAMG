# MATLAB to Python Translation Notes

This document describes the translation of the "Learning to Relax" MATLAB repository to Python.

## File Mapping

| MATLAB File | Python File | Description |
|-------------|-------------|-------------|
| `learners/TsallisINF.m` | `learners/TsallisINF.py` | Tsallis-INF bandit algorithm |
| `learners/TsallisINFCB.m` | `learners/TsallisINFCB.py` | Contextual bandit with discretized contexts |
| `learners/ChebCB.m` | `learners/ChebCB.py` | ChebCB with Chebyshev regression |
| `solvers/sor.m` | `solvers/sor.py` | SOR solver |
| `solvers/omega_opt.m` | `solvers/omega_opt.py` | Optimal omega computation |
| `solvers/omega_grid.m` | `solvers/omega_grid.py` | Omega grid generation |
| `solvers/rho_jacobi.m` | `solvers/rho_jacobi.py` | Jacobi spectral radius |
| `solvers/cgbound.m` | `solvers/cgbound.py` | CG iteration bound |
| `solvers/energy_norm.m` | `solvers/energy_norm.py` | Energy norm bound |
| `solvers/ssor_pcg.m` | `solvers/ssor_pcg.py` | SSOR-preconditioned CG |
| `utils/Heat2D.m` | `utils/Heat2D.py` | 2D heat equation solver |
| `utils/truncated_normal.m` | `utils/truncated_normal.py` | Truncated Gaussian sampling |
| `utils/bump.m` | `utils/bump.py` | Bump function |
| `utils/golden_section.m` | `utils/golden_section.py` | Golden section search |
| `scripts/learning.m` | `scripts/learning.py` | Learning experiment |
| `scripts/contextual.m` | `scripts/contextual.py` | Contextual bandit experiment |
| `scripts/asymptotic.m` | `scripts/asymptotic.py` | Asymptotic bounds |
| `scripts/cg.m` | `scripts/cg.py` | CG experiments |
| `scripts/comparators.m` | `scripts/comparators.py` | Comparator experiments |
| `scripts/degenerate.m` | `scripts/degenerate.py` | Degenerate vector experiments |
| `scripts/h2d.m` | `scripts/h2d.py` | Heat equation simulation |

## Helper Files Added

| File | Description |
|------|-------------|
| `utils/delsq.py` | Provides `delsq()` and `numgrid()` functions equivalent to MATLAB's |

## Translation Caveats

### Indexing
- MATLAB uses 1-based indexing; Python uses 0-based indexing
- All loop indices and array accesses have been adjusted accordingly

### Random Number Generation
- MATLAB's RNG differs from NumPy's RNG
- Results will not be numerically identical but should be statistically equivalent
- Same distributions are used: `betarnd` → `np.random.beta`, `normrnd` → `np.random.randn`

### Linear Solves
- `M\r` (backslash) is translated case-by-case:
  - Lower triangular `M` → `scipy.linalg.solve_triangular(M, r, lower=True)`
  - General sparse → `scipy.sparse.linalg.spsolve(M, r)`
  - General dense → `np.linalg.solve(M, r)`

### Preconditioned CG
- MATLAB `pcg(A, b, tol, maxiter, M1, M2, x0)` with preconditioner `M = M1 * M2`
- Python uses `scipy.sparse.linalg.cg` with a custom `LinearOperator` for the preconditioner
- Preconditioner application: `M \ r = M2 \ (M1 \ r)`

### Chebyshev Polynomials
- MATLAB uses `chebyshevT(j, x)` from Symbolic Toolbox
- Python computes coefficients using exact recurrence: T₀=1, T₁=x, Tₙ₊₁=2xTₙ-Tₙ₋₁
- Polynomial evaluation uses `np.polyval` with coefficients in descending order

### Parallel Execution
- MATLAB `parfor` loops are translated to sequential `for` loops
- This ensures deterministic, reproducible results

### Constrained Least Squares
- MATLAB `lsqlin` → `scipy.optimize.lsq_linear`

## Running the Scripts

```bash
cd d:\Github\RLTune-BoomerAMG\SetupPhase\learning-to-relax-python

# Install dependencies
pip install -r requirements.txt

# Run experiments
python scripts/learning.py
python scripts/contextual.py
python scripts/asymptotic.py
python scripts/cg.py
python scripts/comparators.py
python scripts/degenerate.py
python scripts/h2d.py
```

## Expected Outputs

Each script generates plots in `scripts/plots/`:
- `learning.py` → `learning_high_variance.png`, `learning_low_variance.png`
- `contextual.py` → `contextual_high_variance.png`, `contextual_low_variance.png`
- `asymptotic.py` → `bound_comparison.png`, `asymptocity.png`, `tau_beta.png`
- `cg.py` → `cgbound-*.png`
- `comparators.py` → `low_variance.png`, `high_variance.png`
- `degenerate.py` → `degenerate.png`
- `h2d.py` → `iterations.png`

## Verification

To verify correctness:
1. Compare plot shapes qualitatively with MATLAB outputs
2. Check that learning curves show expected convergence behavior
3. Verify TsallisINF converges toward best fixed omega
4. Verify contextual methods (ChebCB, TsallisINFCB) improve over non-contextual

Note: Due to RNG differences, exact numerical reproduction is not expected.
