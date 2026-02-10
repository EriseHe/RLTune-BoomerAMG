# BoomerAMG Bandit Learning Experiment Results

## Overview

This experiment applies the Tsallis-INF bandit algorithm to tune BoomerAMG's `strong_threshold` parameter across a sequence of linear systems, using **Work Units (WU)** as the cost metric instead of simple iteration counts.

## Loss Function

### Original SOR Loss (Learning to Relax paper)
```
Loss = iteration_count
```

### BoomerAMG Loss (This Implementation)
```
Loss = iterations × cum_nnz_AP
```

Where:
- **iterations**: Number of V-cycles to converge to tolerance
- **cum_nnz_AP**: Cumulative nonzeros ratio of the AMG hierarchy

### The cum_nnz_AP Formula

```
                 L              L-1
                ___            ___
                \              \
cum_nnz_AP =    /   nnz(A_ℓ) + /   nnz(P_ℓ)
                ---            ---
                ℓ=0            ℓ=0
            ─────────────────────────────────
                      nnz(A_0)
```

- **A_ℓ**: Operator matrix at level ℓ (A_0 is the fine grid)
- **P_ℓ**: Interpolation operator from level ℓ+1 to ℓ
- **L**: Number of levels in the AMG hierarchy

This metric captures the **total memory/work complexity** of the AMG hierarchy relative to the original problem.

### Why Work Units Instead of Iterations?

For SOR, each iteration has the same cost. For AMG:
- Different `strong_threshold` values create different hierarchies
- Some hierarchies are "cheaper" (fewer levels, less fill-in)
- Some hierarchies are "more expensive" (more levels, more fill-in)

**Example from experiments:**
| θ (strong_threshold) | Typical cum_nnz_AP | Interpretation |
|----------------------|-------------------|----------------|
| 0.10 | ~3.1 - 3.2 | Aggressive coarsening, compact hierarchy |
| 0.25 | ~3.2 - 3.3 | Default for 2D problems |
| 0.50 | ~4.2 - 4.3 | Conservative coarsening, larger hierarchy |
| 0.70 | ~4.2 - 4.3 | More conservative, similar complexity |

Work Units = iterations × cum_nnz_AP properly accounts for this varying cost.

---

## Experiment Setup

### Problem Definition
- **Matrix**: 2D Poisson equation (5-point stencil) via `delsq(numgrid('S', 20))`
- **Size**: 324 × 324 (from 18×18 interior grid)
- **nnz(A)**: 1,548 nonzeros
- **Tolerance**: 1e-8 (relative residual)

### Diagonal Shift Distribution
Each instance solves:
```
(A + c·I) x = b
```
Where `c` is drawn from a Beta distribution:

| Experiment | Beta Parameters | Distribution Shape |
|------------|-----------------|-------------------|
| High Variance | Beta(0.5, 1.5) | Skewed, heavy tails |
| Low Variance | Beta(2.0, 6.0) | More concentrated |

The diagonal shift range: `c ∈ [-0.15, 0.45]`

### Bandit Configuration
- **Algorithm**: Tsallis-INF with time-varying η = 2/√t
- **Action Grid**: θ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
- **Instances (T)**: 500 per trial
- **Trials**: 2

### Fixed Baselines
Compared against fixed thresholds: θ ∈ {0.10, 0.25, 0.40, 0.50, 0.70}

---

## Log File Format

The CSV log files contain one row per instance with the following columns:

| Column | Description |
|--------|-------------|
| `trial` | Trial number (1 or 2) |
| `instance` | Instance number within trial (1-500) |
| `diagonal_shift` | The value of `c` for this instance |
| `bandit_theta` | Threshold selected by Tsallis-INF |
| `bandit_iterations` | V-cycles with bandit's choice |
| `bandit_cum_nnz_AP` | Hierarchy complexity with bandit's choice |
| `bandit_WU` | Work Units = iterations × cum_nnz_AP |
| `best_fixed_theta` | Best fixed threshold for this instance |
| `best_fixed_iterations` | Iterations with best fixed threshold |
| `best_fixed_cum_nnz_AP` | Complexity with best fixed threshold |
| `best_fixed_WU` | Work Units with best fixed threshold |
| `regret` | bandit_WU - best_fixed_WU |
| `theta_X.XX_WU` | Work Units for each fixed threshold |

### Example Log Entry
```csv
trial,instance,diagonal_shift,bandit_theta,bandit_iterations,bandit_cum_nnz_AP,bandit_WU,best_fixed_theta,best_fixed_iterations,best_fixed_cum_nnz_AP,best_fixed_WU,regret
1,1,-0.149,0.8,149,4.21,627.86,0.25,58,3.20,185.88,441.98
1,2,0.170,0.3,11,3.17,34.86,0.1,11,3.14,34.59,0.27
```

**Interpretation of row 1:**
- Instance 1: diagonal shift c = -0.149
- Bandit chose θ = 0.8, took 149 iterations with complexity 4.21 → WU = 627.86
- Best fixed choice was θ = 0.25 with 58 iterations, complexity 3.20 → WU = 185.88
- Regret (extra cost) = 441.98 work units

**Interpretation of row 2:**
- Instance 2: diagonal shift c = 0.170
- Bandit chose θ = 0.3, took 11 iterations with complexity 3.17 → WU = 34.86
- Best fixed was θ = 0.1 with 11 iterations, complexity 3.14 → WU = 34.59
- Regret = 0.27 (nearly optimal)

---

## Plot Description

### Axes
- **X-axis**: Cumulative Total Work Units
- **Y-axis**: Instances Remaining (starts at T, decreases to 0)

### Interpretation
- Lines further **left** are **better** (lower total cost)
- Straight lines indicate consistent per-instance cost
- Curved lines indicate variable per-instance cost (expected for AMG)
- The bandit (black line) should converge toward the best fixed baseline over time

### Why AMG Lines Are Curved (vs SOR Straight Lines)
For SOR, each omega gives roughly constant iterations per instance → straight cumulative line.

For AMG, the hierarchy is rebuilt for each shifted matrix, so:
- `cum_nnz_AP` varies per instance
- Work Units = iterations × cum_nnz_AP has higher variance
- Cumulative sum shows curvature

---

## Files Generated

| File | Description |
|------|-------------|
| `learning_high_variance.png` | Plot for Beta(0.5, 1.5) distribution |
| `learning_low_variance.png` | Plot for Beta(2.0, 6.0) distribution |
| `experiment_log_high_variance.csv` | Detailed log (1000 rows = 500×2 trials) |
| `experiment_log_low_variance.csv` | Detailed log (1000 rows = 500×2 trials) |

---

## Summary Statistics

### High Variance Experiment
- Total instances: 1000
- The bandit explores aggressively early, then converges

### Low Variance Experiment  
- Total instances: 1000
- With more predictable problems, fixed baselines may perform more consistently

---

## How to Reproduce

```bash
# Set HYPRE path
export HYPRE_DIR=/path/to/hypre/install

# Run experiment with logging
cd SetupPhase/learning-to-relax-python
python scripts/learning_amg_with_logs.py
```

Output will be in `plots/BoomerAMG/`.

---

## Key Takeaways

1. **Work Units is a faithful cost metric** for AMG that accounts for hierarchy complexity
2. **Lower strong_threshold (0.1-0.25)** creates cheaper hierarchies but may need more iterations
3. **Higher strong_threshold (0.5-0.7)** creates more expensive hierarchies but may converge faster
4. **The bandit must balance** iteration count vs. hierarchy cost
5. **Variance in AMG cost** comes from both iteration count AND hierarchy structure
