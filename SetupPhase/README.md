# SetupPhase

This folder contains:
- A copy of the **Learning to Relax** paper (`Learning to Relax.md`)
- The authors’ original MATLAB repo (`learning-to-relax-original/`)
- A Python translation + BoomerAMG adaptation (`learning-to-relax-python/`)

This README documents the **exact commands** I used to run the experiments you asked for (BoomerAMG + a small SOR run), without modifying your solver/learner implementations.

## Environment notes (important)

### BoomerAMG needs a loadable `libHYPRE`
Your Python BoomerAMG wrapper (`learning-to-relax-python/solvers/BoomerAMG/boomeramg.py`) loads HYPRE via:
- `HYPRE_LIBHYPRE` (recommended): full path to `libHYPRE.*`
- or `HYPRE_DIR` / `HYPRE_PREFIX`

In this repo, an existing macOS dylib is at:
- `SolvePhase/hypre/src/lib/libHYPRE.dylib`

### MPI-enabled HYPRE: pass an initialized `MPI_COMM_WORLD`
The provided `libHYPRE.dylib` is linked against OpenMPI (MPI-enabled). To avoid “MPI_Comm_size called before MPI_INIT”, I ran BoomerAMG experiments by:
1) importing `mpi4py` (which initializes MPI)
2) setting `HYPRE_MPI_COMM_WORLD` to `MPI.COMM_WORLD.handle`

This avoids code changes in `boomeramg.py`.

## Commands I ran

All commands below assume you start at repo root:
`/Users/erisehe/Documents/GitHub/RLTune-BoomerAMG`

### 1) BoomerAMG Tsallis-INF (T=5000, trials=1)

This reproduces the BoomerAMG learning curves using **Work Units** (WU = iterations × cum_nnz_AP) and runs both:
- High variance offsets: `Beta(0.5, 1.5)`
- Low variance offsets: `Beta(2.0, 6.0)`

It writes plots under:
`SetupPhase/learning-to-relax-python/plots/BoomerAMG/`

```bash
python - <<'PY'
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye
from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init()

root = os.getcwd()
py_root = os.path.join(root, 'SetupPhase', 'learning-to-relax-python')
os.environ['HYPRE_LIBHYPRE'] = os.path.join(root, 'SolvePhase', 'hypre', 'src', 'lib', 'libHYPRE.dylib')
os.environ['HYPRE_MPI_COMM_WORLD'] = str(MPI.COMM_WORLD.handle)

sys.path.insert(0, py_root)
from learners import TsallisINF_AMG
from solvers.BoomerAMG import boomeramg
from utils import truncated_normal, delsq, numgrid

np.random.seed(0)

out_dir = os.path.join(py_root, 'plots', 'BoomerAMG')
os.makedirs(out_dir, exist_ok=True)

A = delsq(numgrid('S', 20))
n = A.shape[0]
epsilon = 1e-8
T = 5000
thresholds = np.array([0.1, 0.25, 0.4, 0.5, 0.7])
threshold_grid = np.linspace(0.1, 0.9, 9)

def run(beta_a, beta_b, tag):
    threshold_costs = np.zeros((T, len(thresholds)))
    tinf_costs = np.zeros(T)
    tinf = TsallisINF_AMG(threshold_grid, T)

    start = time.time()
    for t in range(T):
        c = -0.15 + 0.6 * np.random.beta(beta_a, beta_b)
        At = A + c * speye(n, format='csr')
        bt = truncated_normal(n)

        theta_pred = float(tinf.predict())
        k, comp, _ = boomeramg(At, bt, np.zeros(n), theta_pred, epsilon)
        tinf_costs[t] = k * comp
        tinf.update(k, comp)

        for i, theta in enumerate(thresholds):
            kf, compf, _ = boomeramg(At, bt, np.zeros(n), float(theta), epsilon)
            threshold_costs[t, i] = kf * compf

        if (t + 1) % 100 == 0 or (t + 1) == T:
            print(f'{tag}: {t+1}/{T} ({(time.time()-start)/60:.1f} min)')

    plt.figure(figsize=(8, 6))
    for i, theta in enumerate(thresholds):
        plt.plot(np.cumsum(threshold_costs[:, i]), T - np.arange(1, T + 1), linestyle='--', linewidth=2)
    plt.plot(np.cumsum(tinf_costs), T - np.arange(1, T + 1), color='black', linewidth=3)
    plt.legend([f'θ={t:.2f}' for t in thresholds] + ['Tsallis-INF (BoomerAMG)'], fontsize=11, loc='upper right')
    plt.xlabel('Total Work Units (WU)')
    plt.ylabel('Instances Remaining')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'learning_{tag}_T{T}_trial1.png')
    plt.savefig(out_path, dpi=256)
    plt.close()
    print('Saved:', out_path)

run(0.5, 1.5, 'high_variance')
run(2.0, 6.0, 'low_variance')
PY
```

### 2) BoomerAMG with a forced “bad” first action (T=5000, trials=1)

You asked to force:
- High variance: first θ = 0.10
- Low variance: first θ = 0.70

This uses the same setup as above but overrides the bandit’s *first* action only.

```bash
python - <<'PY'
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye
from mpi4py import MPI

if not MPI.Is_initialized():
    MPI.Init()

root = os.getcwd()
py_root = os.path.join(root, 'SetupPhase', 'learning-to-relax-python')
os.environ['HYPRE_LIBHYPRE'] = os.path.join(root, 'SolvePhase', 'hypre', 'src', 'lib', 'libHYPRE.dylib')
os.environ['HYPRE_MPI_COMM_WORLD'] = str(MPI.COMM_WORLD.handle)

sys.path.insert(0, py_root)
from learners import TsallisINF_AMG
from solvers.BoomerAMG import boomeramg
from utils import truncated_normal, delsq, numgrid

np.random.seed(0)

out_dir = os.path.join(py_root, 'plots', 'BoomerAMG')
os.makedirs(out_dir, exist_ok=True)

A = delsq(numgrid('S', 20))
n = A.shape[0]
epsilon = 1e-8
T = 5000
thresholds = np.array([0.1, 0.25, 0.4, 0.5, 0.7])
threshold_grid = np.linspace(0.1, 0.9, 9)

def force_first_action(tinf, theta_init):
    _ = tinf.predict()
    idx = int(np.argmin(np.abs(tinf.grid - theta_init)))
    tinf.index = idx
    tinf.prob = 1.0
    if tinf.t <= len(tinf.actions):
        tinf.actions[tinf.t - 1] = idx
    return float(theta_init)

def run(beta_a, beta_b, tag, theta_init):
    tinf = TsallisINF_AMG(threshold_grid, T)
    tinf_costs = np.zeros(T)
    theta_trace = np.zeros(T)

    start = time.time()
    for t in range(T):
        c = -0.15 + 0.6 * np.random.beta(beta_a, beta_b)
        At = A + c * speye(n, format='csr')
        bt = truncated_normal(n)

        theta = force_first_action(tinf, theta_init) if t == 0 else float(tinf.predict())
        k, comp, _ = boomeramg(At, bt, np.zeros(n), theta, epsilon)
        tinf_costs[t] = k * comp
        theta_trace[t] = theta
        tinf.update(k, comp)

        if (t + 1) % 100 == 0 or (t + 1) == T:
            print(f'{tag}: {t+1}/{T} ({(time.time()-start)/60:.1f} min)')

    # learning curve
    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(tinf_costs), T - np.arange(1, T + 1), color='black', linewidth=3)
    plt.xlabel('Total Work Units (WU)')
    plt.ylabel('Instances Remaining')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'learning_{tag}_T{T}_trial1_init{theta_init:.2f}.png')
    plt.savefig(out_path, dpi=256)
    plt.close()

    # theta trace
    plt.figure(figsize=(9, 3))
    plt.plot(theta_trace, linewidth=0.8)
    plt.ylim(0.0, 1.0)
    plt.tight_layout()
    trace_path = os.path.join(out_dir, f'theta_trace_{tag}_T{T}_trial1_init{theta_init:.2f}.png')
    plt.savefig(trace_path, dpi=200)
    plt.close()

    print('Saved:', out_path)
    print('Saved:', trace_path)

run(0.5, 1.5, 'high_variance', theta_init=0.10)
run(2.0, 6.0, 'low_variance', theta_init=0.70)
PY
```

### 3) SOR Tsallis-INF sanity run (small matrix, T=5000, trials=1)

This is the same “Learning to Relax” SOR setup but with a smaller Laplacian grid:
`numgrid('S', 10)` (n=64), so it finishes quickly.

Plots are written under:
`SetupPhase/learning-to-relax-python/plots/SOR/`

```bash
python - <<'PY'
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = os.getcwd()
py_root = os.path.join(root, 'SetupPhase', 'learning-to-relax-python')
sys.path.insert(0, py_root)

from learners import TsallisINF
from solvers import sor
from utils import truncated_normal, delsq, numgrid

np.random.seed(0)

out_dir = os.path.join(py_root, 'plots', 'SOR')
os.makedirs(out_dir, exist_ok=True)

S = 10
A0 = delsq(numgrid('S', S)).toarray()
n = A0.shape[0]
epsilon = 1e-8
T = 5000
omegas = np.linspace(1.0, 1.8, 5)
bandit_grid = np.linspace(1.0, 1.95, 20)

def run(beta_a, beta_b, tag):
    tinf = TsallisINF(bandit_grid, T)
    costs = np.zeros(T)
    trace = np.zeros(T)
    start = time.time()
    for t in range(T):
        c = -0.15 + 0.6 * np.random.beta(beta_a, beta_b)
        At = A0.copy()
        At[np.diag_indices_from(At)] += c
        bt = truncated_normal(n)
        w = float(tinf.predict())
        k, _ = sor(At, bt, np.zeros(n), w, epsilon)
        costs[t] = k
        trace[t] = w
        tinf.update(k)
        if (t + 1) % 200 == 0 or (t + 1) == T:
            print(f'{tag}: {t+1}/{T} ({(time.time()-start)/60:.1f} min)')

    plt.figure(figsize=(8, 6))
    plt.plot(np.cumsum(costs), T - np.arange(1, T + 1), color='black', linewidth=3)
    plt.xlabel('Total iterations')
    plt.ylabel('Instances Remaining')
    plt.tight_layout()
    out_path = os.path.join(out_dir, f'learning_{tag}_T{T}_trial1_S{S}.png')
    plt.savefig(out_path, dpi=256)
    plt.close()

    plt.figure(figsize=(9, 3))
    plt.plot(trace, linewidth=0.8)
    plt.ylim(0.9, 2.0)
    plt.tight_layout()
    trace_path = os.path.join(out_dir, f'omega_trace_{tag}_T{T}_trial1_S{S}.png')
    plt.savefig(trace_path, dpi=200)
    plt.close()

    print('Saved:', out_path)
    print('Saved:', trace_path)

run(0.5, 1.5, 'high_variance')
run(2.0, 6.0, 'low_variance')
PY
```
