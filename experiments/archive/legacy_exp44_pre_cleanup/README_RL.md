## RL BoomerAMG Solve-Phase: Code/Parameter Walkthrough

This README documents **how training and evaluation run** in this repo, how
grid/RHS/parameters are set, how actions map to BoomerAMG controls, and which
HYPRE functions are called in C. It also points to the exact files that
implement each step.

---

### 1) High-level flow

**Training (Python)**
1. `train_ppo.py` reads environment variables (grid, RHS, ranges, penalties).
2. It creates `BoomerAMGRelaxEnv` from `amg_gym_env.py`.
3. PPO (or custom/LSTM) policy is trained to choose AMG cycle parameters
   based on the observation.
4. The policy outputs an action each cycle; the environment maps it to
   `(w, sweeps_down, sweeps_up)` and calls into C.
5. C executes one BoomerAMG V-cycle and returns residual/time.

**Evaluation (Python)**
1. `eval_default_vs_rl.py` creates the same env, loads `vecnormalize_gen.pkl`
   and the trained model.
2. It compares baseline `(w=1, sweeps=1,1)` vs RL policy.
3. It optionally runs a grid sweep (n=10..100) and plots RL actions/residuals.

**C backend**
1. `amg_env.c` builds the matrix A (difconv or Laplacian).
2. It builds RHS and sets up BoomerAMG.
3. Each `amg_env_step()` runs **one** BoomerAMG V-cycle.

---

### 2) Key files

- `hypre/src/test/train_ppo.py`: training driver
- `hypre/src/test/eval_default_vs_rl.py`: evaluation driver
- `hypre/src/test/amg_gym_env.py`: Python Gym env and action mapping
- `hypre/src/test/amg_env.c`: C backend (matrix, RHS, BoomerAMG calls)
- `hypre/src/test/Makefile`: builds `libamg_env.dylib` / `libamg_env.so`

---

### 2.1) How scripts link to each other

**Python stack**
- `train_ppo.py`
  - Reads env vars → builds `BoomerAMGRelaxEnv`.
  - Trains PPO policy and saves `ppo_boomeramg_gen` + `vecnormalize_gen.pkl`.
  - Calls into `amg_gym_env.py` to create the environment and step it.
- `eval_default_vs_rl.py`
  - Loads `ppo_boomeramg_gen` and `vecnormalize_gen.pkl`.
  - Creates the same `BoomerAMGRelaxEnv` to run baseline vs RL.
  - Uses the same `amg_gym_env.py` env wrapper and calls its `step()`.
- `amg_gym_env.py`
  - Loads `libamg_env.dylib` / `libamg_env.so` using `ctypes`.
  - Binds C functions `amg_env_create`, `amg_env_step`, `amg_env_destroy`.
  - Converts RL actions → `(w, sweeps_down, sweeps_up)` and forwards to C.

**C stack**
- `amg_env.c`
  - Implements the C API used by Python:
    - `amg_env_create` (builds matrix, RHS, sets up BoomerAMG)
    - `amg_env_step` (one V-cycle, returns residual/time)
    - `amg_env_destroy`
  - Calls HYPRE BoomerAMG APIs.
- `amg_cycle.c`
  - Provides Laplacian builder helpers used by `amg_env.c`
    (for stencil types 7/27).
- `Makefile`
  - Compiles `amg_env.c` + `amg_cycle.c` into `libamg_env.*`.

---

### 3) Grid control (training and eval)

**Env variables (read in `train_ppo.py` and `eval_default_vs_rl.py`)**
- `RANDOMIZE_GRID=1|0`
  - If `1`, grid size is sampled uniformly from `[GRID_MIN, GRID_MAX]`.
  - If `0`, grid is fixed from `GRID_SIZES` or `GRID_RANGE`.
- `GRID_MIN`, `GRID_MAX`: inclusive range for `n` (cubic grids).
- `GRID_SIZES="60,60,60;80,80,80"`: explicit list of grids.
- `GRID_RANGE="10,100,10"`: range for grids `(start, stop, step)` -> 10..100 by 10.

**Where it happens**
- `train_ppo.py`: `make_env()` reads env vars and passes to `BoomerAMGRelaxEnv`.
- `amg_gym_env.py`: `reset()` selects a grid:
  - If `randomize_grid=True` → sample `n` uniform `[grid_min, grid_max]`.
  - If fixed → choose from `grid_choices`.

---

### 4) RHS generation

**Env variables**
- `RANDOMIZE_B=1|0`
  - `1` → RHS is random Gaussian per episode.
  - `0` → RHS uses a fixed seed (`FIXED_RHS_SEED`) or is constant if `FIXED_RHS_TYPE=0`.
- `FIXED_RHS_TYPE=1` (default): random Gaussian RHS using seed.
- `FIXED_RHS_SEED=123456789` (default).

**Where it happens**
- `amg_gym_env.py`: passes `fixed_rhs_type` and `fixed_rhs_seed` to C.
- `amg_env.c`: `rhs_type=1` → RHS generated using Box-Muller Gaussian per row.

**C code**
```
// amg_env.c
if (rhs_type == 0) { b = ones }
else { Box-Muller Gaussian with seed ^ row index }
```

---

### 5) Matrix A (DifConv diffusion operator)

We use **DifConv** (diffusion-only) when `FIXED_STENCIL=0` (default).

**Env variables**
- `DIFCONV_C_RANGE=lo,hi` (default `1,1000`): samples `cx,cy,cz` uniformly.
- `DIFCONV_A=0,0,0`: convection set to zero → pure diffusion.
- `DIFCONV_ATYPE=0`: forward scheme (not used if convection zero).

**Where it happens**
- `amg_gym_env.py`:
  - If `randomize_A=True`, samples `cx,cy,cz` each episode.
  - Maps difconv params into C slots: `(k,c,a0,a1,a2,a3)` as:
    - `k=cx`, `c=cy`, `a0=cz`, `a1=ax`, `a2=ay`, `a3=az`
  - `ax,ay,az` default to `0`.
- `amg_env.c`:
  - `stencil_type==0` uses `build_difconv_matrix()` and `GenerateDifConv()`.
  - This creates a **7‑point diffusion stencil** on a structured grid.

---

### 6) Action space and mapping (w + sweeps)

**Env variables**
- `W_CENTER`, `W_SCALE` → defines `w ∈ [W_CENTER−W_SCALE, W_CENTER+W_SCALE]`.
- `SWEEPS_MIN`, `SWEEPS_MAX` → integer sweep bounds.
- `W_ONLY=1` → action space is **1-D** (only `w`).
- `W_INIT`, `SWEEPS_INIT` → first-cycle override for initial `w` / sweeps.

**Mapping (in `amg_gym_env.py`)**
- Action `a ∈ [-1,1]^d`
  - `w = W_CENTER + W_SCALE * a_w` (clipped).
  - `sweeps_down = round(center + half * a_d)` (clipped to [min,max]).
  - `sweeps_up = round(center + half * a_u)` (clipped).
- If `W_ONLY=1`:
  - Sweeps are fixed to `sweeps_default`.
  - Action is `[a_w]` only.
  - This **changes the MLP output dimension**: action space becomes 1-D
    instead of 3-D. The policy network still uses the same observation
    (10-dim), but its output layer size is reduced.
  - Code: `amg_gym_env.py` action space block, which drives SB3 policy output.

**First-cycle init (optional)**
- If `W_INIT` or `SWEEPS_INIT` is set, **cycle 0** uses those values only.

---

### 7) Reward and termination

**Key env variables**
- `REWARD_MODE` (default 0): reward form.
- `REWARD_ALPHA` (default 1.3): scaling.
- `CYCLE_PENALTY`, `SWEEP_PENALTY`: optional regularizers.
- `TERM_BONUS` (default +2), `TRUNC_PENALTY` (default +2).

**Reward (mode 0)**
```
reward = REWARD_ALPHA * (log(r_prev) - log(r_cur)) / wall_time
```
Then apply penalties and terminal bonuses.

**Episode end**
- terminates if `r <= tol` (default `1e-8`) or `cycles >= max_cycles` (default `20`).

---

### 8) C backend and HYPRE calls

**C backend file:** `amg_env.c`

**Key calls in `amg_env_create()`**
- `HYPRE_BoomerAMGCreate`
- `HYPRE_BoomerAMGSetRelaxType`
- `HYPRE_BoomerAMGSetCycleType` (env var `AMG_CYCLE_TYPE`, default 1=V)
- `HYPRE_BoomerAMGSetTol(0.0)` → ensures one cycle per call
- `HYPRE_BoomerAMGSetMaxIter(1)` → one V-cycle per step
- `HYPRE_BoomerAMGSetup(A, b, x)`

**Per-step call (`amg_env_step`)**
- `HYPRE_BoomerAMGSetRelaxWt(w)`
- `HYPRE_BoomerAMGSetCycleNumSweeps(sweeps_down, 1)`
- `HYPRE_BoomerAMGSetCycleNumSweeps(sweeps_up, 2)`
- `HYPRE_BoomerAMGSolve(A, b, x)` → one V-cycle
- compute residual norm and return `r, dt`

---

### 9) VecNormalize (why and how)

`VecNormalize` is a Stable-Baselines3 wrapper that **normalizes observations**
(and optionally rewards) using running mean/variance.

- **Where used in training**: `train_ppo.py` wraps the env with
  `VecNormalize(norm_obs=True, norm_reward=False)`.
- **What it does**: scales the 10‑dim observation so the policy sees
  approximately zero‑mean/unit‑variance inputs.
- **Why**: stabilizes PPO training when raw features vary widely
  (residual ratios, grid size, coefficients).
- **Saved stats**: `vecnormalize_gen.pkl` is saved at the end of training.
- **Eval**: `eval_default_vs_rl.py` loads `vecnormalize_gen.pkl` and uses
  `vec_norm.normalize_obs(obs)` before calling `policy.predict(...)`.

### 10) Training configuration details

**Default PPO settings (in `train_ppo.py`)**
- `n_steps=16`, `batch_size=16`, `gamma=0.99`, `clip_range=0.2`,
  `learning_rate=3e-4`, `n_epochs=10`, `gae_lambda=0.95`.
- `ENT_COEF` controls exploration.
- `VecNormalize` used for obs; reward normalization is off.

**Outputs**
- `ppo_boomeramg_gen.zip`
- `vecnormalize_gen.pkl`
- `logs/train_steps.csv` (step-by-step logging)

---

### 11) Evaluation details

**Baseline vs RL**
- Baseline uses `w=1`, `sweeps=1,1`.
- RL uses the trained policy (MLP/LSTM/custom).
- Stats reported: mean/median time, cycles, overhead, inference time.

**How eval actually runs BoomerAMG**
1. `eval_default_vs_rl.py` builds a `BoomerAMGRelaxEnv` (same as training).
2. For each test case, it calls `run_episode(...)`, which loops until
   `done` (tol reached or max cycles).
3. Each loop:
   - Policy outputs an action (or baseline fixed action).
   - The env maps it to `(w, sweeps_down, sweeps_up)` and calls
     `amg_env_step(...)` in C.
4. `amg_env_step(...)` updates BoomerAMG parameters and runs exactly **one**
   V-cycle using:
   - `HYPRE_BoomerAMGSetRelaxWt`
   - `HYPRE_BoomerAMGSetCycleNumSweeps`
   - `HYPRE_BoomerAMGSolve`
5. Residual and wall time are returned to Python and logged.

**Grid sweep**
- `PLOT_GRID_SWEEP=1` runs fixed n = `GRID_MIN .. GRID_MAX` in steps of `PLOT_GRID_STEP`.
- `PLOT_SWEEP_INSTANCES` controls how many random instances per n.
- Plots saved to `hypre/src/test/pltos`.

---

### 12) Example commands (as used)

**Train on fixed n=100**
```
SWEEPS_MIN=1 SWEEPS_MAX=1 \
W_CENTER=1.25 W_SCALE=0.75 \
RANDOMIZE_GRID=0 GRID_SIZES="100,100,100" \
RANDOMIZE_A=1 DIFCONV_C_RANGE=1,1000 \
RANDOMIZE_B=1 \
TOTAL_TIMESTEPS=200000 \
python train_ppo.py
```

**Evaluate on grids 10..100, 10 instances each**
```
MODEL_TYPE=mlp \
SWEEPS_MIN=1 SWEEPS_MAX=1 \
W_CENTER=1.25 W_SCALE=0.75 \
RANDOMIZE_GRID=1 GRID_MIN=10 GRID_MAX=100 \
RANDOMIZE_A=1 DIFCONV_C_RANGE=1,1000 \
RANDOMIZE_B=1 \
EVAL_SEED_COUNT=3 EVAL_A_INSTANCES=20 \
PLOT_GRID_SWEEP=1 PLOT_GRID_STEP=10 \
PLOT_SWEEP_INSTANCES=10 \
python eval_default_vs_rl.py
```

---

### 13) Build libamg_env (macOS/Linux)

```
cd hypre/src/test
make libamg_env.dylib    # macOS
make libamg_env.so       # Linux
```

If Linux build fails with missing symbols, confirm:
```
ldd libamg_env.so | grep "not found"
export LD_LIBRARY_PATH=$HYPRE_BUILD_DIR/lib:$LD_LIBRARY_PATH
```
