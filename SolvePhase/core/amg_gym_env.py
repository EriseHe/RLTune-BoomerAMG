# amg_gym_env.py
import os, math
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time

from problems.amg import build_matrix_kwargs_difconv
from problems.cases import generate_matrix_coeffs, generate_rhs_seed
from hypre.bindings import (
    AMGNativeError,
    AMG_RUNTIME_LIBRARY,
    SolveStatus,
    create_env,
)


DEFAULT_AMG_RUNTIME_LIBRARY = AMG_RUNTIME_LIBRARY


def build_policy_obs(*, r, r_prev, cycle, max_cycles, coeff_triplet, grid_triplet, last_w):
    eps = 1e-30
    r = float(r)
    r_prev = float(r_prev)

    ratio = (r + eps) / (r_prev + eps)
    ratio = float(np.clip(ratio, 1e-30, 1e30))
    log_ratio = math.log(ratio)

    dr = r_prev - r
    sgn = 1.0 if dr > 0 else (-1.0 if dr < 0 else 0.0)
    cycle_frac = float(cycle) / float(max_cycles)

    s1, s2, s3 = coeff_triplet
    nx_norm, ny_norm, nz_norm = grid_triplet

    obs = np.array(
        [
            log_ratio,
            sgn,
            cycle_frac,
            float(s1),
            float(s2),
            float(s3),
            float(nx_norm),
            float(ny_norm),
            float(nz_norm),
            float(last_w),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)


def decode_policy_action(action, *, w_only, w_center, w_scale, sweeps_min, sweeps_max,
                         cycle, last_w, w_init=None, sweeps_init=None, w_smooth_alpha=0.0):
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    a_w = float(np.clip(action[0], -1.0, 1.0))
    if w_only or action.shape[0] == 1:
        a_d = 0.0
        a_u = 0.0
    else:
        a_d = float(np.clip(action[1], -1.0, 1.0))
        a_u = float(np.clip(action[2], -1.0, 1.0))

    w = float(np.clip(
        w_center + w_scale * a_w,
        w_center - w_scale,
        w_center + w_scale
    ))
    if w_smooth_alpha > 0.0:
        alpha_s = float(np.clip(w_smooth_alpha, 0.0, 1.0))
        w = alpha_s * float(last_w) + (1.0 - alpha_s) * w

    sweeps_center = 0.5 * (int(sweeps_min) + int(sweeps_max))
    sweeps_half = 0.5 * (int(sweeps_max) - int(sweeps_min))
    sweeps_default = int(np.clip(np.round(sweeps_center), sweeps_min, sweeps_max))

    if w_only:
        sweeps_down = sweeps_default
        sweeps_up = sweeps_default
    elif sweeps_half <= 0.0:
        sweeps_down = int(sweeps_min)
        sweeps_up = int(sweeps_min)
    else:
        sweeps_down = int(np.clip(
            np.round(sweeps_center + sweeps_half * a_d),
            sweeps_min, sweeps_max
        ))
        sweeps_up = int(np.clip(
            np.round(sweeps_center + sweeps_half * a_u),
            sweeps_min, sweeps_max
        ))

    if cycle == 0:
        if w_init is not None:
            w_min = w_center - w_scale
            w_max = w_center + w_scale
            w = float(np.clip(w_init, w_min, w_max))
        if sweeps_init is not None:
            init_s = int(np.clip(int(sweeps_init), sweeps_min, sweeps_max))
            sweeps_down = init_s
            sweeps_up = init_s

    return float(w), int(sweeps_down), int(sweeps_up)

def decode_policy_action_residual(
    action,
    *,
    w_only,
    w_center,
    w_scale,
    w_min,
    w_max,
    sweeps_min,
    sweeps_max,
    cycle,
    last_w,
    w_init=None,
    sweeps_init=None,
):
    # Residual action means "adjust the current weight" instead of "predict an
    # absolute weight". The first cycle starts from w_center (or w_init); later
    # cycles start from last_w. A clipped residual update keeps the operating
    # family in the global admissible range, e.g. [1, 2] for Exp44.
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    a_w = float(np.clip(action[0], -1.0, 1.0))
    if w_only or action.shape[0] == 1:
        a_d = 0.0
        a_u = 0.0
    else:
        a_d = float(np.clip(action[1], -1.0, 1.0))
        a_u = float(np.clip(action[2], -1.0, 1.0))

    if cycle == 0:
        base_w = float(w_center if w_init is None else w_init)
    else:
        base_w = float(last_w)
    # With action in [-1, 1], the largest single-step movement is exactly
    # w_scale in either direction.
    w = float(np.clip(base_w + w_scale * a_w, w_min, w_max))

    sweeps_center = 0.5 * (int(sweeps_min) + int(sweeps_max))
    sweeps_half = 0.5 * (int(sweeps_max) - int(sweeps_min))
    sweeps_default = int(np.clip(np.round(sweeps_center), sweeps_min, sweeps_max))

    if w_only:
        sweeps_down = sweeps_default
        sweeps_up = sweeps_default
    elif sweeps_half <= 0.0:
        sweeps_down = int(sweeps_min)
        sweeps_up = int(sweeps_min)
    else:
        sweeps_down = int(np.clip(
            np.round(sweeps_center + sweeps_half * a_d),
            sweeps_min, sweeps_max
        ))
        sweeps_up = int(np.clip(
            np.round(sweeps_center + sweeps_half * a_u),
            sweeps_min, sweeps_max
        ))

    if cycle == 0 and sweeps_init is not None:
        init_s = int(np.clip(int(sweeps_init), sweeps_min, sweeps_max))
        sweeps_down = init_s
        sweeps_up = init_s

    return float(w), int(sweeps_down), int(sweeps_up)


def decode_policy_action_hierarchical(
    action,
    *,
    w_only,
    sweeps_min,
    sweeps_max,
    coarse_bins=((1.0, 1.4), (1.4, 1.6), (1.6, 1.8), (1.8, 2.0)),
):
    action = np.asarray(action, dtype=np.float32).reshape(-1)

    a_bin = float(np.clip(action[0], -1.0, 1.0))
    a_off = float(np.clip(action[1], -1.0, 1.0))
    if w_only or action.shape[0] <= 2:
        a_d = 0.0
        a_u = 0.0
    else:
        a_d = float(np.clip(action[2], -1.0, 1.0))
        a_u = float(np.clip(action[3], -1.0, 1.0))

    bins = tuple((float(lo), float(hi)) for lo, hi in coarse_bins)
    if not bins:
        raise ValueError("coarse_bins must be non-empty")
    # Map [-1, 1] to a coarse bin index.
    pos = 0.5 * (a_bin + 1.0)
    idx = int(np.clip(np.floor(pos * len(bins)), 0, len(bins) - 1))
    lo, hi = bins[idx]
    # Map offset back into the chosen bin.
    w = float(lo + 0.5 * (a_off + 1.0) * (hi - lo))

    sweeps_center = 0.5 * (int(sweeps_min) + int(sweeps_max))
    sweeps_half = 0.5 * (int(sweeps_max) - int(sweeps_min))
    sweeps_default = int(np.clip(np.round(sweeps_center), sweeps_min, sweeps_max))
    if w_only:
        sweeps_down = sweeps_default
        sweeps_up = sweeps_default
    elif sweeps_half <= 0.0:
        sweeps_down = int(sweeps_min)
        sweeps_up = int(sweeps_min)
    else:
        sweeps_down = int(np.clip(np.round(sweeps_center + sweeps_half * a_d), sweeps_min, sweeps_max))
        sweeps_up = int(np.clip(np.round(sweeps_center + sweeps_half * a_u), sweeps_min, sweeps_max))

    return float(w), int(sweeps_down), int(sweeps_up)
class BoomerAMGRelaxEnv(gym.Env):
    """
    One RL step = one BoomerAMG V-cycle.
    Episode = full solve until tol or max_cycles.

    Action controls (3-dim, continuous in [-1,1]):
      - relax weight (w)
      - sweeps_down (pre-smoothing)
      - sweeps_up   (post-smoothing)

    Paper-style compact observation (10-dim):
      s = [
        log(r_k / r_{k-1}),
        sign(r_{k-1} - r_k),
        cycle_frac,
        a1/a0, a2/a0, a3/a0,     (Laplacian) OR c_x,c_y,c_z normalized (difconv)
        nx_norm, ny_norm, nz_norm,
        last_w,
      ]
    Reward (paper-like):
      ((r_prev - r) / r_prev) / dt
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lib_path=None,
        fixed_grid=(60, 60, 60),
        fixed_stencil=27,
        fixed_rhs_type=1,   # 1 => uses rhs_seed in your C code
        tol=1e-6,
        max_cycles=50,
        seed=0,
        # Default w range ~[0.8, 1.3]
        w_center=1.05,
        w_scale=0.25,
        w_init=None,
        w_only=False,
        sweeps_min=1,
        sweeps_max=5,
        sweeps_init=None,
        w_smooth_alpha=0.0,
        w_change_penalty=0.0,
        sweeps_change_penalty=0.0,
        cycle_penalty=0.0,
        sweep_penalty=0.0,
        randomize_A=True,
        randomize_b=True,
        fixed_rhs_seed=123456789,
        randomize_grid=False,
        grid_min=10,
        grid_max=80,
        use_grid_bias=False,
        grid_bias=1.0,
        difconv_c=(1.0, 100.0, 100.0),
        difconv_c_range=(1.0, 1000.0),
        difconv_a=(0.0, 0.0, 0.0),
        difconv_atype=0,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.randomize_grid = bool(randomize_grid)
        self.grid_min = int(grid_min)
        self.grid_max = int(grid_max)
        self.use_grid_bias = bool(use_grid_bias)
        self.grid_bias = float(grid_bias)
        if self.grid_min <= 0 or self.grid_max < self.grid_min:
            raise ValueError("grid_min/grid_max must be positive with grid_max >= grid_min")
        if self.use_grid_bias and self.grid_bias <= 0.0:
            raise ValueError("grid_bias must be > 0 when use_grid_bias is enabled")

        self.grid_choices = self._normalize_grid_choices(fixed_grid)
        self.fixed_grid = self.grid_choices[0]
        self.fixed_stencil = int(fixed_stencil)
        self.fixed_rhs_type = int(fixed_rhs_type)

        self.tol = float(tol)
        self.max_cycles = int(max_cycles)

        self.w_center = float(w_center)
        self.w_scale = float(w_scale)
        self.w_init = None if w_init is None else float(w_init)
        self.w_only = bool(w_only)
        self.sweeps_min = int(sweeps_min)
        self.sweeps_max = int(sweeps_max)
        if self.sweeps_min < 1 or self.sweeps_max < self.sweeps_min:
            raise ValueError("sweeps_min/sweeps_max must satisfy 1 <= min <= max")
        self.sweeps_center = 0.5 * (self.sweeps_min + self.sweeps_max)
        self.sweeps_half = 0.5 * (self.sweeps_max - self.sweeps_min)
        self.sweeps_default = int(np.clip(np.round(self.sweeps_center),
                                          self.sweeps_min, self.sweeps_max))
        self.sweeps_init = None if sweeps_init is None else int(sweeps_init)

        # 27pt defaults
        self.a0_base = 26.0
        self.a1_base = -4.0
        self.a2_base = -0.15
        self.a3_base = -0.0125
        self.difconv_c = tuple(float(x) for x in difconv_c)
        self.difconv_c_range = tuple(float(x) for x in difconv_c_range)
        self.difconv_a = tuple(float(x) for x in difconv_a)
        self.difconv_atype = int(difconv_atype)

        self.episode_id = 0

        self.w_smooth_alpha = float(w_smooth_alpha)
        self.w_change_penalty = float(w_change_penalty)
        self.sweeps_change_penalty = float(sweeps_change_penalty)
        self.cycle_penalty = float(cycle_penalty)
        self.sweep_penalty = float(sweep_penalty)
        self.randomize_A = bool(randomize_A)
        self.randomize_b = bool(randomize_b)
        self.fixed_rhs_seed = int(fixed_rhs_seed)

        self.reward_mode   = int(os.environ.get("REWARD_MODE", "0"))
        self.reward_alpha  = float(os.environ.get("REWARD_ALPHA", "1.3"))
        self.term_bonus    = float(os.environ.get("TERM_BONUS", "2.0"))
        self.dt_penalty    = float(os.environ.get("DT_PENALTY", "0.0"))  # optional extra -dt term

        resolved_lib_path = DEFAULT_AMG_RUNTIME_LIBRARY if lib_path is None else Path(lib_path)
        resolved_lib_path = resolved_lib_path.expanduser().resolve()
        if resolved_lib_path != DEFAULT_AMG_RUNTIME_LIBRARY.resolve():
            raise RuntimeError(
                "BoomerAMGRelaxEnv only supports the shared libamg_runtime "
                f"binding at {DEFAULT_AMG_RUNTIME_LIBRARY}."
            )
        if not resolved_lib_path.exists():
            raise RuntimeError(
                f"Compiled AMG runtime not found at {resolved_lib_path}. "
                "Build it with `make -C hypre/interfaces`."
            )

        # Action: either (w, down, up) or w-only
        if self.w_only:
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([ 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
                high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
                dtype=np.float32,
            )

        # Observation: 10 dims
        self.obs_dim = 10
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # runtime state
        self.prepared_env = None
        self.stencil = None
        self.r0 = None
        self.r_prev = None
        self.cycle = 0

        # last action memory
        self.last_w = self.w_init if self.w_init is not None else self.w_center
        init_sweeps = self.sweeps_default if self.sweeps_init is None else int(self.sweeps_init)
        init_sweeps = int(np.clip(init_sweeps, self.sweeps_min, self.sweeps_max))
        self.last_sweeps_down = init_sweeps
        self.last_sweeps_up = init_sweeps

        # coefficients used (fixed per reset)
        self.a0 = self.a0_base
        self.a1 = self.a1_base
        self.a2 = self.a2_base
        self.a3 = self.a3_base
        self.cx, self.cy, self.cz = self.difconv_c
        self.ax, self.ay, self.az = self.difconv_a

        # problem descriptor (computed from a’s)
        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0
        self.nx_norm = 0.0
        self.ny_norm = 0.0
        self.nz_norm = 0.0
        if self.randomize_grid:
            self.grid_norm_div = float(self.grid_max)
        else:
            self.grid_norm_div = float(max(max(g) for g in self.grid_choices))
        if self.grid_norm_div <= 0.0:
            self.grid_norm_div = 1.0
        self.c_norm_div = float(self.difconv_c_range[1]) if self.difconv_c_range else 1.0
        if self.c_norm_div <= 0.0:
            self.c_norm_div = 1.0

        self.residual_curve = None

    def _coeff_ratios(self):
        if int(self.stencil or 0) == 0:
            eps = 1e-30
            denom = max(math.log(self.c_norm_div + eps), eps)
            return (
                math.log(float(self.cx) + eps) / denom,
                math.log(float(self.cy) + eps) / denom,
                math.log(float(self.cz) + eps) / denom,
            )
        eps = 1e-30
        a0 = float(self.a0)
        return (
            float(self.a1) / (a0 + eps),
            float(self.a2) / (a0 + eps),
            float(self.a3) / (a0 + eps),
        )

    def _normalize_grid_choices(self, fixed_grid):
        if fixed_grid is None:
            # Placeholder; real grid chosen in reset when randomize_grid=True
            return [(self.grid_min, self.grid_min, self.grid_min)]
        if isinstance(fixed_grid, (list, tuple)) and fixed_grid:
            first = fixed_grid[0]
            if isinstance(first, (list, tuple)) and len(first) == 3:
                return [tuple(int(x) for x in g) for g in fixed_grid]
            if len(fixed_grid) == 3 and all(isinstance(x, (int, np.integer)) for x in fixed_grid):
                return [tuple(int(x) for x in fixed_grid)]
        raise ValueError("fixed_grid must be (nx,ny,nz) or list of such tuples")

    def _make_obs(self, r, r_prev):
        return build_policy_obs(
            r=r,
            r_prev=r_prev,
            cycle=self.cycle,
            max_cycles=self.max_cycles,
            coeff_triplet=self._coeff_ratios(),
            grid_triplet=(self.nx_norm, self.ny_norm, self.nz_norm),
            last_w=self.last_w,
        )
          

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.prepared_env is not None:
            self.prepared_env.close()
            self.prepared_env = None

        if self.randomize_grid:
            u = float(self.rng.random())
            if self.use_grid_bias and abs(self.grid_bias - 1.0) > 1e-12:
                u = u ** (1.0 / self.grid_bias)
            n = self.grid_min + int(np.floor(u * (self.grid_max - self.grid_min + 1)))
            if n > self.grid_max:
                n = self.grid_max
            self.fixed_grid = (n, n, n)
        else:
            grid_idx = int(self.rng.integers(0, len(self.grid_choices)))
            self.fixed_grid = self.grid_choices[grid_idx]
        nx, ny, nz = self.fixed_grid
        eps = 1e-30
        denom = max(math.log(self.grid_norm_div + eps), eps)
        self.nx_norm = math.log(float(nx) + eps) / denom
        self.ny_norm = math.log(float(ny) + eps) / denom
        self.nz_norm = math.log(float(nz) + eps) / denom
        self.stencil = self.fixed_stencil
        rhs_type = self.fixed_rhs_type

        self.episode_id += 1   # <--- add this

        # -------------------------
        # FIX A (harder but stable)
        # -------------------------
        # if self.stencil == 27:
        #     # Strengthen off-diagonals, recompute diagonal
        #     s1, s2, s3 = 1.4, 2.0, 2.5
        #     self.a1 = self.a1_base * s1
        #     self.a2 = self.a2_base * s2
        #     self.a3 = self.a3_base * s3
        #     self.a0 = -(6.0 * self.a1 + 12.0 * self.a2 + 8.0 * self.a3)
        #     k, c = 1.0, 0.0
        # else:
        #     k, c = 1.0, 0.0
        #     self.a0, self.a1, self.a2, self.a3 = self.a0_base, self.a1_base, self.a2_base, self.a3_base

        # -------------------------
        # A: fixed or randomized
        # -------------------------
        if self.stencil == 0:
            # DifConv: sample diffusion coefficients if randomize_A
            c_min, c_max = self.difconv_c_range
            if self.randomize_A:
                self.cx = float(self.rng.uniform(c_min, c_max))
                self.cy = float(self.rng.uniform(c_min, c_max))
                self.cz = float(self.rng.uniform(c_min, c_max))
            else:
                self.cx, self.cy, self.cz = self.difconv_c
            self.ax, self.ay, self.az = self.difconv_a
            # Map DifConv params into the native runtime slots: k,c,a0..a3.
            self.a0, self.a1, self.a2, self.a3 = self.cz, self.ax, self.ay, self.az
            k, c = float(self.cx), float(self.cy)
        else:
            self.a0, self.a1, self.a2, self.a3, k, c = generate_matrix_coeffs(
                self.rng,
                stencil=self.stencil,
                randomize_A=self.randomize_A,
                a0_base=self.a0_base,
                a1_base=self.a1_base,
                a2_base=self.a2_base,
                a3_base=self.a3_base,
            )
        # Problem descriptor for policy input
        # (If A fixed, these are constants; that’s OK and matches “training on one problem”)
        self.s1, self.s2, self.s3 = self._coeff_ratios()

        # -------------------------
        # FIX b (deterministic)
        # -------------------------
        # rhs_seed = 123456789  # fixed seed => fixed b if rhs_type==1

        # -------------------------
        # b: fixed or randomized
        # -------------------------
        rhs_seed = generate_rhs_seed(
            self.rng,
            randomize_b=self.randomize_b,
            fixed_rhs_seed=self.fixed_rhs_seed,
        )

        if self.stencil == 0:
            matrix_kwargs = build_matrix_kwargs_difconv(
                nx=nx,
                ny=ny,
                nz=nz,
                cx=self.cx,
                cy=self.cy,
                cz=self.cz,
                ax=self.ax,
                ay=self.ay,
                az=self.az,
                rhs_seed=rhs_seed,
                rhs_type=rhs_type,
            )
            k = float(matrix_kwargs["k"])
            c = float(matrix_kwargs["c"])
            self.a0 = float(matrix_kwargs["a0"])
            self.a1 = float(matrix_kwargs["a1"])
            self.a2 = float(matrix_kwargs["a2"])
            self.a3 = float(matrix_kwargs["a3"])

        self.prepared_env = create_env(
            nx=nx, ny=ny, nz=nz,
            stencil=self.stencil,
            rhs_type=rhs_type,
            rhs_seed=rhs_seed,
            k=k, c=c,
            a0=self.a0, a1=self.a1, a2=self.a2, a3=self.a3,
        )
        prepared = self.prepared_env.prepare_rl({})

        self.r0 = float(prepared.initial_residual_norm)
        self.r_prev = self.r0
        self.cycle = 0
        self.setup_time = float(prepared.setup_runtime_sec)

        self.last_w = self.w_init if self.w_init is not None else self.w_center
        init_sweeps = self.sweeps_default if self.sweeps_init is None else int(self.sweeps_init)
        init_sweeps = int(np.clip(init_sweeps, self.sweeps_min, self.sweeps_max))
        self.last_sweeps_down = init_sweeps
        self.last_sweeps_up = init_sweeps

        self.residual_curve = [self.r0]
        r = float(self.prepared_env.r)

        obs = self._make_obs(r, self.r_prev)
        info = {
            "episode_id": self.episode_id,
            "nx": nx, "ny": ny, "nz": nz,
            "stencil": self.stencil,
            "rhs_type": rhs_type,
            "rhs_seed": rhs_seed,
            "setup_time": float(self.setup_time),
            "cycle_type": int(self.prepared_env.cycle_type),
            "relax_type": int(self.prepared_env.relax_type),
            "r0": self.r0,
            "a0": float(self.a0), "a1": float(self.a1), "a2": float(self.a2), "a3": float(self.a3),
            "cx": float(self.cx), "cy": float(self.cy), "cz": float(self.cz),
            "ax": float(self.ax), "ay": float(self.ay), "az": float(self.az),
        }
        return obs, info

    def step(self, action):
        w, sweeps_down, sweeps_up = decode_policy_action(
            action,
            w_only=self.w_only,
            w_center=self.w_center,
            w_scale=self.w_scale,
            sweeps_min=self.sweeps_min,
            sweeps_max=self.sweeps_max,
            cycle=self.cycle,
            last_w=self.last_w,
            w_init=self.w_init,
            sweeps_init=self.sweeps_init,
            w_smooth_alpha=self.w_smooth_alpha,
        )

        prev_w = self.last_w
        prev_sd = self.last_sweeps_down
        prev_su = self.last_sweeps_up

        # Store "last action" (so obs includes weight used to produce the next residual)
        self.last_w = w
        self.last_sweeps_down = sweeps_down
        self.last_sweeps_up = sweeps_up

        # --- Measure both wall time and solver time around the C step
        t0 = time.perf_counter()
        try:
            r_solver, dt_solver = self.prepared_env.step_rl(
                relax_weight=w,
                sweeps_down=sweeps_down,
                sweeps_up=sweeps_up,
                coarse_sweeps=1,
                tol=self.tol,
                max_cycles=self.max_cycles,
            )
        except AMGNativeError as exc:
            dt_wall = time.perf_counter() - t0
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -10.0, False, True, {
                "bad_step": True,
                "failure_reason": str(exc),
                "failure_stage": exc.operation,
                "dt_wall": dt_wall,
                "dt_solver": exc.solve_runtime_sec,
            }
        dt_wall = time.perf_counter() - t0
        r_solver = float(r_solver)
        dt_solver = float(dt_solver)
        status = self.prepared_env.last_step.status
        self.cycle = int(self.prepared_env.cycle)

        # --- Safety checks
        if (
            (not np.isfinite(r_solver)) or (r_solver <= 0.0) or
            (not np.isfinite(dt_solver)) or (dt_solver <= 0.0) or
            (not np.isfinite(dt_wall)) or (dt_wall <= 0.0)
        ):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -10.0, False, True, {"bad_step": True, "r": r_solver, "dt_wall": dt_wall, "dt_solver": dt_solver}

        # --- Reward parameters (read from attributes; fallback to env vars if missing)
        reward_mode = int(getattr(self, "reward_mode", int(os.environ.get("REWARD_MODE", "0"))))
        reward_alpha = float(getattr(self, "reward_alpha", float(os.environ.get("REWARD_ALPHA", "1.3"))))
        term_bonus = float(getattr(self, "term_bonus", float(os.environ.get("TERM_BONUS", "2.0"))))
        dt_penalty = float(getattr(self, "dt_penalty", float(os.environ.get("DT_PENALTY", "0.0"))))

        # --- Compute progress signals
        eps = 1e-30
        r_prev = max(float(self.r_prev), eps)
        r_cur = max(float(r_solver), eps)

        log_prev = math.log(r_prev + eps)
        log_cur = math.log(r_cur + eps)
        log_tol = math.log(self.tol + eps)

        log_drop = max(0.0, log_prev - log_cur)
        rel_drop = (r_prev - r_cur) / r_prev  # can be negative if residual increases

        dt_eff = max(dt_solver, 1e-6)  # use internal BoomerAMGSolve time for reward

        # --- Reward modes
        if reward_mode == 0:
            # current-style: log drop per solver time
            reward = reward_alpha * (log_drop / dt_eff)

        elif reward_mode == 1:
            # literature-like: relative drop per solver time
            reward = reward_alpha * (rel_drop / dt_eff)

        elif reward_mode == 2:
            # remaining-gap normalized (your LaTeX-style), per solver time
            remaining = max(log_prev - log_tol, 1e-6)
            frac = log_drop / remaining
            reward = reward_alpha * (frac / dt_eff)

        elif reward_mode == 3:
            # minimum-time objective: directly minimize solver time
            reward = -dt_eff

        else:
            # fallback
            reward = reward_alpha * (log_drop / dt_eff)

        # Optional extra explicit time penalty (usually 0)
        reward -= dt_penalty * dt_eff

        terminated = status is SolveStatus.CONVERGED
        truncated = status is SolveStatus.MAX_CYCLES

        if terminated:
            reward += term_bonus

        # Optional regularizers (already wired via env vars -> kwargs in your make_env)
        if self.cycle_penalty > 0.0:
            reward -= self.cycle_penalty

        if self.sweep_penalty > 0.0:
            reward -= self.sweep_penalty * float(sweeps_down + sweeps_up)

        if self.w_change_penalty > 0.0:
            reward -= self.w_change_penalty * abs(w - prev_w)

        if self.sweeps_change_penalty > 0.0:
            reward -= self.sweeps_change_penalty * (
                abs(sweeps_down - prev_sd) + abs(sweeps_up - prev_su)
            )

        # Build next obs; note: obs includes w used in the step (stored in self.last_w)
        obs = self._make_obs(r_cur, self.r_prev)
        self.r_prev = r_cur
        self.residual_curve.append(r_cur)

        info = {
            "r": r_cur,
            "r_prev": r_prev,
            "rel_drop": rel_drop,
            # keep both timing measures:
            "dt": dt_wall,               # make dt == wall time for traj logging
            "dt_wall": dt_wall,
            "dt_solver": dt_solver,
            "setup_time": float(getattr(self, "setup_time", 0.0)),
            "w": w,
            "sweeps_down": sweeps_down,
            "sweeps_up": sweeps_up,
            "cycle": self.cycle,
            "reward_mode": reward_mode,
            # problem descriptors (if you want them in logs):
            "cx": float(self.cx), "cy": float(self.cy), "cz": float(self.cz),
        }

        if terminated or truncated:
            info["residual_curve"] = np.array(self.residual_curve, dtype=np.float64)
            info["cycles_to_stop"] = len(self.residual_curve) - 1

        return obs, float(reward), bool(terminated), bool(truncated), info
    def close(self):
        if self.prepared_env is not None:
            self.prepared_env.close()
            self.prepared_env = None
