# amg_gym_env.py
import os, ctypes, math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BoomerAMGRelaxEnv(gym.Env):
    """
    One RL step = one BoomerAMG V-cycle.
    Episode = full solve until tol or max_cycles.

    Action controls (3-dim, continuous in [-1,1]):
      - relax weight (w)
      - sweeps_down (pre-smoothing)
      - sweeps_up   (post-smoothing)

    Paper-style compact observation (7-dim):
      s = [
        log(r_k / r_{k-1}),
        sign(r_{k-1} - r_k),
        cycle_frac,
        a1/a0, a2/a0, a3/a0,     (problem descriptor; constant if A fixed)
        last_w                    (previous action memory)
      ]
    Reward (paper-like):
      ((r_prev - r) / r_prev) / dt
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        lib_path="./libamg_env.dylib",
        fixed_grid=(60, 60, 60),
        fixed_stencil=27,
        fixed_rhs_type=1,   # 1 => uses rhs_seed in your C code
        tol=1e-8,
        max_cycles=30,
        seed=0,
        w_center=0.9,
        w_scale=0.3,
    ):
        super().__init__()
        self.rng = np.random.default_rng(seed)

        self.fixed_grid = tuple(fixed_grid)
        self.fixed_stencil = int(fixed_stencil)
        self.fixed_rhs_type = int(fixed_rhs_type)

        self.tol = float(tol)
        self.max_cycles = int(max_cycles)

        self.w_center = float(w_center)
        self.w_scale = float(w_scale)

        # 27pt defaults
        self.a0_base = 26.0
        self.a1_base = -4.0
        self.a2_base = -0.15
        self.a3_base = -0.0125

        self.episode_id = 0

        self.lib = ctypes.CDLL(os.path.abspath(lib_path))
        self.AMGEnv_p = ctypes.c_void_p
        self._bind()

        # Action: (w, down, up) each in [-1,1]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation: 7 dims (paper-style compact)
        self.obs_dim = 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # runtime state
        self.env_ptr = None
        self.stencil = None
        self.r0 = None
        self.r_prev = None
        self.cycle = 0

        # last action memory
        self.last_w = self.w_center
        self.last_sweeps_down = 3
        self.last_sweeps_up = 3

        # coefficients used (fixed per reset)
        self.a0 = self.a0_base
        self.a1 = self.a1_base
        self.a2 = self.a2_base
        self.a3 = self.a3_base

        # problem descriptor (computed from a’s)
        self.s1 = 0.0
        self.s2 = 0.0
        self.s3 = 0.0

        self.residual_curve = None

    def _bind(self):
        lib = self.lib
        AMGEnv_p = self.AMGEnv_p

        lib.amg_env_create.restype = AMGEnv_p
        lib.amg_env_create.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,   # nx ny nz
            ctypes.c_int, ctypes.c_int,                 # stencil rhs_type
            ctypes.c_double, ctypes.c_int,              # tol max_cycles
            ctypes.c_ulonglong,                         # rhs_seed
            ctypes.c_double, ctypes.c_double,           # k c (7pt)
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double  # a0..a3 (27pt)
        ]

        lib.amg_env_step.restype = ctypes.c_int
        lib.amg_env_step.argtypes = [
            AMGEnv_p,
            ctypes.c_double,  # relax_weight
            ctypes.c_int,     # sweeps_down
            ctypes.c_int,     # sweeps_up
            ctypes.POINTER(ctypes.c_double),  # r
            ctypes.POINTER(ctypes.c_double),  # dt
            ctypes.POINTER(ctypes.c_int),     # status
        ]

        lib.amg_env_get_r0.restype = ctypes.c_double
        lib.amg_env_get_r0.argtypes = [AMGEnv_p]

        lib.amg_env_get_r.restype = ctypes.c_double
        lib.amg_env_get_r.argtypes = [AMGEnv_p]

        lib.amg_env_get_cycle.restype = ctypes.c_int
        lib.amg_env_get_cycle.argtypes = [AMGEnv_p]

        lib.amg_env_destroy.restype = None
        lib.amg_env_destroy.argtypes = [AMGEnv_p]

    def _coeff_ratios(self):
        eps = 1e-30
        a0 = float(self.a0)
        return (
            float(self.a1) / (a0 + eps),
            float(self.a2) / (a0 + eps),
            float(self.a3) / (a0 + eps),
        )

    def _make_obs(self, r, r_prev):
        eps = 1e-30
        r = float(r); r_prev = float(r_prev)

        ratio = (r + eps) / (r_prev + eps)
        ratio = float(np.clip(ratio, 1e-30, 1e30))
        log_ratio = math.log(ratio)

        dr = r_prev - r
        sgn = 1.0 if dr > 0 else (-1.0 if dr < 0 else 0.0)

        cycle_frac = float(self.cycle) / float(self.max_cycles)

        # s1,s2,s3 = normalized A information (fixed A -> constant, but still explicit)
        s1, s2, s3 = self._coeff_ratios()

        obs = np.array(
            [log_ratio, sgn, cycle_frac, s1, s2, s3, float(self.last_w)],
            dtype=np.float32
        )
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
          

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.env_ptr:
            self.lib.amg_env_destroy(self.env_ptr)
            self.env_ptr = None

        nx, ny, nz = self.fixed_grid
        self.stencil = self.fixed_stencil
        rhs_type = self.fixed_rhs_type

        self.episode_id += 1   # <--- add this

        # -------------------------
        # FIX A (harder but stable)
        # -------------------------
        if self.stencil == 27:
            # Strengthen off-diagonals, recompute diagonal
            s1, s2, s3 = 1.4, 2.0, 2.5
            self.a1 = self.a1_base * s1
            self.a2 = self.a2_base * s2
            self.a3 = self.a3_base * s3
            self.a0 = -(6.0 * self.a1 + 12.0 * self.a2 + 8.0 * self.a3)
            k, c = 1.0, 0.0
        else:
            k, c = 1.0, 0.0
            self.a0, self.a1, self.a2, self.a3 = self.a0_base, self.a1_base, self.a2_base, self.a3_base

        # Problem descriptor for policy input
        # (If A fixed, these are constants; that’s OK and matches “training on one problem”)
        a0_eps = float(self.a0) if abs(self.a0) > 1e-30 else 1.0
        self.s1 = float(self.a1 / a0_eps)
        self.s2 = float(self.a2 / a0_eps)
        self.s3 = float(self.a3 / a0_eps)

        # -------------------------
        # FIX b (deterministic)
        # -------------------------
        rhs_seed = 123456789  # fixed seed => fixed b if rhs_type==1

        self.env_ptr = self.lib.amg_env_create(
            nx, ny, nz,
            self.stencil, rhs_type,
            self.tol, self.max_cycles,
            rhs_seed,
            k, c, self.a0, self.a1, self.a2, self.a3
        )

        self.r0 = float(self.lib.amg_env_get_r0(self.env_ptr))
        self.r_prev = self.r0
        self.cycle = 0

        self.last_w = self.w_center
        self.last_sweeps_down = 3
        self.last_sweeps_up = 3

        self.residual_curve = [self.r0]
        r = float(self.lib.amg_env_get_r(self.env_ptr))

        obs = self._make_obs(r, self.r_prev)
        info = {
            "episode_id": self.episode_id,
            "nx": nx, "ny": ny, "nz": nz,
            "stencil": self.stencil,
            "rhs_type": rhs_type,
            "rhs_seed": rhs_seed,
            "r0": self.r0,
            "a0": float(self.a0), "a1": float(self.a1), "a2": float(self.a2), "a3": float(self.a3),
        }
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        a_w = float(np.clip(action[0], -1.0, 1.0))
        a_d = float(np.clip(action[1], -1.0, 1.0))
        a_u = float(np.clip(action[2], -1.0, 1.0))

        # w mapping: centered band
        w = float(np.clip(
            self.w_center + self.w_scale * a_w,
            self.w_center - self.w_scale,
            self.w_center + self.w_scale
        ))

        # sweeps mapping: map [-1,1] -> {1..5} via round(3 + 2*a)
        sweeps_down = int(np.clip(np.round(3.0 + 2.0 * a_d), 1, 5))
        sweeps_up   = int(np.clip(np.round(3.0 + 2.0 * a_u), 1, 5))

        self.last_w = w
        self.last_sweeps_down = sweeps_down
        self.last_sweeps_up = sweeps_up

        r_c  = ctypes.c_double()
        dt_c = ctypes.c_double()
        st_c = ctypes.c_int()

        self.lib.amg_env_step(
            self.env_ptr, w, sweeps_down, sweeps_up,
            ctypes.byref(r_c), ctypes.byref(dt_c), ctypes.byref(st_c)
        )

        r  = float(r_c.value)
        dt = float(dt_c.value)
        self.cycle = int(self.lib.amg_env_get_cycle(self.env_ptr))

        # Safety checks
        if (not np.isfinite(r)) or (r <= 0.0) or (not np.isfinite(dt)) or (dt <= 0.0):
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, -10.0, False, True, {"bad_step": True, "r": r, "dt": dt}

        # # --- Reward:  relative log progress per wall-time, with time penalty
        # eps = 1e-30
        # # clamp to avoid weirdness if residual goes up slightly
        # r_prev_used = max(self.r_prev, eps)
        # r_used      = max(r, eps)

        # log_prev = math.log(r_prev_used)
        # log_curr = math.log(r_used)
        # log_drop = max(0.0, log_prev - log_curr)   # zero if we get worse

        # # scale so that early steps give ~0.5–1.0 reward
        # alpha = 0.01      # << smaller than 0.1
        # improv = alpha * (log_drop / max(dt, 1e-6))

        # # time penalty: each V-cycle costs something
        # time_penalty_coeff = 1.0
        # time_penalty = time_penalty_coeff * dt

        # reward = improv - time_penalty

        # terminated = (st_c.value == 1)
        # truncated  = (st_c.value == 2)

        # if terminated:
        #     reward += 2.0     # small terminal bonus
        # if truncated:
        #     reward -= 2.0     # stronger penalty for not converging

        # --- Reward: log-residual reduction per unit time (speed vs baseline)
        eps = 1e-30
        r_prev_used = max(self.r_prev, eps)
        r_used      = max(r, eps)
        log_prev = math.log(self.r_prev + eps)
        log_curr = math.log(r + eps)
        log_tol  = math.log(self.tol + eps)

        # remaining_prev = max(log_prev - log_tol, 1e-6)     # “how far from tol”
        # log_drop = max(0.0, log_prev - log_curr)           # no reward if worse

        # frac_progress = log_drop / remaining_prev          # fraction of remaining gap removed
        # speed = frac_progress / max(dt, 1e-6)              # per-time version
        log_drop = max(0.0, log_prev - log_curr)
        speed = log_drop / max(dt, 1e-6)

        alpha = 1.3   # strength of progress term

        reward = alpha * speed
        
        terminated = (st_c.value == 1)
        truncated  = (st_c.value == 2)

        if terminated:
            reward += 1.0     # small bonus for actually converging
        if truncated:
            reward -= 1.0     # penalty for failing to reach tol

        obs = self._make_obs(r, self.r_prev)
        self.r_prev = r
        self.residual_curve.append(r)
        rel_drop = (r_prev_used - r_used) / r_prev_used

        info = {
            "r": r,
            "r_prev": r_prev_used,
            "rel_drop": rel_drop,
            "dt": dt,
            "w": w,
            "sweeps_down": sweeps_down,
            "sweeps_up": sweeps_up,
            "cycle": self.cycle,
            # In case you ever care for logging these:
            "a0": float(self.a0),
            "a1": float(self.a1),
            "a2": float(self.a2),
            "a3": float(self.a3),
        }
        if terminated or truncated:
            info["residual_curve"] = np.array(self.residual_curve, dtype=np.float64)
            info["cycles_to_stop"] = len(self.residual_curve) - 1

        return obs, reward, terminated, truncated, info

    def close(self):
        if self.env_ptr:
            self.lib.amg_env_destroy(self.env_ptr)
            self.env_ptr = None
