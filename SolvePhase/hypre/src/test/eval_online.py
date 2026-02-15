import os
import time
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from amg_gym_env import BoomerAMGRelaxEnv

# Eval-online uses the trained relax type (18) by default.
os.environ.setdefault("AMG_RELAX_TYPE", "18")


def _unwrap_single(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0]
    return x


def _env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def _parse_triplet_env(name: str, default: str):
    raw = os.environ.get(name, default)
    parts = [p for p in raw.replace("x", ",").replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must be 3 values (got: {raw})")
    return tuple(float(p) for p in parts)


def _parse_range_env(name: str, default: str):
    raw = os.environ.get(name, default)
    parts = [p for p in raw.replace("x", ",").replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be 2 values (got: {raw})")
    lo, hi = (float(p) for p in parts)
    if hi < lo:
        raise ValueError(f"{name} must be lo,hi with hi>=lo (got: {raw})")
    return (lo, hi)


def _expand_seeds(eval_seeds, instances, stride):
    if instances <= 1:
        return list(eval_seeds)
    expanded = []
    for s in eval_seeds:
        for i in range(instances):
            expanded.append(int(s) + int(i) * int(stride))
    return expanded


def _summarize(results):
    times = np.array([r["time"] for r in results], dtype=float)
    cycles = np.array([r["cycles"] for r in results], dtype=float)
    time_per_cycle = times / np.maximum(cycles, 1.0)
    return {
        "mean_time": float(times.mean()),
        "median_time": float(np.median(times)),
        "mean_cycles": float(cycles.mean()),
        "median_cycles": float(np.median(cycles)),
        "mean_time_per_cycle": float(time_per_cycle.mean()),
        "median_time_per_cycle": float(np.median(time_per_cycle)),
    }


def run_episode(env, policy, obs_normalizer=None, deterministic=True, use_lstm=False):
    obs, info = env.reset()
    info = _unwrap_single(info)
    done = False

    total_dt = 0.0
    steps = 0
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    while not done:
        obs_for_policy = obs
        if obs_normalizer is not None:
            obs_for_policy = obs_normalizer(obs_for_policy)
        if use_lstm:
            action, lstm_state = policy.predict(
                obs_for_policy,
                state=lstm_state,
                episode_start=episode_start,
                deterministic=deterministic,
            )
        else:
            action, _ = policy.predict(obs_for_policy, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)
        info = _unwrap_single(info)

        total_dt += float(info.get("dt", 0.0))
        done = bool(_unwrap_single(terminated)) or bool(_unwrap_single(truncated))
        episode_start[...] = done
        steps += 1

    return {"time": total_dt, "cycles": steps, "info": info}


class CaseSequenceEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, base_kwargs, cases):
        super().__init__()
        self.env = BoomerAMGRelaxEnv(**base_kwargs)
        self.cases = list(cases)
        self.case_idx = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self, *, seed=None, options=None):
        if not self.cases:
            raise RuntimeError("No cases provided for online eval.")
        case = self.cases[self.case_idx % len(self.cases)]
        self.case_idx += 1

        if not self.env.randomize_grid:
            self.env.grid_choices = [case["grid"]]
            self.env.fixed_grid = case["grid"]
        self.env.randomize_A = case["randomize_A"]
        self.env.randomize_b = case["randomize_b"]
        self.env.fixed_rhs_seed = case["fixed_rhs_seed"]
        self.env.fixed_rhs_type = case["fixed_rhs_type"]

        obs, info = self.env.reset(seed=case["seed"])
        info = dict(info)
        info["case_idx"] = self.case_idx - 1
        return obs, info

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


class OnlineEpisodeLogger(BaseCallback):
    def __init__(self, target_episodes, verbose=0):
        super().__init__(verbose)
        self.target_episodes = int(target_episodes)
        self.episode_times = []
        self.episode_cycles = []
        self.episode_case_idx = []
        self.update_wall_times = []
        self._time_accum = None
        self._cycles = None
        self._last_rollout_end = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._time_accum = np.zeros(n_envs, dtype=float)
        self._cycles = np.zeros(n_envs, dtype=int)
        self._last_rollout_end = time.perf_counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for env_idx, info in enumerate(infos):
            self._time_accum[env_idx] += float(info.get("dt", 0.0))
            self._cycles[env_idx] += 1
            if env_idx < len(dones) and bool(dones[env_idx]):
                self.episode_times.append(float(self._time_accum[env_idx]))
                self.episode_cycles.append(int(self._cycles[env_idx]))
                self.episode_case_idx.append(int(info.get("case_idx", -1)))
                self._time_accum[env_idx] = 0.0
                self._cycles[env_idx] = 0

        if len(self.episode_times) >= self.target_episodes:
            return False
        return True

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        if self._last_rollout_end is not None:
            self.update_wall_times.append(now - self._last_rollout_end)
        self._last_rollout_end = now


def _make_case_list(seeds, grids, randomize_A, randomize_b, fixed_rhs_seed, fixed_rhs_type):
    return [
        {
            "seed": s,
            "grid": grids[i % len(grids)],
            "randomize_A": randomize_A,
            "randomize_b": randomize_b,
            "fixed_rhs_seed": fixed_rhs_seed,
            "fixed_rhs_type": fixed_rhs_type,
        }
        for i, s in enumerate(seeds)
    ]


def _load_vecnorm(vec_path, raw_env, update_stats=False):
    if not os.path.exists(vec_path):
        return None
    try:
        vec_norm = VecNormalize.load(vec_path, raw_env)
        vec_norm.training = bool(update_stats)
        vec_norm.norm_reward = bool(update_stats)
        return vec_norm
    except AssertionError as exc:
        print(f"VecNormalize mismatch; skipping {vec_path}. Reason: {exc}")
        return None


def _load_model(model_type, path, env=None):
    if model_type == "lstm":
        return RecurrentPPO.load(path, env=env)
    return PPO.load(path, env=env)


def main():
    lib_path = "./libamg_env.dylib"
    seed_start = int(os.environ.get("EVAL_SEED_START", "100"))
    seed_count = int(os.environ.get("EVAL_SEED_COUNT", "6"))
    eval_seeds = list(range(seed_start, seed_start + seed_count))
    grid_sizes = [(60, 60, 60)]

    fixed_a_mode = _env_flag("EVAL_FIXED_A", "0")
    randomize_A = _env_flag("RANDOMIZE_A", "1")
    randomize_b = _env_flag("RANDOMIZE_B", "1")
    fixed_rhs_seed = int(os.environ.get("FIXED_RHS_SEED", "123456789"))
    fixed_rhs_type = int(os.environ.get("FIXED_RHS_TYPE", "1"))
    fixed_stencil = int(os.environ.get("FIXED_STENCIL", "0"))
    a_instances = int(os.environ.get("EVAL_A_INSTANCES", "3"))
    seed_stride = int(os.environ.get("EVAL_SEED_STRIDE", "100000"))
    randomize_grid = _env_flag("RANDOMIZE_GRID", "1")
    grid_min = int(os.environ.get("GRID_MIN", "10"))
    grid_max = int(os.environ.get("GRID_MAX", "80"))
    difconv_c = _parse_triplet_env("DIFCONV_C", "1,100,100")
    difconv_c_range = _parse_range_env("DIFCONV_C_RANGE", "1,1000")
    difconv_a = _parse_triplet_env("DIFCONV_A", "0,0,0")
    difconv_atype = int(os.environ.get("DIFCONV_ATYPE", "0"))
    sweeps_min = int(os.environ.get("SWEEPS_MIN", "1"))
    sweeps_max = int(os.environ.get("SWEEPS_MAX", "3"))
    w_center = float(os.environ.get("W_CENTER", "1.05"))
    w_scale = float(os.environ.get("W_SCALE", "0.25"))
    if fixed_a_mode:
        randomize_A = False
        randomize_b = False
        fixed_rhs_type = 1
        grid_sizes = [(60, 60, 60)]
        eval_seeds = [seed_start]
    elif randomize_A:
        eval_seeds = _expand_seeds(eval_seeds, a_instances, seed_stride)

    if randomize_grid:
        eval_cases = _make_case_list(
            eval_seeds, [(grid_min, grid_min, grid_min)], randomize_A, randomize_b, fixed_rhs_seed, fixed_rhs_type
        )
    else:
        eval_cases = _make_case_list(
            eval_seeds, grid_sizes, randomize_A, randomize_b, fixed_rhs_seed, fixed_rhs_type
        )

    model_type = os.environ.get("MODEL_TYPE", "ppo").strip().lower()
    model_path = os.environ.get("MODEL_PATH", "ppo_boomeramg_gen")
    vec_path = os.environ.get("VEC_PATH", "vecnormalize_gen.pkl")
    update_norm = _env_flag("ONLINE_UPDATE_NORM", "0")

    base_kwargs = dict(
        lib_path=lib_path,
        seed=0,
        fixed_grid=(None if randomize_grid else grid_sizes[0]),
        randomize_A=randomize_A,
        randomize_b=randomize_b,
        fixed_rhs_seed=fixed_rhs_seed,
        fixed_rhs_type=fixed_rhs_type,
        fixed_stencil=fixed_stencil,
        randomize_grid=randomize_grid,
        grid_min=grid_min,
        grid_max=grid_max,
        difconv_c=difconv_c,
        difconv_c_range=difconv_c_range,
        difconv_a=difconv_a,
        difconv_atype=difconv_atype,
        w_center=w_center,
        w_scale=w_scale,
        sweeps_min=sweeps_min,
        sweeps_max=sweeps_max,
    )

    # Offline eval (no learning)
    vec_norm_eval = None
    raw_env = DummyVecEnv([lambda: BoomerAMGRelaxEnv(**base_kwargs)])
    vec_norm_eval = _load_vecnorm(vec_path, raw_env, update_stats=False)

    model = _load_model(model_type, model_path)

    offline_results = []
    offline_wall_start = time.perf_counter()
    for case in eval_cases:
        env = BoomerAMGRelaxEnv(
            lib_path=lib_path,
            seed=case["seed"],
            fixed_grid=(None if randomize_grid else case["grid"]),
            randomize_A=case["randomize_A"],
            randomize_b=case["randomize_b"],
            fixed_rhs_seed=case["fixed_rhs_seed"],
            fixed_rhs_type=case["fixed_rhs_type"],
            fixed_stencil=fixed_stencil,
            randomize_grid=randomize_grid,
            grid_min=grid_min,
            grid_max=grid_max,
            difconv_c=difconv_c,
            difconv_c_range=difconv_c_range,
            difconv_a=difconv_a,
            difconv_atype=difconv_atype,
            w_center=w_center,
            w_scale=w_scale,
            sweeps_min=sweeps_min,
            sweeps_max=sweeps_max,
        )
        out = run_episode(
            env,
            policy=model,
            obs_normalizer=(vec_norm_eval.normalize_obs if vec_norm_eval is not None else None),
            use_lstm=(model_type == "lstm"),
        )
        env.close()
        out["case"] = case
        offline_results.append(out)
    offline_wall_total = time.perf_counter() - offline_wall_start

    # Online eval (learn during eval stream)
    online_env = CaseSequenceEnv(base_kwargs, eval_cases)
    online_vec_env = DummyVecEnv([lambda: online_env])
    vec_norm_online = _load_vecnorm(vec_path, online_vec_env, update_stats=update_norm)
    train_env = vec_norm_online if vec_norm_online is not None else online_vec_env

    online_model = _load_model(model_type, model_path, env=train_env)

    online_episodes = int(os.environ.get("ONLINE_EPISODES", str(len(eval_cases))))
    tmp_env = BoomerAMGRelaxEnv(**base_kwargs)
    max_cycles = int(tmp_env.max_cycles)
    tmp_env.close()
    total_timesteps = int(os.environ.get("ONLINE_TIMESTEPS", str(online_episodes * max_cycles)))

    callback = OnlineEpisodeLogger(target_episodes=online_episodes)
    online_wall_start = time.perf_counter()
    online_model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, callback=callback)
    online_wall_total = time.perf_counter() - online_wall_start

    online_results = []
    for idx, (t, c) in enumerate(zip(callback.episode_times, callback.episode_cycles)):
        case_idx = callback.episode_case_idx[idx] if idx < len(callback.episode_case_idx) else -1
        case = eval_cases[case_idx] if 0 <= case_idx < len(eval_cases) else {}
        online_results.append({"time": t, "cycles": c, "case": case})

    print("\n=== Online vs Offline ===")
    print("Offline (fixed policy):", _summarize(offline_results))
    print("Online  (policy updates):", _summarize(online_results))
    if online_results:
        online_solver_time = float(sum(r["time"] for r in online_results))
        online_overhead = online_wall_total - online_solver_time
        print(
            "Online wall time (incl updates):",
            online_wall_total,
            "| per-episode:",
            online_wall_total / max(1, len(online_results)),
        )
        print(
            "Online overhead (wall - solver dt):",
            online_overhead,
            "| per-episode:",
            online_overhead / max(1, len(online_results)),
        )
    if offline_results:
        print(
            "Offline wall time (loop only):",
            offline_wall_total,
            "| per-episode:",
            offline_wall_total / max(1, len(offline_results)),
        )
    if callback.update_wall_times:
        upd = np.array(callback.update_wall_times, dtype=float)
        print(
            "Mean wall time per policy update:",
            float(upd.mean()),
            "| median:",
            float(np.median(upd)),
        )

    if fixed_a_mode:
        print("Fixed-A mode: ON (A coefficients and size fixed)")
    if randomize_grid:
        print(f"Eval grid range: [{grid_min}, {grid_max}] (uniform, cubic)")


if __name__ == "__main__":
    main()
