from __future__ import annotations

import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.buffers import RolloutBuffer

from explore_bandit_solve_control import _augment_params
from frozen_bandit_step_env import FrozenBanditStepEnv
from setup_aware_compare_common import generate_difconv_instances, solve_no_rl_case


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_float_list(name: str, default: str) -> tuple[float, ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _env_joint_actions(name: str, default: str) -> tuple[tuple[float, int, int], ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    actions = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid joint action spec: {chunk}")
        actions.append((float(parts[0]), int(parts[1]), int(parts[2])))
    return tuple(actions)


def _env_vec3(name: str, default: str) -> tuple[float, float, float]:
    raw = _env_str(name, default).strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must have 3 comma-separated values, got: {raw!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _constant_action_for_w(*, w: float, action_mode: str, w_center: float, w_scale: float, discrete_w_values: tuple[float, ...]):
    if action_mode == "continuous":
        a_w = 0.0 if float(w_scale) <= 0.0 else (float(w) - float(w_center)) / float(w_scale)
        a_w = float(np.clip(a_w, -1.0, 1.0))
        return np.asarray([a_w], dtype=np.float32)
    if action_mode == "discrete_w":
        idx = int(np.argmin([abs(float(w) - float(v)) for v in discrete_w_values]))
        return idx
    raise ValueError(f"Constant scan unsupported for ACTION_MODE={action_mode}")


def _setup_key(params: dict) -> tuple:
    items = []
    for key, value in sorted(dict(params).items()):
        if isinstance(value, float):
            items.append((key, round(float(value), 12)))
        else:
            items.append((key, value))
    return tuple(items)


def _load_fixed_top1_setup(cache_path: Path, warmup: int) -> tuple[dict, int]:
    with cache_path.open("rb") as fh:
        trace = pickle.load(fh)
    suffix = trace[int(warmup):]
    if not suffix:
        raise ValueError(f"Warmup {warmup} leaves empty suffix for {cache_path}")
    counts = Counter(_setup_key(params) for _mkw, params in suffix)
    top_key, top_count = counts.most_common(1)[0]
    fixed_params = {k: v for k, v in top_key}
    return fixed_params, int(top_count)


def _make_trace(
    *,
    T: int,
    seed: int,
    grid: tuple[int, int, int],
    fixed_params: dict,
    c_min: float,
    c_max: float,
    difconv_a: tuple[float, float, float],
):
    instances = generate_difconv_instances(
        T=int(T),
        seed=int(seed),
        grid_choices=[tuple(int(x) for x in grid)],
        c_min=float(c_min),
        c_max=float(c_max),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )
    return [(dict(mkw), dict(fixed_params)) for (mkw, _ctx) in instances]


def _baseline_runtimes(eval_trace: list[tuple[dict, dict]], *, solver_tol: float, solver_max_iter: int) -> list[float]:
    rows = [
        solve_no_rl_case(
            params=dict(params),
            mkw=dict(mkw),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
            augment_params=_augment_params,
        )
        for (mkw, params) in eval_trace
    ]
    return [float(r["runtime"]) for r in rows]


def _make_env(
    trace,
    *,
    tune_dim: int,
    c_max: float,
    solve_tol: float,
    solve_max_cycles: int,
    w_only: bool,
    w_center: float,
    w_scale: float,
    action_mode: str,
    discrete_w_values: tuple[float, ...],
    discrete_joint_actions: tuple[tuple[float, int, int], ...],
    sweeps_min: int,
    sweeps_max: int,
    reward_mode: int,
    term_bonus: float,
    trunc_penalty: float,
    relative_bonus_scale: float,
    seed: int,
    baseline_runtimes,
):
    return FrozenBanditStepEnv(
        trace=trace,
        tune_dim=int(tune_dim),
        tune7_variant="categorical",
        c_max=float(c_max),
        tol=float(solve_tol),
        max_cycles=int(solve_max_cycles),
        w_only=bool(w_only),
        action_mode=str(action_mode),
        discrete_w_values=tuple(float(x) for x in discrete_w_values),
        discrete_joint_actions=tuple((float(w), int(sd), int(su)) for (w, sd, su) in discrete_joint_actions),
        w_center=float(w_center),
        w_scale=float(w_scale),
        sweeps_min=int(sweeps_min),
        sweeps_max=int(sweeps_max),
        reward_mode=int(reward_mode),
        term_bonus=float(term_bonus),
        trunc_penalty=float(trunc_penalty),
        relative_bonus_scale=float(relative_bonus_scale),
        obs_mode="solve_only",
        baseline_runtimes=baseline_runtimes,
        seed=int(seed),
    )


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int, fractions: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), verbose: int = 0):
        super().__init__(verbose)
        self.total_timesteps = max(1, int(total_timesteps))
        self.targets = [int(self.total_timesteps * frac) for frac in fractions]
        self.index = 0
        self.start_time = 0.0

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        while self.index < len(self.targets) and self.num_timesteps >= self.targets[self.index]:
            frac = float(self.targets[self.index]) / float(self.total_timesteps)
            print(
                json.dumps(
                    {
                        "stage": "train_progress",
                        "timesteps": int(self.num_timesteps),
                        "progress_pct": round(100.0 * frac, 1),
                        "elapsed_sec": float(time.time() - self.start_time),
                    },
                    indent=2,
                ),
                flush=True,
            )
            self.index += 1
        return True


def _rollout_eval(eval_venv, policy_fn, eval_count: int):
    obs = eval_venv.reset()
    runtimes = []
    iterations = []
    final_ws = []
    per_ep_mean_abs_dw = []
    per_ep_std_w = []
    action_hist = Counter()
    start = time.time()
    for i in range(eval_count):
        done = False
        truncated = False
        w_seq = []
        prev_w = None
        abs_dw = []
        while not (done or truncated):
            action = policy_fn(obs)
            obs, _reward, done_arr, info = eval_venv.step(action)
            done = bool(done_arr[0])
            truncated = False
            info0 = info[0]
            w_cur = float(info0["w"])
            w_seq.append(w_cur)
            if prev_w is not None:
                abs_dw.append(abs(w_cur - prev_w))
            prev_w = w_cur
            action_idx = int(info0.get("action_idx", -1))
            if action_idx >= 0:
                action_hist[action_idx] += 1
        info0 = info[0]
        runtimes.append(float(info0["time_with_setup"]))
        iterations.append(int(info0["cycle"]))
        final_ws.append(float(info0["w"]))
        per_ep_mean_abs_dw.append(float(mean(abs_dw)) if abs_dw else 0.0)
        per_ep_std_w.append(float(np.std(np.asarray(w_seq, dtype=np.float32))) if w_seq else 0.0)
        if (i + 1) % 100 == 0:
            print(
                json.dumps(
                    {
                        "stage": "eval_progress",
                        "done": int(i + 1),
                        "elapsed_sec": float(time.time() - start),
                    },
                    indent=2,
                ),
                flush=True,
            )
    return {
        "runtimes": runtimes,
        "iterations": iterations,
        "final_ws": final_ws,
        "mean_abs_dw": per_ep_mean_abs_dw,
        "std_w": per_ep_std_w,
        "action_hist": {str(k): int(v) for k, v in sorted(action_hist.items())},
    }


def main() -> None:
    cache_path = Path(
        os.environ.get(
            "TRACE_CACHE_PATH",
            "/tmp/frozen_bandit_trace_cache/41de2a5973cc59f0dc2310ae6a6fdd8db8539017b7f553ab627409023021491e.pkl",
        )
    )
    warmup = _env_int("WARMUP_CASES", 300)
    grid_n = _env_int("GRID_N_FIXED", 60)
    train_cases = _env_int("TRAIN_CASES", 10000)
    eval_cases = _env_int("EVAL_CASES", 1000)
    train_seed = _env_int("TRAIN_SEED", 39393939)
    eval_seed = _env_int("EVAL_SEED", 39395939)
    difconv_a = _env_vec3("DIFCONV_A", "0,0,0")
    tune_dim = _env_int("SETUP_TUNE_DIM", 5)
    c_min = _env_float("C_MIN", 1.0)
    c_max = _env_float("C_MAX", 1000.0)
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    solve_max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    solver_tol = _env_float("SOLVER_TOL", 1e-6)
    solver_max_iter = _env_int("SOLVER_MAX_ITER", 50)
    w_only = _env_bool("W_ONLY", True)
    w_center = _env_float("W_CENTER", 1.25)
    w_scale = _env_float("W_SCALE", 0.75)
    action_mode = _env_str("ACTION_MODE", "continuous").strip().lower()
    discrete_w_values = _env_float_list("DISCRETE_W_VALUES", "1.3,1.4,1.5,1.6,1.7")
    discrete_joint_actions = _env_joint_actions("DISCRETE_JOINT_ACTIONS", "1.5:1:1;1.6:1:1;1.7:1:1;1.5:1:2;1.6:1:2;1.6:2:1")
    sweeps_min = _env_int("SWEEPS_MIN", 1)
    sweeps_max = _env_int("SWEEPS_MAX", 1)
    algo = _env_str("ALGO", "ppo").strip().lower()
    total_timesteps = _env_int("TOTAL_TIMESTEPS", 320000)
    reward_mode = _env_int("REWARD_MODE", 1)
    term_bonus = _env_float("TERM_BONUS", 0.0)
    trunc_penalty = _env_float("TRUNC_PENALTY", 0.0)
    relative_bonus_scale = _env_float("RELATIVE_BONUS_SCALE", 1.0)
    learning_rate = _env_float("LEARNING_RATE", 3e-4)
    ent_coef = _env_float("ENT_COEF", 0.0)
    n_steps = _env_int("N_STEPS", 128)
    batch_size = _env_int("BATCH_SIZE", 128)
    n_epochs = _env_int("N_EPOCHS", 10)
    learning_starts = _env_int("LEARNING_STARTS", 1000)
    train_freq = _env_int("TRAIN_FREQ", 4)
    gradient_steps = _env_int("GRADIENT_STEPS", 1)
    target_update_interval = _env_int("TARGET_UPDATE_INTERVAL", 1000)
    exploration_fraction = _env_float("EXPLORATION_FRACTION", 0.2)
    exploration_final_eps = _env_float("EXPLORATION_FINAL_EPS", 0.05)
    compute_train_baseline = _env_bool("COMPUTE_TRAIN_BASELINE", reward_mode == 4 and train_cases <= 2000)
    eval_constant_w_values = _env_float_list("EVAL_CONSTANT_W_VALUES", "")
    save_model_path = _env_str("SAVE_MODEL_PATH", "").strip()
    save_vec_path = _env_str("SAVE_VEC_PATH", "").strip()
    init_model_path = _env_str("INIT_MODEL_PATH", "").strip()
    init_vec_path = _env_str("INIT_VEC_PATH", "").strip()
    skip_train = _env_bool("SKIP_TRAIN", False)

    if not cache_path.exists():
        raise FileNotFoundError(f"TRACE_CACHE_PATH does not exist: {cache_path}")
    if algo not in {"ppo", "dqn"}:
        raise ValueError(f"Unsupported ALGO={algo}")
    if algo == "dqn" and action_mode not in {"discrete_w", "discrete_joint"}:
        raise ValueError("ALGO=dqn requires ACTION_MODE=discrete_w or discrete_joint")

    fixed_params, top_count = _load_fixed_top1_setup(cache_path, warmup=warmup)
    print(
        json.dumps(
            {
                "stage": "fixed_setup_selected",
                "trace_cache_path": str(cache_path),
                "warmup_cases": int(warmup),
                "top1_count_in_mature_suffix": int(top_count),
                "fixed_setup_params": fixed_params,
            },
            indent=2,
        ),
        flush=True,
    )

    grid = (int(grid_n), int(grid_n), int(grid_n))
    train_trace = _make_trace(
        T=train_cases,
        seed=train_seed,
        grid=grid,
        fixed_params=fixed_params,
        c_min=c_min,
        c_max=c_max,
        difconv_a=difconv_a,
    )
    eval_trace = _make_trace(
        T=eval_cases,
        seed=eval_seed,
        grid=grid,
        fixed_params=fixed_params,
        c_min=c_min,
        c_max=c_max,
        difconv_a=difconv_a,
    )

    eval_baseline_runtimes = _baseline_runtimes(
        eval_trace,
        solver_tol=solver_tol,
        solver_max_iter=solver_max_iter,
    )
    bandit_only_mean = mean(eval_baseline_runtimes)
    print(
        json.dumps(
            {
                "stage": "baseline_done",
                "eval_cases": int(eval_cases),
                "bandit_only_mean": float(bandit_only_mean),
            },
            indent=2,
        ),
        flush=True,
    )
    if compute_train_baseline:
        train_baseline_runtimes = _baseline_runtimes(
            train_trace,
            solver_tol=solver_tol,
            solver_max_iter=solver_max_iter,
        )
        print(
            json.dumps(
                {
                    "stage": "train_baseline_done",
                    "train_cases": int(train_cases),
                    "train_bandit_only_mean": float(mean(train_baseline_runtimes)),
                },
                indent=2,
            ),
            flush=True,
        )
    else:
        train_baseline_runtimes = [float("nan")] * len(train_trace)

    train_venv = DummyVecEnv(
        [
            lambda: _make_env(
                train_trace,
                tune_dim=tune_dim,
                c_max=c_max,
                solve_tol=solve_tol,
                solve_max_cycles=solve_max_cycles,
                w_only=w_only,
                w_center=w_center,
                w_scale=w_scale,
                action_mode=action_mode,
                discrete_w_values=discrete_w_values,
                discrete_joint_actions=discrete_joint_actions,
                sweeps_min=sweeps_min,
                sweeps_max=sweeps_max,
                reward_mode=reward_mode,
                term_bonus=term_bonus,
                trunc_penalty=trunc_penalty,
                relative_bonus_scale=relative_bonus_scale,
                baseline_runtimes=train_baseline_runtimes,
                seed=0,
            )
        ]
    )
    train_venv = VecMonitor(train_venv)
    if init_vec_path:
        train_venv = VecNormalize.load(init_vec_path, train_venv)
        train_venv.training = True
        train_venv.norm_reward = False
    else:
        train_venv = VecNormalize(train_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    if algo == "ppo":
        if init_model_path:
            model = PPO.load(init_model_path, env=train_venv, device="cpu")
            model.learning_rate = learning_rate
            model.lr_schedule = lambda _progress_remaining: learning_rate
            model.n_steps = n_steps
            model.batch_size = batch_size
            model.n_epochs = n_epochs
            model.ent_coef = ent_coef
            model.rollout_buffer = RolloutBuffer(
                model.n_steps,
                model.observation_space,
                model.action_space,
                device=model.device,
                gae_lambda=model.gae_lambda,
                gamma=model.gamma,
                n_envs=model.n_envs,
            )
        else:
            model = PPO(
                "MlpPolicy",
                train_venv,
                learning_rate=learning_rate,
                gamma=0.99,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=ent_coef,
                vf_coef=0.5,
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                verbose=1,
                device="cpu",
            )
    else:
        if init_model_path:
            model = DQN.load(init_model_path, env=train_venv, device="cpu")
            model.learning_rate = learning_rate
            model.batch_size = batch_size
            model.learning_starts = learning_starts
            model.train_freq = train_freq
            model.gradient_steps = gradient_steps
            model.target_update_interval = target_update_interval
            model.exploration_fraction = exploration_fraction
            model.exploration_final_eps = exploration_final_eps
        else:
            model = DQN(
                "MlpPolicy",
                train_venv,
                learning_rate=learning_rate,
                batch_size=batch_size,
                learning_starts=learning_starts,
                train_freq=train_freq,
                gradient_steps=gradient_steps,
                target_update_interval=target_update_interval,
                exploration_fraction=exploration_fraction,
                exploration_final_eps=exploration_final_eps,
                policy_kwargs={"net_arch": [128, 128]},
                verbose=1,
                device="cpu",
            )

    print(
        json.dumps(
            {
                "stage": "train_start",
                "train_cases": int(train_cases),
                "total_timesteps": int(total_timesteps),
                "algo": str(algo),
                "w_only": bool(w_only),
                "action_mode": str(action_mode),
                "discrete_w_values": list(discrete_w_values),
                "discrete_joint_actions": [list(spec) for spec in discrete_joint_actions],
                "sweeps_min": int(sweeps_min),
                "sweeps_max": int(sweeps_max),
                "reward_mode": int(reward_mode),
                "difconv_a": [float(x) for x in difconv_a],
                "term_bonus": float(term_bonus),
                "trunc_penalty": float(trunc_penalty),
                "relative_bonus_scale": float(relative_bonus_scale),
                "learning_rate": float(learning_rate),
                "ent_coef": float(ent_coef),
                "n_steps": int(n_steps),
                "batch_size": int(batch_size),
                "n_epochs": int(n_epochs),
                "learning_starts": int(learning_starts),
                "train_freq": int(train_freq),
                "gradient_steps": int(gradient_steps),
                "target_update_interval": int(target_update_interval),
                "exploration_fraction": float(exploration_fraction),
                "exploration_final_eps": float(exploration_final_eps),
                "compute_train_baseline": bool(compute_train_baseline),
                "init_model_path": init_model_path,
                "init_vec_path": init_vec_path,
                "skip_train": bool(skip_train),
            },
            indent=2,
        ),
        flush=True,
    )
    if not skip_train:
        model.learn(total_timesteps=total_timesteps, callback=ProgressCallback(total_timesteps))
        if save_model_path:
            model.save(save_model_path)
        if save_vec_path:
            train_venv.save(save_vec_path)
        print(json.dumps({"stage": "train_done", "save_model_path": save_model_path, "save_vec_path": save_vec_path}, indent=2), flush=True)
    else:
        print(json.dumps({"stage": "train_skipped"}, indent=2), flush=True)

    eval_venv = DummyVecEnv(
        [
            lambda: _make_env(
                eval_trace,
                tune_dim=tune_dim,
                c_max=c_max,
                solve_tol=solve_tol,
                solve_max_cycles=solve_max_cycles,
                w_only=w_only,
                w_center=w_center,
                w_scale=w_scale,
                action_mode=action_mode,
                discrete_w_values=discrete_w_values,
                discrete_joint_actions=discrete_joint_actions,
                sweeps_min=sweeps_min,
                sweeps_max=sweeps_max,
                reward_mode=reward_mode,
                term_bonus=term_bonus,
                trunc_penalty=trunc_penalty,
                relative_bonus_scale=relative_bonus_scale,
                baseline_runtimes=eval_baseline_runtimes,
                seed=123,
            )
        ]
    )
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_venv.obs_rms = train_venv.obs_rms
    eval_venv.training = False
    eval_venv.norm_reward = False

    rollout = _rollout_eval(
        eval_venv,
        policy_fn=lambda obs: model.predict(obs, deterministic=True)[0],
        eval_count=len(eval_trace),
    )

    constant_scan = {}
    if eval_constant_w_values and action_mode in {"continuous", "discrete_w"} and bool(w_only):
        for w_const in eval_constant_w_values:
            eval_const = DummyVecEnv(
                [
                    lambda: _make_env(
                        eval_trace,
                        tune_dim=tune_dim,
                        c_max=c_max,
                        solve_tol=solve_tol,
                        solve_max_cycles=solve_max_cycles,
                        w_only=w_only,
                        w_center=w_center,
                        w_scale=w_scale,
                        action_mode=action_mode,
                        discrete_w_values=discrete_w_values,
                        discrete_joint_actions=discrete_joint_actions,
                        sweeps_min=sweeps_min,
                        sweeps_max=sweeps_max,
                        reward_mode=reward_mode,
                        term_bonus=term_bonus,
                        trunc_penalty=trunc_penalty,
                        relative_bonus_scale=relative_bonus_scale,
                        baseline_runtimes=eval_baseline_runtimes,
                        seed=123,
                    )
                ]
            )
            eval_const = VecNormalize(eval_const, norm_obs=True, norm_reward=False, clip_obs=10.0)
            eval_const.obs_rms = train_venv.obs_rms
            eval_const.training = False
            eval_const.norm_reward = False
            const_rollout = _rollout_eval(
                eval_const,
                policy_fn=lambda _obs, wc=float(w_const): _constant_action_for_w(
                    w=wc,
                    action_mode=action_mode,
                    w_center=w_center,
                    w_scale=w_scale,
                    discrete_w_values=discrete_w_values,
                ),
                eval_count=len(eval_trace),
            )
            constant_scan[str(w_const)] = {
                "mean_runtime": float(mean(const_rollout["runtimes"])),
                "delta_vs_bandit_only": float(mean(const_rollout["runtimes"]) - bandit_only_mean),
                "mean_iterations": float(mean(const_rollout["iterations"])),
                "mean_final_w": float(mean(const_rollout["final_ws"])),
            }
            eval_const.close()

    print(
        json.dumps(
            {
                "stage": "final",
                "fixed_setup_top1_count_in_mature_suffix": int(top_count),
                "fixed_setup_params": fixed_params,
                "train_cases": int(train_cases),
                "eval_cases": int(eval_cases),
                "bandit_only_mean": float(bandit_only_mean),
                "rl_mean": float(mean(rollout["runtimes"])),
                "delta_vs_bandit_only": float(mean(rollout["runtimes"]) - bandit_only_mean),
                "mean_iterations": float(mean(rollout["iterations"])),
                "mean_final_w": float(mean(rollout["final_ws"])),
                "mean_abs_delta_w_per_episode": float(mean(rollout["mean_abs_dw"])),
                "mean_std_w_per_episode": float(mean(rollout["std_w"])),
                "action_hist": rollout["action_hist"],
                "constant_w_scan": constant_scan,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
