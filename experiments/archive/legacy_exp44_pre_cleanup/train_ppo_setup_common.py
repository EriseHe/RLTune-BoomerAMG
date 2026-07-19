from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor, VecNormalize

from amg_setup_gym_env import BoomerAMGSetupRelaxEnv, SETUP_OBS_KEYS
from custom_policy import CustomPPOPolicy
from paper_policy import PaperPPOPolicy


def _parse_grid_sizes(spec):
    if not spec:
        return None
    sizes = []
    for part in spec.split(";"):
        clean = part.replace("x", ",")
        nums = [n for n in clean.split(",") if n.strip()]
        if len(nums) != 3:
            continue
        sizes.append(tuple(int(n) for n in nums))
    return sizes if sizes else None


def _parse_grid_range(spec):
    if not spec:
        return None
    vals = [v for v in spec.replace("x", ",").split(",") if v.strip()]
    if len(vals) != 3:
        return None
    start, stop, step = (int(v) for v in vals)
    return [(n, n, n) for n in range(start, stop + 1, step)]


def _env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


def _env_optional_float(name: str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def _env_optional_int(name: str):
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    return int(raw)


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


class SetupStepLoggerCallback(BaseCallback):
    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._episode_steps = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self._episode_steps = np.zeros(n_envs, dtype=int)
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "global_step",
                "env_idx",
                "episode_id",
                "episode_step",
                "cycle",
                "r",
                "r_prev",
                "rel_drop",
                "dt_solver",
                "setup_time",
                "solve_time",
                "time_with_setup",
                "w",
                "sweeps_down",
                "sweeps_up",
                "cx",
                "cy",
                "cz",
                *SETUP_OBS_KEYS,
                "reward",
                "action_raw",
            ])

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        actions = self.locals.get("actions")
        dones = self.locals["dones"]
        self.logger.record("train/mean_step_reward", float(np.mean(rewards)))
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for env_idx, info in enumerate(infos):
                self._episode_steps[env_idx] += 1
                row = [
                    self.num_timesteps,
                    env_idx,
                    info.get("episode_id", -1),
                    self._episode_steps[env_idx],
                    info.get("cycle", -1),
                    info.get("r", np.nan),
                    info.get("r_prev", np.nan),
                    info.get("rel_drop", np.nan),
                    info.get("dt_solver", np.nan),
                    info.get("setup_time", np.nan),
                    info.get("solve_time", np.nan),
                    info.get("time_with_setup", np.nan),
                    info.get("w", np.nan),
                    info.get("sweeps_down", -1),
                    info.get("sweeps_up", -1),
                    info.get("cx", np.nan),
                    info.get("cy", np.nan),
                    info.get("cz", np.nan),
                ]
                row.extend(info.get(k, np.nan) for k in SETUP_OBS_KEYS)
                row.append(float(rewards[env_idx]))
                row.append(actions[env_idx].tolist() if actions is not None else None)
                writer.writerow(row)
        for env_idx, done in enumerate(dones):
            if done:
                self._episode_steps[env_idx] = 0
        return True


class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, update_every=1000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = int(total_timesteps)
        self.update_every = int(update_every)
        self._t0 = None
        self._last = 0

    def _on_training_start(self) -> None:
        self._t0 = time.perf_counter()
        self._last = 0

    def _on_step(self) -> bool:
        if self.total_timesteps <= 0:
            return True
        if (self.num_timesteps - self._last) < self.update_every and self.num_timesteps < self.total_timesteps:
            return True
        self._last = int(self.num_timesteps)
        elapsed = max(1e-6, time.perf_counter() - (self._t0 or time.perf_counter()))
        frac = min(self.num_timesteps / self.total_timesteps, 1.0)
        rate = self.num_timesteps / elapsed
        remaining = (self.total_timesteps - self.num_timesteps) / rate if rate > 0 else float("inf")
        bar_len = 28
        filled = int(round(bar_len * frac))
        bar = "#" * filled + "-" * (bar_len - filled)
        msg = (
            f"\r[{bar}] {self.num_timesteps}/{self.total_timesteps} "
            f"({frac * 100:5.1f}%) | {rate:,.1f} steps/s | ETA {remaining / 60:5.1f} min"
        )
        sys.stdout.write(msg)
        sys.stdout.flush()
        if self.num_timesteps >= self.total_timesteps:
            sys.stdout.write("\n")
            sys.stdout.flush()
        return True


def make_env(rank: int, *, seed: int, setup_mode: str):
    grid_sizes = [(60, 60, 60)]
    grid_sizes_env = _parse_grid_sizes(os.environ.get("GRID_SIZES", ""))
    grid_range_env = _parse_grid_range(os.environ.get("GRID_RANGE", ""))
    if grid_sizes_env:
        grid_sizes = grid_sizes_env
    elif grid_range_env:
        grid_sizes = grid_range_env
    randomize_A = _env_flag("RANDOMIZE_A", "1")
    randomize_b = _env_flag("RANDOMIZE_B", "1")
    fixed_rhs_seed = int(os.environ.get("FIXED_RHS_SEED", "123456789"))
    randomize_grid = _env_flag("RANDOMIZE_GRID", "1")
    grid_min = int(os.environ.get("GRID_MIN", "10"))
    grid_max = int(os.environ.get("GRID_MAX", "80"))
    use_grid_bias = _env_flag("USE_GRID_BIAS", "0")
    grid_bias = float(os.environ.get("GRID_BIAS", "1.0"))
    difconv_c = _parse_triplet_env("DIFCONV_C", "1,100,100")
    difconv_c_range = _parse_range_env("DIFCONV_C_RANGE", "1,1000")
    difconv_a = _parse_triplet_env("DIFCONV_A", "0,0,0")
    cycle_penalty = float(os.environ.get("CYCLE_PENALTY", "0.0"))
    sweep_penalty = float(os.environ.get("SWEEP_PENALTY", "0.0"))
    w_center = float(os.environ.get("W_CENTER", "1.05"))
    w_scale = float(os.environ.get("W_SCALE", "0.25"))
    sweeps_min = int(os.environ.get("SWEEPS_MIN", "1"))
    sweeps_max = int(os.environ.get("SWEEPS_MAX", "5"))
    w_only = _env_flag("W_ONLY", "0")
    w_init = _env_optional_float("W_INIT")
    sweeps_init = _env_optional_int("SWEEPS_INIT")
    tune_dim = int(os.environ.get("SETUP_TUNE_DIM", "5"))
    tune7_variant = os.environ.get("TUNE7_VARIANT", "categorical").strip().lower()
    bandit_update = _env_flag("SETUP_BANDIT_UPDATE", "1")

    def _init():
        return BoomerAMGSetupRelaxEnv(
            setup_mode=setup_mode,
            tune_dim=tune_dim,
            tune7_variant=tune7_variant,
            fixed_grid=None if randomize_grid else grid_sizes,
            tol=float(os.environ.get("SOLVE_TOL", "1e-6")),
            max_cycles=int(os.environ.get("SOLVE_MAX_CYCLES", "50")),
            seed=seed + rank,
            w_center=w_center,
            w_scale=w_scale,
            w_init=w_init,
            w_only=w_only,
            sweeps_min=sweeps_min,
            sweeps_max=sweeps_max,
            sweeps_init=sweeps_init,
            randomize_A=randomize_A,
            randomize_b=randomize_b,
            fixed_rhs_seed=fixed_rhs_seed,
            randomize_grid=randomize_grid,
            grid_min=grid_min,
            grid_max=grid_max,
            use_grid_bias=use_grid_bias,
            grid_bias=grid_bias,
            difconv_c=difconv_c,
            difconv_c_range=difconv_c_range,
            difconv_a=difconv_a,
            cycle_penalty=cycle_penalty,
            sweep_penalty=sweep_penalty,
            bandit_update=bandit_update,
            setup_failure_penalty_multiplier=float(os.environ.get("SETUP_BANDIT_FAILURE_PENALTY_MULTIPLIER", "2.0")),
            setup_failure_min_runtime_sec=float(os.environ.get("SETUP_BANDIT_FAILURE_MIN_RUNTIME_SEC", "1e-3")),
        )

    return _init


def main(*, setup_mode: str, default_model_basename: str, default_vec_name: str, default_tb_dir: str) -> None:
    seed = int(os.environ.get("SEED", "0"))
    set_random_seed(seed)

    n_envs_raw = os.environ.get("N_ENVS", "").strip()
    if n_envs_raw:
        n_envs = int(n_envs_raw)
    else:
        n_envs = 1 if setup_mode == "bandit" else 8
    use_subproc = bool(n_envs > 1)

    if use_subproc:
        venv = SubprocVecEnv([make_env(i, seed=seed, setup_mode=setup_mode) for i in range(n_envs)])
    else:
        venv = DummyVecEnv([make_env(0, seed=seed, setup_mode=setup_mode)])

    venv = VecMonitor(venv)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)

    sample_env = BoomerAMGSetupRelaxEnv(
        setup_mode=setup_mode,
        tune_dim=int(os.environ.get("SETUP_TUNE_DIM", "5")),
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        randomize_grid=_env_flag("RANDOMIZE_GRID", "1"),
        grid_min=int(os.environ.get("GRID_MIN", "10")),
        grid_max=int(os.environ.get("GRID_MAX", "80")),
    )
    max_cycles = int(sample_env.max_cycles)
    sample_env.close()

    total_timesteps = int(os.environ.get("TOTAL_TIMESTEPS", str(700 * max_cycles)))
    device = "cuda" if th.cuda.is_available() else "cpu"
    print(f"Device: {device} | n_envs: {n_envs} | setup_mode: {setup_mode}")

    model_type = os.environ.get("MODEL_TYPE", "mlp").strip().lower()
    ent_coef = float(os.environ.get("ENT_COEF", "0.0"))
    tb_dir = os.environ.get("TENSORBOARD_DIR", default_tb_dir)
    common_kwargs = dict(
        env=venv,
        n_steps=int(os.environ.get("N_STEPS", "16")),
        batch_size=int(os.environ.get("BATCH_SIZE", "16")),
        gamma=float(os.environ.get("GAMMA", "0.99")),
        clip_range=float(os.environ.get("CLIP_RANGE", "0.2")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "3e-4")),
        n_epochs=int(os.environ.get("N_EPOCHS", "10")),
        gae_lambda=float(os.environ.get("GAE_LAMBDA", "0.95")),
        vf_coef=float(os.environ.get("VF_COEF", "0.5")),
        ent_coef=ent_coef,
        max_grad_norm=float(os.environ.get("MAX_GRAD_NORM", "0.5")),
        verbose=1,
        device=device,
        tensorboard_log=tb_dir,
    )

    if model_type == "custom":
        model = PPO(
            policy=CustomPPOPolicy,
            policy_kwargs={
                "pi_dim": int(os.environ.get("CUSTOM_PI_DIM", "256")),
                "vf_dim": int(os.environ.get("CUSTOM_VF_DIM", "128")),
                "n_blocks": int(os.environ.get("CUSTOM_BLOCKS", "2")),
            },
            **common_kwargs,
        )
    elif model_type == "paper":
        model = PPO(policy=PaperPPOPolicy, **common_kwargs)
    elif model_type == "mlp":
        model = PPO(policy="MlpPolicy", **common_kwargs)
    else:
        model = RecurrentPPO("MlpLstmPolicy", **common_kwargs)

    log_dir = Path(os.environ.get("TRAIN_LOG_DIR", "logs"))
    step_logger = SetupStepLoggerCallback(log_path=str(log_dir / f"train_steps_{setup_mode}.csv"))
    progress_cb = ProgressCallback(total_timesteps, update_every=int(os.environ.get("PROGRESS_EVERY", "1000")))
    callbacks = CallbackList([step_logger, progress_cb])

    tb_log_name = os.environ.get("TB_LOG_NAME", f"ppo_setup_{setup_mode}")
    model.learn(total_timesteps=total_timesteps, callback=callbacks, tb_log_name=tb_log_name)

    model_basename = os.environ.get("MODEL_BASENAME", default_model_basename)
    vec_name = os.environ.get("VECNORMALIZE_NAME", default_vec_name)
    model.save(model_basename)
    venv.save(vec_name)
    print(f"Saved model to {model_basename}.zip")
    print(f"Saved VecNormalize to {vec_name}")

    venv.close()


__all__ = ["main"]
