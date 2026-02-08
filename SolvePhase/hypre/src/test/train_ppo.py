import os
import numpy as np
import torch as th
import csv  

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from paper_policy import PaperPPOPolicy
from SolvePhase.hypre.src.test.custom_policy import CustomPPOPolicy

from SolvePhase.hypre.src.test.amg_gym_env import BoomerAMGRelaxEnv


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
    sizes = []
    for n in range(start, stop + 1, step):
        sizes.append((n, n, n))
    return sizes if sizes else None


def _env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


class StepLoggerCallback(BaseCallback):
    """
    Logs one row per environment step to a CSV file.

    Columns:
      global_step, env_idx, episode_id, episode_step,
      cycle, r, r_prev, rel_drop, dt,
      w, sweeps_down, sweeps_up,
      a0, a1, a2, a3,
      reward
    """

    def __init__(self, log_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = log_path
        self._episode_steps = None

    def _on_training_start(self) -> None:
        # Per-env episode step counters
        n_envs = self.training_env.num_envs
        self._episode_steps = np.zeros(n_envs, dtype=int)

        # Make directory & header
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
                "dt",
                "w",
                "sweeps_down",
                "sweeps_up",
                "a0",
                "a1",
                "a2",
                "a3",
                "reward",
                "action_raw",
            ])

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        rewards = self.locals["rewards"]
        actions = self.locals.get("actions", None)
        dones = self.locals["dones"]

        # --- NEW: log mean step reward to TensorBoard ---
        self.logger.record("train/mean_step_reward", float(np.mean(rewards)))

        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            for env_idx, info in enumerate(infos):
                self._episode_steps[env_idx] += 1

                episode_id = info.get("episode_id", -1)
                cycle = info.get("cycle", -1)
                r = info.get("r", np.nan)
                r_prev = info.get("r_prev", np.nan)
                rel_drop = info.get("rel_drop", np.nan)
                dt = info.get("dt", np.nan)
                w = info.get("w", np.nan)
                sweeps_down = info.get("sweeps_down", -1)
                sweeps_up = info.get("sweeps_up", -1)
                a0 = info.get("a0", np.nan)
                a1 = info.get("a1", np.nan)
                a2 = info.get("a2", np.nan)
                a3 = info.get("a3", np.nan)

                if actions is not None:
                    action_raw = actions[env_idx].tolist()
                else:
                    action_raw = None

                writer.writerow([
                    self.num_timesteps,
                    env_idx,
                    episode_id,
                    self._episode_steps[env_idx],
                    cycle,
                    r,
                    r_prev,
                    rel_drop,
                    dt,
                    w,
                    sweeps_down,
                    sweeps_up,
                    a0,
                    a1,
                    a2,
                    a3,
                    float(rewards[env_idx]),
                    action_raw,
                ])

        for env_idx, done in enumerate(dones):
            if done:
                self._episode_steps[env_idx] = 0

        return True

class EvalTrajCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=50_000, save_dir="./eval_traj", verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env  # VecNormalize(DummyVecEnv([...])), n_envs=1
        self.eval_freq = eval_freq
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def _get_reset_info0(self):
        """
        SB3 VecEnv reset() 通常只返回 obs，但会把 info 存在 reset_infos 里。
        VecNormalize 包了一层，所以 info 在 self.eval_env.venv.reset_infos。
        """
        # Try common places safely
        if hasattr(self.eval_env, "reset_infos"):
            infos = self.eval_env.reset_infos
            if infos and isinstance(infos, (list, tuple)):
                return infos[0]
        if hasattr(self.eval_env, "venv") and hasattr(self.eval_env.venv, "reset_infos"):
            infos = self.eval_env.venv.reset_infos
            if infos and isinstance(infos, (list, tuple)):
                return infos[0]
        return {}

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # VecEnv reset: usually returns obs only (batched obs shape: (1, obs_dim))
        obs = self.eval_env.reset()
        info0 = self._get_reset_info0()

        done = False
        residuals = [info0.get("r0", np.nan)]
        ws, downs, ups, dts = [], [], [], []

        # ✅ Recurrent state handling
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)  # n_envs=1

        while not done:
            action, lstm_states = self.model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True,
            )

            # VecEnv step: obs, rewards, dones, infos
            obs, rewards, dones, infos = self.eval_env.step(action)
            done = bool(dones[0])
            inf = infos[0]
            # reward = float(rewards[0])  # 如果你想记录 reward 也可以用

            episode_start = np.array([done], dtype=bool)

            residuals.append(inf.get("r", np.nan))
            ws.append(inf.get("w", np.nan))
            downs.append(inf.get("sweeps_down", -1))
            ups.append(inf.get("sweeps_up", -1))
            dts.append(inf.get("dt", np.nan))

        cycles = len(residuals) - 1
        total_time = float(np.nansum(dts))

        self.logger.record("eval/episode_cycles", cycles)
        self.logger.record("eval/episode_time", total_time)

        step_id = self.num_timesteps
        np.savez(
            os.path.join(self.save_dir, f"traj_{step_id}.npz"),
            residuals=np.array(residuals),
            ws=np.array(ws),
            downs=np.array(downs),
            ups=np.array(ups),
            dts=np.array(dts),
        )

        if self.verbose > 0:
            w0 = ws[0] if len(ws) > 0 else np.nan
            print(
                f"[EvalTrajCallback] t={step_id} | cycles={cycles}, "
                f"time={total_time:.4f}s, w[0]={w0:.3f}"
            )

        return True


def make_env(rank, seed=0):
    # Matrix size fixed at 60^3 for all training runs
    grid_sizes = [(60, 60, 60)]
    randomize_A = _env_flag("RANDOMIZE_A", "1")
    randomize_b = _env_flag("RANDOMIZE_B", "1")
    fixed_rhs_seed = int(os.environ.get("FIXED_RHS_SEED", "123456789"))
    fixed_rhs_type = int(os.environ.get("FIXED_RHS_TYPE", "1"))

    if not randomize_A:
        grid_sizes = [(60, 60, 60)]
        randomize_b = False
        fixed_rhs_type = 1

    def _init():
        kwargs = {}
        if grid_sizes:
            kwargs["fixed_grid"] = grid_sizes
        kwargs["randomize_A"] = randomize_A
        kwargs["randomize_b"] = randomize_b
        kwargs["fixed_rhs_seed"] = fixed_rhs_seed
        kwargs["fixed_rhs_type"] = fixed_rhs_type
        env = BoomerAMGRelaxEnv(
            lib_path="./libamg_env.dylib",
            seed=seed + rank,
            **kwargs,
        )
        return env
    return _init


def main():
    seed = 0
    set_random_seed(seed)

    randomize_A = _env_flag("RANDOMIZE_A", "1")
    randomize_b = _env_flag("RANDOMIZE_B", "1")

    # Parallel envs help SPEED (CPU), especially if your C env step is heavy.
    n_envs = int(os.environ.get("N_ENVS", "8"))
    use_subproc = (n_envs > 1)

    if use_subproc:
        venv = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    else:
        venv = DummyVecEnv([make_env(0, seed)])

    # Normalize obs/reward (common in PPO); must be saved/loaded consistently
    venv = VecMonitor(venv)  # will create rollout/ep_rew_mean, rollout/ep_len_mean
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    # 原本的裸环境先包成 VecEnv
    _eval = DummyVecEnv([lambda: BoomerAMGRelaxEnv(
        lib_path="./libamg_env.dylib",
        seed=123,
        randomize_A=randomize_A,
        randomize_b=randomize_b,
        fixed_rhs_seed=int(os.environ.get("FIXED_RHS_SEED", "123456789")),
        fixed_rhs_type=int(os.environ.get("FIXED_RHS_TYPE", "1")),
    )])

    # VecNormalize：eval 不更新统计量，reward 也不需要 norm
    eval_env = VecNormalize(_eval, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # 共享训练统计量（关键）
    eval_env.obs_rms = venv.obs_rms


    # Paper: timesteps per batch = 16
    n_steps = 16
    batch_size = 16  # paper-like small batch

    # Episodes = 10,000; your episode length is <= max_cycles (e.g., 30)
    # Approx total timesteps:
    max_cycles = BoomerAMGRelaxEnv().max_cycles
    total_timesteps = 700 * max_cycles

    device = "cuda" if th.cuda.is_available() else "cpu"
    print("Device:", device, "| n_envs:", n_envs)

    model_type = os.environ.get("MODEL_TYPE", "lstm").strip().lower()
    common_kwargs = dict(
        env=venv,
        n_steps=16,
        batch_size=16,
        gamma=0.999,
        clip_range=0.2,
        learning_rate=3e-4,
        n_epochs=10,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.0,
        max_grad_norm=0.5,
        verbose=1,
        device=device,
        tensorboard_log="./ppo_logs",
    )

    if model_type == "custom":
        pi_dim = int(os.environ.get("CUSTOM_PI_DIM", "256"))
        vf_dim = int(os.environ.get("CUSTOM_VF_DIM", "128"))
        n_blocks = int(os.environ.get("CUSTOM_BLOCKS", "2"))
        model = PPO(
            policy=CustomPPOPolicy,
            policy_kwargs={"pi_dim": pi_dim, "vf_dim": vf_dim, "n_blocks": n_blocks},
            **common_kwargs,
        )
    elif model_type == "paper":
        model = PPO(
            policy=PaperPPOPolicy,
            **common_kwargs,
        )
    elif model_type == "mlp":
        model = PPO(
            policy="MlpPolicy",
            **common_kwargs,
        )
    else:
        model = RecurrentPPO(
            "MlpLstmPolicy",
            **common_kwargs,
        )

    # model = PPO(
    #     policy="MlpPolicy",      # default small MLP
    #     # change to better neuron-network, or check different options.
    #     env=venv,
    #     n_steps=16,
    #     batch_size=16,
    #     gamma=0.999,
    #     clip_range=0.2,
    #     learning_rate=3e-4,      # single LR for both actor+critic
    #     n_epochs=10,
    #     gae_lambda=0.95,
    #     vf_coef=0.5,
    #     ent_coef=0.0,
    #     max_grad_norm=0.5,
    #     verbose=1,
    #     device=device,
    #     tensorboard_log="./ppo_logs",
    # )
    step_logger = StepLoggerCallback(log_path="logs/train_steps.csv")
    eval_callback = EvalTrajCallback(eval_env=eval_env, eval_freq=50_000)

    callback = CallbackList([step_logger, eval_callback])

    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save("ppo_boomeramg_gen")
    venv.save("vecnormalize_gen.pkl")


if __name__ == "__main__":
    main()
