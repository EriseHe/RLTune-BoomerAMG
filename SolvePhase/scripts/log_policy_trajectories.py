import numpy as np
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from . import _project_paths  # noqa: F401
except ImportError:
    import _project_paths  # type: ignore[no-redef]  # noqa: F401
from amg_gym_env import BoomerAMGRelaxEnv, DEFAULT_AMG_ENV_LIBRARY


def make_env():
    """
    Env factory matching your training setup.
    Adjust fixed_grid, tol, max_cycles, etc. if you changed them.
    """
    def _thunk():
        return BoomerAMGRelaxEnv(
            lib_path=str(DEFAULT_AMG_ENV_LIBRARY),
            fixed_grid=(60, 60, 60),
            fixed_stencil=27,
            fixed_rhs_type=1,
            tol=1e-8,
            max_cycles=30,
            seed=0,
        )
    return _thunk


def log_policy_episodes(
    model_path="ppo_boomeramg_relax_paper",
    vecnorm_path="vecnormalize_paper.pkl",
    n_episodes=20,
    out_csv="policy_trajectories.csv",
):
    # --- 1) Rebuild vec env + load normalization stats ---
    venv = DummyVecEnv([make_env()])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False

    # --- 2) Load trained PPO with this env ---
    model = PPO.load(model_path, env=venv)

    all_rows = []

    for ep in range(n_episodes):
        obs = venv.reset()
        done = False
        ep_step = 0
        ep_return = 0.0
        ep_time = 0.0
        final_r = None

        while not done:
            # a_t from policy
            action, _ = model.predict(obs, deterministic=True)

            # One vectorized step
            obs, reward, dones, infos = venv.step(action)

            r_step = float(reward[0])
            info = infos[0]

            ep_return += r_step

            # Pull out whatever env logged in info.
            # These keys assume your env.step(...) did something like:
            # info = {"r": r, "dt": dt, "w": w,
            #         "sweeps_down": sweeps_down, "sweeps_up": sweeps_up,
            #         "cycle": self.cycle, ...}
            row = {
                "episode": ep,
                "step": ep_step,
                "cycle": info.get("cycle", np.nan),
                "residual_norm": info.get("r", np.nan),
                "dt": info.get("dt", np.nan),
                "w": info.get("w", np.nan),
                "sweeps_down": info.get("sweeps_down", np.nan),
                "sweeps_up": info.get("sweeps_up", np.nan),
                "action_raw": action[0].tolist() if hasattr(action, "shape") else action,
                "reward": r_step,
            }

            final_r = row["residual_norm"]
            ep_time += row["dt"] if not np.isnan(row["dt"]) else 0.0
            all_rows.append(row)

            ep_step += 1
            done = bool(dones[0])

        print(
            f"[RL] episode {ep:3d}: "
            f"steps={ep_step:2d}, time={ep_time:.4f} s, "
            f"return={ep_return:.3f}, final_r={final_r:.3e}"
        )

    df = pd.DataFrame(all_rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved detailed trajectories for {n_episodes} episodes -> {out_csv}")


if __name__ == "__main__":
    log_policy_episodes(
        model_path="ppo_boomeramg_paper",
        n_episodes=20,
        out_csv="policy_trajectories.csv",
    )
