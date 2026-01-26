import os
import itertools
import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from amg_gym_env import BoomerAMGRelaxEnv


def _unwrap_single(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x[0]
    return x


def _is_vec_env(env):
    return hasattr(env, "num_envs")


def _maybe_batch_action(env, action):
    if _is_vec_env(env) and action.ndim == 1:
        return action.reshape((1, -1))
    return action


def run_episode(
    env,
    policy=None,
    fixed_cfg=None,
    deterministic=True,
    record_curve=False,
    obs_normalizer=None,
):
    obs, info = env.reset()
    info = _unwrap_single(info)
    done = False

    total_dt = 0.0
    residual_curve = [float(info.get("r0", np.nan))] if record_curve else None
    cfg_curve = [] if record_curve else None
    steps = 0

    fixed_action = None
    if fixed_cfg is not None:
        w, sd, su = fixed_cfg
        a_w = (w - env.w_center) / env.w_scale
        a_d = (sd - 3.0) / 2.0
        a_u = (su - 3.0) / 2.0
        fixed_action = np.array([a_w, a_d, a_u], dtype=np.float32)

    while not done:
        if fixed_action is not None:
            action = fixed_action
        else:
            obs_for_policy = obs
            if obs_normalizer is not None:
                obs_for_policy = obs_normalizer(obs_for_policy)
            action, _ = policy.predict(obs_for_policy, deterministic=deterministic)

        action = _maybe_batch_action(env, action)
        obs, reward, terminated, truncated, info = env.step(action)
        info = _unwrap_single(info)

        total_dt += float(info.get("dt", 0.0))
        if residual_curve is not None:
            residual_curve.append(float(info.get("r", np.nan)))
            if fixed_cfg is not None:
                w, sd, su = fixed_cfg
                cfg_curve.append((float(w), int(sd), int(su)))
            else:
                cfg_curve.append((float(info.get("w", np.nan)),
                                  int(info.get("sweeps_down", -1)),
                                  int(info.get("sweeps_up", -1))))

        done = bool(_unwrap_single(terminated)) or bool(_unwrap_single(truncated))
        steps += 1

    return {
        "time": total_dt,
        "cycles": steps,
        "residual_curve": (np.array(residual_curve, dtype=np.float64)
                           if residual_curve is not None else None),
        "cfg_curve": cfg_curve,
        "info": info,
    }


def _eval_fixed_cfg(args):
    cfg, eval_seed, lib_path = args
    env = BoomerAMGRelaxEnv(lib_path=lib_path, seed=eval_seed)
    out = run_episode(env, fixed_cfg=cfg, record_curve=False)
    env.close()
    return cfg, out["time"], out["cycles"]


def grid_search_best_constant(eval_seed, w_grid, sd_grid, su_grid, lib_path, n_workers):
    configs = list(itertools.product(w_grid, sd_grid, su_grid))
    args = [(cfg, eval_seed, lib_path) for cfg in configs]

    if n_workers <= 1:
        results = [_eval_fixed_cfg(a) for a in args]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_eval_fixed_cfg, args)

    best_cfg, best_time, best_cycles = min(results, key=lambda x: x[1])
    return {"cfg": best_cfg, "time": best_time, "cycles": best_cycles}


def main():
    lib_path = "./libamg_env.dylib"
    eval_seed = int(os.environ.get("EVAL_SEED", "100"))

    w_grid = np.linspace(0.7, 1.2, 11)
    sd_grid = [1, 2, 3, 4]
    su_grid = [1, 2, 3, 4]

    n_workers = int(os.environ.get("GRID_WORKERS", "4"))
    n_workers = max(1, min(n_workers, os.cpu_count() or 1))
    best = grid_search_best_constant(
        eval_seed, w_grid, sd_grid, su_grid, lib_path, n_workers
    )
    print("\n=== Grid-best constant ===")
    print(best)
    print("Grid-best cfg (w, sweeps_down, sweeps_up):", best["cfg"])

    vec_path = "vecnormalize_paper.pkl"
    vec_norm = None
    if os.path.exists(vec_path):
        try:
            raw_env = DummyVecEnv([lambda: BoomerAMGRelaxEnv(lib_path=lib_path, seed=0)])
            vec_norm = VecNormalize.load(vec_path, raw_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
            print(f"Loaded VecNormalize: {vec_path}")
        except AssertionError as exc:
            print(f"VecNormalize mismatch; skipping {vec_path}. Reason: {exc}")

    model = PPO.load("ppo_boomeramg_paper")

    env = BoomerAMGRelaxEnv(lib_path=lib_path, seed=eval_seed)
    out_best = run_episode(env, fixed_cfg=best["cfg"], record_curve=True)
    env.close()

    env2 = BoomerAMGRelaxEnv(lib_path=lib_path, seed=eval_seed)
    out_rl = run_episode(
        env2,
        policy=model,
        record_curve=True,
        obs_normalizer=(vec_norm.normalize_obs if vec_norm is not None else None),
    )
    env2.close()

    print("\n=== Single-episode eval ===")
    print("Grid-best:", out_best["time"], "s | cycles:", out_best["cycles"])
    print("RL       :", out_rl["time"], "s | cycles:", out_rl["cycles"])
    if out_best["cfg_curve"] and out_rl["cfg_curve"]:
        best_last = out_best["cfg_curve"][-1]
        rl_last = out_rl["cfg_curve"][-1]
        print("\n=== Final params table ===")
        print("method    w        sweeps_down  sweeps_up")
        print(f"grid-best {best_last[0]:.6f} {best_last[1]:>12d} {best_last[2]:>10d}")
        print(f"rl        {rl_last[0]:.6f} {rl_last[1]:>12d} {rl_last[2]:>10d}")

    if out_best["cfg_curve"] and out_rl["cfg_curve"]:
        best_cfg = np.array(out_best["cfg_curve"], dtype=float)
        rl_cfg = np.array(out_rl["cfg_curve"], dtype=float)
        best_steps = np.arange(1, best_cfg.shape[0] + 1)
        rl_steps = np.arange(1, rl_cfg.shape[0] + 1)
        best_w, best_sd, best_su = best["cfg"]

        plt.figure()
        plt.plot(best_steps, best_cfg[:, 0], label="grid-best")
        plt.plot(rl_steps, rl_cfg[:, 0], label="rl")
        plt.title(f"Relaxation weight w (grid-best w={best_w:.3f})")
        plt.xlabel("V-cycle")
        plt.ylabel("w")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(best_steps, best_cfg[:, 1], label="grid-best sweeps_down")
        plt.plot(best_steps, best_cfg[:, 2], label="grid-best sweeps_up")
        plt.plot(rl_steps, rl_cfg[:, 1], label="rl sweeps_down")
        plt.plot(rl_steps, rl_cfg[:, 2], label="rl sweeps_up")
        plt.title(
            "Sweeps per cycle "
            f"(grid-best down={int(best_sd)}, up={int(best_su)})"
        )
        plt.xlabel("V-cycle")
        plt.ylabel("sweeps")
        plt.legend()
        plt.tight_layout()
        plt.show()

    plt.figure()
    plt.semilogy(out_best["residual_curve"], alpha=0.7, label="grid-best")
    plt.title("Grid-best: residual curve")
    plt.xlabel("V-cycle")
    plt.ylabel("||r||")
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.semilogy(out_rl["residual_curve"], alpha=0.7, label="rl")
    plt.title("RL: residual curve")
    plt.xlabel("V-cycle")
    plt.ylabel("||r||")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
