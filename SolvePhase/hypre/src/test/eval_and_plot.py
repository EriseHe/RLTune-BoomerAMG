import os
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


def _env_flag(name, default="1"):
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no")


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


def _make_env(lib_path, seed, grid, randomize_A, randomize_b, fixed_rhs_seed, fixed_rhs_type):
    return BoomerAMGRelaxEnv(
        lib_path=lib_path,
        seed=seed,
        fixed_grid=grid,
        randomize_A=randomize_A,
        randomize_b=randomize_b,
        fixed_rhs_seed=fixed_rhs_seed,
        fixed_rhs_type=fixed_rhs_type,
    )


def _sample_baseline_cfgs(rng, w_min, w_max, samples):
    cfgs = []
    for _ in range(samples):
        w = float(rng.uniform(w_min, w_max))
        sd = int(rng.integers(1, 6))
        su = int(rng.integers(1, 6))
        cfgs.append((w, sd, su))
    return cfgs


def find_rough_best_constant(lib_path, eval_cases, w_min, w_max, samples):
    rng = np.random.default_rng(0)
    cfgs = _sample_baseline_cfgs(rng, w_min, w_max, samples)

    best = None
    for cfg in cfgs:
        times = []
        cycles = []
        for case in eval_cases:
            env = _make_env(
                lib_path,
                case["seed"],
                case["grid"],
                case["randomize_A"],
                case["randomize_b"],
                case["fixed_rhs_seed"],
                case["fixed_rhs_type"],
            )
            out = run_episode(env, fixed_cfg=cfg, record_curve=False)
            env.close()
            times.append(out["time"])
            cycles.append(out["cycles"])
        mean_time = float(np.mean(times))
        mean_cycles = float(np.mean(cycles))
        if best is None or mean_time < best["time"]:
            best = {"cfg": cfg, "time": mean_time, "cycles": mean_cycles}
    return best


def _summarize(results):
    times = np.array([r["time"] for r in results], dtype=float)
    cycles = np.array([r["cycles"] for r in results], dtype=float)
    return {
        "mean_time": float(times.mean()),
        "median_time": float(np.median(times)),
        "mean_cycles": float(cycles.mean()),
        "median_cycles": float(np.median(cycles)),
    }


def main():
    lib_path = "./libamg_env.dylib"
    seed_start = int(os.environ.get("EVAL_SEED_START", "100"))
    seed_count = int(os.environ.get("EVAL_SEED_COUNT", "6"))
    eval_seeds = list(range(seed_start, seed_start + seed_count))
    # Matrix size fixed at 60^3 for all eval runs
    grid_sizes = [(60, 60, 60)]

    fixed_a_mode = _env_flag("EVAL_FIXED_A", "0")
    randomize_A = _env_flag("RANDOMIZE_A", "1")
    randomize_b = _env_flag("RANDOMIZE_B", "1")
    fixed_rhs_seed = int(os.environ.get("FIXED_RHS_SEED", "123456789"))
    fixed_rhs_type = int(os.environ.get("FIXED_RHS_TYPE", "1"))
    if fixed_a_mode:
        randomize_A = False
        randomize_b = False
        fixed_rhs_type = 1
        grid_sizes = [(60, 60, 60)]
        eval_seeds = [seed_start]

    grid_mode = os.environ.get("EVAL_GRID_MODE", "zip").lower()
    if grid_mode == "full":
        eval_cases = [
            {
                "seed": s,
                "grid": g,
                "randomize_A": randomize_A,
                "randomize_b": randomize_b,
                "fixed_rhs_seed": fixed_rhs_seed,
                "fixed_rhs_type": fixed_rhs_type,
            }
            for s in eval_seeds
            for g in grid_sizes
        ]
    else:
        eval_cases = [
            {
                "seed": s,
                "grid": grid_sizes[i % len(grid_sizes)],
                "randomize_A": randomize_A,
                "randomize_b": randomize_b,
                "fixed_rhs_seed": fixed_rhs_seed,
                "fixed_rhs_type": fixed_rhs_type,
            }
            for i, s in enumerate(eval_seeds)
        ]

    baseline_cases = int(os.environ.get("BASELINE_EVAL_CASES", "3"))
    baseline_cases = max(1, min(baseline_cases, len(eval_cases)))
    baseline_samples = int(os.environ.get("BASELINE_SAMPLES", "25"))

    tmp_env = _make_env(
        lib_path,
        seed_start,
        grid_sizes[0],
        randomize_A,
        randomize_b,
        fixed_rhs_seed,
        fixed_rhs_type,
    )
    w_min = tmp_env.w_center - tmp_env.w_scale
    w_max = tmp_env.w_center + tmp_env.w_scale
    tmp_env.close()

    best = find_rough_best_constant(
        lib_path,
        eval_cases[:baseline_cases],
        w_min,
        w_max,
        baseline_samples,
    )
    print("\n=== Grid-best constant ===")
    print(best)
    print("Grid-best cfg (w, sweeps_down, sweeps_up):", best["cfg"])
    print("Eval grid sizes:", grid_sizes)
    print("Eval cases:", len(eval_cases), "| baseline cases:", baseline_cases)
    if fixed_a_mode:
        print("Fixed-A mode: ON (A coefficients and size fixed)")

    vec_path = "vecnormalize_gen.pkl"
    # vec_path = "vecnormalize_fixed.pkl"
    vec_norm = None
    if os.path.exists(vec_path):
        try:
            raw_env = DummyVecEnv(
                [
                    lambda: _make_env(
                        lib_path,
                        seed_start,
                        grid_sizes[0],
                        randomize_A,
                        randomize_b,
                        fixed_rhs_seed,
                        fixed_rhs_type,
                    )
                ]
            )
            vec_norm = VecNormalize.load(vec_path, raw_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
            print(f"Loaded VecNormalize: {vec_path}")
        except AssertionError as exc:
            print(f"VecNormalize mismatch; skipping {vec_path}. Reason: {exc}")

    model = PPO.load("ppo_boomeramg_gen")
    # model = PPO.load("ppo_boomeramg_fixed")

    plot_count = int(os.environ.get("PLOT_COUNT", "3"))
    plot_count = max(1, min(plot_count, len(eval_cases)))

    best_results = []
    rl_results = []
    for i, case in enumerate(eval_cases):
        env = _make_env(
            lib_path,
            case["seed"],
            case["grid"],
            case["randomize_A"],
            case["randomize_b"],
            case["fixed_rhs_seed"],
            case["fixed_rhs_type"],
        )
        out_best = run_episode(env, fixed_cfg=best["cfg"], record_curve=(i < plot_count))
        env.close()
        out_best["case"] = case
        best_results.append(out_best)

        env2 = _make_env(
            lib_path,
            case["seed"],
            case["grid"],
            case["randomize_A"],
            case["randomize_b"],
            case["fixed_rhs_seed"],
            case["fixed_rhs_type"],
        )
        out_rl = run_episode(
            env2,
            policy=model,
            record_curve=(i < plot_count),
            obs_normalizer=(vec_norm.normalize_obs if vec_norm is not None else None),
        )
        env2.close()
        out_rl["case"] = case
        rl_results.append(out_rl)

    print("\n=== Summary ===")
    print("Grid-best:", _summarize(best_results))
    print("RL       :", _summarize(rl_results))

    if best_results and best_results[0]["cfg_curve"] and rl_results[0]["cfg_curve"]:
        best_last = best_results[0]["cfg_curve"][-1]
        rl_last = rl_results[0]["cfg_curve"][-1]
        print("\n=== Final params table (first case) ===")
        print("method    w        sweeps_down  sweeps_up")
        print(f"grid-best {best_last[0]:.6f} {best_last[1]:>12d} {best_last[2]:>10d}")
        print(f"rl        {rl_last[0]:.6f} {rl_last[1]:>12d} {rl_last[2]:>10d}")

        best_cfg = np.array(best_results[0]["cfg_curve"], dtype=float)
        rl_cfg = np.array(rl_results[0]["cfg_curve"], dtype=float)
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
        plt.semilogy(best_results[0]["residual_curve"], alpha=0.7, label="grid-best")
        plt.title("Grid-best: residual curve (first case)")
        plt.xlabel("V-cycle")
        plt.ylabel("||r||")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.semilogy(rl_results[0]["residual_curve"], alpha=0.7, label="rl")
        plt.title("RL: residual curve (first case)")
        plt.xlabel("V-cycle")
        plt.ylabel("||r||")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
