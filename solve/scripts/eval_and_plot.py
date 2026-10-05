from dataclasses import dataclass
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from . import _project_paths  # noqa: F401
except ImportError:
    import _project_paths  # type: ignore[no-redef]  # noqa: F401
from solve.core.amg_gym_env import BoomerAMGRelaxEnv, DEFAULT_AMG_RUNTIME_LIBRARY


def _set_relax_type(value: int) -> None:
    os.environ["AMG_RELAX_TYPE"] = str(int(value))


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


def _cfg_to_action(env, w, sd, su):
    if env.w_scale == 0:
        a_w = 0.0
    else:
        a_w = (w - env.w_center) / env.w_scale
    sweeps_half = 0.5 * (env.sweeps_max - env.sweeps_min)
    sweeps_center = 0.5 * (env.sweeps_max + env.sweeps_min)
    if sweeps_half <= 0:
        a_d = 0.0
        a_u = 0.0
    else:
        a_d = (sd - sweeps_center) / sweeps_half
        a_u = (su - sweeps_center) / sweeps_half
    a_w = float(np.clip(a_w, -1.0, 1.0))
    a_d = float(np.clip(a_d, -1.0, 1.0))
    a_u = float(np.clip(a_u, -1.0, 1.0))
    return np.array([a_w, a_d, a_u], dtype=np.float32)


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


def run_episode(
    env,
    policy=None,
    fixed_cfg=None,
    deterministic=True,
    record_curve=False,
    obs_normalizer=None,
    use_lstm=False,
):
    obs, info = env.reset()
    info = _unwrap_single(info)
    done = False

    total_dt = 0.0
    residual_curve = [float(info.get("r0", np.nan))] if record_curve else None
    cfg_curve = [] if record_curve else None
    steps = 0
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)

    fixed_action = None
    if fixed_cfg is not None:
        w, sd, su = fixed_cfg
        fixed_action = _cfg_to_action(env, w, sd, su)

    while not done:
        if fixed_action is not None:
            action = fixed_action
        else:
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
        episode_start[...] = done
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

def _make_env(
    lib_path,
    seed,
    grid,
    randomize_A,
    randomize_b,
    fixed_rhs_seed,
    fixed_rhs_type,
    *,
    relax_type,
    fixed_stencil,
    randomize_grid,
    grid_min,
    grid_max,
    difconv_c,
    difconv_c_range,
    difconv_a,
    difconv_atype,
    w_center,
    w_scale,
    sweeps_min,
    sweeps_max,
):
    _set_relax_type(relax_type)
    return BoomerAMGRelaxEnv(
        lib_path=lib_path,
        seed=seed,
        fixed_grid=(None if randomize_grid else grid),
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


def _sample_baseline_cfgs(rng, w_min, w_max, sweeps_min, sweeps_max, samples):
    cfgs = []
    for _ in range(samples):
        w = float(rng.uniform(w_min, w_max))
        sd = int(rng.integers(sweeps_min, sweeps_max + 1))
        su = int(rng.integers(sweeps_min, sweeps_max + 1))
        cfgs.append((w, sd, su))
    return cfgs


def find_rough_best_constant(
    lib_path,
    eval_cases,
    w_min,
    w_max,
    sweeps_min,
    sweeps_max,
    samples,
    *,
    relax_type,
    fixed_stencil,
    randomize_grid,
    grid_min,
    grid_max,
    difconv_c,
    difconv_c_range,
    difconv_a,
    difconv_atype,
    w_center,
    w_scale,
):
    rng = np.random.default_rng(0)
    cfgs = _sample_baseline_cfgs(rng, w_min, w_max, sweeps_min, sweeps_max, samples)
    if (1.0, 1, 1) not in cfgs:
        cfgs.append((1.0, 1, 1))

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
                relax_type=relax_type,
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
    time_per_cycle = times / np.maximum(cycles, 1.0)
    return {
        "mean_time": float(times.mean()),
        "median_time": float(np.median(times)),
        "mean_cycles": float(cycles.mean()),
        "median_cycles": float(np.median(cycles)),
        "mean_time_per_cycle": float(time_per_cycle.mean()),
        "median_time_per_cycle": float(np.median(time_per_cycle)),
    }


@dataclass(frozen=True)
class _EvaluationSettings:
    baseline_cases: int
    baseline_samples: int
    difconv_a: tuple[float, ...]
    difconv_atype: int
    difconv_c: tuple[float, ...]
    difconv_c_range: tuple[float, ...]
    eval_cases: list[dict]
    fixed_a_mode: bool
    fixed_rhs_seed: int
    fixed_rhs_type: int
    fixed_stencil: int
    grid_max: int
    grid_min: int
    grid_sizes: list[tuple[int, int, int]]
    lib_path: str
    randomize_A: bool
    randomize_b: bool
    randomize_grid: bool
    seed_start: int
    sweeps_max: int
    sweeps_min: int
    w_center: float
    w_scale: float


def _prepare_evaluation():
    """Read settings and construct the prescribed baseline/policy cases."""
    lib_path = str(DEFAULT_AMG_RUNTIME_LIBRARY)
    seed_start = int(os.environ.get("EVAL_SEED_START", "100"))
    seed_count = int(os.environ.get("EVAL_SEED_COUNT", "6"))
    eval_seeds = list(range(seed_start, seed_start + seed_count))
    # Default grid size; can be overridden by RANDOMIZE_GRID
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
    sweeps_max = int(os.environ.get("SWEEPS_MAX", "5"))
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

    grid_mode = os.environ.get("EVAL_GRID_MODE", "zip").lower()
    if randomize_grid:
        eval_cases = [
            {
                "seed": s,
                "grid": (grid_min, grid_min, grid_min),
                "randomize_A": randomize_A,
                "randomize_b": randomize_b,
                "fixed_rhs_seed": fixed_rhs_seed,
                "fixed_rhs_type": fixed_rhs_type,
            }
            for s in eval_seeds
        ]
    elif grid_mode == "full":
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

    baseline_cases = int(os.environ.get("BASELINE_EVAL_CASES", str(len(eval_cases))))
    baseline_cases = max(1, min(baseline_cases, len(eval_cases)))
    baseline_samples = int(os.environ.get("BASELINE_SAMPLES", "25"))

    return _EvaluationSettings(
        baseline_cases=baseline_cases,
        baseline_samples=baseline_samples,
        difconv_a=difconv_a,
        difconv_atype=difconv_atype,
        difconv_c=difconv_c,
        difconv_c_range=difconv_c_range,
        eval_cases=eval_cases,
        fixed_a_mode=fixed_a_mode,
        fixed_rhs_seed=fixed_rhs_seed,
        fixed_rhs_type=fixed_rhs_type,
        fixed_stencil=fixed_stencil,
        grid_max=grid_max,
        grid_min=grid_min,
        grid_sizes=grid_sizes,
        lib_path=lib_path,
        randomize_A=randomize_A,
        randomize_b=randomize_b,
        randomize_grid=randomize_grid,
        seed_start=seed_start,
        sweeps_max=sweeps_max,
        sweeps_min=sweeps_min,
        w_center=w_center,
        w_scale=w_scale,
    )


def _select_constant_baseline(settings: _EvaluationSettings):
    """Search the original baseline sample and report the selected constant."""
    tmp_env = _make_env(
        settings.lib_path,
        settings.seed_start,
        settings.grid_sizes[0],
        settings.randomize_A,
        settings.randomize_b,
        settings.fixed_rhs_seed,
        settings.fixed_rhs_type,
        relax_type=13,
        fixed_stencil=settings.fixed_stencil,
        randomize_grid=settings.randomize_grid,
        grid_min=settings.grid_min,
        grid_max=settings.grid_max,
        difconv_c=settings.difconv_c,
        difconv_c_range=settings.difconv_c_range,
        difconv_a=settings.difconv_a,
        difconv_atype=settings.difconv_atype,
        w_center=settings.w_center,
        w_scale=settings.w_scale,
        sweeps_min=settings.sweeps_min,
        sweeps_max=settings.sweeps_max,
    )
    w_min = tmp_env.w_center - tmp_env.w_scale
    w_max = tmp_env.w_center + tmp_env.w_scale
    tmp_env.close()

    best = find_rough_best_constant(
        settings.lib_path,
        settings.eval_cases[:settings.baseline_cases],
        w_min,
        w_max,
        settings.sweeps_min,
        settings.sweeps_max,
        settings.baseline_samples,
        relax_type=13,
        fixed_stencil=settings.fixed_stencil,
        randomize_grid=settings.randomize_grid,
        grid_min=settings.grid_min,
        grid_max=settings.grid_max,
        difconv_c=settings.difconv_c,
        difconv_c_range=settings.difconv_c_range,
        difconv_a=settings.difconv_a,
        difconv_atype=settings.difconv_atype,
        w_center=settings.w_center,
        w_scale=settings.w_scale,
    )
    print("\n=== Grid-best constant ===")
    print(best)
    print("Grid-best cfg (w, sweeps_down, sweeps_up):", best["cfg"])
    if settings.randomize_grid:
        print(f"Eval grid range: [{settings.grid_min}, {settings.grid_max}] (uniform, cubic)")
    else:
        print("Eval grid sizes:", settings.grid_sizes)
    print("Eval cases:", len(settings.eval_cases), "| baseline cases:", settings.baseline_cases)
    if settings.fixed_a_mode:
        print("Fixed-A mode: ON (A coefficients and size fixed)")

    return best


def _load_evaluation_policy(settings: _EvaluationSettings):
    """Load optional normalization and the frozen PPO or recurrent policy."""
    vec_path = "vecnormalize_gen.pkl"
    # vec_path = "vecnormalize_fixed.pkl"
    vec_norm = None
    if os.path.exists(vec_path):
        try:
            raw_env = DummyVecEnv(
                [
                    lambda: _make_env(
                        settings.lib_path,
                        settings.seed_start,
                        settings.grid_sizes[0],
                        settings.randomize_A,
                        settings.randomize_b,
                        settings.fixed_rhs_seed,
                        settings.fixed_rhs_type,
                        relax_type=18,
                        fixed_stencil=settings.fixed_stencil,
                        randomize_grid=settings.randomize_grid,
                        grid_min=settings.grid_min,
                        grid_max=settings.grid_max,
                        difconv_c=settings.difconv_c,
                        difconv_c_range=settings.difconv_c_range,
                        difconv_a=settings.difconv_a,
                        difconv_atype=settings.difconv_atype,
                        w_center=settings.w_center,
                        w_scale=settings.w_scale,
                        sweeps_min=settings.sweeps_min,
                        sweeps_max=settings.sweeps_max,
                    )
                ]
            )
            vec_norm = VecNormalize.load(vec_path, raw_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
            print(f"Loaded VecNormalize: {vec_path}")
        except AssertionError as exc:
            print(f"VecNormalize mismatch; skipping {vec_path}. Reason: {exc}")

    model_type = os.environ.get("MODEL_TYPE", "ppo").strip().lower()
    if model_type == "lstm":
        model = RecurrentPPO.load("ppo_boomeramg_gen")
    else:
        model = PPO.load("ppo_boomeramg_gen")
    # model = PPO.load("ppo_boomeramg_fixed")

    plot_count = int(os.environ.get("PLOT_COUNT", "3"))
    plot_count = max(1, min(plot_count, len(settings.eval_cases)))

    return model, vec_norm, model_type, plot_count


def _evaluate_cases(settings: _EvaluationSettings, best, model, vec_norm, model_type, plot_count):
    """Evaluate the selected constant and frozen policy on the same cases."""
    best_results = []
    rl_results = []
    for i, case in enumerate(settings.eval_cases):
        env = _make_env(
            settings.lib_path,
            case["seed"],
            case["grid"],
            case["randomize_A"],
            case["randomize_b"],
            case["fixed_rhs_seed"],
            case["fixed_rhs_type"],
            relax_type=13,
            fixed_stencil=settings.fixed_stencil,
            randomize_grid=settings.randomize_grid,
            grid_min=settings.grid_min,
            grid_max=settings.grid_max,
            difconv_c=settings.difconv_c,
            difconv_c_range=settings.difconv_c_range,
            difconv_a=settings.difconv_a,
            difconv_atype=settings.difconv_atype,
            w_center=settings.w_center,
            w_scale=settings.w_scale,
            sweeps_min=settings.sweeps_min,
            sweeps_max=settings.sweeps_max,
        )
        out_best = run_episode(env, fixed_cfg=best["cfg"], record_curve=(i < plot_count))
        env.close()
        out_best["case"] = case
        best_results.append(out_best)

        env2 = _make_env(
            settings.lib_path,
            case["seed"],
            case["grid"],
            case["randomize_A"],
            case["randomize_b"],
            case["fixed_rhs_seed"],
            case["fixed_rhs_type"],
            relax_type=18,
            fixed_stencil=settings.fixed_stencil,
            randomize_grid=settings.randomize_grid,
            grid_min=settings.grid_min,
            grid_max=settings.grid_max,
            difconv_c=settings.difconv_c,
            difconv_c_range=settings.difconv_c_range,
            difconv_a=settings.difconv_a,
            difconv_atype=settings.difconv_atype,
            w_center=settings.w_center,
            w_scale=settings.w_scale,
            sweeps_min=settings.sweeps_min,
            sweeps_max=settings.sweeps_max,
        )
        out_rl = run_episode(
            env2,
            policy=model,
            record_curve=(i < plot_count),
            obs_normalizer=(vec_norm.normalize_obs if vec_norm is not None else None),
            use_lstm=(model_type == "lstm"),
        )
        env2.close()
        out_rl["case"] = case
        rl_results.append(out_rl)

    return best_results, rl_results


def _report_and_plot(best, best_results, rl_results):
    """Report costs and plot the recorded first-case controls and residuals."""
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


def main():
    settings = _prepare_evaluation()
    best = _select_constant_baseline(settings)
    model, vec_norm, model_type, plot_count = _load_evaluation_policy(settings)
    best_results, rl_results = _evaluate_cases(
        settings, best, model, vec_norm, model_type, plot_count,
    )
    _report_and_plot(best, best_results, rl_results)


if __name__ == "__main__":
    main()
