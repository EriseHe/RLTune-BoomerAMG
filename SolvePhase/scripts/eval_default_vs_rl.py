import os
import time
from pathlib import Path
import json
import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

try:
    from . import _project_paths  # noqa: F401
except ImportError:
    import _project_paths  # type: ignore[no-redef]  # noqa: F401
from amg_gym_env import BoomerAMGRelaxEnv, DEFAULT_AMG_ENV_LIBRARY


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
    if getattr(env, "w_only", False) or env.action_space.shape[0] == 1:
        a_w = float(np.clip(a_w, -1.0, 1.0))
        return np.array([a_w], dtype=np.float32)
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


def _expand_seeds(eval_seeds, instances, stride):
    if instances <= 1:
        return list(eval_seeds)
    expanded = []
    for s in eval_seeds:
        for i in range(instances):
            expanded.append(int(s) + int(i) * int(stride))
    return expanded


def _compose_failure_reason(reasons):
    unique = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(str(reason))
    return ";".join(unique) if unique else ""


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def _classify_episode_failure(*, final_info, terminated, truncated, final_residual, cycles, tol, max_cycles):
    reasons = []
    if bool(final_info.get("bad_step", False)):
        reasons.append("bad_step")
    if not np.isfinite(final_residual):
        reasons.append("non_finite_residual_norm")
    elif final_residual > float(tol):
        reasons.append("residual_above_tol")
    if bool(truncated):
        if np.isfinite(final_residual) and final_residual <= float(tol):
            reasons.append("hit_or_exceeded_max_cycles")
        else:
            reasons.append("max_cycles_reached_without_convergence")
    if (not bool(terminated)) and (not bool(truncated)) and not reasons:
        reasons.append("episode_ended_without_terminal_flag")
    failed = bool(reasons)
    return failed, _compose_failure_reason(reasons)


def _failure_log_record(*, scope, mode, case, result):
    info = dict(result.get("info", {}) or {})
    return {
        "scope": str(scope),
        "mode": str(mode),
        "case": _json_safe(case),
        "failure_reason": str(result.get("failure_reason", "")),
        "cycles": int(result.get("cycles", -1)),
        "final_residual_norm": float(result.get("final_residual_norm", np.nan)),
        "setup_time": float(result.get("setup_time", np.nan)),
        "solve_time": float(result.get("time", np.nan)),
        "time_with_setup": float(result.get("time_with_setup", np.nan)),
        "wall_time": float(result.get("wall_time", np.nan)) if result.get("wall_time") is not None else None,
        "overhead": float(result.get("overhead", np.nan)) if result.get("overhead") is not None else None,
        "terminated": bool(result.get("terminated", False)),
        "truncated": bool(result.get("truncated", False)),
        "bad_step": bool(info.get("bad_step", False)),
        "final_w": float(info.get("w", np.nan)),
        "final_sweeps_down": int(info.get("sweeps_down", -1)),
        "final_sweeps_up": int(info.get("sweeps_up", -1)),
    }


def _write_failure_log(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(_json_safe(rec), sort_keys=True) + "\n")


def run_episode(
    env,
    policy=None,
    fixed_cfg=None,
    deterministic=True,
    record_curve=False,
    obs_normalizer=None,
    use_lstm=False,
    measure_wall=False,
    measure_infer=False,
):
    wall_start = time.perf_counter() if measure_wall else None
    obs, info = env.reset()
    info = _unwrap_single(info)
    done = False

    total_solve_dt = 0.0
    total_step_wall = 0.0
    setup_time = float(info.get("setup_time", 0.0))
    residual_curve = [float(info.get("r0", np.nan))] if record_curve else None
    cfg_curve = [] if record_curve else None
    steps = 0
    lstm_state = None
    episode_start = np.ones((1,), dtype=bool)
    infer_time = 0.0
    preprocess_time = 0.0
    last_terminated = False
    last_truncated = False

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
                t0 = time.perf_counter() if measure_infer else None
                obs_for_policy = obs_normalizer(obs_for_policy)
                if measure_infer and t0 is not None:
                    preprocess_time += time.perf_counter() - t0
            if use_lstm:
                t0 = time.perf_counter() if measure_infer else None
                action, lstm_state = policy.predict(
                    obs_for_policy,
                    state=lstm_state,
                    episode_start=episode_start,
                    deterministic=deterministic,
                )
                if measure_infer and t0 is not None:
                    infer_time += time.perf_counter() - t0
            else:
                t0 = time.perf_counter() if measure_infer else None
                action, _ = policy.predict(obs_for_policy, deterministic=deterministic)
                if measure_infer and t0 is not None:
                    infer_time += time.perf_counter() - t0

        action = _maybe_batch_action(env, action)
        obs, reward, terminated, truncated, info = env.step(action)
        info = _unwrap_single(info)

        total_solve_dt += float(info.get("dt_solver", info.get("dt", 0.0)))
        total_step_wall += float(info.get("dt_wall", info.get("dt", 0.0)))
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
        last_terminated = bool(_unwrap_single(terminated))
        last_truncated = bool(_unwrap_single(truncated))
        episode_start[...] = done
        steps += 1
    wall_time = None
    overhead = None
    if measure_wall and wall_start is not None:
        wall_time = time.perf_counter() - wall_start
        overhead = wall_time - (setup_time + total_solve_dt)
    final_residual = float(info.get("r", np.nan))
    failed, failure_reason = _classify_episode_failure(
        final_info=info,
        terminated=last_terminated,
        truncated=last_truncated,
        final_residual=final_residual,
        cycles=steps,
        tol=float(env.tol),
        max_cycles=int(env.max_cycles),
    )

    return {
        "time": total_solve_dt,
        "step_wall_time": total_step_wall,
        "setup_time": setup_time,
        "time_with_setup": float(total_solve_dt + setup_time),
        "cycles": steps,
        "wall_time": wall_time,
        "overhead": overhead,
        "infer_time": infer_time if measure_infer else None,
        "preprocess_time": preprocess_time if measure_infer else None,
        "residual_curve": (np.array(residual_curve, dtype=np.float64)
                           if residual_curve is not None else None),
        "cfg_curve": cfg_curve,
        "info": info,
        "terminated": bool(last_terminated),
        "truncated": bool(last_truncated),
        "failed": bool(failed),
        "failure_reason": str(failure_reason),
        "final_residual_norm": float(final_residual),
    }


def _summarize(results):
    times = np.array([r["time"] for r in results], dtype=float)
    setup_times = np.array([r.get("setup_time", 0.0) for r in results], dtype=float)
    total_with_setup = np.array([r.get("time_with_setup", r["time"] + r.get("setup_time", 0.0)) for r in results], dtype=float)
    cycles = np.array([r["cycles"] for r in results], dtype=float)
    time_per_cycle = times / np.maximum(cycles, 1.0)
    return {
        "mean_solve_time": float(times.mean()),
        "median_solve_time": float(np.median(times)),
        "mean_setup_time": float(setup_times.mean()),
        "median_setup_time": float(np.median(setup_times)),
        "mean_time_with_setup": float(total_with_setup.mean()),
        "median_time_with_setup": float(np.median(total_with_setup)),
        "mean_cycles": float(cycles.mean()),
        "median_cycles": float(np.median(cycles)),
        "mean_time_per_cycle": float(time_per_cycle.mean()),
        "median_time_per_cycle": float(np.median(time_per_cycle)),
    }


def _summarize_failures(results):
    total = int(sum(1 for r in results if bool(r.get("failed", False))))
    counts = {}
    for r in results:
        if not bool(r.get("failed", False)):
            continue
        reason = str(r.get("failure_reason", "")).strip()
        counts[reason] = counts.get(reason, 0) + 1
    return {"failed_count": total, "failure_reason_counts": counts}


def _summarize_overhead(results):
    vals = []
    cycles = []
    for r in results:
        ov = r.get("overhead")
        if ov is None:
            wt = r.get("wall_time")
            t_with_setup = r.get("time_with_setup")
            if wt is not None and t_with_setup is not None:
                ov = float(wt) - float(t_with_setup)
        if ov is None:
            continue
        vals.append(float(ov))
        cycles.append(float(r.get("cycles", 0)))
    overheads = np.array(vals, dtype=float)
    if overheads.size == 0:
        return None
    cycles = np.array(cycles, dtype=float)
    return {
        "mean_overhead": float(overheads.mean()),
        "median_overhead": float(np.median(overheads)),
        "mean_overhead_per_cycle": float((overheads / np.maximum(cycles, 1.0)).mean()),
        "median_overhead_per_cycle": float(np.median(
            overheads / np.maximum(cycles, 1.0))),
    }


def _summarize_infer(results):
    infer = np.array(
        [r["infer_time"] for r in results if r.get("infer_time") is not None],
        dtype=float,
    )
    if infer.size == 0:
        return None
    cycles = np.array(
        [r["cycles"] for r in results if r.get("infer_time") is not None],
        dtype=float,
    )
    return {
        "mean_infer": float(infer.mean()),
        "median_infer": float(np.median(infer)),
        "mean_infer_per_cycle": float((infer / np.maximum(cycles, 1.0)).mean()),
        "median_infer_per_cycle": float(np.median(infer / np.maximum(cycles, 1.0))),
    }


def _plot_rl_curves(grid_sizes, rl_curves, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, curves in zip(grid_sizes, rl_curves):
        residuals = curves.get("residual_curve")
        cfg = curves.get("cfg_curve")
        if residuals is None or cfg is None:
            continue
        steps = np.arange(len(residuals))
        cfg_arr = np.array(cfg, dtype=float)
        fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
        axes[0].plot(steps, residuals, label="residual")
        axes[0].set_yscale("log")
        axes[0].set_ylabel("Residual")
        axes[0].set_title(f"RL residuals (n={n})")
        axes[0].grid(True, alpha=0.3)

        if cfg_arr.ndim == 2 and cfg_arr.shape[1] >= 3:
            axes[1].plot(steps[1:], cfg_arr[:, 0], label="w")
            axes[1].plot(steps[1:], cfg_arr[:, 1], label="sweeps_down")
            axes[1].plot(steps[1:], cfg_arr[:, 2], label="sweeps_up")
        axes[1].set_xlabel("Cycle")
        axes[1].set_ylabel("Action")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"rl_curve_n{n}.png", dpi=150)
        plt.close(fig)


def _summarize_grid_sweep(grid_sizes, base_results, rl_results):
    rows = []
    for n, b, r in zip(grid_sizes, base_results, rl_results):
        rows.append(
            {
                "n": n,
                "baseline_time": float(b.get("time", np.nan)),
                "baseline_cycles": float(b.get("cycles", np.nan)),
                "rl_time": float(r.get("time", np.nan)),
                "rl_cycles": float(r.get("cycles", np.nan)),
            }
        )
    return rows


def _mean_results(results):
    if not results:
        return {"time": float("nan"), "cycles": float("nan")}
    times = np.array([r.get("time", np.nan) for r in results], dtype=float)
    cycles = np.array([r.get("cycles", np.nan) for r in results], dtype=float)
    return {
        "time": float(np.nanmean(times)),
        "cycles": float(np.nanmean(cycles)),
    }


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
    w_only,
    w_init=None,
    sweeps_init=None,
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
        w_only=w_only,
        w_init=w_init,
        sweeps_init=sweeps_init,
    )


def main():
    lib_path = str(DEFAULT_AMG_ENV_LIBRARY)
    seed_start = int(os.environ.get("EVAL_SEED_START", "100"))
    seed_count = int(os.environ.get("EVAL_SEED_COUNT", "6"))
    eval_seeds = list(range(seed_start, seed_start + seed_count))
    # Default grid size; can be overridden by RANDOMIZE_GRID or GRID_SIZES/GRID_RANGE
    grid_sizes = [(60, 60, 60)]
    grid_sizes_env = _parse_grid_sizes(os.environ.get("GRID_SIZES", ""))
    grid_range_env = _parse_grid_range(os.environ.get("GRID_RANGE", ""))
    if grid_sizes_env:
        grid_sizes = grid_sizes_env
    elif grid_range_env:
        grid_sizes = grid_range_env

    fixed_a_mode = _env_flag("EVAL_FIXED_A", "0")
    randomize_A = _env_flag("RANDOMIZE_A", "1")
    randomize_b = _env_flag("RANDOMIZE_B", "1")
    fixed_rhs_seed = int(os.environ.get("FIXED_RHS_SEED", "123456789"))
    fixed_rhs_type = int(os.environ.get("FIXED_RHS_TYPE", "1"))
    fixed_stencil = int(os.environ.get("FIXED_STENCIL", "0"))
    baseline_relax_type = int(os.environ.get("BASELINE_RELAX_TYPE", "18"))
    rl_relax_type = int(os.environ.get("RL_RELAX_TYPE", "18"))
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
    w_only = _env_flag("W_ONLY", "0")
    w_init = _env_optional_float("W_INIT")
    sweeps_init = _env_optional_int("SWEEPS_INIT")
    plot_grid_sweep = _env_flag("PLOT_GRID_SWEEP", "1")
    plot_grid_step = int(os.environ.get("PLOT_GRID_STEP", "10"))
    plot_sweep_instances = int(os.environ.get("PLOT_SWEEP_INSTANCES", "1"))
    plot_sweep_seed_stride = int(os.environ.get("PLOT_SWEEP_SEED_STRIDE", str(seed_stride)))
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

    baseline_cfg = (1.0, 1, 1)

    model_type = os.environ.get("MODEL_TYPE", "ppo").strip().lower()
    model_path = os.environ.get("MODEL_PATH", "ppo_boomeramg_gen")
    vec_path = os.environ.get("VEC_PATH", "vecnormalize_gen.pkl")
    baseline_lib_path = os.environ.get("BASELINE_LIB_PATH", lib_path)

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
                        relax_type=rl_relax_type,
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
                        w_only=w_only,
                        w_init=w_init,
                        sweeps_init=sweeps_init,
                    )
                ]
            )
            vec_norm = VecNormalize.load(vec_path, raw_env)
            vec_norm.training = False
            vec_norm.norm_reward = False
            print(f"Loaded VecNormalize: {vec_path}")
        except AssertionError as exc:
            print(f"VecNormalize mismatch; skipping {vec_path}. Reason: {exc}")

    if model_type == "lstm":
        model = RecurrentPPO.load(model_path)
    else:
        model = PPO.load(model_path)

    plot_count = int(os.environ.get("PLOT_COUNT", "3"))
    plot_count = max(1, min(plot_count, len(eval_cases)))
    failure_log_path = Path(os.environ.get("EVAL_FAILURE_LOG_PATH", Path.cwd() / "eval_default_vs_rl_failures.jsonl"))

    base_results = []
    rl_results = []
    failure_records = []
    for i, case in enumerate(eval_cases):
        env = _make_env(
            baseline_lib_path,
            case["seed"],
            case["grid"],
            case["randomize_A"],
            case["randomize_b"],
            case["fixed_rhs_seed"],
            case["fixed_rhs_type"],
            relax_type=baseline_relax_type,
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
            w_only=w_only,
            w_init=None,
            sweeps_init=None,
        )
        t0 = time.perf_counter()
        out_base = run_episode(
            env,
            fixed_cfg=baseline_cfg,
            record_curve=(i < plot_count),
            measure_wall=False,
        )
        wall = time.perf_counter() - t0
        out_base["wall_time"] = wall
        out_base["overhead"] = wall - out_base["time"]
        env.close()
        out_base["case"] = case
        base_results.append(out_base)
        if out_base.get("failed", False):
            failure_records.append(_failure_log_record(scope="main_eval", mode="baseline", case=case, result=out_base))

        env2 = _make_env(
            lib_path,
            case["seed"],
            case["grid"],
            case["randomize_A"],
            case["randomize_b"],
            case["fixed_rhs_seed"],
            case["fixed_rhs_type"],
            relax_type=rl_relax_type,
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
            w_only=w_only,
            w_init=w_init,
            sweeps_init=sweeps_init,
        )
        t0 = time.perf_counter()
        out_rl = run_episode(
            env2,
            policy=model,
            record_curve=(i < plot_count),
            obs_normalizer=(vec_norm.normalize_obs if vec_norm is not None else None),
            use_lstm=(model_type == "lstm"),
            measure_wall=False,
            measure_infer=True,
        )
        wall = time.perf_counter() - t0
        out_rl["wall_time"] = wall
        out_rl["overhead"] = wall - out_rl["time"]
        env2.close()
        out_rl["case"] = case
        rl_results.append(out_rl)
        if out_rl.get("failed", False):
            failure_records.append(_failure_log_record(scope="main_eval", mode="rl", case=case, result=out_rl))

    print("\n=== Baseline vs RL (Hypre default baseline) ===")
    print("Baseline config (w, sweeps_down, sweeps_up):", baseline_cfg)
    if baseline_lib_path != lib_path:
        print("Baseline lib:", baseline_lib_path)
    print("Baseline:", _summarize(base_results))
    print("RL      :", _summarize(rl_results))
    print("Baseline failures:", _summarize_failures(base_results))
    print("RL failures      :", _summarize_failures(rl_results))
    base_overhead = _summarize_overhead(base_results)
    if base_overhead is not None:
        print("Baseline overhead (wall - solver dt):", base_overhead)
    rl_overhead = _summarize_overhead(rl_results)
    if rl_overhead is not None:
        print("RL overhead (wall - solver dt):", rl_overhead)
    rl_infer = _summarize_infer(rl_results)
    if rl_infer is not None:
        print("RL inference time (policy.predict only):", rl_infer)
    if randomize_grid:
        print(f"Eval grid range: [{grid_min}, {grid_max}] (uniform, cubic)")
    else:
        print("Eval grid sizes:", grid_sizes)
    print("Eval cases:", len(eval_cases))
    if fixed_a_mode:
        print("Fixed-A mode: ON (A coefficients and size fixed)")

    if base_results and base_results[0]["cfg_curve"] and rl_results[0]["cfg_curve"]:
        base_last = base_results[0]["cfg_curve"][-1]
        rl_last = rl_results[0]["cfg_curve"][-1]
        print("\n=== Final params table (first case) ===")
        print("method    w        sweeps_down  sweeps_up")
        print(f"baseline  {base_last[0]:.6f} {base_last[1]:>12d} {base_last[2]:>10d}")
        print(f"rl        {rl_last[0]:.6f} {rl_last[1]:>12d} {rl_last[2]:>10d}")

        base_cfg = np.array(base_results[0]["cfg_curve"], dtype=float)
        rl_cfg = np.array(rl_results[0]["cfg_curve"], dtype=float)
        base_steps = np.arange(1, base_cfg.shape[0] + 1)
        rl_steps = np.arange(1, rl_cfg.shape[0] + 1)

        plt.figure()
        plt.plot(base_steps, base_cfg[:, 0], label="baseline")
        plt.plot(rl_steps, rl_cfg[:, 0], label="rl")
        plt.title(f"Relaxation weight w (baseline w={baseline_cfg[0]:.3f})")
        plt.xlabel("V-cycle")
        plt.ylabel("w")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(base_steps, base_cfg[:, 1], label="baseline sweeps_down")
        plt.plot(base_steps, base_cfg[:, 2], label="baseline sweeps_up")
        plt.plot(rl_steps, rl_cfg[:, 1], label="rl sweeps_down")
        plt.plot(rl_steps, rl_cfg[:, 2], label="rl sweeps_up")
        plt.title(
            "Sweeps per cycle "
            f"(baseline down={int(baseline_cfg[1])}, up={int(baseline_cfg[2])})"
        )
        plt.xlabel("V-cycle")
        plt.ylabel("sweeps")
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.semilogy(base_results[0]["residual_curve"], alpha=0.7, label="baseline")
        plt.title("Baseline: residual curve (first case)")
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

    if plot_grid_sweep:
        grid_sizes_plot = list(range(grid_min, grid_max + 1, max(1, plot_grid_step)))
        rl_curves = []
        base_means = []
        rl_means = []
        for idx, n in enumerate(grid_sizes_plot):
            base_runs = []
            rl_runs = []
            for j in range(max(1, plot_sweep_instances)):
                seed = seed_start + idx * seed_stride + j * plot_sweep_seed_stride
                case = {
                    "seed": seed,
                    "grid": (n, n, n),
                    "randomize_A": randomize_A,
                    "randomize_b": randomize_b,
                    "fixed_rhs_seed": fixed_rhs_seed,
                    "fixed_rhs_type": fixed_rhs_type,
                }

                envb = _make_env(
                    baseline_lib_path,
                    case["seed"],
                    case["grid"],
                    case["randomize_A"],
                    case["randomize_b"],
                    case["fixed_rhs_seed"],
                    case["fixed_rhs_type"],
                    relax_type=baseline_relax_type,
                    fixed_stencil=fixed_stencil,
                    randomize_grid=False,
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
                    w_only=w_only,
                    w_init=None,
                    sweeps_init=None,
                )
                base_out = run_episode(
                    envb,
                    fixed_cfg=baseline_cfg,
                    record_curve=False,
                )
                envb.close()
                base_runs.append(base_out)
                if base_out.get("failed", False):
                    failure_records.append(_failure_log_record(scope="grid_sweep", mode="baseline", case=case, result=base_out))

                envp = _make_env(
                    lib_path,
                    case["seed"],
                    case["grid"],
                    case["randomize_A"],
                    case["randomize_b"],
                    case["fixed_rhs_seed"],
                    case["fixed_rhs_type"],
                    relax_type=rl_relax_type,
                    fixed_stencil=fixed_stencil,
                    randomize_grid=False,
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
                    w_only=w_only,
                    w_init=w_init,
                    sweeps_init=sweeps_init,
                )
                out = run_episode(
                    envp,
                    policy=model,
                    record_curve=(j == 0),
                    obs_normalizer=(vec_norm.normalize_obs if vec_norm is not None else None),
                    use_lstm=(model_type == "lstm"),
                )
                envp.close()
                rl_runs.append(out)
                if out.get("failed", False):
                    failure_records.append(_failure_log_record(scope="grid_sweep", mode="rl", case=case, result=out))

            base_means.append(_mean_results(base_runs))
            rl_means.append(_mean_results(rl_runs))
            if rl_runs:
                rl_curves.append(rl_runs[0])
        plots_dir = Path(__file__).resolve().parent / "pltos"
        _plot_rl_curves(grid_sizes_plot, rl_curves, plots_dir)
        print(f"Saved RL grid-sweep plots to: {plots_dir}")

        rows = _summarize_grid_sweep(grid_sizes_plot, base_means, rl_means)
        print(f"\n=== Grid Sweep Summary (per n, avg over {max(1, plot_sweep_instances)} inst) ===")
        print("n   baseline_time  baseline_cycles   rl_time   rl_cycles")
        for row in rows:
            print(
                f"{row['n']:>3d} {row['baseline_time']:>14.6f} "
                f"{row['baseline_cycles']:>16.1f} {row['rl_time']:>9.6f} "
                f"{row['rl_cycles']:>10.1f}"
            )

    if failure_records:
        _write_failure_log(failure_log_path, failure_records)
        print("FAILURE LOG:", failure_log_path)
        print("TOTAL FAILED CASES:", len(failure_records))


if __name__ == "__main__":
    main()
