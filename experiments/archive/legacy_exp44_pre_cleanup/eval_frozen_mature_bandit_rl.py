from __future__ import annotations

import json
import os
from pathlib import Path
from statistics import mean

from explore_bandit_solve_control import _augment_params
from explore_step_rl_dynamic_control import _classify_rl_failure
from setup_aware_compare_common import SetupAwareRLConfig, SetupAwareSolvePolicyRunner, solve_no_rl_case, solve_setup_aware_rl_case
from train_ppo_frozen_bandit_step import _cached_fixed_trace, _skip_trace_prefix


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_float_tuple(name: str, default: str = "") -> tuple[float, ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _summarize(rows):
    final_ws = [float(x["final_w"]) for x in rows if x.get("final_w") is not None]
    return {
        "mean_runtime": float(mean(float(x["runtime"]) for x in rows)),
        "mean_setup_runtime": float(mean(float(x["setup_runtime"]) for x in rows)),
        "mean_solve_runtime": float(mean(float(x["solve_runtime"]) for x in rows)),
        "failed_count": int(sum(int(bool(x.get("failed", False))) for x in rows)),
        "mean_iterations": float(mean(int(x["iterations"]) for x in rows)),
        "mean_final_w": (float(mean(final_ws)) if final_ws else None),
    }


def main() -> None:
    grid_n = _env_int("GRID_N_FIXED", 40)
    grid = (grid_n, grid_n, grid_n)
    seed = _env_int("SEED", 39393939)
    eval_seed_offset = _env_int("EVAL_SEED_OFFSET", 1000)
    trace_t = _env_int("TRACE_T", 1500)
    skip_prefix = _env_int("TRACE_SKIP_PREFIX", 1000)
    eval_cases = _env_int("EVAL_CASES", 500)
    tune_dim = _env_int("SETUP_TUNE_DIM", 5)
    bandit_method = _env_str("SETUP_BANDIT_METHOD", "linucbv3").strip().lower()
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    solve_max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    solver_tol = _env_float("SOLVER_TOL", 1e-6)
    solver_max_iter = _env_int("SOLVER_MAX_ITER", 50)

    trace = _cached_fixed_trace(
        T=int(trace_t),
        grid=grid,
        seed=int(seed + eval_seed_offset),
        tune_dim=int(tune_dim),
        bandit_method=str(bandit_method),
        solve_mode="no_rl",
        solve_policy=None,
    )
    eval_trace = list(_skip_trace_prefix(trace, int(skip_prefix)))[: int(eval_cases)]
    if len(eval_trace) != int(eval_cases):
        raise RuntimeError(f"Expected {eval_cases} eval cases after skip, got {len(eval_trace)}")

    cfg = SetupAwareRLConfig(
        tune_dim=int(tune_dim),
        tune7_variant=_env_str("TUNE7_VARIANT", "categorical").strip().lower(),
        algo=_env_str("SETUP_RL_ALGO", "ppo").strip().lower(),
        model_type=_env_str("SETUP_RL_MODEL_TYPE", "mlp").strip().lower(),
        model_path=Path(_env_str("SETUP_RL_MODEL_PATH", "/tmp/ppo_mature40_warmup1000_wonly_solveonly.zip")),
        vec_path=Path(_env_str("SETUP_RL_VEC_PATH", "/tmp/vec_mature40_warmup1000_wonly_solveonly.pkl")),
        fixed_grid=grid,
        difconv_c_range=tuple(float(x) for x in _env_str("DIFCONV_C_RANGE", "1,1000").split(",")),
        w_only=_env_str("W_ONLY", "1").strip().lower() not in {"0", "false", "no"},
        w_center=_env_float("W_CENTER", 1.5),
        w_scale=_env_float("W_SCALE", 0.2),
        sweeps_min=_env_int("SWEEPS_MIN", 1),
        sweeps_max=_env_int("SWEEPS_MAX", 1),
        w_init=None,
        sweeps_init=None,
        solve_max_cycles=int(solve_max_cycles),
        solve_tol=float(solve_tol),
        default_setup_params={},
        obs_mode=_env_str("SETUP_RL_OBS_MODE", "solve_only").strip().lower(),
        action_mode=_env_str("ACTION_MODE", "continuous").strip().lower(),
        discrete_w_values=_env_float_tuple("DISCRETE_W_VALUES"),
    )
    runner = SetupAwareSolvePolicyRunner(cfg)

    no_rl_rows = []
    rl_rows = []
    denom = max(1, len(eval_trace) - 1)
    for idx, (mkw, params) in enumerate(eval_trace):
        no_rl_rows.append(
            solve_no_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solver_tol=float(solver_tol),
                solver_max_iter=int(solver_max_iter),
                augment_params=_augment_params,
            )
        )
        rl_rows.append(
            solve_setup_aware_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solve_policy=runner,
                augment_params=_augment_params,
                classify_rl_failure=lambda *, residual_norm, iterations: _classify_rl_failure(
                    residual_norm=float(residual_norm),
                    iterations=int(iterations),
                    solve_tol=float(solve_tol),
                    solve_max_cycles=int(solve_max_cycles),
                ),
                solve_max_cycles=int(solve_max_cycles),
                case_progress=float(idx) / float(denom),
            )
        )
        if (idx + 1) % 50 == 0:
            print(json.dumps({"stage": "progress", "done": idx + 1, "total": len(eval_trace)}), flush=True)

    a = _summarize(no_rl_rows)
    b = _summarize(rl_rows)
    print(
        json.dumps(
            {
                "stage": "final",
                "grid": list(grid),
                "trace_t": int(trace_t),
                "skip_prefix": int(skip_prefix),
                "eval_cases": int(eval_cases),
                "seed": int(seed),
                "eval_seed": int(seed + eval_seed_offset),
                "bandit_only": a,
                "bandit_plus_rl": b,
                "delta": float(b["mean_runtime"] - a["mean_runtime"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
