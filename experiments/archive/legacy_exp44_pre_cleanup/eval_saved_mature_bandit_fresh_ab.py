from __future__ import annotations

import json
import os
import pickle
from collections import Counter, deque
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np

from explore_bandit_solve_control import _augment_params
from explore_step_rl_dynamic_control import _classify_rl_failure, _solve_schedule_case
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    build_test10_branches,
    compute_failure_scale_min_runtime_sec,
    default_branch_label,
    default_test_final_bandit_config_from_env,
    generate_difconv_instances,
    run_bandit_step_test_final,
    solve_no_rl_case,
    solve_setup_aware_rl_case,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_float_tuple(name: str, default: str = "") -> Tuple[float, ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _env_joint_actions(name: str, default: str = "") -> Tuple[Tuple[float, int, int], ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    out = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        w, sd, su = item.split(":")
        out.append((float(w), int(sd), int(su)))
    return tuple(out)


def _env_extended_actions(name: str, default: str = "") -> Tuple[Tuple[float, int, int, int, int, int, float, float], ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    out = []
    for item in raw.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [x.strip() for x in item.split(":")]
        if len(parts) < 6:
            raise ValueError(f"{name} item must have at least 6 fields: {item!r}")
        w, sd, su, sc, ct, rt = parts[:6]
        ow = parts[6] if len(parts) > 6 else "-1"
        arw = parts[7] if len(parts) > 7 else "-1"
        out.append((float(w), int(sd), int(su), int(sc), int(ct), int(rt), float(ow), float(arw)))
    return tuple(out)


def _param_key(params: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    items = []
    for k, v in sorted(dict(params).items()):
        if isinstance(v, float):
            items.append((k, round(float(v), 12)))
        else:
            items.append((k, v))
    return tuple(items)


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    runtimes = [float(r["runtime"]) for r in results]
    setup_runtimes = [float(r["setup_runtime"]) for r in results]
    solve_runtimes = [float(r["solve_runtime"]) for r in results]
    failures = [bool(r.get("failed", False)) for r in results]
    iterations = [int(r.get("iterations", -1)) for r in results if int(r.get("iterations", -1)) >= 0]
    final_ws = [float(r.get("final_w")) for r in results if np.isfinite(float(r.get("final_w", np.nan)))]
    return {
        "cases": int(len(results)),
        "mean_runtime": float(mean(runtimes)),
        "total_runtime": float(sum(runtimes)),
        "mean_setup_runtime": float(mean(setup_runtimes)),
        "mean_solve_runtime": float(mean(solve_runtimes)),
        "failed_count": int(sum(failures)),
        "mean_iterations": (float(mean(iterations)) if iterations else float("nan")),
        "mean_final_w": (float(mean(final_ws)) if final_ws else float("nan")),
    }


def _bandit_state_path() -> Path:
    return Path(_env_str("BANDIT_STATE_PATH", "/tmp/mature40_tune7_bandit_state.pkl"))


def _load_or_build_mature_bandit():
    state_path = _bandit_state_path()
    if state_path.exists():
        with state_path.open("rb") as fh:
            payload = pickle.load(fh)
        print(json.dumps({"stage": "bandit_state_load", "path": str(state_path)}), flush=True)
        return payload["branch"]

    grid_n = _env_int("GRID_N", 40)
    tune_dim = _env_int("SETUP_TUNE_DIM", 7)
    bandit_method = _env_str("SETUP_BANDIT_METHOD", "linucbv4").strip().lower()
    tune7_variant = _env_str("TUNE7_VARIANT", "categorical").strip().lower()
    warmup_seed = _env_int("WARMUP_SEED", 39393939)
    warmup_cases = _env_int("WARMUP_CASES", 1500)
    c_min = _env_float("C_MIN", 1.0)
    c_max = _env_float("C_MAX", 1000.0)
    difconv_a = tuple(float(x) for x in _env_str("DIFCONV_A", "0,0,0").split(","))

    print(
        json.dumps(
            {
                "stage": "bandit_state_build_start",
                "path": str(state_path),
                "grid_n": grid_n,
                "tune_dim": tune_dim,
                "bandit_method": bandit_method,
                "tune7_variant": tune7_variant,
                "warmup_seed": warmup_seed,
                "warmup_cases": warmup_cases,
            }
        ),
        flush=True,
    )

    bandit_cfg = default_test_final_bandit_config_from_env()
    branches, _bundle = build_test10_branches(
        final_tune_dims=[int(tune_dim)],
        tune7_variant=tune7_variant,
        seed=int(warmup_seed),
        solver_tol=_env_float("SOLVER_TOL", 1e-6),
        solver_max_iter=_env_int("SOLVER_MAX_ITER", 50),
        include_default=False,
        method_filter=bandit_method,
        branch_filter=_env_str(
            "BRANCH_FILTER",
            default_branch_label(method=bandit_method, tune_dim=tune_dim, tune7_variant=tune7_variant),
        ),
        bandit_cfg=bandit_cfg,
    )
    if len(branches) != 1:
        raise RuntimeError(f"Expected exactly one branch, got {[b.label for b in branches]}")
    branch = branches[0]

    instances = generate_difconv_instances(
        T=warmup_cases,
        seed=warmup_seed,
        grid_choices=[(grid_n, grid_n, grid_n)],
        c_min=c_min,
        c_max=c_max,
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )

    prev_update_est = 0.0
    success_runtime_history = deque(maxlen=max(1, int(bandit_cfg.failure_scale_window)))
    failure_scale_min_runtime_sec = None
    for i, (mkw, context) in enumerate(instances, 1):
        def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
            return solve_no_rl_case(
                params=selected_params,
                mkw=dict(mkw),
                solver_tol=_env_float("SOLVER_TOL", 1e-6),
                solver_max_iter=_env_int("SOLVER_MAX_ITER", 50),
                augment_params=_augment_params,
            )

        if failure_scale_min_runtime_sec is None:
            failure_scale_min_runtime_sec = compute_failure_scale_min_runtime_sec(
                solver_fn=solver_fn,
                params=DEFAULT_SETUP_PARAMS,
                mkw=dict(mkw),
                override_value=float(bandit_cfg.failure_scale_min_runtime_sec_override),
            )

        _params, _out, _timing, _failed_attempts, prev_update_est = run_bandit_step_test_final(
            policy=branch.policy,
            parameter_space=branch.parameter_space,
            context=np.asarray(context, dtype=float),
            solver_fn=solver_fn,
            prev_update_est=float(prev_update_est),
            success_runtime_history=success_runtime_history,
            b_min_runtime_sec=float(failure_scale_min_runtime_sec),
            solver_tol=_env_float("SOLVER_TOL", 1e-6),
            cfg=bandit_cfg,
        )
        if i % 100 == 0:
            print(json.dumps({"stage": "bandit_warmup_progress", "done": i, "total": warmup_cases}), flush=True)

    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("wb") as fh:
        pickle.dump(
            {
                "branch": branch,
                "meta": {
                    "grid_n": grid_n,
                    "tune_dim": tune_dim,
                    "bandit_method": bandit_method,
                    "tune7_variant": tune7_variant,
                    "warmup_seed": warmup_seed,
                    "warmup_cases": warmup_cases,
                },
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(json.dumps({"stage": "bandit_state_build_done", "path": str(state_path)}), flush=True)
    return branch


def _load_solve_policy() -> SetupAwareSolvePolicyRunner:
    model_path = Path(_env_str("MODEL_PATH", "ppo_frozen_bandit_step.zip"))
    vec_path = Path(_env_str("VEC_PATH", "vecnormalize_frozen_bandit_step.pkl"))
    cfg = SetupAwareRLConfig(
        tune_dim=_env_int("SETUP_TUNE_DIM", 7),
        tune7_variant=_env_str("TUNE7_VARIANT", "categorical").strip().lower(),
        algo=_env_str("ALGO", "ppo"),
        model_type=_env_str("MODEL_TYPE", "mlp"),
        model_path=model_path,
        vec_path=vec_path,
        fixed_grid=(_env_int("GRID_N", 40),) * 3,
        difconv_c_range=(_env_float("C_MIN", 1.0), _env_float("C_MAX", 1000.0)),
        w_only=_env_str("W_ONLY", "1").strip().lower() not in {"0", "false", "no"},
        w_center=_env_float("W_CENTER", 1.65),
        w_scale=_env_float("W_SCALE", 0.1),
        sweeps_min=_env_int("SWEEPS_MIN", 1),
        sweeps_max=_env_int("SWEEPS_MAX", 1),
        w_init=None,
        sweeps_init=None,
        solve_max_cycles=_env_int("SOLVE_MAX_CYCLES", 50),
        solve_tol=_env_float("SOLVE_TOL", 1e-6),
        default_setup_params=dict(DEFAULT_SETUP_PARAMS),
        obs_mode=_env_str("OBS_MODE", "solve_only"),
        action_mode=_env_str("ACTION_MODE", "continuous"),
        discrete_w_values=_env_float_tuple("DISCRETE_W_VALUES"),
        discrete_joint_actions=_env_joint_actions("DISCRETE_ACTIONS"),
        discrete_extended_actions=_env_extended_actions("DISCRETE_EXTENDED_ACTIONS"),
        discrete_blend_alphas=_env_float_tuple("DISCRETE_BLEND_ALPHAS"),
    )
    return SetupAwareSolvePolicyRunner(cfg)


def main() -> None:
    branch = _load_or_build_mature_bandit()
    solve_policy = _load_solve_policy()

    grid_n = _env_int("GRID_N", 40)
    eval_seed = _env_int("EVAL_SEED", 39394939)
    eval_cases = _env_int("EVAL_CASES", 500)
    c_min = _env_float("C_MIN", 1.0)
    c_max = _env_float("C_MAX", 1000.0)
    difconv_a = tuple(float(x) for x in _env_str("DIFCONV_A", "0,0,0").split(","))
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    solve_max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    solver_tol = _env_float("SOLVER_TOL", 1e-6)
    solver_max_iter = _env_int("SOLVER_MAX_ITER", 50)
    fixed_w = _env_float("FIXED_W", 1.60)

    print(
        json.dumps(
            {
                "stage": "fresh_eval_start",
                "eval_seed": eval_seed,
                "eval_cases": eval_cases,
                "bandit_updates_during_eval": False,
                "fixed_w": fixed_w,
            }
        ),
        flush=True,
    )

    instances = generate_difconv_instances(
        T=eval_cases,
        seed=eval_seed,
        grid_choices=[(grid_n, grid_n, grid_n)],
        c_min=c_min,
        c_max=c_max,
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )

    default_results: List[Dict[str, Any]] = []
    fixed_results: List[Dict[str, Any]] = []
    ppo_results: List[Dict[str, Any]] = []
    selected_params: List[Dict[str, Any]] = []

    for i, (mkw, context) in enumerate(instances, 1):
        params, _info = branch.policy.select(context=np.asarray(context, dtype=float), parameter_space=branch.parameter_space)
        params = dict(params)
        selected_params.append(dict(params))

        default_results.append(
            solve_no_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solver_tol=solver_tol,
                solver_max_iter=solver_max_iter,
                augment_params=_augment_params,
            )
        )
        fixed_results.append(
            _solve_schedule_case(
                params=dict(params),
                mkw=dict(mkw),
                schedule=[(solve_max_cycles, float(fixed_w), 1, 1)],
                solve_tol=solve_tol,
                solve_max_cycles=solve_max_cycles,
            )
        )
        ppo_results.append(
            solve_setup_aware_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solve_policy=solve_policy,
                augment_params=_augment_params,
                classify_rl_failure=lambda *, residual_norm, iterations: _classify_rl_failure(
                    residual_norm=float(residual_norm),
                    iterations=int(iterations),
                    solve_tol=solve_tol,
                    solve_max_cycles=solve_max_cycles,
                ),
                solve_max_cycles=solve_max_cycles,
            )
        )
        if i % 50 == 0:
            print(json.dumps({"stage": "fresh_eval_progress", "done": i, "total": eval_cases}), flush=True)

    param_counter = Counter(_param_key(p) for p in selected_params)
    total = len(selected_params)
    top = []
    for key, count in param_counter.most_common(10):
        top.append({"count": int(count), "share_pct": float(100.0 * count / total), "params": dict(key)})

    s_default = _summarize(default_results)
    s_fixed = _summarize(fixed_results)
    s_ppo = _summarize(ppo_results)

    print(
        json.dumps(
            {
                "stage": "final",
                "protocol": {
                    "bandit_state_source": str(_bandit_state_path()),
                    "bandit_warmup_cases": _env_int("WARMUP_CASES", 1500),
                    "bandit_frozen_during_eval": True,
                    "eval_cases": eval_cases,
                    "new_eval_seed": eval_seed,
                    "grid": [grid_n, grid_n, grid_n],
                },
                "methods": {
                    "mature_bandit_plus_default_solve": s_default,
                    f"mature_bandit_plus_fixed_w_{fixed_w:.2f}": s_fixed,
                    "mature_bandit_plus_best_ppo": s_ppo,
                },
                "relative": {
                    f"fixed_w_{fixed_w:.2f}_vs_default_pct": float(100.0 * (s_default["mean_runtime"] - s_fixed["mean_runtime"]) / s_default["mean_runtime"]),
                    "ppo_vs_default_pct": float(100.0 * (s_default["mean_runtime"] - s_ppo["mean_runtime"]) / s_default["mean_runtime"]),
                    f"ppo_vs_fixed_w_{fixed_w:.2f}_pct": float(100.0 * (s_fixed["mean_runtime"] - s_ppo["mean_runtime"]) / s_fixed["mean_runtime"]),
                },
                "selected_setup_distribution": {
                    "unique_setups": int(len(param_counter)),
                    "top10": top,
                },
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
