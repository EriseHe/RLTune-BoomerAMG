from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    FAIL_RUNTIME_SEC,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    build_test10_branches,
    compute_failure_scale_min_runtime_sec,
    default_test_final_bandit_config_from_env,
    generate_difconv_instances,
    run_bandit_step_test_final,
    solve_no_rl_case,
)
from solver import create_env


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _augment_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    relax_type_raw = os.environ.get("SETUP_RELAX_TYPE", "").strip()
    if relax_type_raw:
        out["relax_type"] = int(relax_type_raw)
    num_sweeps_raw = os.environ.get("SETUP_NUM_SWEEPS", "").strip()
    if num_sweeps_raw:
        out["num_sweeps"] = int(num_sweeps_raw)
    cycle_type_raw = os.environ.get("SETUP_CYCLE_TYPE", "").strip()
    if cycle_type_raw:
        out["cycle_type"] = int(cycle_type_raw)
    max_levels_raw = os.environ.get("SETUP_MAX_LEVELS", "").strip()
    if max_levels_raw:
        out["max_levels"] = int(max_levels_raw)
    return out


def _classify_rl_failure(*, residual_norm: float, iterations: int, solve_tol: float, solve_max_cycles: int) -> str:
    if not np.isfinite(float(residual_norm)):
        return "non_finite_residual_norm"
    if float(residual_norm) > float(solve_tol) and int(iterations) >= int(solve_max_cycles):
        return "residual_above_solve_tol;max_cycles_reached_without_convergence"
    if float(residual_norm) > float(solve_tol):
        return "residual_above_solve_tol"
    if int(iterations) >= int(solve_max_cycles):
        return "hit_or_exceeded_solve_max_cycles"
    return ""


def _solve_fixed_w_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    w: float,
    sweeps_down: int,
    sweeps_up: int,
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    try:
        params = _augment_params(params)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            solve_runtime = 0.0
            residual_norm = float(env.r0)
            iterations = 0
            for cycle in range(int(solve_max_cycles)):
                residual_norm, dt = env.step_rl(
                    relax_weight=float(w),
                    sweeps_down=int(sweeps_down),
                    sweeps_up=int(sweeps_up),
                )
                solve_runtime += float(dt)
                iterations = cycle + 1
                if float(residual_norm) <= float(solve_tol):
                    break
        failure_reason = _classify_rl_failure(
            residual_norm=float(residual_norm),
            iterations=int(iterations),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
        )
        return {
            "runtime": float(prep.setup_runtime_sec + solve_runtime),
            "setup_runtime": float(prep.setup_runtime_sec),
            "solve_runtime": float(solve_runtime),
            "infer_runtime": 0.0,
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "final_w": float(w),
            "final_sweeps_down": int(sweeps_down),
            "final_sweeps_up": int(sweeps_up),
        }
    except Exception as exc:
        return {
            "runtime": float(FAIL_RUNTIME_SEC),
            "setup_runtime": float(FAIL_RUNTIME_SEC),
            "solve_runtime": 0.0,
            "infer_runtime": 0.0,
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(solve_max_cycles),
            "final_w": float(w),
            "final_sweeps_down": int(sweeps_down),
            "final_sweeps_up": int(sweeps_up),
        }


def _make_policy_runner(model_path: Path, vec_path: Path) -> SetupAwareSolvePolicyRunner:
    grid = tuple(int(x) for x in os.environ.get("GRID_SIZES", "60,60,60").split(","))
    cfg = SetupAwareRLConfig(
        tune_dim=int(os.environ.get("SETUP_TUNE_DIM", "5")),
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        model_type=os.environ.get("MODEL_TYPE", "mlp").strip().lower(),
        model_path=model_path,
        vec_path=vec_path,
        fixed_grid=grid,
        difconv_c_range=tuple(float(x) for x in os.environ.get("DIFCONV_C_RANGE", "1,1000").split(",")),
        w_only=_env_flag("W_ONLY", "1"),
        w_center=float(os.environ.get("W_CENTER", "1.25")),
        w_scale=float(os.environ.get("W_SCALE", "0.75")),
        sweeps_min=int(os.environ.get("SWEEPS_MIN", "1")),
        sweeps_max=int(os.environ.get("SWEEPS_MAX", "1")),
        w_init=None,
        sweeps_init=None,
        solve_max_cycles=int(os.environ.get("SOLVE_MAX_CYCLES", "50")),
        solve_tol=float(os.environ.get("SOLVE_TOL", "1e-6")),
        default_setup_params=dict(DEFAULT_SETUP_PARAMS),
    )
    return SetupAwareSolvePolicyRunner(cfg)


def _build_instances() -> Sequence[Tuple[Dict[str, Any], np.ndarray]]:
    grid_choices = [tuple(int(x) for x in os.environ.get("GRID_SIZES", "60,60,60").split(","))]
    c_min, c_max = (float(x) for x in os.environ.get("DIFCONV_C_RANGE", "1,1000").split(","))
    difconv_a = tuple(float(x) for x in os.environ.get("DIFCONV_A", "0,0,0").split(","))
    return generate_difconv_instances(
        T=int(os.environ.get("T", "100")),
        seed=int(os.environ.get("SEED", "39393939")),
        grid_choices=grid_choices,
        c_min=float(c_min),
        c_max=float(c_max),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )


def _build_bandit_branch():
    branches, _bundle = build_test10_branches(
        final_tune_dims=[int(os.environ.get("SETUP_TUNE_DIM", "5"))],
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        seed=int(os.environ.get("SEED", "39393939")),
        solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
        solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
        include_default=False,
        method_filter=os.environ.get("SETUP_BANDIT_METHOD", "linucbv3"),
        branch_filter=os.environ.get("BRANCH_FILTER", ""),
        bandit_cfg=default_test_final_bandit_config_from_env(),
    )
    if len(branches) != 1:
        labels = ", ".join(branch.label for branch in branches)
        raise ValueError(f"Expected exactly one branch, got {len(branches)}: {labels}")
    return branches[0]


def _controller_specs() -> List[Tuple[str, str, Dict[str, Any]]]:
    specs: List[Tuple[str, str, Dict[str, Any]]] = [
        ("bandit_only_replay", "no_rl", {}),
        ("fixed_w_0.90", "fixed", {"w": 0.90, "sd": 1, "su": 1}),
        ("fixed_w_0.95", "fixed", {"w": 0.95, "sd": 1, "su": 1}),
        ("fixed_w_1.00", "fixed", {"w": 1.00, "sd": 1, "su": 1}),
        ("fixed_w_1.10", "fixed", {"w": 1.10, "sd": 1, "su": 1}),
        ("fixed_w_1.20", "fixed", {"w": 1.20, "sd": 1, "su": 1}),
    ]
    if not _env_flag("SKIP_DEFAULT_PPO", "0"):
        repo = Path(__file__).resolve().parent
        candidates = [
            ("ppo_random", repo / "ppo_boomeramg_setup_random.zip", repo / "vecnormalize_setup_random.pkl"),
            ("ppo_bandit", repo / "ppo_boomeramg_setup_bandit.zip", repo / "vecnormalize_setup_bandit.pkl"),
            (
                "ppo_bandit_frozen_small",
                repo / "ppo_boomeramg_setup_bandit_frozen_v3_t5_small.zip",
                repo / "vecnormalize_setup_bandit_frozen_v3_t5_small.pkl",
            ),
        ]
        for name, model_path, vec_path in candidates:
            if model_path.exists() and vec_path.exists():
                specs.append((name, "ppo", {"model_path": model_path, "vec_path": vec_path}))
    extra_name = os.environ.get("EXTRA_MODEL_NAME", "").strip()
    extra_model = os.environ.get("EXTRA_MODEL_PATH", "").strip()
    extra_vec = os.environ.get("EXTRA_VEC_PATH", "").strip()
    if extra_name and extra_model and extra_vec:
        specs.append(
            (
                extra_name,
                "ppo",
                {
                    "model_path": Path(extra_model),
                    "vec_path": Path(extra_vec),
                },
            )
        )
    return specs


def main() -> None:
    instances = _build_instances()
    branch = _build_bandit_branch()
    bandit_cfg = default_test_final_bandit_config_from_env()
    solve_tol = float(os.environ.get("SOLVE_TOL", "1e-6"))
    solve_max_cycles = int(os.environ.get("SOLVE_MAX_CYCLES", "50"))
    solver_tol = float(os.environ.get("SOLVER_TOL", "1e-6"))
    solver_max_iter = int(os.environ.get("SOLVER_MAX_ITER", "50"))

    trace: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
    prev_update_est = 0.0
    success_runtime_history = deque(maxlen=max(1, int(bandit_cfg.failure_scale_window)))
    failure_scale_min_runtime_sec = None

    for mkw, context in instances:
        def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
            return solve_no_rl_case(
                params=selected_params,
                mkw=dict(mkw),
                solver_tol=float(solver_tol),
                solver_max_iter=int(solver_max_iter),
                augment_params=_augment_params,
            )

        if failure_scale_min_runtime_sec is None:
            failure_scale_min_runtime_sec = compute_failure_scale_min_runtime_sec(
                solver_fn=solver_fn,
                params=DEFAULT_SETUP_PARAMS,
                mkw=dict(mkw),
                override_value=float(bandit_cfg.failure_scale_min_runtime_sec_override),
            )

        params, out, _timing, _failed_attempts, prev_update_est = run_bandit_step_test_final(
            policy=branch.policy,
            parameter_space=branch.parameter_space,
            context=np.asarray(context, dtype=float),
            solver_fn=solver_fn,
            prev_update_est=float(prev_update_est),
            success_runtime_history=success_runtime_history,
            b_min_runtime_sec=float(failure_scale_min_runtime_sec),
            solver_tol=float(solver_tol),
            cfg=bandit_cfg,
        )
        trace.append((dict(mkw), dict(params), dict(out)))

    controllers = _controller_specs()
    ppo_runners: Dict[str, SetupAwareSolvePolicyRunner] = {}
    results: Dict[str, Dict[str, Any]] = {}

    for name, kind, extra in controllers:
        runtimes = []
        setup_runtimes = []
        solve_runtimes = []
        failures = 0
        iterations = []
        w_values = []
        for mkw, params, bandit_only_out in trace:
            if kind == "no_rl":
                out = solve_no_rl_case(
                    params=params,
                    mkw=dict(mkw),
                    solver_tol=float(solver_tol),
                    solver_max_iter=int(solver_max_iter),
                    augment_params=_augment_params,
                )
            elif kind == "fixed":
                out = _solve_fixed_w_case(
                    params=params,
                    mkw=dict(mkw),
                    w=float(extra["w"]),
                    sweeps_down=int(extra["sd"]),
                    sweeps_up=int(extra["su"]),
                    solve_tol=float(solve_tol),
                    solve_max_cycles=int(solve_max_cycles),
                )
            elif kind == "ppo":
                if name not in ppo_runners:
                    ppo_runners[name] = _make_policy_runner(
                        model_path=Path(extra["model_path"]),
                        vec_path=Path(extra["vec_path"]),
                    )
                runner = ppo_runners[name]
                out = {
                    **_solve_fixed_w_case(
                        params=params,
                        mkw=dict(mkw),
                        w=1.0,
                        sweeps_down=1,
                        sweeps_up=1,
                        solve_tol=float(solve_tol),
                        solve_max_cycles=int(solve_max_cycles),
                    )
                }
                # Overwrite the fixed-controller result with PPO output using the same setup trace.
                try:
                    from setup_aware_compare_common import solve_setup_aware_rl_case

                    out = solve_setup_aware_rl_case(
                        params=params,
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
                    )
                except Exception as exc:
                    out = {
                        "runtime": float(FAIL_RUNTIME_SEC),
                        "setup_runtime": float(FAIL_RUNTIME_SEC),
                        "solve_runtime": 0.0,
                        "infer_runtime": 0.0,
                        "failed": True,
                        "failure_reason": f"exception:{type(exc).__name__}:{exc}",
                        "residual_norm": float("inf"),
                        "iterations": int(solve_max_cycles),
                        "final_w": float("nan"),
                        "final_sweeps_down": -1,
                        "final_sweeps_up": -1,
                    }
            else:
                raise ValueError(kind)

            runtimes.append(float(out["runtime"]))
            setup_runtimes.append(float(out["setup_runtime"]))
            solve_runtimes.append(float(out["solve_runtime"]))
            failures += int(bool(out.get("failed", False)))
            iterations.append(int(out.get("iterations", 0)))
            if np.isfinite(float(out.get("final_w", np.nan))):
                w_values.append(float(out["final_w"]))

        results[name] = {
            "mean_runtime": float(np.mean(runtimes)),
            "mean_setup_runtime": float(np.mean(setup_runtimes)),
            "mean_solve_runtime": float(np.mean(solve_runtimes)),
            "failed_count": int(failures),
            "mean_iterations": float(np.mean(iterations)),
            "mean_final_w": float(np.mean(w_values)) if w_values else None,
        }

    sorted_items = sorted(results.items(), key=lambda kv: kv[1]["mean_runtime"])
    ref = results["bandit_only_replay"]["mean_runtime"]
    print("Fixed setup-trace replay results")
    print(f"cases={len(trace)} bandit_branch={branch.label}")
    for name, summary in sorted_items:
        delta = float(summary["mean_runtime"] - ref)
        mean_final_w = summary["mean_final_w"]
        w_text = "-" if mean_final_w is None else f"{mean_final_w:.4f}"
        print(
            f"{name:24s} mean_total={summary['mean_runtime']:.6f} "
            f"delta_vs_bandit={delta:+.6f} "
            f"mean_setup={summary['mean_setup_runtime']:.6f} "
            f"mean_solve={summary['mean_solve_runtime']:.6f} "
            f"fail={summary['failed_count']:4d} "
            f"mean_iters={summary['mean_iterations']:.2f} "
            f"mean_final_w={w_text}"
        )


if __name__ == "__main__":
    main()
