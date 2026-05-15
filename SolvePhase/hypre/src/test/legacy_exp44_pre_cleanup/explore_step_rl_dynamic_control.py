from __future__ import annotations

import os
import warnings
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from explore_bandit_solve_control import _augment_params
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    FAIL_RUNTIME_SEC,
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
from solver import create_env


warnings.filterwarnings("ignore", category=RuntimeWarning)


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


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _fixed_trace(
    *,
    T: int,
    grid: Tuple[int, int, int],
    seed: int,
    tune_dim: int,
    bandit_method: str,
    solve_mode: str = "no_rl",
    solve_policy: Optional[SetupAwareSolvePolicyRunner] = None,
) -> Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]:
    difconv_a = tuple(float(x) for x in os.environ.get("DIFCONV_A", "0,0,0").split(","))
    instances = generate_difconv_instances(
        T=int(T),
        seed=int(seed),
        grid_choices=[tuple(int(x) for x in grid)],
        c_min=float(os.environ.get("C_MIN", "1.0")),
        c_max=float(os.environ.get("C_MAX", "1000.0")),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )
    bandit_cfg = default_test_final_bandit_config_from_env()
    branches, _bundle = build_test10_branches(
        final_tune_dims=[int(tune_dim)],
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        seed=int(seed),
        solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
        solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
        include_default=False,
        method_filter=str(bandit_method),
        branch_filter=os.environ.get(
            "BRANCH_FILTER",
            default_branch_label(
                method=str(bandit_method),
                tune_dim=int(tune_dim),
                tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
            ),
        ),
        bandit_cfg=bandit_cfg,
    )
    if len(branches) != 1:
        labels = ", ".join(branch.label for branch in branches)
        raise ValueError(f"Expected one branch, got {len(branches)}: {labels}")
    branch = branches[0]
    prev_update_est = 0.0
    success_runtime_history = deque(maxlen=max(1, int(bandit_cfg.failure_scale_window)))
    failure_scale_min_runtime_sec = None
    trace: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for mkw, context in instances:
        if str(solve_mode).strip().lower() == "no_rl":
            def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_no_rl_case(
                    params=selected_params,
                    mkw=dict(mkw),
                    solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
                    solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
                    augment_params=_augment_params,
                )
            solver_tol = float(os.environ.get("SOLVER_TOL", "1e-6"))
        elif str(solve_mode).strip().lower() == "rl":
            if solve_policy is None:
                raise ValueError("solve_policy is required when solve_mode='rl'")

            def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_setup_aware_rl_case(
                    params=selected_params,
                    mkw=dict(mkw),
                    solve_policy=solve_policy,
                    augment_params=_augment_params,
                    classify_rl_failure=lambda *, residual_norm, iterations: _classify_rl_failure(
                        residual_norm=float(residual_norm),
                        iterations=int(iterations),
                        solve_tol=float(solve_policy.cfg.solve_tol),
                        solve_max_cycles=int(solve_policy.cfg.solve_max_cycles),
                    ),
                    solve_max_cycles=int(solve_policy.cfg.solve_max_cycles),
                )
            solver_tol = float(solve_policy.cfg.solve_tol)
        else:
            raise ValueError(f"Unsupported solve_mode: {solve_mode}")

        if failure_scale_min_runtime_sec is None:
            failure_scale_min_runtime_sec = compute_failure_scale_min_runtime_sec(
                solver_fn=solver_fn,
                params=DEFAULT_SETUP_PARAMS,
                mkw=dict(mkw),
                override_value=float(bandit_cfg.failure_scale_min_runtime_sec_override),
            )

        params, _out, _timing, _failed_attempts, prev_update_est = run_bandit_step_test_final(
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
        trace.append((dict(mkw), dict(params)))
    return trace


def _solve_schedule_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    schedule: Sequence[Tuple[int, float, int, int]],
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    try:
        params = _augment_params(params)
        schedule_sorted = sorted((int(end), float(w), int(sd), int(su)) for end, w, sd, su in schedule)
        with create_env(**mkw) as env:
            prep = env.prepare_rl(params=params)
            solve_runtime = 0.0
            residual_norm = float(env.r0)
            iterations = 0
            last_w = float("nan")
            last_sd = -1
            last_su = -1
            for cycle in range(int(solve_max_cycles)):
                chosen = schedule_sorted[-1]
                for end_cycle, w, sd, su in schedule_sorted:
                    if cycle < int(end_cycle):
                        chosen = (end_cycle, w, sd, su)
                        break
                _end, w, sd, su = chosen
                residual_norm, dt = env.step_rl(
                    relax_weight=float(w),
                    sweeps_down=int(sd),
                    sweeps_up=int(su),
                )
                solve_runtime += float(dt)
                iterations = cycle + 1
                last_w = float(w)
                last_sd = int(sd)
                last_su = int(su)
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
            "failed": bool(failure_reason),
            "failure_reason": str(failure_reason),
            "residual_norm": float(residual_norm),
            "iterations": int(iterations),
            "final_w": float(last_w),
            "final_sweeps_down": int(last_sd),
            "final_sweeps_up": int(last_su),
        }
    except Exception as exc:
        return {
            "runtime": float(FAIL_RUNTIME_SEC),
            "setup_runtime": float(FAIL_RUNTIME_SEC),
            "solve_runtime": 0.0,
            "failed": True,
            "failure_reason": f"exception:{type(exc).__name__}:{exc}",
            "residual_norm": float("inf"),
            "iterations": int(solve_max_cycles),
            "final_w": float("nan"),
            "final_sweeps_down": -1,
            "final_sweeps_up": -1,
        }


def _evaluate_case_schedule_runtime(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    schedule: Sequence[Tuple[int, float, int, int]],
    solve_tol: float,
    solve_max_cycles: int,
) -> float:
    out = _solve_schedule_case(
        params=params,
        mkw=mkw,
        schedule=schedule,
        solve_tol=solve_tol,
        solve_max_cycles=solve_max_cycles,
    )
    return float(out["runtime"])


def _greedy_case_schedule(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    solve_tol: float,
    solve_max_cycles: int,
    action_grid: Sequence[Tuple[float, int, int]],
    tail_action: Tuple[float, int, int],
) -> Tuple[List[Tuple[int, float, int, int]], float]:
    prefix: List[Tuple[int, float, int, int]] = []
    best_runtime = float("inf")
    for cycle in range(int(solve_max_cycles)):
        best_action = None
        best_action_runtime = float("inf")
        for w, sd, su in action_grid:
            trial_schedule = list(prefix)
            trial_schedule.append((cycle + 1, float(w), int(sd), int(su)))
            trial_schedule.append(
                (
                    int(solve_max_cycles),
                    float(tail_action[0]),
                    int(tail_action[1]),
                    int(tail_action[2]),
                )
            )
            runtime = _evaluate_case_schedule_runtime(
                params=params,
                mkw=mkw,
                schedule=trial_schedule,
                solve_tol=solve_tol,
                solve_max_cycles=solve_max_cycles,
            )
            if runtime < best_action_runtime:
                best_action_runtime = float(runtime)
                best_action = (float(w), int(sd), int(su))
        assert best_action is not None
        prefix.append((cycle + 1, best_action[0], best_action[1], best_action[2]))
        best_runtime = float(best_action_runtime)
    return prefix, best_runtime


def _greedy_trace_summary(
    *,
    trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    solve_tol: float,
    solve_max_cycles: int,
    action_grid: Sequence[Tuple[float, int, int]],
    tail_action: Tuple[float, int, int],
) -> Dict[str, Any]:
    runtimes: List[float] = []
    schedule_lengths: List[int] = []
    for mkw, params in trace:
        schedule, runtime = _greedy_case_schedule(
            params=params,
            mkw=dict(mkw),
            solve_tol=solve_tol,
            solve_max_cycles=solve_max_cycles,
            action_grid=action_grid,
            tail_action=tail_action,
        )
        runtimes.append(float(runtime))
        schedule_lengths.append(len(schedule))
    return {
        "mean_runtime": float(np.mean(runtimes)),
        "mean_schedule_len": float(np.mean(schedule_lengths)),
        "tail_action": tuple(float(x) if i == 0 else int(x) for i, x in enumerate(tail_action)),
    }


def _best_two_phase_schedule_for_case(
    *,
    params: Dict[str, Any],
    mkw: Dict[str, Any],
    solve_tol: float,
    solve_max_cycles: int,
    w_grid: Sequence[float],
    sweep_values: Sequence[int],
    switch_points: Sequence[int],
) -> Dict[str, Any]:
    best = None
    for k in switch_points:
        for w1 in w_grid:
            for w2 in w_grid:
                for sd1 in sweep_values:
                    for su1 in sweep_values:
                        for sd2 in sweep_values:
                            for su2 in sweep_values:
                                schedule = [
                                    (int(k), float(w1), int(sd1), int(su1)),
                                    (int(solve_max_cycles), float(w2), int(sd2), int(su2)),
                                ]
                                out = _solve_schedule_case(
                                    params=params,
                                    mkw=mkw,
                                    schedule=schedule,
                                    solve_tol=solve_tol,
                                    solve_max_cycles=solve_max_cycles,
                                )
                                if best is None or float(out["runtime"]) < float(best["runtime"]):
                                    best = {
                                        "runtime": float(out["runtime"]),
                                        "failed": bool(out["failed"]),
                                        "iterations": int(out["iterations"]),
                                        "schedule": schedule,
                                    }
    assert best is not None
    return best


def _case_oracle_two_phase_summary(
    *,
    trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]],
    solve_tol: float,
    solve_max_cycles: int,
    w_grid: Sequence[float],
    sweep_values: Sequence[int],
    switch_points: Sequence[int],
) -> Dict[str, Any]:
    best_runtimes: List[float] = []
    failed_count = 0
    iterations = []
    switch_hist: Dict[int, int] = {}
    w_pair_hist: Dict[Tuple[float, float], int] = {}
    for mkw, params in trace:
        best = _best_two_phase_schedule_for_case(
            params=params,
            mkw=dict(mkw),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
            w_grid=w_grid,
            sweep_values=sweep_values,
            switch_points=switch_points,
        )
        best_runtimes.append(float(best["runtime"]))
        failed_count += int(bool(best["failed"]))
        iterations.append(int(best["iterations"]))
        first, second = best["schedule"]
        switch_hist[int(first[0])] = switch_hist.get(int(first[0]), 0) + 1
        w_pair = (float(first[1]), float(second[1]))
        w_pair_hist[w_pair] = w_pair_hist.get(w_pair, 0) + 1
    top_switch = max(switch_hist.items(), key=lambda kv: kv[1])[0] if switch_hist else None
    top_w_pair = max(w_pair_hist.items(), key=lambda kv: kv[1])[0] if w_pair_hist else None
    return {
        "mean_runtime": float(np.mean(best_runtimes)),
        "failed_count": int(failed_count),
        "mean_iterations": float(np.mean(iterations)) if iterations else 0.0,
        "top_switch": top_switch,
        "top_w_pair": top_w_pair,
    }


def _make_policy_runner() -> SetupAwareSolvePolicyRunner:
    grid = tuple(int(x) for x in os.environ.get("GRID_SIZES", "60,60,60").split(","))
    cfg = SetupAwareRLConfig(
        tune_dim=int(os.environ.get("SETUP_TUNE_DIM", "5")),
        tune7_variant=os.environ.get("TUNE7_VARIANT", "categorical").strip().lower(),
        model_type=os.environ.get("MODEL_TYPE", "mlp").strip().lower(),
        model_path=Path(os.environ["MODEL_PATH"]),
        vec_path=Path(os.environ["VEC_PATH"]),
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


def _eval_runner(runner: SetupAwareSolvePolicyRunner, trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
    from setup_aware_compare_common import solve_setup_aware_rl_case

    solve_tol = float(os.environ.get("SOLVE_TOL", "1e-6"))
    solve_max_cycles = int(os.environ.get("SOLVE_MAX_CYCLES", "50"))
    vals = []
    fails = 0
    action_hist: Dict[int, int] = {}
    denom = max(1, len(trace) - 1)
    for idx_case, (mkw, params) in enumerate(trace):
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
            case_progress=float(idx_case) / float(denom),
        )
        vals.append(out)
        fails += int(bool(out["failed"]))
        for idx, count in dict(out.get("action_counts", {})).items():
            key = int(idx)
            action_hist[key] = int(action_hist.get(key, 0)) + int(count)
    return {
        "mean_runtime": float(np.mean([v["runtime"] for v in vals])),
        "mean_setup_runtime": float(np.mean([v["setup_runtime"] for v in vals])),
        "mean_solve_runtime": float(np.mean([v["solve_runtime"] for v in vals])),
        "failed_count": int(fails),
        "mean_iterations": float(np.mean([v["iterations"] for v in vals])),
        "mean_final_w": float(np.mean([v["final_w"] for v in vals if np.isfinite(v["final_w"])])),
        "action_hist": {int(k): int(action_hist[k]) for k in sorted(action_hist)},
    }


def main() -> None:
    seed = int(os.environ.get("SEED", "39393939"))
    T = int(os.environ.get("T", "10"))
    grid = tuple(int(x) for x in os.environ.get("GRID_SIZES", "60,60,60").split(","))
    tune_dim = int(os.environ.get("SETUP_TUNE_DIM", "5"))
    bandit_method = os.environ.get("SETUP_BANDIT_METHOD", "linucbv3").strip().lower()
    solve_tol = float(os.environ.get("SOLVE_TOL", "1e-6"))
    solve_max_cycles = int(os.environ.get("SOLVE_MAX_CYCLES", "50"))
    w_only = _env_flag("W_ONLY", "1")

    trace = _fixed_trace(T=T, grid=grid, seed=seed, tune_dim=tune_dim, bandit_method=bandit_method)
    bandit_only_vals = [
        solve_no_rl_case(
            params=params,
            mkw=dict(mkw),
            solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
            solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
            augment_params=_augment_params,
        )
        for mkw, params in trace
    ]
    bandit_only_mean = float(np.mean([v["runtime"] for v in bandit_only_vals]))

    w_grid = [float(x) for x in os.environ.get("ORACLE_W_GRID", "0.8,1.0,1.2,1.4,1.6,1.8").split(",")]
    if w_only:
        sweep_values = [1]
    else:
        sweep_values = [int(x) for x in os.environ.get("ORACLE_SWEEPS", "1,2").split(",")]
    switch_points = [int(x) for x in os.environ.get("ORACLE_SWITCHES", "3,5,8,12,20").split(",")]

    best_const = None
    for w in w_grid:
        for sd in sweep_values:
            for su in sweep_values:
                vals = [
                    _solve_schedule_case(
                        params=params,
                        mkw=dict(mkw),
                        schedule=[(solve_max_cycles, float(w), int(sd), int(su))],
                        solve_tol=float(solve_tol),
                        solve_max_cycles=int(solve_max_cycles),
                    )
                    for mkw, params in trace
                ]
                mean_runtime = float(np.mean([v["runtime"] for v in vals]))
                if best_const is None or mean_runtime < best_const["mean_runtime"]:
                    best_const = {
                        "mean_runtime": mean_runtime,
                        "failed_count": int(sum(int(bool(v["failed"])) for v in vals)),
                        "w1": float(w),
                        "sd1": int(sd),
                        "su1": int(su),
                    }

    best_two_phase = None
    for k in switch_points:
        for w1 in w_grid:
            for w2 in w_grid:
                for sd1 in sweep_values:
                    for su1 in sweep_values:
                        for sd2 in sweep_values:
                            for su2 in sweep_values:
                                vals = [
                                    _solve_schedule_case(
                                        params=params,
                                        mkw=dict(mkw),
                                        schedule=[
                                            (int(k), float(w1), int(sd1), int(su1)),
                                            (int(solve_max_cycles), float(w2), int(sd2), int(su2)),
                                        ],
                                        solve_tol=float(solve_tol),
                                        solve_max_cycles=int(solve_max_cycles),
                                    )
                                    for mkw, params in trace
                                ]
                                mean_runtime = float(np.mean([v["runtime"] for v in vals]))
                                if best_two_phase is None or mean_runtime < best_two_phase["mean_runtime"]:
                                    best_two_phase = {
                                        "mean_runtime": mean_runtime,
                                        "failed_count": int(sum(int(bool(v["failed"])) for v in vals)),
                                        "switch": int(k),
                                        "w1": float(w1),
                                        "sd1": int(sd1),
                                        "su1": int(su1),
                                        "w2": float(w2),
                                        "sd2": int(sd2),
                                        "su2": int(su2),
                                    }

    greedy_summary = None
    if _env_flag("RUN_GREEDY_ORACLE", "0"):
        action_grid = [
            (float(w), int(sd), int(su))
            for w in w_grid
            for sd in sweep_values
            for su in sweep_values
        ]
        tail_action = (
            float(best_const["w1"]) if best_const is not None else float(w_grid[0]),
            int(best_const["sd1"]) if best_const is not None else int(sweep_values[0]),
            int(best_const["su1"]) if best_const is not None else int(sweep_values[0]),
        )
        greedy_summary = _greedy_trace_summary(
            trace=trace,
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
            action_grid=action_grid,
            tail_action=tail_action,
        )

    case_oracle_two_phase = None
    if _env_flag("RUN_CASE_ORACLE_TWO_PHASE", "0"):
        case_oracle_two_phase = _case_oracle_two_phase_summary(
            trace=trace,
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
            w_grid=w_grid,
            sweep_values=sweep_values,
            switch_points=switch_points,
        )

    print("Fixed bandit trace step-RL search")
    print(f"grid={grid} cases={len(trace)} bandit_method={bandit_method} tune_dim={tune_dim}")
    print(f"bandit_only_mean={bandit_only_mean:.6f}")
    print(f"best_constant_step_rl={best_const}")
    if best_const is not None:
        print(f"delta_constant_vs_bandit={best_const['mean_runtime'] - bandit_only_mean:+.6f}")
    print(f"best_two_phase_step_rl={best_two_phase}")
    if best_two_phase is not None:
        print(f"delta_two_phase_vs_bandit={best_two_phase['mean_runtime'] - bandit_only_mean:+.6f}")
    if greedy_summary is not None:
        print(f"greedy_step_rl={greedy_summary}")
        print(f"delta_greedy_vs_bandit={greedy_summary['mean_runtime'] - bandit_only_mean:+.6f}")
    if case_oracle_two_phase is not None:
        print(f"case_oracle_two_phase={case_oracle_two_phase}")
        print(f"delta_case_oracle_two_phase_vs_bandit={case_oracle_two_phase['mean_runtime'] - bandit_only_mean:+.6f}")

    model_path_raw = os.environ.get("MODEL_PATH", "").strip()
    vec_path_raw = os.environ.get("VEC_PATH", "").strip()
    if model_path_raw and vec_path_raw:
        runner = _make_policy_runner()
        ppo_summary = _eval_runner(runner, trace)
        print(f"ppo_replay={ppo_summary}")
        print(f"delta_ppo_vs_bandit={ppo_summary['mean_runtime'] - bandit_only_mean:+.6f}")


if __name__ == "__main__":
    main()
