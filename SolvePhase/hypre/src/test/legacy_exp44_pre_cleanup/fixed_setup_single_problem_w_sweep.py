from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple
from itertools import product

from explore_step_rl_dynamic_control import _greedy_case_schedule, _solve_schedule_case
from fixed_setup_top1_wonly_experiment import _baseline_runtimes, _load_fixed_top1_setup, _make_trace


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_vec3(name: str, default: str) -> tuple[float, float, float]:
    raw = _env_str(name, default).strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must have 3 comma-separated values, got {raw!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _env_flag(name: str, default: str = "0") -> bool:
    return _env_str(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _w_grid_from_env() -> tuple[float, ...]:
    raw = _env_str("W_VALUES", "").strip()
    if raw:
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    w_min = _env_float("W_MIN", 0.5)
    w_max = _env_float("W_MAX", 2.0)
    w_step = _env_float("W_STEP", 0.1)
    vals = []
    cur = float(w_min)
    while cur <= float(w_max) + 1e-12:
        vals.append(round(cur, 10))
        cur += float(w_step)
    return tuple(vals)


def main() -> None:
    trace_cache_path = Path(_env_str("TRACE_CACHE_PATH", "/tmp/fixed_setup_top1_schedule_50.pkl"))
    warmup_cases = _env_int("WARMUP_CASES", 300)
    grid_n = _env_int("GRID_N_FIXED", 50)
    grid = (grid_n, grid_n, grid_n)
    case_seed = _env_int("CASE_SEED", 39395939)
    case_index = _env_int("CASE_INDEX", 0)
    c_min = _env_float("C_MIN", 1.0)
    c_max = _env_float("C_MAX", 1000.0)
    difconv_a = _env_vec3("DIFCONV_A", "0,0,0")
    solver_tol = _env_float("SOLVER_TOL", 1e-6)
    solver_max_iter = _env_int("SOLVER_MAX_ITER", 50)
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    solve_max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    switch_max = _env_int("SWITCH_MAX", 10)
    prefix_steps = _env_int("PREFIX_STEPS", 0)
    prefix_default_w = _env_float("PREFIX_DEFAULT_W", 1.0)
    run_constant = _env_flag("RUN_CONSTANT", "1")
    run_two_phase = _env_flag("RUN_TWO_PHASE", "1")
    run_greedy = _env_flag("RUN_GREEDY", "1")
    run_exact_prefix = _env_flag("RUN_EXACT_PREFIX", "1")

    fixed_params, top_count = _load_fixed_top1_setup(trace_cache_path, warmup=int(warmup_cases))
    trace = _make_trace(
        T=max(1, int(case_index) + 1),
        seed=int(case_seed),
        grid=grid,
        fixed_params=fixed_params,
        c_min=float(c_min),
        c_max=float(c_max),
        difconv_a=difconv_a,
    )
    mkw, params = trace[int(case_index)]
    bandit_only_mean = float(
        _baseline_runtimes(
            [(dict(mkw), dict(params))],
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
        )[0]
    )
    print(
        json.dumps(
            {
                "stage": "baseline_done",
                "grid": [int(x) for x in grid],
                "trace_cache_path": str(trace_cache_path),
                "fixed_setup_top1_count_in_mature_suffix": int(top_count),
                "fixed_setup_params": fixed_params,
                "case_index": int(case_index),
                "bandit_only_runtime": float(bandit_only_mean),
            },
            indent=2,
        ),
        flush=True,
    )

    w_grid = _w_grid_from_env()
    action_grid = [(float(w), 1, 1) for w in w_grid]
    switch_points = tuple(range(1, min(int(solve_max_cycles), int(switch_max)) + 1))

    best_constant: Dict[str, Any] | None = None
    if run_constant:
        for w in w_grid:
            out = _solve_schedule_case(
                params=dict(params),
                mkw=dict(mkw),
                schedule=[(int(solve_max_cycles), float(w), 1, 1)],
                solve_tol=float(solve_tol),
                solve_max_cycles=int(solve_max_cycles),
            )
            cand = {
                "runtime": float(out["runtime"]),
                "schedule": [(int(solve_max_cycles), float(w), 1, 1)],
                "failed": bool(out["failed"]),
                "iterations": int(out["iterations"]),
                "delta_vs_bandit_only": float(out["runtime"] - bandit_only_mean),
            }
            if best_constant is None or float(cand["runtime"]) < float(best_constant["runtime"]):
                best_constant = cand

    best_two_phase: Dict[str, Any] | None = None
    if run_two_phase:
        for k in switch_points:
            for w1 in w_grid:
                for w2 in w_grid:
                    out = _solve_schedule_case(
                        params=dict(params),
                        mkw=dict(mkw),
                        schedule=[
                            (int(k), float(w1), 1, 1),
                            (int(solve_max_cycles), float(w2), 1, 1),
                        ],
                        solve_tol=float(solve_tol),
                        solve_max_cycles=int(solve_max_cycles),
                    )
                    cand = {
                        "runtime": float(out["runtime"]),
                        "schedule": [
                            (int(k), float(w1), 1, 1),
                            (int(solve_max_cycles), float(w2), 1, 1),
                        ],
                        "failed": bool(out["failed"]),
                        "iterations": int(out["iterations"]),
                        "delta_vs_bandit_only": float(out["runtime"] - bandit_only_mean),
                    }
                    if best_two_phase is None or float(cand["runtime"]) < float(best_two_phase["runtime"]):
                        best_two_phase = cand

    greedy = None
    if run_greedy:
        greedy_schedule, greedy_runtime = _greedy_case_schedule(
            params=dict(params),
            mkw=dict(mkw),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
            action_grid=action_grid,
            tail_action=(float(best_constant["schedule"][0][1]), 1, 1) if best_constant is not None else (1.0, 1, 1),
        )
        greedy = {
            "runtime": float(greedy_runtime),
            "schedule_len": int(len(greedy_schedule)),
            "schedule_head": list(greedy_schedule[: min(12, len(greedy_schedule))]),
            "schedule_tail": list(greedy_schedule[-min(5, len(greedy_schedule)) :]),
            "delta_vs_bandit_only": float(greedy_runtime - bandit_only_mean),
        }

    exact_prefix: Dict[str, Any] | None = None
    exact_prefix_successful: Dict[str, Any] | None = None
    if run_exact_prefix and int(prefix_steps) > 0:
        prefix_steps_eff = min(int(prefix_steps), int(solve_max_cycles))
        total = len(w_grid) ** prefix_steps_eff
        started = time.time()
        best_runtime = None
        best_schedule = None
        best_iterations = None
        best_failed = None
        best_successful_runtime = None
        best_successful_schedule = None
        best_successful_iterations = None
        for idx, ws in enumerate(product(w_grid, repeat=prefix_steps_eff), start=1):
            schedule = [
                (int(i + 1), float(w), 1, 1)
                for i, w in enumerate(ws)
            ]
            schedule.append((int(solve_max_cycles), float(prefix_default_w), 1, 1))
            out = _solve_schedule_case(
                params=dict(params),
                mkw=dict(mkw),
                schedule=schedule,
                solve_tol=float(solve_tol),
                solve_max_cycles=int(solve_max_cycles),
            )
            runtime = float(out["runtime"])
            if best_runtime is None or runtime < float(best_runtime):
                best_runtime = runtime
                best_schedule = list(schedule)
                best_iterations = int(out["iterations"])
                best_failed = bool(out["failed"])
            if not bool(out["failed"]) and (best_successful_runtime is None or runtime < float(best_successful_runtime)):
                best_successful_runtime = runtime
                best_successful_schedule = list(schedule)
                best_successful_iterations = int(out["iterations"])
            if idx == total or idx % max(200, min(5000, total // 10 if total > 10 else total)) == 0:
                print(
                    json.dumps(
                        {
                            "stage": "exact_prefix_progress",
                            "done": int(idx),
                            "total": int(total),
                            "elapsed_sec": float(time.time() - started),
                            "current_best_runtime": float(best_runtime),
                            "current_best_delta": float(best_runtime - bandit_only_mean),
                            "current_best_successful_runtime": (None if best_successful_runtime is None else float(best_successful_runtime)),
                            "current_best_successful_delta": (None if best_successful_runtime is None else float(best_successful_runtime - bandit_only_mean)),
                        },
                        indent=2,
                    ),
                    flush=True,
                )
        exact_prefix = {
            "runtime": float(best_runtime),
            "schedule_len": int(len(best_schedule)),
            "schedule_head": list(best_schedule[: min(12, len(best_schedule))]),
            "schedule_tail": list(best_schedule[-min(5, len(best_schedule)) :]),
            "failed": bool(best_failed),
            "iterations": int(best_iterations),
            "delta_vs_bandit_only": float(best_runtime - bandit_only_mean),
            "prefix_steps": int(prefix_steps_eff),
            "tail_default_w": float(prefix_default_w),
            "search_size": int(total),
        }
        if best_successful_runtime is not None:
            exact_prefix_successful = {
                "runtime": float(best_successful_runtime),
                "schedule_len": int(len(best_successful_schedule)),
                "schedule_head": list(best_successful_schedule[: min(12, len(best_successful_schedule))]),
                "schedule_tail": list(best_successful_schedule[-min(5, len(best_successful_schedule)) :]),
                "failed": False,
                "iterations": int(best_successful_iterations),
                "delta_vs_bandit_only": float(best_successful_runtime - bandit_only_mean),
                "prefix_steps": int(prefix_steps_eff),
                "tail_default_w": float(prefix_default_w),
                "search_size": int(total),
            }

    print(
        json.dumps(
            {
                "stage": "final",
                "grid": [int(x) for x in grid],
                "trace_cache_path": str(trace_cache_path),
                "case_index": int(case_index),
                "bandit_only_runtime": float(bandit_only_mean),
                "best_constant": best_constant,
                "best_two_phase": best_two_phase,
                "greedy_per_cycle": greedy,
                "exact_prefix": exact_prefix,
                "exact_prefix_successful": exact_prefix_successful,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
