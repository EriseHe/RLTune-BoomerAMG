from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from explore_step_rl_dynamic_control import _fixed_trace, _solve_schedule_case
from fixed_setup_top1_wonly_experiment import (
    _baseline_runtimes,
    _load_fixed_top1_setup,
    _make_trace,
)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _env_float_list(name: str, default: str) -> tuple[float, ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _env_int_list(name: str, default: str) -> tuple[int, ...]:
    raw = _env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def _env_vec3(name: str, default: str) -> tuple[float, float, float]:
    raw = _env_str(name, default).strip()
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"{name} must have 3 comma-separated values, got: {raw!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _ensure_trace_cache(
    *,
    cache_path: Path,
    trace_cases: int,
    warmup_cases: int,
    seed: int,
    grid: tuple[int, int, int],
    tune_dim: int,
    bandit_method: str,
) -> None:
    if cache_path.exists():
        return
    trace = _fixed_trace(
        T=int(trace_cases),
        grid=tuple(int(x) for x in grid),
        seed=int(seed),
        tune_dim=int(tune_dim),
        bandit_method=str(bandit_method),
        solve_mode="no_rl",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as fh:
        pickle.dump(trace, fh)
    print(
        json.dumps(
            {
                "stage": "trace_cache_built",
                "trace_cache_path": str(cache_path),
                "trace_cases": int(trace_cases),
                "warmup_cases": int(warmup_cases),
                "grid": [int(x) for x in grid],
            },
            indent=2,
        ),
        flush=True,
    )


def _schedule_summary(
    *,
    trace: Sequence[tuple[dict, dict]],
    schedule: Sequence[Tuple[int, float, int, int]],
    solve_tol: float,
    solve_max_cycles: int,
) -> Dict[str, Any]:
    rows = [
        _solve_schedule_case(
            params=dict(params),
            mkw=dict(mkw),
            schedule=list(schedule),
            solve_tol=float(solve_tol),
            solve_max_cycles=int(solve_max_cycles),
        )
        for (mkw, params) in trace
    ]
    return {
        "mean_runtime": float(mean(float(row["runtime"]) for row in rows)),
        "mean_setup_runtime": float(mean(float(row["setup_runtime"]) for row in rows)),
        "mean_solve_runtime": float(mean(float(row["solve_runtime"]) for row in rows)),
        "failed_count": int(sum(int(bool(row["failed"])) for row in rows)),
        "mean_iterations": float(mean(int(row["iterations"]) for row in rows)),
        "mean_final_w": float(
            mean(float(row["final_w"]) for row in rows if np.isfinite(float(row["final_w"])))
        ),
    }


def main() -> None:
    seed = _env_int("SEED", 39393939)
    grid_n = _env_int("GRID_N_FIXED", 50)
    grid = (int(grid_n), int(grid_n), int(grid_n))
    tune_dim = _env_int("SETUP_TUNE_DIM", 5)
    bandit_method = _env_str("SETUP_BANDIT_METHOD", "linucbv3").strip().lower()
    trace_cases = _env_int("TRACE_CASES", 400)
    warmup_cases = _env_int("WARMUP_CASES", 300)
    eval_cases = _env_int("EVAL_CASES", 200)
    eval_seed = _env_int("EVAL_SEED", seed + 2000)
    c_min = _env_float("C_MIN", 1.0)
    c_max = _env_float("C_MAX", 1000.0)
    difconv_a = _env_vec3("DIFCONV_A", "0,0,0")
    solver_tol = _env_float("SOLVER_TOL", 1e-6)
    solver_max_iter = _env_int("SOLVER_MAX_ITER", 50)
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    solve_max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    trace_cache_path = Path(_env_str("TRACE_CACHE_PATH", f"/tmp/fixed_setup_top1_schedule_{grid_n}.pkl"))

    w_values = _env_float_list("SCHEDULE_W_VALUES", "1.3,1.4,1.5,1.6,1.7")
    sweeps_down_values = _env_int_list("SCHEDULE_SWEEPS_DOWN", "1")
    sweeps_up_values = _env_int_list("SCHEDULE_SWEEPS_UP", "1,2")
    switch_points = _env_int_list("SCHEDULE_SWITCH_POINTS", "2,4,6,8")

    _ensure_trace_cache(
        cache_path=trace_cache_path,
        trace_cases=int(trace_cases),
        warmup_cases=int(warmup_cases),
        seed=int(seed),
        grid=grid,
        tune_dim=int(tune_dim),
        bandit_method=str(bandit_method),
    )

    fixed_params, top_count = _load_fixed_top1_setup(trace_cache_path, warmup=int(warmup_cases))
    eval_trace = _make_trace(
        T=int(eval_cases),
        seed=int(eval_seed),
        grid=grid,
        fixed_params=fixed_params,
        c_min=float(c_min),
        c_max=float(c_max),
        difconv_a=difconv_a,
    )
    baseline_runtimes = _baseline_runtimes(
        list(eval_trace),
        solver_tol=float(solver_tol),
        solver_max_iter=int(solver_max_iter),
    )
    bandit_only_mean = float(mean(baseline_runtimes))
    print(
        json.dumps(
            {
                "stage": "baseline_done",
                "trace_cache_path": str(trace_cache_path),
                "grid": [int(x) for x in grid],
                "eval_cases": int(eval_cases),
                "fixed_setup_top1_count_in_mature_suffix": int(top_count),
                "fixed_setup_params": fixed_params,
                "bandit_only_mean": float(bandit_only_mean),
            },
            indent=2,
        ),
        flush=True,
    )

    best_constant: Dict[str, Any] | None = None
    constant_total = len(w_values) * len(sweeps_down_values) * len(sweeps_up_values)
    constant_done = 0
    sweep_start = time.time()
    for w in w_values:
        for sd in sweeps_down_values:
            for su in sweeps_up_values:
                schedule = [(int(solve_max_cycles), float(w), int(sd), int(su))]
                summary = _schedule_summary(
                    trace=eval_trace,
                    schedule=schedule,
                    solve_tol=float(solve_tol),
                    solve_max_cycles=int(solve_max_cycles),
                )
                summary["schedule"] = schedule
                summary["delta_vs_bandit_only"] = float(summary["mean_runtime"] - bandit_only_mean)
                if best_constant is None or float(summary["mean_runtime"]) < float(best_constant["mean_runtime"]):
                    best_constant = summary
                constant_done += 1
                if constant_done == constant_total or constant_done % 5 == 0:
                    print(
                        json.dumps(
                            {
                                "stage": "constant_progress",
                                "done": int(constant_done),
                                "total": int(constant_total),
                                "elapsed_sec": float(time.time() - sweep_start),
                                "current_best_mean": float(best_constant["mean_runtime"]),
                                "current_best_delta": float(best_constant["delta_vs_bandit_only"]),
                            },
                            indent=2,
                        ),
                        flush=True,
                    )

    best_two_phase: Dict[str, Any] | None = None
    two_phase_total = (
        len(switch_points)
        * len(w_values)
        * len(sweeps_down_values)
        * len(sweeps_up_values)
        * len(w_values)
        * len(sweeps_down_values)
        * len(sweeps_up_values)
    )
    two_phase_done = 0
    for switch in switch_points:
        for w1 in w_values:
            for sd1 in sweeps_down_values:
                for su1 in sweeps_up_values:
                    for w2 in w_values:
                        for sd2 in sweeps_down_values:
                            for su2 in sweeps_up_values:
                                schedule = [
                                    (int(switch), float(w1), int(sd1), int(su1)),
                                    (int(solve_max_cycles), float(w2), int(sd2), int(su2)),
                                ]
                                summary = _schedule_summary(
                                    trace=eval_trace,
                                    schedule=schedule,
                                    solve_tol=float(solve_tol),
                                    solve_max_cycles=int(solve_max_cycles),
                                )
                                summary["schedule"] = schedule
                                summary["delta_vs_bandit_only"] = float(summary["mean_runtime"] - bandit_only_mean)
                                if best_two_phase is None or float(summary["mean_runtime"]) < float(best_two_phase["mean_runtime"]):
                                    best_two_phase = summary
                                two_phase_done += 1
                                if two_phase_done == two_phase_total or two_phase_done % 20 == 0:
                                    print(
                                        json.dumps(
                                            {
                                                "stage": "two_phase_progress",
                                                "done": int(two_phase_done),
                                                "total": int(two_phase_total),
                                                "elapsed_sec": float(time.time() - sweep_start),
                                                "current_best_mean": float(best_two_phase["mean_runtime"]),
                                                "current_best_delta": float(best_two_phase["delta_vs_bandit_only"]),
                                            },
                                            indent=2,
                                        ),
                                        flush=True,
                                    )

    print(
        json.dumps(
            {
                "stage": "final",
                "trace_cache_path": str(trace_cache_path),
                "grid": [int(x) for x in grid],
                "eval_cases": int(eval_cases),
                "fixed_setup_top1_count_in_mature_suffix": int(top_count),
                "fixed_setup_params": fixed_params,
                "bandit_only_mean": float(bandit_only_mean),
                "best_constant": best_constant,
                "best_two_phase": best_two_phase,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
