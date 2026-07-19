from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from explore_bandit_solve_control import _augment_params
from explore_step_rl_dynamic_control import _solve_schedule_case
from setup_aware_compare_common import DEFAULT_SETUP_PARAMS, generate_difconv_instances, solve_no_rl_case


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def _float_tuple(raw: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _int_tuple(raw: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    runtimes = [float(r["runtime"]) for r in results]
    setup = [float(r["setup_runtime"]) for r in results]
    solve = [float(r["solve_runtime"]) for r in results]
    failures = [bool(r.get("failed", False)) for r in results]
    iterations = [int(r.get("iterations", -1)) for r in results if int(r.get("iterations", -1)) >= 0]
    return {
        "cases": int(len(results)),
        "mean_runtime": float(mean(runtimes)),
        "total_runtime": float(sum(runtimes)),
        "mean_setup_runtime": float(mean(setup)),
        "mean_solve_runtime": float(mean(solve)),
        "failed_count": int(sum(failures)),
        "mean_iterations": float(mean(iterations)) if iterations else float("nan"),
    }


def _schedule_key(k: int, w0: float, w1: float) -> str:
    return f"k{k}_w{w0:.2f}_then_{w1:.2f}"


def main() -> None:
    state_path = Path(_env_str("BANDIT_STATE_PATH", "/tmp/mature40_tune7_bandit_state.pkl"))
    print(json.dumps({"stage": "bandit_state_load_start", "path": str(state_path)}), flush=True)
    with state_path.open("rb") as fh:
        branch = pickle.load(fh)["branch"]
    print(json.dumps({"stage": "bandit_state_load_done", "path": str(state_path)}), flush=True)

    grid_n = _env_int("GRID_N", 40)
    eval_cases = _env_int("EVAL_CASES", 200)
    eval_seed = _env_int("EVAL_SEED", 39394939)
    solve_tol = _env_float("SOLVE_TOL", 1e-6)
    solve_max_cycles = _env_int("SOLVE_MAX_CYCLES", 50)
    solver_tol = _env_float("SOLVER_TOL", 1e-6)
    solver_max_iter = _env_int("SOLVER_MAX_ITER", 50)
    ks = _int_tuple(_env_str("SWITCH_K_VALUES", "1,2,3,4,5,6"))
    w_first_values = _float_tuple(_env_str("W_FIRST_VALUES", "1.6,1.7,1.8,1.9"))
    w_tail_values = _float_tuple(_env_str("W_TAIL_VALUES", "1.4,1.5,1.55,1.6"))
    schedules = [
        (k, w0, w1, [(int(k), float(w0), 1, 1), (solve_max_cycles, float(w1), 1, 1)])
        for k in ks
        for w0 in w_first_values
        for w1 in w_tail_values
    ]

    difconv_a = tuple(float(x) for x in _env_str("DIFCONV_A", "0,0,0").split(","))
    instances = generate_difconv_instances(
        T=eval_cases,
        seed=eval_seed,
        grid_choices=[(grid_n, grid_n, grid_n)],
        c_min=_env_float("C_MIN", 1.0),
        c_max=_env_float("C_MAX", 1000.0),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )

    default_results: List[Dict[str, Any]] = []
    bandit_results: List[Dict[str, Any]] = []
    schedule_results: Dict[str, List[Dict[str, Any]]] = {
        _schedule_key(k, w0, w1): [] for k, w0, w1, _schedule in schedules
    }
    oracle_best = []
    rng = np.random.default_rng(_env_int("EVAL_ORDER_SEED", eval_seed + 9001))

    for i, (mkw, context) in enumerate(instances, 1):
        params, _info = branch.policy.select(np.asarray(context, dtype=float), parameter_space=branch.parameter_space)
        params = dict(params)
        base_bandit = solve_no_rl_case(
            params=dict(params),
            mkw=dict(mkw),
            solver_tol=solver_tol,
            solver_max_iter=solver_max_iter,
            augment_params=_augment_params,
        )
        base_default = solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(mkw),
            solver_tol=solver_tol,
            solver_max_iter=solver_max_iter,
            augment_params=_augment_params,
        )
        default_results.append(base_default)
        bandit_results.append(base_bandit)

        case_sched_out = []
        order = [schedules[j] for j in rng.permutation(len(schedules))]
        for k, w0, w1, schedule in order:
            key = _schedule_key(k, w0, w1)
            out = _solve_schedule_case(
                params=dict(params),
                mkw=dict(mkw),
                schedule=schedule,
                solve_tol=solve_tol,
                solve_max_cycles=solve_max_cycles,
            )
            schedule_results[key].append(out)
            case_sched_out.append((key, out))
        best_key, best_out = min(case_sched_out, key=lambda item: float(item[1]["runtime"]))
        oracle_out = dict(best_out)
        oracle_out["oracle_key"] = best_key
        oracle_best.append(oracle_out)
        if i % max(1, _env_int("PROGRESS_EVERY", 20)) == 0 or i == eval_cases:
            print(json.dumps({"stage": "eval_progress", "done": i, "total": eval_cases}), flush=True)

    s_default = _summarize(default_results)
    s_bandit = _summarize(bandit_results)
    s_oracle = _summarize(oracle_best)
    sched_summaries = {key: _summarize(val) for key, val in schedule_results.items()}
    top_schedules = sorted(
        (
            {
                "schedule": key,
                **summary,
                "vs_bandit_pct": float(100.0 * (s_bandit["mean_runtime"] - summary["mean_runtime"]) / s_bandit["mean_runtime"]),
            }
            for key, summary in sched_summaries.items()
        ),
        key=lambda row: float(row["mean_runtime"]),
    )[:20]
    oracle_hist: Dict[str, int] = {}
    for out in oracle_best:
        key = str(out["oracle_key"])
        oracle_hist[key] = int(oracle_hist.get(key, 0)) + 1
    result = {
        "stage": "final",
        "protocol": {
            "grid": [grid_n, grid_n, grid_n],
            "eval_cases": eval_cases,
            "eval_seed": eval_seed,
            "bandit_updates": False,
        },
        "methods": {
            "default_setup_default_solve": s_default,
            "mature_bandit_default_solve": s_bandit,
            "oracle_best_two_phase_per_case": s_oracle,
        },
        "relative_pct": {
            "bandit_vs_default_setup_pct": float(100.0 * (s_default["mean_runtime"] - s_bandit["mean_runtime"]) / s_default["mean_runtime"]),
            "oracle_best_two_phase_vs_bandit_pct": float(100.0 * (s_bandit["mean_runtime"] - s_oracle["mean_runtime"]) / s_bandit["mean_runtime"]),
        },
        "top_fixed_two_phase_schedules": top_schedules,
        "oracle_schedule_hist_top20": sorted(
            ({"schedule": k, "count": v} for k, v in oracle_hist.items()),
            key=lambda row: int(row["count"]),
            reverse=True,
        )[:20],
    }
    result_path = _env_str("RESULT_PATH", "").strip()
    if result_path:
        Path(result_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"stage": "result_write", "path": result_path}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
