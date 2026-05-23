from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Sequence

import numpy as np

def _repo_root() -> Path:
    env_root = os.environ.get("REPO_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[2]


REPO = _repo_root()
TEST_DIR = REPO / "SolvePhase/hypre/src/test"
sys.path.insert(0, str(TEST_DIR))

from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    augment_setup_params,
    eval_runner,
    fixed_trace,
    solve_schedule_case,
    solve_no_rl_case,
)


OUT_DIR = REPO / "results/mature_tune7_ppo_repro_20260423"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(OUT_DIR / "checkpoint_10000.zip")))
# The retained Exp44 path does not use VecNormalize. The runner only loads
# VecNormalize when the file exists, so a stable nonexistent path is enough.
UNUSED_VEC_PATH = OUT_DIR / ".unused_vecnormalize.pkl"


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def env_str(name: str, default: str) -> str:
    return str(os.environ.get(name, default))


def env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def env_float_tuple(name: str, default: str) -> tuple[float, ...]:
    raw = env_str(name, default).strip()
    if not raw:
        return ()
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def enabled_methods() -> set[str]:
    raw = env_str("FORWARD_METHODS", "default,bandit,fixed,ppo")
    allowed = {"default", "bandit", "fixed", "ppo"}
    methods = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = methods - allowed
    if unknown:
        raise ValueError(f"unknown FORWARD_METHODS values: {sorted(unknown)}")
    if not methods:
        raise ValueError("FORWARD_METHODS must enable at least one method")
    return methods


def result_path_for_seed(seed: int) -> Path:
    return OUT_DIR / f"forward_dual_seed{int(seed)}.json"


def summarize(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    runtimes = [float(row["runtime"]) for row in results]
    setup = [float(row["setup_runtime"]) for row in results]
    solve = [float(row["solve_runtime"]) for row in results]
    failures = [bool(row.get("failed", False)) for row in results]
    iterations = [int(row.get("iterations", -1)) for row in results if int(row.get("iterations", -1)) >= 0]
    return {
        "cases": int(len(results)),
        "mean_runtime": float(mean(runtimes)),
        "total_runtime": float(sum(runtimes)),
        "mean_setup_runtime": float(mean(setup)),
        "mean_solve_runtime": float(mean(solve)),
        "failed_count": int(sum(failures)),
        "mean_iterations": float(mean(iterations)) if iterations else None,
    }


def make_runner() -> SetupAwareSolvePolicyRunner:
    return SetupAwareSolvePolicyRunner(
        SetupAwareRLConfig(
            tune_dim=7,
            tune7_variant="categorical",
            algo=env_str("ALGO", "ppo").strip().lower(),
            model_type=env_str("MODEL_TYPE", "mlp").strip().lower(),
            model_path=MODEL_PATH,
            vec_path=UNUSED_VEC_PATH,
            fixed_grid=(40, 40, 40),
            difconv_c_range=(1.0, 1000.0),
            w_only=True,
            w_center=env_float("W_CENTER", 1.65),
            w_scale=env_float("W_SCALE", 0.1),
            w_global_min=env_float("W_GLOBAL_MIN", 1.0),
            w_global_max=env_float("W_GLOBAL_MAX", 2.0),
            sweeps_min=1,
            sweeps_max=1,
            w_init=None,
            sweeps_init=None,
            solve_max_cycles=50,
            solve_tol=1e-6,
            default_setup_params={},
            obs_mode=env_str("OBS_MODE", "solve_only").strip().lower(),
            action_mode=env_str("ACTION_MODE", "continuous").strip().lower(),
            discrete_w_values=env_float_tuple("DISCRETE_W_VALUES", "1.4,1.6,1.8"),
        )
    )


def pct(base: float, other: float) -> float:
    return 100.0 * (base - other) / base


def ensure_runner_summary(summary: Dict[str, Any], cases: int) -> Dict[str, Any]:
    out = dict(summary)
    out.setdefault("cases", int(cases))
    out.setdefault("total_runtime", float(out["mean_runtime"]) * int(cases))
    return out


def window_result(
    *,
    label: str,
    trace_slice: Sequence[tuple[dict, dict]],
    methods_enabled: set[str],
    default_rows: Sequence[Dict[str, Any]] | None,
    bandit_rows: Sequence[Dict[str, Any]] | None,
    fixed_rows: Sequence[Dict[str, Any]] | None,
    runner: SetupAwareSolvePolicyRunner | None,
) -> Dict[str, Any]:
    method_map: Dict[str, Any] = {}
    rel_map: Dict[str, float] = {}

    default_summary = summarize(default_rows) if default_rows is not None else None
    bandit_summary = summarize(bandit_rows) if bandit_rows is not None else None
    fixed_summary = summarize(fixed_rows) if fixed_rows is not None else None
    ppo_summary = ensure_runner_summary(eval_runner(runner, list(trace_slice)), len(trace_slice)) if runner is not None else None

    if default_summary is not None:
        method_map["default_setup_default_solve"] = default_summary
    if bandit_summary is not None:
        method_map["bandit_only"] = bandit_summary
    if fixed_summary is not None:
        method_map["fixed_w_1.60"] = fixed_summary
    if ppo_summary is not None:
        method_map["ppo_best"] = ppo_summary

    if default_summary is not None and bandit_summary is not None:
        rel_map["bandit_vs_default_setup"] = pct(default_summary["mean_runtime"], bandit_summary["mean_runtime"])
    if default_summary is not None and fixed_summary is not None:
        rel_map["fixed_w_1.60_vs_default_setup"] = pct(default_summary["mean_runtime"], fixed_summary["mean_runtime"])
    if default_summary is not None and ppo_summary is not None:
        rel_map["ppo_best_vs_default_setup"] = pct(default_summary["mean_runtime"], ppo_summary["mean_runtime"])
    if bandit_summary is not None and fixed_summary is not None:
        rel_map["fixed_w_1.60_vs_bandit"] = pct(bandit_summary["mean_runtime"], fixed_summary["mean_runtime"])
    if bandit_summary is not None and ppo_summary is not None:
        rel_map["ppo_best_vs_bandit"] = pct(bandit_summary["mean_runtime"], ppo_summary["mean_runtime"])
    if fixed_summary is not None and ppo_summary is not None:
        rel_map["ppo_best_vs_fixed_w_1.60"] = pct(fixed_summary["mean_runtime"], ppo_summary["mean_runtime"])

    return {
        "label": label,
        "enabled_methods": sorted(methods_enabled),
        "methods": method_map,
        "relative_pct": rel_map,
    }


def main() -> None:
    # Held-out forward evaluation flow:
    # 1) regenerate one archived same-trace continuation from FORWARD_SEED
    # 2) slice the requested eval windows from that trace
    # 3) run default/bandit/fixed baselines on the sliced cases
    # 4) run the PPO solve policy on the same sliced cases
    # 5) write one per-seed JSON summary used later by the aggregator
    seed = env_int("FORWARD_SEED", 39393939)
    trace_t = env_int("TRACE_T", 2500)
    train_segment_start = env_int("TRAIN_SEGMENT_START", 1000)
    train_segment_end = env_int("TRAIN_SEGMENT_END", 1500)
    eval_a_name = env_str("EVAL_A_NAME", "eval_500").strip()
    eval_a_start = env_int("EVAL_A_START", 1500)
    eval_a_end = env_int("EVAL_A_END", 2000)
    eval_b_name = env_str("EVAL_B_NAME", "eval_1000").strip()
    eval_b_start = env_int("EVAL_B_START", 1500)
    eval_b_end = env_int("EVAL_B_END", 2500)
    methods_enabled = enabled_methods()
    result_path = Path(os.environ.get("RESULT_PATH", str(result_path_for_seed(seed))))
    print(json.dumps({"stage": "trace_build_start", "seed": seed, "T": trace_t}), flush=True)
    trace = list(
        fixed_trace(
            T=trace_t,
            grid=(40, 40, 40),
            seed=seed,
            tune_dim=7,
            bandit_method="linucbv4",
            solve_mode="no_rl",
        )
    )
    trace_a = [(dict(mkw), dict(params)) for mkw, params in trace[eval_a_start:eval_a_end]]
    trace_b = [(dict(mkw), dict(params)) for mkw, params in trace[eval_b_start:eval_b_end]]
    print(
        json.dumps(
            {
                "stage": "trace_build_done",
                eval_a_name: len(trace_a),
                eval_b_name: len(trace_b),
            }
        ),
        flush=True,
    )

    fixed_w = 1.60
    default_rows_1000 = [] if "default" in methods_enabled else None
    bandit_rows_1000 = [] if "bandit" in methods_enabled else None
    fixed_rows_1000 = [] if "fixed" in methods_enabled else None
    max_eval_end = max(eval_a_end, eval_b_end)
    eval_trace = [(dict(mkw), dict(params)) for mkw, params in trace[eval_a_start:max_eval_end]]
    for idx, (mkw, params) in enumerate(eval_trace, 1):
        if default_rows_1000 is not None:
            default_rows_1000.append(
                solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=1e-6,
                    solver_max_iter=50,
                    augment_params=augment_setup_params,
                )
            )
        if bandit_rows_1000 is not None:
            bandit_rows_1000.append(
                solve_no_rl_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    solver_tol=1e-6,
                    solver_max_iter=50,
                    augment_params=augment_setup_params,
                )
            )
        if fixed_rows_1000 is not None:
            fixed_rows_1000.append(
                solve_schedule_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    schedule=[(50, fixed_w, 1, 1)],
                    solve_tol=1e-6,
                    solve_max_cycles=50,
                )
            )
        if idx % 100 == 0 or idx == len(eval_trace):
            print(json.dumps({"stage": "baseline_progress", "done": idx, "total": len(eval_trace)}), flush=True)

    runner = make_runner() if "ppo" in methods_enabled else None
    result = {
        "stage": "final",
        "protocol": {
            "seed": seed,
            "trace_T": trace_t,
            "train_segment_used_for_existing_model": [train_segment_start, train_segment_end],
            "eval_windows": {
                eval_a_name: [eval_a_start, eval_a_end],
                eval_b_name: [eval_b_start, eval_b_end],
            },
            "model_path": str(MODEL_PATH),
            "fixed_w": fixed_w,
            "enabled_methods": sorted(methods_enabled),
            "note": "same online bandit trace continuation; one trace build, two eval windows",
        },
        "windows": {
            eval_a_name: window_result(
                label=eval_a_name,
                trace_slice=trace_a,
                methods_enabled=methods_enabled,
                default_rows=default_rows_1000[: len(trace_a)] if default_rows_1000 is not None else None,
                bandit_rows=bandit_rows_1000[: len(trace_a)] if bandit_rows_1000 is not None else None,
                fixed_rows=fixed_rows_1000[: len(trace_a)] if fixed_rows_1000 is not None else None,
                runner=runner,
            ),
            eval_b_name: window_result(
                label=eval_b_name,
                trace_slice=trace_b,
                methods_enabled=methods_enabled,
                default_rows=default_rows_1000[: len(trace_b)] if default_rows_1000 is not None else None,
                bandit_rows=bandit_rows_1000[: len(trace_b)] if bandit_rows_1000 is not None else None,
                fixed_rows=fixed_rows_1000[: len(trace_b)] if fixed_rows_1000 is not None else None,
                runner=runner,
            ),
        },
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "result_write", "path": str(result_path)}), flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
