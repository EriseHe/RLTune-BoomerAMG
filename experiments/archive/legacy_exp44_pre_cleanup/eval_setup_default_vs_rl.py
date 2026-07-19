from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from eval_default_vs_rl import (
    _env_flag,
    _env_optional_float,
    _env_optional_int,
    _parse_grid_range,
    _parse_grid_sizes,
    _parse_range_env,
    _parse_triplet_env,
    _summarize,
    _summarize_failures,
    _summarize_infer,
    _summarize_overhead,
    _write_failure_log,
)
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    RandomPolicy,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    build_action_space_bundle,
    build_single_branch,
    default_test_final_bandit_config_from_env,
    family_seed_map,
    generate_difconv_instances,
    run_interleaved_branch_scenario,
)


def _augment_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    relax_type = os.environ.get("SETUP_RELAX_TYPE", "").strip()
    if relax_type:
        out["relax_type"] = int(relax_type)
    num_sweeps = os.environ.get("SETUP_NUM_SWEEPS", "").strip()
    if num_sweeps:
        out["num_sweeps"] = int(num_sweeps)
    cycle_type = os.environ.get("SETUP_CYCLE_TYPE", "").strip()
    if cycle_type:
        out["cycle_type"] = int(cycle_type)
    max_levels = os.environ.get("SETUP_MAX_LEVELS", "").strip()
    if max_levels:
        out["max_levels"] = int(max_levels)
    return out


def _make_solve_policy(
    *,
    tune_dim: int,
    tune7_variant: str,
    model_type: str,
    model_path: Path,
    vec_path: Path,
    fixed_grid: Tuple[int, int, int],
    difconv_c_range: Tuple[float, float],
    w_center: float,
    w_scale: float,
    sweeps_min: int,
    sweeps_max: int,
    w_only: bool,
    w_init,
    sweeps_init,
    solve_tol: float,
    solve_max_cycles: int,
) -> SetupAwareSolvePolicyRunner:
    cfg = SetupAwareRLConfig(
        tune_dim=int(tune_dim),
        tune7_variant=str(tune7_variant),
        model_type=str(model_type),
        model_path=Path(model_path),
        vec_path=Path(vec_path),
        fixed_grid=tuple(int(x) for x in fixed_grid),
        difconv_c_range=(float(difconv_c_range[0]), float(difconv_c_range[1])),
        w_only=bool(w_only),
        w_center=float(w_center),
        w_scale=float(w_scale),
        sweeps_min=int(sweeps_min),
        sweeps_max=int(sweeps_max),
        w_init=w_init,
        sweeps_init=sweeps_init,
        solve_max_cycles=int(solve_max_cycles),
        solve_tol=float(solve_tol),
        default_setup_params=dict(DEFAULT_SETUP_PARAMS),
    )
    return SetupAwareSolvePolicyRunner(cfg)


def _make_random_branch(*, label: str, tune_dim: int, tune7_variant: str, seed: int, solver_tol: float, solver_max_iter: int, bundle):
    if int(tune_dim) == 3:
        parameter_space = bundle.parameter_space_tune3
        tune_set = "tune3"
    elif int(tune_dim) == 5:
        parameter_space = bundle.parameter_space_tune5
        tune_set = "tune5"
    elif int(tune_dim) == 7:
        parameter_space = bundle.parameter_space_tune7
        tune_set = "tune7"
    else:
        raise ValueError(f"Unsupported tune_dim: {tune_dim}")
    if parameter_space is None:
        raise ValueError(f"No parameter space available for tune_dim={tune_dim}")
    from setup_aware_compare_common import BranchRun

    return BranchRun(
        label=str(label),
        family="random",
        tune_set=str(tune_set),
        seed=int(seed),
        policy=RandomPolicy(parameter_space["actions"], seed=int(seed)),
        parameter_space=parameter_space,
        solver_tol=float(solver_tol),
        solver_max_iter=int(solver_max_iter),
    )


def _results_from_metrics(*, metrics: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i in range(len(metrics["runtime_sec"][label])):
        time_with_setup = float(metrics["runtime_sec"][label][i])
        overhead = float(metrics["overhead_sec"][label][i])
        out.append(
            {
                "time": float(metrics["solve_runtime_sec"][label][i]),
                "setup_time": float(metrics["setup_runtime_sec"][label][i]),
                "time_with_setup": float(time_with_setup),
                "cycles": int(metrics["iterations"][label][i]),
                "final_residual_norm": float(metrics["residual_norm"][label][i]),
                "failed": bool(metrics["failed_flags"][label][i]),
                "failure_reason": str(metrics["failure_reason"][label][i]),
                "terminated": not bool(metrics["failed_flags"][label][i]),
                "truncated": bool(metrics["failure_reason"][label][i]) and "max_" in str(metrics["failure_reason"][label][i]),
                "infer_time": float(metrics["infer_runtime_sec"][label][i]),
                "preprocess_time": None,
                "wall_time": float(time_with_setup + overhead),
                "overhead": float(overhead),
                "info": {},
            }
        )
    return out


def _failure_records_from_results(*, mode: str, results: Sequence[Dict[str, Any]], instances: Sequence[Tuple[Dict[str, Any], np.ndarray]]) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, (result, (mkw, _context)) in enumerate(zip(results, instances)):
        if not bool(result.get("failed", False)):
            continue
        records.append(
            {
                "scope": "main_eval",
                "mode": str(mode),
                "case": {
                    "seed_index": int(idx),
                    "grid": [int(mkw["nx"]), int(mkw["ny"]), int(mkw["nz"])],
                    "k": float(mkw["k"]),
                    "c": float(mkw["c"]),
                    "a0": float(mkw["a0"]),
                    "rhs_seed": int(mkw["rhs_seed"]),
                },
                "failure_reason": str(result.get("failure_reason", "")),
                "cycles": int(result.get("cycles", 0)),
                "final_residual_norm": float(result.get("final_residual_norm", np.nan)),
                "setup_time": float(result.get("setup_time", np.nan)),
                "solve_time": float(result.get("time", np.nan)),
                "time_with_setup": float(result.get("time_with_setup", np.nan)),
                "wall_time": float(result.get("wall_time", np.nan)),
                "overhead": float(result.get("overhead", np.nan)),
                "terminated": bool(result.get("terminated", False)),
                "truncated": bool(result.get("truncated", False)),
            }
        )
    return records


def main():
    seed_start = int(os.environ.get("SEED", "39393939"))
    setup_mode = os.environ.get("SETUP_MODE", "random").strip().lower()
    tune_dim = int(os.environ.get("SETUP_TUNE_DIM", "5"))
    tune7_variant = os.environ.get("TUNE7_VARIANT", "categorical").strip().lower()
    setup_bandit_method = os.environ.get("SETUP_BANDIT_METHOD", "").strip().lower()
    if not setup_bandit_method:
        setup_bandit_method = "linucbv4" if int(tune_dim) == 7 else "linucbv3"
    bandit_update_enabled = _env_flag("SETUP_BANDIT_UPDATE", "1" if setup_mode == "bandit" else "0")

    grid_sizes = [(50, 50, 50)]
    grid_sizes_env = _parse_grid_sizes(os.environ.get("GRID_SIZES", ""))
    grid_range_env = _parse_grid_range(os.environ.get("GRID_RANGE", ""))
    if grid_sizes_env:
        grid_sizes = grid_sizes_env
    elif grid_range_env:
        grid_sizes = grid_range_env
    randomize_grid = _env_flag("RANDOMIZE_GRID", "0")
    if randomize_grid:
        raise ValueError("eval_setup_default_vs_rl.py now requires explicit GRID_SIZES/GRID_RANGE so both methods share the same case list")

    a_instances = int(os.environ.get("EVAL_A_INSTANCES", "20"))
    difconv_c_range = _parse_range_env("DIFCONV_C_RANGE", "1,1000")
    w_center = float(os.environ.get("W_CENTER", "1.25"))
    w_scale = float(os.environ.get("W_SCALE", "0.75"))
    sweeps_min = int(os.environ.get("SWEEPS_MIN", "1"))
    sweeps_max = int(os.environ.get("SWEEPS_MAX", "1"))
    w_only = _env_flag("W_ONLY", "0")
    w_init = _env_optional_float("W_INIT")
    sweeps_init = _env_optional_int("SWEEPS_INIT")
    solve_tol = float(os.environ.get("SOLVE_TOL", "1e-6"))
    solve_max_cycles = int(os.environ.get("SOLVE_MAX_CYCLES", "50"))
    solver_tol = float(os.environ.get("SOLVER_TOL", os.environ.get("SOLVE_TOL", "1e-6")))
    solver_max_iter = int(os.environ.get("SOLVER_MAX_ITER", os.environ.get("SOLVE_MAX_CYCLES", "50")))
    baseline_solve_mode = os.environ.get("BASELINE_SOLVE_MODE", "no_rl").strip().lower()
    if baseline_solve_mode != "no_rl":
        raise ValueError("eval_setup_default_vs_rl.py now uses BASELINE_SOLVE_MODE=no_rl only")

    model_type = os.environ.get("MODEL_TYPE", "mlp").strip().lower()
    model_path = Path(os.environ["MODEL_PATH"])
    vec_path = Path(os.environ["VEC_PATH"])

    bundle = build_action_space_bundle(final_tune_dims=(int(tune_dim),), tune7_variant=tune7_variant)
    difconv_a = tuple(float(x) for x in os.environ.get("DIFCONV_A", "0,0,0").split(","))
    instances = generate_difconv_instances(
        T=int(a_instances),
        seed=int(seed_start),
        grid_choices=[tuple(int(x) for x in grid) for grid in grid_sizes],
        c_min=float(difconv_c_range[0]),
        c_max=float(difconv_c_range[1]),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )

    solve_policy = _make_solve_policy(
        tune_dim=int(tune_dim),
        tune7_variant=tune7_variant,
        model_type=model_type,
        model_path=model_path,
        vec_path=vec_path,
        fixed_grid=tuple(int(x) for x in grid_sizes[0]),
        difconv_c_range=(float(difconv_c_range[0]), float(difconv_c_range[1])),
        w_center=float(w_center),
        w_scale=float(w_scale),
        sweeps_min=int(sweeps_min),
        sweeps_max=int(sweeps_max),
        w_only=bool(w_only),
        w_init=w_init,
        sweeps_init=sweeps_init,
        solve_tol=float(solve_tol),
        solve_max_cycles=int(solve_max_cycles),
    )

    bandit_cfg = default_test_final_bandit_config_from_env()
    seed_map = family_seed_map(seed=int(seed_start))

    if setup_mode == "bandit":
        base_branch = build_single_branch(
            method=setup_bandit_method,
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            seed=int(seed_start),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
            bandit_cfg=bandit_cfg,
            bundle=bundle,
        )
        rl_branch = build_single_branch(
            method=setup_bandit_method,
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            seed=int(seed_start),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
            bandit_cfg=bandit_cfg,
            bundle=bundle,
        )
        setup_policy_summary = {
            "method": setup_bandit_method,
            "tune_dim": int(tune_dim),
            "seed": int(base_branch.seed or seed_map.get(base_branch.family, seed_start)),
            "bandit_update": bool(bandit_update_enabled),
            "loss": "test_final_shared runtime loss",
        }
        if not bandit_update_enabled:
            base_branch.policy.update = lambda *args, **kwargs: None
            rl_branch.policy.update = lambda *args, **kwargs: None
    elif setup_mode == "random":
        random_seed = int(seed_start + 17003)
        base_branch = _make_random_branch(
            label="Random setup | baseline",
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            seed=int(random_seed),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
            bundle=bundle,
        )
        rl_branch = _make_random_branch(
            label="Random setup | rl",
            tune_dim=int(tune_dim),
            tune7_variant=tune7_variant,
            seed=int(random_seed),
            solver_tol=float(solver_tol),
            solver_max_iter=int(solver_max_iter),
            bundle=bundle,
        )
        setup_policy_summary = {
            "method": "random",
            "tune_dim": int(tune_dim),
            "seed": int(random_seed),
            "bandit_update": False,
            "loss": None,
        }
    else:
        raise ValueError(f"Unsupported SETUP_MODE: {setup_mode}")

    base_metrics = run_interleaved_branch_scenario(
        phase_label="baseline",
        branches=[base_branch],
        instances=instances,
        solve_mode="no_rl",
        solve_policy=None,
        permutation_seed=int(seed_start ^ 0x12345678),
        solve_max_cycles=int(solve_max_cycles),
        augment_params=_augment_params,
        classify_rl_failure=lambda **_: "",
        bandit_cfg=bandit_cfg,
        progress_every=None,
    )
    rl_metrics = run_interleaved_branch_scenario(
        phase_label="rl",
        branches=[rl_branch],
        instances=instances,
        solve_mode="rl",
        solve_policy=solve_policy,
        permutation_seed=int(seed_start ^ 0x12345678),
        solve_max_cycles=int(solve_max_cycles),
        augment_params=_augment_params,
        classify_rl_failure=lambda *, residual_norm, iterations: (
            "non_finite_residual_norm"
            if not np.isfinite(float(residual_norm))
            else (
                "residual_above_solve_tol;max_cycles_reached_without_convergence"
                if float(residual_norm) > float(solve_tol) and int(iterations) >= int(solve_max_cycles)
                else (
                    "residual_above_solve_tol"
                    if float(residual_norm) > float(solve_tol)
                    else (
                        "hit_or_exceeded_solve_max_cycles"
                        if int(iterations) >= int(solve_max_cycles)
                        else ""
                    )
                )
            )
        ),
        bandit_cfg=bandit_cfg,
        progress_every=None,
    )

    base_results = _results_from_metrics(metrics=base_metrics, label=base_branch.label)
    rl_results = _results_from_metrics(metrics=rl_metrics, label=rl_branch.label)

    print(f"\n=== Setup-aware Baseline vs RL ({setup_mode}) ===")
    print("Baseline solve mode:", baseline_solve_mode)
    print("Baseline config (plain solve):", {"tol": float(solver_tol), "max_iter": int(solver_max_iter)})
    if setup_mode == "bandit":
        print("Bandit setup policy:", setup_policy_summary)
    else:
        print("Random setup policy:", setup_policy_summary)
    print("Baseline:", _summarize(base_results))
    print("RL      :", _summarize(rl_results))
    print("Baseline failures:", _summarize_failures(base_results))
    print("RL failures      :", _summarize_failures(rl_results))
    base_overhead = _summarize_overhead(base_results)
    if base_overhead is not None:
        print("Baseline overhead (bandit/update overhead):", base_overhead)
    rl_overhead = _summarize_overhead(rl_results)
    if rl_overhead is not None:
        print("RL overhead (bandit/update overhead):", rl_overhead)
    rl_infer = _summarize_infer(rl_results)
    if rl_infer is not None:
        print("RL inference time (policy.predict only):", rl_infer)
    print("Eval grid sizes:", grid_sizes)
    print("Eval cases:", len(instances))

    failure_records = _failure_records_from_results(mode="baseline", results=base_results, instances=instances)
    failure_records.extend(_failure_records_from_results(mode="rl", results=rl_results, instances=instances))
    if failure_records:
        failure_log_path = Path(
            os.environ.get(
                "EVAL_FAILURE_LOG_PATH",
                Path.cwd() / f"eval_setup_default_vs_rl_{setup_mode}_failures.jsonl",
            )
        )
        _write_failure_log(failure_log_path, failure_records)
        print(f"FAILURE LOG: {failure_log_path}")
        print(f"TOTAL FAILED CASES: {len(failure_records)}")


if __name__ == "__main__":
    main()
