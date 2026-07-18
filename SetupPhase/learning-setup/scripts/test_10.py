from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

import test_9 as base

REPO_ROOT = Path(__file__).resolve().parents[3]
SOLVE_TEST_DIR = REPO_ROOT / "SolvePhase" / "hypre" / "src" / "test"
if str(SOLVE_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVE_TEST_DIR))

from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    SetupAwareRLConfig,
    SetupAwareSolvePolicyRunner,
    build_test10_branches,
    default_test_final_bandit_config_from_env,
    generate_difconv_instances,
    run_interleaved_branch_scenario,
)

SETUP_RL_MODE = os.environ.get("SETUP_RL_MODE", "random").strip().lower()
SETUP_RL_TUNE_DIM = int(
    os.environ.get(
        "SETUP_RL_TUNE_DIM",
        str(max(base.FINAL_TUNE_DIMS) if base.FINAL_TUNE_DIMS else 5),
    )
)
SETUP_RL_TUNE7_VARIANT = os.environ.get(
    "SETUP_RL_TUNE7_VARIANT",
    os.environ.get("TUNE7_VARIANT", "categorical"),
).strip().lower()
SETUP_RL_MODEL_TYPE = os.environ.get("SETUP_RL_MODEL_TYPE", os.environ.get("MODEL_TYPE", "mlp")).strip().lower()
SETUP_RL_ALGO = os.environ.get("SETUP_RL_ALGO", os.environ.get("ALGO", "ppo")).strip().lower()
SETUP_RL_OBS_MODE = os.environ.get("SETUP_RL_OBS_MODE", "full").strip().lower()
SETUP_RL_START_CASE = int(os.environ.get("SETUP_RL_START_CASE", "0"))
_SETUP_RL_DEFAULT_BASENAME = f"ppo_boomeramg_setup_{SETUP_RL_MODE}"
_SETUP_RL_DEFAULT_VEC = f"vecnormalize_setup_{SETUP_RL_MODE}.pkl"
SETUP_RL_MODEL_PATH = Path(os.environ.get("SETUP_RL_MODEL_PATH", str(SOLVE_TEST_DIR / _SETUP_RL_DEFAULT_BASENAME)))
SETUP_RL_VEC_PATH = Path(os.environ.get("SETUP_RL_VEC_PATH", str(SOLVE_TEST_DIR / _SETUP_RL_DEFAULT_VEC)))
FINAL_INCLUDE_DEFAULT = os.environ.get("FINAL_INCLUDE_DEFAULT", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}


def _progress_every() -> int | None:
    if int(base.PROGRESS_EVERY) <= 0:
        return None
    return int(base.PROGRESS_EVERY)


def _augment_params(params: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(params)
    if base.TEST9_RELAX_TYPE:
        out["relax_type"] = int(base.TEST9_RELAX_TYPE)
    return out


def _failure_reason_counts(reason_map: Dict[str, np.ndarray]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for label, arr in reason_map.items():
        counts: Dict[str, int] = {}
        for raw in arr.tolist():
            reason = str(raw or "").strip()
            if not reason:
                continue
            counts[reason] = counts.get(reason, 0) + 1
        out[label] = counts
    return out


def _build_label_meta(branches) -> Dict[str, Dict[str, Any]]:
    return {
        branch.label: {
            "method": str(branch.family),
            "tune_set": str(branch.tune_set),
            "fixed_params": dict(getattr(branch.policy, "_params", {})),
        }
        for branch in branches
    }


def _make_solve_policy() -> SetupAwareSolvePolicyRunner:
    cfg = SetupAwareRLConfig(
        tune_dim=int(SETUP_RL_TUNE_DIM),
        tune7_variant=str(SETUP_RL_TUNE7_VARIANT),
        algo=str(SETUP_RL_ALGO),
        model_type=str(SETUP_RL_MODEL_TYPE),
        model_path=SETUP_RL_MODEL_PATH,
        vec_path=SETUP_RL_VEC_PATH,
        fixed_grid=(base.FIXED_N, base.FIXED_N, base.FIXED_N),
        difconv_c_range=(float(base.C_MIN), float(base.C_MAX)),
        w_only=bool(os.environ.get("W_ONLY", "0").strip().lower() in {"1", "true", "yes", "on"}),
        w_center=float(base.SOLVE_W_CENTER),
        w_scale=float(base.SOLVE_W_SCALE),
        sweeps_min=int(base.SOLVE_SWEEPS_MIN),
        sweeps_max=int(base.SOLVE_SWEEPS_MAX),
        w_init=float(base.SOLVE_W_INIT) if base.SOLVE_W_INIT else None,
        sweeps_init=int(base.SOLVE_SWEEPS_INIT) if base.SOLVE_SWEEPS_INIT else None,
        solve_max_cycles=int(base.SOLVE_MAX_CYCLES),
        solve_tol=float(base.SOLVE_TOL),
        default_setup_params=dict(DEFAULT_SETUP_PARAMS),
        obs_mode=str(SETUP_RL_OBS_MODE),
    )
    return SetupAwareSolvePolicyRunner(cfg)


def _build_instances() -> Sequence[tuple[Dict[str, Any], np.ndarray]]:
    difconv_a = tuple(float(x) for x in os.environ.get("DIFCONV_A", "0,0,0").split(","))
    return generate_difconv_instances(
        T=int(base.T),
        seed=int(base.SEED),
        grid_choices=[(int(base.FIXED_N), int(base.FIXED_N), int(base.FIXED_N))],
        c_min=float(base.C_MIN),
        c_max=float(base.C_MAX),
        difconv_a=(float(difconv_a[0]), float(difconv_a[1]), float(difconv_a[2])),
    )


def _finalize_scenario(
    *,
    scenario_name: str,
    scenario_desc: str,
    solve_mode: str,
    branches,
    metrics: Dict[str, Any],
    run_dir: Path,
    bundle,
    permutation_seed: int,
) -> Dict[str, Any]:
    labels = [branch.label for branch in branches]
    label_meta = _build_label_meta(branches)
    runtime_sec = metrics["runtime_sec"]
    setup_runtime_sec = metrics["setup_runtime_sec"]
    solve_runtime_sec = metrics["solve_runtime_sec"]
    overhead_sec = metrics["overhead_sec"]
    select_sec = metrics["select_sec"]
    loss_eval_sec = metrics["loss_eval_sec"]
    update_sec = metrics["update_sec"]
    failed_flags = metrics["failed_flags"]
    failure_reason = metrics["failure_reason"]
    traces = metrics["traces"]
    failure_records = metrics["failure_records"]
    failure_reason_counts = _failure_reason_counts(failure_reason)

    run_prefix = f"{scenario_name}_interleaved_runtime"
    solve_mode_title = "RL solve" if solve_mode == "rl" else "non-RL solve"
    default_method_name = "default (fixed)" if "default (fixed)" in labels else labels[0]
    summary = base.save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix=run_prefix,
        method_names=labels,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        T=int(base.T),
        title=(
            f"{scenario_desc} cumulative runtime ({solve_mode_title})  "
            f"T={base.T}  n={base.FIXED_N}^3  c={base.C_MIN:g}..{base.C_MAX:g}"
        ),
        default_method_name=default_method_name,
        traces=traces,
        trace_keys=(),
        default_params=DEFAULT_SETUP_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "script": Path(__file__).name,
            "scenario_name": str(scenario_name),
            "scenario_desc": str(scenario_desc),
            "seed": int(base.SEED),
            "alpha": float(base.ALPHA),
            "l2": float(base.L2),
            "sigma": float(base.SIGMA),
            "fixed_n": int(base.FIXED_N),
            "c_min": float(base.C_MIN),
            "c_max": float(base.C_MAX),
            "context_dim": int(base.DIFCONV_CONTEXT_DIM),
            "candidate_pool_size": int(base.CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(base.ELITE_CACHE_SIZE),
            "method_filter": str(base.METHOD_FILTER),
            "branch_filter": str(base.BRANCH_FILTER),
            "solver_tol": float(base.SOLVER_TOL),
            "solver_max_iter": int(base.SOLVER_MAX_ITER),
            "test9_relax_type": (int(base.TEST9_RELAX_TYPE) if base.TEST9_RELAX_TYPE else None),
            "solve_model_type": str(SETUP_RL_MODEL_TYPE),
            "solve_model_path": str(SETUP_RL_MODEL_PATH),
            "solve_vec_path": str(SETUP_RL_VEC_PATH),
            "solve_tol": float(base.SOLVE_TOL),
            "solve_max_cycles": int(base.SOLVE_MAX_CYCLES),
            "solve_w_center": float(base.SOLVE_W_CENTER),
            "solve_w_scale": float(base.SOLVE_W_SCALE),
            "solve_sweeps_min": int(base.SOLVE_SWEEPS_MIN),
            "solve_sweeps_max": int(base.SOLVE_SWEEPS_MAX),
            "setup_param_resolution": int(bundle.param_resolution),
            "actions_count_tune3": int(len(bundle.actions_tune3)),
            "actions_count_tune5": int(len(bundle.actions_tune5)),
            "p_max_values": [int(v) for v in bundle.p_max_values],
            "agg_num_levels_values": [int(v) for v in bundle.agg_nl_values],
            "T": int(base.T),
            "continuation": False,
            "bias_mitigation": "within-step permutation on identical instance stream across scenario branches",
            "within_step_method_permutation": True,
            "within_step_tune_set_permutation": True,
            "permutation_seed": int(permutation_seed),
            "solve_mode": str(solve_mode),
            "solve_policy_start_case": int(SETUP_RL_START_CASE if solve_mode == "rl" else 0),
            "failed_count_total": {k: int(np.sum(v.astype(int))) for k, v in failed_flags.items()},
            "failed_reason_counts": failure_reason_counts,
            "total_setup_runtime_sec": {k: float(np.sum(v)) for k, v in setup_runtime_sec.items()},
            "mean_setup_runtime_sec": {k: float(np.mean(v)) for k, v in setup_runtime_sec.items()},
            "total_solve_runtime_sec": {k: float(np.sum(v)) for k, v in solve_runtime_sec.items()},
            "mean_solve_runtime_sec": {k: float(np.mean(v)) for k, v in solve_runtime_sec.items()},
            "total_select_sec": {k: float(np.sum(v)) for k, v in select_sec.items()},
            "mean_select_sec": {k: float(np.mean(v)) for k, v in select_sec.items()},
            "total_loss_eval_sec": {k: float(np.sum(v)) for k, v in loss_eval_sec.items()},
            "mean_loss_eval_sec": {k: float(np.mean(v)) for k, v in loss_eval_sec.items()},
            "total_update_sec": {k: float(np.sum(v)) for k, v in update_sec.items()},
            "mean_update_sec": {k: float(np.mean(v)) for k, v in update_sec.items()},
            "total_overhead_sec": {k: float(np.sum(v)) for k, v in overhead_sec.items()},
            "mean_overhead_sec": {k: float(np.mean(v)) for k, v in overhead_sec.items()},
            "total_end_to_end_sec": {k: float(np.sum(runtime_sec[k] + overhead_sec[k])) for k in labels},
            "mean_end_to_end_sec": {k: float(np.mean(runtime_sec[k] + overhead_sec[k])) for k in labels},
            "mean_test_problem_runtime_sec": {k: float(np.mean(v)) for k, v in runtime_sec.items()},
            "bandit_logic": "test_final_shared",
        },
    )

    data_csv_path = run_dir / f"per_instance_runtime_data_{scenario_name}.csv"
    base._save_dual_csv(
        out_csv=data_csv_path,
        solve_mode=solve_mode,
        labels=labels,
        label_meta=label_meta,
        runtime_sec=runtime_sec,
        setup_runtime_sec=setup_runtime_sec,
        solve_runtime_sec=solve_runtime_sec,
        overhead_sec=overhead_sec,
        select_sec=select_sec,
        loss_eval_sec=loss_eval_sec,
        update_sec=update_sec,
        failed=failed_flags,
        failure_reason=failure_reason,
        traces=traces,
        t_total=int(base.T),
    )

    failure_log_path = run_dir / f"failed_cases_{scenario_name}.jsonl"
    base._write_failure_records(failure_log_path, failure_records)
    scenario_wall_time_sec = float(time.perf_counter() - metrics["scenario_wall_start"])
    summary["scenario_wall_time_sec"] = float(scenario_wall_time_sec)
    summary["mean_scenario_wall_time_per_instance_sec"] = float(scenario_wall_time_sec / max(1, int(base.T)))
    summary_path = Path(summary["summary_path"])
    summary_path.write_text(json.dumps(base._json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return {
        "scenario_name": scenario_name,
        "scenario_desc": scenario_desc,
        "solve_mode": solve_mode,
        "branch_labels": labels,
        "summary": summary,
        "data_csv_path": data_csv_path,
        "failure_log_path": failure_log_path,
        "runtime_sec": runtime_sec,
        "setup_runtime_sec": setup_runtime_sec,
        "solve_runtime_sec": solve_runtime_sec,
        "overhead_sec": overhead_sec,
        "select_sec": select_sec,
        "update_sec": update_sec,
        "failed_flags": failed_flags,
        "failure_reason_counts": failure_reason_counts,
        "scenario_wall_time_sec": float(scenario_wall_time_sec),
    }


def _run_scenario(
    *,
    scenario_name: str,
    scenario_desc: str,
    phase_label: str,
    branches,
    instances,
    solve_mode: str,
    solve_policy,
    run_dir: Path,
    permutation_seed: int,
    bundle,
    bandit_cfg,
    solve_policy_start_case: int,
) -> Dict[str, Any] | None:
    if not branches:
        return None
    print(f"Single run: {scenario_desc}, T={base.T}")
    scenario_wall_start = time.perf_counter()
    metrics = run_interleaved_branch_scenario(
        phase_label=phase_label,
        branches=branches,
        instances=instances,
        solve_mode=solve_mode,
        solve_policy=solve_policy,
        permutation_seed=int(permutation_seed),
        solve_max_cycles=int(base.SOLVE_MAX_CYCLES),
        augment_params=_augment_params,
        classify_rl_failure=base._classify_rl_failure,
        bandit_cfg=bandit_cfg,
        progress_every=_progress_every(),
        solve_policy_start_case=int(solve_policy_start_case),
    )
    metrics["scenario_wall_start"] = scenario_wall_start
    return _finalize_scenario(
        scenario_name=scenario_name,
        scenario_desc=scenario_desc,
        solve_mode=solve_mode,
        branches=branches,
        metrics=metrics,
        run_dir=run_dir,
        bundle=bundle,
        permutation_seed=int(permutation_seed),
    )


def _build_branches_and_bundle():
    return build_test10_branches(
        final_tune_dims=base.FINAL_TUNE_DIMS,
        tune7_variant=base.TUNE7_VARIANT,
        seed=int(base.SEED),
        solver_tol=float(base.SOLVER_TOL),
        solver_max_iter=int(base.SOLVER_MAX_ITER),
        include_default=bool(FINAL_INCLUDE_DEFAULT),
        method_filter=str(base.METHOD_FILTER),
        branch_filter=str(base.BRANCH_FILTER),
        bandit_cfg=default_test_final_bandit_config_from_env(),
    )


def main() -> None:
    print(f"Using setup-aware RL model: {SETUP_RL_MODEL_PATH}")
    print(f"Using setup-aware VecNormalize: {SETUP_RL_VEC_PATH}")
    print(f"Setup-aware RL mode={SETUP_RL_MODE} tune_dim={SETUP_RL_TUNE_DIM} tune7_variant={SETUP_RL_TUNE7_VARIANT} obs_mode={SETUP_RL_OBS_MODE} start_case={SETUP_RL_START_CASE}")

    warm_rng = np.random.default_rng(int(base.SEED) ^ 0xBADC0FFE)
    warm_mkw, _, _ = base.stencil_0_difconv_rl(
        rng=warm_rng,
        t=0,
        trial=0,
        nx=int(base.FIXED_N),
        ny=int(base.FIXED_N),
        nz=int(base.FIXED_N),
        n_min=int(base.FIXED_N),
        n_max=int(base.FIXED_N),
        c_min=float(base.C_MIN),
        c_max=float(base.C_MAX),
    )
    _ = base.solve(params=DEFAULT_SETUP_PARAMS, **warm_mkw)
    solve_policy = _make_solve_policy()
    instances = _build_instances()
    bandit_cfg = default_test_final_bandit_config_from_env()

    run_dir = base.create_run_output_dir(
        base_dir=base.plots_base_dir,
        script_name=Path(__file__).stem,
        problem_name="difconv",
        size_tag=f"{base.FIXED_N}x{base.FIXED_N}x{base.FIXED_N}",
        T=int(base.T),
        seed=int(base.SEED),
    )

    print("Within-step permutation over all branches: enabled")
    scenario_results: List[Dict[str, Any]] = []

    branches, bundle = _build_branches_and_bundle()
    fixed_branches = [branch for branch in branches if branch.family == "default"]
    bandit_branches = [branch for branch in branches if branch.family != "default"]

    permutation_seed = int(base.SEED ^ 0x1A2B3C4D)

    scenarios = [
        ("rl_only", "RL only (fixed setup + RL solve)", "rl only", fixed_branches, "rl"),
        ("rl_bandit", "RL + Bandit (bandit setup + RL solve)", "rl+bandit", bandit_branches, "rl"),
        ("bandit_only", "Bandit only (bandit setup + non-RL solve)", "bandit only", bandit_branches, "no_rl"),
        ("fixed_only", "Fixed only (fixed setup + non-RL solve)", "fixed only", fixed_branches, "no_rl"),
    ]

    for offset, (scenario_name, scenario_desc, phase_label, scenario_branches, solve_mode) in enumerate(scenarios):
        if not scenario_branches:
            continue
        if scenario_name in {"rl_bandit", "bandit_only"}:
            scenario_branches, bundle = _build_branches_and_bundle()
            scenario_branches = [branch for branch in scenario_branches if branch.family != "default"]
        elif scenario_name in {"rl_only", "fixed_only"}:
            scenario_branches, bundle = _build_branches_and_bundle()
            scenario_branches = [branch for branch in scenario_branches if branch.family == "default"]
        result = _run_scenario(
            scenario_name=scenario_name,
            scenario_desc=scenario_desc,
            phase_label=phase_label,
            branches=scenario_branches,
            instances=instances,
            solve_mode=solve_mode,
            solve_policy=solve_policy if solve_mode == "rl" else None,
            run_dir=run_dir,
            permutation_seed=int(permutation_seed + offset * 1009),
            bundle=bundle,
            bandit_cfg=bandit_cfg,
            solve_policy_start_case=(int(SETUP_RL_START_CASE) if solve_mode == "rl" else 0),
        )
        if result is not None:
            scenario_results.append(result)
            base._print_scenario_summary(result)

    base._print_overall_comparison(scenario_results)

    export = {
        "script": Path(__file__).name,
        "seed": int(base.SEED),
        "T": int(base.T),
        "fixed_n": int(base.FIXED_N),
        "c_min": float(base.C_MIN),
        "c_max": float(base.C_MAX),
        "setup_param_resolution": int(bundle.param_resolution),
        "final_tune_dims": [int(v) for v in base.FINAL_TUNE_DIMS],
        "method_filter": str(base.METHOD_FILTER),
        "branch_filter": str(base.BRANCH_FILTER),
        "test9_relax_type": (int(base.TEST9_RELAX_TYPE) if base.TEST9_RELAX_TYPE else None),
        "solve_model_type": str(SETUP_RL_MODEL_TYPE),
        "solve_model_path": str(SETUP_RL_MODEL_PATH),
        "solve_vec_path": str(SETUP_RL_VEC_PATH),
        "solve_tol": float(base.SOLVE_TOL),
        "solve_max_cycles": int(base.SOLVE_MAX_CYCLES),
        "solver_tol": float(base.SOLVER_TOL),
        "solver_max_iter": int(base.SOLVER_MAX_ITER),
        "scenarios": [result["summary"] for result in scenario_results],
    }
    export_path = run_dir / "test_10_export_summary.json"
    export_path.write_text(json.dumps(base._json_safe(export), indent=2) + "\n", encoding="utf-8")
    print(f"EXPORT SUMMARY: {export_path}")


if __name__ == "__main__":
    main()
