"""
Dedicated online benchmark for tune7 AMG setup tuning.

This harness compares:
  - Shared LinUCB v2 | tune3
  - Shared LinTS v4 | tune7
  - SquareCB linear | tune7
  - SquareCB quadratic | tune7
  - Nyström KernelUCB | tune7
  - Hierarchical SquareCB | tune7
  - Hierarchical LinUCB v5 | tune7

Protocol
--------
- one solver evaluation per round
- no retry loop
- raw penalized runtime is the learner update target
- explicit failed-case accounting per method

Primary metrics
---------------
- full-stream cumulative raw penalized runtime
"""

from __future__ import annotations

from collections import deque
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learners._tune7_online_features import Tune7FeatureBuilder
from utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
from utils.setup_amg import init_param_trace, progress_bar, record_param_trace
from utils.tune7_online_benchmark import (
    DEFAULT_AGG_NUM_LEVELS_VALUES,
    DEFAULT_COARSEN_TYPE_VALUES,
    DEFAULT_P_MAX_ELMTS_VALUES,
    DEFAULT_TUNE7_INTERP_TYPES,
    SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM,
    TRACE_KEYS_FINAL,
    BenchmarkBranch,
    build_actions_tune3,
    build_actions_tune7,
    build_grids,
    build_online_benchmark_branches,
    compute_failure_scale_min_runtime_sec,
    ensure_default_arm,
    evaluate_action,
    generate_instances,
    parse_int_list_env,
    save_per_instance_csv,
    DEFAULT_PARAMS,
)
from solver import solve
from utils.scalar_anisotropic_diffusion import stencil_0_scalar_anisotropic_diffusion_rl


T = int(os.environ.get("T", "1000"))
SEED = int(os.environ.get("SEED", "39393939"))
FIXED_N = int(os.environ.get("FIXED_N", "60"))
FIXED_NX = int(os.environ.get("FIXED_NX", str(FIXED_N)))
FIXED_NY = int(os.environ.get("FIXED_NY", str(FIXED_N)))
FIXED_NZ = int(os.environ.get("FIXED_NZ", str(FIXED_N)))
C_MIN = float(os.environ.get("C_MIN", "1.0"))
C_MAX = float(os.environ.get("C_MAX", "1000.0"))
GRID_N = int(os.environ.get("GRID_N", "20"))
GRID_MAX = float(os.environ.get("GRID_MAX", "0.95"))

SOLVER_TOL = float(os.environ.get("SOLVER_TOL", "1e-8"))
SOLVER_MAX_ITER = int(os.environ.get("SOLVER_MAX_ITER", "10000"))
LEGACY_SOLVER_TOL = float(os.environ.get("LEGACY_SOLVER_TOL", "1e-8"))
LEGACY_SOLVER_MAX_ITER = int(os.environ.get("LEGACY_SOLVER_MAX_ITER", "10000"))

FAILURE_PENALTY_MULTIPLIER = float(os.environ.get("FAILURE_PENALTY_MULTIPLIER", "2.0"))
FAILURE_SCALE_WINDOW = int(os.environ.get("FAILURE_SCALE_WINDOW", "200"))
FAILURE_SEVERITY_CAP = float(os.environ.get("FAILURE_SEVERITY_CAP", "6.0"))
FAILURE_SCALE_MIN_RUNTIME_SEC = float(os.environ.get("FAILURE_SCALE_MIN_RUNTIME_SEC", "0.0"))
STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER = float(
    os.environ.get("STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER", "3.0")
)

KERNEL_CONTEXT_COUNT = int(os.environ.get("TUNE7_BENCH_KERNEL_CONTEXT_COUNT", "64"))
KERNEL_SIGMA_Z_SAMPLE_SIZE = int(os.environ.get("TUNE7_BENCH_KERNEL_SIGMA_Z_SAMPLE_SIZE", "512"))
KERNEL_SIGMA_Z_PAIR_COUNT = int(os.environ.get("TUNE7_BENCH_KERNEL_SIGMA_Z_PAIR_COUNT", "10000"))
KERNEL_LANDMARK_COUNT = int(os.environ.get("TUNE7_BENCH_KERNEL_LANDMARK_COUNT", "32"))
BRANCH_FILTER = tuple(
    x.strip()
    for x in os.environ.get("TUNE7_BENCH_BRANCH_FILTER", "").split(",")
    if x.strip()
)
PLOT_TRACES = os.environ.get("TUNE7_BENCH_PLOT_TRACES", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
REPEAT_SAME_INSTANCE = os.environ.get("TUNE7_BENCH_REPEAT_SAME_INSTANCE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

PLOTS_BASE_DIR = Path(__file__).resolve().parent.parent / "plots" / "Combined"
PLOTS_BASE_DIR.mkdir(parents=True, exist_ok=True)


def _select_trace_keys(traces: Dict[str, Dict[str, np.ndarray]]) -> List[str]:
    active: List[str] = []
    for key in TRACE_KEYS_FINAL:
        arrays = []
        for trace in traces.values():
            arr = np.asarray(trace[key], dtype=float).reshape(-1)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                arrays.append(finite)
        if not arrays:
            continue
        data = np.concatenate(arrays, axis=0)
        uniq = np.unique(np.round(data, 12))
        if uniq.size > 1:
            active.append(str(key))
    return active


def main() -> None:
    if T <= 0:
        raise ValueError("T must be positive")
    if FIXED_NX <= 0 or FIXED_NY <= 0 or FIXED_NZ <= 0:
        raise ValueError("FIXED_NX/FIXED_NY/FIXED_NZ must all be positive")

    sampler_kwargs = {
        "nx": int(FIXED_NX),
        "ny": int(FIXED_NY),
        "nz": int(FIXED_NZ),
        "n_min": int(max(FIXED_NX, FIXED_NY, FIXED_NZ)),
        "n_max": int(max(FIXED_NX, FIXED_NY, FIXED_NZ)),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
    }

    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_0_scalar_anisotropic_diffusion_rl(
        rng=warm_rng,
        t=0,
        trial=0,
        **sampler_kwargs,
    )
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    th_grid, mxrs_grid, tr_grid = build_grids(grid_n=int(GRID_N), grid_max=float(GRID_MAX))
    p_max_values = parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES, os.environ)
    agg_nl_values = parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES, os.environ)
    coarsen_type_values = parse_int_list_env("COARSEN_TYPE_VALUES", DEFAULT_COARSEN_TYPE_VALUES, os.environ)
    interp_values = parse_int_list_env("TUNE7_INTERP_TYPES", DEFAULT_TUNE7_INTERP_TYPES, os.environ)

    actions_tune3 = build_actions_tune3(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune3, _default_arm_index_tune3 = ensure_default_arm(actions_tune3)

    actions_tune7, parameter_spec_tune7 = build_actions_tune7(
        th_grid=th_grid,
        mxrs_grid=mxrs_grid,
        tr_grid=tr_grid,
        p_max_values=p_max_values,
        agg_nl_values=agg_nl_values,
        coarsen_type_values=coarsen_type_values,
        interp_values=interp_values,
    )
    actions_tune7, _default_arm_index_tune7 = ensure_default_arm(actions_tune7)

    instances = generate_instances(
        T=int(T),
        seed=int(SEED),
        sampler_kwargs=sampler_kwargs,
        repeat_same_instance=bool(REPEAT_SAME_INSTANCE),
    )

    tune7_feature_builder = Tune7FeatureBuilder(tuple(actions_tune7), parameter_spec_tune7)

    b_min_runtime_tune7 = compute_failure_scale_min_runtime_sec(
        mkw=instances[0][0],
        solver_tol=float(SOLVER_TOL),
        solver_max_iter=int(SOLVER_MAX_ITER),
        failure_scale_min_runtime_sec=float(FAILURE_SCALE_MIN_RUNTIME_SEC),
    )
    b_min_runtime_tune3 = compute_failure_scale_min_runtime_sec(
        mkw=instances[0][0],
        solver_tol=float(LEGACY_SOLVER_TOL),
        solver_max_iter=int(LEGACY_SOLVER_MAX_ITER),
        failure_scale_min_runtime_sec=float(FAILURE_SCALE_MIN_RUNTIME_SEC),
    )

    branches, branch_metadata = build_online_benchmark_branches(
        seed=int(SEED),
        actions_tune3=actions_tune3,
        actions_tune7=actions_tune7,
        parameter_spec_tune7=parameter_spec_tune7,
        instance_contexts=[context for (_mkw, context) in instances],
        solver_tol=float(SOLVER_TOL),
        solver_max_iter=int(SOLVER_MAX_ITER),
        legacy_solver_tol=float(LEGACY_SOLVER_TOL),
        legacy_solver_max_iter=int(LEGACY_SOLVER_MAX_ITER),
        kernel_context_count=int(KERNEL_CONTEXT_COUNT),
        kernel_sigma_z_sample_size=int(KERNEL_SIGMA_Z_SAMPLE_SIZE),
        kernel_sigma_z_pair_count=int(KERNEL_SIGMA_Z_PAIR_COUNT),
        kernel_landmark_count=int(KERNEL_LANDMARK_COUNT),
    )
    if BRANCH_FILTER:
        keep = set(BRANCH_FILTER)
        branches = [branch for branch in branches if branch.label in keep]
        if not branches:
            raise ValueError("TUNE7_BENCH_BRANCH_FILTER removed every branch")
        branch_metadata["branch_hyperparams"] = {
            label: params
            for label, params in branch_metadata["branch_hyperparams"].items()
            if label in keep
        }

    labels = [branch.label for branch in branches]
    runtime_sec = {label: np.zeros(T, dtype=float) for label in labels}
    select_sec = {label: np.zeros(T, dtype=float) for label in labels}
    loss_eval_sec = {label: np.zeros(T, dtype=float) for label in labels}
    update_sec = {label: np.zeros(T, dtype=float) for label in labels}
    overhead_sec = {label: np.zeros(T, dtype=float) for label in labels}
    penalized_loss_sec = {label: np.zeros(T, dtype=float) for label in labels}
    failed_flags = {label: np.zeros(T, dtype=bool) for label in labels}
    traces = {label: init_param_trace(TRACE_KEYS_FINAL, T) for label in labels}

    success_history_by_label = {
        branch.label: deque(maxlen=int(FAILURE_SCALE_WINDOW))
        for branch in branches
    }

    order_rng = np.random.default_rng(SEED ^ 0x1A2B3C4D)
    fail_runtime_sec = 1e9
    for t in range(int(T)):
        progress_bar(t + 1, int(T), prefix="benchmark")
        mkw, context = instances[t]
        for branch_idx in order_rng.permutation(len(branches)):
            branch = branches[int(branch_idx)]
            select_start_ns = time.perf_counter_ns()
            params, _info = branch.policy.select(context=context)
            select_elapsed_sec = float((time.perf_counter_ns() - select_start_ns) / 1e9)

            b_min_runtime_sec = float(b_min_runtime_tune3 if branch.tune_set == "tune3" else b_min_runtime_tune7)
            outcome, raw_loss_sec, loss_eval_elapsed_sec = evaluate_action(
                params=params,
                mkw=mkw,
                success_runtime_history=success_history_by_label[branch.label],
                b_min_runtime_sec=float(b_min_runtime_sec),
                solver_tol=float(branch.solver_tol),
                solver_max_iter=int(branch.solver_max_iter),
                fail_runtime_sec=float(fail_runtime_sec),
                failure_penalty_multiplier=float(FAILURE_PENALTY_MULTIPLIER),
                failure_severity_cap=float(FAILURE_SEVERITY_CAP),
                structural_failure_surcharge_multiplier=float(STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER),
            )

            update_start_ns = time.perf_counter_ns()
            branch.policy.update(
                loss=float(raw_loss_sec),
                context=context,
                params=params,
                outcome=outcome,
            )
            update_elapsed_sec = float((time.perf_counter_ns() - update_start_ns) / 1e9)

            runtime_sec[branch.label][t] = float(outcome["runtime"])
            select_sec[branch.label][t] = float(select_elapsed_sec)
            loss_eval_sec[branch.label][t] = float(loss_eval_elapsed_sec)
            update_sec[branch.label][t] = float(update_elapsed_sec)
            overhead_sec[branch.label][t] = float(select_elapsed_sec + loss_eval_elapsed_sec + update_elapsed_sec)
            penalized_loss_sec[branch.label][t] = float(raw_loss_sec)
            failed_flags[branch.label][t] = bool(outcome.get("failed", False))
            record_param_trace(traces[branch.label], t=t, params=params, keys=TRACE_KEYS_FINAL)

    progress_bar(int(T), int(T), prefix="benchmark")

    size_tag = f"{int(FIXED_NX)}x{int(FIXED_NY)}x{int(FIXED_NZ)}"
    run_dir = create_run_output_dir(
        base_dir=PLOTS_BASE_DIR,
        script_name=(Path(__file__).stem + ("_same_instance" if REPEAT_SAME_INSTANCE else "")),
        problem_name="scalar_anisotropic_diffusion",
        size_tag=size_tag,
        T=int(T),
        seed=int(SEED),
    )

    full_stream_raw_penalized = {
        label: float(np.sum(np.asarray(penalized_loss_sec[label], dtype=float)))
        for label in labels
    }
    total_raw_runtime = {
        label: float(np.sum(np.asarray(runtime_sec[label], dtype=float)))
        for label in labels
    }
    total_select_overhead = {
        label: float(np.sum(np.asarray(select_sec[label], dtype=float)))
        for label in labels
    }
    total_loss_eval_overhead = {
        label: float(np.sum(np.asarray(loss_eval_sec[label], dtype=float)))
        for label in labels
    }
    total_update_overhead = {
        label: float(np.sum(np.asarray(update_sec[label], dtype=float)))
        for label in labels
    }
    total_overhead = {
        label: float(np.sum(np.asarray(overhead_sec[label], dtype=float)))
        for label in labels
    }
    failed_case_count = {
        label: int(np.sum(np.asarray(failed_flags[label], dtype=int)))
        for label in labels
    }

    active_trace_keys = _select_trace_keys(traces) if PLOT_TRACES else []
    summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="benchmark_tune7_online",
        method_names=labels,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        T=int(T),
        title=(
            f"Online tune7 benchmark runtime  {size_tag}  T={int(T)}"
            + ("  repeated same instance" if REPEAT_SAME_INSTANCE else "")
        ),
        traces=traces,
        trace_keys=active_trace_keys,
        trace_title=(
            (
                f"Online tune7 benchmark traces  {size_tag}  T={int(T)}"
                + ("  repeated same instance" if REPEAT_SAME_INSTANCE else "")
            )
            if active_trace_keys
            else None
        ),
        default_params=DEFAULT_PARAMS,
        summary_extra={
            "problem": "scalar_anisotropic_diffusion",
            "size_tag": size_tag,
            "instance_stream_mode": ("repeated_same_instance" if REPEAT_SAME_INSTANCE else "iid_stream"),
            "repeat_same_instance": bool(REPEAT_SAME_INSTANCE),
            "full_stream_cumulative_raw_penalized_loss_sec": full_stream_raw_penalized,
            "total_raw_runtime_sec": total_raw_runtime,
            "total_select_overhead_sec": total_select_overhead,
            "total_loss_eval_overhead_sec": total_loss_eval_overhead,
            "total_update_overhead_sec": total_update_overhead,
            "total_overhead_sec": total_overhead,
            "failed_case_count": failed_case_count,
            "grid_n": int(GRID_N),
            "grid_max": float(GRID_MAX),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "legacy_solver_tol": float(LEGACY_SOLVER_TOL),
            "legacy_solver_max_iter": int(LEGACY_SOLVER_MAX_ITER),
            "branch_hyperparams": branch_metadata["branch_hyperparams"],
            "tune3_action_count": int(len(actions_tune3)),
            "tune7_action_count": int(len(actions_tune7)),
            "context_dim": int(SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM),
            "v4_feature_dim": int(tune7_feature_builder.phi_v4_matrix(instances[0][1], np.asarray([0], dtype=int)).shape[1]),
            "p_max_elmts_values": [int(v) for v in p_max_values],
            "agg_num_levels_values": [int(v) for v in agg_nl_values],
            "coarsen_type_values": [int(v) for v in coarsen_type_values],
            "interp_type_values": [int(v) for v in interp_values],
            "branch_labels": labels,
        },
    )

    save_per_instance_csv(
        out_csv=run_dir / "per_instance_runtime_data.csv",
        branches=branches,
        instances=instances,
        runtime_sec=runtime_sec,
        select_sec=select_sec,
        loss_eval_sec=loss_eval_sec,
        update_sec=update_sec,
        overhead_sec=overhead_sec,
        penalized_loss_sec=penalized_loss_sec,
        failed_flags=failed_flags,
        traces=traces,
    )

    print(f"Saved summary to {run_dir / 'benchmark_tune7_online_summary.json'}")
    print(f"Full-stream cumulative raw penalized loss: {summary['full_stream_cumulative_raw_penalized_loss_sec']}")
    print(f"Failed-case counts: {summary['failed_case_count']}")


if __name__ == "__main__":
    main()
