"""Trace evaluation and interleaved comparison accounting."""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from solve.controllers.ppo import SetupAwareSolvePolicyRunner
from hypre.bindings import augment_setup_params
from solve.core.outcomes import classify_rl_failure
from setup.utils.setup_amg import init_param_trace, progress_bar, record_param_trace
from .action_spaces import DEFAULT_SETUP_PARAMS, TRACE_KEYS_FINAL
from .native_evaluation import solve_no_rl_case, solve_setup_aware_rl_case
from .setup_branches import BranchRun, TestFinalBanditConfig, build_test10_branches, default_branch_label, default_test_final_bandit_config_from_env, generate_difconv_instances, run_bandit_step_test_final


def fixed_trace(
    *,
    T: int,
    grid: Tuple[int, int, int],
    seed: int,
    tune_dim: int,
    bandit_method: str,
    solve_mode: str = "no_rl",
    solve_policy: Optional[SetupAwareSolvePolicyRunner] = None,
    trace_records: Optional[List[Dict[str, Any]]] = None,
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
    trace: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for case_index, (mkw, context) in enumerate(instances):
        if str(solve_mode).strip().lower() == "no_rl":
            def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_no_rl_case(
                    params=selected_params,
                    mkw=dict(mkw),
                    solver_tol=float(os.environ.get("SOLVER_TOL", "1e-6")),
                    solver_max_iter=int(os.environ.get("SOLVER_MAX_ITER", "50")),
                    augment_params=augment_setup_params,
                )
        elif str(solve_mode).strip().lower() == "rl":
            if solve_policy is None:
                raise ValueError("solve_policy is required when solve_mode='rl'")

            def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_setup_aware_rl_case(
                    params=selected_params,
                    mkw=dict(mkw),
                    solve_policy=solve_policy,
                    augment_params=augment_setup_params,
                    classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
                        residual_norm=float(residual_norm),
                        iterations=int(iterations),
                        solve_tol=float(solve_policy.cfg.solve_tol),
                        solve_max_cycles=int(solve_policy.cfg.solve_max_cycles),
                    ),
                    solve_max_cycles=int(solve_policy.cfg.solve_max_cycles),
                )
        else:
            raise ValueError(f"Unsupported solve_mode: {solve_mode}")

        def fallback_solver_fn(_params: Dict[str, Any]) -> Dict[str, Any]:
            return solve_no_rl_case(
                params=dict(DEFAULT_SETUP_PARAMS),
                mkw=dict(mkw),
                solver_tol=float(
                    os.environ.get("SOLVER_TOL", "1e-6")
                    if solve_policy is None
                    else solve_policy.cfg.solve_tol
                ),
                solver_max_iter=int(
                    os.environ.get("SOLVER_MAX_ITER", "50")
                    if solve_policy is None
                    else solve_policy.cfg.solve_max_cycles
                ),
                augment_params=augment_setup_params,
            )

        params, out, timing, fallback_used, prev_update_est = run_bandit_step_test_final(
            policy=branch.policy,
            parameter_space=branch.parameter_space,
            problem_context=np.asarray(context, dtype=float),
            solver_fn=solver_fn,
            fallback_solver_fn=fallback_solver_fn,
            prev_update_est=float(prev_update_est),
        )
        trace.append((dict(mkw), dict(params)))
        if trace_records is not None:
            trace_records.append(
                {
                    "instance_index": int(case_index),
                    "mkw": dict(mkw),
                    "context": np.asarray(context, dtype=float).tolist(),
                    "params": dict(params),
                    "fallback_used": int(fallback_used),
                    "bandit_timing": dict(timing),
                    "feedback_outcome": dict(out),
                }
            )
    return trace

def eval_runner(runner: SetupAwareSolvePolicyRunner, trace: Sequence[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, Any]:
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
            augment_params=augment_setup_params,
            classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
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
    runtimes = [float(v["runtime"]) for v in vals]
    setup_runtimes = [float(v["setup_runtime"]) for v in vals]
    solve_runtimes = [float(v["solve_runtime"]) for v in vals]
    infer_runtimes = [float(v.get("infer_runtime", 0.0)) for v in vals]
    per_case_mean_w = [
        float(np.mean(actions)) if actions else float("nan")
        for actions in (list(v.get("cycle_actions", ())) for v in vals)
    ]
    max_cycles = max((len(v.get("cycle_actions", ())) for v in vals), default=0)
    mean_w_by_cycle = []
    for cycle in range(max_cycles):
        cycle_values = [
            float(v["cycle_actions"][cycle])
            for v in vals
            if cycle < len(v.get("cycle_actions", ()))
        ]
        mean_w_by_cycle.append(float(np.mean(cycle_values)))
    return {
        "cases": int(len(vals)),
        "mean_runtime": float(np.mean(runtimes)),
        "total_runtime": float(np.sum(runtimes)),
        "mean_setup_runtime": float(np.mean(setup_runtimes)),
        "total_setup_runtime": float(np.sum(setup_runtimes)),
        "mean_solve_runtime": float(np.mean(solve_runtimes)),
        "total_solve_runtime": float(np.sum(solve_runtimes)),
        "mean_infer_runtime": float(np.mean(infer_runtimes)),
        "total_infer_runtime": float(np.sum(infer_runtimes)),
        "mean_runtime_with_controller": float(
            np.mean(np.asarray(runtimes) + np.asarray(infer_runtimes))
        ),
        "total_runtime_with_controller": float(
            np.sum(np.asarray(runtimes) + np.asarray(infer_runtimes))
        ),
        "failed_count": int(fails),
        "mean_iterations": float(np.mean([v["iterations"] for v in vals])),
        "mean_final_w": float(np.mean([v["final_w"] for v in vals if np.isfinite(v["final_w"])])),
        "action_hist": {int(k): int(action_hist[k]) for k in sorted(action_hist)},
        "per_case_mean_w": per_case_mean_w,
        "mean_w_by_cycle": mean_w_by_cycle,
    }

def alloc_branch_metrics(*, labels: Sequence[str], T: int) -> Dict[str, Any]:
    return {
        "runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "setup_runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "solve_runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "infer_runtime_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "overhead_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "select_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "loss_eval_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "update_sec": {label: np.zeros(T, dtype=float) for label in labels},
        "failed_flags": {label: np.zeros(T, dtype=bool) for label in labels},
        "failure_reason": {label: np.full(T, "", dtype=object) for label in labels},
        "iterations": {label: np.zeros(T, dtype=int) for label in labels},
        "residual_norm": {label: np.full(T, np.nan, dtype=float) for label in labels},
        "final_w": {label: np.full(T, np.nan, dtype=float) for label in labels},
        "final_sweeps_down": {label: np.full(T, -1, dtype=int) for label in labels},
        "final_sweeps_up": {label: np.full(T, -1, dtype=int) for label in labels},
        "traces": {label: init_param_trace(TRACE_KEYS_FINAL, T) for label in labels},
        "prev_update_est": {label: 0.0 for label in labels},
    }

def run_interleaved_branch_scenario(
    *,
    phase_label: str,
    branches: Sequence[BranchRun],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    solve_mode: str,
    solve_policy: Optional[SetupAwareSolvePolicyRunner],
    permutation_seed: int,
    solve_max_cycles: int,
    augment_params: Callable[[Dict[str, Any]], Dict[str, Any]],
    classify_rl_failure: Callable[..., str],
    bandit_cfg: TestFinalBanditConfig,
    progress_every: Optional[int],
    solve_policy_start_case: int = 0,
) -> Dict[str, Any]:
    labels = [branch.label for branch in branches]
    metrics = alloc_branch_metrics(labels=labels, T=len(instances))
    phase_start_time = time.perf_counter()
    rng_order = np.random.default_rng(int(permutation_seed))
    failure_records: List[Dict[str, Any]] = []

    branch_by_label = {branch.label: branch for branch in branches}
    branch_meta = {
        branch.label: {
            "method": str(branch.family),
            "tune_set": str(branch.tune_set),
            "fixed_params": dict(getattr(branch.policy, "_params", {})),
        }
        for branch in branches
    }

    for local_t, (mkw, context) in enumerate(instances):
        order = rng_order.permutation(len(branches))
        for idx in order:
            branch = branches[int(idx)]
            label = branch.label
            if solve_mode == "rl":
                if solve_policy is None:
                    raise ValueError("solve_policy is required for solve_mode='rl'")

                if int(local_t) < int(solve_policy_start_case):

                    def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                        return solve_no_rl_case(
                            params=selected_params,
                            mkw=dict(mkw),
                            solver_tol=float(branch.solver_tol),
                            solver_max_iter=int(branch.solver_max_iter),
                            augment_params=augment_params,
                        )

                else:

                    def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                        return solve_setup_aware_rl_case(
                            params=selected_params,
                            mkw=dict(mkw),
                            solve_policy=solve_policy,
                            augment_params=augment_params,
                            classify_rl_failure=classify_rl_failure,
                            solve_max_cycles=int(solve_max_cycles),
                        )

            elif solve_mode == "no_rl":

                def solver_fn(selected_params: Dict[str, Any]) -> Dict[str, Any]:
                    return solve_no_rl_case(
                        params=selected_params,
                        mkw=dict(mkw),
                        solver_tol=float(branch.solver_tol),
                        solver_max_iter=int(branch.solver_max_iter),
                        augment_params=augment_params,
                    )

            else:
                raise ValueError(f"Unsupported solve_mode: {solve_mode}")

            def fallback_solver_fn(_params: Dict[str, Any]) -> Dict[str, Any]:
                return solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=float(
                        branch.solver_tol
                        if solve_mode == "no_rl"
                        else solve_policy.cfg.solve_tol
                    ),
                    solver_max_iter=int(
                        branch.solver_max_iter
                        if solve_mode == "no_rl"
                        else solve_policy.cfg.solve_max_cycles
                    ),
                    augment_params=augment_params,
                )

            params, out, timing, _fallback_used, last_update_sec = run_bandit_step_test_final(
                policy=branch.policy,
                parameter_space=branch.parameter_space,
                problem_context=np.asarray(context, dtype=float),
                solver_fn=solver_fn,
                fallback_solver_fn=fallback_solver_fn,
                prev_update_est=float(metrics["prev_update_est"][label]),
            )

            metrics["runtime_sec"][label][local_t] = float(out["runtime"])
            metrics["setup_runtime_sec"][label][local_t] = float(out["setup_runtime"])
            metrics["solve_runtime_sec"][label][local_t] = float(out["solve_runtime"])
            metrics["infer_runtime_sec"][label][local_t] = float(out.get("infer_runtime", 0.0))
            metrics["overhead_sec"][label][local_t] = float(timing["overhead_sec"])
            metrics["select_sec"][label][local_t] = float(timing["select_sec"])
            metrics["loss_eval_sec"][label][local_t] = float(timing["loss_eval_sec"])
            metrics["update_sec"][label][local_t] = float(timing["update_sec"])
            metrics["failed_flags"][label][local_t] = bool(out.get("failed", False))
            metrics["failure_reason"][label][local_t] = str(out.get("failure_reason", ""))
            metrics["iterations"][label][local_t] = int(out.get("iterations", 0))
            metrics["residual_norm"][label][local_t] = float(out.get("residual_norm", np.nan))
            metrics["final_w"][label][local_t] = float(out.get("final_w", np.nan))
            metrics["final_sweeps_down"][label][local_t] = int(out.get("final_sweeps_down", -1))
            metrics["final_sweeps_up"][label][local_t] = int(out.get("final_sweeps_up", -1))
            record_param_trace(metrics["traces"][label], t=local_t, params=params, keys=TRACE_KEYS_FINAL)
            metrics["prev_update_est"][label] = float(last_update_sec)

            if bool(out.get("failed", False)):
                failure_records.append(
                    {
                        "phase": "dual_permuted",
                        "solve_mode": str(solve_mode),
                        "t": int(local_t + 1),
                        "method": str(branch.family),
                        "label": str(label),
                        "tune_set": str(branch.tune_set),
                        "failure_reason": str(out.get("failure_reason", "")),
                        "residual_norm": float(out.get("residual_norm", np.nan)),
                        "iterations": int(out.get("iterations", -1)),
                        "setup_runtime_sec": float(out.get("setup_runtime", np.nan)),
                        "solve_runtime_sec": float(out.get("solve_runtime", np.nan)),
                        "runtime_sec": float(out.get("runtime", np.nan)),
                        "select_sec": float(timing["select_sec"]),
                        "loss_eval_sec": float(timing["loss_eval_sec"]),
                        "update_sec": float(timing["update_sec"]),
                        "overhead_sec": float(timing["overhead_sec"]),
                        "end_to_end_sec": float(out.get("runtime", np.nan) + timing["overhead_sec"]),
                        "final_w": float(out.get("final_w", np.nan)),
                        "final_sweeps_down": int(out.get("final_sweeps_down", -1)),
                        "final_sweeps_up": int(out.get("final_sweeps_up", -1)),
                        "params": dict(params),
                        "fixed_params": dict(branch_meta.get(label, {}).get("fixed_params", {})),
                        "problem": {
                            "nx": int(mkw["nx"]),
                            "ny": int(mkw["ny"]),
                            "nz": int(mkw["nz"]),
                            "k": float(mkw["k"]),
                            "c": float(mkw["c"]),
                            "a0": float(mkw["a0"]),
                        },
                    }
                )

        progress_bar(
            local_t + 1,
            len(instances),
            prefix=str(phase_label),
            every=progress_every,
            start_time=phase_start_time,
        )

    metrics["failure_records"] = failure_records
    metrics["branch_meta"] = branch_meta
    metrics["labels"] = labels
    metrics["branches"] = {label: branch_by_label[label] for label in labels}
    return metrics
