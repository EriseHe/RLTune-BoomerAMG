"""
Test 8: tune3 and tune5 run separately, both interleaved per step.

Protocol
--------
All methods are evaluated on the same instance stream.
At each step, execution order is randomly permuted over all branches:
  (method x tune_set), where tune_set in {tune3, tune5}.
No continuation between tune3 and tune5 branches.

Outputs
-------
1) One cumulative runtime plot across all branches
2) One per-instance CSV export (t, tune_set, runtime_sec, overhead_sec, params)
3) One JSON summary for paths + run metadata
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

# Render plots headlessly by default (safe for local runs too).
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learners.Bayesianbandits_AMG_v2 import Bayesianbandits_AMG_v2
from learners.SharedLinTS_AMG import SharedLinTS_AMG
from learners.SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from learners.SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from solver import solve
from utils.problem_amg import DIFCONV_CONTEXT_DIM, stencil_0_difconv_rl
from utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
from utils.setup_amg import build_actions_th_mxrs_tr, init_param_trace, progress_bar, record_param_trace


T = int(os.environ.get("T", "1000"))
if T <= 0:
    raise ValueError("T must be positive")

PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "0"))

SEED = int(os.environ.get("SEED", "39393939"))

ALPHA = float(os.environ.get("ALPHA", "1.0"))
L2 = float(os.environ.get("L2", "1.0"))
SIGMA = float(os.environ.get("SIGMA", "0.1"))

FIXED_N = int(os.environ.get("FIXED_N", "100"))
C_MIN = float(os.environ.get("C_MIN", "1.0"))
C_MAX = float(os.environ.get("C_MAX", "1000.0"))

CANDIDATE_POOL_SIZE = int(os.environ.get("CANDIDATE_POOL_SIZE", "512"))
ELITE_CACHE_SIZE = int(os.environ.get("ELITE_CACHE_SIZE", "64"))

SOLVER_TOL = float(os.environ.get("SOLVER_TOL", "1e-8"))
SOLVER_MAX_ITER = int(os.environ.get("SOLVER_MAX_ITER", "10000"))

plots_base_dir = Path(__file__).resolve().parent.parent / "plots" / "Combined"
plots_base_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_PARAMS = {
    "strong_threshold": 0.25,
    "max_row_sum": 0.90,
    "trunc_factor": 0.00,
    "coarsen_type": 10,
    "interp_type": 6,
    "P_max_elmts": 4,
    "agg_num_levels": 0,
}
DEFAULT_P_MAX_ELMTS_VALUES = (2, 4, 6, 8, 12, 16)
DEFAULT_AGG_NUM_LEVELS_VALUES = (0, 1, 2, 3, 4, 5)
TRACE_KEYS_5 = ("strong_threshold", "max_row_sum", "trunc_factor", "P_max_elmts", "agg_num_levels")


def runtime_loss_sec(*, outcome: Dict[str, Any], fail_runtime_sec: float, **_) -> float:
    rt = float(outcome["runtime"])
    if not np.isfinite(rt):
        return float(fail_runtime_sec)
    if bool(outcome.get("failed", False)) or rt >= 0.999 * float(fail_runtime_sec):
        return rt + 10.0
    return rt


class FixedPolicy:
    def __init__(self, params: Dict[str, Any]):
        self._params = dict(params)

    def select(self, context, **_):
        return dict(self._params), {}

    def update(self, loss, **_):
        return None


class SharedFactory:
    def __init__(self, model_cls, alpha: float, l2: float, *, model_kwargs=None):
        self.model_cls = model_cls
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.model_kwargs = dict(model_kwargs or {})

    def new_trial(self, *, parameter_space, seed: int, **_):
        class _Policy:
            def __init__(self, actions, context_dim, alpha, l2, seed):
                self.m = model_cls(
                    actions,
                    context_dim=int(context_dim),
                    alpha=float(alpha),
                    l2_reg=float(l2),
                    seed=int(seed),
                    **model_kwargs,
                )

            def select(self, context, **_):
                params = self.m.predict(context)
                if getattr(self.m, "history", None):
                    step = self.m.history[-1]
                    info = {
                        "pred_mean": float(getattr(step, "pred_mean", np.nan)),
                        "pred_uncert": float(getattr(step, "pred_uncert", np.nan)),
                    }
                else:
                    info = {}
                return params, info

            def update(self, loss, **_):
                self.m.update(float(loss))

        model_cls = self.model_cls
        model_kwargs = self.model_kwargs
        return _Policy(
            parameter_space["actions"],
            parameter_space["context_dim"],
            self.alpha,
            self.l2,
            seed,
        )


def _same_action(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return (
        np.isclose(float(a["strong_threshold"]), float(b["strong_threshold"]), rtol=0.0, atol=1e-12)
        and np.isclose(float(a["max_row_sum"]), float(b["max_row_sum"]), rtol=0.0, atol=1e-12)
        and np.isclose(float(a["trunc_factor"]), float(b["trunc_factor"]), rtol=0.0, atol=1e-12)
        and int(a["coarsen_type"]) == int(b["coarsen_type"])
        and int(a["interp_type"]) == int(b["interp_type"])
        and int(a["P_max_elmts"]) == int(b["P_max_elmts"])
        and int(a["agg_num_levels"]) == int(b["agg_num_levels"])
    )


def _parse_int_list_env(name: str, default_values: Sequence[int]) -> List[int]:
    raw = os.environ.get(name, ",".join(str(v) for v in default_values))
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    return vals or [int(v) for v in default_values]


def _build_grids() -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    grid_n = int(os.environ.get("GRID_N", "20"))
    grid_max = float(os.environ.get("GRID_MAX", "0.95"))
    th_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, grid_n)
    return grid_n, th_grid, mxrs_grid, tr_grid


def _build_actions_tune3(*, th_grid, mxrs_grid, tr_grid) -> List[Dict[str, Any]]:
    base_actions = build_actions_th_mxrs_tr(
        th_grid,
        mxrs_grid,
        tr_grid,
        fixed_params={
            "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
            "interp_type": DEFAULT_PARAMS["interp_type"],
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        params = dict(base)
        params["P_max_elmts"] = int(DEFAULT_PARAMS["P_max_elmts"])
        params["agg_num_levels"] = int(DEFAULT_PARAMS["agg_num_levels"])
        actions.append(params)
    return actions


def _build_actions_tune5(*, th_grid, mxrs_grid, tr_grid) -> Tuple[List[Dict[str, Any]], List[int], List[int]]:
    p_max_values = _parse_int_list_env("P_MAX_ELMTS_VALUES", DEFAULT_P_MAX_ELMTS_VALUES)
    agg_nl_values = _parse_int_list_env("AGG_NUM_LEVELS_VALUES", DEFAULT_AGG_NUM_LEVELS_VALUES)

    base_actions = build_actions_th_mxrs_tr(
        th_grid,
        mxrs_grid,
        tr_grid,
        fixed_params={
            "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
            "interp_type": DEFAULT_PARAMS["interp_type"],
        },
    )
    actions: List[Dict[str, Any]] = []
    for base in base_actions:
        for p_max in p_max_values:
            for agg_nl in agg_nl_values:
                params = dict(base)
                params["P_max_elmts"] = int(p_max)
                params["agg_num_levels"] = int(agg_nl)
                actions.append(params)
    return actions, p_max_values, agg_nl_values


def _ensure_default_arm(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    default_arm_index = next((i for i, a in enumerate(actions) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        return [*actions, dict(DEFAULT_PARAMS)], int(len(actions))
    return actions, int(default_arm_index)


def _make_methods(*, parameter_space: Dict[str, Any], default_arm_index: int, seed_base: int):
    return [
        ("default (fixed)", FixedPolicy(DEFAULT_PARAMS)),
        (
            "Shared LinUCB v2",
            SharedFactory(
                SharedLinUCB_AMG_v2,
                ALPHA,
                L2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 11003, T=T, trial=0),
        ),
        (
            "Shared LinUCB v3",
            SharedFactory(
                SharedLinUCB_AMG_v3,
                ALPHA,
                L2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 12003, T=T, trial=0),
        ),
        (
            "Shared LinTS",
            SharedFactory(
                SharedLinTS_AMG,
                ALPHA,
                L2,
                model_kwargs={
                    "sigma": SIGMA,
                    "action_center": DEFAULT_PARAMS,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 13003, T=T, trial=0),
        ),
        (
            "Bayesianbandits v2",
            SharedFactory(
                Bayesianbandits_AMG_v2,
                ALPHA,
                L2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 15003, T=T, trial=0),
        ),
    ]


def _generate_instances(*, T: int, seed: int, sampler_kwargs: Dict[str, Any]):
    rng = np.random.default_rng(seed)
    instances = []
    for t in range(int(T)):
        mkw, context, _meta = stencil_0_difconv_rl(rng=rng, t=t, trial=0, **sampler_kwargs)
        instances.append((mkw, np.asarray(context, dtype=float)))
    return instances


def _progress_every() -> int | None:
    if PROGRESS_EVERY <= 0:
        return None
    return int(PROGRESS_EVERY)


def _safe_solve(params: Dict[str, Any], mkw: Dict[str, Any], *, fail_runtime_sec: float) -> Dict[str, Any]:
    try:
        res = solve(params=params, tol=SOLVER_TOL, max_iter=SOLVER_MAX_ITER, **mkw)
        res_norm = float(res.residual_norm)
        iters = int(res.iterations)
        converged = bool(np.isfinite(res_norm) and res_norm <= float(SOLVER_TOL) and iters < int(SOLVER_MAX_ITER))
        return {
            "runtime": float(res.runtime_sec),
            "failed": (not converged),
            "residual_norm": res_norm,
            "iterations": iters,
        }
    except Exception:
        return {
            "runtime": float(fail_runtime_sec),
            "failed": True,
            "residual_norm": float("inf"),
            "iterations": int(SOLVER_MAX_ITER),
        }


def _run_one_step(
    *,
    policy: Any,
    parameter_space: Dict[str, Any],
    context: np.ndarray,
    mkw: Dict[str, Any],
    prev_update_est: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    fail_runtime_sec = 1e9

    sel_start = time.perf_counter_ns()
    selected = policy.select(context=context, parameter_space=parameter_space)
    sel_sec = (time.perf_counter_ns() - sel_start) / 1e9
    params = selected[0] if isinstance(selected, tuple) else selected

    out = _safe_solve(params, mkw, fail_runtime_sec=fail_runtime_sec)

    loss_start = time.perf_counter_ns()
    base_loss_sec = float(runtime_loss_sec(outcome=out, fail_runtime_sec=fail_runtime_sec))
    loss_eval_sec = (time.perf_counter_ns() - loss_start) / 1e9
    end_to_end_loss_sec = base_loss_sec + float(sel_sec) + float(loss_eval_sec) + float(prev_update_est)

    upd_sec = 0.0
    if hasattr(policy, "update"):
        upd_start = time.perf_counter_ns()
        policy.update(loss=end_to_end_loss_sec, context=context, params=params, outcome=out)
        upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

    step_overhead = float(sel_sec + loss_eval_sec + upd_sec)
    return params, out, step_overhead, float(upd_sec)


def _run_phase_dual_interleaved(
    *,
    phase_label: str,
    branch_entries: Sequence[Tuple[str, str, str, Any, Dict[str, Any]]],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    failed_flags: Dict[str, np.ndarray],
    traces: Dict[str, Dict[str, np.ndarray]],
    prev_update_est: Dict[str, float],
    rng_order: np.random.Generator,
) -> None:
    for local_t, (mkw, context) in enumerate(instances):
        order = rng_order.permutation(len(branch_entries))
        for i in order:
            label, _base_name, _tune_set, policy, parameter_space = branch_entries[int(i)]
            params, out, step_overhead, upd_sec = _run_one_step(
                policy=policy,
                parameter_space=parameter_space,
                context=context,
                mkw=mkw,
                prev_update_est=float(prev_update_est[label]),
            )
            runtime_sec[label][local_t] = float(out["runtime"])
            overhead_sec[label][local_t] = float(step_overhead)
            failed_flags[label][local_t] = bool(out.get("failed", False))
            if label in traces:
                record_param_trace(traces[label], t=local_t, params=params, keys=TRACE_KEYS_5)
            prev_update_est[label] = float(upd_sec)

        progress_bar(local_t + 1, len(instances), prefix=phase_label, every=_progress_every())


def _save_dual_csv(
    *,
    out_csv: Path,
    labels: Sequence[str],
    label_meta: Dict[str, Dict[str, str]],
    runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    failed: Dict[str, np.ndarray],
    traces: Dict[str, Dict[str, np.ndarray]],
    t_total: int,
) -> None:
    fieldnames = [
        "phase",
        "t",
        "t_phase",
        "method",
        "tune_set",
        "runtime_sec",
        "overhead_sec",
        "end_to_end_sec",
        "failed",
        "strong_threshold",
        "max_row_sum",
        "trunc_factor",
        "P_max_elmts",
        "agg_num_levels",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for local_t in range(int(t_total)):
            global_t = int(local_t + 1)
            for label in labels:
                trace = traces.get(label, {})

                def _param_value(key: str) -> float:
                    arr = trace.get(key)
                    if arr is None:
                        return float(DEFAULT_PARAMS[key])
                    val = float(arr[local_t])
                    return val if np.isfinite(val) else float(DEFAULT_PARAMS[key])

                rt = float(runtime_sec[label][local_t])
                ov = float(overhead_sec[label][local_t])
                writer.writerow(
                    {
                        "phase": "dual_permuted",
                        "t": global_t,
                        "t_phase": int(local_t + 1),
                        "method": str(label_meta[label]["method"]),
                        "tune_set": str(label_meta[label]["tune_set"]),
                        "runtime_sec": rt,
                        "overhead_sec": ov,
                        "end_to_end_sec": float(rt + ov),
                        "failed": int(bool(failed[label][local_t])),
                        "strong_threshold": _param_value("strong_threshold"),
                        "max_row_sum": _param_value("max_row_sum"),
                        "trunc_factor": _param_value("trunc_factor"),
                        "P_max_elmts": _param_value("P_max_elmts"),
                        "agg_num_levels": _param_value("agg_num_levels"),
                    }
                )


def main() -> None:
    sampler_kwargs = {
        "nx": int(FIXED_N),
        "ny": int(FIXED_N),
        "nz": int(FIXED_N),
        "n_min": int(FIXED_N),
        "n_max": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
    }

    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_0_difconv_rl(rng=warm_rng, t=0, trial=0, **sampler_kwargs)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    grid_n, th_grid, mxrs_grid, tr_grid = _build_grids()
    actions_tune3 = _build_actions_tune3(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune3, default_arm_index_tune3 = _ensure_default_arm(actions_tune3)
    actions_tune5, p_max_values, agg_nl_values = _build_actions_tune5(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune5, default_arm_index_tune5 = _ensure_default_arm(actions_tune5)

    instances = _generate_instances(T=T, seed=SEED, sampler_kwargs=sampler_kwargs)

    parameter_space_tune3 = {"actions": actions_tune3, "context_dim": DIFCONV_CONTEXT_DIM}
    parameter_space_tune5 = {"actions": actions_tune5, "context_dim": DIFCONV_CONTEXT_DIM}

    methods_tune3 = _make_methods(
        parameter_space=parameter_space_tune3,
        default_arm_index=int(default_arm_index_tune3),
        seed_base=SEED + 10_000,
    )
    methods_tune5 = _make_methods(
        parameter_space=parameter_space_tune5,
        default_arm_index=int(default_arm_index_tune5),
        seed_base=SEED + 20_000,
    )
    method_names = [name for name, _ in methods_tune3]
    if method_names != [name for name, _ in methods_tune5]:
        raise RuntimeError("Method lists for tune3 and tune5 do not match")

    run_dir = create_run_output_dir(
        base_dir=plots_base_dir,
        script_name=Path(__file__).stem,
        problem_name="difconv",
        size_tag=f"{FIXED_N}x{FIXED_N}x{FIXED_N}",
        T=T,
        seed=SEED,
    )

    branch_entries: List[Tuple[str, str, str, Any, Dict[str, Any]]] = []
    label_meta: Dict[str, Dict[str, str]] = {}
    for name, policy in methods_tune3:
        if name == "default (fixed)":
            label = "default (fixed)"
            branch_entries.append((label, name, "default", policy, parameter_space_tune3))
            label_meta[label] = {"method": str(name), "tune_set": "default"}
            continue
        label = f"{name} | tune3"
        branch_entries.append((label, name, "tune3", policy, parameter_space_tune3))
        label_meta[label] = {"method": str(name), "tune_set": "tune3"}
    for name, policy in methods_tune5:
        if name == "default (fixed)":
            continue
        label = f"{name} | tune5"
        branch_entries.append((label, name, "tune5", policy, parameter_space_tune5))
        label_meta[label] = {"method": str(name), "tune_set": "tune5"}
    branch_labels = [entry[0] for entry in branch_entries]

    permutation_seed = int(SEED ^ 0x1A2B3C4D)
    rng_order = np.random.default_rng(permutation_seed)
    print("Within-step permutation over all branches: enabled")

    runtime_sec = {label: np.zeros(T, dtype=float) for label in branch_labels}
    overhead_sec = {label: np.zeros(T, dtype=float) for label in branch_labels}
    failed_flags = {label: np.zeros(T, dtype=bool) for label in branch_labels}
    traces = {
        label: init_param_trace(TRACE_KEYS_5, T)
        for label, _base, _set, policy, _ps in branch_entries
        if not isinstance(policy, FixedPolicy)
    }
    prev_update_est = {label: 0.0 for label in branch_labels}

    print(f"Single run: tune3 + tune5 separate branches, T={T}")
    _run_phase_dual_interleaved(
        phase_label="  dual run",
        branch_entries=branch_entries,
        instances=instances,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        failed_flags=failed_flags,
        traces=traces,
        prev_update_est=prev_update_est,
        rng_order=rng_order,
    )

    summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="dual_tune3_tune5_interleaved_permuted_runtime",
        method_names=branch_labels,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        T=T,
        title=(
            f"BoomerAMG setup cumulative runtime (test 8 tune3/tune5 separate, fully permuted)  "
            f"T={T}  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=traces,
        trace_keys=(),
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "script": Path(__file__).name,
            "seed": int(SEED),
            "alpha": float(ALPHA),
            "l2": float(L2),
            "sigma": float(SIGMA),
            "fixed_n": int(FIXED_N),
            "c_min": float(C_MIN),
            "c_max": float(C_MAX),
            "context_dim": int(DIFCONV_CONTEXT_DIM),
            "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(ELITE_CACHE_SIZE),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "actions_grid_n": int(grid_n),
            "actions_count_tune3": int(len(actions_tune3)),
            "actions_count_tune5": int(len(actions_tune5)),
            "p_max_values": [int(v) for v in p_max_values],
            "agg_num_levels_values": [int(v) for v in agg_nl_values],
            "T": int(T),
            "continuation": False,
            "bias_mitigation": "within-step permutation on identical instance stream across method+tune_set branches",
            "within_step_method_permutation": True,
            "within_step_tune_set_permutation": True,
            "permutation_seed": permutation_seed,
            "failed_count_total": {k: int(np.sum(v.astype(int))) for k, v in failed_flags.items()},
        },
    )

    data_csv_path = run_dir / "per_instance_runtime_data.csv"
    _save_dual_csv(
        out_csv=data_csv_path,
        labels=branch_labels,
        label_meta=label_meta,
        runtime_sec=runtime_sec,
        overhead_sec=overhead_sec,
        failed=failed_flags,
        traces=traces,
        t_total=T,
    )

    bundle_summary = {
        "script": Path(__file__).name,
        "seed": int(SEED),
        "T": int(T),
        "fixed_n": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
        "continuation": False,
        "bias_mitigation": "within-step permutation on identical instance stream across method+tune_set branches",
        "within_step_method_permutation": True,
        "within_step_tune_set_permutation": True,
        "permutation_seed": permutation_seed,
        "runtime_plot": str(summary["plot"]),
        "runtime_summary": str(summary["summary_path"]),
        "data_csv": str(data_csv_path),
    }
    bundle_summary_path = run_dir / "test_8_export_summary.json"
    bundle_summary_path.write_text(json.dumps(bundle_summary, indent=2) + "\n")

    print("RUNTIME PLOT:", summary["plot"])
    print("RUNTIME SUMMARY:", summary["summary_path"])
    print("DATA CSV:", data_csv_path)
    print("EXPORT SUMMARY:", bundle_summary_path)


if __name__ == "__main__":
    main()
