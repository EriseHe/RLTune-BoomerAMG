"""
Continue a single setup-phase run:
1) tune 3 params first
2) keep learned model state and continue tuning 5 params

Outputs:
1) Per-phase runtime/trace plots (phase 1 and phase 2)
2) Combined runtime/trace plots over both phases
3) Side-by-side phase comparison plots
"""

import json
import os
import sys
import time
import csv
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
from learners._candidate_subset import CandidateSelector
from solver import solve
from utils.scalar_anisotropic_diffusion import (
    SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM,
    stencil_0_scalar_anisotropic_diffusion_rl,
)
from utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
from utils.setup_amg import (
    build_actions_th_mxrs_tr,
    init_param_trace,
    progress_bar,
    record_param_trace,
)


T_DEFAULT = int(os.environ.get("T", "1000"))
T_STAGE3 = int(os.environ.get("T_STAGE3", str(T_DEFAULT)))
T_STAGE5 = int(os.environ.get("T_STAGE5", str(T_DEFAULT)))
if T_STAGE3 <= 0 or T_STAGE5 <= 0:
    raise ValueError("T_STAGE3 and T_STAGE5 must be positive")
T_TOTAL = int(T_STAGE3 + T_STAGE5)

PROGRESS_EVERY = int(os.environ.get("PROGRESS_EVERY", "0"))

SEED = int(os.environ.get("SEED", "39393939"))

ALPHA = float(os.environ.get("ALPHA", "1.0"))
L2 = float(os.environ.get("L2", "1.0"))
SIGMA = float(os.environ.get("SIGMA", "0.1"))

# Fixed-size comparison (new requirement)
FIXED_N = int(os.environ.get("FIXED_N", "75"))

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

DEFAULT_P_MAX_ELMTS_VALUES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
DEFAULT_AGG_NUM_LEVELS_VALUES = (0, 1, 2, 3, 4, 5)
TRACE_KEYS_3 = ("strong_threshold", "max_row_sum", "trunc_factor")
TRACE_KEYS_5 = ("strong_threshold", "max_row_sum", "trunc_factor", "P_max_elmts", "agg_num_levels")


def runtime_loss_sec(*, outcome, fail_runtime_sec: float, **_) -> float:
    rt = float(outcome["runtime"])
    if not np.isfinite(rt):
        return float(fail_runtime_sec)
    if bool(outcome.get("failed", False)) or rt >= 0.999 * float(fail_runtime_sec):
        return rt + 10.0
    return rt


class FixedPolicy:
    def __init__(self, params):
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


def _same_action(a: dict, b: dict) -> bool:
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


def _build_grids() -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray]:
    grid_n = int(os.environ.get("GRID_N", "20"))
    grid_max = float(os.environ.get("GRID_MAX", "0.95"))
    th_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, grid_n)
    return grid_n, grid_max, th_grid, mxrs_grid, tr_grid


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
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 11003, T=T_TOTAL, trial=0),
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
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 12003, T=T_TOTAL, trial=0),
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
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 13003, T=T_TOTAL, trial=0),
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
            ).new_trial(parameter_space=parameter_space, seed=seed_base + 15003, T=T_TOTAL, trial=0),
        ),
    ]


def _generate_instances(*, T: int, seed: int, sampler_kwargs: Dict[str, Any]):
    rng = np.random.default_rng(seed)
    instances = []
    for t in range(int(T)):
        mkw, context, _meta = stencil_0_scalar_anisotropic_diffusion_rl(rng=rng, t=t, trial=0, **sampler_kwargs)
        instances.append((mkw, np.asarray(context, dtype=float)))
    return instances


def _ensure_default_arm(actions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    default_arm_index = next((i for i, a in enumerate(actions) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        return [*actions, dict(DEFAULT_PARAMS)], int(len(actions))
    return actions, int(default_arm_index)


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


def _update_model_action_space(model: Any, *, actions: List[Dict[str, Any]], default_arm_index: int) -> None:
    model.actions = [dict(a) for a in actions]
    model.K = int(len(model.actions))
    model._g_actions = np.zeros((model.K, model.g_dim), dtype=float)
    for i, a in enumerate(model.actions):
        model._g_actions[i] = model._g_from_action(a)

    if hasattr(model, "_last_phi"):
        model._last_phi = None
    if hasattr(model, "_last_arm"):
        model._last_arm = None

    if isinstance(model, SharedLinUCB_AMG_v2):
        model._always_include_arms = np.asarray([int(default_arm_index)], dtype=int)
        model._elite_arms = np.zeros(0, dtype=int)
        model._arm_best_loss = np.full(model.K, np.inf, dtype=float) if model.elite_cache_size > 0 else None
        return

    if isinstance(model, (SharedLinUCB_AMG_v3, SharedLinTS_AMG, Bayesianbandits_AMG_v2)):
        old_cand = getattr(model, "_cand", None)
        model._cand = CandidateSelector(
            model.K,
            candidate_pool_size=getattr(old_cand, "candidate_pool_size", None),
            always_include_arms=[int(default_arm_index)],
            elite_cache_size=int(getattr(old_cand, "elite_cache_size", 0)),
            rng=model.rng,
        )
        return

    raise TypeError(f"Unsupported policy model type for action-space update: {type(model)!r}")


def _switch_methods_to_action_space(
    *,
    methods: Sequence[Tuple[str, Any]],
    actions: List[Dict[str, Any]],
    default_arm_index: int,
) -> None:
    for name, policy in methods:
        if isinstance(policy, FixedPolicy):
            continue
        model = getattr(policy, "m", None)
        if model is None:
            raise TypeError(f"Method {name!r} does not expose model state as .m")
        _update_model_action_space(model, actions=actions, default_arm_index=int(default_arm_index))


def _run_phase(
    *,
    phase_label: str,
    methods: Sequence[Tuple[str, Any]],
    parameter_space: Dict[str, Any],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    global_offset: int,
    rt_total: Dict[str, np.ndarray],
    overhead_total: Dict[str, np.ndarray],
    traces_total: Dict[str, Dict[str, np.ndarray]],
    failed_count: Dict[str, int],
    prev_update_est: Dict[str, float],
    rng_order: np.random.Generator,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    fail_runtime_sec = 1e9
    phase_T = int(len(instances))
    rt_phase = {name: np.zeros(phase_T, dtype=float) for name, _ in methods}
    overhead_phase = {name: np.zeros(phase_T, dtype=float) for name, _ in methods}

    for local_t in range(phase_T):
        mkw, context = instances[local_t]
        order = rng_order.permutation(len(methods))

        for i in order:
            name, policy = methods[int(i)]
            global_t = int(global_offset + local_t)

            sel_start = time.perf_counter_ns()
            selected = policy.select(context=context, parameter_space=parameter_space)
            sel_sec = (time.perf_counter_ns() - sel_start) / 1e9
            params = selected[0] if isinstance(selected, tuple) else selected

            out = _safe_solve(params, mkw, fail_runtime_sec=fail_runtime_sec)
            rt_total[name][global_t] = float(out["runtime"])
            rt_phase[name][local_t] = float(out["runtime"])
            failed_count[name] += int(bool(out.get("failed", False)))

            if name in traces_total:
                record_param_trace(traces_total[name], t=global_t, params=params, keys=TRACE_KEYS_5)

            loss_start = time.perf_counter_ns()
            base_loss_sec = float(runtime_loss_sec(outcome=out, fail_runtime_sec=fail_runtime_sec))
            loss_eval_sec = (time.perf_counter_ns() - loss_start) / 1e9
            end_to_end_loss_sec = base_loss_sec + float(sel_sec) + float(loss_eval_sec) + float(prev_update_est[name])

            upd_sec = 0.0
            if hasattr(policy, "update"):
                upd_start = time.perf_counter_ns()
                policy.update(loss=end_to_end_loss_sec, context=context, params=params, outcome=out)
                upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

            prev_update_est[name] = float(upd_sec)
            step_overhead = float(sel_sec + loss_eval_sec + upd_sec)
            overhead_total[name][global_t] = step_overhead
            overhead_phase[name][local_t] = step_overhead

        progress_bar(local_t + 1, phase_T, prefix=phase_label, every=_progress_every())

    return rt_phase, overhead_phase


def _slice_traces(
    *,
    traces: Dict[str, Dict[str, np.ndarray]],
    keys: Sequence[str],
    start: int,
    length: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    s0 = int(start)
    s1 = int(start + length)
    for name, trace in traces.items():
        trace_out: Dict[str, np.ndarray] = {}
        for k in keys:
            if k in trace:
                trace_out[str(k)] = np.asarray(trace[k][s0:s1], dtype=float).copy()
        if trace_out:
            out[name] = trace_out
    return out


def _save_per_instance_csv(
    *,
    out_csv: Path,
    method_names: Sequence[str],
    runtime_sec: Dict[str, np.ndarray],
    overhead_sec: Dict[str, np.ndarray],
    traces: Dict[str, Dict[str, np.ndarray]],
    t_stage3: int,
    t_total: int,
) -> None:
    fieldnames = [
        "phase",
        "t",
        "t_phase",
        "method",
        "runtime_sec",
        "overhead_sec",
        "end_to_end_sec",
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

        for t in range(int(t_total)):
            phase = "tune3" if t < int(t_stage3) else "tune5_continued"
            t_phase = (t + 1) if t < int(t_stage3) else (t - int(t_stage3) + 1)

            for name in method_names:
                rt = float(runtime_sec[name][t])
                ov = float(overhead_sec[name][t])
                trace = traces.get(name, {})

                def _param_value(key: str) -> float:
                    arr = trace.get(key)
                    if arr is None:
                        return float(DEFAULT_PARAMS[key])
                    val = float(arr[t])
                    return val if np.isfinite(val) else float(DEFAULT_PARAMS[key])

                writer.writerow(
                    {
                        "phase": phase,
                        "t": int(t + 1),
                        "t_phase": int(t_phase),
                        "method": str(name),
                        "runtime_sec": rt,
                        "overhead_sec": ov,
                        "end_to_end_sec": float(rt + ov),
                        "strong_threshold": _param_value("strong_threshold"),
                        "max_row_sum": _param_value("max_row_sum"),
                        "trunc_factor": _param_value("trunc_factor"),
                        "P_max_elmts": _param_value("P_max_elmts"),
                        "agg_num_levels": _param_value("agg_num_levels"),
                    }
                )


def _stitch_images_side_by_side(
    *,
    left_image: Path,
    right_image: Path,
    out_image: Path,
    left_title: str,
    right_title: str,
    suptitle: str,
) -> None:
    import matplotlib.pyplot as plt

    left = plt.imread(left_image)
    right = plt.imread(right_image)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), constrained_layout=True)
    for ax, img, title in [
        (axes[0], left, left_title),
        (axes[1], right, right_title),
    ]:
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    fig.suptitle(suptitle, fontsize=14)
    fig.savefig(out_image, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    fail_warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)

    sampler_kwargs = {
        "nx": int(FIXED_N),
        "ny": int(FIXED_N),
        "nz": int(FIXED_N),
        "n_min": int(FIXED_N),
        "n_max": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
    }

    # Warm-up one solve to avoid measuring first-call overhead in the experiments.
    warm_mkw, _, _ = stencil_0_scalar_anisotropic_diffusion_rl(rng=fail_warm_rng, t=0, trial=0, **sampler_kwargs)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    grid_n, _grid_max, th_grid, mxrs_grid, tr_grid = _build_grids()
    actions_tune3 = _build_actions_tune3(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune3, default_arm_index_tune3 = _ensure_default_arm(actions_tune3)
    actions_tune5, p_max_values, agg_nl_values = _build_actions_tune5(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune5, default_arm_index_tune5 = _ensure_default_arm(actions_tune5)

    instances = _generate_instances(T=T_TOTAL, seed=SEED, sampler_kwargs=sampler_kwargs)
    instances_tune3 = instances[:T_STAGE3]
    instances_tune5 = instances[T_STAGE3:]

    parameter_space_tune3 = {"actions": actions_tune3, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM}
    parameter_space_tune5 = {"actions": actions_tune5, "context_dim": SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM}
    methods = _make_methods(
        parameter_space=parameter_space_tune3,
        default_arm_index=int(default_arm_index_tune3),
        seed_base=SEED + 10_000,
    )
    method_names = [name for name, _ in methods]

    rt_total = {name: np.zeros(T_TOTAL, dtype=float) for name in method_names}
    overhead_total = {name: np.zeros(T_TOTAL, dtype=float) for name in method_names}
    failed_count = {name: 0 for name in method_names}
    traces_total = {
        name: init_param_trace(TRACE_KEYS_5, T_TOTAL)
        for name, policy in methods
        if not isinstance(policy, FixedPolicy)
    }
    prev_update_est = {name: 0.0 for name in method_names}
    rng_order = np.random.default_rng((SEED + 10_000) ^ 0xA5A5A5A5)

    run_dir = create_run_output_dir(
        base_dir=plots_base_dir,
        script_name=Path(__file__).stem,
    problem_name="scalar_anisotropic_diffusion",
        size_tag=f"{FIXED_N}x{FIXED_N}x{FIXED_N}",
        T=T_TOTAL,
        seed=SEED,
    )

    print(f"Phase 1/2: tune3 for T_STAGE3={T_STAGE3}")
    rt_tune3, overhead_tune3 = _run_phase(
        phase_label="  phase tune3",
        methods=methods,
        parameter_space=parameter_space_tune3,
        instances=instances_tune3,
        global_offset=0,
        rt_total=rt_total,
        overhead_total=overhead_total,
        traces_total=traces_total,
        failed_count=failed_count,
        prev_update_est=prev_update_est,
        rng_order=rng_order,
    )

    # Continue the same learned policies after expanding the action space.
    _switch_methods_to_action_space(
        methods=methods,
        actions=actions_tune5,
        default_arm_index=int(default_arm_index_tune5),
    )

    print(f"Phase 2/2: tune5 continuation for T_STAGE5={T_STAGE5}")
    rt_tune5, overhead_tune5 = _run_phase(
        phase_label="  phase tune5",
        methods=methods,
        parameter_space=parameter_space_tune5,
        instances=instances_tune5,
        global_offset=T_STAGE3,
        rt_total=rt_total,
        overhead_total=overhead_total,
        traces_total=traces_total,
        failed_count=failed_count,
        prev_update_est=prev_update_est,
        rng_order=rng_order,
    )

    phase3_summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="phase1_tune3_runtime_interleaved_end2end",
        method_names=method_names,
        runtime_sec=rt_tune3,
        overhead_sec=overhead_tune3,
        T=T_STAGE3,
        title=(
        f"BoomerAMG setup cumulative runtime (phase 1 tune3, scalar anisotropic diffusion)  "
            f"T={T_STAGE3}  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=_slice_traces(traces=traces_total, keys=TRACE_KEYS_3, start=0, length=T_STAGE3),
        trace_keys=TRACE_KEYS_3,
        trace_title=f"Bandit parameter traces (phase 1 tune3)  T={T_STAGE3}  seed={SEED}",
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "script": Path(__file__).name,
            "phase": "tune3",
            "continuation": False,
            "seed": int(SEED),
            "alpha": float(ALPHA),
            "l2": float(L2),
            "sigma": float(SIGMA),
            "fixed_n": int(FIXED_N),
            "c_min": float(C_MIN),
            "c_max": float(C_MAX),
    "context_dim": int(SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM),
            "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(ELITE_CACHE_SIZE),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "actions_grid_n": int(grid_n),
            "actions_count": int(len(actions_tune3)),
            "p_max_values": [int(DEFAULT_PARAMS["P_max_elmts"])],
            "agg_num_levels_values": [int(DEFAULT_PARAMS["agg_num_levels"])],
            "failed_count_cumulative": {k: int(v) for k, v in failed_count.items()},
        },
    )

    phase5_summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="phase2_tune5_continued_runtime_interleaved_end2end",
        method_names=method_names,
        runtime_sec=rt_tune5,
        overhead_sec=overhead_tune5,
        T=T_STAGE5,
        title=(
        f"BoomerAMG setup cumulative runtime (phase 2 tune5 continued, scalar anisotropic diffusion)  "
            f"T={T_STAGE5}  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=_slice_traces(traces=traces_total, keys=TRACE_KEYS_5, start=T_STAGE3, length=T_STAGE5),
        trace_keys=TRACE_KEYS_5,
        trace_title=f"Bandit parameter traces (phase 2 tune5 continued)  T={T_STAGE5}  seed={SEED}",
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "script": Path(__file__).name,
            "phase": "tune5_continued",
            "continuation": True,
            "seed": int(SEED),
            "alpha": float(ALPHA),
            "l2": float(L2),
            "sigma": float(SIGMA),
            "fixed_n": int(FIXED_N),
            "c_min": float(C_MIN),
            "c_max": float(C_MAX),
    "context_dim": int(SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM),
            "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(ELITE_CACHE_SIZE),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "actions_grid_n": int(grid_n),
            "actions_count": int(len(actions_tune5)),
            "p_max_values": [int(v) for v in p_max_values],
            "agg_num_levels_values": [int(v) for v in agg_nl_values],
            "failed_count_cumulative": {k: int(v) for k, v in failed_count.items()},
        },
    )

    combined_summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="continuation_tune3_to_tune5_runtime_interleaved_end2end",
        method_names=method_names,
        runtime_sec=rt_total,
        overhead_sec=overhead_total,
        T=T_TOTAL,
        title=(
        f"BoomerAMG setup cumulative runtime (test 7 continuation tune3->tune5, scalar anisotropic diffusion)  "
            f"T_total={T_TOTAL} (phase boundary at {T_STAGE3})  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=traces_total,
        trace_keys=TRACE_KEYS_5,
        trace_title=f"Bandit parameter traces (test 7 continuation)  T_total={T_TOTAL}  seed={SEED}",
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
    "context_dim": int(SCALAR_ANISOTROPIC_DIFFUSION_CONTEXT_DIM),
            "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(ELITE_CACHE_SIZE),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "actions_grid_n": int(grid_n),
            "actions_count_tune3": int(len(actions_tune3)),
            "actions_count_tune5": int(len(actions_tune5)),
            "p_max_values": [int(v) for v in p_max_values],
            "agg_num_levels_values": [int(v) for v in agg_nl_values],
            "t_stage3": int(T_STAGE3),
            "t_stage5": int(T_STAGE5),
            "t_total": int(T_TOTAL),
            "phase_boundary_round": int(T_STAGE3),
            "continuation": True,
            "failed_count_total": {k: int(v) for k, v in failed_count.items()},
        },
    )

    data_csv_path = run_dir / "per_instance_runtime_data.csv"
    _save_per_instance_csv(
        out_csv=data_csv_path,
        method_names=method_names,
        runtime_sec=rt_total,
        overhead_sec=overhead_total,
        traces=traces_total,
        t_stage3=T_STAGE3,
        t_total=T_TOTAL,
    )

    runtime_compare_path = run_dir / "runtime_interleaved_end2end_compare_side_by_side.png"
    _stitch_images_side_by_side(
        left_image=Path(phase3_summary["plot"]),
        right_image=Path(phase5_summary["plot"]),
        out_image=runtime_compare_path,
        left_title="Phase 1: tune 3 params",
        right_title="Phase 2: tune 5 params (continued)",
        suptitle="BoomerAMG setup cumulative runtime by phase (test 7 continuation)",
    )

    trace_compare_path = run_dir / "runtime_interleaved_end2end_param_trace_compare_side_by_side.png"
    _stitch_images_side_by_side(
        left_image=Path(phase3_summary["trace_plot"]),
        right_image=Path(phase5_summary["trace_plot"]),
        out_image=trace_compare_path,
        left_title="Phase 1 traces (3 params)",
        right_title="Phase 2 traces (5 params, continued)",
        suptitle="BoomerAMG setup parameter traces by phase (test 7 continuation)",
    )

    compare_summary = {
        "script": Path(__file__).name,
        "seed": int(SEED),
        "t_stage3": int(T_STAGE3),
        "t_stage5": int(T_STAGE5),
        "t_total": int(T_TOTAL),
        "fixed_n": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
        "continuation": True,
        "runtime_compare_plot": str(runtime_compare_path),
        "trace_compare_plot": str(trace_compare_path),
        "data_csv": str(data_csv_path),
        "phases": {
            "phase1_tune3": {
                "summary_path": str(phase3_summary["summary_path"]),
                "plot": str(phase3_summary["plot"]),
                "trace_plot": str(phase3_summary["trace_plot"]),
                "actions_count": int(len(actions_tune3)),
                "trace_keys": list(phase3_summary["trace_keys"]),
            },
            "phase2_tune5_continued": {
                "summary_path": str(phase5_summary["summary_path"]),
                "plot": str(phase5_summary["plot"]),
                "trace_plot": str(phase5_summary["trace_plot"]),
                "actions_count": int(len(actions_tune5)),
                "trace_keys": list(phase5_summary["trace_keys"]),
                "p_max_values": [int(v) for v in p_max_values],
                "agg_num_levels_values": [int(v) for v in agg_nl_values],
            },
        },
        "combined": {
            "summary_path": str(combined_summary["summary_path"]),
            "plot": str(combined_summary["plot"]),
            "trace_plot": str(combined_summary["trace_plot"]),
            "trace_keys": list(combined_summary["trace_keys"]),
        },
    }
    compare_summary_path = run_dir / "runtime_interleaved_end2end_compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, indent=2) + "\n")

    print("COMPARE RUNTIME PLOT:", runtime_compare_path)
    print("COMPARE TRACE PLOT:", trace_compare_path)
    print("DATA CSV:", data_csv_path)
    print("COMBINED SUMMARY:", combined_summary["summary_path"])
    print("COMPARE SUMMARY:", compare_summary_path)
    print("PHASE1 TUNE3 SUMMARY:", phase3_summary["summary_path"])
    print("PHASE2 TUNE5 CONTINUED SUMMARY:", phase5_summary["summary_path"])


if __name__ == "__main__":
    main()
