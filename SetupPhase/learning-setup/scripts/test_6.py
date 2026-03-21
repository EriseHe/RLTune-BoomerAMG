"""
Compare setup-phase runtime between tuning 3 params vs tuning 5 params.

Outputs:
1) Side-by-side cumulative runtime plot (two experiments)
2) Side-by-side parameter trace plot (two experiments)

Experiments use the same sampled problem instances for fair comparison.
"""

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
from utils.setup_amg import build_actions_th_mxrs_tr, init_param_trace, record_param_trace


T = int(os.environ.get("T", "1000"))
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

DEFAULT_P_MAX_ELMTS_VALUES = (2, 4, 6, 8, 12, 16)
DEFAULT_AGG_NUM_LEVELS_VALUES = (0, 1, 2, 3, 4, 5)


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


def _run_experiment(
    *,
    experiment_key: str,
    experiment_label: str,
    tuned_label: str,
    actions: List[Dict[str, Any]],
    trace_keys: Sequence[str],
    instances: Sequence[Tuple[Dict[str, Any], np.ndarray]],
    run_dir: Path,
    seed_base: int,
    summary_extra: Dict[str, Any],
) -> Dict[str, Any]:
    fail_runtime_sec = 1e9

    default_arm_index = next((i for i, a in enumerate(actions) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        actions = [*actions, dict(DEFAULT_PARAMS)]
        default_arm_index = len(actions) - 1

    parameter_space = {"actions": actions, "context_dim": DIFCONV_CONTEXT_DIM}
    methods = _make_methods(parameter_space=parameter_space, default_arm_index=int(default_arm_index), seed_base=seed_base)

    rt = {name: np.zeros(T, dtype=float) for name, _ in methods}
    overhead = {name: np.zeros(T, dtype=float) for name, _ in methods}
    failed_count = {name: 0 for name, _ in methods}
    traces = {name: init_param_trace(trace_keys, T) for name, _ in methods if not isinstance(_, FixedPolicy)}

    rng_order = np.random.default_rng(seed_base ^ 0xA5A5A5A5)
    prev_update_est = {name: 0.0 for name, _ in methods}

    def safe_solve(params: dict, mkw: dict) -> dict:
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

    for t in range(T):
        mkw, context = instances[t]
        order = rng_order.permutation(len(methods))

        for i in order:
            name, policy = methods[int(i)]

            sel_start = time.perf_counter_ns()
            selected = policy.select(context=context, parameter_space=parameter_space)
            sel_sec = (time.perf_counter_ns() - sel_start) / 1e9
            params = selected[0] if isinstance(selected, tuple) else selected

            out = safe_solve(params, mkw)
            rt[name][t] = float(out["runtime"])
            failed_count[name] += int(bool(out.get("failed", False)))

            if name in traces:
                record_param_trace(traces[name], t=t, params=params, keys=trace_keys)

            loss_start = time.perf_counter_ns()
            base_loss_sec = float(runtime_loss_sec(outcome=out, fail_runtime_sec=fail_runtime_sec))
            loss_eval_sec = (time.perf_counter_ns() - loss_start) / 1e9

            end_to_end_loss_sec = base_loss_sec + float(sel_sec) + float(loss_eval_sec) + float(prev_update_est[name])
            loss_value = end_to_end_loss_sec

            upd_sec = 0.0
            if hasattr(policy, "update"):
                upd_start = time.perf_counter_ns()
                policy.update(loss=loss_value, context=context, params=params, outcome=out)
                upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

            prev_update_est[name] = float(upd_sec)
            overhead[name][t] = float(sel_sec + loss_eval_sec + upd_sec)

    method_names = [name for name, _ in methods]
    summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix=f"{experiment_key}_runtime_interleaved_end2end",
        method_names=method_names,
        runtime_sec=rt,
        overhead_sec=overhead,
        T=T,
        title=(
            f"BoomerAMG Setup runtime cumulative ({experiment_label}, difconv)  "
            f"T={T}  n={FIXED_N}^3  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=traces,
        trace_keys=tuple(trace_keys),
        trace_title=f"Bandit parameter traces ({experiment_label})  T={T}  seed={SEED}",
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "experiment_key": str(experiment_key),
            "experiment_label": str(experiment_label),
            "tuned_param_set": str(tuned_label),
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
            "actions_count": int(len(actions)),
            "failed_count": {k: int(v) for k, v in failed_count.items()},
            **summary_extra,
        },
    )

    summary["runtime_sec"] = rt
    summary["overhead_sec"] = overhead
    summary["traces_obj"] = traces
    summary["method_names_obj"] = method_names
    summary["trace_keys_obj"] = tuple(trace_keys)
    return summary


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
    warm_mkw, _, _ = stencil_0_difconv_rl(rng=fail_warm_rng, t=0, trial=0, **sampler_kwargs)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    grid_n, _grid_max, th_grid, mxrs_grid, tr_grid = _build_grids()
    actions_tune3 = _build_actions_tune3(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)
    actions_tune5, p_max_values, agg_nl_values = _build_actions_tune5(th_grid=th_grid, mxrs_grid=mxrs_grid, tr_grid=tr_grid)

    instances = _generate_instances(T=T, seed=SEED, sampler_kwargs=sampler_kwargs)

    run_dir = create_run_output_dir(
        base_dir=plots_base_dir,
        script_name=Path(__file__).stem,
        problem_name="difconv",
        size_tag=f"{FIXED_N}x{FIXED_N}x{FIXED_N}",
        T=T,
        seed=SEED,
    )

    exp3 = _run_experiment(
        experiment_key="tune3",
        experiment_label="test 6 tune 3 params (th, mxrs, tr)",
        tuned_label="3_params",
        actions=actions_tune3,
        trace_keys=("strong_threshold", "max_row_sum", "trunc_factor"),
        instances=instances,
        run_dir=run_dir,
        seed_base=SEED + 10_000,
        summary_extra={
            "actions_grid_n": int(grid_n),
            "p_max_values": [int(DEFAULT_PARAMS["P_max_elmts"])],
            "agg_num_levels_values": [int(DEFAULT_PARAMS["agg_num_levels"])],
        },
    )
    exp5 = _run_experiment(
        experiment_key="tune5",
        experiment_label="test 6 tune 5 params (th, mxrs, tr, Pmax, agg_nl)",
        tuned_label="5_params",
        actions=actions_tune5,
        trace_keys=("strong_threshold", "max_row_sum", "trunc_factor", "P_max_elmts", "agg_num_levels"),
        instances=instances,
        run_dir=run_dir,
        seed_base=SEED + 20_000,
        summary_extra={
            "actions_grid_n": int(grid_n),
            "p_max_values": [int(v) for v in p_max_values],
            "agg_num_levels_values": [int(v) for v in agg_nl_values],
        },
    )

    runtime_compare_path = run_dir / "runtime_interleaved_end2end_compare_side_by_side.png"
    _stitch_images_side_by_side(
        left_image=Path(exp3["plot"]),
        right_image=Path(exp5["plot"]),
        out_image=runtime_compare_path,
        left_title="Tune 3 params",
        right_title="Tune 5 params",
        suptitle="BoomerAMG setup cumulative end-to-end runtime comparison (test 6)",
    )

    trace_compare_path = run_dir / "runtime_interleaved_end2end_param_trace_compare_side_by_side.png"
    _stitch_images_side_by_side(
        left_image=Path(exp3["trace_plot"]),
        right_image=Path(exp5["trace_plot"]),
        out_image=trace_compare_path,
        left_title="Tune 3 params traces",
        right_title="Tune 5 params traces",
        suptitle="BoomerAMG setup bandit parameter traces comparison (test 6)",
    )

    compare_summary = {
        "script": Path(__file__).name,
        "seed": int(SEED),
        "T": int(T),
        "fixed_n": int(FIXED_N),
        "c_min": float(C_MIN),
        "c_max": float(C_MAX),
        "runtime_compare_plot": str(runtime_compare_path),
        "trace_compare_plot": str(trace_compare_path),
        "experiments": {
            "tune3": {
                "summary_path": str(exp3["summary_path"]),
                "plot": str(exp3["plot"]),
                "trace_plot": str(exp3["trace_plot"]),
                "actions_count": int(exp3["actions_count"]),
                "trace_keys": list(exp3["trace_keys"]),
            },
            "tune5": {
                "summary_path": str(exp5["summary_path"]),
                "plot": str(exp5["plot"]),
                "trace_plot": str(exp5["trace_plot"]),
                "actions_count": int(exp5["actions_count"]),
                "trace_keys": list(exp5["trace_keys"]),
                "p_max_values": [int(v) for v in p_max_values],
                "agg_num_levels_values": [int(v) for v in agg_nl_values],
            },
        },
    }
    compare_summary_path = run_dir / "runtime_interleaved_end2end_compare_summary.json"
    compare_summary_path.write_text(json.dumps(compare_summary, indent=2) + "\n")

    print("COMPARE RUNTIME PLOT:", runtime_compare_path)
    print("COMPARE TRACE PLOT:", trace_compare_path)
    print("COMPARE SUMMARY:", compare_summary_path)
    print("TUNE3 SUMMARY:", exp3["summary_path"])
    print("TUNE5 SUMMARY:", exp5["summary_path"])


if __name__ == "__main__":
    main()