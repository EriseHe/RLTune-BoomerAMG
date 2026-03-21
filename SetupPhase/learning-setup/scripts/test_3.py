import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learners.SharedLinTS_AMG import SharedLinTS_AMG
from learners.SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from learners.SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from solver import solve
from utils.problem_amg import DIFCONV_CONTEXT_DIM, stencil_0_difconv_rl
from utils.plotting_amg import create_run_output_dir, save_runtime_artifacts
from utils.setup_amg import (
    build_actions_th_mxrs_tr,
    init_param_trace,
    record_param_trace,
)


T = int(os.environ.get("T", "5000"))
SEED = int(os.environ.get("SEED", "20260215"))

ALPHA = float(os.environ.get("ALPHA", "1.0"))
L2 = float(os.environ.get("L2", "1.0"))
SIGMA = float(os.environ.get("SIGMA", "0.1"))

N_MIN = int(os.environ.get("N_MIN", "10"))
N_MAX = int(os.environ.get("N_MAX", "100"))
C_MIN = float(os.environ.get("C_MIN", "1.0"))
C_MAX = float(os.environ.get("C_MAX", "1000.0"))

FIXED_N = int(os.environ.get("FIXED_N", "0"))

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
}


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
    )


def main() -> None:
    fail_runtime_sec = 1e9

    grid_n = int(os.environ.get("GRID_N", "20"))
    grid_max = float(os.environ.get("GRID_MAX", "0.95"))
    th_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid = np.linspace(0.0, grid_max, grid_n)
    mxrs_grid[0] = 1e-6
    tr_grid = np.linspace(0.0, grid_max, grid_n)

    actions = build_actions_th_mxrs_tr(
        th_grid,
        mxrs_grid,
        tr_grid,
        fixed_params={"coarsen_type": DEFAULT_PARAMS["coarsen_type"], "interp_type": DEFAULT_PARAMS["interp_type"]},
    )
    default_arm_index = next((i for i, a in enumerate(actions) if _same_action(a, DEFAULT_PARAMS)), None)
    if default_arm_index is None:
        actions.append(dict(DEFAULT_PARAMS))
        default_arm_index = len(actions) - 1

    parameter_space = {"actions": actions, "context_dim": DIFCONV_CONTEXT_DIM}

    sampler_kwargs = dict(n_min=N_MIN, n_max=N_MAX, c_min=C_MIN, c_max=C_MAX)
    if FIXED_N > 0:
        sampler_kwargs.update(
            {
                "nx": int(FIXED_N),
                "ny": int(FIXED_N),
                "nz": int(FIXED_N),
                "n_min": int(FIXED_N),
                "n_max": int(FIXED_N),
            }
        )

    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_0_difconv_rl(rng=warm_rng, **sampler_kwargs)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    methods = [
        ("default (fixed)", FixedPolicy(DEFAULT_PARAMS)),
        (
            f"Shared LinUCB v2: {grid_n}^3",
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
            ).new_trial(parameter_space=parameter_space, seed=SEED + 11003, T=T, trial=0),
        ),
        (
            f"Shared LinUCB v3: {grid_n}^3",
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
            ).new_trial(parameter_space=parameter_space, seed=SEED + 12003, T=T, trial=0),
        ),
        (
            f"Shared LinTS: {grid_n}^3",
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
            ).new_trial(parameter_space=parameter_space, seed=SEED + 13003, T=T, trial=0),
        ),
    ]

    rt = {name: np.zeros(T, dtype=float) for name, _ in methods}
    overhead = {name: np.zeros(T, dtype=float) for name, _ in methods}
    failed_count = {name: 0 for name, _ in methods}
    trace_keys = ("strong_threshold", "max_row_sum", "trunc_factor")
    traces = {name: init_param_trace(trace_keys, T) for name, _ in methods if not isinstance(_, FixedPolicy)}

    rng_inst = np.random.default_rng(SEED)
    rng_order = np.random.default_rng(SEED ^ 0xA5A5A5A5)
    prev_update_est = {name: 0.0 for name, _ in methods}

    def safe_solve(params: dict, mkw: dict) -> dict:
        try:
            res = solve(params=params, tol=SOLVER_TOL, max_iter=SOLVER_MAX_ITER, **mkw)
            res_norm = float(res.residual_norm)
            iters = int(res.iterations)
            converged = bool(np.isfinite(res_norm) and res_norm <= float(SOLVER_TOL) and iters < int(SOLVER_MAX_ITER))
            return {"runtime": float(res.runtime_sec), "failed": (not converged), "residual_norm": res_norm, "iterations": iters}
        except Exception:
            return {
                "runtime": float(fail_runtime_sec),
                "failed": True,
                "residual_norm": float("inf"),
                "iterations": int(SOLVER_MAX_ITER),
            }

    for t in range(T):
        mkw, context, _meta = stencil_0_difconv_rl(rng=rng_inst, **sampler_kwargs)
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

    n_desc = f"{FIXED_N}^3" if FIXED_N > 0 else f"n={N_MIN}..{N_MAX} (cube)"
    size_tag = f"{FIXED_N}x{FIXED_N}x{FIXED_N}" if FIXED_N > 0 else f"n{N_MIN}-{N_MAX}-cube"
    run_dir = create_run_output_dir(
        base_dir=plots_base_dir,
        script_name=Path(__file__).stem,
        problem_name="difconv",
        size_tag=size_tag,
        T=T,
        seed=SEED,
    )
    method_names = [name for name, _ in methods]
    summary = save_runtime_artifacts(
        run_dir=run_dir,
        run_prefix="runtime_interleaved_end2end",
        method_names=method_names,
        runtime_sec=rt,
        overhead_sec=overhead,
        T=T,
        title=(
            f"BoomerAMG Setup runtime cumulative (test 3, difconv)  "
            f"T={T}  {n_desc}  c={C_MIN:g}..{C_MAX:g}"
        ),
        default_method_name="default (fixed)",
        traces=traces,
        trace_keys=trace_keys,
        trace_title=f"Bandit parameter traces (test 3)  T={T}  seed={SEED}",
        default_params=DEFAULT_PARAMS,
        diagnostics_window=500,
        summary_extra={
            "seed": int(SEED),
            "alpha": float(ALPHA),
            "l2": float(L2),
            "sigma": float(SIGMA),
            "n_min": int(N_MIN),
            "n_max": int(N_MAX),
            "fixed_n": int(FIXED_N),
            "c_min": float(C_MIN),
            "c_max": float(C_MAX),
            "context_dim": int(DIFCONV_CONTEXT_DIM),
            "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
            "elite_cache_size": int(ELITE_CACHE_SIZE),
            "actions_grid_n": int(grid_n),
            "actions_count": int(len(actions)),
            "solver_tol": float(SOLVER_TOL),
            "solver_max_iter": int(SOLVER_MAX_ITER),
            "failed_count": {k: int(v) for k, v in failed_count.items()},
        },
    )

    print("PLOT:", summary["plot"])
    print("SUMMARY:", summary["summary_path"])
    print("TOTAL hypre seconds:", summary["total_hypre_runtime_sec"])
    print("TOTAL overhead seconds:", summary["total_bandit_overhead_sec"])


if __name__ == "__main__":
    main()
