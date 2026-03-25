import json
import os
import sys
import time
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.stencil27_laplace import stencil_27_laplace
from utils.setup_amg import build_actions_th_mxrs_tr, init_param_trace, record_param_trace

from learners.LinUCB_AMG import LinUCB_AMG
from learners.SharedLinUCB_AMG import SharedLinUCB_AMG
from learners.LinUCB_AMG_v2 import LinUCB_AMG_v2
from learners.SharedLinUCB_AMG_v2 import SharedLinUCB_AMG_v2
from learners.SharedLinUCB_AMG_v3 import SharedLinUCB_AMG_v3
from learners.SharedLinTS_AMG import SharedLinTS_AMG
from learners.SharedBootstrapTS_AMG import SharedBootstrapTS_AMG
from learners.RFF_TS_AMG import RFF_TS_AMG
from learners.TsallisINF_AMG import TsallisINF_AMG
from solver import solve


T = int(os.environ.get("T", "5000"))
SEED = int(os.environ.get("SEED", "20260209"))
RUN_TAG = os.environ.get("RUN_TAG", "").strip() or time.strftime("%Y%m%d_%H%M%S")

ALPHA = 1.0
L2 = 1.0
NX = int(os.environ.get("NX", "60"))
NY = int(os.environ.get("NY", "60"))
NZ = int(os.environ.get("NZ", "1"))
CANDIDATE_POOL_SIZE = int(os.environ.get("CANDIDATE_POOL_SIZE", "512"))
ELITE_CACHE_SIZE = int(os.environ.get("ELITE_CACHE_SIZE", "64"))

SOLVER_TOL = float(os.environ.get("SOLVER_TOL", "1e-8"))
SOLVER_MAX_ITER = int(os.environ.get("SOLVER_MAX_ITER", "10000"))

# Shared action scaling (v3 linear models, TS, bootstrap, RFF).
ACTION_SCALE_TH = float(os.environ.get("ACTION_SCALE_TH", "0.25"))
ACTION_SCALE_MXRS = float(os.environ.get("ACTION_SCALE_MXRS", "0.1"))
ACTION_SCALE_TR = float(os.environ.get("ACTION_SCALE_TR", "0.2"))

LINTS_SIGMA = os.environ.get("LINTS_SIGMA", "").strip()
LINTS_SIGMA_MULT = float(os.environ.get("LINTS_SIGMA_MULT", "0.15"))

BOOTSTRAP_HEADS = int(os.environ.get("BOOTSTRAP_HEADS", "10"))

RFF_DIM = int(os.environ.get("RFF_DIM", "128"))
RFF_LENGTHSCALE = float(os.environ.get("RFF_LENGTHSCALE", "1.0"))
RFF_CDIAG_SCALE = float(os.environ.get("RFF_CDIAG_SCALE", "2.5"))
RFF_SIGMA = os.environ.get("RFF_SIGMA", "").strip()
RFF_SIGMA_MULT = float(os.environ.get("RFF_SIGMA_MULT", "0.15"))

plots_root = Path(__file__).resolve().parent.parent / "plots" / "Combined"
# One folder per run to avoid overwriting plots/results (and to keep runs grouped).
run_dir_name = f"{Path(__file__).stem}_seed{SEED}_{NX}x{NY}x{NZ}_T{T}_tag{RUN_TAG}"
out_dir = plots_root / run_dir_name
out_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_PARAMS = {
    "strong_threshold": 0.25,
    "max_row_sum": 0.90,
    "trunc_factor": 0.00,
    "coarsen_type": 10,
    "interp_type": 6,
}

FIXED_MODE_V2_19CUBE = {
    # From prior best-performing shared-v2 run.
    "strong_threshold": 0.50,
    # NOTE: HYPRE rejects max_row_sum <= 0, so use a tiny positive epsilon.
    "max_row_sum": 1e-6,
    # NOTE: HYPRE rejects trunc_factor >= 1, so use a value just below 1.
    "trunc_factor": 0.999999,
    "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
    "interp_type": DEFAULT_PARAMS["interp_type"],
}

FIXED_MODE_TSALLIS_3CUBE = {
    # From prior Tsallis 3^3 mode action.
    "strong_threshold": 0.05,
    "max_row_sum": 0.10,
    "trunc_factor": 0.80,
    "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
    "interp_type": DEFAULT_PARAMS["interp_type"],
}


def runtime_loss_sec(*, outcome, fail_runtime_sec: float, **_) -> float:
    rt = float(outcome["runtime"])
    if not np.isfinite(rt):
        return float(fail_runtime_sec)
    # Prevent "fast failures" from looking good under runtime loss.
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


class DisjointLinUCBFactory:
    def __init__(self, alpha: float, l2: float, *, model_cls=LinUCB_AMG, model_kwargs=None):
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.model_cls = model_cls
        self.model_kwargs = dict(model_kwargs or {})

    def new_trial(self, *, parameter_space, seed: int, **_):
        class _Policy:
            def __init__(self, actions, alpha, l2, seed):
                self.m = model_cls(
                    actions,
                    context_dim=5,
                    alpha=float(alpha),
                    l2_reg=float(l2),
                    seed=int(seed),
                    **model_kwargs,
                )

            def select(self, context, **_):
                params = self.m.predict(context)
                step = self.m.history[-1]
                return params, {"pred_mean": step.pred_mean, "pred_uncert": step.pred_uncert}

            def update(self, loss, **_):
                self.m.update(float(loss))

        model_cls = self.model_cls
        model_kwargs = self.model_kwargs
        return _Policy(parameter_space["actions"], self.alpha, self.l2, seed)


class SharedLinUCBFactory:
    def __init__(self, alpha: float, l2: float, *, model_cls=SharedLinUCB_AMG, model_kwargs=None):
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.model_cls = model_cls
        self.model_kwargs = dict(model_kwargs or {})

    def new_trial(self, *, parameter_space, seed: int, **_):
        class _Policy:
            def __init__(self, actions, alpha, l2, seed):
                self.m = model_cls(
                    actions,
                    context_dim=5,
                    alpha=float(alpha),
                    l2_reg=float(l2),
                    seed=int(seed),
                    **model_kwargs,
                )

            def select(self, context, **_):
                params = self.m.predict(context)
                step = self.m.history[-1]
                return params, {"pred_mean": step.pred_mean, "pred_uncert": step.pred_uncert}

            def update(self, loss, **_):
                self.m.update(float(loss))

        model_cls = self.model_cls
        model_kwargs = self.model_kwargs
        return _Policy(parameter_space["actions"], self.alpha, self.l2, seed)


class TsallisActionFactory:
    def __init__(self, actions):
        self.actions = [dict(a) for a in actions]

    def new_trial(self, *, seed: int, T: int, **_):
        class _Policy:
            def __init__(self, actions, seed, T):
                self.m = TsallisINF_AMG(np.asarray(actions, dtype=object), int(T), seed=int(seed))

            def select(self, **_):
                params = self.m.predict()
                return dict(params), {}

            def update(self, loss, **_):
                # TsallisINF_AMG assumes updates with (loss - 1).
                # Runtime loss can be << 1, so shift to keep updates stable.
                self.m.update(float(loss) + 1.0)

        return _Policy(self.actions, seed, T)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run selected AMG setup bandit methods (default: default only)."
    )
    parser.add_argument(
        "--methods",
        action="append",
        default=[],
        help=(
            "Comma-separated or repeated list of method IDs to run. "
            "Use --list-methods to see options. Default: default"
        ),
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List available method IDs and exit.",
    )
    return parser.parse_args()


def _normalize_method_ids(raw: list[str], *, available: list[str]) -> list[str]:
    if not raw:
        return ["default"]
    ids: list[str] = []
    for entry in raw:
        for token in str(entry).split(","):
            token = token.strip()
            if token:
                ids.append(token)
    if not ids:
        return ["default"]
    if "all" in ids:
        return list(available)
    unknown = sorted(set(ids) - set(available))
    if unknown:
        raise ValueError(f"Unknown method id(s): {unknown}. Use --list-methods.")
    seen = set()
    ordered: list[str] = []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            ordered.append(mid)
    return ordered


def main() -> None:
    args = _parse_args()

    # Available method IDs (used for CLI selection).
    available_methods = [
        "default",
        "fixed_tsallis_3cube",
        "linucb_disjoint_v1",
        "linucb_disjoint_v2",
        "tsallis",
        "linucb_shared",
        "linucb_shared_v2",
        "linucb_shared_v3",
        "lints",
        "bootstrap_ts",
        "rff_ts",
    ]
    if args.list_methods:
        print("Available method IDs:")
        for mid in available_methods:
            print(" -", mid)
        return
    selected_methods = _normalize_method_ids(args.methods, available=available_methods)

    fail_runtime_sec = 1e9
    # Warmup: the first `solve()` in a fresh Python process pays MPI/HYPRE init
    # inside the C library (amg_setup_solver.c::_ensure_init). This can be
    # several seconds and is noisy run-to-run, so do one untimed warmup solve
    # before any timing begins.
    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_27_laplace(rng=warm_rng, nx=NX, ny=NY, nz=NZ)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    # Baseline runtime (for TS sigma defaults): median of 3 solves on the warmup matrix.
    # Note: runtime_sec is hypre-only (setup+solve) measured inside C; matrix build is excluded.
    baseline_runtime_sec = None
    if any(mid in selected_methods for mid in ["lints", "rff_ts"]):
        rts: list[float] = []
        for _ in range(10):
            try:
                res = solve(params=DEFAULT_PARAMS, tol=SOLVER_TOL, max_iter=SOLVER_MAX_ITER, **warm_mkw)
                rt = float(res.runtime_sec)
            except Exception:
                rt = float("nan")
            if np.isfinite(rt):
                rts.append(rt)
            if len(rts) >= 3:
                break
        baseline_runtime_sec = float(np.median(np.asarray(rts, dtype=float))) if rts else 1.0

    # 2D problem size for this experiment.
    # Grids: evenly spaced and reduced for 3-parameter tuning (sample-efficiency).
    grid3_n = 3
    th_grid_3d = np.linspace(0.05, 0.45, grid3_n)   # includes default th=0.25
    mxrs_grid_3d = np.linspace(0.10, 0.90, grid3_n)  # includes default mxrs=0.90
    tr_grid_3d = np.linspace(0.00, 0.80, grid3_n)    # includes default tr=0.00

    # User request: 19x19x19 uniform grids over [0, 1] for (th,mxrs,tr).
    # Note: strong_threshold and max_row_sum are conceptually in (0,1); to
    # avoid potential solver edge-case failures at exactly 0 or 1, we clip
    # them slightly inward while keeping an approximately-uniform grid.
    # Updated discretization: 20 points up to 0.95 (reference-style).
    # Constraints from HYPRE:
    # - strong_threshold in [0,1] so 0 is allowed
    # - trunc_factor in [0,1) so 0 is allowed (1 is not)
    # - max_row_sum must be in (0,1] so 0 is NOT allowed
    grid5_n = 20
    grid5_max = 0.95
    th_grid_5d = np.linspace(0.0, grid5_max, grid5_n)
    mxrs_grid_5d = np.linspace(0.0, grid5_max, grid5_n)
    mxrs_grid_5d[0] = 1e-6
    tr_grid_5d = np.linspace(0.0, grid5_max, grid5_n)

    needs_actions = any(
        mid in selected_methods
        for mid in [
            "linucb_disjoint_v1",
            "linucb_disjoint_v2",
            "tsallis",
            "linucb_shared",
            "linucb_shared_v2",
            "linucb_shared_v3",
            "lints",
            "bootstrap_ts",
            "rff_ts",
        ]
    )

    actions_cont3_3 = []
    actions_cont3_5 = []
    default_arm_index_5 = None
    if needs_actions:
        # Action spaces for (th,mxrs,tr) at two discretization levels.
        actions_cont3_3 = build_actions_th_mxrs_tr(
            th_grid_3d,
            mxrs_grid_3d,
            tr_grid_3d,
            fixed_params={"coarsen_type": DEFAULT_PARAMS["coarsen_type"], "interp_type": DEFAULT_PARAMS["interp_type"]},
        )
        actions_cont3_5 = build_actions_th_mxrs_tr(
            th_grid_5d,
            mxrs_grid_5d,
            tr_grid_5d,
            fixed_params={"coarsen_type": DEFAULT_PARAMS["coarsen_type"], "interp_type": DEFAULT_PARAMS["interp_type"]},
        )

    # Ensure the default configuration is always an available arm (safety baseline).
    # The uniform grids do not necessarily contain th=0.25 or mxrs=0.90.
    def _same_action(a: dict, b: dict) -> bool:
        return (
            np.isclose(float(a["strong_threshold"]), float(b["strong_threshold"]), rtol=0.0, atol=1e-12)
            and np.isclose(float(a["max_row_sum"]), float(b["max_row_sum"]), rtol=0.0, atol=1e-12)
            and np.isclose(float(a["trunc_factor"]), float(b["trunc_factor"]), rtol=0.0, atol=1e-12)
            and int(a["coarsen_type"]) == int(b["coarsen_type"])
            and int(a["interp_type"]) == int(b["interp_type"])
        )

    default_arm = dict(DEFAULT_PARAMS)
    default_arm_index_5 = next((i for i, a in enumerate(actions_cont3_5) if _same_action(a, default_arm)), None)
    if default_arm_index_5 is None:
        actions_cont3_5.append(default_arm)
        default_arm_index_5 = len(actions_cont3_5) - 1

    # ---------------------------------------------------------------------
    # Interleaved evaluation (fair timing):
    # - one shared instance stream (rng_inst)
    # - evaluate *all* methods per round on the same instance
    # - randomize method evaluation order each round to reduce drift bias
    # Loss is hypre-only runtime (solver walltime); overhead is tracked separately.
    # ---------------------------------------------------------------------

    ps_default = {"actions": [DEFAULT_PARAMS], "context_dim": 5}
    ps_cont3_3 = {"actions": actions_cont3_3, "context_dim": 5}
    ps_cont3_5 = {"actions": actions_cont3_5, "context_dim": 5}

    def _require_actions() -> None:
        if not actions_cont3_5:
            raise RuntimeError("Selected methods require action grids, but none were built.")

    methods: list[tuple[str, object, dict, bool]] = []
    for mid in selected_methods:
        if mid == "default":
            methods.append(("default (fixed)", FixedPolicy(DEFAULT_PARAMS), ps_default, False))
        elif mid == "fixed_tsallis_3cube":
            methods.append(("fixed: (th,mxrs,tr) = (0.05,0.10,0.80)", FixedPolicy(FIXED_MODE_TSALLIS_3CUBE), ps_default, False))
        elif mid == "linucb_disjoint_v2":
            _require_actions()
            policy = DisjointLinUCBFactory(
                ALPHA,
                L2,
                model_cls=LinUCB_AMG_v2,
                model_kwargs={"alpha_decay": True},
            ).new_trial(parameter_space=ps_cont3_5, seed=SEED + 12003, T=T, trial=0)
            methods.append((f"LinUCB disjoint v2: {grid5_n}^3", policy, ps_cont3_5, False))
        elif mid == "linucb_disjoint_v1":
            _require_actions()
            policy = DisjointLinUCBFactory(
                ALPHA,
                L2,
                model_cls=LinUCB_AMG,
                model_kwargs={},
            ).new_trial(parameter_space=ps_cont3_5, seed=SEED + 12001, T=T, trial=0)
            methods.append((f"LinUCB disjoint v1: {grid5_n}^3", policy, ps_cont3_5, False))
        elif mid == "tsallis":
            _require_actions()
            policy = TsallisActionFactory(actions_cont3_5).new_trial(parameter_space=ps_cont3_5, seed=SEED + 10004, T=T, trial=0)
            methods.append((f"Tsallis-INF: {grid5_n}^3", policy, ps_cont3_5, True))
        elif mid == "linucb_shared":
            _require_actions()
            policy = SharedLinUCBFactory(ALPHA, L2).new_trial(parameter_space=ps_cont3_5, seed=SEED + 10003, T=T, trial=0)
            methods.append((f"LinUCB shared: {grid5_n}^3", policy, ps_cont3_5, False))
        elif mid == "linucb_shared_v2":
            _require_actions()
            if default_arm_index_5 is None:
                raise RuntimeError("default_arm_index_5 not set; action grid may be missing default arm.")
            policy = SharedLinUCBFactory(
                ALPHA,
                L2,
                model_cls=SharedLinUCB_AMG_v2,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index_5)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=ps_cont3_5, seed=SEED + 11003, T=T, trial=0)
            methods.append((f"LinUCB shared v2: {grid5_n}^3", policy, ps_cont3_5, False))
        elif mid == "linucb_shared_v3":
            _require_actions()
            if default_arm_index_5 is None:
                raise RuntimeError("default_arm_index_5 not set; action grid may be missing default arm.")
            policy = SharedLinUCBFactory(
                ALPHA,
                L2,
                model_cls=SharedLinUCB_AMG_v3,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "action_scales": (ACTION_SCALE_TH, ACTION_SCALE_MXRS, ACTION_SCALE_TR),
                    "alpha_decay": True,
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index_5)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=ps_cont3_5, seed=SEED + 21003, T=T, trial=0)
            methods.append((f"LinUCB shared v3: {grid5_n}^3", policy, ps_cont3_5, False))
        elif mid == "lints":
            _require_actions()
            if default_arm_index_5 is None:
                raise RuntimeError("default_arm_index_5 not set; action grid may be missing default arm.")
            if baseline_runtime_sec is None:
                raise RuntimeError("baseline_runtime_sec not computed")
            sigma = float(LINTS_SIGMA) if LINTS_SIGMA else float(LINTS_SIGMA_MULT) * float(baseline_runtime_sec)
            policy = SharedLinUCBFactory(
                ALPHA,
                L2,
                model_cls=SharedLinTS_AMG,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "action_scales": (ACTION_SCALE_TH, ACTION_SCALE_MXRS, ACTION_SCALE_TR),
                    "sigma": float(sigma),
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index_5)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=ps_cont3_5, seed=SEED + 22003, T=T, trial=0)
            methods.append((f"LinTS: {grid5_n}^3", policy, ps_cont3_5, False))
        elif mid == "bootstrap_ts":
            _require_actions()
            if default_arm_index_5 is None:
                raise RuntimeError("default_arm_index_5 not set; action grid may be missing default arm.")
            policy = SharedLinUCBFactory(
                ALPHA,
                L2,
                model_cls=SharedBootstrapTS_AMG,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "action_scales": (ACTION_SCALE_TH, ACTION_SCALE_MXRS, ACTION_SCALE_TR),
                    "heads": int(BOOTSTRAP_HEADS),
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index_5)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=ps_cont3_5, seed=SEED + 23003, T=T, trial=0)
            methods.append((f"BootstrapTS: {grid5_n}^3", policy, ps_cont3_5, False))
        elif mid == "rff_ts":
            _require_actions()
            if default_arm_index_5 is None:
                raise RuntimeError("default_arm_index_5 not set; action grid may be missing default arm.")
            if baseline_runtime_sec is None:
                raise RuntimeError("baseline_runtime_sec not computed")
            sigma = float(RFF_SIGMA) if RFF_SIGMA else float(RFF_SIGMA_MULT) * float(baseline_runtime_sec)
            policy = SharedLinUCBFactory(
                ALPHA,
                L2,
                model_cls=RFF_TS_AMG,
                model_kwargs={
                    "action_center": DEFAULT_PARAMS,
                    "action_scales": (ACTION_SCALE_TH, ACTION_SCALE_MXRS, ACTION_SCALE_TR),
                    "sigma": float(sigma),
                    "rff_dim": int(RFF_DIM),
                    "lengthscale": float(RFF_LENGTHSCALE),
                    "cdiag_scale": float(RFF_CDIAG_SCALE),
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "always_include_arms": [int(default_arm_index_5)],
                    "elite_cache_size": ELITE_CACHE_SIZE,
                },
            ).new_trial(parameter_space=ps_cont3_5, seed=SEED + 24003, T=T, trial=0)
            methods.append((f"RFF-TS: {grid5_n}^3", policy, ps_cont3_5, False))
        else:
            raise RuntimeError(f"Unreachable method id: {mid}")

    rt = {name: np.zeros(T, dtype=float) for name, *_ in methods}
    overhead = {name: np.zeros(T, dtype=float) for name, *_ in methods}
    failed_count = {name: 0 for name, *_ in methods}
    trace_keys = ("strong_threshold", "max_row_sum", "trunc_factor")
    traces = {name: init_param_trace(trace_keys, T) for name, *_ in methods}

    rng_inst = np.random.default_rng(SEED)
    rng_order = np.random.default_rng(SEED ^ 0xA5A5A5A5)
    prev_update_est = {name: 0.0 for name, *_ in methods}

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
        mkw, context, meta = stencil_27_laplace(rng=rng_inst, nx=NX, ny=NY, nz=NZ)
        order = rng_order.permutation(len(methods))

        for i in order:
            name, policy, parameter_space, is_tsallis = methods[int(i)]

            sel_start = time.perf_counter_ns()
            selected = policy.select(context=context, parameter_space=parameter_space)
            sel_sec = (time.perf_counter_ns() - sel_start) / 1e9

            if isinstance(selected, tuple):
                params = selected[0]
            else:
                params = selected

            out = safe_solve(params, mkw)
            rt[name][t] = float(out["runtime"])
            failed_count[name] += int(bool(out.get("failed", False)))
            record_param_trace(traces[name], t=t, params=params, keys=trace_keys)

            loss_start = time.perf_counter_ns()
            base_loss_sec = float(runtime_loss_sec(outcome=out, fail_runtime_sec=fail_runtime_sec))
            loss_sec = (time.perf_counter_ns() - loss_start) / 1e9

            # Loss used for learning: hypre walltime + bandit overhead estimate.
            # Use previous round's update-time as an estimate to avoid circularity.
            end_to_end_loss_sec = base_loss_sec + float(sel_sec) + float(loss_sec) + float(prev_update_est[name])
            loss_value = end_to_end_loss_sec

            upd_sec = 0.0
            if hasattr(policy, "update"):
                upd_start = time.perf_counter_ns()
                policy.update(loss=loss_value, context=context, params=params, outcome=out)
                upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

            prev_update_est[name] = float(upd_sec)
            overhead[name][t] = float(sel_sec + loss_sec + upd_sec)

    y = T - np.arange(1, T + 1)
    # Plot end-to-end runtime (hypre walltime + bandit overhead).
    plt.figure(figsize=(10, 5))
    for name, *_ in methods:
        cum = np.cumsum(rt[name] + overhead[name])
        if name == "default (fixed)":
            plt.plot(cum, y, "--", linewidth=2.0, label=name)
        else:
            plt.plot(cum, y, linewidth=2.3, label=name)

    plt.xlabel("cumulative runtime (seconds)")
    plt.ylabel("instances remaining")
    plt.title(
        f"BoomerAMG Setup runtime cumulative (test 2, interleaved)  T={T}  {NX}x{NY}x{NZ} stencil_27_laplace  [loss=hypre+overhead runtime]",
        fontsize=12,
    )
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = out_dir / f"test2_runtime_cumulative_T{T}_interleaved_end2end_{NX}x{NY}x{NZ}_seed{SEED}_tag{RUN_TAG}.png"
    plt.savefig(plot_path, dpi=256)
    plt.close()

    # Bandit-only parameter trace plot:
    # - One row per bandit method (no baseline/default/fixed methods).
    # - Three subplots per bandit (one for each parameter).
    trace_plot_path = None
    bandit_names = [name for name, policy, *_ in methods if not isinstance(policy, FixedPolicy)]
    if bandit_names:
        n_bandits = int(len(bandit_names))
        nrows = n_bandits if n_bandits else 1
        ncols = len(trace_keys)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.2 * nrows), sharex=True, sharey=True)
        axes = np.atleast_2d(axes)
        t_axis = np.arange(1, T + 1)

        # High-contrast, colorblind-friendly colors per parameter.
        param_colors = {
            "strong_threshold": "#0072B2",
            "max_row_sum": "#D55E00",
            "trunc_factor": "#009E73",
        }

        for row_i, name in enumerate(bandit_names):
            for col_i, key in enumerate(trace_keys):
                ax = axes[row_i, col_i]
                series = traces[name][key]
                ax.scatter(
                    t_axis,
                    series,
                    s=14,
                    marker=".",
                    label=None,
                    color=param_colors.get(key, None),
                    alpha=0.9,
                    linewidths=0.0,
                    rasterized=True,
                )
                if row_i == 0:
                    ax.set_title(key, fontsize=10)
                if col_i == 0:
                    ax.set_ylabel(name, fontsize=10)
                ax.set_ylim(-0.02, 1.02)
                ax.grid(True, alpha=0.25)
                if row_i == nrows - 1:
                    ax.set_xlabel("t")

        fig.suptitle(f"Bandit parameter traces (3 panels per bandit)  T={T}  seed={SEED}", fontsize=12)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))

        trace_plot_path = out_dir / f"test2_param_trace_bandits_T{T}_interleaved_end2end_{NX}x{NY}x{NZ}_seed{SEED}_tag{RUN_TAG}.png"
        fig.savefig(trace_plot_path, dpi=256)
        plt.close(fig)

    # Quick 鈥渄id it match default?鈥?diagnostics on the last window.
    def _action_key(name: str, idx: int) -> tuple:
        return tuple(round(float(traces[name][k][idx]), 6) for k in trace_keys)

    def _fraction_default(name: str, window: int = 500) -> float:
        start = max(0, T - int(window))
        mask = np.ones(T - start, dtype=bool)
        for k in trace_keys:
            mask &= np.isclose(traces[name][k][start:], float(DEFAULT_PARAMS[k]), rtol=0.0, atol=1e-12)
        return float(np.mean(mask)) if mask.size else 0.0

    def _mode_action(name: str, window: int = 500) -> dict:
        from collections import Counter

        start = max(0, T - int(window))
        keys = [_action_key(name, i) for i in range(start, T)]
        if not keys:
            return {"action": None, "count": 0, "window": int(window)}
        action, count = Counter(keys).most_common(1)[0]
        return {"action": action, "count": int(count), "window": int(window)}

    summary = {
        "T": int(T),
        "seed": int(SEED),
        "run_tag": str(RUN_TAG),
        "alpha": float(ALPHA),
        "l2": float(L2),
        "nx": int(NX),
        "ny": int(NY),
        "nz": int(NZ),
        "solver_tol": float(SOLVER_TOL),
        "solver_max_iter": int(SOLVER_MAX_ITER),
        "candidate_pool_size": int(CANDIDATE_POOL_SIZE),
        "elite_cache_size": int(ELITE_CACHE_SIZE),
        "action_scales": [float(ACTION_SCALE_TH), float(ACTION_SCALE_MXRS), float(ACTION_SCALE_TR)],
        "baseline_runtime_sec": float(baseline_runtime_sec) if baseline_runtime_sec is not None else None,
        "lints_sigma": (float(LINTS_SIGMA) if LINTS_SIGMA else None),
        "lints_sigma_mult": float(LINTS_SIGMA_MULT),
        "bootstrap_heads": int(BOOTSTRAP_HEADS),
        "rff_dim": int(RFF_DIM),
        "rff_lengthscale": float(RFF_LENGTHSCALE),
        "rff_cdiag_scale": float(RFF_CDIAG_SCALE),
        "rff_sigma": (float(RFF_SIGMA) if RFF_SIGMA else None),
        "rff_sigma_mult": float(RFF_SIGMA_MULT),
        "actions_cont3_grid3": int(len(actions_cont3_3)),
        "actions_cont3_grid5": int(len(actions_cont3_5)),
        "grid3_n": int(grid3_n),
        "grid5_n": int(grid5_n),
        "total_hypre_runtime_sec": {k: float(np.sum(v)) for k, v in rt.items()},
        "total_bandit_overhead_sec": {k: float(np.sum(v)) for k, v in overhead.items()},
        "total_end_to_end_sec": {k: float(np.sum(rt[k]) + np.sum(overhead[k])) for k in rt},
        "failed_count": {k: int(v) for k, v in failed_count.items()},
        "plot": str(plot_path),
        "trace_plot": str(trace_plot_path),
        "trace_keys": list(trace_keys),
        "last500_fraction_equal_default": {name: _fraction_default(name, window=500) for name, *_ in methods},
        "last500_mode_action": {name: _mode_action(name, window=500) for name, *_ in methods},
    }
    summary_path = out_dir / f"test2_runtime_summary_T{T}_interleaved_end2end_{NX}x{NY}x{NZ}_seed{SEED}_tag{RUN_TAG}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("PLOT:", plot_path)
    print("SUMMARY:", summary_path)
    print("TOTAL hypre seconds:", summary["total_hypre_runtime_sec"])
    print("TOTAL overhead seconds:", summary["total_bandit_overhead_sec"])


if __name__ == "__main__":
    main()
