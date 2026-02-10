import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.problem_amg import stencil_27_laplace
from utils.setup_amg import build_actions_th_mxrs_tr, init_param_trace, moving_average, record_param_trace

from learners.LinUCB_AMG import LinUCB_AMG
from learners.SharedLinUCB_AMG import SharedLinUCB_AMG
from learners.TsallisINF_AMG import TsallisINF_AMG
from solver import solve


T = 5000
SEED = 20260209

ALPHA = 1.0
L2 = 1.0
NX, NY, NZ = 60, 60, 1

out_dir = Path(__file__).resolve().parent.parent / "plots" / "Combined"
out_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_PARAMS = {
    "strong_threshold": 0.25,
    "max_row_sum": 0.90,
    "trunc_factor": 0.00,
    "coarsen_type": 10,
    "interp_type": 6,
}


def runtime_loss_sec(*, outcome, fail_penalty: float, **_) -> float:
    rt = float(outcome["runtime"])
    wu = float(outcome["wu"])
    if not np.isfinite(rt):
        return 1e9
    # Prevent "fast failures" from looking good under runtime loss.
    if wu >= 0.999 * float(fail_penalty):
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
    def __init__(self, alpha: float, l2: float):
        self.alpha = float(alpha)
        self.l2 = float(l2)

    def new_trial(self, *, parameter_space, seed: int, **_):
        class _Policy:
            def __init__(self, actions, alpha, l2, seed):
                self.m = LinUCB_AMG(actions, context_dim=5, alpha=float(alpha), l2_reg=float(l2), seed=int(seed))

            def select(self, context, **_):
                params = self.m.predict(context)
                step = self.m.history[-1]
                return params, {"pred_mean": step.pred_mean, "pred_uncert": step.pred_uncert}

            def update(self, loss, **_):
                self.m.update(float(loss))

        return _Policy(parameter_space["actions"], self.alpha, self.l2, seed)


class SharedLinUCBFactory:
    def __init__(self, alpha: float, l2: float):
        self.alpha = float(alpha)
        self.l2 = float(l2)

    def new_trial(self, *, parameter_space, seed: int, **_):
        class _Policy:
            def __init__(self, actions, alpha, l2, seed):
                self.m = SharedLinUCB_AMG(actions, context_dim=5, alpha=float(alpha), l2_reg=float(l2), seed=int(seed))

            def select(self, context, **_):
                params = self.m.predict(context)
                step = self.m.history[-1]
                return params, {"pred_mean": step.pred_mean, "pred_uncert": step.pred_uncert}

            def update(self, loss, **_):
                self.m.update(float(loss))

        return _Policy(parameter_space["actions"], self.alpha, self.l2, seed)


class TsallisActionFactory:
    def __init__(self, actions):
        self.actions = [dict(a) for a in actions]

    def new_trial(self, *, seed: int, T: int, **_):
        class _Policy:
            def __init__(self, actions, seed, T):
                self.m = TsallisINF_AMG(np.asarray(actions, dtype=object), int(T))
                # Keep behavior deterministic across runs.
                np.random.seed(int(seed))

            def select(self, **_):
                params = self.m.predict()
                return dict(params), {}

            def update(self, loss, **_):
                # TsallisINF_AMG assumes updates with (loss - 1).
                # Runtime loss can be << 1, so shift to keep updates stable.
                self.m.update(float(loss) + 1.0)

        return _Policy(self.actions, seed, T)


def main() -> None:
    fail_penalty = 1e6
    # Warmup: the first `solve()` in a fresh Python process pays MPI/HYPRE init
    # inside the C library (amg_setup_solver.c::_ensure_init). This can be
    # several seconds and is noisy run-to-run, so do one untimed warmup solve
    # before any timing begins.
    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_27_laplace(rng=warm_rng, nx=NX, ny=NY, nz=NZ)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    # 2D problem size for this experiment.
    # Grids: evenly spaced and reduced for 3-parameter tuning (sample-efficiency).
    grid3_n = 3
    th_grid_3d = np.linspace(0.05, 0.45, grid3_n)   # includes default th=0.25
    mxrs_grid_3d = np.linspace(0.10, 0.90, grid3_n)  # includes default mxrs=0.90
    tr_grid_3d = np.linspace(0.00, 0.80, grid3_n)    # includes default tr=0.00

    grid5_n = 5
    th_grid_5d = np.linspace(0.05, 0.85, grid5_n)    # includes default th=0.25
    mxrs_grid_5d = np.linspace(0.10, 0.90, grid5_n)  # includes default mxrs=0.90
    tr_grid_5d = np.linspace(0.00, 0.80, grid5_n)    # includes default tr=0.00

    th_grid_19 = np.linspace(0.05, 0.95, 19)

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
    actions_th_19 = [
        {
            "strong_threshold": float(th),
            "max_row_sum": DEFAULT_PARAMS["max_row_sum"],
            "trunc_factor": DEFAULT_PARAMS["trunc_factor"],
            "coarsen_type": DEFAULT_PARAMS["coarsen_type"],
            "interp_type": DEFAULT_PARAMS["interp_type"],
        }
        for th in th_grid_19
    ]

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
    ps_th_19 = {"actions": actions_th_19, "context_dim": 5}

    policy_default = FixedPolicy(DEFAULT_PARAMS)
    policy_disjoint_3 = DisjointLinUCBFactory(ALPHA, L2).new_trial(parameter_space=ps_cont3_3, seed=SEED + 10000, T=T, trial=0)
    policy_shared_th_19 = SharedLinUCBFactory(ALPHA, L2).new_trial(parameter_space=ps_th_19, seed=SEED + 10001, T=T, trial=0)
    policy_tsallis_cont3_3 = TsallisActionFactory(actions_cont3_3).new_trial(parameter_space=ps_cont3_3, seed=SEED + 10002, T=T, trial=0)
    policy_shared_5 = SharedLinUCBFactory(ALPHA, L2).new_trial(parameter_space=ps_cont3_5, seed=SEED + 10003, T=T, trial=0)

    methods = [
        ("default (fixed)", policy_default, ps_default, False),
        (f"LinUCB disjoint: (th,mxrs,tr) {grid3_n}^3", policy_disjoint_3, ps_cont3_3, False),
        ("LinUCB shared: (th) 19 arms", policy_shared_th_19, ps_th_19, False),
        (f"Tsallis-INF: (th,mxrs,tr) {grid3_n}^3", policy_tsallis_cont3_3, ps_cont3_3, True),
        (f"LinUCB shared: (th,mxrs,tr) {grid5_n}^3", policy_shared_5, ps_cont3_5, False),
    ]

    rt = {name: np.zeros(T, dtype=float) for name, *_ in methods}
    overhead = {name: np.zeros(T, dtype=float) for name, *_ in methods}
    trace_keys = ("strong_threshold", "max_row_sum", "trunc_factor")
    traces = {name: init_param_trace(trace_keys, T) for name, *_ in methods}

    rng_inst = np.random.default_rng(SEED)
    rng_order = np.random.default_rng(SEED ^ 0xA5A5A5A5)
    prev_update_est = {name: 0.0 for name, *_ in methods}

    def safe_solve(params: dict, mkw: dict) -> dict:
        try:
            res = solve(params=params, **mkw)
            return {"wu": float(res.work_units), "runtime": float(res.runtime_sec)}
        except Exception:
            return {"wu": float(fail_penalty), "runtime": 1e9}

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
            record_param_trace(traces[name], t=t, params=params, keys=trace_keys)

            loss_start = time.perf_counter_ns()
            base_loss_sec = float(runtime_loss_sec(outcome=out, fail_penalty=fail_penalty))
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
        f"BoomerAMG Setup runtime cumulative (test 2, interleaved)  T={T}  2D {NX}x{NY} stencil_27_laplace  [loss=hypre+overhead runtime]",
        fontsize=12,
    )
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = out_dir / f"test2_runtime_cumulative_T{T}_interleaved_end_to_end_loss_end2end_{NX}x{NY}_tsallis3d.png"
    plt.savefig(plot_path, dpi=256)
    plt.close()

    # Parameter trace plot (moving average) to see where the bandits drift/converge.
    ma_window = 25
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    t_axis = np.arange(1, T + 1)
    for ax, key in zip(axes, trace_keys):
        for name, *_ in methods:
            series = traces[name][key]
            ax.plot(t_axis, moving_average(series, ma_window), linewidth=1.4, label=name)
        ax.axhline(float(DEFAULT_PARAMS[key]), color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="default" if key == trace_keys[0] else None)
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("t")
    axes[0].set_title(f"Chosen parameter traces (MA{ma_window})  T={T}  seed={SEED}", fontsize=12)
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()

    trace_plot_path = out_dir / f"test2_param_trace_T{T}_interleaved_loss_end2end_{NX}x{NY}_tsallis3d.png"
    fig.savefig(trace_plot_path, dpi=256)
    plt.close(fig)

    # Quick “did it match default?” diagnostics on the last window.
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
        "alpha": float(ALPHA),
        "l2": float(L2),
        "nx": int(NX),
        "ny": int(NY),
        "nz": int(NZ),
        "actions_cont3_grid3": int(len(actions_cont3_3)),
        "actions_cont3_grid5": int(len(actions_cont3_5)),
        "actions_th_19": int(len(actions_th_19)),
        "grid3_n": int(grid3_n),
        "grid5_n": int(grid5_n),
        "total_hypre_runtime_sec": {k: float(np.sum(v)) for k, v in rt.items()},
        "total_bandit_overhead_sec": {k: float(np.sum(v)) for k, v in overhead.items()},
        "total_end_to_end_sec": {k: float(np.sum(rt[k]) + np.sum(overhead[k])) for k in rt},
        "plot": str(plot_path),
        "trace_plot": str(trace_plot_path),
        "trace_keys": list(trace_keys),
        "last500_fraction_equal_default": {name: _fraction_default(name, window=500) for name, *_ in methods},
        "last500_mode_action": {name: _mode_action(name, window=500) for name, *_ in methods},
    }
    summary_path = out_dir / f"test2_runtime_summary_T{T}_interleaved_loss_end2end_{NX}x{NY}_tsallis3d.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("PLOT:", plot_path)
    print("SUMMARY:", summary_path)
    print("TOTAL hypre seconds:", summary["total_hypre_runtime_sec"])
    print("TOTAL overhead seconds:", summary["total_bandit_overhead_sec"])


if __name__ == "__main__":
    main()
