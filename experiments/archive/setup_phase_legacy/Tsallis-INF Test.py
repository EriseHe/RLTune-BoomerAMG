import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from learners.TsallisINF_AMG import TsallisINF_AMG
from solver import solve
from utils.problem_amg import stencil_27_laplace
from utils.setup_amg import build_actions_th_mxrs_tr


T = 5000
SEED = 20260209
NX, NY, NZ = 60, 60, 1

# Tsallis-INF expects loss >= 1 and updates with (loss - 1). Runtime per round
# is typically < 1s, so scale it up for stability.
LOSS_SCALE = 1e3

out_dir = Path(__file__).resolve().parent.parent / "plots" / "Combined"
out_dir.mkdir(parents=True, exist_ok=True)


DEFAULT_PARAMS = {
    "strong_threshold": 0.25,
    "max_row_sum": 0.90,
    "trunc_factor": 0.00,
    "coarsen_type": 10,
    "interp_type": 6,
}


def runtime_loss_sec(*, outcome, fail_penalty: float) -> float:
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

    def select(self, **_):
        return dict(self._params), {}

    def update(self, **_):
        return None


class TsallisPolicy:
    def __init__(self, actions, *, T: int, seed: int):
        self._actions = [dict(a) for a in actions]
        # Use a per-policy RNG so randomized evaluation order does not couple policies.
        self._bandit = TsallisINF_AMG(np.asarray(self._actions, dtype=object), int(T), seed=int(seed))

    def select(self, **_):
        params = self._bandit.predict()
        return dict(params), {}

    def update(self, *, loss: float, **_):
        self._bandit.update(float(loss))


def main() -> None:
    fail_penalty = 1e6

    # Warmup solve: avoid measuring one-time HYPRE init costs.
    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_27_laplace(rng=warm_rng, nx=NX, ny=NY, nz=NZ)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    fixed_params = {"coarsen_type": DEFAULT_PARAMS["coarsen_type"], "interp_type": DEFAULT_PARAMS["interp_type"]}

    # 1) Tsallis th-only (3)
    th_grid_3 = np.array([0.05, 0.25, 0.45], dtype=float)
    actions_th_3 = build_actions_th_mxrs_tr(th_grid_3, [DEFAULT_PARAMS["max_row_sum"]], [DEFAULT_PARAMS["trunc_factor"]], fixed_params=fixed_params)

    # 2) Tsallis th-only (19)
    th_grid_19 = np.linspace(0.05, 0.95, 19)
    actions_th_19 = build_actions_th_mxrs_tr(th_grid_19, [DEFAULT_PARAMS["max_row_sum"]], [DEFAULT_PARAMS["trunc_factor"]], fixed_params=fixed_params)

    # 3) Tsallis cont3 (5^3)
    th_5 = np.linspace(0.05, 0.85, 5)    # includes default th=0.25
    mxrs_5 = np.linspace(0.10, 0.90, 5)  # includes default mxrs=0.90
    tr_5 = np.linspace(0.00, 0.80, 5)    # includes default tr=0.00
    actions_cont3_5 = build_actions_th_mxrs_tr(th_5, mxrs_5, tr_5, fixed_params=fixed_params)

    # 4) Tsallis cont3 (19^3)
    th_19 = np.linspace(0.05, 0.95, 19)
    mxrs_19 = np.linspace(0.10, 0.90, 19)
    tr_19 = np.linspace(0.00, 0.80, 19)
    actions_cont3_19 = build_actions_th_mxrs_tr(th_19, mxrs_19, tr_19, fixed_params=fixed_params)

    methods = [
        ("default (fixed)", FixedPolicy(DEFAULT_PARAMS), {"actions": [DEFAULT_PARAMS]}, False),
        ("Tsallis-INF: th-only (3)", TsallisPolicy(actions_th_3, T=T, seed=SEED + 10000), {"actions": actions_th_3}, True),
        ("Tsallis-INF: th-only (19)", TsallisPolicy(actions_th_19, T=T, seed=SEED + 10001), {"actions": actions_th_19}, True),
        ("Tsallis-INF: (th,mxrs,tr) 5^3", TsallisPolicy(actions_cont3_5, T=T, seed=SEED + 10002), {"actions": actions_cont3_5}, True),
        ("Tsallis-INF: (th,mxrs,tr) 19^3", TsallisPolicy(actions_cont3_19, T=T, seed=SEED + 10003), {"actions": actions_cont3_19}, True),
    ]

    rt = {name: np.zeros(T, dtype=float) for name, *_ in methods}
    overhead = {name: np.zeros(T, dtype=float) for name, *_ in methods}

    rng_inst = np.random.default_rng(SEED)
    rng_order = np.random.default_rng(SEED ^ 0xA5A5A5A5)

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

            params = selected[0] if isinstance(selected, tuple) else selected

            out = safe_solve(params, mkw)
            rt[name][t] = float(out["runtime"])

            loss_start = time.perf_counter_ns()
            base_loss_sec = float(runtime_loss_sec(outcome=out, fail_penalty=fail_penalty))
            loss_sec = (time.perf_counter_ns() - loss_start) / 1e9

            upd_sec = 0.0
            if is_tsallis:
                # Learning loss: hypre-only runtime (solver walltime).
                loss_value = 1.0 + float(LOSS_SCALE) * base_loss_sec
                upd_start = time.perf_counter_ns()
                policy.update(loss=loss_value, context=context, params=params, outcome=out)
                upd_sec = (time.perf_counter_ns() - upd_start) / 1e9
            overhead[name][t] = float(sel_sec + loss_sec + upd_sec)

    y = T - np.arange(1, T + 1)
    plt.figure(figsize=(10, 5))
    for name, *_ in methods:
        cum = np.cumsum(rt[name] + overhead[name])
        if name == "default (fixed)":
            plt.plot(cum, y, "--", linewidth=2.0, label=name)
        else:
            plt.plot(cum, y, linewidth=2.3, label=name)

    plt.xlabel("cumulative runtime (seconds)")
    plt.ylabel("instances remaining")
    plt.title(f"Tsallis-INF Test  T={T}  2D {NX}x{NY} stencil_27_laplace  [loss=hypre-only runtime]", fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = out_dir / f"tsallis_inf_test_runtime_cumulative_T{T}_{NX}x{NY}_seed{SEED}.png"
    plt.savefig(plot_path, dpi=256)
    plt.close()

    summary = {
        "T": int(T),
        "seed": int(SEED),
        "nx": int(NX),
        "ny": int(NY),
        "nz": int(NZ),
        "loss_scale": float(LOSS_SCALE),
        "loss": "hypre_only_runtime",
        "actions": {
            "th_3": int(len(actions_th_3)),
            "th_19": int(len(actions_th_19)),
            "cont3_5": int(len(actions_cont3_5)),
            "cont3_19": int(len(actions_cont3_19)),
        },
        "total_hypre_runtime_sec": {k: float(np.sum(v)) for k, v in rt.items()},
        "total_bandit_overhead_sec": {k: float(np.sum(v)) for k, v in overhead.items()},
        "total_end_to_end_sec": {k: float(np.sum(rt[k]) + np.sum(overhead[k])) for k in rt},
        "plot": str(plot_path),
    }
    summary_path = out_dir / f"tsallis_inf_test_runtime_summary_T{T}_{NX}x{NY}_seed{SEED}.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("PLOT:", plot_path)
    print("SUMMARY:", summary_path)
    print("TOTAL hypre seconds:", summary["total_hypre_runtime_sec"])
    print("TOTAL overhead seconds:", summary["total_bandit_overhead_sec"])


if __name__ == "__main__":
    main()
