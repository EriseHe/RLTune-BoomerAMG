import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.problem_amg import stencil_27_laplace
from utils.setup_amg import build_actions_th_mxrs_tr

from learners.SharedLinUCB_AMG import SharedLinUCB_AMG
from solver import solve


T = 5000
SEED = 20260211

ALPHA = 1.0
L2 = 1.0

out_dir = Path(__file__).resolve().parent.parent / "plots" / "Combined" / f"seed{SEED}"
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


class SharedLinUCBPolicy:
    def __init__(self, actions, *, alpha: float, l2: float, seed: int):
        self.m = SharedLinUCB_AMG(actions, context_dim=5, alpha=float(alpha), l2_reg=float(l2), seed=int(seed))

    def select(self, context, **_):
        params = self.m.predict(context)
        step = self.m.history[-1]
        return params, {"pred_mean": step.pred_mean, "pred_uncert": step.pred_uncert}

    def update(self, loss, **_):
        self.m.update(float(loss))


def main() -> None:
    fail_penalty = 1e6

    # Warmup: first solve pays MPI/HYPRE init cost; do one untimed warmup.
    warm_rng = np.random.default_rng(SEED ^ 0xBADC0FFE)
    warm_mkw, _, _ = stencil_27_laplace(rng=warm_rng)
    _ = solve(params=DEFAULT_PARAMS, **warm_mkw)

    # Action space: 3 continuous parameters, 19^3 evenly discretized.
    th_grid_19 = np.linspace(0.05, 0.95, 19)
    mxrs_grid_19 = np.linspace(0.05, 0.95, 19)
    tr_grid_19 = np.linspace(0.0, 0.90, 19)  # includes default tr=0.0
    actions_cont3 = build_actions_th_mxrs_tr(
        th_grid_19,
        mxrs_grid_19,
        tr_grid_19,
        fixed_params={"coarsen_type": DEFAULT_PARAMS["coarsen_type"], "interp_type": DEFAULT_PARAMS["interp_type"]},
    )
    parameter_space = {"actions": actions_cont3, "context_dim": 5}

    # Policies (same algorithm, different loss used for update)
    policy_default = FixedPolicy(DEFAULT_PARAMS)
    policy_loss_hypre = SharedLinUCBPolicy(actions_cont3, alpha=ALPHA, l2=L2, seed=SEED + 10001)
    policy_loss_end2end = SharedLinUCBPolicy(actions_cont3, alpha=ALPHA, l2=L2, seed=SEED + 10001)

    methods = [
        ("default (fixed)", policy_default),
        ("shared LinUCB (loss=hypre)", policy_loss_hypre),
        ("shared LinUCB (loss=hypre+overhead)", policy_loss_end2end),
    ]

    rt = {name: np.zeros(T, dtype=float) for name, _ in methods}
    overhead = {name: np.zeros(T, dtype=float) for name, _ in methods}

    # For the overhead-in-loss variant, include an estimate of update time
    # from the previous round to avoid circularity (loss depends on update time,
    # but update depends on loss).
    prev_update_est = {name: 0.0 for name, _ in methods}

    rng_inst = np.random.default_rng(SEED)
    rng_order = np.random.default_rng(SEED ^ 0xA5A5A5A5)

    def safe_solve(params: dict, mkw: dict) -> dict:
        try:
            res = solve(params=params, **mkw)
            return {"wu": float(res.work_units), "runtime": float(res.runtime_sec)}
        except Exception:
            return {"wu": float(fail_penalty), "runtime": 1e9}

    for t in range(T):
        mkw, context, meta = stencil_27_laplace(rng=rng_inst)
        order = rng_order.permutation(len(methods))

        for i in order:
            name, policy = methods[int(i)]

            sel_start = time.perf_counter_ns()
            selected = policy.select(context=context, parameter_space=parameter_space)
            sel_sec = (time.perf_counter_ns() - sel_start) / 1e9

            if isinstance(selected, tuple):
                params = selected[0]
            else:
                params = selected

            out = safe_solve(params, mkw)
            rt[name][t] = float(out["runtime"])

            loss_start = time.perf_counter_ns()
            hypre_loss = float(
                runtime_loss_sec(
                    outcome=out,
                    context=context,
                    params=params,
                    meta=meta,
                    mkw=mkw,
                    t=t + 1,
                    trial=1,
                    fail_penalty=fail_penalty,
                )
            )
            loss_eval_sec = (time.perf_counter_ns() - loss_start) / 1e9

            if name == "shared LinUCB (loss=hypre+overhead)":
                loss_value = hypre_loss + float(sel_sec) + float(prev_update_est[name])
            else:
                loss_value = hypre_loss

            upd_sec = 0.0
            if hasattr(policy, "update"):
                upd_start = time.perf_counter_ns()
                policy.update(loss=loss_value, context=context, params=params, outcome=out)
                upd_sec = (time.perf_counter_ns() - upd_start) / 1e9

            prev_update_est[name] = float(upd_sec)
            overhead[name][t] = float(sel_sec + loss_eval_sec + upd_sec)

    y = T - np.arange(1, T + 1)
    cum_default = np.cumsum(rt["default (fixed)"] + overhead["default (fixed)"])
    cum_hypre = np.cumsum(rt["shared LinUCB (loss=hypre)"] + overhead["shared LinUCB (loss=hypre)"])
    cum_end2end = np.cumsum(rt["shared LinUCB (loss=hypre+overhead)"] + overhead["shared LinUCB (loss=hypre+overhead)"])

    plt.figure(figsize=(10, 5))
    plt.plot(cum_default, y, "--", linewidth=2.0, label="default (fixed)")
    plt.plot(cum_hypre, y, linewidth=2.3, label="shared LinUCB (loss=hypre)")
    plt.plot(cum_end2end, y, linewidth=2.3, label="shared LinUCB (loss=hypre+overhead)")
    plt.xlabel("cumulative runtime (seconds)")
    plt.ylabel("instances remaining")
    plt.title(f"BoomerAMG Setup runtime cumulative (loss compare)  T={T}  2D stencil_27_laplace", fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = out_dir / "test2_loss_compare_shared_linucb_T5000.png"
    plt.savefig(plot_path, dpi=256)
    plt.close()

    summary = {
        "T": int(T),
        "seed": int(SEED),
        "alpha": float(ALPHA),
        "l2": float(L2),
        "actions_cont3": int(len(actions_cont3)),
        "note": "loss=hypre+overhead uses (hypre_loss + select_time + prev_update_time_estimate) to avoid circularity",
        "total_hypre_runtime_sec": {k: float(np.sum(v)) for k, v in rt.items()},
        "total_bandit_overhead_sec": {k: float(np.sum(v)) for k, v in overhead.items()},
        "total_end_to_end_sec": {k: float(np.sum(rt[k]) + np.sum(overhead[k])) for k in rt},
        "plot": str(plot_path),
    }
    summary_path = out_dir / "test2_loss_compare_shared_linucb_T5000.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("PLOT:", plot_path)
    print("SUMMARY:", summary_path)
    print("TOTAL end-to-end seconds:", summary["total_end_to_end_sec"])


if __name__ == "__main__":
    main()

