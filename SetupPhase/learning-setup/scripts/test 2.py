import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.problem_amg import stencil_27_laplace
from utils.setup_amg import build_actions_th_mxrs_tr, run_amg_setup_experiment

from learners.LinUCB_AMG import LinUCB_AMG
from learners.SharedLinUCB_AMG import SharedLinUCB_AMG
from learners.TsallisINF_AMG import TsallisINF_AMG


T = 5000
TRIALS = 1
SEED = 20260209

ALPHA = 1.0
L2 = 1.0

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


class TsallisThPolicy:
    def __init__(self, grid: np.ndarray, *, T: int, default_params: dict, seed: int):
        np.random.seed(int(seed))  # TsallisINF_AMG uses NumPy global RNG
        self._default_params = dict(default_params)
        self._bandit = TsallisINF_AMG(np.asarray(grid, dtype=float), int(T))

    def select(self, context, **_):
        th = float(self._bandit.predict())
        params = dict(self._default_params)
        params["strong_threshold"] = th
        return params, {}

    def update(self, loss, **_):
        self._bandit.update(float(loss))


def main() -> None:
    fail_penalty = 1e6

    # 2D: utils.problem_amg.NZ defaults to 1, so stencil_27_laplace is already 60x60x1.
    # Grids: evenly spaced, within [0,1]. Use 0.05 steps where possible.
    th_grid_19 = np.linspace(0.05, 0.95, 19)
    mxrs_grid_19 = np.linspace(0.05, 0.95, 19)
    tr_grid_19 = np.linspace(0.0, 0.90, 19)  # includes default tr=0.0

    # 19^3 actions for (th,mxrs,tr).
    actions_cont3 = build_actions_th_mxrs_tr(
        th_grid_19,
        mxrs_grid_19,
        tr_grid_19,
        fixed_params={"coarsen_type": DEFAULT_PARAMS["coarsen_type"], "interp_type": DEFAULT_PARAMS["interp_type"]},
    )

    # 19 actions for th-only (shared LinUCB), others fixed to default.
    actions_th_only = build_actions_th_mxrs_tr(
        th_grid_19,
        [DEFAULT_PARAMS["max_row_sum"]],
        [DEFAULT_PARAMS["trunc_factor"]],
        fixed_params={"coarsen_type": DEFAULT_PARAMS["coarsen_type"], "interp_type": DEFAULT_PARAMS["interp_type"]},
    )

    # Baseline: fixed default parameters (timed independently).
    res_default = run_amg_setup_experiment(
        stencil_27_laplace,
        {"actions": [DEFAULT_PARAMS], "context_dim": 5},
        FixedPolicy(DEFAULT_PARAMS),
        lambda **kw: runtime_loss_sec(fail_penalty=fail_penalty, **kw),
        T=T,
        trials=TRIALS,
        seed=SEED,
        baselines=[],
        fail_penalty=fail_penalty,
    )

    # 1) Disjoint LinUCB on 3 continuous parameters.
    res_disjoint = run_amg_setup_experiment(
        stencil_27_laplace,
        {"actions": actions_cont3, "context_dim": 5},
        DisjointLinUCBFactory(ALPHA, L2),
        lambda **kw: runtime_loss_sec(fail_penalty=fail_penalty, **kw),
        T=T,
        trials=TRIALS,
        seed=SEED,
        baselines=[],
        fail_penalty=fail_penalty,
    )

    # 2) Shared LinUCB on 3 continuous parameters.
    res_shared = run_amg_setup_experiment(
        stencil_27_laplace,
        {"actions": actions_cont3, "context_dim": 5},
        SharedLinUCBFactory(ALPHA, L2),
        lambda **kw: runtime_loss_sec(fail_penalty=fail_penalty, **kw),
        T=T,
        trials=TRIALS,
        seed=SEED,
        baselines=[],
        fail_penalty=fail_penalty,
    )

    # 3) Shared LinUCB on th-only, others default.
    res_shared_th = run_amg_setup_experiment(
        stencil_27_laplace,
        {"actions": actions_th_only, "context_dim": 5},
        SharedLinUCBFactory(ALPHA, L2),
        lambda **kw: runtime_loss_sec(fail_penalty=fail_penalty, **kw),
        T=T,
        trials=TRIALS,
        seed=SEED,
        baselines=[],
        fail_penalty=fail_penalty,
    )

    # 4) Tsallis-INF on th-only (19 arms), others default.
    res_tsallis_th = run_amg_setup_experiment(
        stencil_27_laplace,
        {"actions": [], "context_dim": 5},
        TsallisThPolicy(th_grid_19, T=T, default_params=DEFAULT_PARAMS, seed=SEED + 3),
        # TsallisINF_AMG expects loss >= 1 (it updates with (loss - 1)).
        lambda **kw: 1.0 + 1e3 * runtime_loss_sec(fail_penalty=fail_penalty, **kw),
        T=T,
        trials=TRIALS,
        seed=SEED,
        baselines=[],
        fail_penalty=fail_penalty,
    )

    y = T - np.arange(1, T + 1)
    cum_default = np.mean(np.cumsum(res_default["bandit_total_runtime"], axis=0), axis=1)
    cum_disjoint = np.mean(np.cumsum(res_disjoint["bandit_total_runtime"], axis=0), axis=1)
    cum_shared = np.mean(np.cumsum(res_shared["bandit_total_runtime"], axis=0), axis=1)
    cum_shared_th = np.mean(np.cumsum(res_shared_th["bandit_total_runtime"], axis=0), axis=1)
    cum_tsallis_th = np.mean(np.cumsum(res_tsallis_th["bandit_total_runtime"], axis=0), axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(cum_default, y, "--", linewidth=2.0, label="default (fixed)")
    plt.plot(cum_disjoint, y, linewidth=2.3, label="LinUCB disjoint: (th,mxrs,tr) 19^3")
    plt.plot(cum_shared, y, linewidth=2.3, label="LinUCB shared: (th,mxrs,tr) 19^3")
    plt.plot(cum_shared_th, y, linewidth=2.3, label="LinUCB shared: th-only (19)")
    plt.plot(cum_tsallis_th, y, linewidth=2.3, label="Tsallis-INF: th-only (19)")

    plt.xlabel("cumulative runtime (seconds)")
    plt.ylabel("instances remaining")
    plt.title(f"BoomerAMG Setup runtime cumulative (test 2)  T={T}  2D stencil_27_laplace  [includes bandit overhead]", fontsize=12)
    plt.legend(fontsize=9)
    plt.tight_layout()

    plot_path = out_dir / "test2_runtime_cumulative_T5000_disjoint_vs_shared_cont3_vs_shared_th_vs_tsallis_th_vs_default.png"
    plt.savefig(plot_path, dpi=256)
    plt.close()

    summary = {
        "T": int(T),
        "seed": int(SEED),
        "alpha": float(ALPHA),
        "l2": float(L2),
        "actions_cont3": int(len(actions_cont3)),
        "actions_th_only": int(len(actions_th_only)),
        "total_runtime_sec": {
            "default": float(np.sum(res_default["bandit_total_runtime"])),
            "linucb_disjoint_cont3": float(np.sum(res_disjoint["bandit_total_runtime"])),
            "linucb_shared_cont3": float(np.sum(res_shared["bandit_total_runtime"])),
            "linucb_shared_th_only": float(np.sum(res_shared_th["bandit_total_runtime"])),
            "tsallis_th_only": float(np.sum(res_tsallis_th["bandit_total_runtime"])),
        },
        "total_overhead_sec": {
            "default": float(np.sum(res_default["bandit_overhead_runtime"])),
            "linucb_disjoint_cont3": float(np.sum(res_disjoint["bandit_overhead_runtime"])),
            "linucb_shared_cont3": float(np.sum(res_shared["bandit_overhead_runtime"])),
            "linucb_shared_th_only": float(np.sum(res_shared_th["bandit_overhead_runtime"])),
            "tsallis_th_only": float(np.sum(res_tsallis_th["bandit_overhead_runtime"])),
        },
        "plot": str(plot_path),
    }
    summary_path = out_dir / "test2_runtime_summary_T5000.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("PLOT:", plot_path)
    print("SUMMARY:", summary_path)
    print("TOTAL seconds:", summary["total_runtime_sec"])


if __name__ == "__main__":
    main()
