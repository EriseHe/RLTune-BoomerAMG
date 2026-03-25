import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.setup_amg import run_amg_setup_experiment
from utils.setup_amg import build_actions_th_coarsen_interp, build_actions_th_mxrs_tr
from utils.stencil27_laplace import CONTEXT_DIM, stencil_27_laplace
from learners.LinUCB_AMG import LinUCB_AMG

T = 5000
TRIALS = 1
SEED = 20260209

out_dir = Path(__file__).resolve().parent.parent / "plots" / "Combined"
out_dir.mkdir(parents=True, exist_ok=True)

default_params = {
    "strong_threshold": 0.25,
    "max_row_sum": 0.90,
    "trunc_factor": 0.00,
    "coarsen_type": 10,
    "interp_type": 6,
}
baselines = [("default", default_params)]

def runtime_loss_ms(*, outcome, **kwargs):
    return float(outcome["runtime"]) * 1000.0

class FixedPolicy:
    def __init__(self, params):
        self._params = dict(params)
    def select(self, context, **kwargs):
        return dict(self._params), {}
    def update(self, loss, **kwargs):
        return None

class LinFactory:
    def __init__(self, alpha=1.0, l2=1.0):
        self.alpha = alpha
        self.l2 = l2
    def new_trial(self, *, parameter_space, seed, **kwargs):
        class P:
            def __init__(self, actions, alpha, l2, seed):
                self.m = LinUCB_AMG(actions, context_dim=CONTEXT_DIM, alpha=float(alpha), l2_reg=float(l2), seed=int(seed))
            def select(self, context, **kwargs):
                return self.m.predict(context), {}
            def update(self, loss, **kwargs):
                self.m.update(float(loss))
        return P(parameter_space["actions"], self.alpha, self.l2, seed)

# Default baseline timed independently (no interleaving bias)
res_default = run_amg_setup_experiment(
    stencil_27_laplace,
    {"actions": [default_params], "context_dim": CONTEXT_DIM},
    FixedPolicy(default_params),
    runtime_loss_ms,
    T=T, trials=TRIALS, seed=SEED, baselines=[],
)

# Case A: LinUCB tune 3 continuous params, 5x5x5
th_grid = [0.10, 0.25, 0.40, 0.55, 0.70]
mxrs_grid = [0.10, 0.30, 0.50, 0.70, 0.90]
tr_grid = [0.00, 0.05, 0.10, 0.15, 0.20]
actions_cont3 = build_actions_th_mxrs_tr(
    th_grid,
    mxrs_grid,
    tr_grid,
    fixed_params={"coarsen_type": 10, "interp_type": 6},
)

res_cont3 = run_amg_setup_experiment(
    stencil_27_laplace,
    {"actions": actions_cont3, "context_dim": CONTEXT_DIM},
    LinFactory(alpha=1.0, l2=1.0),
    runtime_loss_ms,
    T=T, trials=TRIALS, seed=SEED, baselines=[],
)

# Case B: LinUCB tune strong_threshold only, ORIGINAL grid resolution (19)
th_grid_original = np.linspace(0.05, 0.95, 19)
actions_th = build_actions_th_coarsen_interp(
    th_grid_original,
    [10],
    [6],
    fixed_params={"max_row_sum": 0.90, "trunc_factor": 0.00},
)

res_th = run_amg_setup_experiment(
    stencil_27_laplace,
    {"actions": actions_th, "context_dim": CONTEXT_DIM},
    LinFactory(alpha=1.0, l2=1.0),
    runtime_loss_ms,
    T=T, trials=TRIALS, seed=SEED, baselines=[],
)

# Combined cumulative runtime plot
y = T - np.arange(1, T + 1)
cum_default = np.mean(np.cumsum(res_default["bandit_runtime"], axis=0), axis=1)
cum_cont3 = np.mean(np.cumsum(res_cont3["bandit_runtime"], axis=0), axis=1)
cum_th = np.mean(np.cumsum(res_th["bandit_runtime"], axis=0), axis=1)

plt.figure(figsize=(9, 5))
plt.plot(cum_default, y, "--", linewidth=2.0, label="default")
plt.plot(cum_cont3, y, linewidth=2.5, label="LinUCB: (th,mxrs,tr) 5x5x5")
plt.plot(cum_th, y, linewidth=2.5, label="LinUCB: th-only (19-grid)")
plt.xlabel("cumulative runtime (seconds)")
plt.ylabel("instances remaining")
plt.title("BoomerAMG Setup runtime cumulative (T=5000, 60x60x1)")
plt.legend()
plt.tight_layout()

plot_path = out_dir / "runtime_cumulative_T5000_linucb_cont3_5x5x5_vs_linucb_th19_vs_default.png"
plt.savefig(plot_path, dpi=256)
plt.close()

summary = {
    "T": T,
    "seed": SEED,
    "totals_sec": {
        "default": float(np.sum(res_default["bandit_runtime"])),
        "linucb_cont3_5x5x5": float(np.sum(res_cont3["bandit_runtime"])),
        "linucb_th_only_19grid": float(np.sum(res_th["bandit_runtime"])),
    },
    "plot": str(plot_path),
}
summary_path = out_dir / "runtime_summary_T5000_linucb_cont3_5x5x5_vs_linucb_th19_vs_default.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

print("PLOT:", plot_path)
print("SUMMARY:", summary_path)
print("TOTAL seconds:", summary["totals_sec"])
