"""Replay two recorded cycle-50 boundary cases against linked BoomerAMG.

Fixed weight and saved setups only; no learner updates or experiment reruns.
"""

from experiments.paper_final.common.artifacts import ROOT as REPOSITORY_ROOT

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

from pathlib import Path

from experiments.diagnostics.solve_control.diagnose_native_timing_outlier import _trace_row
from hypre.bindings import SolveStatus, create_env
from hypre.bindings.config import augment_setup_params, configure_smoother_profile
from experiments.joint.solve_control.online_td_experiment_common import _write_json
from solve.core.outcomes import classify_rl_failure


def main():
    root = REPOSITORY_ROOT
    configure_smoother_profile("l1_jacobi_direct_coarse")
    records = []
    for seed, case in ((1, 255), (2, 35)):
        trace = root / f"results/paper_final/03_activation/advection_80_s{seed}/trajectories/start_750.jsonl"
        row = _trace_row(trace, case-1)
        assert not row["outcome"].get("cycle_actions")
        params = augment_setup_params(row["params"])
        with create_env(**row["mkw"]) as env:
            native = env.solve(params, tol=1e-6, max_iter=50)
            env.prepare_rl(params)
            for cycles in range(1, 51):
                residual, _ = env.step_rl(relax_weight=1.0, sweeps_down=1,
                                          sweeps_up=1, tol=1e-6, max_cycles=50)
                if env.last_step.status is not SolveStatus.CONTINUE:
                    break
            status = env.last_step.status
            with_headroom = env.solve(params, tol=1e-6, max_iter=51)
        reason = classify_rl_failure(residual_norm=residual, iterations=cycles,
                                     solve_tol=1e-6, solve_max_cycles=50)
        assert native.iterations == cycles == with_headroom.iterations == 50
        assert native.status is status is SolveStatus.MAX_CYCLES
        assert with_headroom.status is SolveStatus.CONVERGED
        assert residual < 1e-6 and abs(residual-native.residual_norm) < 1e-12
        assert reason == "max_cycles_reached_without_convergence"
        records.append({"source": str(trace.relative_to(root)), "case": case,
                        "mkw": row["mkw"], "params": row["params"],
                        "cycles": cycles, "full_solve_status": native.status.name,
                        "rl_step_status": status.name, "rl_failure_reason": reason,
                        "full_residual": native.residual_norm, "rl_residual": residual,
                        "full_status_with_budget_51": with_headroom.status.name,
                        "budget_51_cycles": with_headroom.iterations})
    output = root / "results/paper_final/01_numerics/stopping_alignment/historical_cases.json"
    _write_json(output, {"passed": True, "native_solves": 6, "cases": records})
    print(output)


if __name__ == "__main__":
    main()
