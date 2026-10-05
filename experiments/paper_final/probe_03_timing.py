"""Bounded frozen native replay at roster sizes 1, 3 and 6; no training.

Each slot repeats the same saved matrix, setup and finite action sequence.
All six permutations of roster sizes counterbalance the order of the blocks.
This isolates immediate native timing; it cannot replay historical thermal
conditions, learner memory pressure or the adaptive effect of timing feedback.
"""

from experiments.runtime import configure_single_thread

if __name__ == "__main__":
    configure_single_thread()

import argparse
from itertools import permutations
from pathlib import Path
import time

import numpy as np

from experiments.diagnostics.solve_control.diagnose_native_timing_outlier import _cpu_number, _trace_row
from hypre.bindings import create_env
from hypre.bindings.config import augment_setup_params, configure_smoother_profile
from experiments.joint.solve_control.online_td_experiment_common import _write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/paper_final/03_activation"))
    parser.add_argument("--case", type=int, default=4001)
    args = parser.parse_args()
    configure_smoother_profile("l1_jacobi_direct_coarse")
    output = args.root / "advection_80_s2/analysis/timing_probe.json"
    records, sources = [], {}
    for seed in (1, 2):
        trace = args.root / f"advection_80_s{seed}/trajectories/start_750.jsonl"
        source = _trace_row(trace, args.case-1)
        actions = source["outcome"]["cycle_actions"]
        assert source["online_index"] == args.case-1
        assert source["outcome"]["first_primary_status"] == "success"
        sources[seed] = {"trace": str(trace), "case": args.case, "mkw": source["mkw"],
                         "params": source["params"], "actions": actions,
                         "recorded_residuals": source["outcome"]["cycle_residuals"],
                         "recorded_cycle_ms": 1000*np.mean(source["outcome"]["cycle_times"])}
        for repeat, sizes in enumerate(permutations((1, 3, 6))):
            for roster_size in sizes:
                for rank in range(roster_size):
                    cpu_before = _cpu_number()
                    started = time.perf_counter()
                    cpu_started = time.process_time()
                    with create_env(**source["mkw"]) as env:
                        prep_start = time.perf_counter()
                        prep = env.prepare_rl(params=augment_setup_params(source["params"]))
                        prep_wall = time.perf_counter()-prep_start
                        cycles = []
                        for weight in actions:
                            cycle_start = time.perf_counter()
                            residual, native = env.step_rl(relax_weight=weight, sweeps_down=1,
                                                          sweeps_up=1, tol=1e-6, max_cycles=50)
                            cycles.append({"native_sec": native, "outer_sec": time.perf_counter()-cycle_start,
                                           "residual": residual, "status": env.last_step.status.name})
                    record = {"source_seed": seed, "repeat": repeat, "roster_size": roster_size,
                              "rank": rank, "cycles": cycles, "setup_native_sec": prep.setup_runtime_sec,
                              "prepare_outer_sec": prep_wall, "episode_outer_sec": time.perf_counter()-started,
                              "episode_cpu_sec": time.process_time()-cpu_started,
                              "hardware_cpu_before": cpu_before, "hardware_cpu_after": _cpu_number()}
                    record["residuals_match_saved"] = bool(np.allclose(
                        [c["residual"] for c in cycles], source["outcome"]["cycle_residuals"],
                        rtol=1e-8, atol=1e-14))
                    records.append(record)
            print(f"source s{seed}: counterbalanced block {repeat+1}/6 complete", flush=True)
    summary = {}
    for seed in (1, 2):
        summary[seed] = {}
        for size in (1, 3, 6):
            selected = [r for r in records if r["source_seed"] == seed and r["roster_size"] == size]
            native = [c["native_sec"] for r in selected for c in r["cycles"]]
            outer = [c["outer_sec"] for r in selected for c in r["cycles"]]
            summary[seed][size] = {
                "episodes": len(selected), "cycles": len(native),
                "mean_native_cycle_ms": float(1000*np.mean(native)),
                "median_native_cycle_ms": float(1000*np.median(native)),
                "mean_step_outer_ms": float(1000*np.mean(outer)),
                "mean_unrecorded_step_ms": float(1000*np.mean(np.subtract(outer, native))),
                "mean_native_setup_ms": float(1000*np.mean([r["setup_native_sec"] for r in selected])),
                "all_residuals_match_saved": all(r["residuals_match_saved"] for r in selected),
                "rank_cycle_ms": {rank: float(1000*np.mean([c["native_sec"] for r in selected
                                     if r["rank"] == rank for c in r["cycles"]])) for rank in range(size)},
            }
    _write_json(output, {"development_only": True, "native_episodes": len(records),
                         "sources": sources, "summary": summary, "records": records,
                         "limitation": "Immediate frozen native replay; not a controlled reproduction of two full adaptive runs."})
    print(output, flush=True)


if __name__ == "__main__":
    main()
