"""Locate native timing spikes at individual AMG-cycle granularity."""

from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.diagnostics.solve_control.diagnose_native_timing_outlier import (
    _cpu_number,
    _delta,
    _proc_counters,
    _trace_row,
    _usage,
)
from hypre.bindings import create_env
from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--row", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.episodes <= 0 or args.cycles <= 0:
        raise ValueError("episodes and cycles must be positive")

    source = _trace_row(args.trace.resolve(), args.row)
    records = []
    for episode in range(args.episodes):
        cycles = []
        with create_env(**source["mkw"]) as env:
            preparation = env.prepare_rl(params=source["params"])
            for cycle in range(args.cycles):
                usage_before = _usage()
                proc_counters_before = _proc_counters()
                cpu_time_before = time.process_time()
                hardware_cpu_before = _cpu_number()
                wall_before = time.perf_counter()
                residual, native_sec = env.step_rl(
                    relax_weight=1.0,
                    sweeps_down=1,
                    sweeps_up=1,
                    tol=1.0e-6,
                    max_cycles=args.cycles,
                )
                outer_wall_sec = time.perf_counter() - wall_before
                hardware_cpu_after = _cpu_number()
                process_cpu_sec = time.process_time() - cpu_time_before
                record = {
                    "episode": episode + 1,
                    "cycle": cycle + 1,
                    "native_sec": float(native_sec),
                    "outer_wall_sec": outer_wall_sec,
                    "process_cpu_sec": process_cpu_sec,
                    "wall_minus_cpu_sec": outer_wall_sec - process_cpu_sec,
                    "hardware_cpu_before": hardware_cpu_before,
                    "hardware_cpu_after": hardware_cpu_after,
                    "resource_delta": _delta(_usage(), usage_before),
                    "proc_counter_delta": _delta(
                        _proc_counters(), proc_counters_before
                    ),
                    "residual_norm": float(residual),
                }
                cycles.append(record)
                if env.last_step.status.name != "CONTINUE":
                    break
        episode_record = {
            "episode": episode + 1,
            "setup_sec": preparation.setup_runtime_sec,
            "cycles": cycles,
            "native_cycle_total_sec": float(
                sum(row["native_sec"] for row in cycles)
            ),
            "outer_cycle_total_sec": float(
                sum(row["outer_wall_sec"] for row in cycles)
            ),
        }
        records.append(episode_record)
        print(
            json.dumps(
                _json_ready(
                    {
                        key: value
                        for key, value in episode_record.items()
                        if key != "cycles"
                    }
                ),
                separators=(",", ":"),
            ),
            flush=True,
        )

    flat = [cycle for episode in records for cycle in episode["cycles"]]
    native = np.asarray([cycle["native_sec"] for cycle in flat])
    median = float(np.median(native))
    mad = float(np.median(np.abs(native - median)))
    threshold = median + max(0.002, 8.0 * 1.4826 * mad)
    summary = {
        "trace": str(args.trace.resolve()),
        "row": args.row,
        "arm_index": source.get("arm_index"),
        "episodes": args.episodes,
        "cycles_observed": len(flat),
        "native_cycle_median_sec": median,
        "native_cycle_min_sec": float(np.min(native)),
        "native_cycle_max_sec": float(np.max(native)),
        "slow_cycle_threshold_sec": threshold,
        "slow_cycles": [
            cycle for cycle in flat if cycle["native_sec"] > threshold
        ],
        "records": records,
    }
    _write_json(args.output.resolve(), summary)
    print(
        json.dumps(
            _json_ready(
                {key: value for key, value in summary.items() if key != "records"}
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
