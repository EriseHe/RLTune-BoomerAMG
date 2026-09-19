"""Fix Module 04 failure penalties from independent Default timing samples.

No learner runs here. For each family/grid, the fixed rule is
Lambda = 2 * (median setup time + H * median time per cycle), rounded up
to the next millisecond. The factor two represents a primary budget and
one restarted fallback budget, not an estimate of eventual completion time.
"""
from experiments.diagnostics.solve_control import _project_paths  # noqa: F401

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

from hypre.bindings import create_env
from hypre.bindings.config import augment_setup_params
from hypre.bindings.recovery import validate_runtime_cost
from joint_experiment_config import parse_joint_experiment_config, runtime_config_from_spec
from joint_online_common import _build_paired_instance_stream, configure_paired_environment
from online_td_experiment_common import _write_json
from run_paper_final import ROOT, file_hash, source_state
from setup.space import DEFAULT_SETUP_PARAMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace prescribed calibration: {args.output}")
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Independent calibration only; no learner or benchmark outcome selection",
        "rule": "ceil(1000 * 2 * (median(setup_sec) + H * median(solve_sec / cycles))) / 1000",
        "warmup_solves_per_group": 1,
        "measured_solves_per_group": 8,
        "generator_seeds": [74000123 + 6000 * j for j in range(8)],
        "shuffle_seed": 74048123,
        "source": source_state(),
        "power_state": subprocess.check_output(["pmset", "-g", "batt"], text=True),
        "groups": [],
    }
    for n in (40, 60):
        for family in ("diffusion", "advection"):
            name = f"{family}_{n}_s1"
            config_path = ROOT / "experiments/paper_final/04_online/20260918" / f"{name}.json"
            original = json.loads(config_path.read_text())
            original_runtime = runtime_config_from_spec(parse_joint_experiment_config(original))
            original_stream, _ = _build_paired_instance_stream(original_runtime)
            raw = copy.deepcopy(original)
            raw["stream"].update(cases=8, online_cases=8, cases_per_seed=1,
                                 group_take=8, seed_groups=report["generator_seeds"],
                                 shuffle_seeds=[report["shuffle_seed"]], expected_sha256="")
            runtime = runtime_config_from_spec(parse_joint_experiment_config(raw))
            configure_paired_environment(runtime)
            stream, manifest = _build_paired_instance_stream(runtime)
            original_inputs = {json.dumps(mkw, sort_keys=True) for mkw, _ in original_stream}
            if any(json.dumps(mkw, sort_keys=True) in original_inputs for mkw, _ in stream):
                raise ValueError("Calibration overlaps the benchmark inputs")
            measurements = []
            for index in [-1, *range(len(stream))]:
                mkw, _ = stream[max(0, index)]
                with create_env(**mkw) as env:
                    started = time.perf_counter()
                    solved = env.solve(augment_setup_params(DEFAULT_SETUP_PARAMS),
                                       tol=runtime.tol, max_iter=runtime.max_cycles)
                    elapsed = time.perf_counter() - started
                validate_runtime_cost(solved.runtime_sec)
                if solved.iterations <= 0 or solved.solve_runtime_sec <= 0:
                    raise ValueError("Cannot estimate positive cycle cost from this calibration")
                if index >= 0:
                    measurements.append({
                        "mkw": mkw, "cycles": solved.iterations, "status": solved.status.name,
                        "setup_sec": solved.setup_runtime_sec, "solve_sec": solved.solve_runtime_sec,
                        "per_cycle_sec": solved.solve_runtime_sec / solved.iterations,
                        "solve_call_wall_sec": elapsed,
                        "residual": solved.residual_norm if math.isfinite(solved.residual_norm) else None,
                    })
            setup = statistics.median(row["setup_sec"] for row in measurements)
            cycle = statistics.median(row["per_cycle_sec"] for row in measurements)
            penalty = math.ceil(1000 * 2 * (setup + runtime.max_cycles * cycle)) / 1000
            group = {
                "name": name, "cap": runtime.max_cycles, "penalty_sec": penalty,
                "median_setup_sec": setup, "median_cycle_sec": cycle,
                "source_config_sha256": file_hash(config_path),
                "calibration_stream_sha256": manifest["sha256"],
                "benchmark_input_overlap": 0, "measurements": measurements,
            }
            report["groups"].append(group)
            print(json.dumps({k: group[k] for k in ("name", "cap", "penalty_sec")}), flush=True)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(args.output, report)
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
