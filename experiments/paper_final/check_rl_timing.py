"""Compare native timer coverage before/after the fix on two frozen 80-cubed cases.

No learning: replay saved matrices, setups and actions, alternating binary order.
Outer-minus-native time measures omitted work within the same call; differences
in raw solve speed between runs are not used as evidence of timer coverage.
"""
from experiments.diagnostics.solve_control import _project_paths  # noqa: F401

import argparse
import ctypes
import hashlib
from pathlib import Path
import time

import numpy as np

from diagnose_native_timing_outlier import _trace_row
from hypre.bindings import boomeramg, create_env
from hypre.bindings.config import augment_setup_params, configure_smoother_profile
from online_td_experiment_common import _write_json


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=root / "results/paper_final/01_numerics/timing")
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 2 or args.repeats % 2:
        parser.error("--repeats must be a positive even number for balanced order")
    output = args.output_dir / "replay.json"
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    configure_smoother_profile("l1_jacobi_direct_coarse")
    current = boomeramg._lib
    paths = {
        "before": args.output_dir / "before/libamg_runtime.dylib",
        "after": root / "hypre/interfaces/libamg_runtime.dylib",
    }
    baseline = ctypes.CDLL(str(paths["before"].resolve()))
    for name, function in current.__dict__.copy().items():
        if name.startswith("amg_"):
            getattr(baseline, name).argtypes = function.argtypes
            getattr(baseline, name).restype = function.restype
    libraries = {"before": baseline, "after": current}
    sources, records = {}, []
    try:
        for seed in (1, 2):
            trace = root / f"results/paper_final/03_activation/advection_80_s{seed}/trajectories/start_750.jsonl"
            source = _trace_row(trace, 4000)
            assert source["online_index"] == 4000
            assert source["outcome"]["first_primary_status"] == "success"
            sources[seed] = {
                "trace": str(trace.relative_to(root)), "case": 4001,
                "mkw": source["mkw"], "params": source["params"],
                "actions": source["outcome"]["cycle_actions"],
                "recorded_residuals": source["outcome"]["cycle_residuals"],
            }
            for repeat in range(args.repeats):
                order = ("before", "after") if repeat % 2 == 0 else ("after", "before")
                for rank, version in enumerate(order):
                    boomeramg._lib = libraries[version]
                    with create_env(**source["mkw"]) as env:
                        started = time.perf_counter()
                        prepared = env.prepare_rl(params=augment_setup_params(source["params"]))
                        prepare_outer = time.perf_counter() - started
                        cycles = []
                        for weight in sources[seed]["actions"]:
                            started = time.perf_counter()
                            residual, native = env.step_rl(
                                relax_weight=weight, sweeps_down=1, sweeps_up=1,
                                tol=1e-6, max_cycles=50,
                            )
                            outer = time.perf_counter() - started
                            cycles.append({"native_sec": native, "outer_sec": outer,
                                           "residual": residual, "status": env.last_step.status.name})
                    np.testing.assert_allclose(
                        [c["residual"] for c in cycles], sources[seed]["recorded_residuals"],
                        rtol=1e-8, atol=1e-14,
                    )
                    assert cycles[-1]["status"] == "CONVERGED"
                    records.append({
                        "seed": seed, "repeat": repeat, "version": version, "rank": rank,
                        "setup_native_sec": prepared.setup_runtime_sec,
                        "prepare_outer_sec": prepare_outer,
                        "initial_residual": prepared.absolute_initial_residual_norm,
                        "cycles": cycles, "residuals_match_saved": True,
                    })
            print(f"s{seed}: {2 * args.repeats} frozen solves matched saved residuals", flush=True)
    finally:
        boomeramg._lib = current
    summary = {}
    for version in libraries:
        selected = [r for r in records if r["version"] == version]
        cycles = [c for r in selected for c in r["cycles"]]
        summary[version] = {
            "episodes": len(selected), "cycles": len(cycles),
            "mean_native_step_ms": float(1000 * np.mean([c["native_sec"] for c in cycles])),
            "mean_outer_step_ms": float(1000 * np.mean([c["outer_sec"] for c in cycles])),
            "mean_unrecorded_step_ms": float(1000 * np.mean([
                c["outer_sec"] - c["native_sec"] for c in cycles])),
            "mean_unrecorded_prepare_ms": float(1000 * np.mean([
                r["prepare_outer_sec"] - r["setup_native_sec"] for r in selected])),
        }
    _write_json(output, {
        "passed": True, "development_only": True,
        "binary_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in paths.items()},
        "sources": sources, "summary": summary, "records": records,
        "limitation": "Two frozen cases, no training. Remaining outer-call overhead is outside native timers.",
    })
    print(summary, flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    main()
