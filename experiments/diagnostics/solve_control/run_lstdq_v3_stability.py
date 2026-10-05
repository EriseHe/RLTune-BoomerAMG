"""Audit LSTDQ v2/v3 sensitivity to method execution order."""

from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np

from experiments.joint.solve_control.joint_online_common import (
    _method_stream_summary,
    _report_online_outcome,
)
from experiments.joint.solve_control.online_td_experiment_common import _write_json
from experiments.joint.solve_control.run_online_methods_2k import _as_feedback
from experiments.diagnostics.solve_control.run_solve_controller_calibration import (
    V2_CALIBRATION_BASELINE,
    _build_bundle,
    _controller_is_finite,
    _write_json_line,
)
from setup.space import DEFAULT_SETUP_PARAMS
from hypre.bindings import augment_setup_params
from experiments.joint.solve_control.native_evaluation import solve_no_rl_case
from solve.controllers.common import ControllerBundle, OnlineSolveCase
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqV2LcbSpec,
    RecursiveLstdqV3LcbSpec,
)


V2_METHOD = V2_CALIBRATION_BASELINE
V3_METHOD = "lstdq_v3_selected"
TRAIN_CASES = 400
HELD_OUT_CASES = 200
REPETITIONS = 3


def _load_json_lines(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _resolve_setup_trace(
    calibration_result_path: Path,
    result: Mapping[str, Any],
) -> Path:
    configured = Path(str(result["setup_trace"]["path"]))
    if configured.exists():
        return configured
    beside_result = calibration_result_path.parent / configured.name
    if beside_result.exists():
        return beside_result
    raise FileNotFoundError(
        f"Cannot locate calibration fixed-setup trace: {configured}"
    )


def _selected_v3_beta(
    result: Mapping[str, Any],
) -> tuple[float, str]:
    selected = result["selected"]["lstdq_v3"]
    if not bool(selected.get("eligible", False)):
        raise ValueError(
            "Calibration did not produce an eligible LSTDQ v3 candidate"
        )
    method = str(selected["method"])
    summary = result["candidates"][method]
    gate = summary.get("v3_calibration_gate", {})
    if not bool(gate.get("passed", False)):
        raise ValueError(
            "Selected LSTDQ v3 candidate did not pass the calibration gate"
        )
    return float(selected["beta"]), method


def _build_controllers(
    args: argparse.Namespace,
    *,
    v3_beta: float,
) -> Dict[str, ControllerBundle]:
    return {
        V2_METHOD: _build_bundle(
            args=args,
            kind="recursive_lstdq_v2",
            algorithm=RecursiveLstdqV2LcbSpec(
                ridge=1.0,
                uncertainty_beta=4.0,
                residual_floor_sec=1.0e-3,
                lcb_lower_bound_sec=0.0,
                coverage_ridge=1.0,
                residual_scale_window=2048,
                residual_scale_min_samples=32,
            ),
            trace_lambda=0.8,
        ),
        V3_METHOD: _build_bundle(
            args=args,
            kind="recursive_lstdq_v3",
            algorithm=RecursiveLstdqV3LcbSpec(
                ridge=1.0,
                uncertainty_beta=float(v3_beta),
                residual_floor_sec=1.0e-3,
                lcb_lower_bound_sec=0.0,
            ),
            trace_lambda=0.8,
        ),
    }


def _run_case(
    *,
    bundle: ControllerBundle,
    setup_row: Mapping[str, Any],
    args: argparse.Namespace,
    learn: bool,
    explore: bool,
) -> Dict[str, Any]:
    mkw = dict(setup_row["mkw"])
    fallback_result: Dict[str, Any] = {}

    def fallback_attempt():
        result = solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS), mkw=mkw,
            solver_tol=float(args.tol), solver_max_iter=int(args.max_cycles),
            augment_params=augment_setup_params,
        )
        fallback_result.update(result)
        return result

    native = bundle.run_case(
        OnlineSolveCase(
            mkw=mkw,
            params=dict(setup_row["params"]),
            solve_tol=float(args.tol),
            solve_max_cycles=int(args.max_cycles),
            learn=bool(learn),
            explore=bool(explore),
            record_action_metadata=True,
            fallback_attempt=fallback_attempt,
        )
    )
    outcome = _report_online_outcome(
        _as_feedback(native, include_controller=True),
        bandit_timing={},
    )
    outcome["completed_residual_norm"] = (
        fallback_result.get("residual_norm", float("nan"))
        if outcome.get("fallback_used", False) else outcome["residual_norm"]
    )
    return outcome


def _summarize_stability(
    repetitions: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    method_metrics: Dict[str, Dict[str, Any]] = {}
    for method in (V2_METHOD, V3_METHOD):
        summaries = [
            repetition["held_out"][method]
            for repetition in repetitions
        ]
        means = np.asarray(
            [
                summary["means_sec"]["end_to_end_runtime"]
                for summary in summaries
            ],
            dtype=float,
        )
        spread = (
            float((np.max(means) - np.min(means)) / np.min(means))
            if np.all(np.isfinite(means)) and np.min(means) > 0.0
            else float("inf")
        )
        method_metrics[method] = {
            "held_out_mean_end_to_end_sec": means.tolist(),
            "relative_max_min_spread": spread,
            "primary_failure_counts": [
                int(summary["primary_failure_count"])
                for summary in summaries
            ],
            "unrecovered_failure_counts": [
                int(summary["unrecovered_failure_count"])
                for summary in summaries
            ],
            "finite_controller": [
                bool(repetition["finite_controller"][method])
                for repetition in repetitions
            ],
        }
    failure_excess = [
        int(v3 - v2)
        for v3, v2 in zip(
            method_metrics[V3_METHOD]["primary_failure_counts"],
            method_metrics[V2_METHOD]["primary_failure_counts"],
        )
    ]
    v3 = method_metrics[V3_METHOD]
    passed = bool(
        float(v3["relative_max_min_spread"]) <= 0.05
        and all(v3["finite_controller"])
        and max(v3["unrecovered_failure_counts"], default=0) == 0
        and max(failure_excess, default=0) <= 5
    )
    return {
        "methods": method_metrics,
        "v3_primary_failure_excess_vs_v2": failure_excess,
        "acceptance": {
            "maximum_relative_runtime_spread": 0.05,
            "maximum_primary_failure_excess_per_repetition": 5,
            "requires_finite_controller": True,
            "requires_zero_unrecovered_failures": True,
            "passed": passed,
        },
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite stability output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = args.calibration_result
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    v3_beta, selected_method = _selected_v3_beta(calibration)
    setup_path = _resolve_setup_trace(calibration_path, calibration)
    setup_rows = _load_json_lines(setup_path)
    if len(setup_rows) < TRAIN_CASES + HELD_OUT_CASES:
        raise ValueError(
            "Stability audit requires at least 600 fixed-setup cases"
        )
    training_rows = setup_rows[:TRAIN_CASES]
    held_out_rows = setup_rows[
        TRAIN_CASES : TRAIN_CASES + HELD_OUT_CASES
    ]

    repetition_results: list[Dict[str, Any]] = []
    for repetition in range(REPETITIONS):
        repetition_dir = args.output_dir / f"repetition_{repetition}"
        repetition_dir.mkdir()
        bundles = _build_controllers(args, v3_beta=v3_beta)
        names = tuple(bundles)
        training_records: Dict[str, list[Dict[str, Any]]] = {
            name: [] for name in names
        }
        held_out_records: Dict[str, list[Dict[str, Any]]] = {
            name: [] for name in names
        }
        handles = {
            (phase, name): (
                repetition_dir / f"{phase}_{name}.jsonl"
            ).open("w", encoding="utf-8")
            for phase in ("train", "held_out")
            for name in names
        }
        order_rng = np.random.default_rng(
            int(args.method_order_seed + repetition * 1009)
        )
        try:
            for phase, rows, learn, explore, records in (
                (
                    "train",
                    training_rows,
                    True,
                    True,
                    training_records,
                ),
                (
                    "held_out",
                    held_out_rows,
                    False,
                    False,
                    held_out_records,
                ),
            ):
                for index, setup_row in enumerate(rows):
                    case_rows: Dict[str, Dict[str, Any]] = {}
                    for rank, method_index in enumerate(
                        order_rng.permutation(len(names))
                    ):
                        name = names[int(method_index)]
                        row = {
                            "index": int(index),
                            "execution_rank": int(rank),
                            "mkw": dict(setup_row["mkw"]),
                            "params": dict(setup_row["params"]),
                            "outcome": _run_case(
                                bundle=bundles[name],
                                setup_row=setup_row,
                                args=args,
                                learn=learn,
                                explore=explore,
                            ),
                        }
                        case_rows[name] = row
                    for name in names:
                        records[name].append(case_rows[name])
                        _write_json_line(
                            handles[(phase, name)],
                            case_rows[name],
                        )
        finally:
            for handle in handles.values():
                handle.close()

        finite = {
            name: _controller_is_finite(bundle.controller)
            for name, bundle in bundles.items()
        }
        for name, bundle in bundles.items():
            bundle.save(repetition_dir / f"{name}_trained.npz")
        repetition_result = {
            "repetition": int(repetition),
            "method_order_seed": int(
                args.method_order_seed + repetition * 1009
            ),
            "train": {
                name: _method_stream_summary(rows)
                for name, rows in training_records.items()
            },
            "held_out": {
                name: _method_stream_summary(rows)
                for name, rows in held_out_records.items()
            },
            "finite_controller": finite,
        }
        repetition_results.append(repetition_result)
        _write_json(
            repetition_dir / "summary.json",
            repetition_result,
        )
        print(
            json.dumps(
                {
                    "stage": "lstdq_v3_stability",
                    "completed_repetition": repetition + 1,
                    "total_repetitions": REPETITIONS,
                }
            ),
            flush=True,
        )

    result = {
        "protocol": {
            "purpose": "execution-order stability audit",
            "training_cases": TRAIN_CASES,
            "held_out_cases": HELD_OUT_CASES,
            "repetitions": REPETITIONS,
            "v2_beta": 4.0,
            "v3_beta": v3_beta,
            "selected_v3_calibration_method": selected_method,
            "fixed_setup_trace": str(setup_path),
        },
        "repetitions": repetition_results,
        "stability": _summarize_stability(repetition_results),
    }
    _write_json(args.output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-seed", type=int, default=41966039)
    parser.add_argument("--method-order-seed", type=int, default=41972039)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
