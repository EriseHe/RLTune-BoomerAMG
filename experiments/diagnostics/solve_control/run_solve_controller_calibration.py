"""Calibrate solve-controller confidence on a disjoint fixed-setup trace."""

from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from experiments.joint.solve_control.joint_online_common import (
    _build_paired_instance_stream,
    _configure_paired_environment,
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
    _validate_recovery_stream,
)
from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json
from experiments.joint.solve_control.joint_controller_build import (
    make_setup_obs_encoder,
)
from experiments.joint.solve_control.joint_reporting import _action_summary
from experiments.joint.solve_control.run_online_methods_2k import _as_feedback
from setup.space import DEFAULT_SETUP_PARAMS
from experiments.joint.solve_control.setup_aware_compare_common import (
    augment_setup_params,
    build_online_linucb_branch,
    run_bandit_step_test_final,
    solve_fixed_w_case,
    solve_no_rl_case,
    validate_expected_setup_action_count,
)
from solve.controllers.common import (
    ControllerBundle,
    EpsilonScheduleSpec,
    OnlineSolveCase,
    SharedActionSpec,
    SolveStateSpec,
)
from solve.controllers.lsvi import HierarchicalLsviLcbSpec
from solve.controllers.model_based import StructuredModelBasedSpec
from solve.controllers.recursive_lstdq import (
    RecursiveLstdqV2LcbSpec,
    RecursiveLstdqV3LcbSpec,
)
from solve.registry import (
    OnlineControllerBuildSpec,
    build_online_solve_controller,
)


V2_BETAS = (1.0, 2.0, 4.0)
V3_BETAS = (1.0, 2.0, 4.0)
LSVI_BETAS = (1.0, 2.0, 4.0)
MODEL_METHOD = "structured_model_based"
V2_CALIBRATION_BASELINE = "lstdq_v2_beta_4"


def _write_json_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(_json_ready(row), separators=(",", ":")))
    handle.write("\n")


def _shared_controller_specs(
    args: argparse.Namespace,
) -> tuple[SolveStateSpec, SharedActionSpec]:
    weights = tuple(float(value) for value in np.linspace(1.0, 3.0, 41))
    centers = tuple(float(value) for value in np.linspace(1.0, 3.0, 9))
    return (
        SolveStateSpec(
            tol=float(args.tol),
            max_cycles=int(args.max_cycles),
            c_max=float(args.c_max),
            mode="setup_full",
        ),
        SharedActionSpec(
            weights=weights,
            rbf_centers=centers,
            rbf_sigma=0.2,
            epsilon=EpsilonScheduleSpec(
                start=0.30,
                final=0.03,
                decay_steps=20_000.0,
            ),
            anchor_weight=weights[0],
            force_default_first_action=True,
        ),
    )


def _build_bundle(
    *,
    args: argparse.Namespace,
    kind: str,
    algorithm: Any,
    trace_lambda: float | None,
) -> ControllerBundle:
    state, actions = _shared_controller_specs(args)
    return build_online_solve_controller(
        OnlineControllerBuildSpec(
            kind=kind,
            state=state,
            actions=actions,
            algorithm=algorithm,
            trace_lambda=trace_lambda,
        ),
        setup_obs_encoder=make_setup_obs_encoder(),
        seed=int(args.controller_seed),
    )


def _build_controllers(
    args: argparse.Namespace,
) -> tuple[Dict[str, ControllerBundle], Dict[str, float | None]]:
    bundles: Dict[str, ControllerBundle] = {}
    betas: Dict[str, float | None] = {}
    for beta in V2_BETAS:
        name = f"lstdq_v2_beta_{beta:g}"
        bundles[name] = _build_bundle(
            args=args,
            kind="recursive_lstdq_v2",
            algorithm=RecursiveLstdqV2LcbSpec(
                ridge=1.0,
                uncertainty_beta=float(beta),
                residual_floor_sec=1.0e-3,
                q_max_sec=0.1,
                inverse_denominator_floor=1.0e-10,
                lcb_lower_bound_sec=0.0,
                coverage_ridge=1.0,
                residual_scale_window=2048,
                residual_scale_min_samples=32,
            ),
            trace_lambda=0.8,
        )
        betas[name] = float(beta)

    for beta in V3_BETAS:
        name = f"lstdq_v3_beta_{beta:g}"
        bundles[name] = _build_bundle(
            args=args,
            kind="recursive_lstdq_v3",
            algorithm=RecursiveLstdqV3LcbSpec(
                ridge=1.0,
                uncertainty_beta=float(beta),
                residual_floor_sec=1.0e-3,
                q_max_sec=0.1,
                inverse_denominator_floor=1.0e-10,
                lcb_lower_bound_sec=0.0,
            ),
            trace_lambda=0.8,
        )
        betas[name] = float(beta)

    bundles[MODEL_METHOD] = _build_bundle(
        args=args,
        kind="structured_model_based",
        algorithm=StructuredModelBasedSpec(
            ridge=1.0,
            minimum_samples=32,
            scale_window=2048,
        ),
        trace_lambda=None,
    )
    betas[MODEL_METHOD] = None

    for beta in LSVI_BETAS:
        name = f"recalibrated_lsvi_beta_{beta:g}"
        bundles[name] = _build_bundle(
            args=args,
            kind="recalibrated_lsvi",
            algorithm=HierarchicalLsviLcbSpec(
                horizon=int(args.max_cycles),
                ridge=1.0,
                uncertainty_beta=float(beta),
                residual_floor_sec=1.0e-3,
                refit_interval_episodes=100,
                refit_sweeps=3,
                residual_shrinkage_samples=32.0,
            ),
            trace_lambda=None,
        )
        betas[name] = float(beta)
    return bundles, betas


def _generate_setup_trace(
    args: argparse.Namespace,
    stream: Sequence[tuple[Dict[str, Any], np.ndarray]],
) -> list[Dict[str, Any]]:
    branch, _config = build_online_linucb_branch(
        seed=int(args.bandit_seed),
        tune_dim=7,
        tune7_variant="categorical",
    )
    validate_expected_setup_action_count(branch)
    rows: list[Dict[str, Any]] = []
    previous_update = 0.0
    path = args.output_dir / "fixed_setup_trace.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for index, (mkw, context) in enumerate(stream):
            def solve_selected(params: Dict[str, Any]) -> Dict[str, Any]:
                native = solve_fixed_w_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    w=1.6,
                    sweeps_down=1,
                    sweeps_up=1,
                    solve_tol=float(args.tol),
                    solve_max_cycles=int(args.max_cycles),
                )
                return _as_feedback(native, include_controller=False)

            def solve_fallback(_params: Dict[str, Any]) -> Dict[str, Any]:
                native = solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=float(args.tol),
                    solver_max_iter=int(args.max_cycles),
                    augment_params=augment_setup_params,
                )
                return _as_feedback(native, include_controller=False)

            params, outcome, timing, fallback_used, update_sec = (
                run_bandit_step_test_final(
                    policy=branch.policy,
                    parameter_space=branch.parameter_space,
                    problem_context=np.asarray(context, dtype=float),
                    solver_fn=solve_selected,
                    fallback_solver_fn=solve_fallback,
                    prev_update_est=float(previous_update),
                )
            )
            previous_update = float(update_sec)
            row = {
                "index": int(index),
                "mkw": dict(mkw),
                "context": np.asarray(context, dtype=float).tolist(),
                "params": dict(params),
                "arm_index": _policy_last_arm(branch.policy),
                "fallback_used": int(fallback_used),
                "bandit_timing": dict(timing),
                "outcome": _report_online_outcome(
                    outcome, bandit_timing=dict(timing)
                ),
            }
            rows.append(row)
            _write_json_line(handle, row)
            if (index + 1) % int(args.progress_every) == 0:
                handle.flush()
                print(
                    json.dumps(
                        {
                            "stage": "calibration_setup_trace",
                            "done": int(index + 1),
                            "total": int(len(stream)),
                        }
                    ),
                    flush=True,
                )
    branch.policy.model.save_mutable_state(
        args.output_dir / "fixed_setup_trace_bandit_final.npz",
        metadata={"cases": int(len(rows))},
    )
    return rows


def _primary_failed(row: Dict[str, Any]) -> bool:
    return str(row["outcome"].get("primary_status", "success")) != "success"


def _longest_excess_failure_chain(
    rows: Sequence[Dict[str, Any]],
    baseline_rows: Sequence[Dict[str, Any]],
) -> int:
    longest = 0
    current = 0
    for row, baseline in zip(rows, baseline_rows):
        controller_specific_failure = _primary_failed(row) and not _primary_failed(
            baseline
        )
        current = current + 1 if controller_specific_failure else 0
        longest = max(longest, current)
    return int(longest)


def _controller_is_finite(controller: Any) -> bool:
    return all(
        bool(np.all(np.isfinite(value)))
        for value in vars(controller).values()
        if isinstance(value, np.ndarray)
    )


def _choose_candidate(
    names: Iterable[str],
    *,
    summaries: Dict[str, Dict[str, Any]],
    betas: Dict[str, float | None],
) -> str:
    eligible = [
        name
        for name in names
        if bool(summaries[name]["eligible"])
    ]
    if not eligible:
        raise RuntimeError(f"No eligible calibration candidates among {tuple(names)}")
    best_total = min(
        float(summaries[name]["means_sec"]["end_to_end_runtime"])
        for name in eligible
    )
    tied = [
        name
        for name in eligible
        if float(summaries[name]["means_sec"]["end_to_end_runtime"])
        <= best_total * 1.005
    ]
    return min(
        tied,
        key=lambda name: (
            float(summaries[name]["means_sec"]["controller_runtime"]),
            float("inf") if betas[name] is None else float(betas[name]),
            name,
        ),
    )


def _apply_v3_calibration_gates(
    summaries: Dict[str, Dict[str, Any]],
) -> None:
    """Apply the locked v3 uncertainty, runtime, and overhead gates in place."""

    baseline = summaries[V2_CALIBRATION_BASELINE]
    if not bool(baseline["eligible"]):
        raise RuntimeError(
            f"{V2_CALIBRATION_BASELINE} must pass the base safety gates"
        )
    baseline_diagnostic = baseline["action_diagnostics"][
        "residual_to_parameter_width"
    ]
    baseline_median_ratio = float(
        baseline_diagnostic[
            "median_abs_residual_over_parameter_width"
        ]
    )
    baseline_end_to_end = float(
        baseline["means_sec"]["end_to_end_runtime"]
    )
    baseline_overhead = float(
        baseline["means_sec"]["controller_runtime"]
    )
    overhead_limit = min(
        baseline_overhead * 1.25,
        baseline_overhead + 0.005,
    )

    for name, summary in summaries.items():
        if not name.startswith("lstdq_v3_beta_"):
            continue
        diagnostic = summary["action_diagnostics"][
            "residual_to_parameter_width"
        ]
        reasons: list[str] = []
        if not bool(summary["eligible"]):
            reasons.append("base_safety_gate")
        fraction_within_4x = float(
            diagnostic["fraction_within_4x_parameter_width"]
        )
        if fraction_within_4x < 0.70:
            reasons.append("fraction_within_4x_below_0.70")
        median_ratio = float(
            diagnostic["median_abs_residual_over_parameter_width"]
        )
        if (
            baseline_median_ratio > 0.0
            and median_ratio > baseline_median_ratio / 3.0
        ):
            reasons.append("median_ratio_not_3x_better_than_v2")
        if (
            float(summary["means_sec"]["end_to_end_runtime"])
            > baseline_end_to_end * 1.02
        ):
            reasons.append("end_to_end_more_than_2pct_slower_than_v2")
        if (
            float(summary["means_sec"]["controller_runtime"])
            > overhead_limit
        ):
            reasons.append("controller_overhead_exceeds_v2_gate")
        summary["v3_calibration_gate"] = {
            "passed": not reasons,
            "reasons": reasons,
            "baseline": V2_CALIBRATION_BASELINE,
            "minimum_fraction_within_4x": 0.70,
            "maximum_median_ratio": (
                baseline_median_ratio / 3.0
                if baseline_median_ratio > 0.0
                else None
            ),
            "maximum_end_to_end_runtime_sec": (
                baseline_end_to_end * 1.02
            ),
            "maximum_controller_runtime_sec": overhead_limit,
        }
        summary["eligible"] = bool(not reasons)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite calibration output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "trajectories").mkdir()
    (args.output_dir / "checkpoints").mkdir()
    if not bool(args.smoke) and int(args.cases) != 600:
        raise ValueError("The locked calibration contains exactly 600 cases")
    _configure_paired_environment(args)
    stream, manifest = _build_paired_instance_stream(args)
    _write_json(args.output_dir / "stream_manifest.json", manifest)
    setup_rows = _generate_setup_trace(args, stream)

    controller_bundles, betas = _build_controllers(args)
    records: Dict[str, list[Dict[str, Any]]] = {
        name: [] for name in controller_bundles
    }
    handles = {
        name: (args.output_dir / "trajectories" / f"{name}.jsonl").open(
            "w", encoding="utf-8"
        )
        for name in controller_bundles
    }
    order_rng = np.random.default_rng(int(args.method_order_seed))
    names = tuple(controller_bundles)
    try:
        for index, ((mkw, context), setup_row) in enumerate(zip(stream, setup_rows)):
            case_rows: Dict[str, Dict[str, Any]] = {}
            for rank, method_index in enumerate(order_rng.permutation(len(names))):
                name = names[int(method_index)]
                native = controller_bundles[name].run_case(
                    OnlineSolveCase(
                        mkw=dict(mkw),
                        params=dict(setup_row["params"]),
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                        learn=True,
                        explore=True,
                        problem_context=np.asarray(
                            context,
                            dtype=float,
                        ),
                        record_action_metadata=True,
                        fallback_attempt=lambda: solve_no_rl_case(
                            params=dict(DEFAULT_SETUP_PARAMS),
                            mkw=dict(mkw),
                            solver_tol=float(args.tol),
                            solver_max_iter=int(args.max_cycles),
                            augment_params=augment_setup_params,
                        ),
                    ),
                )
                row = {
                    "index": int(index),
                    "execution_rank": int(rank),
                    "mkw": dict(mkw),
                    "context": setup_row["context"],
                    "params": dict(setup_row["params"]),
                    "arm_index": int(setup_row["arm_index"]),
                    "bandit_timing": dict(setup_row["bandit_timing"]),
                    "outcome": _report_online_outcome(
                        _as_feedback(native, include_controller=True),
                        bandit_timing=dict(setup_row["bandit_timing"]),
                    ),
                }
                case_rows[name] = row
            for name in names:
                records[name].append(case_rows[name])
                _write_json_line(handles[name], case_rows[name])
            if (index + 1) % int(args.progress_every) == 0:
                for handle in handles.values():
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "stage": "calibration_replay",
                            "done": int(index + 1),
                            "total": int(len(stream)),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    summaries: Dict[str, Dict[str, Any]] = {}
    for name, rows in records.items():
        summary = _method_stream_summary(rows)
        longest_chain = _longest_excess_failure_chain(rows, setup_rows)
        bundle = controller_bundles[name]
        finite = _controller_is_finite(bundle.controller)
        summary.update(
            {
                "finite_parameters": bool(finite),
                "longest_excess_primary_failure_chain": int(longest_chain),
                "eligible": bool(
                    finite
                    and int(summary["unrecovered_failure_count"]) == 0
                    and longest_chain < 25
                ),
                "controller": bundle.summary(),
                "recovery_audit": _validate_recovery_stream(
                    rows, expect_bandit_transaction=False
                ),
                "action_diagnostics": _action_summary(rows),
            }
        )
        summaries[name] = summary
        bundle.save(
            args.output_dir / "checkpoints" / f"{name}_final.npz"
        )

    _apply_v3_calibration_gates(summaries)
    selected_v2 = _choose_candidate(
        (name for name in names if name.startswith("lstdq_v2_beta_")),
        summaries=summaries,
        betas=betas,
    )
    eligible_v3 = tuple(
        name
        for name in names
        if name.startswith("lstdq_v3_beta_")
        and bool(summaries[name]["eligible"])
    )
    selected_v3 = (
        None
        if not eligible_v3
        else _choose_candidate(
            eligible_v3,
            summaries=summaries,
            betas=betas,
        )
    )
    selected_lsvi = _choose_candidate(
        (name for name in names if name.startswith("recalibrated_lsvi_beta_")),
        summaries=summaries,
        betas=betas,
    )
    selected = {
        "lstdq_v2": {
            "method": selected_v2,
            "beta": float(betas[selected_v2]),
        },
        "lstdq_v3": {
            "method": selected_v3,
            "beta": (
                None
                if selected_v3 is None
                else float(betas[selected_v3])
            ),
            "eligible": bool(selected_v3 is not None),
        },
        "structured_model_based": {"method": MODEL_METHOD},
        "recalibrated_lsvi": {
            "method": selected_lsvi,
            "beta": float(betas[selected_lsvi]),
        },
    }
    result = {
        "protocol": {
            "purpose": "disjoint fixed-setup solve-controller calibration",
            "cases": int(len(stream)),
            "grid_n": int(args.grid_n),
            "setup_param_resolution": int(args.setup_param_resolution),
            "action_grid": "1.00:0.05:3.00",
            "betas": {
                "lstdq_v2": list(V2_BETAS),
                "lstdq_v3": list(V3_BETAS),
                "lsvi": list(LSVI_BETAS),
            },
            "seeds": {
                "stream": str(args.train_seed_groups),
                "shuffle": int(args.train_shuffle_seeds),
                "bandit": int(args.bandit_seed),
                "controller": int(args.controller_seed),
                "method_order": int(args.method_order_seed),
            },
        },
        "stream": manifest,
        "setup_trace": {
            "summary": _method_stream_summary(setup_rows),
            "path": str(args.output_dir / "fixed_setup_trace.jsonl"),
        },
        "candidates": summaries,
        "selected": selected,
    }
    _write_json(args.output_dir / "result.json", result)
    _write_json(args.output_dir / "selected_config.json", selected)
    print(
        json.dumps(
            {
                "stage": "calibration_done",
                "output": str(args.output_dir / "result.json"),
                "selected": selected,
            }
        ),
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=600)
    parser.add_argument("--seed", type=int, default=41900039)
    parser.add_argument("--bandit-seed", type=int, default=41960039)
    parser.add_argument("--controller-seed", type=int, default=41966039)
    parser.add_argument("--method-order-seed", type=int, default=41972039)
    parser.add_argument(
        "--train-seed-groups", default="41900039,41906039,41912039"
    )
    parser.add_argument("--train-shuffle-seeds", default="41948039")
    parser.add_argument("--train-cases-per-seed", type=int, default=200)
    parser.add_argument("--instance-offset", type=int, default=0)
    parser.add_argument("--grid-n", type=int, default=60)
    parser.add_argument("--setup-param-resolution", type=int, default=20)
    parser.add_argument("--setup-action-space", default="full_cartesian")
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parsed = parser.parse_args()
    parsed.train_cases = int(parsed.cases)
    parsed.train_group_take = int(parsed.cases)
    run(parsed)


if __name__ == "__main__":
    main()
