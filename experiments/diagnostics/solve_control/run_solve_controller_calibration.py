"""Calibrate solve-controller confidence on a disjoint fixed-setup trace."""

from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from joint_online_common import (
    _build_paired_instance_stream,
    _configure_paired_environment,
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
    _validate_recovery_stream,
)
from online_td_experiment_common import _json_ready, _write_json
from run_joint_online_sarsa_4k import (
    _make_recalibrated_lsvi_controller,
    _make_recursive_lstdq_v2_controller,
    _make_structured_model_based_controller,
)
from run_online_methods_2k import _as_feedback
from setup_action_space import DEFAULT_SETUP_PARAMS
from setup_aware_compare_common import (
    augment_setup_params,
    build_online_linucb_branch,
    run_bandit_step_test_final,
    solve_fixed_w_case,
    solve_no_rl_case,
    validate_expected_setup_action_count,
)
from SolvePhase.algorithms.sarsa import run_td_episode


V2_BETAS = (1.0, 2.0, 4.0)
LSVI_BETAS = (1.0, 2.0, 4.0)
MODEL_METHOD = "structured_model_based"


def _write_json_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(_json_ready(row), separators=(",", ":")))
    handle.write("\n")


def _controller_args(args: argparse.Namespace, *, v2_beta: float, lsvi_beta: float) -> SimpleNamespace:
    return SimpleNamespace(
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        weights="",
        action_rbf_centers="",
        action_rbf_sigma=0.2,
        epsilon_start=0.30,
        epsilon_final=0.03,
        epsilon_decay_steps=20_000.0,
        recursive_lstdq_ridge=1.0,
        recursive_lstdq_beta=2.0,
        recursive_lstdq_lambda=0.8,
        recursive_lstdq_residual_floor_sec=1.0e-3,
        recursive_lstdq_lcb_lower_bound_sec=0.0,
        recursive_lstdq_v2_beta=float(v2_beta),
        recursive_lstdq_v2_coverage_ridge=1.0,
        recursive_lstdq_v2_residual_window=2048,
        recursive_lstdq_v2_min_samples=32,
        structured_model_ridge=1.0,
        structured_model_min_samples=32,
        structured_model_scale_window=2048,
        lsvi_ridge=1.0,
        lsvi_residual_floor_sec=1.0e-3,
        lsvi_refit_interval_episodes=100,
        recalibrated_lsvi_beta=float(lsvi_beta),
        recalibrated_lsvi_refit_sweeps=3,
        recalibrated_lsvi_shrinkage_samples=32.0,
    )


def _set_action_grid(controller_args: SimpleNamespace) -> None:
    weights = tuple(float(value) for value in np.linspace(1.0, 3.0, 41))
    centers = tuple(float(value) for value in np.linspace(1.0, 3.0, 9))
    controller_args.weights = ",".join(f"{value:g}" for value in weights)
    controller_args.action_rbf_centers = ",".join(
        f"{value:g}" for value in centers
    )


def _build_controllers(args: argparse.Namespace) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, float | None]]:
    controllers: Dict[str, Any] = {}
    encoders: Dict[str, Any] = {}
    betas: Dict[str, float | None] = {}
    for beta in V2_BETAS:
        name = f"lstdq_v2_beta_{beta:g}"
        controller_args = _controller_args(args, v2_beta=beta, lsvi_beta=2.0)
        _set_action_grid(controller_args)
        controller, encoder = _make_recursive_lstdq_v2_controller(
            controller_args,
            seed=int(args.controller_seed),
        )
        controllers[name] = controller
        encoders[name] = encoder
        betas[name] = float(beta)

    controller_args = _controller_args(args, v2_beta=2.0, lsvi_beta=2.0)
    _set_action_grid(controller_args)
    controller, encoder = _make_structured_model_based_controller(
        controller_args,
        seed=int(args.controller_seed),
    )
    controllers[MODEL_METHOD] = controller
    encoders[MODEL_METHOD] = encoder
    betas[MODEL_METHOD] = None

    for beta in LSVI_BETAS:
        name = f"recalibrated_lsvi_beta_{beta:g}"
        controller_args = _controller_args(args, v2_beta=2.0, lsvi_beta=beta)
        _set_action_grid(controller_args)
        controller, encoder = _make_recalibrated_lsvi_controller(
            controller_args,
            seed=int(args.controller_seed),
        )
        controllers[name] = controller
        encoders[name] = encoder
        betas[name] = float(beta)
    return controllers, encoders, betas


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
                    context=np.asarray(context, dtype=float),
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

    controllers, encoders, betas = _build_controllers(args)
    records: Dict[str, list[Dict[str, Any]]] = {
        name: [] for name in controllers
    }
    handles = {
        name: (args.output_dir / "trajectories" / f"{name}.jsonl").open(
            "w", encoding="utf-8"
        )
        for name in controllers
    }
    order_rng = np.random.default_rng(int(args.method_order_seed))
    names = tuple(controllers)
    try:
        for index, ((mkw, _context), setup_row) in enumerate(zip(stream, setup_rows)):
            case_rows: Dict[str, Dict[str, Any]] = {}
            for rank, method_index in enumerate(order_rng.permutation(len(names))):
                name = names[int(method_index)]
                native = run_td_episode(
                    mkw=dict(mkw),
                    params=dict(setup_row["params"]),
                    controller=controllers[name],
                    encoder=encoders[name],
                    solve_tol=float(args.tol),
                    solve_max_cycles=int(args.max_cycles),
                    learn=True,
                    explore=True,
                    record_action_metadata=True,
                    fallback_attempt=lambda: solve_no_rl_case(
                        params=dict(DEFAULT_SETUP_PARAMS),
                        mkw=dict(mkw),
                        solver_tol=float(args.tol),
                        solver_max_iter=int(args.max_cycles),
                        augment_params=augment_setup_params,
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
        finite = _controller_is_finite(controllers[name])
        summary.update(
            {
                "finite_parameters": bool(finite),
                "longest_excess_primary_failure_chain": int(longest_chain),
                "eligible": bool(
                    finite
                    and int(summary["unrecovered_failure_count"]) == 0
                    and longest_chain < 25
                ),
                "controller": controllers[name].summary(),
                "recovery_audit": _validate_recovery_stream(
                    rows, expect_bandit_transaction=False
                ),
            }
        )
        summaries[name] = summary
        controllers[name].save(
            args.output_dir / "checkpoints" / f"{name}_final.npz"
        )

    selected_v2 = _choose_candidate(
        (name for name in names if name.startswith("lstdq_v2_beta_")),
        summaries=summaries,
        betas=betas,
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
            "betas": {"lstdq_v2": list(V2_BETAS), "lsvi": list(LSVI_BETAS)},
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
