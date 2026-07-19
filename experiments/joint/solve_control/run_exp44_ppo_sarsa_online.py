from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

from online_td_experiment_common import _action_diagnostics, _git_revision, _json_ready, _write_json
from online_td_lambda import ExpectedSarsaLambda, SolveStateEncoder, run_td_episode
from run_online_bandit_td_lambda import _method_stream_summary, _report_online_outcome
from run_online_methods_2k import (
    _continuous_action_diagnostics,
    _method_comparison,
    make_exp44_ppo_runner,
    make_true_online_sarsa_controller,
)
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    EXP44_MATRIX_GRID_N,
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
    SetupAwareSolvePolicyRunner,
    augment_setup_params,
    classify_rl_failure,
    fixed_trace,
    solve_fixed_w_case,
    solve_no_rl_case,
    solve_setup_aware_rl_case,
)


METHODS = (
    "default_setup",
    "bandit_default",
    "bandit_fixed_w1.6",
    "bandit_ppo",
    "bandit_sarsa",
)
BANDIT_METHODS = METHODS[1:]
COMPARISON_PAIRS = (
    ("sarsa_vs_ppo", "bandit_ppo", "bandit_sarsa"),
    ("sarsa_vs_fixed_w1.6", "bandit_fixed_w1.6", "bandit_sarsa"),
    ("sarsa_vs_bandit_default", "bandit_default", "bandit_sarsa"),
    ("ppo_vs_fixed_w1.6", "bandit_fixed_w1.6", "bandit_ppo"),
)
METRIC_ACCESSORS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "setup_runtime": lambda row: float(row["outcome"]["setup_runtime"]),
    "native_solve_runtime": lambda row: float(row["outcome"]["solve_runtime"]),
    "native_total_runtime": lambda row: float(row["outcome"]["runtime"]),
    "controller_runtime": lambda row: float(row["outcome"].get("infer_runtime", 0.0)),
    "solve_plus_controller": lambda row: float(
        row["outcome"]["solve_runtime"] + row["outcome"].get("infer_runtime", 0.0)
    ),
    "end_to_end_runtime": lambda row: float(row["outcome"]["end_to_end_runtime"]),
}


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _configure_environment(args: argparse.Namespace) -> Dict[str, str]:
    values = {
        "MATRIX_GRID_N": str(int(args.grid_n)),
        "SETUP_PARAM_RESOLUTION": str(int(args.setup_param_resolution)),
        "EXPECTED_SETUP_ACTION_COUNT": str(EXP44_TUNE7_CATEGORICAL_ACTION_COUNT),
        "SETUP_ACTION_SPACE": "full_cartesian",
        "C_MIN": str(float(args.c_min)),
        "C_MAX": str(float(args.c_max)),
        "DIFCONV_A": "0,0,0",
        "SOLVER_TOL": str(float(args.tol)),
        "SOLVER_MAX_ITER": str(int(args.max_cycles)),
        "SOLVE_TOL": str(float(args.tol)),
        "SOLVE_MAX_CYCLES": str(int(args.max_cycles)),
        "SETUP_BANDIT_METHOD": "linucbv4",
        "SETUP_TUNE_DIM": "7",
        "TUNE7_VARIANT": "categorical",
        "ALPHA": "1.0",
        "L2": "1.0",
        "RETRY_MAX_ATTEMPTS": "1000",
        "TUNE7_CANDIDATE_POOL_SIZE": "1024",
        "TUNE7_CANDIDATE_POOL_SIZE_BURNIN": "4096",
        "TUNE7_CANDIDATE_POOL_BURNIN_ROUNDS": "200",
        "TUNE7_ALPHA_DECAY_BURNIN_ROUNDS": "250",
        "TUNE7_CANDIDATE_STRATEGY": "adaptive_local",
        "TUNE7_LOCAL_NEIGHBOR_RADIUS": "1",
        "TUNE7_CANDIDATE_LOCAL_FRACTION": "0.60",
        "TUNE7_CANDIDATE_ELITE_FRACTION": "0.20",
        "SETUP_INITIAL_GUESS_ROUNDS": "1",
        "FAILURE_PENALTY_MULTIPLIER": "2.0",
        "FAILURE_SEVERITY_CAP": "6.0",
        "FAILURE_SCALE_WINDOW": "200",
        "STRUCTURAL_FAILURE_SURCHARGE_MULTIPLIER": "3.0",
    }
    os.environ.update(values)
    return values


def _trace_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    timing_fields = ("select_sec", "loss_eval_sec", "update_sec", "overhead_sec")
    feedback_fields = ("runtime", "setup_runtime", "solve_runtime")
    return {
        "cases": int(len(records)),
        "setup_retries": int(sum(int(row.get("failed_attempts", 0)) for row in records)),
        "feedback_failures": int(
            sum(bool(row.get("feedback_outcome", {}).get("failed", False)) for row in records)
        ),
        "bandit_timing_totals_sec": {
            field: float(
                sum(float(row.get("bandit_timing", {}).get(field, 0.0)) for row in records)
            )
            for field in timing_fields
        },
        "feedback_totals_sec": {
            field: float(
                sum(float(row.get("feedback_outcome", {}).get(field, 0.0)) for row in records)
            )
            for field in feedback_fields
        },
    }


def _sarsa_outcome(
    *,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    native = run_td_episode(
        mkw=dict(mkw),
        params=dict(params),
        controller=controller,
        encoder=encoder,
        solve_tol=float(args.tol),
        solve_max_cycles=int(args.max_cycles),
        learn=True,
        explore=True,
        record_action_metadata=True,
    )
    return _report_online_outcome(native, bandit_timing={})


def _evaluation_outcome(
    method: str,
    *,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    case_progress: float,
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    ppo_runner: SetupAwareSolvePolicyRunner,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if method == "default_setup":
        native = solve_no_rl_case(
            params=dict(DEFAULT_SETUP_PARAMS),
            mkw=dict(mkw),
            solver_tol=float(args.tol),
            solver_max_iter=int(args.max_cycles),
            augment_params=augment_setup_params,
        )
    elif method == "bandit_default":
        native = solve_no_rl_case(
            params=dict(params),
            mkw=dict(mkw),
            solver_tol=float(args.tol),
            solver_max_iter=int(args.max_cycles),
            augment_params=augment_setup_params,
        )
    elif method == "bandit_fixed_w1.6":
        native = solve_fixed_w_case(
            params=dict(params),
            mkw=dict(mkw),
            w=float(args.fixed_weight),
            sweeps_down=1,
            sweeps_up=1,
            solve_tol=float(args.tol),
            solve_max_cycles=int(args.max_cycles),
        )
    elif method == "bandit_ppo":
        native = solve_setup_aware_rl_case(
            params=dict(params),
            mkw=dict(mkw),
            solve_policy=ppo_runner,
            augment_params=augment_setup_params,
            classify_rl_failure=lambda *, residual_norm, iterations: classify_rl_failure(
                residual_norm=float(residual_norm),
                iterations=int(iterations),
                solve_tol=float(args.tol),
                solve_max_cycles=int(args.max_cycles),
            ),
            solve_max_cycles=int(args.max_cycles),
            case_progress=float(case_progress),
        )
    elif method == "bandit_sarsa":
        return _sarsa_outcome(
            mkw=mkw,
            params=params,
            controller=controller,
            encoder=encoder,
            args=args,
        )
    else:
        raise ValueError(f"Unknown evaluation method: {method}")
    return _report_online_outcome(native, bandit_timing={})


def _record(
    *,
    seed: int,
    instance_index: int,
    phase: str,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    outcome: Dict[str, Any],
    execution_order: Sequence[str],
) -> Dict[str, Any]:
    return {
        "forward_seed": int(seed),
        "instance_index": int(instance_index),
        "phase": str(phase),
        "mkw": dict(mkw),
        "params": dict(params),
        "execution_order": list(execution_order),
        "failed_attempts": 0,
        "outcome": dict(outcome),
    }


def _per_seed_comparisons(
    records: Dict[str, list[Dict[str, Any]]],
    *,
    seed: int,
) -> Dict[str, Any]:
    return {
        label: _method_comparison(
            records[candidate],
            records[reference],
            seed=int(seed + 1000 * offset),
        )
        for offset, (label, reference, candidate) in enumerate(COMPARISON_PAIRS)
    }


def _clustered_metric(
    seed_records: Sequence[Dict[str, list[Dict[str, Any]]]],
    *,
    reference: str,
    candidate: str,
    accessor: Callable[[Dict[str, Any]], float],
    seed: int,
) -> Dict[str, Any]:
    reference_arrays = [
        np.asarray([accessor(row) for row in records[reference]], dtype=float)
        for records in seed_records
    ]
    candidate_arrays = [
        np.asarray([accessor(row) for row in records[candidate]], dtype=float)
        for records in seed_records
    ]
    if not reference_arrays or any(
        ref.shape != cand.shape or ref.size == 0
        for ref, cand in zip(reference_arrays, candidate_arrays)
    ):
        raise ValueError("Clustered comparison requires aligned non-empty seed streams")

    reference_all = np.concatenate(reference_arrays)
    candidate_all = np.concatenate(candidate_arrays)
    reference_mean = float(np.mean(reference_all))
    candidate_mean = float(np.mean(candidate_all))
    improvement = (
        100.0 * (reference_mean - candidate_mean) / reference_mean
        if reference_mean != 0.0
        else float("nan")
    )

    rng = np.random.default_rng(int(seed))
    bootstrap = np.empty(2000, dtype=float)
    cluster_count = len(reference_arrays)
    for draw in range(bootstrap.size):
        sampled_reference: list[np.ndarray] = []
        sampled_candidate: list[np.ndarray] = []
        for cluster_index in rng.integers(0, cluster_count, size=cluster_count):
            ref = reference_arrays[int(cluster_index)]
            cand = candidate_arrays[int(cluster_index)]
            case_indices = rng.integers(0, ref.size, size=ref.size)
            sampled_reference.append(ref[case_indices])
            sampled_candidate.append(cand[case_indices])
        ref_mean = float(np.mean(np.concatenate(sampled_reference)))
        cand_mean = float(np.mean(np.concatenate(sampled_candidate)))
        bootstrap[draw] = (
            100.0 * (ref_mean - cand_mean) / ref_mean
            if ref_mean != 0.0
            else float("nan")
        )
    finite_bootstrap = bootstrap[np.isfinite(bootstrap)]
    return {
        "cases": int(reference_all.size),
        "forward_seed_clusters": int(cluster_count),
        "reference_mean_sec": reference_mean,
        "candidate_mean_sec": candidate_mean,
        "candidate_improvement_pct": float(improvement),
        "candidate_improvement_95pct": [
            float(np.percentile(finite_bootstrap, 2.5)) if finite_bootstrap.size else float("nan"),
            float(np.percentile(finite_bootstrap, 97.5)) if finite_bootstrap.size else float("nan"),
        ],
        "candidate_win_rate": float(np.mean(candidate_all < reference_all)),
    }


def _clustered_comparisons(
    seed_records: Sequence[Dict[str, list[Dict[str, Any]]]],
    *,
    seed: int,
) -> Dict[str, Any]:
    return {
        label: {
            metric: _clustered_metric(
                seed_records,
                reference=reference,
                candidate=candidate,
                accessor=accessor,
                seed=int(seed + 10_000 * pair_index + metric_index),
            )
            for metric_index, (metric, accessor) in enumerate(METRIC_ACCESSORS.items())
        }
        for pair_index, (label, reference, candidate) in enumerate(COMPARISON_PAIRS)
    }


def _write_main_table(path: Path, summaries: Dict[str, Dict[str, Any]]) -> None:
    fields = (
        "method",
        "cases",
        "mean_setup_runtime_sec",
        "mean_native_solve_runtime_sec",
        "mean_native_total_runtime_sec",
        "mean_controller_runtime_sec",
        "mean_end_to_end_runtime_sec",
        "failed_count",
        "mean_iterations",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, summary in summaries.items():
            means = summary["means_sec"]
            writer.writerow(
                {
                    "method": method,
                    "cases": summary["cases"],
                    "mean_setup_runtime_sec": means["setup_runtime"],
                    "mean_native_solve_runtime_sec": means["native_solve_runtime"],
                    "mean_native_total_runtime_sec": means["native_total_runtime"],
                    "mean_controller_runtime_sec": means["controller_runtime"],
                    "mean_end_to_end_runtime_sec": means["end_to_end_runtime"],
                    "failed_count": summary["failed_count"],
                    "mean_iterations": summary["mean_iterations"],
                }
            )


def _save_progress(
    path: Path,
    *,
    seed: int,
    stage: str,
    done: int,
    total: int,
    controller: ExpectedSarsaLambda,
) -> None:
    _write_json(
        path,
        {
            "forward_seed": int(seed),
            "stage": str(stage),
            "done": int(done),
            "total": int(total),
            "controller_steps": int(controller.steps),
            "controller_episodes": int(controller.episodes),
            "epsilon": float(controller.epsilon),
        },
    )


def _run_forward_seed(
    args: argparse.Namespace,
    *,
    forward_seed: int,
    ppo_runner: SetupAwareSolvePolicyRunner,
) -> Dict[str, Any]:
    seed_dir = args.output_dir / f"seed_{int(forward_seed)}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    result_path = seed_dir / "result.json"
    if bool(args.resume) and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if bool(existing.get("completed", False)):
            print(
                json.dumps({"stage": "exp44_seed_resume_skip", "seed": int(forward_seed)}),
                flush=True,
            )
            return existing

    trace_records: list[Dict[str, Any]] = []
    print(
        json.dumps(
            {"stage": "exp44_trace_start", "seed": int(forward_seed), "cases": int(args.trace_t)}
        ),
        flush=True,
    )
    trace = list(
        fixed_trace(
            T=int(args.trace_t),
            grid=(int(args.grid_n),) * 3,
            seed=int(forward_seed),
            tune_dim=7,
            bandit_method="linucbv4",
            solve_mode="no_rl",
            trace_records=trace_records,
        )
    )
    trace_hash = _stable_hash(
        [{"mkw": mkw, "params": params} for mkw, params in trace]
    )
    trace_audit = {
        "forward_seed": int(forward_seed),
        "trace_hash": trace_hash,
        "summary": _trace_summary(trace_records),
        "records": trace_records,
    }
    _write_json(seed_dir / "trace_audit.json", trace_audit)
    print(
        json.dumps(
            {
                "stage": "exp44_trace_done",
                "seed": int(forward_seed),
                "cases": len(trace),
                "trace_hash": trace_hash,
            }
        ),
        flush=True,
    )

    controller_args = copy.copy(args)
    controller_args.seed = int(args.controller_seed)
    controller, encoder = make_true_online_sarsa_controller(controller_args)
    controller.save(seed_dir / "controller_case_0.npz")

    adaptation_records: list[Dict[str, Any]] = []
    for instance_index, (mkw, params) in enumerate(trace[: int(args.eval_start)]):
        outcome = _sarsa_outcome(
            mkw=dict(mkw),
            params=dict(params),
            controller=controller,
            encoder=encoder,
            args=args,
        )
        adaptation_records.append(
            _record(
                seed=int(forward_seed),
                instance_index=int(instance_index),
                phase="online_adaptation",
                mkw=dict(mkw),
                params=dict(params),
                outcome=outcome,
                execution_order=("bandit_sarsa",),
            )
        )
        done = instance_index + 1
        if done % max(1, int(args.progress_every)) == 0 or done == int(args.eval_start):
            _save_progress(
                args.output_dir / "progress.json",
                seed=int(forward_seed),
                stage="online_adaptation",
                done=done,
                total=int(args.eval_start),
                controller=controller,
            )
            controller.save(seed_dir / "controller_partial.npz")
            print(
                json.dumps(
                    {
                        "stage": "exp44_sarsa_adaptation",
                        "seed": int(forward_seed),
                        "done": done,
                        "total": int(args.eval_start),
                        "steps": int(controller.steps),
                        "epsilon": float(controller.epsilon),
                    }
                ),
                flush=True,
            )
    controller.save(seed_dir / f"controller_case_{int(args.eval_start)}.npz")

    evaluation_records: Dict[str, list[Dict[str, Any]]] = {
        method: [] for method in METHODS
    }
    order_rng = np.random.default_rng(int(args.method_order_seed + int(forward_seed)))
    eval_trace = trace[int(args.eval_start) : int(args.eval_end)]
    denominator = max(1, len(eval_trace) - 1)
    midpoint = int(args.eval_start + len(eval_trace) // 2)
    for offset, (mkw, params) in enumerate(eval_trace):
        instance_index = int(args.eval_start + offset)
        method_order = list(METHODS)
        order_rng.shuffle(method_order)
        case_outcomes: Dict[str, Dict[str, Any]] = {}
        for method in method_order:
            case_outcomes[method] = _evaluation_outcome(
                method,
                mkw=dict(mkw),
                params=dict(params),
                case_progress=float(offset) / float(denominator),
                controller=controller,
                encoder=encoder,
                ppo_runner=ppo_runner,
                args=args,
            )
        for method in METHODS:
            selected_params = (
                dict(DEFAULT_SETUP_PARAMS) if method == "default_setup" else dict(params)
            )
            evaluation_records[method].append(
                _record(
                    seed=int(forward_seed),
                    instance_index=instance_index,
                    phase="evaluation_1000",
                    mkw=dict(mkw),
                    params=selected_params,
                    outcome=case_outcomes[method],
                    execution_order=method_order,
                )
            )
        completed = offset + 1
        absolute_done = instance_index + 1
        if absolute_done == midpoint:
            controller.save(seed_dir / f"controller_case_{midpoint}.npz")
        if completed % max(1, int(args.progress_every)) == 0 or completed == len(eval_trace):
            _save_progress(
                args.output_dir / "progress.json",
                seed=int(forward_seed),
                stage="evaluation_1000",
                done=completed,
                total=len(eval_trace),
                controller=controller,
            )
            controller.save(seed_dir / "controller_partial.npz")
            print(
                json.dumps(
                    {
                        "stage": "exp44_online_evaluation",
                        "seed": int(forward_seed),
                        "done": completed,
                        "total": len(eval_trace),
                        "steps": int(controller.steps),
                        "epsilon": float(controller.epsilon),
                    }
                ),
                flush=True,
            )
    controller.save(seed_dir / f"controller_case_{int(args.eval_end)}.npz")

    bandit_params = [record["params"] for record in evaluation_records["bandit_default"]]
    same_setup_rates = {
        method: float(
            np.mean(
                [
                    record["params"] == reference
                    for record, reference in zip(evaluation_records[method], bandit_params)
                ]
            )
        )
        for method in BANDIT_METHODS
    }
    method_summaries = {
        method: _method_stream_summary(records)
        for method, records in evaluation_records.items()
    }
    result = {
        "completed": True,
        "forward_seed": int(forward_seed),
        "trace": {
            "cases": int(len(trace)),
            "sha256": trace_hash,
            "audit_path": str(seed_dir / "trace_audit.json"),
            "setup_bandit": trace_audit["summary"],
        },
        "window": {
            "name": "eval_1000",
            "start": int(args.eval_start),
            "end": int(args.eval_end),
            "cases": int(len(eval_trace)),
            "no_eval_500_window": True,
        },
        "adaptation": {
            "start": 0,
            "end": int(args.eval_start),
            "cases": int(len(adaptation_records)),
            "excluded_from_main_comparison": True,
            "summary": _method_stream_summary(adaptation_records),
            "action_diagnostics": _action_diagnostics(
                [record["outcome"] for record in adaptation_records], controller.weights
            ),
            "records": adaptation_records,
        },
        "evaluation": {
            "methods": method_summaries,
            "comparisons": _per_seed_comparisons(
                evaluation_records,
                seed=int(args.bootstrap_seed + int(forward_seed)),
            ),
            "same_setup_rate": same_setup_rates,
            "action_diagnostics": {
                "bandit_sarsa": _action_diagnostics(
                    [record["outcome"] for record in evaluation_records["bandit_sarsa"]],
                    controller.weights,
                ),
                "bandit_ppo": _continuous_action_diagnostics(
                    [record["outcome"] for record in evaluation_records["bandit_ppo"]]
                ),
            },
            "records": evaluation_records,
        },
        "controller": {
            "seed": int(args.controller_seed),
            "steps": int(controller.steps),
            "episodes": int(controller.episodes),
            "epsilon": float(controller.epsilon),
            "feature_dim": int(encoder.feature_dim),
            "config": controller.config.__dict__,
            "checkpoints": {
                "case_0": str(seed_dir / "controller_case_0.npz"),
                f"case_{int(args.eval_start)}": str(
                    seed_dir / f"controller_case_{int(args.eval_start)}.npz"
                ),
                f"case_{midpoint}": str(seed_dir / f"controller_case_{midpoint}.npz"),
                f"case_{int(args.eval_end)}": str(
                    seed_dir / f"controller_case_{int(args.eval_end)}.npz"
                ),
            },
        },
    }
    _write_json(result_path, result)
    return result


def _aggregate(
    args: argparse.Namespace,
    *,
    seed_results: Sequence[Dict[str, Any]],
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    seed_records = [dict(result["evaluation"]["records"]) for result in seed_results]
    combined_records = {
        method: [
            record
            for records in seed_records
            for record in records[method]
        ]
        for method in METHODS
    }
    method_summaries = {
        method: _method_stream_summary(records)
        for method, records in combined_records.items()
    }
    aggregate = {
        "completed": True,
        "protocol": protocol,
        "window": {
            "name": "eval_1000",
            "start": int(args.eval_start),
            "end": int(args.eval_end),
            "cases_per_seed": int(args.eval_end - args.eval_start),
            "total_cases": int(len(seed_results) * (args.eval_end - args.eval_start)),
            "no_eval_500_window": True,
        },
        "methods": method_summaries,
        "clustered_comparisons": _clustered_comparisons(
            seed_records,
            seed=int(args.bootstrap_seed),
        ),
        "per_seed": {
            str(result["forward_seed"]): {
                "result_path": str(
                    args.output_dir / f"seed_{int(result['forward_seed'])}" / "result.json"
                ),
                "trace_sha256": result["trace"]["sha256"],
                "methods": result["evaluation"]["methods"],
                "comparisons": result["evaluation"]["comparisons"],
                "same_setup_rate": result["evaluation"]["same_setup_rate"],
                "controller": result["controller"],
            }
            for result in seed_results
        },
        "action_diagnostics": {
            "bandit_sarsa": _action_diagnostics(
                [record["outcome"] for record in combined_records["bandit_sarsa"]],
                _parse_values(args.weights, float),
            ),
            "bandit_ppo": _continuous_action_diagnostics(
                [record["outcome"] for record in combined_records["bandit_ppo"]]
            ),
        },
    }
    _write_json(args.output_dir / "result.json", aggregate)
    _write_main_table(args.output_dir / "main_table.csv", method_summaries)
    return aggregate


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if (
        int(args.eval_end - args.eval_start) != 1000
        and not bool(args.allow_noncanonical_smoke)
    ):
        raise ValueError("This formal run requires one 1000-case evaluation window")
    if not (0 < int(args.eval_start) < int(args.eval_end) <= int(args.trace_t)):
        raise ValueError("Evaluation bounds must fit within the forward trace")
    if str(args.td_algorithm) != "true_online_sarsa":
        raise ValueError("This comparison requires true_online_sarsa")
    if str(args.exploration_mode) != "uniform":
        raise ValueError("This comparison requires uniform exploration")
    if float(args.td_decay_power) != 0.0:
        raise ValueError("true_online_sarsa requires td_decay_power=0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.ppo_model = args.ppo_model.resolve()
    if not args.ppo_model.exists():
        raise FileNotFoundError(args.ppo_model)
    ppo_sha = _sha256_file(args.ppo_model)
    if args.expected_ppo_sha256 and ppo_sha != str(args.expected_ppo_sha256):
        raise ValueError(
            f"PPO SHA256 mismatch: expected {args.expected_ppo_sha256}, got {ppo_sha}"
        )
    snapshot_dir = args.output_dir / "model_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"model_best_{ppo_sha}.zip"
    if not snapshot_path.exists():
        shutil.copy2(args.ppo_model, snapshot_path)
    args.ppo_model = snapshot_path

    environment = _configure_environment(args)
    forward_seeds = _parse_values(args.forward_seeds, int)
    protocol = {
        "name": "Active Exp44 frozen PPO vs on-stream true-online SARSA(lambda)",
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "forward_seeds": [int(seed) for seed in forward_seeds],
        "trace_cases_per_seed": int(args.trace_t),
        "online_adaptation_window": [0, int(args.eval_start)],
        "evaluation_window": [int(args.eval_start), int(args.eval_end)],
        "evaluation_cases_per_seed": int(args.eval_end - args.eval_start),
        "only_1000_case_window": True,
        "methods": list(METHODS),
        "shared_setup_continuation": True,
        "setup_feedback_policy": "online LinUCBv4 updated by default solve, as in Active Exp44 README",
        "randomized_method_order_per_case": True,
        "method_order_seed": int(args.method_order_seed),
        "ppo": {
            "model": str(args.ppo_model),
            "sha256": ppo_sha,
            "training": "offline Active Exp44; excluded from comparison",
            "frozen": True,
            "observation": "cycle_action_setup",
            "action": "continuous residual, w += 0.02 * action",
        },
        "sarsa": {
            "training": "online from forward case 0; reset independently for each forward seed",
            "adaptation_cost_excluded_from_main_table": True,
            "continues_learning_during_evaluation": True,
            "controller_seed_reused_per_forward_seed": int(args.controller_seed),
            "algorithm": str(args.td_algorithm),
            "alpha": float(args.alpha),
            "trace_lambda": float(args.trace_lambda),
            "gamma": float(args.gamma),
            "state_mode": "setup_full",
            "weights": list(_parse_values(args.weights, float)),
            "exploration_mode": str(args.exploration_mode),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "potential_scale_sec": float(args.potential_scale_sec),
            "failure_penalty_sec": float(args.failure_penalty_sec),
        },
        "timing": {
            "primary": "native setup + native solve",
            "secondary": "native total + controller inference/update",
            "ppo_offline_training_excluded": True,
            "sarsa_cases_before_eval_excluded_but_audited": True,
            "setup_bandit_overhead_reported_separately": True,
        },
        "environment": environment,
    }
    _write_json(args.output_dir / "config.json", protocol)
    ppo_runner = make_exp44_ppo_runner(args)

    seed_results = []
    for forward_seed in forward_seeds:
        seed_results.append(
            _run_forward_seed(
                args,
                forward_seed=int(forward_seed),
                ppo_runner=ppo_runner,
            )
        )
    aggregate = _aggregate(args, seed_results=seed_results, protocol=protocol)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "exp44_ppo_sarsa_done",
                    "output": args.output_dir / "result.json",
                    "window": aggregate["window"],
                    "methods": aggregate["methods"],
                    "comparisons": aggregate["clustered_comparisons"],
                }
            )
        ),
        flush=True,
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Active Exp44 1000-case comparison of frozen PPO and on-stream "
            "true-online SARSA(lambda)."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ppo-model", type=Path, required=True)
    parser.add_argument(
        "--expected-ppo-sha256",
        default="83503658ec315c0b001a207970d451c36c6d180a3fab6a4a93eaf06fd88670d4",
    )
    parser.add_argument(
        "--forward-seeds",
        default="39393939,39394939,39400939,39406939,39412939",
    )
    parser.add_argument("--trace-t", type=int, default=2500)
    parser.add_argument("--eval-start", type=int, default=1500)
    parser.add_argument("--eval-end", type=int, default=2500)
    parser.add_argument(
        "--matrix-grid-n",
        "--grid-n",
        dest="grid_n",
        type=int,
        default=EXP44_MATRIX_GRID_N,
    )
    parser.add_argument(
        "--setup-param-resolution",
        type=int,
        default=EXP44_SETUP_PARAM_RESOLUTION,
    )
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--fixed-weight", type=float, default=1.6)
    parser.add_argument(
        "--weights",
        default="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--controller-seed", type=int, default=39394016)
    parser.add_argument("--method-order-seed", type=int, default=39394016)
    parser.add_argument("--bootstrap-seed", type=int, default=39394016)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--td-algorithm", default="true_online_sarsa")
    parser.add_argument("--exploration-mode", default="uniform")
    parser.add_argument("--td-decay-power", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=float, default=20000.0)
    parser.add_argument("--potential-scale-sec", type=float, default=0.001)
    parser.add_argument("--failure-penalty-sec", type=float, default=0.1)
    parser.add_argument("--initial-q-sec", type=float, default=0.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-noncanonical-smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
