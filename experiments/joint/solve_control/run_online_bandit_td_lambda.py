from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import copy
import hashlib
import json
import os
import pickle
import platform
import subprocess
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from online_td_lambda import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    run_td_episode,
    shaped_cycle_cost,
)
from online_td_experiment_common import (
    _action_diagnostics,
    _json_ready,
    _paired,
    _summarize,
    _write_json,
)
from run_online_bandit_rl import (
    _build_bandit,
    _deployment_safe_arms,
    _make_env,
    _select_frozen_eval_trace,
    _summarize_online_records,
)
from setup_aware_compare_common import (
    DEFAULT_SETUP_PARAMS,
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
    SetupObsEncoder,
    augment_setup_params,
    build_setup_parameter_spec,
    default_test_final_bandit_config_from_env,
    generate_difconv_instances,
    run_bandit_step_test_final,
    solve_fixed_w_case,
    solve_no_rl_case,
)


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def _instance_stream_hash(
    stream: Sequence[tuple[Dict[str, Any], np.ndarray]],
) -> str:
    digest = hashlib.sha256()
    for mkw, context in stream:
        payload = {
            "mkw": dict(mkw),
            "context": np.asarray(context, dtype=float).tolist(),
        }
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def _run_joint_episode(
    *,
    env: Any,
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
) -> Dict[str, Any]:
    feature_runtime = 0.0
    decision_runtime = 0.0
    update_runtime = 0.0
    td_errors: list[float] = []
    transitions: list[tuple[np.ndarray, int, float]] = []
    action_counts: Counter[float] = Counter()

    _obs, reset_info = env.reset()
    mkw = dict(env._mkw)
    setup_params = dict(getattr(env, "setup_params", {}))
    initial_residual = float(env._r_cur)
    residual = initial_residual
    previous_residual = initial_residual
    last_weight = float(controller.config.anchor_weight)
    last_cycle_time = 0.0
    cycle_index = 0
    controller.start_episode()

    feature_started = time.perf_counter()
    features = encoder.encode(
        mkw=mkw,
        setup_params=setup_params,
        initial_residual=initial_residual,
        residual=residual,
        previous_residual=previous_residual,
        cycle=0,
        last_weight=last_weight,
        last_cycle_time=last_cycle_time,
    )
    feature_runtime += float(time.perf_counter() - feature_started)

    terminated = False
    truncated = False
    step_info: Dict[str, Any] = {}
    while not (terminated or truncated):
        decision_started = time.perf_counter()
        action_index, weight, _selection = controller.select_action(
            features,
            explore=True,
            cycle=cycle_index,
            include_metadata=False,
        )
        decision_runtime += float(time.perf_counter() - decision_started)
        _obs, _reward, terminated, truncated, step_info = env.step(action_index)
        if "r" not in step_info or "dt_solver" not in step_info:
            break

        residual_new = float(step_info["r"])
        cycle_time = float(step_info["dt_solver"])
        cycle = int(step_info["cycle"])
        feature_started = time.perf_counter()
        next_features = encoder.encode(
            mkw=mkw,
            setup_params=setup_params,
            initial_residual=initial_residual,
            residual=residual_new,
            previous_residual=residual,
            cycle=cycle,
            last_weight=float(weight),
            last_cycle_time=cycle_time,
        )
        feature_runtime += float(time.perf_counter() - feature_started)

        transition_cost = shaped_cycle_cost(
            cycle_time,
            residual=residual,
            next_residual=residual_new,
            tol=encoder.tol,
            scale_sec=float(controller.config.potential_scale_sec),
            gamma=float(controller.config.gamma),
            terminal=bool(terminated or truncated),
        )
        if truncated:
            transition_cost += float(controller.config.failure_penalty_sec)
        transitions.append((features.copy(), int(action_index), float(transition_cost)))

        update_started = time.perf_counter()
        td_errors.append(
            controller.update(
                features=features,
                action_index=action_index,
                cost=transition_cost,
                next_features=next_features,
                terminal=bool(terminated or truncated),
                next_cycle=cycle,
            )
        )
        update_runtime += float(time.perf_counter() - update_started)
        action_counts[float(weight)] += 1
        features = next_features
        previous_residual = residual
        residual = residual_new
        last_weight = float(weight)
        last_cycle_time = cycle_time
        cycle_index = cycle

    if controller.config.monte_carlo_alpha > 0.0 and transitions:
        update_started = time.perf_counter()
        td_errors.extend(controller.monte_carlo_update(transitions))
        update_runtime += float(time.perf_counter() - update_started)
    controller.finish_episode(learned=bool(transitions))

    record = env.completed_episodes[-1]
    outcome = record["outcome"]
    policy_runtime = float(feature_runtime + decision_runtime + update_runtime)
    outcome["infer_runtime"] = policy_runtime
    outcome["feature_runtime"] = float(feature_runtime)
    outcome["decision_runtime"] = float(decision_runtime)
    outcome["update_runtime"] = float(update_runtime)
    outcome["runtime_with_overhead"] = float(outcome["runtime"] + policy_runtime)
    outcome["action_counts"] = {str(weight): int(count) for weight, count in sorted(action_counts.items())}
    outcome["cycle_actions"] = [float(action["w"]) for action in record["actions"]]
    outcome["mean_abs_td_error"] = float(np.mean(np.abs(td_errors))) if td_errors else 0.0
    outcome["epsilon"] = float(controller.epsilon)
    record["reset_info"] = dict(reset_info)
    return record


def _evaluate(
    *,
    trace: Sequence[tuple[Dict[str, Any], Dict[str, Any]]],
    selection_times: Sequence[float],
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    tol: float,
    max_cycles: int,
    seed: int,
) -> Dict[str, Any]:
    methods = ("default_solve", "fixed_w1.0", "fixed_w1.4", "td_lambda")
    results: Dict[str, list[Dict[str, Any]]] = {method: [] for method in methods}
    rng = np.random.default_rng(seed)
    for case_index, (mkw, params) in enumerate(trace):
        for method_index in rng.permutation(len(methods)):
            method = methods[int(method_index)]
            if method == "default_solve":
                outcome = solve_no_rl_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    solver_tol=float(tol),
                    solver_max_iter=int(max_cycles),
                    augment_params=augment_setup_params,
                )
            elif method.startswith("fixed_w"):
                outcome = solve_fixed_w_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    w=float(method.removeprefix("fixed_w")),
                    sweeps_down=1,
                    sweeps_up=1,
                    solve_tol=float(tol),
                    solve_max_cycles=int(max_cycles),
                )
            else:
                outcome = run_td_episode(
                    mkw=dict(mkw),
                    params=dict(params),
                    controller=controller,
                    encoder=encoder,
                    solve_tol=float(tol),
                    solve_max_cycles=int(max_cycles),
                    learn=False,
                    explore=False,
                )
            outcome = dict(outcome)
            outcome["infer_runtime"] = float(outcome.get("infer_runtime", 0.0)) + float(
                selection_times[case_index]
            )
            results[method].append(outcome)

    diagnostics = _action_diagnostics(results["td_lambda"], controller.weights)
    modal_weight = float(diagnostics["modal_weight"])
    modal_reference_key: str | None = None
    if np.isfinite(modal_weight):
        fixed_key = f"fixed_w{modal_weight:.1f}"
        if fixed_key in results:
            modal_reference_key = fixed_key
        else:
            modal_reference_key = f"modal_fixed_w{modal_weight:.3f}"
            results[modal_reference_key] = []
            for case_index, (mkw, params) in enumerate(trace):
                outcome = solve_fixed_w_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    w=modal_weight,
                    sweeps_down=1,
                    sweeps_up=1,
                    solve_tol=float(tol),
                    solve_max_cycles=int(max_cycles),
                )
                outcome["infer_runtime"] = float(selection_times[case_index])
                results[modal_reference_key].append(outcome)

    summaries = {method: _summarize(rows) for method, rows in results.items()}
    comparisons = {
        f"td_lambda_vs_{reference}": _paired(
            results[reference],
            results["td_lambda"],
            seed=seed + 100 + index * 2,
        )
        for index, reference in enumerate(("default_solve", "fixed_w1.0", "fixed_w1.4"))
    }
    if modal_reference_key is not None:
        comparisons["td_lambda_vs_modal_fixed"] = _paired(
            results[modal_reference_key],
            results["td_lambda"],
            seed=seed + 110,
        )
    return {
        "methods": summaries,
        "comparisons": comparisons,
        "action_diagnostics": diagnostics,
        "modal_reference_method": modal_reference_key,
        "per_case": results,
    }


def _configure_paired_environment(args: argparse.Namespace) -> None:
    setup_param_resolution = int(args.setup_param_resolution)
    values = {
        "MATRIX_GRID_N": int(args.grid_n),
        "SETUP_PARAM_RESOLUTION": setup_param_resolution,
        "SETUP_ACTION_SPACE": str(args.setup_action_space),
        "C_MIN": float(args.c_min),
        "C_MAX": float(args.c_max),
        "SOLVE_TOL": float(args.tol),
        "SOLVE_MAX_CYCLES": int(args.max_cycles),
        "ACTION_MODE": "discrete_w",
    }
    for name, value in values.items():
        os.environ[name] = str(value)
    if (
        str(args.setup_action_space) == "full_cartesian"
        and setup_param_resolution == EXP44_SETUP_PARAM_RESOLUTION
    ):
        os.environ["EXPECTED_SETUP_ACTION_COUNT"] = str(
            EXP44_TUNE7_CATEGORICAL_ACTION_COUNT
        )
    else:
        os.environ.pop("EXPECTED_SETUP_ACTION_COUNT", None)


def _policy_last_arm(policy: Any) -> int:
    model = getattr(policy, "model", policy)
    last_arm = getattr(model, "_last_arm", None)
    return -1 if last_arm is None else int(last_arm)


def _report_online_outcome(
    outcome: Dict[str, Any],
    *,
    bandit_timing: Dict[str, float],
) -> Dict[str, Any]:
    reported = dict(outcome)
    controller_runtime = float(reported.get("infer_runtime", 0.0))
    native_runtime = float(
        reported.get("native_runtime", float(reported["runtime"]) - controller_runtime)
    )
    native_solve_runtime = float(
        reported.get(
            "native_solve_runtime",
            float(reported.get("solve_runtime", 0.0)) - controller_runtime,
        )
    )
    bandit_overhead = float(bandit_timing.get("overhead_sec", 0.0))
    reported.update(
        {
            "runtime": native_runtime,
            "solve_runtime": native_solve_runtime,
            "infer_runtime": controller_runtime,
            "bandit_select_runtime": float(bandit_timing.get("select_sec", 0.0)),
            "bandit_loss_eval_runtime": float(
                bandit_timing.get("loss_eval_sec", 0.0)
            ),
            "bandit_update_runtime": float(bandit_timing.get("update_sec", 0.0)),
            "bandit_overhead_runtime": bandit_overhead,
            "end_to_end_runtime": float(
                native_runtime + controller_runtime + bandit_overhead
            ),
        }
    )
    return reported


def _method_stream_summary(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    outcomes = [record["outcome"] for record in records]
    summary = _summarize(outcomes)
    fields = {
        "setup_runtime": "setup_runtime",
        "native_solve_runtime": "solve_runtime",
        "native_total_runtime": "runtime",
        "controller_runtime": "infer_runtime",
        "setup_bandit_overhead": "bandit_overhead_runtime",
        "setup_bandit_select": "bandit_select_runtime",
        "setup_bandit_loss_eval": "bandit_loss_eval_runtime",
        "setup_bandit_update": "bandit_update_runtime",
        "end_to_end_runtime": "end_to_end_runtime",
    }
    summary["totals_sec"] = {
        label: float(sum(float(outcome.get(field, 0.0)) for outcome in outcomes))
        for label, field in fields.items()
    }
    summary["means_sec"] = {
        label: float(np.mean([float(outcome.get(field, 0.0)) for outcome in outcomes]))
        for label, field in fields.items()
    }
    summary["unique_setup_count"] = int(
        len({json.dumps(record["params"], sort_keys=True) for record in records})
    )
    summary["setup_retry_count"] = int(
        sum(int(record.get("failed_attempts", 0)) for record in records)
    )
    return summary


def _paired_metric(
    baseline_values: np.ndarray,
    rl_values: np.ndarray,
    *,
    seed: int,
) -> Dict[str, Any]:
    baseline = np.asarray(baseline_values, dtype=float)
    rl = np.asarray(rl_values, dtype=float)
    if baseline.shape != rl.shape or baseline.size == 0:
        raise ValueError("Paired metric arrays must be non-empty and aligned")
    difference = baseline - rl
    baseline_mean = float(np.mean(baseline))
    rl_mean = float(np.mean(rl))
    improvement = (
        float(100.0 * (baseline_mean - rl_mean) / baseline_mean)
        if baseline_mean != 0.0
        else float("nan")
    )
    rng = np.random.default_rng(int(seed))
    boot_difference = np.empty(2000, dtype=float)
    boot_improvement = np.empty(2000, dtype=float)
    for index in range(boot_difference.size):
        sample = rng.integers(0, baseline.size, size=baseline.size)
        sampled_baseline = float(np.mean(baseline[sample]))
        sampled_rl = float(np.mean(rl[sample]))
        boot_difference[index] = sampled_baseline - sampled_rl
        boot_improvement[index] = (
            100.0 * (sampled_baseline - sampled_rl) / sampled_baseline
            if sampled_baseline != 0.0
            else float("nan")
        )
    return {
        "cases": int(baseline.size),
        "baseline_mean_sec": baseline_mean,
        "online_rl_mean_sec": rl_mean,
        "baseline_total_sec": float(np.sum(baseline)),
        "online_rl_total_sec": float(np.sum(rl)),
        "online_rl_savings_sec": float(np.sum(difference)),
        "online_rl_mean_savings_sec": float(np.mean(difference)),
        "online_rl_mean_savings_95pct_sec": [
            float(np.nanpercentile(boot_difference, 2.5)),
            float(np.nanpercentile(boot_difference, 97.5)),
        ],
        "online_rl_improvement_pct": improvement,
        "online_rl_improvement_95pct": [
            float(np.nanpercentile(boot_improvement, 2.5)),
            float(np.nanpercentile(boot_improvement, 97.5)),
        ],
        "online_rl_win_rate": float(np.mean(difference > 0.0)),
    }


def _paired_stream_comparison(
    rl_records: Sequence[Dict[str, Any]],
    default_records: Sequence[Dict[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    if len(rl_records) != len(default_records):
        raise ValueError("Online streams must contain the same number of instances")
    accessors = {
        "setup_runtime": lambda row: float(row["outcome"]["setup_runtime"]),
        "native_solve_runtime": lambda row: float(row["outcome"]["solve_runtime"]),
        "native_total_runtime": lambda row: float(row["outcome"]["runtime"]),
        "controller_runtime": lambda row: float(row["outcome"].get("infer_runtime", 0.0)),
        "setup_bandit_overhead": lambda row: float(
            row["outcome"].get("bandit_overhead_runtime", 0.0)
        ),
        "solve_plus_controller": lambda row: float(
            row["outcome"]["solve_runtime"]
            + row["outcome"].get("infer_runtime", 0.0)
        ),
        "end_to_end_runtime": lambda row: float(row["outcome"]["end_to_end_runtime"]),
    }
    comparison = {}
    for offset, (name, accessor) in enumerate(accessors.items()):
        comparison[name] = _paired_metric(
            np.asarray([accessor(row) for row in default_records], dtype=float),
            np.asarray([accessor(row) for row in rl_records], dtype=float),
            seed=int(seed + offset),
        )
    comparison["same_setup_rate"] = float(
        np.mean(
            [
                rl_record["params"] == default_record["params"]
                for rl_record, default_record in zip(rl_records, default_records)
            ]
        )
    )
    return comparison


def _cumulative_checkpoints(
    rl_records: Sequence[Dict[str, Any]],
    default_records: Sequence[Dict[str, Any]],
    *,
    every: int,
) -> list[Dict[str, Any]]:
    checkpoints = []
    for end in range(int(every), len(rl_records) + 1, int(every)):
        rl_slice = rl_records[:end]
        default_slice = default_records[:end]
        row = {"cases": int(end)}
        for name, field in (
            ("setup_runtime", "setup_runtime"),
            ("native_solve_runtime", "solve_runtime"),
            ("native_total_runtime", "runtime"),
            ("end_to_end_runtime", "end_to_end_runtime"),
        ):
            rl_total = float(sum(float(record["outcome"][field]) for record in rl_slice))
            default_total = float(
                sum(float(record["outcome"][field]) for record in default_slice)
            )
            row[name] = {
                "online_rl_total_sec": rl_total,
                "baseline_total_sec": default_total,
                "online_rl_savings_sec": float(default_total - rl_total),
                "online_rl_improvement_pct": float(
                    100.0 * (default_total - rl_total) / default_total
                ),
            }
        checkpoints.append(row)
    return checkpoints


def _build_paired_instance_stream(
    args: argparse.Namespace,
) -> tuple[list[tuple[Dict[str, Any], np.ndarray]], Dict[str, Any]]:
    raw_groups = str(args.train_seed_groups).strip()
    if not raw_groups:
        instance_offset = int(args.instance_offset)
        if instance_offset < 0:
            raise ValueError("instance_offset must be non-negative")
        full_stream = generate_difconv_instances(
            T=int(args.train_cases) + instance_offset,
            seed=int(args.seed),
            grid_choices=[(int(args.grid_n),) * 3],
            c_min=float(args.c_min),
            c_max=float(args.c_max),
            difconv_a=(0.0, 0.0, 0.0),
        )
        selected_stream = full_stream[instance_offset:]
        return selected_stream, {
            "mode": "single_seed_continuation",
            "seed": int(args.seed),
            "window": [instance_offset, instance_offset + int(args.train_cases)],
            "sha256": _instance_stream_hash(selected_stream),
        }

    seed_groups = [
        _parse_values(group, int)
        for group in raw_groups.split(";")
        if group.strip()
    ]
    shuffle_seeds = _parse_values(args.train_shuffle_seeds, int)
    if len(seed_groups) != len(shuffle_seeds):
        raise ValueError("train_seed_groups and train_shuffle_seeds must have equal lengths")

    stream: list[tuple[Dict[str, Any], np.ndarray]] = []
    segment_metadata = []
    for segment_index, (seeds, shuffle_seed) in enumerate(
        zip(seed_groups, shuffle_seeds)
    ):
        candidates: list[tuple[Dict[str, Any], np.ndarray]] = []
        for seed in seeds:
            generated = generate_difconv_instances(
                T=int(args.instance_offset + args.train_cases_per_seed),
                seed=int(seed),
                grid_choices=[(int(args.grid_n),) * 3],
                c_min=float(args.c_min),
                c_max=float(args.c_max),
                difconv_a=(0.0, 0.0, 0.0),
            )
            candidates.extend(generated[int(args.instance_offset) :])
        order = np.random.default_rng(int(shuffle_seed)).permutation(len(candidates))
        selected = [
            candidates[int(index)]
            for index in order[: int(args.train_group_take)]
        ]
        stream.extend(selected)
        segment_metadata.append(
            {
                "segment": int(segment_index),
                "seeds": [int(seed) for seed in seeds],
                "cases_per_seed": int(args.train_cases_per_seed),
                "window": [
                    int(args.instance_offset),
                    int(args.instance_offset + args.train_cases_per_seed),
                ],
                "candidate_cases": int(len(candidates)),
                "shuffle_seed": int(shuffle_seed),
                "selected_cases": int(len(selected)),
            }
        )
    if len(stream) != int(args.train_cases):
        raise ValueError(
            f"Grouped training stream produced {len(stream)} cases, expected {args.train_cases}"
        )
    return stream, {
        "mode": "grouped_shuffled_segments",
        "segments": segment_metadata,
        "sha256": _instance_stream_hash(stream),
    }


def _run_paired_online_comparison(
    *,
    args: argparse.Namespace,
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    config: ExpectedSarsaLambdaConfig,
) -> None:
    _configure_paired_environment(args)
    if args.initial_bandit_state is None:
        initial_branch, bandit_cfg = _build_bandit(
            seed=int(args.seed),
            tune_dim=7,
            tune7_variant="categorical",
        )
    else:
        with args.initial_bandit_state.open("rb") as handle:
            initial_branch = pickle.load(handle)["branch"]
        bandit_cfg = default_test_final_bandit_config_from_env()
    rl_branch = copy.deepcopy(initial_branch)
    default_branch = copy.deepcopy(initial_branch)
    instances, instance_stream = _build_paired_instance_stream(args)
    action_count = int(rl_branch.policy.model.K)
    protocol = {
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "seed": int(args.seed),
        "instance_stream": instance_stream,
        "instances": int(args.train_cases),
        "grid": [int(args.grid_n)] * 3,
        "coefficient_range": [float(args.c_min), float(args.c_max)],
        "difconv_a": [0.0, 0.0, 0.0],
        "tol": float(args.tol),
        "max_cycles": int(args.max_cycles),
        "comparison": {
            "online_rl": "online LinUCBv4 setup + continually online per-cycle Expected SARSA(lambda)",
            "default": "independent online LinUCBv4 setup + native default solve",
            "same_instance_stream": True,
            "identical_initial_bandit_state": True,
            "randomized_method_order_per_instance": True,
        },
        "setup_bandit": {
            "action_space": str(args.setup_action_space),
            "actions": action_count,
            "feature_dim": int(rl_branch.policy.model.d_phi),
            "starts_from_scratch": args.initial_bandit_state is None,
            "initial_state": (
                None
                if args.initial_bandit_state is None
                else str(args.initial_bandit_state)
            ),
            "retry_max_attempts": int(bandit_cfg.retry_max_attempts),
        },
        "controller": config.__dict__,
        "state_mode": str(args.state_mode),
        "feature_dim": int(encoder.feature_dim),
    }
    _write_json(args.output_dir / "config.json", protocol)
    print(json.dumps(_json_ready({"stage": "paired_online_start", **protocol})), flush=True)

    histories = {
        "online_rl": deque(maxlen=max(1, int(bandit_cfg.failure_scale_window))),
        "default": deque(maxlen=max(1, int(bandit_cfg.failure_scale_window))),
    }
    previous_update = {"online_rl": 0.0, "default": 0.0}
    records: Dict[str, list[Dict[str, Any]]] = {"online_rl": [], "default": []}
    rng = np.random.default_rng(int(args.seed + 91_003))

    for index, (mkw, context) in enumerate(instances):
        method_order = ["online_rl", "default"]
        rng.shuffle(method_order)
        case_records: Dict[str, Dict[str, Any]] = {}
        for method in method_order:
            branch = rl_branch if method == "online_rl" else default_branch

            if method == "online_rl":
                def solver_fn(params: Dict[str, Any]) -> Dict[str, Any]:
                    native = run_td_episode(
                        mkw=dict(mkw),
                        params=dict(params),
                        controller=controller,
                        encoder=encoder,
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                        learn=True,
                        explore=True,
                    )
                    feedback = dict(native)
                    feedback["native_runtime"] = float(native["runtime"])
                    feedback["native_solve_runtime"] = float(native["solve_runtime"])
                    feedback["runtime"] = float(
                        native["runtime"] + native.get("infer_runtime", 0.0)
                    )
                    feedback["solve_runtime"] = float(
                        native["solve_runtime"] + native.get("infer_runtime", 0.0)
                    )
                    return feedback
            else:
                def solver_fn(params: Dict[str, Any]) -> Dict[str, Any]:
                    native = solve_no_rl_case(
                        params=dict(params),
                        mkw=dict(mkw),
                        solver_tol=float(args.tol),
                        solver_max_iter=int(args.max_cycles),
                        augment_params=augment_setup_params,
                    )
                    feedback = dict(native)
                    feedback["native_runtime"] = float(native["runtime"])
                    feedback["native_solve_runtime"] = float(native["solve_runtime"])
                    feedback["infer_runtime"] = 0.0
                    return feedback

            params, outcome, timing, failed_attempts, update_sec = run_bandit_step_test_final(
                policy=branch.policy,
                parameter_space=branch.parameter_space,
                context=np.asarray(context, dtype=float),
                solver_fn=solver_fn,
                prev_update_est=float(previous_update[method]),
                success_runtime_history=histories[method],
                b_min_runtime_sec=1.0e-3,
                solver_tol=float(args.tol),
                cfg=bandit_cfg,
            )
            previous_update[method] = float(update_sec)
            reported = _report_online_outcome(outcome, bandit_timing=timing)
            case_records[method] = {
                "instance_index": int(index),
                "mkw": dict(mkw),
                "context": np.asarray(context, dtype=float).tolist(),
                "params": dict(params),
                "arm_index": _policy_last_arm(branch.policy),
                "failed_attempts": int(failed_attempts),
                "bandit_timing": dict(timing),
                "outcome": reported,
            }

        for method in ("online_rl", "default"):
            records[method].append(case_records[method])

        done = index + 1
        if done % max(1, int(args.progress_every)) == 0 or done == int(args.train_cases):
            partial = {
                "protocol": protocol,
                "completed_instances": int(done),
                "methods": {
                    method: _method_stream_summary(method_records)
                    for method, method_records in records.items()
                },
                "comparison": _paired_stream_comparison(
                    records["online_rl"],
                    records["default"],
                    seed=int(args.seed + done),
                ),
            }
            _write_json(args.output_dir / "progress.json", partial)
            controller.save(args.output_dir / "td_lambda_controller_partial.npz")
            print(
                json.dumps(
                    _json_ready(
                        {
                            "stage": "paired_online_progress",
                            "done": int(done),
                            "total": int(args.train_cases),
                            "td_steps": int(controller.steps),
                            "epsilon": float(controller.epsilon),
                            "setup": partial["comparison"]["setup_runtime"],
                            "solve": partial["comparison"]["native_solve_runtime"],
                            "end_to_end": partial["comparison"]["end_to_end_runtime"],
                        }
                    )
                ),
                flush=True,
            )

    controller_path = args.output_dir / "td_lambda_controller.npz"
    controller.save(controller_path)
    with (args.output_dir / "online_bandit_states.pkl").open("wb") as handle:
        pickle.dump(
            {"online_rl": rl_branch, "default": default_branch},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    all_window_name = f"all_{len(records['online_rl'])}"
    windows = {
        all_window_name: (records["online_rl"], records["default"]),
        "last_1000": (records["online_rl"][-1000:], records["default"][-1000:]),
        "last_500": (records["online_rl"][-500:], records["default"][-500:]),
    }
    result = {
        "protocol": protocol,
        "methods": {
            method: _method_stream_summary(method_records)
            for method, method_records in records.items()
        },
        "comparisons": {
            name: _paired_stream_comparison(
                rl_window,
                default_window,
                seed=int(args.seed + 100_000 + offset),
            )
            for offset, (name, (rl_window, default_window)) in enumerate(windows.items())
        },
        "cumulative_checkpoints": _cumulative_checkpoints(
            records["online_rl"],
            records["default"],
            every=max(1, int(args.progress_every)),
        ),
        "action_diagnostics": _action_diagnostics(
            [record["outcome"] for record in records["online_rl"]],
            controller.weights,
        ),
        "controller": {
            "checkpoint": str(controller_path),
            "steps": int(controller.steps),
            "episodes": int(controller.episodes),
            "epsilon": float(controller.epsilon),
        },
        "online_records": records,
    }
    _write_json(args.output_dir / "result.json", result)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "paired_online_done",
                    "output": args.output_dir / "result.json",
                    all_window_name: result["comparisons"][all_window_name],
                }
            )
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Joint online SharedLinUCB + per-cycle TD(lambda).")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--train-cases", type=int, default=1024)
    parser.add_argument("--eval-cases", type=int, default=96)
    parser.add_argument("--matrix-grid-n", "--grid-n", dest="grid_n", type=int, default=30)
    parser.add_argument("--setup-param-resolution", type=int, default=8)
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--paired-online-default", action="store_true")
    parser.add_argument("--initial-bandit-state", type=Path)
    parser.add_argument("--instance-offset", type=int, default=0)
    parser.add_argument("--train-seed-groups", default="")
    parser.add_argument("--train-shuffle-seeds", default="")
    parser.add_argument("--train-cases-per-seed", type=int, default=500)
    parser.add_argument("--train-group-take", type=int, default=1000)
    parser.add_argument(
        "--setup-action-space",
        choices=("safe_one_at_a_time", "full_cartesian"),
        default="safe_one_at_a_time",
    )
    parser.add_argument(
        "--state-mode",
        choices=("cycle_tabular", "full", "setup_full"),
        default="cycle_tabular",
    )
    parser.add_argument("--weights", default="1.4,1.6")
    parser.add_argument("--anchor-weight", type=float, default=1.4)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--td-decay-power", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--failure-penalty-sec", type=float, default=0.05)
    parser.add_argument("--monte-carlo-alpha", type=float, default=0.02)
    parser.add_argument("--monte-carlo-decay-power", type=float, default=0.0)
    parser.add_argument("--adaptive-cycles", type=int, default=2)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-final", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=float, default=8000.0)
    parser.add_argument("--initial-q-sec", type=float, default=0.02)
    parser.add_argument("--potential-scale-sec", type=float, default=0.0)
    parser.add_argument(
        "--exploration-mode",
        choices=("uniform", "least_visited"),
        default="uniform",
    )
    parser.add_argument("--action-rbf-sigma", type=float, default=0.0)
    parser.add_argument("--policy-cost-initial-sec", type=float, default=3.0e-5)
    parser.add_argument("--pretrained-controller", type=Path)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = _parse_values(args.weights, float)
    config = ExpectedSarsaLambdaConfig(
        weights=weights,
        anchor_weight=float(args.anchor_weight),
        alpha=float(args.alpha),
        td_decay_power=float(args.td_decay_power),
        gamma=float(args.gamma),
        trace_lambda=float(args.trace_lambda),
        epsilon_start=float(args.epsilon_start),
        epsilon_final=float(args.epsilon_final),
        epsilon_decay_steps=float(args.epsilon_decay_steps),
        initial_q_sec=float(args.initial_q_sec),
        monte_carlo_alpha=float(args.monte_carlo_alpha),
        monte_carlo_decay_power=float(args.monte_carlo_decay_power),
        adaptive_cycles=(
            None if int(args.adaptive_cycles) < 0 else int(args.adaptive_cycles)
        ),
        potential_scale_sec=float(args.potential_scale_sec),
        failure_penalty_sec=float(args.failure_penalty_sec),
        exploration_mode=str(args.exploration_mode),
        action_rbf_sigma=float(args.action_rbf_sigma),
    )
    setup_obs_encoder = None
    if str(args.state_mode) == "setup_full":
        setup_spec, _fixed_setup_params = build_setup_parameter_spec(
            tune_dim=7,
            tune7_variant="categorical",
        )
        setup_keys = tuple(setup_spec.parameter_names)
        setup_obs_encoder = SetupObsEncoder(
            setup_spec,
            dict(DEFAULT_SETUP_PARAMS),
            setup_keys,
        )
    encoder = SolveStateEncoder(
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        mode=str(args.state_mode),
        setup_obs_encoder=setup_obs_encoder,
    )
    controller = ExpectedSarsaLambda(
        feature_dim=encoder.feature_dim,
        config=config,
        seed=int(args.seed),
        initial_parameters=encoder.constant_value_parameters(config.initial_q_sec),
    )
    pretrained_controller = None
    if args.pretrained_controller is not None:
        pretrained_controller = controller.load(args.pretrained_controller.resolve())

    if args.paired_online_default:
        if pretrained_controller is not None:
            raise ValueError("Paired online comparison must start without a pretrained controller")
        _run_paired_online_comparison(
            args=args,
            controller=controller,
            encoder=encoder,
            config=config,
        )
        return

    branch, bandit_cfg = _build_bandit(seed=args.seed, tune_dim=7, tune7_variant="categorical")
    train_instances = generate_difconv_instances(
        T=int(args.train_cases),
        seed=int(args.seed),
        grid_choices=[(int(args.grid_n),) * 3],
        c_min=float(args.c_min),
        c_max=float(args.c_max),
    )
    env = _make_env(
        instances=train_instances,
        branch=branch,
        bandit_cfg=bandit_cfg,
        seed=args.seed,
        discrete_w_values=weights,
        policy_step_cost_sec=float(args.policy_cost_initial_sec),
        potential_progress_scale=float(args.potential_scale_sec),
    )

    protocol = {
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "seed": int(args.seed),
        "train_cases": int(args.train_cases),
        "eval_cases": int(args.eval_cases),
        "grid": [int(args.grid_n)] * 3,
        "c_range": [float(args.c_min), float(args.c_max)],
        "tol": float(args.tol),
        "max_cycles": int(args.max_cycles),
        "controller": config.__dict__,
        "pretrained_controller": pretrained_controller,
        "setup_bandit": {
            "actions": int(branch.policy.model.K),
            "feature_dim": int(branch.policy.model.d_phi),
        },
    }
    _write_json(args.output_dir / "config.json", protocol)
    print(json.dumps(_json_ready({"stage": "joint_td_start", **protocol})), flush=True)

    overhead_per_cycle = deque([float(args.policy_cost_initial_sec)], maxlen=100)
    for episode in range(int(args.train_cases)):
        env.policy_step_cost_sec = float(np.mean(overhead_per_cycle))
        record = _run_joint_episode(env=env, controller=controller, encoder=encoder)
        outcome = record["outcome"]
        if int(outcome.get("iterations", 0)) > 0:
            overhead_per_cycle.append(
                float(outcome.get("infer_runtime", 0.0)) / float(outcome["iterations"])
            )
        if (episode + 1) % max(1, args.progress_every) == 0 or episode + 1 == int(args.train_cases):
            print(
                json.dumps(
                    {
                        "stage": "joint_td_progress",
                        "done": episode + 1,
                        "total": int(args.train_cases),
                        "bandit_updates": len(env.bandit_updates),
                        "td_steps": controller.steps,
                        "epsilon": controller.epsilon,
                        "policy_cost_estimate_sec": env.policy_step_cost_sec,
                    }
                ),
                flush=True,
            )

    online_records = copy.deepcopy(env.completed_episodes)
    bandit_updates = copy.deepcopy(env.bandit_updates)
    env.close()
    controller.save(args.output_dir / "td_lambda_controller.npz")
    with (args.output_dir / "online_bandit_state.pkl").open("wb") as handle:
        pickle.dump({"branch": branch}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    eval_instances = generate_difconv_instances(
        T=int(args.eval_cases),
        seed=int(args.seed + 100_000),
        grid_choices=[(int(args.grid_n),) * 3],
        c_min=float(args.c_min),
        c_max=float(args.c_max),
    )
    deployment_arms, deployment_safety = _deployment_safe_arms(
        branch,
        online_records,
        max_cycle_fraction=0.7,
    )
    eval_trace, selection_times, selection_wall_sec = _select_frozen_eval_trace(
        branch,
        eval_instances,
        candidate_arms=deployment_arms,
    )
    evaluation = _evaluate(
        trace=eval_trace,
        selection_times=selection_times,
        controller=copy.deepcopy(controller),
        encoder=encoder,
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        seed=int(args.seed + 100_001),
    )
    evaluation["bandit_selection_wall_sec"] = float(selection_wall_sec)
    evaluation["deployment_safety"] = deployment_safety

    payload = {
        "protocol": protocol,
        "training": {
            "summary": _summarize_online_records(online_records, window=min(100, len(online_records))),
            "action_diagnostics": _action_diagnostics(
                [record["outcome"] for record in online_records],
                weights,
            ),
            "policy_cost_per_cycle_sec": float(np.mean(overhead_per_cycle)),
        },
        "evaluation": evaluation,
        "controller": {
            "steps": int(controller.steps),
            "episodes": int(controller.episodes),
            "epsilon": float(controller.epsilon),
            "theta": controller.theta.tolist(),
        },
        "online_records": online_records,
        "bandit_updates": bandit_updates,
    }
    _write_json(args.output_dir / "result.json", payload)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "joint_td_done",
                    "output": args.output_dir / "result.json",
                    "comparisons": evaluation["comparisons"],
                    "actions": evaluation["action_diagnostics"],
                }
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
