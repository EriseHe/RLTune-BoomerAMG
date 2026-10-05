from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import copy
import json
import pickle
import platform
from collections import Counter, deque
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import numpy as np

from experiments.joint.solve_control.comparison import method_comparison, paired_metric
from experiments.joint.solve_control.feedback import as_feedback

from solve.controllers.sarsa import ExpectedSarsaLambda
from solve.controllers.common.state_encoder import SolveStateEncoder
from solve.core.episode import run_td_episode
from solve.controllers.ppo import (
    FrozenPpoConfig,
    SetupAwareSolvePolicyRunner,
    build_frozen_ppo_runner,
)
from experiments.joint.solve_control.online_td_experiment_common import _action_diagnostics, _json_ready, _write_json
from experiments.joint.solve_control.run_exp44_online_rl import make_controller
from experiments.joint.solve_control.joint_online_common import (
    _build_paired_instance_stream,
    _configure_paired_environment,
    _git_revision,
    _method_stream_summary,
    _policy_last_arm,
    _report_online_outcome,
    _validate_recovery_stream,
)
from setup.space import DEFAULT_SETUP_PARAMS
from experiments.joint.solve_control.action_spaces import EXP44_MATRIX_GRID_N, EXP44_SETUP_PARAM_RESOLUTION
from hypre.bindings import augment_setup_params
from solve.core.outcomes import classify_rl_failure
from experiments.joint.solve_control.setup_branches import default_test_final_bandit_config_from_env, run_bandit_step_test_final, validate_expected_setup_action_count
from experiments.joint.solve_control.native_evaluation import solve_default_baseline_case, solve_fixed_w_case, solve_no_rl_case, solve_setup_aware_rl_case


METHODS = (
    "default_setup",
    "bandit_default",
    "bandit_fixed_w1.6",
    "bandit_ppo",
    "bandit_td",
)
BANDIT_METHODS = METHODS[1:]
BASELINE_METHOD = "bandit_default"

# Compatibility names for historical scripts and result analysis.
_as_feedback = as_feedback
_method_comparison = method_comparison
_paired_metric = paired_metric


def _window_comparisons(
    records: Dict[str, list[Dict[str, Any]]],
    *,
    seed: int,
) -> Dict[str, Any]:
    baseline = records[BASELINE_METHOD]
    return {
        method: method_comparison(method_records, baseline, seed=int(seed + offset * 100))
        for offset, (method, method_records) in enumerate(records.items())
        if method != BASELINE_METHOD
    }


def _cumulative_checkpoints(
    records: Dict[str, list[Dict[str, Any]]],
    *,
    every: int,
) -> list[Dict[str, Any]]:
    checkpoints: list[Dict[str, Any]] = []
    total_cases = len(records[BASELINE_METHOD])
    for end in range(int(every), total_cases + 1, int(every)):
        row: Dict[str, Any] = {"cases": int(end), "methods": {}}
        for method, method_records in records.items():
            outcomes = [record["outcome"] for record in method_records[:end]]
            row["methods"][method] = {
                "setup_runtime": float(sum(float(out["setup_runtime"]) for out in outcomes)),
                "native_solve_runtime": float(sum(float(out["solve_runtime"]) for out in outcomes)),
                "native_total_runtime": float(sum(float(out["runtime"]) for out in outcomes)),
                "controller_runtime": float(
                    sum(float(out.get("infer_runtime", 0.0)) for out in outcomes)
                ),
                "setup_bandit_overhead": float(
                    sum(float(out.get("bandit_overhead_runtime", 0.0)) for out in outcomes)
                ),
                "end_to_end_runtime": float(
                    sum(float(out["end_to_end_runtime"]) for out in outcomes)
                ),
            }
        checkpoints.append(row)
    return checkpoints


def _continuous_action_diagnostics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_cycle: Dict[int, list[float]] = {}
    all_weights: list[float] = []
    for row in rows:
        for cycle, weight in enumerate(row.get("cycle_actions", [])):
            by_cycle.setdefault(int(cycle), []).append(float(weight))
            all_weights.append(float(weight))

    def summarize(values: Sequence[float]) -> Dict[str, float]:
        array = np.asarray(values, dtype=float)
        return {
            "count": int(array.size),
            "mean": float(np.mean(array)),
            "p10": float(np.percentile(array, 10)),
            "median": float(np.median(array)),
            "p90": float(np.percentile(array, 90)),
        }

    return {
        "overall": summarize(all_weights) if all_weights else {},
        "by_cycle": {
            str(cycle): summarize(values) for cycle, values in sorted(by_cycle.items())
        },
    }


def make_exp44_ppo_runner(args: argparse.Namespace) -> SetupAwareSolvePolicyRunner:
    """Compatibility delegate to the solve-owned frozen PPO factory."""

    return build_frozen_ppo_runner(
        FrozenPpoConfig.from_runtime(args)
    )


def make_true_online_sarsa_controller(
    args: argparse.Namespace,
) -> tuple[ExpectedSarsaLambda, SolveStateEncoder]:
    controller_args = copy.copy(args)
    controller_args.adaptive_cycles = -1
    controller_args.state_mode = "setup_full"
    controller_args.action_rbf_sigma = 0.0
    controller_args.monte_carlo_alpha = 0.0
    controller_args.monte_carlo_decay_power = 0.0
    controller, encoder, _config = make_controller(controller_args)
    return controller, encoder


# Backward-compatible private aliases for the existing 2K entrypoint and tests.
_ppo_runner = make_exp44_ppo_runner
_td_controller = make_true_online_sarsa_controller


def _solver_for_method(
    method: str,
    *,
    args: argparse.Namespace,
    mkw: Dict[str, Any],
    case_progress: float,
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    ppo_runner: SetupAwareSolvePolicyRunner,
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    if method == "bandit_td":
        def solve_td(params: Dict[str, Any]) -> Dict[str, Any]:
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
                fallback_attempt=lambda: solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=float(args.tol),
                    solver_max_iter=int(args.max_cycles),
                    augment_params=augment_setup_params,
                ),
            )
            return as_feedback(native, include_controller=True)

        return solve_td
    if method == "bandit_ppo":
        def solve_ppo(params: Dict[str, Any]) -> Dict[str, Any]:
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
            return as_feedback(native, include_controller=True)

        return solve_ppo
    if method == "bandit_fixed_w1.6":
        def solve_fixed(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_fixed_w_case(
                params=dict(params),
                mkw=dict(mkw),
                w=float(args.fixed_weight),
                sweeps_down=1,
                sweeps_up=1,
                solve_tol=float(args.tol),
                solve_max_cycles=int(args.max_cycles),
            )
            return as_feedback(native, include_controller=False)

        return solve_fixed

    if method == "bandit_default":
        def solve_default(params: Dict[str, Any]) -> Dict[str, Any]:
            native = solve_no_rl_case(
                params=dict(params),
                mkw=dict(mkw),
                solver_tol=float(args.tol),
                solver_max_iter=int(args.max_cycles),
                augment_params=augment_setup_params,
            )
            return as_feedback(native, include_controller=False)

        return solve_default
    raise ValueError(f"Unsupported bandit method: {method}")


def run(args: argparse.Namespace) -> None:
    _configure_paired_environment(args)
    with args.initial_bandit_state.open("rb") as handle:
        initial_branch = pickle.load(handle)["branch"]
    validate_expected_setup_action_count(initial_branch)
    bandit_cfg = default_test_final_bandit_config_from_env()
    branches = {method: copy.deepcopy(initial_branch) for method in BANDIT_METHODS}
    instances, instance_stream = _build_paired_instance_stream(args)
    controller, encoder = _td_controller(args)
    ppo_runner = _ppo_runner(args)

    action_count = int(branches[BASELINE_METHOD].policy.model.K)
    protocol = {
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "seed": int(args.seed),
        "methods": list(METHODS),
        "instances": int(len(instances)),
        "instance_stream": instance_stream,
        "grid": [int(args.grid_n)] * 3,
        "coefficient_range": [float(args.c_min), float(args.c_max)],
        "difconv_a": [0.0, 0.0, 0.0],
        "tol": float(args.tol),
        "max_cycles": int(args.max_cycles),
        "setup_bandit": {
            "common_warmup_instances": int(args.warmup_instances),
            "common_initial_state": str(args.initial_bandit_state),
            "action_space": str(args.setup_action_space),
            "actions": int(action_count),
            "feature_dim": int(branches[BASELINE_METHOD].policy.model.d_phi),
            "independent_branch_updates_after_warmup": True,
            "failure_protocol": "single_default_fallback",
        },
        "ppo": {
            "model": str(args.ppo_model),
            "training_result": (
                None if args.ppo_training_result is None else str(args.ppo_training_result)
            ),
            "training_mode": "offline Active Exp44; frozen during held-out stream",
            "observation": "cycle_action_setup",
            "action": "continuous residual, w += 0.02 * action",
            "reward": "Active Exp44 reward_mode=4 with precomputed default reference",
        },
        "td": {
            "training_mode": "continual online from held-out instance 1",
            "method_label": "true-online SARSA(lambda)",
            "state_mode": "setup_full",
            "feature_dim": int(encoder.feature_dim),
            "controller": controller.config.__dict__,
        },
        "randomized_method_order_per_instance": True,
        "timing_fields": [
            "setup_runtime",
            "native_solve_runtime",
            "controller_runtime",
            "setup_bandit_overhead",
            "end_to_end_runtime",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "config.json", protocol)
    print(json.dumps(_json_ready({"stage": "methods_2k_start", **protocol})), flush=True)

    previous_update = {method: 0.0 for method in BANDIT_METHODS}
    records: Dict[str, list[Dict[str, Any]]] = {method: [] for method in METHODS}
    rng = np.random.default_rng(int(args.seed + 91_003))
    progress_denom = max(1, len(instances) - 1)

    for index, (mkw, context) in enumerate(instances):
        method_order = list(METHODS)
        rng.shuffle(method_order)
        case_records: Dict[str, Dict[str, Any]] = {}
        for method in method_order:
            if method == "default_setup":
                native = solve_default_baseline_case(
                    mkw=dict(mkw),
                    solver_tol=float(args.tol),
                    solver_max_iter=int(args.max_cycles),
                    augment_params=augment_setup_params,
                )
                outcome = _report_online_outcome(
                    as_feedback(native, include_controller=False),
                    bandit_timing={},
                )
                case_records[method] = {
                    "instance_index": int(index),
                    "mkw": dict(mkw),
                    "context": np.asarray(context, dtype=float).tolist(),
                    "params": dict(DEFAULT_SETUP_PARAMS),
                    "arm_index": -1,
                    "fallback_used": 0,
                    "bandit_timing": {},
                    "outcome": outcome,
                }
                continue

            branch = branches[method]
            solver_fn = _solver_for_method(
                method,
                args=args,
                mkw=dict(mkw),
                case_progress=float(index) / float(progress_denom),
                controller=controller,
                encoder=encoder,
                ppo_runner=ppo_runner,
            )
            def fallback_solver(_params: Dict[str, Any]) -> Dict[str, Any]:
                fallback_native = solve_no_rl_case(
                    params=dict(DEFAULT_SETUP_PARAMS),
                    mkw=dict(mkw),
                    solver_tol=float(args.tol),
                    solver_max_iter=int(args.max_cycles),
                    augment_params=augment_setup_params,
                )
                return as_feedback(fallback_native, include_controller=False)
            params, native, timing, fallback_used, update_sec = run_bandit_step_test_final(
                policy=branch.policy,
                parameter_space=branch.parameter_space,
                problem_context=np.asarray(context, dtype=float),
                solver_fn=solver_fn,
                fallback_solver_fn=fallback_solver,
                prev_update_est=float(previous_update[method]),
            )
            previous_update[method] = float(update_sec)
            case_records[method] = {
                "instance_index": int(index),
                "mkw": dict(mkw),
                "context": np.asarray(context, dtype=float).tolist(),
                "params": dict(params),
                "arm_index": int(native.get("selected_arm_index", -1)),
                "fallback_used": int(fallback_used),
                "bandit_timing": dict(timing),
                "outcome": _report_online_outcome(native, bandit_timing=timing),
            }

        for method in METHODS:
            records[method].append(case_records[method])

        done = index + 1
        if done == 1000:
            controller.save(args.output_dir / "controller_episode_1000.npz")
        if done % max(1, int(args.progress_every)) == 0 or done == len(instances):
            partial_records = {method: method_records[:done] for method, method_records in records.items()}
            partial = {
                "protocol": protocol,
                "completed_instances": int(done),
                "methods": {
                    method: _method_stream_summary(method_records)
                    for method, method_records in partial_records.items()
                },
                "comparisons_vs_bandit_default": _window_comparisons(
                    partial_records,
                    seed=int(args.seed + done),
                ),
                "td": {
                    "steps": int(controller.steps),
                    "episodes": int(controller.episodes),
                    "epsilon": float(controller.epsilon),
                },
            }
            _write_json(args.output_dir / "progress.json", partial)
            controller.save(args.output_dir / "td_lambda_controller_partial.npz")
            print(
                json.dumps(
                    _json_ready(
                        {
                            "stage": "methods_2k_progress",
                            "done": int(done),
                            "total": int(len(instances)),
                            "td_steps": int(controller.steps),
                            "td_epsilon": float(controller.epsilon),
                            "end_to_end_vs_bandit_default": {
                                method: comparison["end_to_end_runtime"]
                                for method, comparison in partial[
                                    "comparisons_vs_bandit_default"
                                ].items()
                            },
                        }
                    )
                ),
                flush=True,
            )

    controller_path = args.output_dir / "td_lambda_controller.npz"
    controller.save(controller_path)
    final_audit_path = args.output_dir / f"controller_episode_{len(instances)}.npz"
    controller.save(final_audit_path)
    midpoint_path = args.output_dir / "controller_episode_1000.npz"
    with (args.output_dir / "online_bandit_states.pkl").open("wb") as handle:
        pickle.dump(branches, handle, protocol=pickle.HIGHEST_PROTOCOL)

    recovery_audit = {
        method: _validate_recovery_stream(
            method_records,
            expect_bandit_transaction=method in BANDIT_METHODS,
        )
        for method, method_records in records.items()
    }

    windows = {
        f"all_{len(instances)}": records,
        "first_1000": {method: method_records[:1000] for method, method_records in records.items()},
        "last_1000": {method: method_records[-1000:] for method, method_records in records.items()},
        "last_500": {method: method_records[-500:] for method, method_records in records.items()},
    }
    window_results = {
        window: {
            "methods": {
                method: _method_stream_summary(method_records)
                for method, method_records in window_records.items()
            },
            "comparisons_vs_bandit_default": _window_comparisons(
                window_records,
                seed=int(args.seed + 100_000 + offset * 1000),
            ),
        }
        for offset, (window, window_records) in enumerate(windows.items())
    }
    result = {
        "protocol": protocol,
        "recovery_audit": recovery_audit,
        "methods": {
            method: _method_stream_summary(method_records)
            for method, method_records in records.items()
        },
        "windows": window_results,
        "comparisons_vs_bandit_default": {
            window: value["comparisons_vs_bandit_default"]
            for window, value in window_results.items()
        },
        "cumulative_checkpoints": _cumulative_checkpoints(
            records,
            every=max(1, int(args.progress_every)),
        ),
        "action_diagnostics": {
            "bandit_td": _action_diagnostics(
                [record["outcome"] for record in records["bandit_td"]],
                controller.weights,
            ),
            "bandit_ppo": _continuous_action_diagnostics(
                [record["outcome"] for record in records["bandit_ppo"]]
            ),
        },
        "controller": {
            "checkpoint": str(controller_path),
            "episode_1000_checkpoint": (
                str(midpoint_path) if midpoint_path.exists() else None
            ),
            "episode_2000_checkpoint": (
                str(final_audit_path) if len(instances) == 2000 else None
            ),
            "final_episode_checkpoint": str(final_audit_path),
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
                    "stage": "methods_2k_done",
                    "output": args.output_dir / "result.json",
                    "methods": result["methods"],
                    "comparisons": result["comparisons_vs_bandit_default"][
                        f"all_{len(instances)}"
                    ],
                }
            )
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out 2K comparison of default, online bandit, fixed, PPO, and "
            "continual online solve control."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-bandit-state", type=Path, required=True)
    parser.add_argument("--warmup-instances", type=int, default=1000)
    parser.add_argument("--ppo-model", type=Path, required=True)
    parser.add_argument("--ppo-training-result", type=Path)
    parser.add_argument("--seed", type=int, default=39394016)
    parser.add_argument("--train-cases", type=int, default=2000)
    parser.add_argument("--instance-offset", type=int, default=1500)
    parser.add_argument(
        "--train-seed-groups",
        default="39394939,39400939;39406939,39412939",
    )
    parser.add_argument("--train-shuffle-seeds", default="39394016,39394022")
    parser.add_argument("--train-cases-per-seed", type=int, default=500)
    parser.add_argument("--train-group-take", type=int, default=1000)
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
    parser.add_argument("--setup-action-space", choices=("full_cartesian",), default="full_cartesian")
    parser.add_argument("--fixed-weight", type=float, default=1.6)
    parser.add_argument(
        "--weights",
        default="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument(
        "--td-algorithm",
        choices=("expected_sarsa", "sarsa", "true_online_sarsa"),
        default="true_online_sarsa",
    )
    parser.add_argument(
        "--exploration-mode",
        choices=("uniform", "least_visited"),
        default="uniform",
    )
    parser.add_argument("--td-decay-power", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--epsilon-start", type=float, default=0.3)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=float, default=20000.0)
    parser.add_argument("--initial-q-sec", type=float, default=0.0)
    parser.add_argument(
        "--force-default-first-action",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.td_algorithm == "true_online_sarsa" and args.td_decay_power != 0.0:
        parser.error("true_online_sarsa requires --td-decay-power 0")
    run(args)


if __name__ == "__main__":
    main()
