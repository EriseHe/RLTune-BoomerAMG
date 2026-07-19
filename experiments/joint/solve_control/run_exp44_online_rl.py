from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from amg_setup_gym_env import (
    DEFAULT_SETUP_PARAMS,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from SolvePhase.algorithms.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    OnlineFixedWeightIncumbent,
    SolveStateEncoder,
)
from online_td_experiment_common import (
    _action_diagnostics,
    _git_revision,
    _method_outcome,
    _paired,
    _run_trace_stream,
    _summarize,
    _write_json,
)
from run_mature_bandit_rl_pipeline import _build_train_trace, _frozen_trace, _warmup_bandit
from setup_aware_compare_common import (
    EXP44_MATRIX_GRID_N,
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
)


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def _runtime_with_overhead(row: Dict[str, Any]) -> float:
    return float(row["runtime"]) + float(row.get("infer_runtime", 0.0))


def _online_break_even(
    reference: Sequence[Dict[str, Any]],
    candidate: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    reference_cost = np.asarray([_runtime_with_overhead(row) for row in reference], dtype=float)
    candidate_cost = np.asarray([_runtime_with_overhead(row) for row in candidate], dtype=float)
    cumulative_savings = np.cumsum(reference_cost - candidate_cost)
    break_even = None
    for index in range(cumulative_savings.size):
        if np.all(cumulative_savings[index:] >= 0.0):
            break_even = int(index + 1)
            break
    return {
        "cases": int(candidate_cost.size),
        "reference_total_sec": float(np.sum(reference_cost)),
        "candidate_total_sec": float(np.sum(candidate_cost)),
        "final_savings_sec": float(cumulative_savings[-1]),
        "final_improvement_pct": float(
            100.0 * (np.sum(reference_cost) - np.sum(candidate_cost)) / np.sum(reference_cost)
        ),
        "persistent_break_even_case": break_even,
    }


def _configure_exp44_environment(args: argparse.Namespace) -> None:
    values = {
        "MATRIX_GRID_N": args.grid_n,
        "SETUP_PARAM_RESOLUTION": args.setup_param_resolution,
        "EXPECTED_SETUP_ACTION_COUNT": EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
        "C_MIN": args.c_min,
        "C_MAX": args.c_max,
        "DIFCONV_A": "0,0,0",
        "SETUP_BANDIT_METHOD": "linucbv4",
        "SETUP_TUNE_DIM": 7,
        "TUNE7_VARIANT": "categorical",
        "SOLVER_TOL": args.tol,
        "SOLVER_MAX_ITER": args.max_cycles,
        "SOLVE_TOL": args.tol,
        "SOLVE_MAX_CYCLES": args.max_cycles,
        "WARMUP_SEED": 39393939,
        "WARMUP_CASES": 1500,
        "RL_TRAIN_SEEDS": args.train_seeds,
        "RL_TRAIN_CASES": args.train_cases_per_seed,
        "TRAIN_TRACE_SHUFFLE": 1,
        "TRAIN_TRACE_SHUFFLE_SEED": getattr(args, "trace_shuffle_seed", args.seed),
        "RL_SEED": args.seed,
        "PROGRESS_EVERY": args.progress_every,
    }
    for name, value in values.items():
        os.environ[name] = str(value)
    if args.bandit_state.exists():
        os.environ["USE_SAVED_BANDIT"] = "1"
        os.environ["BANDIT_STATE_PATH"] = str(args.bandit_state)
        os.environ.pop("SAVE_BANDIT_STATE_PATH", None)
    else:
        os.environ.pop("USE_SAVED_BANDIT", None)
        os.environ["SAVE_BANDIT_STATE_PATH"] = str(args.bandit_state)


def make_controller(
    args: argparse.Namespace,
    *,
    anchor_weight: float | None = None,
) -> tuple[ExpectedSarsaLambda, SolveStateEncoder, ExpectedSarsaLambdaConfig]:
    adaptive_cycles = None if int(args.adaptive_cycles) < 0 else int(args.adaptive_cycles)
    selected_anchor = float(args.anchor_weight) if anchor_weight is None else float(anchor_weight)
    config = ExpectedSarsaLambdaConfig(
        weights=_parse_values(args.weights, float),
        anchor_weight=selected_anchor,
        alpha=float(args.alpha),
        td_decay_power=float(args.td_decay_power),
        gamma=float(args.gamma),
        trace_lambda=float(args.trace_lambda),
        epsilon_start=float(args.epsilon_start),
        epsilon_final=float(args.epsilon_final),
        epsilon_decay_steps=float(args.epsilon_decay_steps),
        potential_scale_sec=float(args.potential_scale_sec),
        failure_penalty_sec=float(args.failure_penalty_sec),
        initial_q_sec=float(args.initial_q_sec),
        monte_carlo_alpha=float(args.monte_carlo_alpha),
        monte_carlo_decay_power=float(args.monte_carlo_decay_power),
        adaptive_cycles=adaptive_cycles,
        exploration_mode=str(args.exploration_mode),
        action_rbf_sigma=float(args.action_rbf_sigma),
        td_algorithm=str(args.td_algorithm),
        force_default_first_action=bool(
            getattr(args, "force_default_first_action", False)
        ),
    )
    setup_obs_encoder = None
    if str(args.state_mode) == "setup_full":
        parameter_spec, _fixed_params = build_setup_parameter_spec(
            tune_dim=7,
            tune7_variant="categorical",
        )
        setup_obs_encoder = SetupObsEncoder(
            parameter_spec,
            dict(DEFAULT_SETUP_PARAMS),
            tuple(parameter_spec.parameter_names),
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
    return controller, encoder, config


# Keep the existing private name for callers created before the shared builder
# became part of the Exp44 tuning/final-run protocol.
_make_controller = make_controller


def _run_online_incumbent_calibration(
    *,
    trace: Sequence[tuple[Dict[str, Any], Dict[str, Any]]],
    episodes: int,
    weights: Sequence[float],
    initial_weight: float,
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    tol: float,
    max_cycles: int,
    failure_penalty_sec: float,
    seed: int,
    progress_every: int,
) -> tuple[OnlineFixedWeightIncumbent, list[Dict[str, Any]]]:
    if episodes < 0 or episodes > len(trace):
        raise ValueError("calibration episodes must fit within the training trace")
    estimator = OnlineFixedWeightIncumbent(
        weights=weights,
        initial_weight=initial_weight,
        seed=seed,
    )
    rows: list[Dict[str, Any]] = []
    for index, (mkw, params) in enumerate(trace[:episodes]):
        action_index, weight = estimator.select_action()
        outcome = _method_outcome(
            f"fixed_w{weight:g}",
            mkw=mkw,
            params=params,
            controller=controller,
            encoder=encoder,
            tol=tol,
            max_cycles=max_cycles,
            learn=False,
            explore=False,
        )
        observed_cost = float(outcome["solve_runtime"])
        if bool(outcome.get("failed", False)):
            observed_cost += float(failure_penalty_sec)
        estimator.update(action_index, observed_cost)
        row = dict(outcome)
        row.update(
            {
                "calibration_action_index": int(action_index),
                "calibration_weight": float(weight),
                "observed_cost_sec": float(observed_cost),
                "incumbent_after_weight": estimator.incumbent_weight,
            }
        )
        rows.append(row)
        done = index + 1
        if done % max(1, int(progress_every)) == 0 or done == episodes:
            print(
                json.dumps(
                    {
                        "stage": "exp44_black_box_calibration",
                        "done": done,
                        "total": episodes,
                        "incumbent_weight": estimator.incumbent_weight,
                    }
                ),
                flush=True,
            )
    return estimator, rows


def _average_repeated_rows(
    repeats: Sequence[Sequence[Dict[str, Any]]],
) -> list[Dict[str, Any]]:
    if not repeats:
        return []
    cases = len(repeats[0])
    if any(len(rows) != cases for rows in repeats):
        raise ValueError("Repeated evaluation rows must have matching case counts")
    numeric_fields = (
        "runtime",
        "setup_runtime",
        "solve_runtime",
        "infer_runtime",
        "feature_runtime",
        "decision_runtime",
        "update_runtime",
    )
    averaged: list[Dict[str, Any]] = []
    for case_index in range(cases):
        case_rows = [rows[case_index] for rows in repeats]
        row = {
            field: float(np.mean([float(case.get(field, 0.0)) for case in case_rows]))
            for field in numeric_fields
        }
        row["failed"] = bool(any(bool(case.get("failed", False)) for case in case_rows))
        averaged.append(row)
    return averaged


def _evaluate(
    *,
    branch: Any,
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    eval_seeds: Sequence[int],
    eval_cases: int,
    fixed_weights: Sequence[float],
    args: argparse.Namespace,
) -> tuple[Dict[str, Any], Dict[str, list[Dict[str, Any]]]]:
    methods = tuple(["default_solve", *[f"fixed_w{weight:g}" for weight in fixed_weights], "td_lambda"])
    per_seed: Dict[str, Any] = {}
    combined_rows: Dict[str, list[Dict[str, Any]]] = {method: [] for method in methods}
    comparison_rows: Dict[str, list[Dict[str, Any]]] = {method: [] for method in methods}
    for offset, seed in enumerate(eval_seeds):
        trace = _frozen_trace(branch, cases=int(eval_cases), seed=int(seed))
        repeated_rows = []
        for repeat in range(int(args.eval_repeats)):
            rows = _run_trace_stream(
                trace=trace,
                controller=controller,
                encoder=encoder,
                tol=float(args.tol),
                max_cycles=int(args.max_cycles),
                methods=methods,
                learn=False,
                explore=False,
                seed=int(args.seed + 20_000 + 100 * offset + repeat),
                progress_stage=f"exp44_eval_seed_{seed}_repeat_{repeat + 1}",
                progress_every=int(args.progress_every),
            )
            repeated_rows.append(rows)
            for method, method_rows in rows.items():
                combined_rows[method].extend(method_rows)
        averaged_rows = {
            method: _average_repeated_rows([rows[method] for rows in repeated_rows])
            for method in methods
        }
        for method, method_rows in averaged_rows.items():
            comparison_rows[method].extend(method_rows)
        per_seed[str(seed)] = {
            "methods": {
                method: _summarize(
                    [row for repeat_rows in repeated_rows for row in repeat_rows[method]]
                )
                for method in methods
            },
            "vs_default": {
                method: _paired(
                    averaged_rows["default_solve"],
                    method_rows,
                    seed=int(seed + offset),
                )
                for method, method_rows in averaged_rows.items()
                if method != "default_solve"
            },
        }

    combined = {method: _summarize(rows) for method, rows in combined_rows.items()}
    fixed_methods = [method for method in methods if method.startswith("fixed_w")]
    best_fixed_method = min(
        fixed_methods,
        key=lambda method: combined[method]["mean_runtime_with_overhead_sec"],
    )
    evaluation = {
        "per_seed": per_seed,
        "combined": combined,
        "best_fixed_method": best_fixed_method,
        "comparisons": {
            "online_rl_vs_default": _paired(
                comparison_rows["default_solve"],
                comparison_rows["td_lambda"],
                seed=int(args.seed + 30_001),
            ),
            "online_rl_vs_best_fixed": _paired(
                comparison_rows[best_fixed_method],
                comparison_rows["td_lambda"],
                seed=int(args.seed + 30_002),
            ),
        },
        "repeats": int(args.eval_repeats),
        "action_diagnostics": _action_diagnostics(
            combined_rows["td_lambda"],
            controller.weights,
        ),
    }
    return evaluation, combined_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Budget-matched online per-cycle RL under the retained Exp44 protocol."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bandit-state",
        type=Path,
        default=Path(
            "results/joint/mature_tune7_ppo_repro_20260423/run_logs/"
            "mature40_tune7_bandit_state_case2.pkl"
        ),
    )
    parser.add_argument("--seed", type=int, default=39394016)
    parser.add_argument("--trace-shuffle-seed", type=int, default=39394016)
    parser.add_argument("--train-seeds", default="39396939,39402939,39408939")
    parser.add_argument("--train-cases-per-seed", type=int, default=500)
    parser.add_argument("--eval-seeds", default="39414939,39420939,39426939")
    parser.add_argument("--eval-cases", type=int, default=500)
    parser.add_argument("--eval-repeats", type=int, default=1)
    parser.add_argument("--transition-budget", type=int, default=2560)
    parser.add_argument("--episode-budget", type=int, default=0)
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
    parser.add_argument(
        "--weights",
        default="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )
    parser.add_argument("--fixed-eval-weights", default="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0")
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--online-calibration-episodes", type=int, default=0)
    parser.add_argument("--adaptive-cycles", type=int, default=2)
    parser.add_argument(
        "--state-mode",
        choices=("full", "setup_full", "cycle_tabular"),
        default="cycle_tabular",
    )
    parser.add_argument("--action-rbf-sigma", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(
        "--td-algorithm",
        choices=("expected_sarsa", "sarsa", "true_online_sarsa"),
        default="expected_sarsa",
    )
    parser.add_argument("--td-decay-power", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--monte-carlo-alpha", type=float, default=1.0)
    parser.add_argument("--monte-carlo-decay-power", type=float, default=1.0)
    parser.add_argument(
        "--monte-carlo-control-variate",
        choices=("none", "anchor"),
        default="none",
    )
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-final", type=float, default=1.0)
    parser.add_argument("--epsilon-decay-steps", type=float, default=2560.0)
    parser.add_argument("--exploration-mode", choices=("uniform", "least_visited"), default="least_visited")
    parser.add_argument("--potential-scale-sec", type=float, default=0.0)
    parser.add_argument("--failure-penalty-sec", type=float, default=0.05)
    parser.add_argument("--initial-q-sec", type=float, default=0.02)
    parser.add_argument(
        "--force-default-first-action",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--controller-checkpoint", type=Path)
    parser.add_argument("--skip-training", action="store_true")
    args = parser.parse_args()

    if args.skip_training and args.controller_checkpoint is None:
        parser.error("--skip-training requires --controller-checkpoint")
    if args.online_calibration_episodes < 0:
        parser.error("--online-calibration-episodes must be non-negative")
    if args.online_calibration_episodes and args.skip_training:
        parser.error("online calibration cannot be combined with --skip-training")
    if args.online_calibration_episodes and args.controller_checkpoint is not None:
        parser.error("online calibration cannot be combined with a pretrained controller")
    if args.online_calibration_episodes and args.monte_carlo_control_variate != "none":
        parser.error("black-box online calibration cannot use an oracle control variate")
    if (
        args.episode_budget > 0
        and args.online_calibration_episodes >= args.episode_budget
    ):
        parser.error("episode budget must leave at least one episode for per-cycle RL")
    if args.td_algorithm == "true_online_sarsa" and args.td_decay_power != 0.0:
        parser.error("true_online_sarsa requires --td-decay-power 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.bandit_state = args.bandit_state.resolve()
    _configure_exp44_environment(args)
    controller, encoder, config = _make_controller(args)
    checkpoint_metadata = None
    if args.controller_checkpoint is not None:
        args.controller_checkpoint = args.controller_checkpoint.resolve()
        checkpoint_metadata = controller.load(args.controller_checkpoint)
    protocol = {
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "grid": [int(args.grid_n)] * 3,
        "coefficient_range": [float(args.c_min), float(args.c_max)],
        "tol": float(args.tol),
        "max_cycles": int(args.max_cycles),
        "bandit_state": str(args.bandit_state),
        "train_seeds": list(_parse_values(args.train_seeds, int)),
        "train_cases_per_seed": int(args.train_cases_per_seed),
        "trace_shuffle_seed": int(args.trace_shuffle_seed),
        "eval_seeds": list(_parse_values(args.eval_seeds, int)),
        "eval_cases_per_seed": int(args.eval_cases),
        "eval_repeats": int(args.eval_repeats),
        "nominal_transition_budget": (
            int(args.transition_budget) if int(args.transition_budget) > 0 else None
        ),
        "episode_budget": int(args.episode_budget) if int(args.episode_budget) > 0 else None,
        "online_calibration_episodes": int(args.online_calibration_episodes),
        "initial_anchor_weight": float(args.anchor_weight),
        "ppo_actual_reference_budget": 2560,
        "state_mode": str(args.state_mode),
        "feature_dim": int(encoder.feature_dim),
        "controller": asdict(config),
        "monte_carlo_control_variate": str(args.monte_carlo_control_variate),
        "pretrained_checkpoint": (
            str(args.controller_checkpoint) if args.controller_checkpoint is not None else None
        ),
    }
    _write_json(args.output_dir / "config.json", protocol)
    print(json.dumps({"stage": "exp44_online_rl_start", **protocol}), flush=True)

    branch = _warmup_bandit()
    anchor_method = f"fixed_w{float(config.anchor_weight):g}"
    if args.skip_training:
        train_meta = None
        training_result: Dict[str, Any] = {
            "skipped": True,
            "source_checkpoint": str(args.controller_checkpoint),
            "checkpoint_metadata": checkpoint_metadata,
        }
        checkpoint_path = args.controller_checkpoint
    else:
        train_trace, train_meta = _build_train_trace(branch)
        calibration_rows: list[Dict[str, Any]] = []
        calibration_result = None
        calibration_episodes = int(args.online_calibration_episodes)
        if calibration_episodes:
            estimator, calibration_rows = _run_online_incumbent_calibration(
                trace=train_trace,
                episodes=calibration_episodes,
                weights=_parse_values(args.weights, float),
                initial_weight=float(args.anchor_weight),
                controller=controller,
                encoder=encoder,
                tol=float(args.tol),
                max_cycles=int(args.max_cycles),
                failure_penalty_sec=float(args.failure_penalty_sec),
                seed=int(args.seed + 701),
                progress_every=int(args.progress_every),
            )
            learned_anchor = estimator.incumbent_weight
            controller, encoder, config = _make_controller(
                args,
                anchor_weight=learned_anchor,
            )
            anchor_method = f"fixed_w{learned_anchor:g}"
            calibration_result = {
                "information_model": "one_observed_fixed-weight trajectory per instance",
                "counterfactual_rollouts": 0,
                "episodes": int(len(calibration_rows)),
                "transitions": int(sum(int(row["iterations"]) for row in calibration_rows)),
                "trajectory_total_sec": float(
                    sum(_runtime_with_overhead(row) for row in calibration_rows)
                ),
                "methods": _summarize(calibration_rows),
                "estimator": estimator.summary(),
            }
        remaining_trace = train_trace[calibration_episodes:]
        remaining_episode_budget = (
            int(args.episode_budget) - calibration_episodes
            if int(args.episode_budget) > 0
            else None
        )
        calibration_transitions = int(sum(int(row["iterations"]) for row in calibration_rows))
        remaining_transition_budget = (
            max(1, int(args.transition_budget) - calibration_transitions)
            if int(args.transition_budget) > 0
            else None
        )
        training_methods = (
            (anchor_method, "td_lambda")
            if args.monte_carlo_control_variate == "anchor"
            else ("td_lambda",)
        )
        training = _run_trace_stream(
            trace=remaining_trace,
            controller=controller,
            encoder=encoder,
            tol=float(args.tol),
            max_cycles=int(args.max_cycles),
            methods=training_methods,
            learn=True,
            explore=True,
            seed=int(args.seed + 1),
            progress_stage="exp44_online_rl_train",
            progress_every=int(args.progress_every),
            transition_budget=remaining_transition_budget,
            episode_budget=remaining_episode_budget,
            monte_carlo_baseline_method=(
                anchor_method if args.monte_carlo_control_variate == "anchor" else None
            ),
        )
        checkpoint_path = args.output_dir / "online_rl_controller.npz"
        controller.save(checkpoint_path)
        rl_trajectory_total = float(
            sum(_runtime_with_overhead(row) for row in training["td_lambda"])
        )
        training_result = {
            "information_model": (
                "paired oracle reference"
                if args.monte_carlo_control_variate == "anchor"
                else "one trajectory per instance"
            ),
            "counterfactual_rollouts": (
                int(len(training[anchor_method]))
                if args.monte_carlo_control_variate == "anchor"
                else 0
            ),
            "processed_cases": int(calibration_episodes + len(training["td_lambda"])),
            "actual_transitions": int(calibration_transitions + controller.steps),
            "transition_budget_overshoot": (
                int(calibration_transitions + controller.steps - args.transition_budget)
                if int(args.transition_budget) > 0
                else None
            ),
            "episodes": int(calibration_episodes + controller.episodes),
            "per_cycle_rl_episodes": int(controller.episodes),
            "calibration": calibration_result,
            "methods": {method: _summarize(rows) for method, rows in training.items()},
            "online_trajectory_total_sec": float(
                sum(_runtime_with_overhead(row) for row in calibration_rows)
                + rl_trajectory_total
            ),
            "reference_rollout_total_sec": (
                float(sum(_runtime_with_overhead(row) for row in training[anchor_method]))
                if args.monte_carlo_control_variate == "anchor"
                else 0.0
            ),
            "online_break_even_vs_anchor": (
                _online_break_even(training[anchor_method], training["td_lambda"])
                if args.monte_carlo_control_variate == "anchor"
                else None
            ),
            "action_diagnostics": _action_diagnostics(training["td_lambda"], controller.weights),
        }

        protocol["controller"] = asdict(config)
        protocol["learned_anchor_weight"] = float(config.anchor_weight)
        _write_json(args.output_dir / "config.json", protocol)

    evaluation, _combined_rows = _evaluate(
        branch=branch,
        controller=controller,
        encoder=encoder,
        eval_seeds=_parse_values(args.eval_seeds, int),
        eval_cases=int(args.eval_cases),
        fixed_weights=_parse_values(args.fixed_eval_weights, float),
        args=args,
    )
    result = {
        "stage": "final",
        "protocol": protocol,
        "train_trace": train_meta,
        "training": training_result,
        "evaluation": evaluation,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(args.output_dir / "result.json", result)
    print(json.dumps({"stage": "exp44_online_rl_done", "result": str(args.output_dir / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
