from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from amg_setup_gym_env import (
    DEFAULT_SETUP_PARAMS,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from online_td_experiment_common import _git_revision, _json_ready, _write_json
from SolvePhase.algorithms.sarsa import (
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    run_td_episode,
)
from run_online_methods_2k import _paired_metric, make_exp44_ppo_runner
from run_true_online_sarsa_tuning import (
    _build_traces,
    _configure_trace_environment,
)
from sarsa_exploration_policies import (
    ExplorationStudyController,
    ExplorationStudySpec,
)
from setup_aware_compare_common import (
    augment_setup_params,
    classify_rl_failure,
    solve_fixed_w_case,
    solve_setup_aware_rl_case,
)


BEHAVIOR_MODES = ("uniform", "local", "uncertainty_lcb")
ACTION_MODES = ("absolute", "residual")
REFERENCE_METHODS = ("fixed_w1.6", "ppo")
STUDY_METHODS = tuple(
    f"{action_mode}_{behavior_mode}"
    for action_mode in ACTION_MODES
    for behavior_mode in BEHAVIOR_MODES
)


def study_specs(*, initial_weight: float = 1.0) -> tuple[ExplorationStudySpec, ...]:
    return tuple(
        ExplorationStudySpec(
            action_mode=action_mode,
            behavior_mode=behavior_mode,
            initial_weight=float(initial_weight),
        )
        for action_mode in ACTION_MODES
        for behavior_mode in BEHAVIOR_MODES
    )


def _make_encoder(args: argparse.Namespace) -> SolveStateEncoder:
    parameter_spec, _fixed_params = build_setup_parameter_spec(
        tune_dim=7,
        tune7_variant="categorical",
    )
    setup_encoder = SetupObsEncoder(
        parameter_spec,
        dict(DEFAULT_SETUP_PARAMS),
        tuple(parameter_spec.parameter_names),
    )
    return SolveStateEncoder(
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        mode="setup_full",
        setup_obs_encoder=setup_encoder,
    )


def _make_controller(
    args: argparse.Namespace,
    spec: ExplorationStudySpec,
) -> tuple[ExplorationStudyController, SolveStateEncoder]:
    encoder = _make_encoder(args)
    if spec.action_mode == "absolute":
        actions = tuple(np.round(np.arange(1.0, 2.0 + 0.05, 0.1), 1))
        anchor = float(spec.initial_weight)
    else:
        actions = (-0.02, -0.01, 0.0, 0.01, 0.02)
        anchor = 0.0
    config = ExpectedSarsaLambdaConfig(
        weights=tuple(float(value) for value in actions),
        anchor_weight=anchor,
        alpha=float(args.alpha),
        td_decay_power=0.0,
        gamma=1.0,
        trace_lambda=float(args.trace_lambda),
        epsilon_start=float(args.epsilon_start),
        epsilon_final=float(args.epsilon_final),
        epsilon_decay_steps=float(args.epsilon_decay_steps),
        potential_scale_sec=float(args.potential_scale_sec),
        failure_penalty_sec=float(args.failure_penalty_sec),
        initial_q_sec=0.0,
        monte_carlo_alpha=0.0,
        adaptive_cycles=None,
        # The diagnosis subclass implements local and LCB behavior. The base
        # value is retained so the shared true-online controller validates.
        exploration_mode="uniform",
        action_rbf_sigma=0.0,
        td_algorithm="true_online_sarsa",
    )
    controller = ExplorationStudyController(
        study_spec=spec,
        feature_dim=encoder.feature_dim,
        config=config,
        seed=int(args.controller_seed),
        initial_parameters=encoder.constant_value_parameters(0.0),
    )
    return controller, encoder


def _setup_hash(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_outcome(outcome: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(outcome)
    normalized["infer_runtime"] = float(normalized.get("infer_runtime", 0.0))
    normalized["feature_runtime"] = float(normalized.get("feature_runtime", 0.0))
    normalized["decision_runtime"] = float(normalized.get("decision_runtime", 0.0))
    normalized["update_runtime"] = float(normalized.get("update_runtime", 0.0))
    normalized["native_total_runtime"] = float(normalized["runtime"])
    normalized["end_to_end_runtime"] = float(
        normalized["runtime"] + normalized["infer_runtime"]
    )
    return normalized


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    outcomes = [row["outcome"] for row in rows]
    return {
        "cases": int(len(rows)),
        "failed_count": int(sum(bool(outcome.get("failed", False)) for outcome in outcomes)),
        "mean_setup_runtime_sec": float(
            np.mean([float(outcome["setup_runtime"]) for outcome in outcomes])
        ),
        "mean_native_solve_runtime_sec": float(
            np.mean([float(outcome["solve_runtime"]) for outcome in outcomes])
        ),
        "mean_native_total_runtime_sec": float(
            np.mean([float(outcome["runtime"]) for outcome in outcomes])
        ),
        "mean_controller_runtime_sec": float(
            np.mean([float(outcome.get("infer_runtime", 0.0)) for outcome in outcomes])
        ),
        "mean_end_to_end_runtime_sec": float(
            np.mean([float(outcome["end_to_end_runtime"]) for outcome in outcomes])
        ),
        "mean_iterations": float(
            np.mean([float(outcome["iterations"]) for outcome in outcomes])
        ),
    }


def _comparison(
    candidate: Sequence[Dict[str, Any]],
    reference: Sequence[Dict[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    if len(candidate) != len(reference):
        raise ValueError("Paired method streams must have the same length")
    accessors = {
        "setup_runtime": lambda row: float(row["outcome"]["setup_runtime"]),
        "native_solve_runtime": lambda row: float(row["outcome"]["solve_runtime"]),
        "native_total_runtime": lambda row: float(row["outcome"]["runtime"]),
        "controller_runtime": lambda row: float(row["outcome"].get("infer_runtime", 0.0)),
        "end_to_end_runtime": lambda row: float(row["outcome"]["end_to_end_runtime"]),
    }
    return {
        name: _paired_metric(
            np.asarray([accessor(row) for row in reference], dtype=float),
            np.asarray([accessor(row) for row in candidate], dtype=float),
            seed=int(seed + offset),
        )
        for offset, (name, accessor) in enumerate(accessors.items())
    }


def _action_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    physical_weights: list[float] = []
    action_values: list[float] = []
    explored: list[bool] = []
    uncertainties: list[float] = []
    by_cycle: Dict[int, list[float]] = {}
    for row in rows:
        outcome = row["outcome"]
        for cycle, weight in enumerate(outcome.get("cycle_actions", [])):
            physical_weights.append(float(weight))
            by_cycle.setdefault(int(cycle), []).append(float(weight))
        action_values.extend(float(value) for value in outcome.get("cycle_action_values", []))
        explored.extend(bool(value) for value in outcome.get("cycle_explored", []))
        uncertainties.extend(
            float(value)
            for value in outcome.get("cycle_selected_uncertainties", [])
        )
    return {
        "decisions": int(len(physical_weights)),
        "mean_physical_weight": float(np.mean(physical_weights)),
        "mean_action_value": float(np.mean(action_values)),
        "explored_rate": float(np.mean(explored)) if explored else 0.0,
        "mean_selected_uncertainty_sec": (
            float(np.mean(uncertainties)) if uncertainties else 0.0
        ),
        "by_cycle": {
            str(cycle): {
                "count": int(len(values)),
                "mean": float(np.mean(values)),
                "p10": float(np.percentile(values, 10)),
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
            }
            for cycle, values in sorted(by_cycle.items())
        },
    }


def _window_summary(
    records: Dict[str, list[Dict[str, Any]]],
    *,
    start: int,
    stop: int,
    seed: int,
) -> Dict[str, Any]:
    sliced = {method: rows[start:stop] for method, rows in records.items()}
    return {
        "start": int(start),
        "stop": int(stop),
        "methods": {method: _summarize(rows) for method, rows in sliced.items()},
        "comparisons_vs_fixed_w1.6": {
            method: _comparison(
                rows,
                sliced["fixed_w1.6"],
                seed=int(seed + index * 101),
            )
            for index, (method, rows) in enumerate(sliced.items())
            if method != "fixed_w1.6"
        },
        "comparisons_vs_ppo": {
            method: _comparison(
                rows,
                sliced["ppo"],
                seed=int(seed + 50_000 + index * 101),
            )
            for index, (method, rows) in enumerate(sliced.items())
            if method != "ppo"
        },
        "actions": {
            method: _action_summary(rows)
            for method, rows in sliced.items()
            if method not in REFERENCE_METHODS
        },
    }


def _write_json_line(handle: Any, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(_json_ready(row), separators=(",", ":")))
    handle.write("\n")


def _write_performance_csv(
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
) -> None:
    fields = (
        "case_index",
        "method",
        "setup_runtime_sec",
        "native_solve_runtime_sec",
        "native_total_runtime_sec",
        "controller_runtime_sec",
        "end_to_end_runtime_sec",
        "iterations",
        "failed",
        "epsilon",
        "mean_physical_weight",
        "first_weight",
        "last_weight",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, rows in records.items():
            for row in rows:
                outcome = row["outcome"]
                actions = [float(value) for value in outcome.get("cycle_actions", [])]
                writer.writerow(
                    {
                        "case_index": int(row["case_index"]),
                        "method": method,
                        "setup_runtime_sec": float(outcome["setup_runtime"]),
                        "native_solve_runtime_sec": float(outcome["solve_runtime"]),
                        "native_total_runtime_sec": float(outcome["runtime"]),
                        "controller_runtime_sec": float(outcome.get("infer_runtime", 0.0)),
                        "end_to_end_runtime_sec": float(outcome["end_to_end_runtime"]),
                        "iterations": int(outcome["iterations"]),
                        "failed": bool(outcome.get("failed", False)),
                        "epsilon": float(outcome.get("epsilon", 0.0)),
                        "mean_physical_weight": (
                            float(np.mean(actions)) if actions else float("nan")
                        ),
                        "first_weight": actions[0] if actions else float("nan"),
                        "last_weight": actions[-1] if actions else float("nan"),
                    }
                )


def _write_action_csv(
    path: Path,
    records: Dict[str, list[Dict[str, Any]]],
) -> None:
    fields = (
        "case_index",
        "method",
        "cycle",
        "physical_weight",
        "action_value",
        "residual",
        "cycle_time_sec",
        "explored",
        "epsilon",
        "selected_uncertainty_sec",
        "uncertainty_td_scale_sec",
        "selection_score_sec",
        "forced_default_first_action",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, rows in records.items():
            if method in REFERENCE_METHODS:
                continue
            for row in rows:
                outcome = row["outcome"]
                weights = outcome.get("cycle_actions", [])
                action_values = outcome.get("cycle_action_values", [])
                residuals = outcome.get("cycle_residuals", [])
                cycle_times = outcome.get("cycle_times", [])
                explored = outcome.get("cycle_explored", [])
                epsilons = outcome.get("cycle_epsilons", [])
                uncertainties = outcome.get("cycle_selected_uncertainties", [])
                td_scales = outcome.get("cycle_uncertainty_td_scales", [])
                scores = outcome.get("cycle_selection_scores", [])
                forced_actions = outcome.get("cycle_forced_default_actions", [])
                for cycle, weight in enumerate(weights):
                    writer.writerow(
                        {
                            "case_index": int(row["case_index"]),
                            "method": method,
                            "cycle": int(cycle),
                            "physical_weight": float(weight),
                            "action_value": float(action_values[cycle]),
                            "residual": float(residuals[cycle]),
                            "cycle_time_sec": float(cycle_times[cycle]),
                            "explored": bool(explored[cycle]),
                            "epsilon": float(epsilons[cycle]),
                            "selected_uncertainty_sec": float(uncertainties[cycle]),
                            "uncertainty_td_scale_sec": float(td_scales[cycle]),
                            "selection_score_sec": float(scores[cycle]),
                            "forced_default_first_action": bool(
                                forced_actions[cycle]
                            ),
                        }
                    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if int(args.train_take) <= 0:
        raise ValueError("train_take must be positive")
    if not args.bandit_state.exists():
        raise FileNotFoundError(args.bandit_state)
    if not args.ppo_model.exists():
        raise FileNotFoundError(args.ppo_model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = args.output_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    _configure_trace_environment(args)
    with args.bandit_state.open("rb") as handle:
        branch = pickle.load(handle)["branch"]
    train_trace, _unused_eval, trace_manifest = _build_traces(args, branch)

    requested_variants = tuple(
        value.strip() for value in str(args.variants).split(",") if value.strip()
    )
    unknown_variants = set(requested_variants) - set(STUDY_METHODS)
    if unknown_variants:
        raise ValueError(f"Unknown study variants: {sorted(unknown_variants)}")
    initial_weight = 1.0 if args.initial_weight is None else float(args.initial_weight)
    specs = tuple(
        spec
        for spec in study_specs(initial_weight=initial_weight)
        if spec.name in set(requested_variants)
    )
    if not specs:
        raise ValueError("At least one study variant is required")
    controllers: Dict[str, ExplorationStudyController] = {}
    encoders: Dict[str, SolveStateEncoder] = {}
    for spec in specs:
        controller, encoder = _make_controller(args, spec)
        controllers[spec.name] = controller
        encoders[spec.name] = encoder
    ppo_args = SimpleNamespace(
        ppo_model=args.ppo_model,
        output_dir=args.output_dir,
        grid_n=int(args.grid_n),
        c_min=float(args.c_min),
        c_max=float(args.c_max),
        max_cycles=int(args.max_cycles),
        tol=float(args.tol),
    )
    ppo_runner = make_exp44_ppo_runner(ppo_args)
    methods = (*REFERENCE_METHODS, *[spec.name for spec in specs])
    protocol = {
        "git_revision": _git_revision(),
        "purpose": "direct 1000-case online trajectory and performance comparison",
        "information_model": (
            "one common mature-bandit frozen (instance, setup) trace; one trajectory "
            "per method; no learner receives reference feedback"
        ),
        "cases": int(len(train_trace)),
        "trace": trace_manifest["train"],
        "warmup_bandit_state": str(args.bandit_state),
        "ppo_model": str(args.ppo_model),
        "randomized_method_order_per_case": True,
        "controller_seed": int(args.controller_seed),
        "method_order_seed": int(args.method_order_seed),
        "initial_environment_weight_source": (
            "prepared_solver" if args.initial_weight is None else "explicit_override"
        ),
        "initial_environment_weight_override": (
            None if args.initial_weight is None else float(args.initial_weight)
        ),
        "initialization": (
            "only the first online decision (instance 0, cycle 0) is forced to "
            "the default physical weight; every later cycle and episode is policy-controlled"
        ),
        "common_sarsa": {
            "algorithm": "true-online SARSA(lambda)",
            "alpha": float(args.alpha),
            "lambda": float(args.trace_lambda),
            "gamma": 1.0,
            "state": "setup_full",
            "feature_dim": int(next(iter(encoders.values())).feature_dim),
            "epsilon": {
                "start": float(args.epsilon_start),
                "final": float(args.epsilon_final),
                "decay_steps": float(args.epsilon_decay_steps),
            },
            "potential_scale_sec": float(args.potential_scale_sec),
            "failure_penalty_sec": float(args.failure_penalty_sec),
        },
        "variants": {spec.name: spec.__dict__ for spec in specs},
        "references": {
            "fixed_w1.6": "same case and setup, fixed relaxation weight 1.6",
            "ppo": "frozen Active Exp44 recurrent PPO",
        },
    }
    _write_json(args.output_dir / "config.json", protocol)
    _write_json(args.output_dir / "trace_manifest.json", trace_manifest)

    records: Dict[str, list[Dict[str, Any]]] = {method: [] for method in methods}
    handles = {
        method: (trajectories_dir / f"{method}.jsonl").open("w", encoding="utf-8")
        for method in methods
    }
    order_rng = np.random.default_rng(int(args.method_order_seed))
    progress_denom = max(1, len(train_trace) - 1)
    try:
        for case_index, (mkw, params) in enumerate(train_trace):
            order = [methods[index] for index in order_rng.permutation(len(methods))]
            for method in order:
                if method == "fixed_w1.6":
                    outcome = solve_fixed_w_case(
                        params=dict(params),
                        mkw=dict(mkw),
                        w=1.6,
                        sweeps_down=1,
                        sweeps_up=1,
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                    )
                elif method == "ppo":
                    outcome = solve_setup_aware_rl_case(
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
                        case_progress=float(case_index) / float(progress_denom),
                    )
                else:
                    outcome = run_td_episode(
                        mkw=dict(mkw),
                        params=dict(params),
                        controller=controllers[method],
                        encoder=encoders[method],
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                        learn=True,
                        explore=True,
                        record_action_metadata=True,
                        initial_environment_weight_override=args.initial_weight,
                    )
                row = {
                    "case_index": int(case_index),
                    "mkw": dict(mkw),
                    "params": dict(params),
                    "setup_hash": _setup_hash(params),
                    "execution_order": list(order),
                    "outcome": _normalize_outcome(outcome),
                }
                records[method].append(row)
                _write_json_line(handles[method], row)

            done = case_index + 1
            if done % max(1, int(args.progress_every)) == 0 or done == len(train_trace):
                progress = {
                    "stage": "sarsa_exploration_online",
                    "done": int(done),
                    "total": int(len(train_trace)),
                    "methods": {
                        method: _summarize(rows) for method, rows in records.items()
                    },
                    "controllers": {
                        method: {
                            "steps": int(controller.steps),
                            "episodes": int(controller.episodes),
                            "epsilon": float(controller.epsilon),
                            **controller.study_summary(),
                        }
                        for method, controller in controllers.items()
                    },
                }
                _write_json(args.output_dir / "progress.json", progress)
                for method, controller in controllers.items():
                    controller.save(args.output_dir / f"{method}_partial.npz")
                for handle in handles.values():
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "stage": "sarsa_exploration_online",
                            "done": int(done),
                            "total": int(len(train_trace)),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()

    case_count = len(train_trace)
    if case_count == 1000:
        windows = {
            "all_1000": (0, 1000),
            "first_500": (0, 500),
            "last_500": (500, 1000),
            "last_250": (750, 1000),
        }
    elif case_count == 2000:
        windows = {
            "all_2000": (0, 2000),
            "first_1000": (0, 1000),
            "last_1000": (1000, 2000),
            "last_500": (1500, 2000),
        }
    else:
        windows = {f"all_{case_count}": (0, case_count)}
    setup_sequences = [
        tuple(row["setup_hash"] for row in records[method]) for method in methods
    ]
    same_setup_rate = float(
        np.mean(
            [
                len({sequence[index] for sequence in setup_sequences}) == 1
                for index in range(len(train_trace))
            ]
        )
    )
    result = {
        "protocol": protocol,
        "same_setup_rate": same_setup_rate,
        "windows": {
            name: _window_summary(
                records,
                start=start,
                stop=stop,
                seed=int(args.controller_seed + start + stop),
            )
            for name, (start, stop) in windows.items()
        },
        "controllers": {
            method: {
                "steps": int(controller.steps),
                "episodes": int(controller.episodes),
                "epsilon": float(controller.epsilon),
                **controller.study_summary(),
            }
            for method, controller in controllers.items()
        },
        "artifacts": {
            "trajectories": str(trajectories_dir),
            "online_performance_csv": str(args.output_dir / "online_performance.csv"),
            "action_trajectory_csv": str(args.output_dir / "action_trajectory.csv"),
        },
    }
    for method, controller in controllers.items():
        controller.save(args.output_dir / f"{method}_controller.npz")
    _write_performance_csv(args.output_dir / "online_performance.csv", records)
    _write_action_csv(args.output_dir / "action_trajectory.csv", records)
    _write_json(args.output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare selected SARSA policies on one common frozen-setup trace."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--bandit-state",
        type=Path,
        default=Path(
            "results/online_td_lambda_v1/run_logs/"
            "paired_online_bandit_warmup1000_seed39393939/bandit_state.pkl"
        ),
    )
    parser.add_argument(
        "--ppo-model",
        type=Path,
        default=Path(
            "results/joint/mature_tune7_ppo_repro_20260423/run_logs/"
            "exp44_warm1000_original_2500_20260716/model_best.zip"
        ),
    )
    parser.add_argument("--train-seeds", default="39396939,39402939,39408939")
    parser.add_argument("--train-cases-per-seed", type=int, default=500)
    parser.add_argument("--train-take", type=int, default=1000)
    parser.add_argument(
        "--variants",
        default=",".join(STUDY_METHODS),
        help="Comma-separated SARSA study variants to run.",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--trace-shuffle-seed", type=int, default=39394016)
    # _build_traces expects validation arguments; this study deliberately has none.
    parser.add_argument("--eval-seeds", default="")
    parser.add_argument("--eval-cases", type=int, default=0)
    parser.add_argument("--controller-seed", type=int, default=39394016)
    parser.add_argument("--method-order-seed", type=int, default=39485019)
    parser.add_argument(
        "--initial-weight",
        type=float,
        default=None,
        help="Optional diagnostic override; by default read the prepared solver weight.",
    )
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=float, default=20_000.0)
    parser.add_argument("--potential-scale-sec", type=float, default=0.001)
    parser.add_argument("--failure-penalty-sec", type=float, default=0.1)
    parser.add_argument(
        "--matrix-grid-n", "--grid-n", dest="grid_n", type=int, default=40
    )
    parser.add_argument("--setup-param-resolution", type=int, default=20)
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=100)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
