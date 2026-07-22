from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


_DIAGNOSIS_DIR = Path(__file__).resolve().parent
_TEST_DIR = _DIAGNOSIS_DIR.parent
for _path in (_TEST_DIR, _DIAGNOSIS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from setup_action_space import (
    DEFAULT_SETUP_PARAMS,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from run_sarsa_exploration_study import (
    _comparison,
    _normalize_outcome,
    _setup_hash,
    _summarize,
    _write_json_line,
)
from online_td_experiment_common import _git_revision, _write_json
from SolvePhase.algorithms.sarsa import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    run_td_episode,
)
from run_true_online_sarsa_tuning import (
    _build_traces,
    _configure_trace_environment,
    _parse_values,
    validate_seed_partition,
)
from setup_aware_compare_common import solve_fixed_w_case
from SolvePhase.algorithms.lcb import (
    BootstrapLcbSarsaController,
    BootstrapSarsaSpec,
    StagewiseLsviLcbController,
    StagewiseLsviLcbSpec,
)


FIXED_METHOD = "fixed_w1.6"
INDEPENDENT_METHOD = "independent_sarsa"
CANDIDATE_METHODS = (
    "shared_sarsa",
    "bootstrap_lcb_sarsa",
    "stagewise_lsvi_lcb",
)
METHODS = (FIXED_METHOD, INDEPENDENT_METHOD, *CANDIDATE_METHODS)


def _encoder(args: argparse.Namespace) -> SolveStateEncoder:
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


def _config(
    args: argparse.Namespace,
    *,
    shared: bool,
) -> ExpectedSarsaLambdaConfig:
    weights = _parse_values(args.weights, float)
    return ExpectedSarsaLambdaConfig(
        weights=tuple(float(weight) for weight in weights),
        anchor_weight=float(weights[0]),
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
        monte_carlo_decay_power=0.0,
        adaptive_cycles=None,
        exploration_mode="uniform",
        action_rbf_sigma=(float(args.action_rbf_sigma) if shared else 0.0),
        action_basis_mode=("compact_rbf" if shared else "legacy"),
        action_basis_centers=(
            tuple(_parse_values(args.action_rbf_centers, float)) if shared else ()
        ),
        td_algorithm="true_online_sarsa",
        force_default_first_action=True,
    )


def _make_controllers(
    args: argparse.Namespace,
    *,
    seed: int,
) -> tuple[Dict[str, Any], Dict[str, SolveStateEncoder]]:
    encoders = {method: _encoder(args) for method in METHODS if method != FIXED_METHOD}
    independent_config = _config(args, shared=False)
    shared_config = _config(args, shared=True)
    controllers: Dict[str, Any] = {
        INDEPENDENT_METHOD: ExpectedSarsaLambda(
            feature_dim=encoders[INDEPENDENT_METHOD].feature_dim,
            config=independent_config,
            seed=int(seed + 101),
            initial_parameters=encoders[INDEPENDENT_METHOD].constant_value_parameters(0.0),
        ),
        "shared_sarsa": ExpectedSarsaLambda(
            feature_dim=encoders["shared_sarsa"].feature_dim,
            config=shared_config,
            seed=int(seed + 211),
            initial_parameters=encoders["shared_sarsa"].constant_value_parameters(0.0),
        ),
        "bootstrap_lcb_sarsa": BootstrapLcbSarsaController(
            feature_dim=encoders["bootstrap_lcb_sarsa"].feature_dim,
            config=shared_config,
            spec=BootstrapSarsaSpec(
                members=int(args.bootstrap_members),
                episode_inclusion_probability=float(args.bootstrap_inclusion_probability),
                uncertainty_beta=float(args.bootstrap_beta),
            ),
            seed=int(seed + 307),
            initial_parameters=encoders[
                "bootstrap_lcb_sarsa"
            ].constant_value_parameters(0.0),
        ),
        "stagewise_lsvi_lcb": StagewiseLsviLcbController(
            feature_dim=encoders["stagewise_lsvi_lcb"].feature_dim,
            config=shared_config,
            spec=StagewiseLsviLcbSpec(
                horizon=int(args.max_cycles),
                ridge=float(args.lsvi_ridge),
                uncertainty_beta=float(args.lsvi_beta),
                residual_floor_sec=float(args.lsvi_residual_floor_sec),
                q_max_sec=float(args.failure_penalty_sec),
            ),
            seed=int(seed + 401),
        ),
    }
    return controllers, encoders


def _controller_summary(controller: Any) -> Dict[str, Any]:
    summary = {
        "steps": int(controller.steps),
        "episodes": int(controller.episodes),
        "epsilon": float(controller.epsilon),
    }
    if hasattr(controller, "summary"):
        summary.update(controller.summary())
    return summary


def _run_stream(
    *,
    args: argparse.Namespace,
    trace: Sequence[tuple[Dict[str, Any], Dict[str, Any]]],
    controllers: Dict[str, Any],
    encoders: Dict[str, SolveStateEncoder],
    output_dir: Path,
    phase: str,
    controller_seed: int,
    learn: bool,
    explore: bool,
    eval_seed: int | None = None,
) -> Dict[str, list[Dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records: Dict[str, list[Dict[str, Any]]] = {method: [] for method in METHODS}
    handles = {
        method: (output_dir / f"{method}.jsonl").open("w", encoding="utf-8")
        for method in METHODS
    }
    order_seed = int(
        args.method_order_seed
        + controller_seed
        + (0 if eval_seed is None else 10_000 + int(eval_seed))
    )
    order_rng = np.random.default_rng(order_seed)
    try:
        for case_index, (mkw, params) in enumerate(trace):
            order = [METHODS[index] for index in order_rng.permutation(len(METHODS))]
            for method in order:
                if method == FIXED_METHOD:
                    outcome = solve_fixed_w_case(
                        params=dict(params),
                        mkw=dict(mkw),
                        w=1.6,
                        sweeps_down=1,
                        sweeps_up=1,
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                    )
                else:
                    outcome = run_td_episode(
                        mkw=dict(mkw),
                        params=dict(params),
                        controller=controllers[method],
                        encoder=encoders[method],
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                        learn=learn,
                        explore=explore,
                        record_action_metadata=True,
                    )
                row = {
                    "case_index": int(case_index),
                    "eval_seed": None if eval_seed is None else int(eval_seed),
                    "mkw": dict(mkw),
                    "params": dict(params),
                    "setup_hash": _setup_hash(params),
                    "execution_order": list(order),
                    "outcome": _normalize_outcome(outcome),
                }
                records[method].append(row)
                _write_json_line(handles[method], row)

            done = case_index + 1
            if done % max(1, int(args.progress_every)) == 0 or done == len(trace):
                progress = {
                    "phase": phase,
                    "controller_seed": int(controller_seed),
                    "eval_seed": None if eval_seed is None else int(eval_seed),
                    "done": int(done),
                    "total": int(len(trace)),
                    "methods": {
                        method: _summarize(rows) for method, rows in records.items()
                    },
                    "controllers": {
                        method: _controller_summary(controller)
                        for method, controller in controllers.items()
                    },
                }
                _write_json(output_dir / "progress.json", progress)
                if learn:
                    for method, controller in controllers.items():
                        controller.save(output_dir / f"{method}_partial.npz")
                for handle in handles.values():
                    handle.flush()
                print(
                    json.dumps(
                        {
                            "phase": phase,
                            "controller_seed": int(controller_seed),
                            "eval_seed": eval_seed,
                            "done": int(done),
                            "total": int(len(trace)),
                        }
                    ),
                    flush=True,
                )
    finally:
        for handle in handles.values():
            handle.close()
    return records


def _validate_pairing(records: Dict[str, Sequence[Dict[str, Any]]]) -> float:
    lengths = {len(rows) for rows in records.values()}
    if len(lengths) != 1:
        raise ValueError("All methods must contain the same number of records")
    same = []
    for index in range(next(iter(lengths), 0)):
        keys = {
            (
                records[method][index]["case_index"],
                records[method][index]["eval_seed"],
                records[method][index]["setup_hash"],
            )
            for method in METHODS
        }
        same.append(len(keys) == 1)
    rate = float(np.mean(same)) if same else 1.0
    if rate != 1.0:
        raise ValueError(f"Methods do not share an identical setup stream: {rate}")
    return rate


def _calibration_summary(
    rows: Sequence[Dict[str, Any]],
    *,
    failure_penalty_sec: float,
) -> Dict[str, Any]:
    predictions: list[float] = []
    uncertainties: list[float] = []
    returns: list[float] = []
    for row in rows:
        outcome = row["outcome"]
        q_values = outcome.get("cycle_selected_q_values", [])
        widths = outcome.get("cycle_selected_uncertainties", [])
        costs = np.asarray(outcome.get("cycle_times", []), dtype=float)
        if not (len(q_values) == len(widths) == costs.size):
            continue
        future = np.cumsum(costs[::-1])[::-1]
        if bool(outcome.get("failed", False)) and future.size:
            future += float(failure_penalty_sec)
        for prediction, width, observed in zip(q_values, widths, future):
            if np.isfinite(prediction) and np.isfinite(width) and np.isfinite(observed):
                predictions.append(float(prediction))
                uncertainties.append(max(float(width), 0.0))
                returns.append(float(observed))
    if not predictions:
        return {"samples": 0}
    predicted = np.asarray(predictions, dtype=float)
    width = np.asarray(uncertainties, dtype=float)
    observed = np.asarray(returns, dtype=float)
    error = np.abs(observed - predicted)
    z_values = {"50": 0.67448975, "80": 1.28155157, "95": 1.95996398}
    return {
        "samples": int(predicted.size),
        "mean_abs_error_sec": float(np.mean(error)),
        "mean_uncertainty_sec": float(np.mean(width)),
        "coverage": {
            level: float(np.mean(error <= z_value * width))
            for level, z_value in z_values.items()
        },
        "lower_bound_coverage": {
            level: float(np.mean(observed >= predicted - z_value * width))
            for level, z_value in z_values.items()
        },
    }


def _improvement(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(100.0 * (np.mean(reference) - np.mean(candidate)) / np.mean(reference))


def _hierarchical_improvement(
    seed_records: Sequence[Dict[str, Any]],
    *,
    candidate: str,
    reference: str,
    metric: str,
    bootstrap_samples: int,
    seed: int,
) -> Dict[str, Any]:
    clusters: list[list[tuple[np.ndarray, np.ndarray]]] = []
    all_reference: list[np.ndarray] = []
    all_candidate: list[np.ndarray] = []
    for seed_result in seed_records:
        eval_clusters = []
        for eval_seed in sorted(seed_result["validation_records"]):
            records = seed_result["validation_records"][eval_seed]
            reference_values = np.asarray(
                [float(row["outcome"][metric]) for row in records[reference]],
                dtype=float,
            )
            candidate_values = np.asarray(
                [float(row["outcome"][metric]) for row in records[candidate]],
                dtype=float,
            )
            eval_clusters.append((reference_values, candidate_values))
            all_reference.append(reference_values)
            all_candidate.append(candidate_values)
        clusters.append(eval_clusters)
    point = _improvement(np.concatenate(all_reference), np.concatenate(all_candidate))
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(bootstrap_samples), dtype=float)
    for sample_index in range(int(bootstrap_samples)):
        sampled_reference: list[np.ndarray] = []
        sampled_candidate: list[np.ndarray] = []
        for controller_index in rng.integers(len(clusters), size=len(clusters)):
            eval_clusters = clusters[int(controller_index)]
            for eval_index in rng.integers(len(eval_clusters), size=len(eval_clusters)):
                reference_values, candidate_values = eval_clusters[int(eval_index)]
                case_indices = rng.integers(reference_values.size, size=reference_values.size)
                sampled_reference.append(reference_values[case_indices])
                sampled_candidate.append(candidate_values[case_indices])
        samples[sample_index] = _improvement(
            np.concatenate(sampled_reference),
            np.concatenate(sampled_candidate),
        )
    return {
        "improvement_pct": point,
        "hierarchical_bootstrap_95pct": [
            float(np.percentile(samples, 2.5)),
            float(np.percentile(samples, 97.5)),
        ],
        "bootstrap_samples": int(bootstrap_samples),
    }


def _aggregate(
    seed_results: Sequence[Dict[str, Any]],
    *,
    bootstrap_samples: int,
) -> tuple[Dict[str, Any], str | None]:
    aggregate: Dict[str, Any] = {}
    eligible: list[tuple[float, float, str]] = []
    for offset, candidate in enumerate(CANDIDATE_METHODS):
        native_vs_fixed = _hierarchical_improvement(
            seed_results,
            candidate=candidate,
            reference=FIXED_METHOD,
            metric="solve_runtime",
            bootstrap_samples=bootstrap_samples,
            seed=90210 + offset,
        )
        native_vs_independent = _hierarchical_improvement(
            seed_results,
            candidate=candidate,
            reference=INDEPENDENT_METHOD,
            metric="solve_runtime",
            bootstrap_samples=bootstrap_samples,
            seed=90310 + offset,
        )
        e2e_vs_fixed = _hierarchical_improvement(
            seed_results,
            candidate=candidate,
            reference=FIXED_METHOD,
            metric="end_to_end_runtime",
            bootstrap_samples=bootstrap_samples,
            seed=90410 + offset,
        )
        by_controller_seed = []
        failures_increased = False
        catastrophic_seed = False
        for seed_result in seed_results:
            candidate_rows = seed_result["validation_flat"][candidate]
            fixed_rows = seed_result["validation_flat"][FIXED_METHOD]
            improvement = _improvement(
                np.asarray([row["outcome"]["solve_runtime"] for row in fixed_rows]),
                np.asarray([row["outcome"]["solve_runtime"] for row in candidate_rows]),
            )
            by_controller_seed.append(float(improvement))
            failures_increased |= sum(
                bool(row["outcome"].get("failed", False)) for row in candidate_rows
            ) > sum(bool(row["outcome"].get("failed", False)) for row in fixed_rows)
            catastrophic_seed |= improvement < -5.0
        score = float(np.mean(by_controller_seed) - np.std(by_controller_seed))
        lower_vs_independent = float(
            native_vs_independent["hierarchical_bootstrap_95pct"][0]
        )
        passed = bool(
            not failures_increased
            and not catastrophic_seed
            and lower_vs_independent > 0.0
        )
        aggregate[candidate] = {
            "native_solve_vs_fixed": native_vs_fixed,
            "native_solve_vs_independent_sarsa": native_vs_independent,
            "end_to_end_vs_fixed": e2e_vs_fixed,
            "native_improvement_vs_fixed_by_controller_seed_pct": by_controller_seed,
            "score_mean_minus_std_pct": score,
            "failures_increased": failures_increased,
            "any_seed_slower_than_fixed_by_over_5pct": catastrophic_seed,
            "passed": passed,
        }
        if passed:
            mean_e2e = float(
                np.mean(
                    [
                        row["outcome"]["end_to_end_runtime"]
                        for seed_result in seed_results
                        for row in seed_result["validation_flat"][candidate]
                    ]
                )
            )
            eligible.append((score, -mean_e2e, candidate))
    winner = max(eligible)[2] if eligible else None
    return aggregate, winner


def _write_summary_csv(path: Path, aggregate: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "method",
        "passed",
        "score_mean_minus_std_pct",
        "native_vs_fixed_pct",
        "native_vs_fixed_ci_low",
        "native_vs_fixed_ci_high",
        "native_vs_independent_pct",
        "native_vs_independent_ci_low",
        "native_vs_independent_ci_high",
        "end_to_end_vs_fixed_pct",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, values in aggregate.items():
            fixed = values["native_solve_vs_fixed"]
            independent = values["native_solve_vs_independent_sarsa"]
            writer.writerow(
                {
                    "method": method,
                    "passed": values["passed"],
                    "score_mean_minus_std_pct": values["score_mean_minus_std_pct"],
                    "native_vs_fixed_pct": fixed["improvement_pct"],
                    "native_vs_fixed_ci_low": fixed["hierarchical_bootstrap_95pct"][0],
                    "native_vs_fixed_ci_high": fixed["hierarchical_bootstrap_95pct"][1],
                    "native_vs_independent_pct": independent["improvement_pct"],
                    "native_vs_independent_ci_low": independent[
                        "hierarchical_bootstrap_95pct"
                    ][0],
                    "native_vs_independent_ci_high": independent[
                        "hierarchical_bootstrap_95pct"
                    ][1],
                    "end_to_end_vs_fixed_pct": values["end_to_end_vs_fixed"][
                        "improvement_pct"
                    ],
                }
            )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.smoke:
        args.train_cases_per_seed = 2
        args.train_take = 2
        args.eval_cases = 1
        args.controller_seeds = str(_parse_values(args.controller_seeds, int)[0])
        args.bootstrap_samples = min(int(args.bootstrap_samples), 100)
        args.progress_every = 1
    if float(args.potential_scale_sec) != 0.0:
        raise ValueError("The controlled shared-action screen requires potential_scale=0")
    if not args.bandit_state.exists():
        raise FileNotFoundError(args.bandit_state)

    validate_seed_partition(
        train_seeds=_parse_values(args.train_seeds, int),
        eval_seeds=_parse_values(args.eval_seeds, int),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_trace_environment(args)
    with args.bandit_state.open("rb") as handle:
        branch = pickle.load(handle)["branch"]
    train_trace, eval_traces, trace_manifest = _build_traces(args, branch)
    _write_json(args.output_dir / "trace_manifest.json", trace_manifest)
    controller_seeds = _parse_values(args.controller_seeds, int)
    protocol = {
        "git_revision": _git_revision(),
        "purpose": "controlled shared-action online RL screen",
        "methods": list(METHODS),
        "common_trace": "one frozen (instance, setup) stream per phase",
        "train_cases": int(len(train_trace)),
        "validation_cases_per_seed": int(args.eval_cases),
        "controller_seeds": list(controller_seeds),
        "state_mode": "setup_full",
        "actions": list(_parse_values(args.weights, float)),
        "action_basis": {
            "mode": "compact_rbf",
            "centers": list(_parse_values(args.action_rbf_centers, float)),
            "sigma": float(args.action_rbf_sigma),
        },
        "sarsa": {
            "alpha": float(args.alpha),
            "lambda": float(args.trace_lambda),
            "gamma": 1.0,
            "epsilon_start": float(args.epsilon_start),
            "epsilon_final": float(args.epsilon_final),
            "epsilon_decay_steps": float(args.epsilon_decay_steps),
        },
        "potential_scale_sec": 0.0,
        "bootstrap": {
            "members": int(args.bootstrap_members),
            "episode_inclusion_probability": float(
                args.bootstrap_inclusion_probability
            ),
            "beta": float(args.bootstrap_beta),
        },
        "lsvi": {
            "ridge": float(args.lsvi_ridge),
            "beta": float(args.lsvi_beta),
            "residual_floor_sec": float(args.lsvi_residual_floor_sec),
            "update": "backward refit after every solve",
        },
        "selection_rule": {
            "no_extra_failures": True,
            "no_controller_seed_below_fixed_by_more_than_pct": 5.0,
            "pooled_95pct_lower_vs_independent_sarsa_must_exceed_pct": 0.0,
            "rank": "mean improvement vs fixed minus controller-seed std",
            "tie_break": "lower controller-inclusive runtime",
        },
    }
    _write_json(args.output_dir / "config.json", protocol)

    seed_results: list[Dict[str, Any]] = []
    for controller_seed in controller_seeds:
        seed_dir = args.output_dir / f"controller_seed_{controller_seed}"
        controllers, encoders = _make_controllers(
            args,
            seed=int(controller_seed),
        )
        training_records = _run_stream(
            args=args,
            trace=train_trace,
            controllers=controllers,
            encoders=encoders,
            output_dir=seed_dir / "training",
            phase="training",
            controller_seed=int(controller_seed),
            learn=True,
            explore=True,
        )
        _validate_pairing(training_records)
        for method, controller in controllers.items():
            controller.save(seed_dir / f"{method}_controller.npz")

        validation_records: Dict[int, Dict[str, list[Dict[str, Any]]]] = {}
        validation_flat: Dict[str, list[Dict[str, Any]]] = {
            method: [] for method in METHODS
        }
        for eval_seed, trace in eval_traces.items():
            records = _run_stream(
                args=args,
                trace=trace,
                controllers=controllers,
                encoders=encoders,
                output_dir=seed_dir / "validation" / f"seed_{eval_seed}",
                phase="frozen_validation",
                controller_seed=int(controller_seed),
                learn=False,
                explore=False,
                eval_seed=int(eval_seed),
            )
            _validate_pairing(records)
            validation_records[int(eval_seed)] = records
            for method in METHODS:
                validation_flat[method].extend(records[method])

        seed_summary = {
            "controller_seed": int(controller_seed),
            "training": {
                method: _summarize(rows) for method, rows in training_records.items()
            },
            "validation": {
                method: _summarize(rows) for method, rows in validation_flat.items()
            },
            "validation_comparisons_vs_fixed": {
                method: _comparison(
                    validation_flat[method],
                    validation_flat[FIXED_METHOD],
                    seed=int(controller_seed + index),
                )
                for index, method in enumerate(METHODS)
                if method != FIXED_METHOD
            },
            "calibration": {
                method: _calibration_summary(
                    validation_flat[method],
                    failure_penalty_sec=float(args.failure_penalty_sec),
                )
                for method in ("bootstrap_lcb_sarsa", "stagewise_lsvi_lcb")
            },
            "controllers": {
                method: _controller_summary(controller)
                for method, controller in controllers.items()
            },
        }
        _write_json(seed_dir / "result.json", seed_summary)
        seed_results.append(
            {
                **seed_summary,
                "validation_records": validation_records,
                "validation_flat": validation_flat,
            }
        )

    aggregate, winner = _aggregate(
        seed_results,
        bootstrap_samples=int(args.bootstrap_samples),
    )
    result = {
        "protocol": protocol,
        "trace_manifest": trace_manifest,
        "aggregate": aggregate,
        "selected_candidate": winner,
        "joint_online_recommendation": (
            "advance selected candidate to a fresh 2K+2K joint-online run"
            if winner is not None
            else "do not run joint-online; inspect sharing and confidence diagnostics"
        ),
        "controller_seed_results": [
            {
                key: value
                for key, value in seed_result.items()
                if key not in {"validation_records", "validation_flat"}
            }
            for seed_result in seed_results
        ],
        "artifacts": {
            "summary_csv": str(args.output_dir / "summary.csv"),
            "controller_seed_dirs": [
                str(args.output_dir / f"controller_seed_{seed}")
                for seed in controller_seeds
            ],
        },
    }
    _write_summary_csv(args.output_dir / "summary.csv", aggregate)
    _write_json(args.output_dir / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Screen shared-action SARSA, bootstrap SARSA, and LSVI-LCB."
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
    parser.add_argument("--train-seeds", default="39396939,39402939,39408939")
    parser.add_argument("--train-cases-per-seed", type=int, default=500)
    parser.add_argument("--train-take", type=int, default=1000)
    parser.add_argument("--trace-shuffle-seed", type=int, default=39394016)
    parser.add_argument("--eval-seeds", default="39414939,39420939,39426939")
    parser.add_argument("--eval-cases", type=int, default=250)
    parser.add_argument("--controller-seeds", default="39394016,39394017,39394018")
    parser.add_argument("--method-order-seed", type=int, default=39485019)
    parser.add_argument(
        "--weights",
        default="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )
    parser.add_argument("--action-rbf-centers", default="1.0,1.25,1.5,1.75,2.0")
    parser.add_argument("--action-rbf-sigma", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--epsilon-start", type=float, default=0.30)
    parser.add_argument("--epsilon-final", type=float, default=0.03)
    parser.add_argument("--epsilon-decay-steps", type=float, default=20_000.0)
    parser.add_argument("--potential-scale-sec", type=float, default=0.0)
    parser.add_argument("--failure-penalty-sec", type=float, default=0.1)
    parser.add_argument("--bootstrap-members", type=int, default=5)
    parser.add_argument(
        "--bootstrap-inclusion-probability", type=float, default=0.8
    )
    parser.add_argument("--bootstrap-beta", type=float, default=1.0)
    parser.add_argument("--lsvi-ridge", type=float, default=1.0)
    parser.add_argument("--lsvi-beta", type=float, default=2.0)
    parser.add_argument("--lsvi-residual-floor-sec", type=float, default=1.0e-3)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument(
        "--matrix-grid-n", "--grid-n", dest="grid_n", type=int, default=40
    )
    parser.add_argument("--setup-param-resolution", type=int, default=20)
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
