from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import csv
import hashlib
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from SolvePhase.algorithms.sarsa import ExpectedSarsaLambda, SolveStateEncoder, run_td_episode
from run_exp44_online_rl import make_controller
from online_td_experiment_common import _json_ready, _write_json
from run_mature_bandit_rl_pipeline import _frozen_trace
from setup_aware_compare_common import (
    EXP44_MATRIX_GRID_N,
    EXP44_SETUP_PARAM_RESOLUTION,
    EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
    augment_setup_params,
    solve_fixed_w_case,
    solve_no_rl_case,
    validate_expected_setup_action_count,
)


DEFAULT_ALPHAS = (0.001, 0.005)
DEFAULT_LAMBDAS = (0.2, 0.5, 0.8)
DEFAULT_CONTROLLER_SEEDS = (39394016, 39394017, 39394018)
DEFAULT_TRAIN_SEEDS = (39396939, 39402939, 39408939)
DEFAULT_EVAL_SEEDS = (39414939, 39420939, 39426939)
FINAL_STREAM_SEEDS = (39394939, 39400939, 39406939, 39412939)


@dataclass(frozen=True)
class CandidateSpec:
    alpha: float
    trace_lambda: float

    @property
    def name(self) -> str:
        alpha = f"{self.alpha:g}".replace(".", "p")
        trace_lambda = f"{self.trace_lambda:g}".replace(".", "p")
        return f"alpha_{alpha}_lambda_{trace_lambda}"


def candidate_grid(
    alphas: Sequence[float] = DEFAULT_ALPHAS,
    trace_lambdas: Sequence[float] = DEFAULT_LAMBDAS,
) -> tuple[CandidateSpec, ...]:
    return tuple(
        CandidateSpec(float(alpha), float(trace_lambda))
        for alpha in alphas
        for trace_lambda in trace_lambdas
    )


def validate_seed_partition(
    *,
    train_seeds: Sequence[int],
    eval_seeds: Sequence[int],
    final_seeds: Sequence[int] = FINAL_STREAM_SEEDS,
) -> None:
    groups = {
        "tuning train": set(int(seed) for seed in train_seeds),
        "tuning validation": set(int(seed) for seed in eval_seeds),
        "final 2K": set(int(seed) for seed in final_seeds),
    }
    names = tuple(groups)
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            overlap = groups[left_name] & groups[right_name]
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} seeds overlap: {sorted(overlap)}"
                )


def trace_hash(trace: Sequence[tuple[Dict[str, Any], Dict[str, Any]]]) -> str:
    digest = hashlib.sha256()
    for mkw, params in trace:
        payload = {"mkw": dict(mkw), "params": dict(params)}
        digest.update(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def select_tuning_candidate(
    rows: Sequence[Dict[str, Any]],
    *,
    tie_tolerance_percentage_points: float = 0.1,
) -> Dict[str, Any] | None:
    eligible = [row for row in rows if not bool(row.get("rejected", False))]
    if not eligible:
        return None
    best_score = max(float(row["score_mean_minus_std_pct"]) for row in eligible)
    tied = [
        row
        for row in eligible
        if best_score - float(row["score_mean_minus_std_pct"])
        <= float(tie_tolerance_percentage_points) + 1.0e-12
    ]
    return min(
        tied,
        key=lambda row: (
            float(row["mean_end_to_end_runtime_sec"]),
            float(row["alpha"]),
            float(row["trace_lambda"]),
        ),
    )


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in str(raw).split(",") if part.strip())


def _controller_args(
    args: argparse.Namespace,
    *,
    spec: CandidateSpec,
    controller_seed: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        weights=str(args.weights),
        anchor_weight=float(args.anchor_weight),
        alpha=float(spec.alpha),
        td_decay_power=0.0,
        gamma=1.0,
        trace_lambda=float(spec.trace_lambda),
        epsilon_start=0.30,
        epsilon_final=0.03,
        epsilon_decay_steps=20_000.0,
        initial_q_sec=0.0,
        monte_carlo_alpha=0.0,
        monte_carlo_decay_power=0.0,
        adaptive_cycles=-1,
        exploration_mode="uniform",
        action_rbf_sigma=0.0,
        td_algorithm="true_online_sarsa",
        force_default_first_action=True,
        state_mode="setup_full",
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        seed=int(controller_seed),
    )


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize an empty outcome stream")
    return {
        "cases": int(len(rows)),
        "failed_count": int(sum(bool(row.get("failed", False)) for row in rows)),
        "mean_setup_runtime_sec": float(
            np.mean([float(row["setup_runtime"]) for row in rows])
        ),
        "mean_native_solve_runtime_sec": float(
            np.mean([float(row["solve_runtime"]) for row in rows])
        ),
        "mean_native_total_runtime_sec": float(
            np.mean([float(row["runtime"]) for row in rows])
        ),
        "mean_controller_runtime_sec": float(
            np.mean([float(row.get("infer_runtime", 0.0)) for row in rows])
        ),
        "mean_end_to_end_runtime_sec": float(
            np.mean(
                [
                    float(row["runtime"]) + float(row.get("infer_runtime", 0.0))
                    for row in rows
                ]
            )
        ),
        "mean_iterations": float(np.mean([float(row["iterations"]) for row in rows])),
    }


def _native_solve_improvement_pct(
    fixed_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
) -> float:
    fixed = float(np.mean([float(row["solve_runtime"]) for row in fixed_rows]))
    candidate = float(
        np.mean([float(row["solve_runtime"]) for row in candidate_rows])
    )
    if not np.isfinite(fixed) or fixed <= 0.0 or not np.isfinite(candidate):
        return float("nan")
    return float(100.0 * (fixed - candidate) / fixed)


def _trajectory_row(
    *,
    case_index: int,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    outcome: Dict[str, Any],
    eval_seed: int | None = None,
) -> Dict[str, Any]:
    row = {
        "case_index": int(case_index),
        "mkw": dict(mkw),
        "params": dict(params),
        "outcome": dict(outcome),
    }
    if eval_seed is not None:
        row["eval_seed"] = int(eval_seed)
    return row


def _write_json_lines(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), separators=(",", ":")))
            handle.write("\n")


def _read_json_lines(path: Path) -> list[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _configure_trace_environment(args: argparse.Namespace) -> None:
    import os

    values = {
        "MATRIX_GRID_N": int(args.grid_n),
        "SETUP_PARAM_RESOLUTION": int(args.setup_param_resolution),
        "EXPECTED_SETUP_ACTION_COUNT": EXP44_TUNE7_CATEGORICAL_ACTION_COUNT,
        "C_MIN": float(args.c_min),
        "C_MAX": float(args.c_max),
        "DIFCONV_A": "0,0,0",
        "SOLVER_TOL": float(args.tol),
        "SOLVER_MAX_ITER": int(args.max_cycles),
        "SETUP_BANDIT_METHOD": "linucbv4",
        "SETUP_TUNE_DIM": 7,
        "TUNE7_VARIANT": "categorical",
        "PROGRESS_EVERY": int(args.progress_every),
    }
    for name, value in values.items():
        os.environ[name] = str(value)


def _build_traces(
    args: argparse.Namespace,
    branch: Any,
) -> tuple[
    list[tuple[Dict[str, Any], Dict[str, Any]]],
    Dict[int, list[tuple[Dict[str, Any], Dict[str, Any]]]],
    Dict[str, Any],
]:
    train_seeds = _parse_values(args.train_seeds, int)
    eval_seeds = _parse_values(args.eval_seeds, int)
    train_parts = {
        int(seed): _frozen_trace(
            branch,
            cases=int(args.train_cases_per_seed),
            seed=int(seed),
        )
        for seed in train_seeds
    }
    candidates = [item for seed in train_seeds for item in train_parts[int(seed)]]
    order = np.random.default_rng(int(args.trace_shuffle_seed)).permutation(len(candidates))
    train_trace = [candidates[int(index)] for index in order[: int(args.train_take)]]
    eval_traces = {
        int(seed): _frozen_trace(branch, cases=int(args.eval_cases), seed=int(seed))
        for seed in eval_seeds
    }
    manifest = {
        "warmup_bandit_state": str(args.bandit_state),
        "train": {
            "seeds": [int(seed) for seed in train_seeds],
            "cases_per_seed": int(args.train_cases_per_seed),
            "candidate_cases": int(len(candidates)),
            "shuffle_seed": int(args.trace_shuffle_seed),
            "selected_cases": int(len(train_trace)),
            "sha256": trace_hash(train_trace),
            "per_seed_sha256": {
                str(seed): trace_hash(trace) for seed, trace in train_parts.items()
            },
        },
        "validation": {
            "seeds": [int(seed) for seed in eval_seeds],
            "cases_per_seed": int(args.eval_cases),
            "per_seed_sha256": {
                str(seed): trace_hash(trace) for seed, trace in eval_traces.items()
            },
        },
    }
    return train_trace, eval_traces, manifest


def _candidate_aggregate(
    *,
    spec: CandidateSpec,
    seed_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    rejection_reasons: list[str] = []
    for result in seed_results:
        seed = int(result["controller_seed"])
        candidate = result["validation"]["candidate"]
        fixed = result["validation"]["fixed_w1.6"]
        default = result["validation"]["default_solve"]
        allowed_failures = min(
            int(fixed["failed_count"]),
            int(default["failed_count"]),
        )
        if int(candidate["failed_count"]) > allowed_failures:
            rejection_reasons.append(f"seed {seed}: failures increased")
        iteration_limit = 1.05 * float(fixed["mean_iterations"])
        if float(candidate["mean_iterations"]) > iteration_limit:
            rejection_reasons.append(
                f"seed {seed}: mean iterations exceed fixed 1.6 by more than 5%"
            )
        improvement_vs_default = float(
            result["native_solve_improvement_vs_default_pct"]
        )
        if not np.isfinite(improvement_vs_default):
            rejection_reasons.append(
                f"seed {seed}: default comparison is not finite"
            )
        elif improvement_vs_default <= 0.0:
            rejection_reasons.append(
                f"seed {seed}: native solve did not exceed default solve"
            )

    improvements = np.asarray(
        [float(result["native_solve_improvement_pct"]) for result in seed_results],
        dtype=float,
    )
    improvements_vs_default = np.asarray(
        [
            float(result["native_solve_improvement_vs_default_pct"])
            for result in seed_results
        ],
        dtype=float,
    )
    end_to_end = np.asarray(
        [
            float(result["validation"]["candidate"]["mean_end_to_end_runtime_sec"])
            for result in seed_results
        ],
        dtype=float,
    )
    return {
        "candidate": spec.name,
        "alpha": float(spec.alpha),
        "trace_lambda": float(spec.trace_lambda),
        "rejected": bool(rejection_reasons),
        "rejection_reasons": rejection_reasons,
        "native_solve_improvement_by_controller_seed_pct": improvements.tolist(),
        "native_solve_improvement_vs_default_by_controller_seed_pct": (
            improvements_vs_default.tolist()
        ),
        "mean_native_solve_improvement_pct": float(np.mean(improvements)),
        "std_native_solve_improvement_pct": float(np.std(improvements)),
        "score_mean_minus_std_pct": float(np.mean(improvements) - np.std(improvements)),
        "mean_native_solve_improvement_vs_default_pct": float(
            np.mean(improvements_vs_default)
        ),
        "min_native_solve_improvement_vs_default_pct": float(
            np.min(improvements_vs_default)
        ),
        "mean_end_to_end_runtime_sec": float(np.mean(end_to_end)),
        "controller_seed_results": list(seed_results),
    }


def _write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = (
        "candidate",
        "alpha",
        "trace_lambda",
        "rejected",
        "rejection_reasons",
        "mean_native_solve_improvement_pct",
        "std_native_solve_improvement_pct",
        "score_mean_minus_std_pct",
        "mean_native_solve_improvement_vs_default_pct",
        "min_native_solve_improvement_vs_default_pct",
        "mean_end_to_end_runtime_sec",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            csv_row = {field: row.get(field) for field in fields}
            csv_row["rejection_reasons"] = "; ".join(row["rejection_reasons"])
            writer.writerow(csv_row)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    train_seeds = _parse_values(args.train_seeds, int)
    eval_seeds = _parse_values(args.eval_seeds, int)
    controller_seeds = _parse_values(args.controller_seeds, int)
    validate_seed_partition(train_seeds=train_seeds, eval_seeds=eval_seeds)
    if int(args.train_take) > len(train_seeds) * int(args.train_cases_per_seed):
        raise ValueError("train_take exceeds the generated tuning trace")
    if not args.bandit_state.exists():
        raise FileNotFoundError(args.bandit_state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _configure_trace_environment(args)
    with args.bandit_state.open("rb") as handle:
        branch = pickle.load(handle)["branch"]
    validate_expected_setup_action_count(branch)
    train_trace, eval_traces, trace_manifest = _build_traces(args, branch)
    _write_json(args.output_dir / "trace_manifest.json", trace_manifest)

    specs = candidate_grid(
        _parse_values(args.alphas, float),
        _parse_values(args.trace_lambdas, float),
    )
    protocol = {
        "algorithm": "true-online SARSA(lambda)",
        "td_algorithm": "true_online_sarsa",
        "exploration_mode": "uniform",
        "td_decay_power": 0.0,
        "gamma": 1.0,
        "state_mode": "setup_full",
        "weights": list(_parse_values(args.weights, float)),
        "epsilon": {"start": 0.30, "final": 0.03, "decay_steps": 20_000},
        "calibration_episodes": 0,
        "adaptive_cycles": None,
        "paired_reference": False,
        "controller_seeds": [int(seed) for seed in controller_seeds],
        "candidates": [asdict(spec) | {"name": spec.name} for spec in specs],
        "trace_manifest": trace_manifest,
        "selection": {
            "failure_gate": "candidate failures must not exceed fixed w=1.6",
            "iteration_gate": "each controller seed <= 1.05 * fixed w=1.6 mean iterations",
            "default_gate": "native solve improvement vs default must be positive for every controller seed",
            "score": "mean(native solve improvement %) - std across controller seeds",
            "tie_tolerance_percentage_points": 0.1,
            "tie_break": ["end-to-end runtime", "smaller alpha", "smaller lambda"],
        },
    }
    _write_json(args.output_dir / "config.json", protocol)

    results_by_candidate: Dict[str, list[Dict[str, Any]]] = {
        spec.name: [] for spec in specs
    }
    for controller_seed in controller_seeds:
        controllers: Dict[str, ExpectedSarsaLambda] = {}
        encoders: Dict[str, SolveStateEncoder] = {}
        train_rows: Dict[str, list[Dict[str, Any]]] = {spec.name: [] for spec in specs}
        for spec in specs:
            controller, encoder, _config = make_controller(
                _controller_args(args, spec=spec, controller_seed=int(controller_seed))
            )
            controllers[spec.name] = controller
            encoders[spec.name] = encoder

        seed_dir = args.output_dir / f"controller_seed_{controller_seed}"
        checkpoint_paths = {
            spec.name: seed_dir / spec.name / "controller.npz"
            for spec in specs
        }
        trajectory_paths = {
            spec.name: seed_dir / spec.name / "training_trajectory.jsonl"
            for spec in specs
        }
        reusable = {
            spec.name: (
                checkpoint_paths[spec.name].exists()
                and trajectory_paths[spec.name].exists()
            )
            for spec in specs
        }
        reuse_seed = bool(
            getattr(args, "reuse_controller_checkpoints", False)
            and all(reusable.values())
        )
        if getattr(args, "reuse_controller_checkpoints", False) and any(
            reusable.values()
        ) and not all(reusable.values()):
            raise RuntimeError(
                f"Controller seed {controller_seed} has an incomplete checkpoint set"
            )

        if reuse_seed:
            for spec in specs:
                controllers[spec.name].load(checkpoint_paths[spec.name])
                train_rows[spec.name] = _read_json_lines(
                    trajectory_paths[spec.name]
                )
                if len(train_rows[spec.name]) != len(train_trace):
                    raise RuntimeError(
                        f"Training trajectory length mismatch for {spec.name}"
                    )
            print(
                json.dumps(
                    {
                        "stage": "true_online_tuning_train_reused",
                        "controller_seed": int(controller_seed),
                        "cases": int(len(train_trace)),
                    }
                ),
                flush=True,
            )
        else:
            order_rng = np.random.default_rng(int(controller_seed) + 704_003)
            for case_index, (mkw, params) in enumerate(train_trace):
                for candidate_index in order_rng.permutation(len(specs)):
                    spec = specs[int(candidate_index)]
                    outcome = run_td_episode(
                        mkw=dict(mkw),
                        params=dict(params),
                        controller=controllers[spec.name],
                        encoder=encoders[spec.name],
                        solve_tol=float(args.tol),
                        solve_max_cycles=int(args.max_cycles),
                        learn=True,
                        explore=True,
                        record_action_metadata=True,
                    )
                    train_rows[spec.name].append(
                        _trajectory_row(
                            case_index=case_index,
                            mkw=mkw,
                            params=params,
                            outcome=outcome,
                        )
                    )
                done = case_index + 1
                if (
                    done % max(1, int(args.progress_every)) == 0
                    or done == len(train_trace)
                ):
                    print(
                        json.dumps(
                            {
                                "stage": "true_online_tuning_train",
                                "controller_seed": int(controller_seed),
                                "done": int(done),
                                "total": int(len(train_trace)),
                            }
                        ),
                        flush=True,
                    )

            for spec in specs:
                candidate_dir = seed_dir / spec.name
                controllers[spec.name].save(checkpoint_paths[spec.name])
                _write_json_lines(
                    trajectory_paths[spec.name],
                    train_rows[spec.name],
                )

        default_rows: list[Dict[str, Any]] = []
        fixed_rows: list[Dict[str, Any]] = []
        validation_rows: Dict[str, list[Dict[str, Any]]] = {
            spec.name: [] for spec in specs
        }
        validation_by_seed: Dict[int, Dict[str, list[Dict[str, Any]]]] = {
            int(seed): {
                "default_solve": [],
                "fixed_w1.6": [],
                **{spec.name: [] for spec in specs},
            }
            for seed in eval_seeds
        }
        eval_order_rng = np.random.default_rng(int(controller_seed) + 804_003)
        for eval_seed in eval_seeds:
            trace = eval_traces[int(eval_seed)]
            methods = (
                "default_solve",
                "fixed_w1.6",
                *[spec.name for spec in specs],
            )
            for case_index, (mkw, params) in enumerate(trace):
                for method_index in eval_order_rng.permutation(len(methods)):
                    method = methods[int(method_index)]
                    if method == "default_solve":
                        outcome = solve_no_rl_case(
                            params=dict(params),
                            mkw=dict(mkw),
                            solver_tol=float(args.tol),
                            solver_max_iter=int(args.max_cycles),
                            augment_params=augment_setup_params,
                        )
                        outcome = dict(outcome)
                        outcome["infer_runtime"] = 0.0
                        default_rows.append(outcome)
                    elif method == "fixed_w1.6":
                        outcome = solve_fixed_w_case(
                            params=dict(params),
                            mkw=dict(mkw),
                            w=1.6,
                            sweeps_down=1,
                            sweeps_up=1,
                            solve_tol=float(args.tol),
                            solve_max_cycles=int(args.max_cycles),
                        )
                        outcome = dict(outcome)
                        outcome["infer_runtime"] = 0.0
                        fixed_rows.append(outcome)
                    else:
                        outcome = run_td_episode(
                            mkw=dict(mkw),
                            params=dict(params),
                            controller=controllers[method],
                            encoder=encoders[method],
                            solve_tol=float(args.tol),
                            solve_max_cycles=int(args.max_cycles),
                            learn=False,
                            explore=False,
                            epsilon=0.0,
                            record_action_metadata=True,
                        )
                        validation_rows[method].append(outcome)
                    validation_by_seed[int(eval_seed)][method].append(dict(outcome))
            print(
                json.dumps(
                    {
                        "stage": "true_online_tuning_validation",
                        "controller_seed": int(controller_seed),
                        "eval_seed": int(eval_seed),
                        "cases": int(len(trace)),
                    }
                ),
                flush=True,
            )
            raw_dir = seed_dir / "validation_raw" / str(eval_seed)
            for method in methods:
                _write_json_lines(
                    raw_dir / f"{method}.jsonl",
                    validation_by_seed[int(eval_seed)][method],
                )

        default_summary = _summarize(default_rows)
        fixed_summary = _summarize(fixed_rows)
        for spec in specs:
            candidate_rows = validation_rows[spec.name]
            candidate_summary = _summarize(candidate_rows)
            per_eval_seed = {
                str(seed): {
                    "default_solve": _summarize(
                        validation_by_seed[int(seed)]["default_solve"]
                    ),
                    "fixed_w1.6": _summarize(
                        validation_by_seed[int(seed)]["fixed_w1.6"]
                    ),
                    "candidate": _summarize(
                        validation_by_seed[int(seed)][spec.name]
                    ),
                    "native_solve_improvement_pct": _native_solve_improvement_pct(
                        validation_by_seed[int(seed)]["fixed_w1.6"],
                        validation_by_seed[int(seed)][spec.name],
                    ),
                    "native_solve_improvement_vs_default_pct": (
                        _native_solve_improvement_pct(
                            validation_by_seed[int(seed)]["default_solve"],
                            validation_by_seed[int(seed)][spec.name],
                        )
                    ),
                }
                for seed in eval_seeds
            }
            seed_result = {
                "candidate": spec.name,
                "alpha": float(spec.alpha),
                "trace_lambda": float(spec.trace_lambda),
                "controller_seed": int(controller_seed),
                "training": _summarize(
                    [row["outcome"] for row in train_rows[spec.name]]
                ),
                "validation": {
                    "default_solve": default_summary,
                    "fixed_w1.6": fixed_summary,
                    "candidate": candidate_summary,
                    "per_eval_seed": per_eval_seed,
                },
                "native_solve_improvement_pct": _native_solve_improvement_pct(
                    fixed_rows,
                    candidate_rows,
                ),
                "native_solve_improvement_vs_default_pct": (
                    _native_solve_improvement_pct(
                        default_rows,
                        candidate_rows,
                    )
                ),
            }
            results_by_candidate[spec.name].append(seed_result)
            candidate_dir = seed_dir / spec.name
            _write_json(candidate_dir / "result.json", seed_result)
            paired_validation = [
                {
                    "case_index": int(index),
                    "default_solve": default,
                    "fixed_w1.6": fixed,
                    "candidate": candidate,
                }
                for index, (default, fixed, candidate) in enumerate(
                    zip(default_rows, fixed_rows, candidate_rows)
                )
            ]
            _write_json_lines(
                candidate_dir / "validation_details.jsonl",
                paired_validation,
            )

    aggregate_rows = [
        _candidate_aggregate(
            spec=spec,
            seed_results=results_by_candidate[spec.name],
        )
        for spec in specs
    ]
    selected = select_tuning_candidate(aggregate_rows)
    summary = {
        "protocol": protocol,
        "candidates": aggregate_rows,
        "selected": (
            None
            if selected is None
            else {
                "candidate": selected["candidate"],
                "alpha": float(selected["alpha"]),
                "trace_lambda": float(selected["trace_lambda"]),
                "score_mean_minus_std_pct": float(
                    selected["score_mean_minus_std_pct"]
                ),
                "mean_end_to_end_runtime_sec": float(
                    selected["mean_end_to_end_runtime_sec"]
                ),
            }
        ),
        "final_2k_allowed": bool(selected is not None),
    }
    _write_json(args.output_dir / "tuning_summary.json", summary)
    _write_summary_csv(args.output_dir / "tuning_summary.csv", aggregate_rows)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "true_online_tuning_done",
                    "selected": summary["selected"],
                    "final_2k_allowed": summary["final_2k_allowed"],
                    "output": args.output_dir / "tuning_summary.json",
                }
            )
        ),
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Controlled alpha/lambda selection for Exp44 true-online SARSA(lambda)."
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
    parser.add_argument("--controller-seeds", default="39394016,39394017,39394018")
    parser.add_argument("--eval-seeds", default="39414939,39420939,39426939")
    parser.add_argument("--eval-cases", type=int, default=96)
    parser.add_argument("--alphas", default="0.001,0.005")
    parser.add_argument("--trace-lambdas", default="0.2,0.5,0.8")
    parser.add_argument(
        "--weights",
        default="1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0",
    )
    parser.add_argument("--anchor-weight", type=float, default=1.0)
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
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--reuse-controller-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
