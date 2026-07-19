from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import copy
import json
import platform
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from online_td_lambda import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    run_td_episode,
)
from setup_aware_compare_common import (
    augment_setup_params,
    generate_difconv_instances,
    solve_fixed_w_case,
    solve_no_rl_case,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2), encoding="utf-8")


def _git_revision() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def _load_dominant_setup(path: Path) -> tuple[Dict[str, Any], int, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = [json.dumps(case["params"], sort_keys=True) for case in payload["evaluation"]["per_case"]]
    dominant, count = Counter(encoded).most_common(1)[0]
    return dict(json.loads(dominant)), int(count), int(len(encoded))


def _runtime_with_overhead(row: Dict[str, Any]) -> float:
    return float(row["runtime"]) + float(row.get("infer_runtime", 0.0))


def _solve_with_overhead(row: Dict[str, Any]) -> float:
    return float(row["solve_runtime"]) + float(row.get("infer_runtime", 0.0))


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    mean_optional = lambda field: float(
        np.mean([float(row.get(field, 0.0)) for row in rows])
    )
    return {
        "cases": int(len(rows)),
        "failed_count": int(sum(bool(row.get("failed", False)) for row in rows)),
        "mean_runtime_sec": float(np.mean([float(row["runtime"]) for row in rows])),
        "mean_runtime_with_overhead_sec": float(np.mean([_runtime_with_overhead(row) for row in rows])),
        "mean_setup_runtime_sec": float(np.mean([float(row["setup_runtime"]) for row in rows])),
        "mean_solve_runtime_sec": float(np.mean([float(row["solve_runtime"]) for row in rows])),
        "mean_solve_with_overhead_sec": float(np.mean([_solve_with_overhead(row) for row in rows])),
        "mean_policy_overhead_sec": float(np.mean([float(row.get("infer_runtime", 0.0)) for row in rows])),
        "mean_feature_runtime_sec": mean_optional("feature_runtime"),
        "mean_decision_runtime_sec": mean_optional("decision_runtime"),
        "mean_update_runtime_sec": mean_optional("update_runtime"),
        "mean_iterations": float(np.mean([int(row["iterations"]) for row in rows])),
    }


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(2000, dtype=float)
    for index in range(means.size):
        sample = rng.integers(0, values.size, size=values.size)
        means[index] = float(np.mean(values[sample]))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _bootstrap_aggregate_improvement(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    seed: int,
) -> list[float]:
    rng = np.random.default_rng(seed)
    improvements = np.empty(2000, dtype=float)
    for index in range(improvements.size):
        sample = rng.integers(0, reference.size, size=reference.size)
        reference_mean = float(np.mean(reference[sample]))
        candidate_mean = float(np.mean(candidate[sample]))
        improvements[index] = 100.0 * (reference_mean - candidate_mean) / reference_mean
    return [
        float(np.percentile(improvements, 2.5)),
        float(np.percentile(improvements, 97.5)),
    ]


def _paired(reference: Sequence[Dict[str, Any]], candidate: Sequence[Dict[str, Any]], *, seed: int) -> Dict[str, Any]:
    reference_total = np.asarray([_runtime_with_overhead(row) for row in reference], dtype=float)
    candidate_total = np.asarray([_runtime_with_overhead(row) for row in candidate], dtype=float)
    reference_solve = np.asarray([_solve_with_overhead(row) for row in reference], dtype=float)
    candidate_solve = np.asarray([_solve_with_overhead(row) for row in candidate], dtype=float)
    total_improvement = 100.0 * (reference_total - candidate_total) / reference_total
    solve_improvement = 100.0 * (reference_solve - candidate_solve) / reference_solve
    total_difference = reference_total - candidate_total
    solve_difference = reference_solve - candidate_solve
    return {
        "aggregate_total_improvement_pct": float(
            100.0 * (np.mean(reference_total) - np.mean(candidate_total)) / np.mean(reference_total)
        ),
        "aggregate_total_improvement_95pct": _bootstrap_aggregate_improvement(
            reference_total,
            candidate_total,
            seed=seed,
        ),
        "aggregate_solve_improvement_pct": float(
            100.0 * (np.mean(reference_solve) - np.mean(candidate_solve)) / np.mean(reference_solve)
        ),
        "aggregate_solve_improvement_95pct": _bootstrap_aggregate_improvement(
            reference_solve,
            candidate_solve,
            seed=seed + 1,
        ),
        "mean_total_difference_sec": float(np.mean(total_difference)),
        "total_difference_95pct_sec": _bootstrap_interval(total_difference, seed=seed + 2),
        "mean_solve_difference_sec": float(np.mean(solve_difference)),
        "solve_difference_95pct_sec": _bootstrap_interval(solve_difference, seed=seed + 3),
        "mean_total_improvement_pct": float(np.mean(total_improvement)),
        "total_improvement_95pct": _bootstrap_interval(total_improvement, seed=seed + 4),
        "mean_solve_improvement_pct": float(np.mean(solve_improvement)),
        "solve_improvement_95pct": _bootstrap_interval(solve_improvement, seed=seed + 5),
        "total_win_rate": float(np.mean(candidate_total < reference_total)),
        "solve_win_rate": float(np.mean(candidate_solve < reference_solve)),
    }


def _method_outcome(
    method: str,
    *,
    mkw: Dict[str, Any],
    params: Dict[str, Any],
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    tol: float,
    max_cycles: int,
    learn: bool,
    explore: bool,
    defer_monte_carlo_update: bool = False,
) -> Dict[str, Any]:
    if method == "default_solve":
        return solve_no_rl_case(
            params=dict(params),
            mkw=dict(mkw),
            solver_tol=float(tol),
            solver_max_iter=int(max_cycles),
            augment_params=augment_setup_params,
        )
    if method.startswith("fixed_w"):
        return solve_fixed_w_case(
            params=dict(params),
            mkw=dict(mkw),
            w=float(method.removeprefix("fixed_w")),
            sweeps_down=1,
            sweeps_up=1,
            solve_tol=float(tol),
            solve_max_cycles=int(max_cycles),
        )
    if method == "td_lambda":
        return run_td_episode(
            mkw=dict(mkw),
            params=dict(params),
            controller=controller,
            encoder=encoder,
            solve_tol=float(tol),
            solve_max_cycles=int(max_cycles),
            learn=learn,
            explore=explore,
            defer_monte_carlo_update=defer_monte_carlo_update,
        )
    raise ValueError(f"Unknown method: {method}")


def _run_stream(
    *,
    instances: Sequence[tuple[Dict[str, Any], np.ndarray]],
    params: Dict[str, Any],
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    tol: float,
    max_cycles: int,
    methods: Sequence[str],
    learn: bool,
    explore: bool,
    seed: int,
    progress_stage: str,
    progress_every: int,
) -> Dict[str, list[Dict[str, Any]]]:
    trace = [(dict(mkw), dict(params)) for mkw, _context in instances]
    return _run_trace_stream(
        trace=trace,
        controller=controller,
        encoder=encoder,
        tol=tol,
        max_cycles=max_cycles,
        methods=methods,
        learn=learn,
        explore=explore,
        seed=seed,
        progress_stage=progress_stage,
        progress_every=progress_every,
    )


def _run_trace_stream(
    *,
    trace: Sequence[tuple[Dict[str, Any], Dict[str, Any]]],
    controller: ExpectedSarsaLambda,
    encoder: SolveStateEncoder,
    tol: float,
    max_cycles: int,
    methods: Sequence[str],
    learn: bool,
    explore: bool,
    seed: int,
    progress_stage: str,
    progress_every: int,
    transition_budget: int | None = None,
    episode_budget: int | None = None,
    monte_carlo_baseline_method: str | None = None,
) -> Dict[str, list[Dict[str, Any]]]:
    if monte_carlo_baseline_method is not None:
        if monte_carlo_baseline_method not in methods:
            raise ValueError("monte_carlo_baseline_method must be present in methods")
        if "td_lambda" not in methods:
            raise ValueError("paired Monte Carlo updates require td_lambda in methods")
    results: Dict[str, list[Dict[str, Any]]] = {method: [] for method in methods}
    rng = np.random.default_rng(seed)
    for index, (mkw, params) in enumerate(trace):
        order = [methods[i] for i in rng.permutation(len(methods))]
        case_outcomes: Dict[str, Dict[str, Any]] = {}
        for method in order:
            outcome = _method_outcome(
                method,
                mkw=mkw,
                params=params,
                controller=controller,
                encoder=encoder,
                tol=tol,
                max_cycles=max_cycles,
                learn=bool(learn and method == "td_lambda"),
                explore=bool(explore and method == "td_lambda"),
                defer_monte_carlo_update=bool(
                    learn
                    and method == "td_lambda"
                    and monte_carlo_baseline_method is not None
                ),
            )
            case_outcomes[method] = dict(outcome)
        if monte_carlo_baseline_method is not None:
            td_outcome = case_outcomes["td_lambda"]
            transitions = td_outcome.pop("_monte_carlo_transitions", None)
            if transitions is not None:
                baseline = case_outcomes[monte_carlo_baseline_method]
                update_started = time.perf_counter()
                errors = controller.monte_carlo_update(
                    transitions,
                    baseline_cost_sec=float(baseline["solve_runtime"]),
                )
                deferred_runtime = float(time.perf_counter() - update_started)
                td_outcome["update_runtime"] = float(
                    td_outcome.get("update_runtime", 0.0) + deferred_runtime
                )
                td_outcome["infer_runtime"] = float(
                    td_outcome.get("infer_runtime", 0.0) + deferred_runtime
                )
                if errors:
                    td_outcome["mean_abs_td_error"] = float(np.mean(np.abs(errors)))
        for method in methods:
            results[method].append(case_outcomes[method])
        transition_budget_reached = bool(
            transition_budget is not None and controller.steps >= int(transition_budget)
        )
        episode_budget_reached = bool(
            episode_budget is not None and controller.episodes >= int(episode_budget)
        )
        budget_reached = bool(transition_budget_reached or episode_budget_reached)
        if (index + 1) % max(1, progress_every) == 0 or index + 1 == len(trace) or budget_reached:
            print(
                json.dumps(
                    {
                        "stage": progress_stage,
                        "done": index + 1,
                        "total": len(trace),
                        "episodes": controller.episodes,
                        "steps": controller.steps,
                        "epsilon": controller.epsilon,
                        "transition_budget": transition_budget,
                        "episode_budget": episode_budget,
                    }
                ),
                flush=True,
            )
        if budget_reached:
            break
    return results


def _action_diagnostics(rows: Sequence[Dict[str, Any]], weights: Sequence[float]) -> Dict[str, Any]:
    counts = Counter()
    by_cycle: Dict[int, Counter] = {}
    for row in rows:
        for cycle, weight in enumerate(row.get("cycle_actions", [])):
            counts[float(weight)] += 1
            by_cycle.setdefault(int(cycle), Counter())[float(weight)] += 1
    modal_weight = float(counts.most_common(1)[0][0]) if counts else float("nan")
    return {
        "weights": [float(weight) for weight in weights],
        "counts": {str(weight): int(counts.get(float(weight), 0)) for weight in weights},
        "modal_weight": modal_weight,
        "by_cycle": {
            str(cycle): {str(weight): int(count) for weight, count in sorted(cycle_counts.items())}
            for cycle, cycle_counts in sorted(by_cycle.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent per-cycle Expected SARSA(lambda) on a frozen setup.")
    parser.add_argument("--trace-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--train-cases", type=int, default=1024)
    parser.add_argument("--eval-cases", type=int, default=96)
    parser.add_argument("--matrix-grid-n", "--grid-n", dest="grid_n", type=int, default=30)
    parser.add_argument("--c-min", type=float, default=1.0)
    parser.add_argument("--c-max", type=float, default=1000.0)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--weights", default="1.4,1.5,1.6")
    parser.add_argument("--anchor-weight", type=float, default=1.4)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--td-decay-power", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--trace-lambda", type=float, default=0.8)
    parser.add_argument("--epsilon-start", type=float, default=0.20)
    parser.add_argument("--epsilon-final", type=float, default=0.01)
    parser.add_argument("--epsilon-decay-steps", type=float, default=4000.0)
    parser.add_argument("--potential-scale-sec", type=float, default=0.0)
    parser.add_argument("--failure-penalty-sec", type=float, default=0.05)
    parser.add_argument("--initial-q-sec", type=float, default=0.02)
    parser.add_argument("--monte-carlo-alpha", type=float, default=0.0)
    parser.add_argument("--monte-carlo-decay-power", type=float, default=0.0)
    parser.add_argument("--adaptive-cycles", type=int, default=2)
    parser.add_argument("--state-mode", choices=("full", "cycle_tabular"), default="full")
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--skip-training-baselines", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixed_params, dominant_count, trace_cases = _load_dominant_setup(args.trace_result.resolve())
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
        potential_scale_sec=float(args.potential_scale_sec),
        failure_penalty_sec=float(args.failure_penalty_sec),
        initial_q_sec=float(args.initial_q_sec),
        monte_carlo_alpha=float(args.monte_carlo_alpha),
        monte_carlo_decay_power=float(args.monte_carlo_decay_power),
        adaptive_cycles=int(args.adaptive_cycles),
    )
    encoder = SolveStateEncoder(
        tol=float(args.tol),
        max_cycles=int(args.max_cycles),
        c_max=float(args.c_max),
        mode=str(args.state_mode),
    )
    controller = ExpectedSarsaLambda(
        feature_dim=encoder.feature_dim,
        config=config,
        seed=args.seed,
        initial_parameters=encoder.constant_value_parameters(config.initial_q_sec),
    )
    protocol = {
        "git_revision": _git_revision(),
        "platform": platform.platform(),
        "seed": int(args.seed),
        "trace_result": str(args.trace_result.resolve()),
        "fixed_setup_params": fixed_params,
        "dominant_setup_count": int(dominant_count),
        "dominant_setup_trace_cases": int(trace_cases),
        "train_cases": int(args.train_cases),
        "eval_cases": int(args.eval_cases),
        "grid": [int(args.grid_n)] * 3,
        "tol": float(args.tol),
        "max_cycles": int(args.max_cycles),
        "feature_dim": int(encoder.feature_dim),
        "state_mode": str(args.state_mode),
        "controller": config.__dict__,
    }
    _write_json(args.output_dir / "config.json", protocol)
    print(json.dumps(_json_ready({"stage": "td_start", **protocol})), flush=True)

    train_instances = generate_difconv_instances(
        T=int(args.train_cases),
        seed=int(args.seed),
        grid_choices=[(int(args.grid_n),) * 3],
        c_min=float(args.c_min),
        c_max=float(args.c_max),
    )
    train_methods = (
        ("td_lambda",)
        if args.skip_training_baselines
        else ("default_solve", "fixed_w1.4", "td_lambda")
    )
    training = _run_stream(
        instances=train_instances,
        params=fixed_params,
        controller=controller,
        encoder=encoder,
        tol=args.tol,
        max_cycles=args.max_cycles,
        methods=train_methods,
        learn=True,
        explore=True,
        seed=args.seed + 1,
        progress_stage="td_train_progress",
        progress_every=args.progress_every,
    )
    controller.save(args.output_dir / "td_lambda_controller.npz")
    frozen_controller = copy.deepcopy(controller)

    eval_instances = generate_difconv_instances(
        T=int(args.eval_cases),
        seed=int(args.seed + 100_000),
        grid_choices=[(int(args.grid_n),) * 3],
        c_min=float(args.c_min),
        c_max=float(args.c_max),
    )
    evaluation = _run_stream(
        instances=eval_instances,
        params=fixed_params,
        controller=frozen_controller,
        encoder=encoder,
        tol=args.tol,
        max_cycles=args.max_cycles,
        methods=("default_solve", "fixed_w1.0", "fixed_w1.4", "td_lambda"),
        learn=False,
        explore=False,
        seed=args.seed + 100_001,
        progress_stage="td_eval_progress",
        progress_every=args.progress_every,
    )
    diagnostics = _action_diagnostics(evaluation["td_lambda"], weights)
    modal_weight = float(diagnostics["modal_weight"])
    modal_reference_key: str | None = None
    if np.isfinite(modal_weight):
        fixed_key = f"fixed_w{modal_weight:.1f}"
        if fixed_key in evaluation:
            modal_reference_key = fixed_key
        else:
            modal_reference_key = f"modal_fixed_w{modal_weight:.3f}"
            evaluation[modal_reference_key] = [
                solve_fixed_w_case(
                    params=dict(fixed_params),
                    mkw=dict(mkw),
                    w=modal_weight,
                    sweeps_down=1,
                    sweeps_up=1,
                    solve_tol=float(args.tol),
                    solve_max_cycles=int(args.max_cycles),
                )
                for mkw, _context in eval_instances
            ]

    training_summaries = {name: _summarize(rows) for name, rows in training.items()}
    evaluation_summaries = {name: _summarize(rows) for name, rows in evaluation.items()}
    comparisons = {
        f"td_lambda_vs_{reference}": _paired(
            evaluation[reference],
            evaluation["td_lambda"],
            seed=args.seed + 200_000 + index * 2,
        )
        for index, reference in enumerate(("default_solve", "fixed_w1.0", "fixed_w1.4"))
    }
    if modal_reference_key is not None:
        comparisons["td_lambda_vs_modal_fixed"] = _paired(
            evaluation[modal_reference_key],
            evaluation["td_lambda"],
            seed=args.seed + 200_010,
        )
    training_comparisons = {}
    if "default_solve" in training:
        training_comparisons["td_lambda_vs_default_solve"] = _paired(
            training["default_solve"], training["td_lambda"], seed=args.seed + 300_000
        )
    if "fixed_w1.4" in training:
        training_comparisons["td_lambda_vs_fixed_w1.4"] = _paired(
            training["fixed_w1.4"], training["td_lambda"], seed=args.seed + 300_002
        )
    payload = {
        "protocol": protocol,
        "training": {
            "methods": training_summaries,
            "action_diagnostics": _action_diagnostics(training["td_lambda"], weights),
            "comparisons": training_comparisons,
            "first_100": _summarize(training["td_lambda"][: min(100, len(training["td_lambda"]))]),
            "last_100": _summarize(training["td_lambda"][-min(100, len(training["td_lambda"])) :]),
        },
        "evaluation": {
            "methods": evaluation_summaries,
            "comparisons": comparisons,
            "action_diagnostics": diagnostics,
        },
        "controller": {
            "steps": int(controller.steps),
            "episodes": int(controller.episodes),
            "epsilon": float(controller.epsilon),
            "theta": controller.theta.tolist(),
        },
    }
    _write_json(args.output_dir / "result.json", payload)
    print(
        json.dumps(
            _json_ready(
                {
                    "stage": "td_done",
                    "output": args.output_dir / "result.json",
                    "comparisons": comparisons,
                    "actions": diagnostics,
                }
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
