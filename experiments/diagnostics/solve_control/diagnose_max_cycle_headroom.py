"""Replay 50-cycle nonconvergence cases with a larger cycle budget.

For staged solve-controller runs, the first cohort uses the native BoomerAMG
solve and the active cohort uses one-cycle RL stepping.  The two interfaces
report different residual conventions, so this diagnostic replays each case
through the same interface that originally produced it instead of attempting
to infer convergence from residual values alone.
"""

from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence, Tuple

import numpy as np

from experiments.joint.solve_control.online_td_experiment_common import _json_ready, _write_json
from experiments.joint.solve_control.joint_experiment_plotting import _read_json_lines
from hypre.bindings import augment_setup_params
from experiments.joint.solve_control.native_evaluation import solve_no_rl_case, solve_schedule_case


DEFAULT_METHOD = "default_setup_default_solve"
MAX_ITER_REASON = "max_iter_reached_without_convergence"
MAX_CYCLES_REASON = "max_cycles_reached_without_convergence"


def recorded_action_tail_schedule(
    actions: Sequence[float],
    extended_max_cycles: int,
) -> Tuple[Tuple[int, float, int, int], ...]:
    """Compress recorded weights, then hold the last weight to the new cap."""

    if not actions:
        raise ValueError("recorded action sequence must not be empty")
    if extended_max_cycles < len(actions):
        raise ValueError("extended_max_cycles cannot truncate recorded actions")

    schedule = []
    previous = float(actions[0])
    for index, action in enumerate(actions[1:], start=1):
        action = float(action)
        if action != previous:
            schedule.append((index, previous, 1, 1))
            previous = action
    schedule.append((len(actions), previous, 1, 1))
    if extended_max_cycles > len(actions):
        schedule.append((extended_max_cycles, previous, 1, 1))
    return tuple(schedule)


def _trajectory_path(result_dir: Path, method: str) -> Path:
    return result_dir / "trajectories" / f"{method}.jsonl"


def _primary_reason(row: Dict[str, Any]) -> str:
    outcome = row.get("outcome", {})
    return str(
        outcome.get("primary_failure_reason")
        or outcome.get("failure_reason")
        or ""
    )


def _limited(rows: Iterable[Dict[str, Any]], limit: int | None) -> list[Dict[str, Any]]:
    selected = list(rows)
    return selected if limit is None else selected[:limit]


def _tail_geometric_ratio(residuals: Sequence[float], window: int = 10) -> float | None:
    positive = np.asarray(
        [float(value) for value in residuals if np.isfinite(value) and value > 0.0],
        dtype=np.float64,
    )
    if positive.size < 2:
        return None
    positive = positive[-(window + 1) :]
    ratios = positive[1:] / positive[:-1]
    return float(np.exp(np.mean(np.log(ratios))))


def _initial_residual_from_trace(outcome: Dict[str, Any]) -> float:
    residuals = outcome.get("cycle_residuals", [])
    ratios = outcome.get("cycle_residual_ratios", [])
    if not residuals or not ratios or float(ratios[0]) <= 0.0:
        raise ValueError("active replay needs the first residual and residual ratio")
    initial_residual = float(residuals[0]) / float(ratios[0])
    if not np.isfinite(initial_residual) or initial_residual <= 0.0:
        raise ValueError("reconstructed initial residual must be positive and finite")
    return initial_residual


def _iteration_bucket(
    *,
    failed: bool,
    iterations: int,
    original_cap: int,
    extended_cap: int,
) -> str:
    if failed:
        return "still_failed_at_extended_cap"
    if iterations <= original_cap:
        return "converged_by_original_cap_on_replay"
    if iterations <= original_cap + 10:
        return f"{original_cap + 1}-{original_cap + 10}"
    midpoint = original_cap + (extended_cap - original_cap) // 2
    if iterations <= midpoint:
        return f"{original_cap + 11}-{midpoint}"
    return f"{midpoint + 1}-{extended_cap}"


def _replay_record(
    *,
    row: Dict[str, Any],
    replay: Dict[str, Any],
    cohort: str,
    requested_tolerance: float,
    comparison_tolerance: float,
    original_cap: int,
    extended_cap: int,
    residual_convention: str,
) -> Dict[str, Any]:
    outcome = row["outcome"]
    original_primary_solve = float(outcome.get("primary_solve_runtime", 0.0))
    fallback_native = float(outcome.get("fallback_setup_runtime", 0.0)) + float(
        outcome.get("fallback_solve_runtime", 0.0)
    )
    converged = not bool(replay.get("failed", True))
    replay_solve = float(replay.get("solve_runtime", 0.0))
    estimated_delta = replay_solve - original_primary_solve
    if converged and bool(outcome.get("fallback_used", False)):
        estimated_delta -= fallback_native

    iterations = int(replay.get("iterations", 0))
    original_residual = float(
        outcome.get("primary_residual_norm", outcome.get("residual_norm", math.nan))
    )
    replay_residual = float(replay.get("residual_norm", math.nan))
    residuals = outcome.get("cycle_residuals", [])
    return {
        "cohort": cohort,
        "online_index": int(row["online_index"]),
        "stream_index": int(row["stream_index"]),
        "arm_index": row.get("arm_index"),
        "original_primary_reason": _primary_reason(row),
        "original_primary_cycles": int(outcome.get("primary_cycles", original_cap)),
        "original_primary_residual_norm": original_residual,
        "original_residual_over_tolerance": (
            original_residual / comparison_tolerance
            if np.isfinite(original_residual)
            else None
        ),
        "original_primary_solve_runtime_sec": original_primary_solve,
        "original_fallback_used": bool(outcome.get("fallback_used", False)),
        "original_recovered": bool(outcome.get("recovered", False)),
        "original_fallback_native_runtime_sec": fallback_native,
        "recorded_tail_geometric_residual_ratio": _tail_geometric_ratio(residuals),
        "residual_convention": residual_convention,
        "requested_tolerance": requested_tolerance,
        "effective_comparison_tolerance": comparison_tolerance,
        "extended_converged": converged,
        "extended_status": str(replay.get("native_status", "")),
        "extended_failure_reason": str(replay.get("failure_reason", "")),
        "extended_iterations": iterations,
        "extended_residual_norm": replay_residual,
        "extended_residual_over_tolerance": (
            replay_residual / comparison_tolerance
            if np.isfinite(replay_residual)
            else None
        ),
        "extended_setup_runtime_sec": float(replay.get("setup_runtime", 0.0)),
        "extended_solve_runtime_sec": replay_solve,
        "additional_cycles_if_converged": (
            max(0, iterations - original_cap) if converged else None
        ),
        "iteration_bucket": _iteration_bucket(
            failed=not converged,
            iterations=iterations,
            original_cap=original_cap,
            extended_cap=extended_cap,
        ),
        "estimated_native_runtime_delta_vs_original_protocol_sec": estimated_delta,
        "params": row["params"],
        "mkw": row["mkw"],
    }


def summarize_replays(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate replay outcomes for a single protocol cohort."""

    count = len(rows)
    rescued = [row for row in rows if bool(row["extended_converged"])]
    buckets: Dict[str, int] = {}
    for row in rows:
        bucket = str(row["iteration_bucket"])
        buckets[bucket] = buckets.get(bucket, 0) + 1

    rescued_iterations = np.asarray(
        [int(row["extended_iterations"]) for row in rescued],
        dtype=np.int64,
    )
    deltas = np.asarray(
        [
            float(row["estimated_native_runtime_delta_vs_original_protocol_sec"])
            for row in rows
        ],
        dtype=np.float64,
    )
    tail_ratios = np.asarray(
        [
            float(row["recorded_tail_geometric_residual_ratio"])
            for row in rows
            if row.get("recorded_tail_geometric_residual_ratio") is not None
        ],
        dtype=np.float64,
    )
    return {
        "cases_at_original_cap": count,
        "converged_by_extended_cap": len(rescued),
        "still_failed_at_extended_cap": count - len(rescued),
        "rescue_rate": (len(rescued) / count) if count else 0.0,
        "iteration_buckets": buckets,
        "rescued_iteration_median": (
            float(np.median(rescued_iterations)) if rescued_iterations.size else None
        ),
        "rescued_iteration_p90": (
            float(np.percentile(rescued_iterations, 90))
            if rescued_iterations.size
            else None
        ),
        "original_recovered_count": sum(
            bool(row["original_recovered"]) for row in rows
        ),
        "original_unrecovered_count": sum(
            not bool(row["original_recovered"]) for row in rows
        ),
        "estimated_native_runtime_delta_total_sec": (
            float(np.sum(deltas)) if deltas.size else 0.0
        ),
        "estimated_native_runtime_delta_mean_ms_per_cap_event": (
            float(np.mean(deltas) * 1_000.0) if deltas.size else 0.0
        ),
        "recorded_tail_geometric_residual_ratio_median": (
            float(np.median(tail_ratios)) if tail_ratios.size else None
        ),
    }


def _print_progress(
    *,
    cohort: str,
    completed: int,
    total: int,
    converged: int,
) -> None:
    print(
        json.dumps(
            {
                "cohort": cohort,
                "completed": completed,
                "total": total,
                "converged_by_extended_cap": converged,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def _replay_no_rl_cohort(
    *,
    rows: Sequence[Dict[str, Any]],
    cohort: str,
    tolerance: float,
    original_cap: int,
    extended_cap: int,
    progress_every: int,
) -> list[Dict[str, Any]]:
    records = []
    for index, row in enumerate(rows, start=1):
        replay = solve_no_rl_case(
            params=dict(row["params"]),
            mkw=dict(row["mkw"]),
            solver_tol=tolerance,
            solver_max_iter=extended_cap,
            augment_params=augment_setup_params,
        )
        records.append(
            _replay_record(
                row=row,
                replay=replay,
                cohort=cohort,
                requested_tolerance=tolerance,
                comparison_tolerance=tolerance,
                original_cap=original_cap,
                extended_cap=extended_cap,
                residual_convention="relative",
            )
        )
        if index % progress_every == 0 or index == len(rows):
            _print_progress(
                cohort=cohort,
                completed=index,
                total=len(rows),
                converged=sum(
                    bool(record["extended_converged"]) for record in records
                ),
            )
    return records


def _replay_active_cohort(
    *,
    rows: Sequence[Dict[str, Any]],
    tolerance: float,
    original_cap: int,
    extended_cap: int,
    progress_every: int,
    tolerance_mode: str,
) -> list[Dict[str, Any]]:
    cohort = "learned_active_recorded_schedule_fixed_tail"
    records = []
    for index, row in enumerate(rows, start=1):
        actions = [float(value) for value in row["outcome"]["cycle_actions"]]
        schedule = recorded_action_tail_schedule(actions, extended_cap)
        comparison_tolerance = float(tolerance)
        residual_convention = "absolute"
        if tolerance_mode == "relative":
            comparison_tolerance *= _initial_residual_from_trace(row["outcome"])
            residual_convention = (
                "absolute residual checked against requested relative tolerance "
                "times reconstructed initial residual"
            )
        replay = solve_schedule_case(
            params=dict(row["params"]),
            mkw=dict(row["mkw"]),
            schedule=schedule,
            solve_tol=comparison_tolerance,
            solve_max_cycles=extended_cap,
        )
        records.append(
            _replay_record(
                row=row,
                replay=replay,
                cohort=cohort,
                requested_tolerance=tolerance,
                comparison_tolerance=comparison_tolerance,
                original_cap=original_cap,
                extended_cap=extended_cap,
                residual_convention=residual_convention,
            )
        )
        if index % progress_every == 0 or index == len(rows):
            _print_progress(
                cohort=cohort,
                completed=index,
                total=len(rows),
                converged=sum(
                    bool(record["extended_converged"]) for record in records
                ),
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--learned-method", required=True)
    parser.add_argument("--tolerance", type=float, required=True)
    parser.add_argument("--activation-case", type=int, default=1_000)
    parser.add_argument("--original-max-cycles", type=int, default=50)
    parser.add_argument("--extended-max-cycles", type=int, default=100)
    parser.add_argument("--limit-per-cohort", type=int)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--rl-tolerance-mode",
        choices=("absolute", "relative"),
        default="absolute",
    )
    parser.add_argument("--active-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if args.original_max_cycles <= 0:
        raise ValueError("original-max-cycles must be positive")
    if args.extended_max_cycles <= args.original_max_cycles:
        raise ValueError("extended-max-cycles must exceed original-max-cycles")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    if args.limit_per_cohort is not None and args.limit_per_cohort <= 0:
        raise ValueError("limit-per-cohort must be positive")

    result_dir = args.result_dir.resolve()
    default_path = _trajectory_path(result_dir, DEFAULT_METHOD)
    learned_path = _trajectory_path(result_dir, args.learned_method)
    default_rows = _read_json_lines(default_path)
    learned_rows = _read_json_lines(learned_path)

    default_cap_rows = _limited(
        (
            row
            for row in default_rows
            if _primary_reason(row) == MAX_ITER_REASON
        ),
        args.limit_per_cohort,
    )
    learned_prefix_rows = _limited(
        (
            row
            for row in learned_rows
            if int(row["online_index"]) < args.activation_case
            and _primary_reason(row) == MAX_ITER_REASON
        ),
        args.limit_per_cohort,
    )
    learned_active_rows = _limited(
        (
            row
            for row in learned_rows
            if int(row["online_index"]) >= args.activation_case
            and _primary_reason(row) == MAX_CYCLES_REASON
        ),
        args.limit_per_cohort,
    )

    default_records = []
    learned_prefix_records = []
    if not args.active_only:
        default_records = _replay_no_rl_cohort(
            rows=default_cap_rows,
            cohort="default_baseline",
            tolerance=args.tolerance,
            original_cap=args.original_max_cycles,
            extended_cap=args.extended_max_cycles,
            progress_every=args.progress_every,
        )
        learned_prefix_records = _replay_no_rl_cohort(
            rows=learned_prefix_rows,
            cohort="learned_prefix_default_solve",
            tolerance=args.tolerance,
            original_cap=args.original_max_cycles,
            extended_cap=args.extended_max_cycles,
            progress_every=args.progress_every,
        )
    learned_active_records = _replay_active_cohort(
        rows=learned_active_rows,
        tolerance=args.tolerance,
        original_cap=args.original_max_cycles,
        extended_cap=args.extended_max_cycles,
        progress_every=args.progress_every,
        tolerance_mode=args.rl_tolerance_mode,
    )

    cohorts = {
        "default_baseline": default_records,
        "learned_prefix_default_solve": learned_prefix_records,
        "learned_active_recorded_schedule_fixed_tail": learned_active_records,
    }
    payload = {
        "diagnostic": "max_cycle_headroom",
        "source_result_dir": str(result_dir),
        "source_trajectories": {
            "default": str(default_path),
            "learned": str(learned_path),
        },
        "protocol": {
            "tolerance": args.tolerance,
            "activation_case": args.activation_case,
            "original_max_cycles": args.original_max_cycles,
            "extended_max_cycles": args.extended_max_cycles,
            "rl_tolerance_mode": args.rl_tolerance_mode,
            "learned_active_extension": (
                "Replay all 50 recorded actions exactly, then hold the final "
                "recorded relaxation weight through cycle 100."
            ),
            "residual_conventions": {
                "default_baseline": "native BoomerAMG relative residual norm",
                "learned_prefix_default_solve": (
                    "native BoomerAMG relative residual norm"
                ),
                "learned_active_recorded_schedule_fixed_tail": (
                    "RL-step absolute residual norm"
                    if args.rl_tolerance_mode == "absolute"
                    else (
                        "requested relative tolerance converted per case to "
                        "tol * reconstructed initial residual"
                    )
                ),
            },
            "timing_note": (
                "Runtime deltas compare the new replay timing with original "
                "primary/fallback timings and are approximate; convergence "
                "counts and iteration counts are the primary evidence."
            ),
            "limit_per_cohort": args.limit_per_cohort,
        },
        "summary": {
            name: summarize_replays(records)
            for name, records in cohorts.items()
        },
        "records": cohorts,
    }
    _write_json(args.output.resolve(), payload)
    print(
        json.dumps(
            _json_ready(
                {
                    "output": str(args.output.resolve()),
                    "summary": payload["summary"],
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
