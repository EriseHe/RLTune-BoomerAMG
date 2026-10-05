from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np

from experiments.joint.solve_control.setup_aware_compare_common import solve_schedule_case


def _parse_values(raw: str, cast: Any) -> tuple[Any, ...]:
    return tuple(cast(part.strip()) for part in raw.split(",") if part.strip())


def _load_trace(path: Path, max_cases: int | None) -> list[tuple[Dict[str, Any], Dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload["evaluation"]["per_case"]
    if max_cases is not None:
        cases = cases[:max_cases]
    return [(dict(case["mkw"]), dict(case["params"])) for case in cases]


def _schedule_key(schedule: Sequence[tuple[int, float, int, int]]) -> str:
    return "__".join(f"until{end}_w{weight:.3f}" for end, weight, _sd, _su in schedule)


def _build_schedules(
    *,
    weights: Sequence[float],
    switch_cycles: Sequence[int],
    max_cycles: int,
) -> Dict[str, list[tuple[int, float, int, int]]]:
    schedules: Dict[str, list[tuple[int, float, int, int]]] = {}
    for weight in weights:
        schedule = [(int(max_cycles), float(weight), 1, 1)]
        schedules[f"fixed_w{weight:.3f}"] = schedule
    for switch_cycle in switch_cycles:
        for first_weight in weights:
            for tail_weight in weights:
                if first_weight == tail_weight:
                    continue
                schedule = [
                    (int(switch_cycle), float(first_weight), 1, 1),
                    (int(max_cycles), float(tail_weight), 1, 1),
                ]
                schedules[_schedule_key(schedule)] = schedule
    return schedules


def _bootstrap_mean_interval(values: np.ndarray, *, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    means = np.empty(2000, dtype=float)
    for index in range(means.size):
        sample = rng.integers(0, values.size, size=values.size)
        means[index] = float(np.mean(values[sample]))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    solve = np.asarray([float(row["solve_runtime"]) for row in rows], dtype=float)
    total = np.asarray([float(row["runtime"]) for row in rows], dtype=float)
    iterations = np.asarray([int(row["iterations"]) for row in rows], dtype=float)
    return {
        "observations": int(len(rows)),
        "failed_count": int(sum(bool(row.get("failed", False)) for row in rows)),
        "mean_solve_runtime_sec": float(np.mean(solve)),
        "mean_total_runtime_sec": float(np.mean(total)),
        "mean_iterations": float(np.mean(iterations)),
    }


def _paired_stats(
    reference: Sequence[Dict[str, Any]],
    candidate: Sequence[Dict[str, Any]],
    *,
    seed: int,
) -> Dict[str, Any]:
    reference_solve = np.asarray([float(row["solve_runtime"]) for row in reference], dtype=float)
    candidate_solve = np.asarray([float(row["solve_runtime"]) for row in candidate], dtype=float)
    improvement = 100.0 * (reference_solve - candidate_solve) / reference_solve
    return {
        "mean_solve_improvement_pct": float(np.mean(improvement)),
        "median_solve_improvement_pct": float(np.median(improvement)),
        "win_rate": float(np.mean(candidate_solve < reference_solve)),
        "bootstrap_mean_95pct": _bootstrap_mean_interval(improvement, seed=seed),
    }


def _top_rows(
    schedules: Dict[str, Sequence[tuple[int, float, int, int]]],
    results: Dict[str, Sequence[Dict[str, Any]]],
    reference_key: str,
    *,
    seed: int,
    limit: int,
) -> list[Dict[str, Any]]:
    rows = []
    for index, (key, outcomes) in enumerate(results.items()):
        summary = _summarize(outcomes)
        rows.append(
            {
                "name": key,
                "schedule": [list(item) for item in schedules[key]],
                **summary,
                "vs_reference": _paired_stats(
                    results[reference_key],
                    outcomes,
                    seed=seed + index,
                ),
            }
        )
    return sorted(rows, key=lambda row: float(row["mean_solve_runtime_sec"]))[:limit]


def _oracle_summary(results: Dict[str, Sequence[Dict[str, Any]]]) -> Dict[str, Any]:
    method_names = tuple(results)
    rows = []
    selections: Dict[str, int] = {}
    for case_index in range(len(next(iter(results.values())))):
        name = min(method_names, key=lambda key: float(results[key][case_index]["solve_runtime"]))
        selections[name] = int(selections.get(name, 0) + 1)
        row = dict(results[name][case_index])
        row["selected_schedule"] = name
        rows.append(row)
    return {
        **_summarize(rows),
        "selection_counts": dict(sorted(selections.items(), key=lambda item: item[1], reverse=True)),
    }


def _iter_progress(total: int, every: int) -> Iterable[bool]:
    for index in range(total):
        yield (index + 1) % every == 0 or index + 1 == total


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure per-cycle solve-control headroom on a frozen setup trace.")
    parser.add_argument("--trace-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", default="1.0,1.2,1.4,1.5,1.6")
    parser.add_argument("--switch-cycles", default="1,2,4,6,8")
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    trace = _load_trace(args.trace_result.resolve(), args.max_cases)
    weights = _parse_values(args.weights, float)
    switch_cycles = _parse_values(args.switch_cycles, int)
    schedules = _build_schedules(weights=weights, switch_cycles=switch_cycles, max_cycles=args.max_cycles)
    reference_key = "fixed_w1.400"
    if reference_key not in schedules:
        raise ValueError("--weights must include 1.4 for the required reference comparison")

    results: Dict[str, list[Dict[str, Any]]] = {key: [] for key in schedules}
    rng = np.random.default_rng(args.seed)
    total_cases = int(len(trace) * args.repeats)
    progress = _iter_progress(total_cases, max(1, args.progress_every))
    completed = 0
    for repeat in range(args.repeats):
        for mkw, params in trace:
            order = [tuple(schedules)[index] for index in rng.permutation(len(schedules))]
            for key in order:
                outcome = solve_schedule_case(
                    params=dict(params),
                    mkw=dict(mkw),
                    schedule=schedules[key],
                    solve_tol=args.tol,
                    solve_max_cycles=args.max_cycles,
                )
                outcome["repeat"] = int(repeat)
                results[key].append(outcome)
            completed += 1
            if next(progress):
                print(json.dumps({"stage": "headroom_progress", "done": completed, "total": total_cases}), flush=True)

    payload = {
        "protocol": {
            "trace_result": str(args.trace_result.resolve()),
            "cases": int(len(trace)),
            "repeats": int(args.repeats),
            "weights": list(weights),
            "switch_cycles": list(switch_cycles),
            "max_cycles": int(args.max_cycles),
            "tol": float(args.tol),
            "seed": int(args.seed),
        },
        "reference": {
            "name": reference_key,
            "schedule": [list(item) for item in schedules[reference_key]],
            **_summarize(results[reference_key]),
        },
        "top_schedules": _top_rows(
            schedules,
            results,
            reference_key,
            seed=args.seed + 10_000,
            limit=args.top,
        ),
        "per_case_oracle": _oracle_summary(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "headroom_done", "output": str(args.output), "top": payload["top_schedules"][:5]}), flush=True)


if __name__ == "__main__":
    main()
