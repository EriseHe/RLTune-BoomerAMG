"""Replay one recorded solve trace into identical LSTDQ-v3 controllers.

The recorded cycle observations are treated as an immutable shared input.  No
HYPRE work is executed, so this diagnostic separates controller determinism
from native setup/solve timing variation.
"""

from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from online_td_experiment_common import _json_ready, _write_json
from run_solve_controller_calibration import _build_bundle
from solve.controllers.recursive_lstdq import RecursiveLstdqV3LcbSpec


def _assert_identical(label: str, values: Sequence[Any]) -> None:
    reference = values[0]
    for replica, value in enumerate(values[1:], start=2):
        if isinstance(reference, np.ndarray):
            if not isinstance(value, np.ndarray) or not np.array_equal(
                reference,
                value,
                equal_nan=True,
            ):
                raise AssertionError(f"{label} differs in replica {replica}")
        elif isinstance(reference, dict):
            if not isinstance(value, dict) or reference.keys() != value.keys():
                raise AssertionError(f"{label} keys differ in replica {replica}")
            for key in reference:
                _assert_identical(
                    f"{label}.{key}",
                    (reference[key], value[key]),
                )
        elif isinstance(reference, (list, tuple)):
            if not isinstance(value, type(reference)) or len(reference) != len(value):
                raise AssertionError(f"{label} shape differs in replica {replica}")
            for index, item in enumerate(reference):
                _assert_identical(
                    f"{label}[{index}]",
                    (item, value[index]),
                )
        elif isinstance(reference, float) and math.isnan(reference):
            if not isinstance(value, float) or not math.isnan(value):
                raise AssertionError(f"{label} differs in replica {replica}")
        elif reference != value:
            raise AssertionError(f"{label} differs in replica {replica}")


def _controller_state(controller: Any) -> dict[str, Any]:
    state = controller.snapshot_learning_state()
    state["rng_state"] = controller.rng.bit_generator.state
    return state


def _features_for_row(bundle: Any, row: dict[str, Any]) -> list[np.ndarray]:
    outcome = row["outcome"]
    residuals = tuple(float(value) for value in outcome["cycle_residuals"])
    ratios = tuple(float(value) for value in outcome["cycle_residual_ratios"])
    times = tuple(float(value) for value in outcome["cycle_times"])
    weights = tuple(float(value) for value in outcome["cycle_actions"])
    if not residuals or not (
        len(residuals) == len(ratios) == len(times) == len(weights)
    ):
        raise ValueError("trace row does not contain a complete successful episode")
    initial_residual = residuals[0] / ratios[0]
    initial_weight = float(outcome["initial_environment_weight"])
    features: list[np.ndarray] = []
    for cycle in range(len(residuals)):
        residual = initial_residual if cycle == 0 else residuals[cycle - 1]
        previous_residual = (
            initial_residual if cycle <= 1 else residuals[cycle - 2]
        )
        features.append(
            bundle.encoder.encode(
                mkw=row["mkw"],
                setup_params=row["params"],
                initial_residual=initial_residual,
                residual=residual,
                previous_residual=previous_residual,
                cycle=cycle,
                last_weight=(initial_weight if cycle == 0 else weights[cycle - 1]),
                last_cycle_time=(0.0 if cycle == 0 else times[cycle - 1]),
            )
        )
    return features


def _load_successful_rows(path: Path, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            outcome = row.get("outcome", {})
            if (
                not outcome.get("failed", True)
                and len(outcome.get("cycle_residuals", ()))
                == len(outcome.get("cycle_residual_ratios", ()))
            ):
                rows.append(row)
                if len(rows) == count:
                    break
    if len(rows) != count:
        raise ValueError(f"only found {len(rows)} complete successful trace rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=20)
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--controller-seed", type=int, default=41866039)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument("--c-max", type=float, default=1000.0)
    args = parser.parse_args()
    if args.cases <= 0 or args.replicas < 2:
        raise ValueError("cases must be positive and replicas must be at least two")

    bundles = [
        _build_bundle(
            args=args,
            kind="recursive_lstdq_v3",
            algorithm=RecursiveLstdqV3LcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
                q_max_sec=0.1,
                inverse_denominator_floor=1.0e-10,
                lcb_lower_bound_sec=0.0,
            ),
            trace_lambda=0.8,
        )
        for _ in range(args.replicas)
    ]
    rows = _load_successful_rows(args.trace.resolve(), args.cases)
    transitions = 0
    selected_actions: list[list[int]] = []

    for episode, row in enumerate(rows):
        feature_sets = [_features_for_row(bundle, row) for bundle in bundles]
        _assert_identical(f"episode[{episode}].features", feature_sets)
        initial_weight = float(row["outcome"]["initial_environment_weight"])
        for bundle in bundles:
            bundle.controller.start_episode(
                initial_environment_weight=initial_weight,
            )

        costs = tuple(float(value) for value in row["outcome"]["cycle_times"])
        pending: list[tuple[int, float, dict[str, Any]]] | None = None
        episode_actions: list[int] = []
        for cycle, features in enumerate(feature_sets[0]):
            selections = pending or [
                bundle.controller.select_action(
                    feature_sets[index][cycle],
                    explore=True,
                    epsilon=None,
                    cycle=cycle,
                    include_metadata=True,
                )
                for index, bundle in enumerate(bundles)
            ]
            pending = None
            _assert_identical(
                f"episode[{episode}].cycle[{cycle}].selection",
                selections,
            )
            action_index = int(selections[0][0])
            episode_actions.append(action_index)
            terminal = cycle + 1 == len(feature_sets[0])
            next_features = features if terminal else feature_sets[0][cycle + 1]
            next_action_index = None
            if not terminal:
                pending = [
                    bundle.controller.select_action(
                        feature_sets[index][cycle + 1],
                        explore=True,
                        epsilon=None,
                        cycle=cycle + 1,
                        include_metadata=True,
                    )
                    for index, bundle in enumerate(bundles)
                ]
                _assert_identical(
                    f"episode[{episode}].cycle[{cycle}].next_selection",
                    pending,
                )
                next_action_index = int(pending[0][0])

            td_errors = [
                bundle.controller.update(
                    features=feature_sets[index][cycle],
                    action_index=action_index,
                    cost=costs[cycle],
                    next_features=(
                        feature_sets[index][cycle]
                        if terminal
                        else feature_sets[index][cycle + 1]
                    ),
                    terminal=terminal,
                    next_cycle=cycle + 1,
                    next_action_index=next_action_index,
                    native_cycle_cost=costs[cycle],
                    residual_ratio=float(
                        row["outcome"]["cycle_residual_ratios"][cycle]
                    ),
                )
                for index, bundle in enumerate(bundles)
            ]
            _assert_identical(
                f"episode[{episode}].cycle[{cycle}].td_error",
                td_errors,
            )
            _assert_identical(
                f"episode[{episode}].cycle[{cycle}].state",
                [_controller_state(bundle.controller) for bundle in bundles],
            )
            transitions += 1

        for bundle in bundles:
            bundle.controller.finish_episode(learned=True)
        _assert_identical(
            f"episode[{episode}].committed_state",
            [_controller_state(bundle.controller) for bundle in bundles],
        )
        selected_actions.append(episode_actions)

    result = {
        "status": "exact_match",
        "trace": str(args.trace.resolve()),
        "cases": len(rows),
        "transitions": transitions,
        "replicas": args.replicas,
        "controller_seed": args.controller_seed,
        "episodes": bundles[0].controller.episodes,
        "steps": bundles[0].controller.steps,
        "episode_moment_count": bundles[0].controller.episode_moment_count,
        "selected_action_indices": selected_actions,
    }
    _write_json(args.output.resolve(), result)
    print(json.dumps(_json_ready(result), indent=2), flush=True)


if __name__ == "__main__":
    main()
