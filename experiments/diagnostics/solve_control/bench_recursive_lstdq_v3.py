"""Microbenchmark recursive LSTDQ v2/v3 at production feature dimensions."""

from __future__ import annotations

if __package__ in {None, ""}:
    import _project_paths  # noqa: F401

import argparse
import json
import time
from typing import Any, Dict

import numpy as np

from solve.controllers.recursive_lstdq import (
    RecursiveLstdqV2LcbController,
    RecursiveLstdqV2LcbSpec,
    RecursiveLstdqV3LcbController,
    RecursiveLstdqV3LcbSpec,
)
from solve.controllers.common.td_config import ExpectedSarsaLambdaConfig


FEATURE_DIM = 32
WEIGHTS = tuple(float(value) for value in np.linspace(1.0, 3.0, 41))
CENTERS = tuple(float(value) for value in np.linspace(1.0, 3.0, 9))


def _config() -> ExpectedSarsaLambdaConfig:
    return ExpectedSarsaLambdaConfig(
        weights=WEIGHTS,
        anchor_weight=1.0,
        alpha=0.001,
        gamma=1.0,
        trace_lambda=0.8,
        epsilon_start=0.0,
        epsilon_final=0.0,
        epsilon_decay_steps=20_000.0,
        initial_q_sec=0.0,
        action_rbf_sigma=0.2,
        action_basis_mode="compact_rbf",
        action_basis_centers=CENTERS,
        td_algorithm="true_online_sarsa",
        force_default_first_action=False,
    )


def _controllers() -> Dict[str, Any]:
    config = _config()
    return {
        "v2": RecursiveLstdqV2LcbController(
            feature_dim=FEATURE_DIM,
            config=config,
            spec=RecursiveLstdqV2LcbSpec(
                ridge=1.0,
                uncertainty_beta=4.0,
                residual_floor_sec=1.0e-3,
                coverage_ridge=1.0,
                residual_scale_window=2048,
                residual_scale_min_samples=32,
            ),
            seed=81,
        ),
        "v3": RecursiveLstdqV3LcbController(
            feature_dim=FEATURE_DIM,
            config=config,
            spec=RecursiveLstdqV3LcbSpec(
                ridge=1.0,
                uncertainty_beta=2.0,
                residual_floor_sec=1.0e-3,
            ),
            seed=81,
        ),
    }


def _run_training_episode(
    controller: Any,
    *,
    rng: np.random.Generator,
    cycles: int,
) -> tuple[float, float, float]:
    started = time.perf_counter()
    controller.start_episode(initial_environment_weight=1.0)
    start_runtime = time.perf_counter() - started
    features = rng.normal(size=FEATURE_DIM)
    update_runtime = 0.0
    for cycle in range(int(cycles)):
        terminal = cycle == int(cycles) - 1
        next_features = rng.normal(size=FEATURE_DIM)
        action_index = int(rng.integers(0, len(WEIGHTS)))
        next_action_index = (
            None if terminal else int(rng.integers(0, len(WEIGHTS)))
        )
        started = time.perf_counter()
        controller.update(
            features=features,
            action_index=action_index,
            cost=float(rng.uniform(5.0e-4, 5.0e-3)),
            next_features=next_features,
            terminal=terminal,
            next_action_index=next_action_index,
        )
        update_runtime += time.perf_counter() - started
        features = next_features
    started = time.perf_counter()
    controller.finish_episode(learned=True)
    commit_runtime = time.perf_counter() - started
    return start_runtime, update_runtime, commit_runtime


def benchmark(
    *,
    decision_repetitions: int = 500,
    training_episodes: int = 100,
    cycles_per_episode: int = 20,
    seed: int = 8101,
) -> Dict[str, Any]:
    if min(
        int(decision_repetitions),
        int(training_episodes),
        int(cycles_per_episode),
    ) <= 0:
        raise ValueError("benchmark repetitions and cycle count must be positive")
    controllers = _controllers()
    measurements: Dict[str, Dict[str, float]] = {}
    for index, (name, controller) in enumerate(controllers.items()):
        rng = np.random.default_rng(int(seed + index * 10_000))
        start_total = 0.0
        update_total = 0.0
        commit_total = 0.0
        for _ in range(int(training_episodes)):
            start, update, commit = _run_training_episode(
                controller,
                rng=rng,
                cycles=int(cycles_per_episode),
            )
            start_total += start
            update_total += update
            commit_total += commit

        decision_features = rng.normal(
            size=(int(decision_repetitions), FEATURE_DIM)
        )
        started = time.perf_counter()
        for features in decision_features:
            controller._values(features, cycle=0)
        decision_total = time.perf_counter() - started

        episodes = float(training_episodes)
        transitions = float(training_episodes * cycles_per_episode)
        decision_mean = decision_total / float(decision_repetitions)
        update_mean = update_total / transitions
        start_mean = start_total / episodes
        commit_mean = commit_total / episodes
        estimated_case = (
            float(cycles_per_episode) * (decision_mean + update_mean)
            + start_mean
            + commit_mean
        )
        measurements[name] = {
            "decision_sec_per_cycle": decision_mean,
            "update_sec_per_cycle": update_mean,
            "episode_start_sec": start_mean,
            "episode_commit_sec": commit_mean,
            "estimated_controller_sec_per_case": estimated_case,
        }

    v2_case = measurements["v2"]["estimated_controller_sec_per_case"]
    v3_case = measurements["v3"]["estimated_controller_sec_per_case"]
    ratio = v3_case / v2_case
    absolute_increase = v3_case - v2_case
    return {
        "dimensions": {
            "state_features": FEATURE_DIM,
            "action_basis": len(CENTERS),
            "actions": len(WEIGHTS),
            "joint_features": FEATURE_DIM * len(CENTERS),
        },
        "workload": {
            "decision_repetitions": int(decision_repetitions),
            "training_episodes": int(training_episodes),
            "cycles_per_episode": int(cycles_per_episode),
        },
        "measurements": measurements,
        "v3_vs_v2": {
            "case_runtime_ratio": ratio,
            "absolute_increase_sec": absolute_increase,
            "within_1p25x": bool(ratio <= 1.25),
            "within_plus_5ms": bool(absolute_increase <= 0.005),
            "passes_locked_overhead_gate": bool(
                ratio <= 1.25 and absolute_increase <= 0.005
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-repetitions", type=int, default=500)
    parser.add_argument("--training-episodes", type=int, default=100)
    parser.add_argument("--cycles-per-episode", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8101)
    args = parser.parse_args()
    print(
        json.dumps(
            benchmark(
                decision_repetitions=args.decision_repetitions,
                training_episodes=args.training_episodes,
                cycles_per_episode=args.cycles_per_episode,
                seed=args.seed,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
