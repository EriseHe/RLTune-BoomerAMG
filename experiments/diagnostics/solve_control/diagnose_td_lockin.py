from __future__ import annotations

import _project_paths  # noqa: F401

import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np

from amg_setup_gym_env import (
    DEFAULT_SETUP_PARAMS,
    SetupObsEncoder,
    build_setup_parameter_spec,
)
from online_td_lambda import (
    ExpectedSarsaLambda,
    ExpectedSarsaLambdaConfig,
    SolveStateEncoder,
    shaped_cycle_cost,
)
from setup_aware_compare_common import augment_setup_params
from hypre.bindings import create_env


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _make_encoders(*, tol: float, max_cycles: int, c_max: float) -> Dict[str, SolveStateEncoder]:
    parameter_spec, _fixed_params = build_setup_parameter_spec(
        tune_dim=7,
        tune7_variant="categorical",
    )
    setup_encoder = SetupObsEncoder(
        parameter_spec,
        dict(DEFAULT_SETUP_PARAMS),
        tuple(parameter_spec.parameter_names),
    )
    return {
        "setup_full": SolveStateEncoder(
            tol=tol,
            max_cycles=max_cycles,
            c_max=c_max,
            mode="setup_full",
            setup_obs_encoder=setup_encoder,
        ),
        "full": SolveStateEncoder(
            tol=tol,
            max_cycles=max_cycles,
            c_max=c_max,
            mode="full",
        ),
    }


def _make_controller(
    *,
    encoder: SolveStateEncoder,
    config: ExpectedSarsaLambdaConfig,
    seed: int,
) -> ExpectedSarsaLambda:
    return ExpectedSarsaLambda(
        feature_dim=encoder.feature_dim,
        config=config,
        seed=seed,
        initial_parameters=encoder.constant_value_parameters(config.initial_q_sec),
    )


def _sarsa_update(
    controller: ExpectedSarsaLambda,
    *,
    features: np.ndarray,
    action_index: int,
    cost: float,
    next_features: np.ndarray,
    next_action_index: int | None,
    terminal: bool,
) -> float:
    current_q = float(controller.q_values(features)[int(action_index)])
    next_q = 0.0 if terminal else float(
        controller.q_values(next_features)[int(next_action_index)]
    )
    td_error = float(cost + controller.config.gamma * next_q - current_q)
    controller.eligibility *= float(
        controller.config.gamma * controller.config.trace_lambda
    )
    action_gradient = controller._action_gradient(int(action_index), features)
    controller.eligibility += action_gradient
    controller.td_counts += np.abs(action_gradient)
    step_sizes = np.full_like(controller.theta, float(controller.config.alpha))
    if controller.config.td_decay_power > 0.0:
        step_sizes /= np.maximum(controller.td_counts, 1.0) ** float(
            controller.config.td_decay_power
        )
    controller.theta += td_error * step_sizes * controller.eligibility
    controller.steps += 1
    return td_error


def _replacing_trace_update(
    controller: ExpectedSarsaLambda,
    *,
    features: np.ndarray,
    action_index: int,
    cost: float,
    next_features: np.ndarray,
    terminal: bool,
    next_cycle: int,
) -> float:
    current_q = float(controller.q_values(features)[int(action_index)])
    if terminal:
        expected_next_q = 0.0
    else:
        next_q_values = controller.q_values(next_features)
        probabilities = controller._probabilities_from_q(
            next_q_values,
            epsilon=controller.epsilon,
            allowed_indices=controller.allowed_action_indices(next_cycle),
        )
        expected_next_q = float(probabilities @ next_q_values)
    td_error = float(
        cost + controller.config.gamma * expected_next_q - current_q
    )
    controller.eligibility *= float(
        controller.config.gamma * controller.config.trace_lambda
    )
    action_gradient = controller._action_gradient(int(action_index), features)
    active = np.abs(action_gradient) > 1.0e-15
    controller.eligibility[active] = action_gradient[active]
    controller.td_counts += np.abs(action_gradient)
    step_sizes = np.full_like(controller.theta, float(controller.config.alpha))
    if controller.config.td_decay_power > 0.0:
        step_sizes /= np.maximum(controller.td_counts, 1.0) ** float(
            controller.config.td_decay_power
        )
    controller.theta += td_error * step_sizes * controller.eligibility
    controller.steps += 1
    return td_error


def _true_online_sarsa_update(
    controller: ExpectedSarsaLambda,
    *,
    features: np.ndarray,
    action_index: int,
    cost: float,
    next_features: np.ndarray,
    next_action_index: int | None,
    terminal: bool,
    q_old: float,
) -> tuple[float, float]:
    """Apply Algorithm 3 from van Seijen et al. (2016)."""

    action_features = controller._action_gradient(int(action_index), features)
    if terminal:
        next_action_features = np.zeros_like(action_features)
    else:
        next_action_features = controller._action_gradient(
            int(next_action_index),
            next_features,
        )
    current_q = float(np.sum(controller.theta * action_features))
    next_q = float(np.sum(controller.theta * next_action_features))
    td_error = float(cost + controller.config.gamma * next_q - current_q)
    alpha = float(controller.config.alpha)
    trace_dot = float(np.sum(controller.eligibility * action_features))
    controller.eligibility *= float(
        controller.config.gamma * controller.config.trace_lambda
    )
    controller.eligibility += action_features
    controller.eligibility -= (
        alpha
        * float(controller.config.gamma * controller.config.trace_lambda)
        * trace_dot
        * action_features
    )
    controller.theta += (
        alpha * (td_error + current_q - float(q_old)) * controller.eligibility
        - alpha * (current_q - float(q_old)) * action_features
    )
    controller.td_counts += np.abs(action_features)
    controller.steps += 1
    return td_error, next_q


def _phase(cycle: int) -> str:
    if cycle <= 3:
        return "early_0_3"
    if cycle <= 8:
        return "middle_4_8"
    return "late_9_plus"


def _features_for_variant(
    variant: str,
    setup_features: np.ndarray,
    full_features: np.ndarray,
) -> np.ndarray:
    if variant.startswith("full_"):
        features = full_features
        if variant == "full_drop_cycle_time":
            features = full_features.copy()
            features[5] = 0.0
        return features
    if variant == "setup_zeroed":
        features = setup_features.copy()
        features[-7:] = 0.0
        return features
    if variant.startswith("setup_drop_"):
        features = setup_features.copy()
        setup_indices = {
            "setup_drop_strong": (25,),
            "setup_drop_pmax": (28,),
            "setup_drop_interp": (31,),
            "setup_drop_strong_pmax_interp": (25, 28, 31),
            "setup_drop_cycle_time": (5,),
            "setup_drop_last_time_weight": (5, 6),
        }
        features[list(setup_indices[variant])] = 0.0
        return features
    return setup_features


def _summarize_controller(
    *,
    name: str,
    controller: ExpectedSarsaLambda,
    transitions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    weights = controller.weights
    greedy_counts: Dict[str, Counter[float]] = defaultdict(Counter)
    q_gaps: Dict[str, list[float]] = defaultdict(list)
    observed_errors: Dict[str, list[float]] = defaultdict(list)
    for transition in transitions:
        features = _features_for_variant(
            name,
            transition["setup_features"],
            transition["full_features"],
        )
        q_values = controller.q_values(features)
        phase = _phase(int(transition["cycle"]))
        greedy = float(weights[int(np.argmin(q_values))])
        greedy_counts[phase][greedy] += 1
        q_gaps[phase].append(float(q_values[2] - q_values[6]))
        action = int(transition["action_index"])
        observed_return_key = "observed_return"
        if name == "setup_no_shaping":
            observed_return_key = "raw_observed_return"
        elif name == "setup_unit_cycle_cost":
            observed_return_key = "unit_observed_return"
        observed_errors[phase].append(
            float(
                q_values[action]
                - transition[observed_return_key]
            )
        )

    return {
        "steps": int(controller.steps),
        "episodes": int(controller.episodes),
        "epsilon": float(controller.epsilon),
        "theta_norm": float(np.linalg.norm(controller.theta)),
        "direct_count_by_weight": {
            f"{weight:.1f}": float(np.sum(controller.td_counts[index]))
            for index, weight in enumerate(weights)
        },
        "phase": {
            phase: {
                "states": int(sum(counts.values())),
                "greedy_counts": {
                    f"{weight:.1f}": int(count)
                    for weight, count in sorted(counts.items())
                },
                "mean_q_1p2_minus_1p6_sec": float(np.mean(q_gaps[phase])),
                "observed_action_q_minus_return_mean_sec": float(
                    np.mean(observed_errors[phase])
                ),
                "observed_action_q_minus_return_rmse_sec": float(
                    np.sqrt(np.mean(np.square(observed_errors[phase])))
                ),
            }
            for phase, counts in greedy_counts.items()
        },
    }


def run(args: argparse.Namespace) -> None:
    result = json.loads(args.result.read_text())
    rows = result["online_records"]["bandit_td"]
    saved = np.load(args.checkpoint)
    base_config = ExpectedSarsaLambdaConfig(**json.loads(str(saved["config"])))
    tol = float(result["protocol"]["tol"])
    max_cycles = int(result["protocol"]["max_cycles"])
    c_max = float(result["protocol"]["coefficient_range"][1])
    encoders = _make_encoders(tol=tol, max_cycles=max_cycles, c_max=c_max)

    configs = {
        "exact_setup_full": base_config,
        "full_no_setup": base_config,
        "setup_zeroed": base_config,
        "setup_lambda0": replace(base_config, trace_lambda=0.0),
        "setup_lambda0p2": replace(base_config, trace_lambda=0.2),
        "setup_lambda0p5": replace(base_config, trace_lambda=0.5),
        "setup_lambda0p95": replace(base_config, trace_lambda=0.95),
        "setup_replacing_trace": base_config,
        "setup_no_shaping": replace(base_config, potential_scale_sec=0.0),
        "setup_unit_cycle_cost": replace(
            base_config,
            potential_scale_sec=0.0,
        ),
        "setup_drop_cycle_time": base_config,
        "setup_drop_last_time_weight": base_config,
        "full_drop_cycle_time": base_config,
        "setup_drop_strong": base_config,
        "setup_drop_pmax": base_config,
        "setup_drop_interp": base_config,
        "setup_drop_strong_pmax_interp": base_config,
        "setup_rbf0p1": replace(base_config, action_rbf_sigma=0.1),
        "setup_constant_alpha1e3": replace(
            base_config,
            alpha=1.0e-3,
            td_decay_power=0.0,
        ),
        "setup_actual_sarsa": base_config,
        "full_actual_sarsa": base_config,
        "setup_true_online_a0p02": replace(
            base_config,
            alpha=2.0e-2,
            td_decay_power=0.0,
        ),
        "setup_true_online_a0p005": replace(
            base_config,
            alpha=5.0e-3,
            td_decay_power=0.0,
        ),
        "setup_true_online_a0p001": replace(
            base_config,
            alpha=1.0e-3,
            td_decay_power=0.0,
        ),
        "full_true_online_a0p02": replace(
            base_config,
            alpha=2.0e-2,
            td_decay_power=0.0,
        ),
    }
    variants: Dict[str, ExpectedSarsaLambda] = {}
    for index, (name, config) in enumerate(configs.items()):
        encoder = encoders["full" if name.startswith("full_") else "setup_full"]
        variants[name] = _make_controller(
            encoder=encoder,
            config=config,
            seed=int(args.seed + index),
        )

    transitions: list[Dict[str, Any]] = []
    for episode_index, record in enumerate(rows):
        outcome = record["outcome"]
        mkw = record["mkw"]
        params = record["params"]
        residuals = [float(value) for value in outcome["cycle_residuals"]]
        times = [float(value) for value in outcome["cycle_times"]]
        actions = [float(value) for value in outcome["cycle_actions"]]
        with create_env(**dict(mkw)) as env:
            prep = env.prepare_rl(params=augment_setup_params(dict(params)))
            initial_residual = float(prep.initial_residual_norm)

        episode: list[Dict[str, Any]] = []
        for cycle, (weight, next_residual, cycle_time) in enumerate(
            zip(actions, residuals, times)
        ):
            residual = initial_residual if cycle == 0 else residuals[cycle - 1]
            previous_residual = (
                initial_residual if cycle <= 1 else residuals[cycle - 2]
            )
            last_weight = (
                base_config.anchor_weight if cycle == 0 else actions[cycle - 1]
            )
            last_cycle_time = 0.0 if cycle == 0 else times[cycle - 1]
            terminal = bool(cycle == len(residuals) - 1)
            setup_features = encoders["setup_full"].encode(
                mkw=mkw,
                setup_params=params,
                initial_residual=initial_residual,
                residual=residual,
                previous_residual=previous_residual,
                cycle=cycle,
                last_weight=last_weight,
                last_cycle_time=last_cycle_time,
            ).copy()
            full_features = encoders["full"].encode(
                mkw=mkw,
                initial_residual=initial_residual,
                residual=residual,
                previous_residual=previous_residual,
                cycle=cycle,
                last_weight=last_weight,
                last_cycle_time=last_cycle_time,
            ).copy()
            if terminal:
                next_setup_features = setup_features
                next_full_features = full_features
            else:
                next_setup_features = encoders["setup_full"].encode(
                    mkw=mkw,
                    setup_params=params,
                    initial_residual=initial_residual,
                    residual=next_residual,
                    previous_residual=residual,
                    cycle=cycle + 1,
                    last_weight=weight,
                    last_cycle_time=cycle_time,
                ).copy()
                next_full_features = encoders["full"].encode(
                    mkw=mkw,
                    initial_residual=initial_residual,
                    residual=next_residual,
                    previous_residual=residual,
                    cycle=cycle + 1,
                    last_weight=weight,
                    last_cycle_time=cycle_time,
                ).copy()
            cost = shaped_cycle_cost(
                cycle_time,
                residual=residual,
                next_residual=next_residual,
                tol=tol,
                scale_sec=base_config.potential_scale_sec,
                gamma=base_config.gamma,
                terminal=terminal,
            )
            if terminal and not (
                np.isfinite(next_residual) and next_residual <= tol
            ):
                cost += float(base_config.failure_penalty_sec)
            raw_cost = float(cycle_time)
            if terminal and not (
                np.isfinite(next_residual) and next_residual <= tol
            ):
                raw_cost += float(base_config.failure_penalty_sec)
            unit_cost = 1.0e-3
            if terminal and not (
                np.isfinite(next_residual) and next_residual <= tol
            ):
                unit_cost += float(base_config.failure_penalty_sec)
            action_index = int(np.argmin(np.abs(np.asarray(base_config.weights) - weight)))
            episode.append(
                {
                    "episode": int(episode_index),
                    "cycle": int(cycle),
                    "action_index": action_index,
                    "cost": float(cost),
                    "raw_cost": raw_cost,
                    "unit_cost": unit_cost,
                    "setup_features": setup_features,
                    "full_features": full_features,
                    "next_setup_features": next_setup_features,
                    "next_full_features": next_full_features,
                    "terminal": terminal,
                }
            )

        observed_return = 0.0
        raw_observed_return = 0.0
        unit_observed_return = 0.0
        for transition in reversed(episode):
            observed_return = float(
                transition["cost"] + base_config.gamma * observed_return
            )
            raw_observed_return = float(
                transition["raw_cost"]
                + base_config.gamma * raw_observed_return
            )
            unit_observed_return = float(
                transition["unit_cost"]
                + base_config.gamma * unit_observed_return
            )
            transition["observed_return"] = observed_return
            transition["raw_observed_return"] = raw_observed_return
            transition["unit_observed_return"] = unit_observed_return

        true_online_q_old = {
            name: 0.0 for name in variants if "_true_online_" in name
        }
        for controller in variants.values():
            controller.start_episode()
        for cycle, transition in enumerate(episode):
            next_action_index = (
                None if transition["terminal"] else episode[cycle + 1]["action_index"]
            )
            for name, controller in variants.items():
                features = _features_for_variant(
                    name,
                    transition["setup_features"],
                    transition["full_features"],
                )
                next_features = _features_for_variant(
                    name,
                    transition["next_setup_features"],
                    transition["next_full_features"],
                )
                transition_cost = (
                    transition["unit_cost"]
                    if name == "setup_unit_cycle_cost"
                    else (
                        transition["raw_cost"]
                        if name == "setup_no_shaping"
                        else transition["cost"]
                    )
                )
                if "_true_online_" in name:
                    _td_error, next_q = _true_online_sarsa_update(
                        controller,
                        features=features,
                        action_index=transition["action_index"],
                        cost=transition_cost,
                        next_features=next_features,
                        next_action_index=next_action_index,
                        terminal=transition["terminal"],
                        q_old=true_online_q_old[name],
                    )
                    true_online_q_old[name] = next_q
                elif name.endswith("actual_sarsa"):
                    _sarsa_update(
                        controller,
                        features=features,
                        action_index=transition["action_index"],
                        cost=transition_cost,
                        next_features=next_features,
                        next_action_index=next_action_index,
                        terminal=transition["terminal"],
                    )
                elif name == "setup_replacing_trace":
                    _replacing_trace_update(
                        controller,
                        features=features,
                        action_index=transition["action_index"],
                        cost=transition_cost,
                        next_features=next_features,
                        terminal=transition["terminal"],
                        next_cycle=cycle + 1,
                    )
                else:
                    controller.update(
                        features=features,
                        action_index=transition["action_index"],
                        cost=transition_cost,
                        next_features=next_features,
                        terminal=transition["terminal"],
                        next_cycle=cycle + 1,
                    )
            transitions.append(transition)
        for controller in variants.values():
            controller.finish_episode(learned=True)

        done = episode_index + 1
        if done % max(1, int(args.progress_every)) == 0 or done == len(rows):
            print(json.dumps({"stage": "replay", "episodes": done}), flush=True)

    exact = variants["exact_setup_full"]
    output = {
        "source": {
            "result": str(args.result),
            "checkpoint": str(args.checkpoint),
            "episodes": int(len(rows)),
            "transitions": int(len(transitions)),
        },
        "exact_replay": {
            "theta_max_abs_difference": float(
                np.max(np.abs(exact.theta - saved["theta"]))
            ),
            "td_count_max_abs_difference": float(
                np.max(np.abs(exact.td_counts - saved["td_counts"]))
            ),
        },
        "variants": {
            name: _summarize_controller(
                name=name,
                controller=controller,
                transitions=transitions,
            )
            for name, controller in variants.items()
        },
    }
    _write_json(args.output, output)
    print(json.dumps(output["exact_replay"], sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded online TD stream to diagnose action lock-in."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=39394016)
    parser.add_argument("--progress-every", type=int, default=250)
    args = parser.parse_args()
    os.environ.setdefault("SETUP_PARAM_RESOLUTION", "20")
    run(args)


if __name__ == "__main__":
    main()
